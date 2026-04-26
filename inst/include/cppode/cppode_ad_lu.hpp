/*
 AD-aware dense LU solver for CppODE — LAPACK backend.

 Uses dgetrf/dgetrs from R's bundled LAPACK for the base case.
 AD types (F<T>, F<F<T>>, ...) are handled by recursive IFT peeling.

 BLAS-3 optimization: IFT derivative propagation uses batched
 dgetrs (nrhs = n_derivs) instead of n_derivs separate solves.
 The matvec phase fuses extraction and subtraction in a single
 pass over W_stored for cache-optimal access.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_AD_LU_HPP
#define CPPODE_AD_LU_HPP

#include <type_traits>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>

#include <cppode/cppode_types.hpp>
#include <cppode/cppode_ad_traits.hpp>
#include <fadbad++/fadiff.h>

// LAPACK declarations via R's headers (handles ILP64 automatically)
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

// --- Runtime detection of MKL's thread-control function ---
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace cppode {
namespace detail {
inline void ensure_single_thread_blas() {
  static bool done = false;
  if (!done) {
    // 1. Try MKL runtime API (works even without MKL headers)
#ifdef _WIN32
    auto mkl_fn = (void(*)(int))GetProcAddress(
      GetModuleHandle(NULL), "MKL_Set_Num_Threads");
#else
    auto mkl_fn = (void(*)(int))dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads");
#endif
    if (mkl_fn) mkl_fn(1);

    // 2. Try OpenBLAS runtime API
#ifndef _WIN32
    auto oblas_fn = (void(*)(int))dlsym(RTLD_DEFAULT, "openblas_set_num_threads");
    if (oblas_fn) oblas_fn(1);
#endif

    // 3. Environment variables as last-resort fallback
#ifdef _WIN32
    _putenv_s("MKL_NUM_THREADS",      "1");
    _putenv_s("OPENBLAS_NUM_THREADS",  "1");
    _putenv_s("GOTO_NUM_THREADS",      "1");
    _putenv_s("OMP_NUM_THREADS",       "1");
#else
    // Use overwrite=1 so our setting always wins.
    setenv("MKL_NUM_THREADS",      "1", 1);
    setenv("OPENBLAS_NUM_THREADS",  "1", 1);
    setenv("GOTO_NUM_THREADS",      "1", 1);
    setenv("OMP_NUM_THREADS",       "1", 1);
#endif

    done = true;
  }
}
} // namespace detail
} // namespace cppode

namespace cppode {
namespace ad_lu {

// ============================================================================
//  AD type traits / bulk helpers — pulled in from cppode_ad_traits.hpp.
//  Re-exported here so existing consumers using `ad_lu::is_ad`,
//  `ad_lu::scalar_value`, etc. compile unchanged.
// ============================================================================

using cppode::ad_traits::is_ad;
using cppode::ad_traits::inner_type;
using cppode::ad_traits::inner_type_t;
using cppode::ad_traits::scalar_type;
using cppode::ad_traits::scalar_type_t;
using cppode::ad_traits::scalar_value;
using cppode::ad_traits::extract_values;
using cppode::ad_traits::extract_derivs;
using cppode::ad_traits::max_deriv_size;
using cppode::ad_traits::bulk_extract_derivs;
using cppode::ad_traits::bulk_inject_results;

// ============================================================================
//  dense_lu_solver<double> — Base case: LAPACK dgetrf + dgetrs
// ============================================================================

template<class T, class Enable = void>
class dense_lu_solver;

template<class Scalar>
class dense_lu_solver<Scalar, std::enable_if_t<!is_ad<Scalar>::value>>
{
public:
  void factorize(const dense_matrix<Scalar>& W)
  {
    detail::ensure_single_thread_blas();
    const int n = W.rows();
    m_n = n;
    m_lu_data = W.data;
    m_ipiv.resize(n);

    int info = 0;
    F77_CALL(dgetrf)(&m_n, &m_n, m_lu_data.data(), &m_n, m_ipiv.data(), &info);
  }

  // Factorize by MOVING the data out of the source matrix (zero-copy).
  // The source matrix is left in a valid but unspecified state.
  // Use when the caller no longer needs the original W (e.g. m_W_temp
  // which gets rebuilt from scratch on every factorize_W call).
  void factorize_move(dense_matrix<Scalar>& W)
  {
    detail::ensure_single_thread_blas();
    const int n = W.rows();
    m_n = n;
    m_lu_data.swap(W.data);   // O(1) pointer swap, no memcpy
    m_ipiv.resize(n);

    int info = 0;
    F77_CALL(dgetrf)(&m_n, &m_n, m_lu_data.data(), &m_n, m_ipiv.data(), &info);
  }

  // Solve in-place: b ← W⁻¹ b
  void solve(std::vector<Scalar>& b) const
  {
    char trans = 'N';
    int nrhs = 1;
    int info = 0;
    F77_CALL(dgetrs)(&trans, &m_n, &nrhs,
             const_cast<double*>(m_lu_data.data()), &m_n,
             const_cast<int*>(m_ipiv.data()),
             b.data(), &m_n, &info
                       FCONE);
  }

  // Batched solve: B ← W⁻¹ B  (B is column-major n × nrhs)
  //
  // Uses BLAS-3 internally (dtrsm) via dgetrs with nrhs > 1.
  // This is the key optimization: cache-blocked triangular solves
  // instead of nrhs separate memory-bound triangular back-subs.
  void solve_batch(std::vector<Scalar>& B, int nrhs) const
  {
    if (nrhs <= 0) return;
    char trans = 'N';
    int info = 0;
    F77_CALL(dgetrs)(&trans, &m_n, &nrhs,
             const_cast<double*>(m_lu_data.data()), &m_n,
             const_cast<int*>(m_ipiv.data()),
             B.data(), &m_n, &info
                       FCONE);
  }

  // Solve with separate output
  std::vector<Scalar> solve_copy(const std::vector<Scalar>& b) const
  {
    std::vector<Scalar> x = b;
    solve(x);
    return x;
  }

  // Scalar-only solve (identity for base case — same as solve)
  void solve_scalar(std::vector<double>& b) const
  { solve(b); }

private:
  mutable int m_n = 0;
  std::vector<double> m_lu_data;   // LU-factorized copy of W
  std::vector<int>    m_ipiv;      // pivot indices
};

// ============================================================================
//  dense_lu_solver<F<Inner>> — Recursive AD case (IFT peeling)
//
//  Given W·x = b (all entries F<Inner>):
//    Value:   W_val · x_val = b_val       (solved recursively)
//    Deriv j: W_val · dx_j  = db_j - dW_j · x_val  (reuses factorization)
//
//  Optimizations:
//    1. Persistent m_W_val: scalar matrix extracted ONCE, only Ax values
//       updated on subsequent factorize calls.
//    2. Persistent m_W_stored: full AD matrix stored for IFT, only data
//       array overwritten after the first call.
//    3. Mutable solve buffers: m_b_val, m_rhs_all, m_col_buf allocated
//       once and reused across all solve() calls.
//    4. Compile-time n_derivs: when N > 0, skip the O(n²) scan to
//       determine active derivative directions.
//    5. BLAS-2 IFT matvec (dgemv): when Inner = double, the fused
//       derivative matvec is replaced by a single dgemv call on a
//       pre-extracted (n*n_derivs) × n derivative block.
//    6. BLAS-3 batched solve: dgetrs with nrhs = n_derivs (unchanged).
// ============================================================================

// Generic AD specialization: works for fadbad::F<Inner,N> and cppode::dual<Inner,N>.
// The body uses only the FADBAD-compatible accessor surface (.x(), .d(j),
// .size(), .diff(), operator[], .depend()) which both backends provide.
template<class AD_T>
class dense_lu_solver<AD_T, std::enable_if_t<is_ad<AD_T>::value>>
{
  using F = AD_T;
  using Inner = inner_type_t<AD_T>;
  static constexpr unsigned N = ad_traits::detail::ad_static_size<AD_T>::value;

public:

  void factorize(const dense_matrix<F>& W)
  {
    const int n = W.rows();
    m_n = n;
    const int nn = n * n;

    // --- Extract scalar values into persistent m_W_val ---
    // First call: allocate.  Subsequent: only update values.
    if (m_W_val.rows() != n)
      m_W_val.resize(n, n);
    for (int k = 0; k < nn; ++k)
      m_W_val.data[k] = const_cast<F&>(W.data[k]).x();

    if constexpr (!is_ad<Inner>::value) {
      // ============================================================
      //  Inner = double: pre-extract dW_block for BLAS-2 IFT
      //
      //  The IFT matvec needs the derivative components of W as a
      //  (n*n_derivs) × n column-major double block.  W doesn't
      //  change between factorize() and solve(), so we extract ONCE
      //  here instead of on every solve() call.
      //
      //  m_W_stored is NOT needed — all derivative information lives
      //  in m_dW_block.  This saves an n×n F<double,N> deep copy.
      // ============================================================

      // Determine n_derivs
      unsigned nd;
      if constexpr (N > 0) {
        nd = N;
      } else {
        nd = max_deriv_size(W);
      }
      m_n_derivs_cached = nd;

      if (nd > 0) {
        int m = n * static_cast<int>(nd);
        m_dW_block.resize(static_cast<size_t>(m) * n);

        // AoS→SoA deinterleave: extract all derivative components
        for (int col = 0; col < n; ++col) {
          double* col_dst = m_dW_block.data() + static_cast<size_t>(col) * m;
          for (int row = 0; row < n; ++row) {
            auto& w = const_cast<F&>(W(row, col));
            unsigned wsz = w.size();
            for (unsigned j = 0; j < nd; ++j)
              col_dst[j * n + row] = (j < wsz) ?
                static_cast<double>(w.d(j)) : 0.0;
          }
        }
      }

    } else {
      // ============================================================
      //  Inner = F<...> (nested AD): store full AD matrix for the
      //  generic element-wise IFT path.
      // ============================================================
      if (m_W_stored.rows() != n) {
        m_W_stored = W;
      } else {
        std::copy(W.data.begin(), W.data.end(), m_W_stored.data.begin());
      }
    }

    // Recursive: inner solver factorizes the scalar matrix.
    m_inner.factorize(m_W_val);
  }

  void solve(std::vector<F>& b) const
  {
    const int n = m_n;

    // 1. Extract and solve value part (reuse buffer)
    m_b_val.resize(n);
    for (int i = 0; i < n; ++i)
      m_b_val[i] = const_cast<F&>(b[i]).x();
    m_inner.solve(m_b_val);

    // 2. Determine derivative directions
    unsigned n_derivs;
    if constexpr (!is_ad<Inner>::value) {
      // Inner = double: n_derivs was determined at factorize time.
      // For RHS-only derivatives (W has none), check b too.
      if constexpr (N > 0) {
        n_derivs = N;
      } else {
        n_derivs = std::max(m_n_derivs_cached, max_deriv_size(b));
      }
    } else {
      // Inner = F<...>: need to scan both b and W_stored
      if constexpr (N > 0) {
        n_derivs = N;
      } else {
        n_derivs = max_deriv_size(b);
        n_derivs = std::max(n_derivs, max_deriv_size(m_W_stored));
      }
    }
    if (n_derivs == 0) {
      for (int i = 0; i < n; ++i)
        b[i].x() = m_b_val[i];
      return;
    }

    // 3. Bulk-extract ALL derivative RHS into column-major n × n_derivs
    m_rhs_all.resize(static_cast<size_t>(n) * n_derivs);
    for (int i = 0; i < n; ++i) {
      auto& bi = const_cast<F&>(b[i]);
      unsigned sz = bi.size();
      for (unsigned j = 0; j < n_derivs; ++j)
        m_rhs_all[j * n + i] = (j < sz) ? bi.d(j) : Inner(0);
    }

    // 4. IFT matvec: rhs_all -= dW · b_val
    ift_matvec(n, n_derivs);

    // 5. Batched solve: solve ALL n_derivs RHS in one call
    m_inner.solve_batch(m_rhs_all, static_cast<int>(n_derivs));

    // 6. Bulk-inject: write values and derivatives in one pass
    bulk_inject_results(b, m_b_val, m_rhs_all, n, n_derivs);
  }

  // Batched solve for nrhs RHS vectors stored column-major.
  void solve_batch(std::vector<F>& B_flat, int nrhs) const
  {
    const int n = m_n;
    m_col_buf.resize(n);
    for (int k = 0; k < nrhs; ++k) {
      for (int i = 0; i < n; ++i)
        m_col_buf[i] = B_flat[k * n + i];
      solve(m_col_buf);
      for (int i = 0; i < n; ++i)
        B_flat[k * n + i] = m_col_buf[i];
    }
  }

  // Scalar-only solve: use the value-level factorization on plain doubles.
  // No IFT, no derivative propagation — just the base-case LAPACK solve.
  void solve_scalar(std::vector<double>& b) const
  {
    m_inner.solve_scalar(b);
  }

private:

  // ================================================================
  //  IFT matvec: rhs_all -= dW · b_val
  //
  //  Inner = double:
  //    m_dW_block was pre-extracted in factorize().  Only the dgemv
  //    call remains — no per-solve extraction overhead.
  //
  //  Inner = F<...> (nested AD):
  //    Single pass over m_W_stored (element-wise loop).
  // ================================================================

  void ift_matvec(int n, unsigned n_derivs) const
  {
    if constexpr (!is_ad<Inner>::value) {
      // =============================================================
      //  BLAS-2 path: dgemv only (extraction done in factorize)
      //
      //  m_dW_block is (m × n) column-major, m = n * n_derivs_cached.
      //  If n_derivs > n_derivs_cached (RHS has more derivs than W),
      //  the extra derivative directions have dW = 0, so no correction
      //  is needed for them — dgemv covers [0, n_derivs_cached) and
      //  the remaining rhs_all entries stay untouched.
      // =============================================================

      unsigned nd_W = m_n_derivs_cached;
      if (nd_W == 0) return;  // W has no derivatives → no IFT correction

      int m = n * static_cast<int>(nd_W);
      double alpha = -1.0, beta = 1.0;
      int inc = 1;
      char trans = 'N';
      F77_CALL(dgemv)(&trans, &m, &n, &alpha,
               const_cast<double*>(m_dW_block.data()), &m,
               const_cast<double*>(m_b_val.data()), &inc,
               &beta, m_rhs_all.data(), &inc  FCONE);

    } else {
      // =============================================================
      //  Generic path: nested AD types (Inner = F<...>)
      //
      //  Single pass over W_stored — each F<Inner,N> element is
      //  touched exactly once.
      // =============================================================

      for (int col = 0; col < n; ++col) {
        const Inner b_col = m_b_val[col];
        for (int row = 0; row < n; ++row) {
          auto& w_entry = const_cast<F&>(m_W_stored(row, col));
          unsigned wsz = w_entry.size();
          for (unsigned j = 0; j < n_derivs; ++j) {
            Inner dw = (j < wsz) ? w_entry.d(j) : Inner(0);
            m_rhs_all[j * n + row] -= dw * b_col;
          }
        }
      }
    }
  }

  int m_n = 0;
  dense_matrix<F>     m_W_stored;      // Full AD matrix (nested AD path only)
  dense_matrix<Inner> m_W_val;         // Persistent scalar extraction buffer
  dense_lu_solver<Inner> m_inner;

  // Pre-extracted derivative block for BLAS-2 IFT (Inner = double only).
  // Built once in factorize(), reused across all solve() calls.
  // Layout: (n*n_derivs_cached) × n, column-major.
  std::vector<double> m_dW_block;
  unsigned m_n_derivs_cached = 0;     // n_derivs at last factorize

  // Persistent solve buffers — avoid per-call heap allocations.
  mutable std::vector<Inner>  m_b_val;     // n scalars: extracted value part
  mutable std::vector<Inner>  m_rhs_all;   // n × n_derivs: derivative RHS
  mutable std::vector<F>      m_col_buf;   // n entries: column buffer for solve_batch
};

} // namespace ad_lu
} // namespace cppode

#endif // CPPODE_AD_LU_HPP
