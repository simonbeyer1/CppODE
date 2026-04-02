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
    //    Use overwrite=1 so our setting always wins.
    setenv("MKL_NUM_THREADS",      "1", 1);
    setenv("OPENBLAS_NUM_THREADS",  "1", 1);
    setenv("GOTO_NUM_THREADS",      "1", 1);
    setenv("OMP_NUM_THREADS",       "1", 1);

    done = true;
  }
}
} // namespace detail
} // namespace cppode

namespace cppode {
namespace ad_lu {

// ============================================================================
//  AD type traits
// ============================================================================

template<class T> struct is_ad : std::false_type {};
template<class T, unsigned int N> struct is_ad<fadbad::F<T,N>> : std::true_type {};

template<class T> struct inner_type       { using type = T; };
template<class T, unsigned int N> struct inner_type<fadbad::F<T,N>> { using type = T; };
template<class T> using inner_type_t = typename inner_type<T>::type;

/// Recursively unwrap F<F<...F<T>...>> to the innermost scalar type T.
/// Used to derive time_type from value_type: time is always a plain scalar.
///   scalar_type<double>::type            = double
///   scalar_type<F<double,N>>::type       = double
///   scalar_type<F<F<double,N>,M>>::type  = double
template<class T> struct scalar_type       { using type = T; };
template<class T, unsigned int N> struct scalar_type<fadbad::F<T,N>>
  : scalar_type<T> {};
template<class T> using scalar_type_t = typename scalar_type<T>::type;

/// Extract the innermost scalar value from any (nested) AD type.
/// For double, returns the value unchanged. For F<T,N>, recursively
/// unwraps to the base scalar. Used throughout the stepper/controller
/// hierarchy to convert value_type time arguments to scalar time_type.
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v) { return static_cast<double>(v); }

template<class T, unsigned int N>
inline double scalar_value(const fadbad::F<T,N>& v) {
  return scalar_value(const_cast<fadbad::F<T,N>&>(v).x());
}

// ============================================================================
//  Value/derivative extraction helpers for std::vector<F<T>>
// ============================================================================

/// Extract value (.x()) from each element: vector<F<Inner>> → vector<Inner>
template<class Inner, unsigned int N>
inline std::vector<Inner> extract_values(const std::vector<fadbad::F<Inner,N>>& v)
{
  const size_t n = v.size();
  std::vector<Inner> out(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = const_cast<fadbad::F<Inner,N>&>(v[i]).x();
  return out;
}

/// Extract value (.x()) from each element: dense_matrix<F<Inner>> → dense_matrix<Inner>
template<class Inner, unsigned int N>
inline dense_matrix<Inner> extract_values(const dense_matrix<fadbad::F<Inner,N>>& M)
{
  dense_matrix<Inner> out(M.rows(), M.cols());
  const size_t sz = M.data.size();
  for (size_t k = 0; k < sz; ++k)
    out.data[k] = const_cast<fadbad::F<Inner,N>&>(M.data[k]).x();
  return out;
}

/// Extract derivative d(j) from each element: vector<F<Inner>> → vector<Inner>
template<class Inner, unsigned int N>
inline std::vector<Inner> extract_derivs(const std::vector<fadbad::F<Inner,N>>& v, unsigned j)
{
  const size_t n = v.size();
  std::vector<Inner> out(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = const_cast<fadbad::F<Inner,N>&>(v[i]).d(j);
  return out;
}

/// Extract derivative d(j) from each element: dense_matrix<F<Inner>> → dense_matrix<Inner>
template<class Inner, unsigned int N>
inline dense_matrix<Inner> extract_derivs(const dense_matrix<fadbad::F<Inner,N>>& M, unsigned j)
{
  dense_matrix<Inner> out(M.rows(), M.cols());
  const size_t sz = M.data.size();
  for (size_t k = 0; k < sz; ++k)
    out.data[k] = const_cast<fadbad::F<Inner,N>&>(M.data[k]).d(j);
  return out;
}

/// Max number of derivative directions in a vector
template<class Inner, unsigned int N>
inline unsigned max_deriv_size(const std::vector<fadbad::F<Inner,N>>& v)
{
  unsigned mx = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    unsigned sz = const_cast<fadbad::F<Inner,N>&>(v[i]).size();
    if (sz > mx) mx = sz;
  }
  return mx;
}

/// Max number of derivative directions in a dense_matrix
template<class Inner, unsigned int N>
inline unsigned max_deriv_size(const dense_matrix<fadbad::F<Inner,N>>& M)
{
  unsigned mx = 0;
  for (size_t k = 0; k < M.data.size(); ++k) {
    unsigned sz = const_cast<fadbad::F<Inner,N>&>(M.data[k]).size();
    if (sz > mx) mx = sz;
  }
  return mx;
}

// ============================================================================
//  Bulk extraction helpers — single-pass AoS→SoA
//
//  Instead of n_derivs separate passes calling extract_derivs(v, j),
//  these extract ALL derivative directions in ONE pass over the vector,
//  producing a column-major n × n_derivs matrix.
// ============================================================================

/// Bulk-extract all derivatives from vector<F<Inner,N>> into column-major
/// Inner array of size n × n_derivs.  One pass over the source vector.
template<class Inner, unsigned int N>
inline void bulk_extract_derivs(
    const std::vector<fadbad::F<Inner,N>>& v,
    std::vector<Inner>& out,
    int n, unsigned n_derivs)
{
  out.resize(static_cast<size_t>(n) * n_derivs);
  for (int i = 0; i < n; ++i) {
    auto& vi = const_cast<fadbad::F<Inner,N>&>(v[i]);
    unsigned sz = vi.size();
    for (unsigned j = 0; j < n_derivs; ++j) {
      out[j * n + i] = (j < sz) ? vi.d(j) : Inner(0);
    }
  }
}

/// Bulk-inject all derivatives from column-major Inner array into vector<F<Inner,N>>.
/// Also sets the value part.  One pass over the target vector.
///
/// IMPORTANT: For FADBAD++ F<T,N> with static N, writing to .d(j) only works
/// when m_depend == true.  Otherwise .d(j) returns a reference to a static
/// zero variable and the write is silently discarded.  We call .diff(0) first
/// to activate the derivative slots (sets m_depend = true and initializes all
/// slots), then overwrite ALL slots with the correct values.
template<class Inner, unsigned int N>
inline void bulk_inject_results(
    std::vector<fadbad::F<Inner,N>>& b,
    const std::vector<Inner>& b_val,
    const std::vector<Inner>& dx_all,
    int n, unsigned n_derivs)
{
  for (int i = 0; i < n; ++i) {
    b[i].x() = b_val[i];
    if (n_derivs > 0) {
      // Activate derivative storage: sets m_depend = true.
      // For static N > 0: diff(0) sets m_diff[0]=1, rest=0, m_depend=true.
      // For dynamic N == 0: diff(0, n_derivs) allocates and sets m_diff[0]=1.
      // We overwrite ALL slots below, so the initial values don't matter.
      if constexpr (N > 0) {
        if (!b[i].depend()) b[i].diff(0);
      } else {
        if (!b[i].depend()) b[i].diff(0, n_derivs);
      }
      for (unsigned j = 0; j < n_derivs; ++j)
        b[i][j] = dx_all[j * n + i];   // use operator[] (direct array access)
    }
  }
}

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

  /// Solve in-place: b ← W⁻¹ b
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

  /// Batched solve: B ← W⁻¹ B  (B is column-major n × nrhs)
  ///
  /// Uses BLAS-3 internally (dtrsm) via dgetrs with nrhs > 1.
  /// This is the key optimization: cache-blocked triangular solves
  /// instead of nrhs separate memory-bound triangular back-subs.
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

  /// Solve with separate output
  std::vector<Scalar> solve_copy(const std::vector<Scalar>& b) const
  {
    std::vector<Scalar> x = b;
    solve(x);
    return x;
  }

  /// Scalar-only solve (identity for base case — same as solve)
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

template<class Inner, unsigned int N>
class dense_lu_solver<fadbad::F<Inner,N>>
{
  using F = fadbad::F<Inner, N>;

public:

  void factorize(const dense_matrix<F>& W)
  {
    const int n = W.rows();
    m_n = n;
    const int nn = n * n;

    // --- Store full AD matrix for IFT derivative propagation ---
    // First call: full deep copy.  Subsequent: only overwrite data vector.
    if (m_W_stored.rows() != n) {
      m_W_stored = W;
    } else {
      std::copy(W.data.begin(), W.data.end(), m_W_stored.data.begin());
    }

    // --- Extract scalar values into persistent m_W_val ---
    // First call: allocate.  Subsequent: only update values.
    if (m_W_val.rows() != n)
      m_W_val.resize(n, n);
    for (int k = 0; k < nn; ++k)
      m_W_val.data[k] = const_cast<F&>(W.data[k]).x();

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
    //    When N > 0, it is the compile-time upper bound — skip the scan.
    unsigned n_derivs;
    if constexpr (N > 0) {
      n_derivs = N;
    } else {
      n_derivs = max_deriv_size(b);
      n_derivs = std::max(n_derivs, max_deriv_size(m_W_stored));
    }
    if (n_derivs == 0) {
      for (int i = 0; i < n; ++i)
        b[i].x() = m_b_val[i];
      return;
    }

    // 3. Bulk-extract ALL derivative RHS into column-major n × n_derivs
    //    Reuse persistent buffer (resize is no-op after first call)
    m_rhs_all.resize(static_cast<size_t>(n) * n_derivs);
    for (int i = 0; i < n; ++i) {
      auto& bi = const_cast<F&>(b[i]);
      unsigned sz = bi.size();
      for (unsigned j = 0; j < n_derivs; ++j)
        m_rhs_all[j * n + i] = (j < sz) ? bi.d(j) : Inner(0);
    }

    // 4. IFT matvec: rhs_all -= dW · b_val
    //    Dispatches to dgemv (BLAS-2) when Inner = double, or
    //    generic element-wise loop for nested AD types.
    ift_matvec(n, n_derivs);

    // 5. Batched solve: solve ALL n_derivs RHS in one call
    //    For Inner = double: single dgetrs with nrhs = n_derivs (BLAS-3!)
    //    For Inner = F<...>: recursive batched solve
    m_inner.solve_batch(m_rhs_all, static_cast<int>(n_derivs));

    // 6. Bulk-inject: write values and derivatives in one pass
    bulk_inject_results(b, m_b_val, m_rhs_all, n, n_derivs);
  }

  /// Batched solve for nrhs RHS vectors stored column-major.
  /// Reuses persistent column buffer to avoid per-call allocation.
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

  /// Scalar-only solve: use the value-level factorization on plain doubles.
  /// No IFT, no derivative propagation — just the base-case LAPACK solve.
  void solve_scalar(std::vector<double>& b) const
  {
    m_inner.solve_scalar(b);
  }

private:

  // ================================================================
  //  IFT matvec: rhs_all -= dW · b_val
  //
  //  The IFT derivative correction for ALL n_derivs directions:
  //    rhs_j -= dW_j · b_val   for j = 0, ..., n_derivs-1
  //
  //  Reformulated as a single dgemv:
  //    rhs_vec -= dW_block · b_val
  //
  //  where dW_block is (n*n_derivs) × n, column-major, with
  //    dW_block[j*n + row, col] = W_stored(row, col).d(j)
  //
  //  This replaces the element-wise triple loop with a vectorized
  //  BLAS-2 operation.  The extraction (AoS→SoA deinterleave) runs
  //  once; the dgemv multiply-accumulate is fully SIMD-vectorized.
  // ================================================================

  void ift_matvec(int n, unsigned n_derivs) const
  {
    if constexpr (!is_ad<Inner>::value) {
      // =============================================================
      //  BLAS-2 path: dgemv (Inner = double)
      //
      //  Build dW_block: (m × n) column-major where m = n * n_derivs.
      //  Column col of dW_block contains all derivative slices of
      //  column col of W_stored, stacked as:
      //    [d0_row0..d0_rowN-1, d1_row0..d1_rowN-1, ...]
      // =============================================================

      int m = n * static_cast<int>(n_derivs);
      m_dW_block.resize(static_cast<size_t>(m) * n);

      // Extraction: deinterleave AoS (F objects) → SoA (flat doubles)
      for (int col = 0; col < n; ++col) {
        double* col_dst = m_dW_block.data() + static_cast<size_t>(col) * m;
        for (int row = 0; row < n; ++row) {
          auto& w = const_cast<F&>(m_W_stored(row, col));
          unsigned wsz = w.size();
          // Write derivatives for this entry into the column:
          // positions j*n + row for j = 0..n_derivs-1
          for (unsigned j = 0; j < n_derivs; ++j)
            col_dst[j * n + row] = (j < wsz) ?
          static_cast<double>(w.d(j)) : 0.0;
        }
      }

      // dgemv: rhs_all = 1.0 * rhs_all + (-1.0) * dW_block · b_val
      double alpha = -1.0, beta = 1.0;
      int inc = 1;
      char trans = 'N';
      F77_CALL(dgemv)(&trans, &m, &n, &alpha,
               m_dW_block.data(), &m,
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
  dense_matrix<F>     m_W_stored;      // Full AD matrix (for IFT derivatives)
  dense_matrix<Inner> m_W_val;         // Persistent scalar extraction buffer
  dense_lu_solver<Inner> m_inner;

  // Persistent solve buffers — avoid per-call heap allocations.
  // Mutable because solve() is logically const (same results, no state change).
  mutable std::vector<Inner>  m_b_val;     // n scalars: extracted value part
  mutable std::vector<Inner>  m_rhs_all;   // n × n_derivs: derivative RHS
  mutable std::vector<F>      m_col_buf;   // n entries: column buffer for solve_batch
  mutable std::vector<double> m_dW_block;  // (n*n_derivs) × n: derivative matrix (dgemv path only)
};

} // namespace ad_lu
} // namespace cppode

#endif // CPPODE_AD_LU_HPP
