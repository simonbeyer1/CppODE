/*
 AD-aware sparse LU solver for CppODE — raw CSC + KLU.

 Backend: KLU (KLU) or Eigen::SparseLU fallback.
 Operates on csc_matrix<T> with raw Ap/Ai/Ax arrays.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_SPARSE_AD_LU_HPP
#define CPPODE_SPARSE_AD_LU_HPP

#include <type_traits>
#include <algorithm>
#include <vector>

#include <cppode/cppode_types.hpp>
#include <cppode/cppode_ad_lu.hpp>   // for is_ad, extract_values, etc.
#include <fadbad++/fadiff.h>

#ifdef KLU
#include <cppode/cppode_klu_solver.hpp>
#endif

namespace cppode {
namespace ad_lu {

// ============================================================================
//  extract_csc_values: csc_matrix<F<Inner>> → csc_matrix<Inner>
//
//  Copies the CSC structure (Ap, Ai) and extracts value part of Ax.
// ============================================================================

template<class Inner, unsigned int N>
inline csc_matrix<Inner> extract_csc_values(const csc_matrix<fadbad::F<Inner,N>>& W)
{
  csc_matrix<Inner> out;
  out.n   = W.n;
  out.nnz = W.nnz;
  out.Ap  = W.Ap;
  out.Ai  = W.Ai;
  out.Ax.resize(W.nnz);
  out.pattern_built = W.pattern_built;

  for (int k = 0; k < W.nnz; ++k)
    out.Ax[k] = const_cast<fadbad::F<Inner,N>&>(W.Ax[k]).x();

  return out;
}

// ============================================================================
//  sparse_lu_solver<T> — Base case: T is a non-AD scalar
//
//  KLU → klu_lu_solver (preferred)
//  otherwise      → compile error (KLU is now required for sparse)
// ============================================================================

template<class T, class Enable = void>
class sparse_lu_solver;

template<class Scalar>
class sparse_lu_solver<Scalar, std::enable_if_t<!is_ad<Scalar>::value>>
{
public:

#if defined(KLU)

  sparse_lu_solver() = default;
  sparse_lu_solver(sparse_lu_solver&& o) noexcept
    : m_solver(std::move(o.m_solver)) {}

  void factorize(const csc_matrix<Scalar>& W)
  {
    m_solver.factorize(W.n, W.Ap.data(), W.Ai.data(), W.Ax.data());
  }

  void analyze_pattern(const csc_matrix<Scalar>& W)
  {
    m_solver.analyze_pattern(W.n, W.Ap.data(), W.Ai.data());
  }

  void factorize_numeric(const csc_matrix<Scalar>& W)
  {
    m_solver.factorize(W.n, W.Ap.data(), W.Ai.data(), W.Ax.data());
  }

  void solve(std::vector<Scalar>& b) const
  { m_solver.solve(b); }

  /// Batched solve: B ← W⁻¹ B (column-major n × nrhs)
  void solve_batch(std::vector<Scalar>& B, int nrhs) const
  { m_solver.solve_batch(B.data(), nrhs); }

  std::vector<Scalar> solve_copy(const std::vector<Scalar>& b) const
  {
    std::vector<Scalar> x = b;
    m_solver.solve(x);
    return x;
  }

  /// Scalar-only solve (identity for base case — same as solve)
  void solve_scalar(std::vector<double>& b) const
  { m_solver.solve(b); }

  bool pattern_analyzed() const { return m_solver.pattern_analyzed(); }
  void reset_pattern() { m_solver.reset_pattern(); }

private:
  klu_lu_solver m_solver;

#else
  // Without KLU, sparse is not supported — static_assert at compile time
  static_assert(sizeof(Scalar) == 0,
                "CppODE sparse LU requires KLU (KLU). "
                "Rebuild the package or use dense mode.");
#endif
};

// ============================================================================
//  sparse_lu_solver<F<Inner>> — Recursive AD case (IFT peeling)
//
//  Same approach as dense: extract value part, factorize, propagate
//  derivatives via IFT.  CSC structure is shared between levels.
//
//  Optimizations vs. naive implementation:
//    1. Persistent m_W_val: CSC structure (Ap, Ai) is copied ONCE,
//       subsequent factorize calls only update Ax values.
//    2. Persistent m_W_stored: full AD matrix copied once, then only
//       the Ax array is overwritten (Ap/Ai never change).
//    3. Mutable solve buffers: m_b_val, m_rhs_all, m_col_buf are
//       allocated once and reused across all solve() calls.
//    4. Compile-time n_derivs: when N > 0, skip the O(nnz) scan
//       over W_stored to determine the number of active derivatives.
// ============================================================================

template<class Inner, unsigned int N>
class sparse_lu_solver<fadbad::F<Inner,N>>
{
  using F = fadbad::F<Inner, N>;

public:

  void factorize(const csc_matrix<F>& W)
  {
    m_n = W.n;
    const int nnz = W.nnz;

    // --- Extract scalar values into persistent m_W_val ---
    // First call: copy structure (Ap, Ai).  Subsequent: only update Ax.
    if (m_W_val.nnz != nnz) {
      m_W_val.n   = W.n;
      m_W_val.nnz = nnz;
      m_W_val.Ap  = W.Ap;
      m_W_val.Ai  = W.Ai;
      m_W_val.Ax.resize(nnz);
      m_W_val.pattern_built = W.pattern_built;
    }

    if constexpr (!is_ad<Inner>::value) {
      // ============================================================
      //  Inner = double: fused value extraction + dW pre-extraction
      //
      //  Extract scalar values and derivative components in a single
      //  pass over W.Ax.  The derivative block m_dW_ax is laid out as
      //  nnz × n_derivs (row-major per entry) so that the IFT matvec
      //  can iterate over CSC entries with contiguous derivative
      //  access per entry.
      //
      //  m_W_stored is NOT needed — eliminates nnz F-object copies.
      // ============================================================

      // Determine n_derivs
      unsigned nd;
      if constexpr (N > 0) {
        nd = N;
      } else {
        nd = 0;
        for (int k = 0; k < nnz; ++k) {
          unsigned sz = const_cast<F&>(W.Ax[k]).size();
          if (sz > nd) nd = sz;
        }
      }
      m_n_derivs_cached = nd;

      if (nd > 0)
        m_dW_ax.resize(static_cast<size_t>(nnz) * nd);

      // Fused extraction: values + derivatives in one pass
      for (int k = 0; k < nnz; ++k) {
        auto& wk = const_cast<F&>(W.Ax[k]);
        m_W_val.Ax[k] = static_cast<double>(wk.x());
        if (nd > 0) {
          unsigned wsz = wk.size();
          double* dst = m_dW_ax.data() + static_cast<size_t>(k) * nd;
          for (unsigned j = 0; j < nd; ++j)
            dst[j] = (j < wsz) ? static_cast<double>(wk.d(j)) : 0.0;
        }
      }

      // Cache CSC structure for IFT matvec (Ap, Ai needed in solve)
      if (m_Ap_cached.empty()) {
        m_Ap_cached = W.Ap;
        m_Ai_cached = W.Ai;
      }
      m_nnz_cached = nnz;

    } else {
      // ============================================================
      //  Inner = F<...> (nested AD): store full AD matrix for the
      //  generic element-wise IFT path.
      // ============================================================
      if (m_W_stored.nnz != nnz) {
        m_W_stored = W;
      } else {
        std::copy(W.Ax.begin(), W.Ax.end(), m_W_stored.Ax.begin());
      }
      for (int k = 0; k < nnz; ++k)
        m_W_val.Ax[k] = const_cast<F&>(W.Ax[k]).x();
    }

    m_inner.factorize(m_W_val);
  }

  void analyze_pattern(const csc_matrix<F>& W)
  {
    // Structural analysis only — reuse m_W_val buffer
    const int nnz = W.nnz;
    if (m_W_val.nnz != nnz) {
      m_W_val.n   = W.n;
      m_W_val.nnz = nnz;
      m_W_val.Ap  = W.Ap;
      m_W_val.Ai  = W.Ai;
      m_W_val.Ax.resize(nnz);
      m_W_val.pattern_built = W.pattern_built;
    }
    for (int k = 0; k < nnz; ++k)
      m_W_val.Ax[k] = const_cast<F&>(W.Ax[k]).x();
    m_inner.analyze_pattern(m_W_val);
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
      if constexpr (N > 0) {
        n_derivs = N;
      } else {
        n_derivs = std::max(m_n_derivs_cached, max_deriv_size(b));
      }
    } else {
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

    // 4. IFT sparse matvec: rhs_all -= dW · b_val
    ift_sparse_matvec(n, n_derivs);

    // 5. Batched solve
    m_inner.solve_batch(m_rhs_all, static_cast<int>(n_derivs));

    // 6. Bulk-inject results
    bulk_inject_results(b, m_b_val, m_rhs_all, n, n_derivs);
  }

  /// Batched solve for nrhs RHS vectors (column-major).
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

  bool pattern_analyzed() const { return m_inner.pattern_analyzed(); }
  void reset_pattern() { m_inner.reset_pattern(); }

  /// Scalar-only solve: forward to inner solver, no IFT
  void solve_scalar(std::vector<double>& b) const
  {
    m_inner.solve_scalar(b);
  }

private:

  // ================================================================
  //  IFT sparse matvec: rhs_all -= dW · b_val
  //
  //  Inner = double:
  //    Derivative components were pre-extracted into m_dW_ax in
  //    factorize().  No F<> objects touched — pure double arithmetic.
  //    Layout: m_dW_ax[p * nd + j] = d(j) of the p-th CSC entry.
  //
  //  Inner = F<...> (nested AD):
  //    Single pass over m_W_stored (element-wise loop).
  // ================================================================

  void ift_sparse_matvec(int n, unsigned n_derivs) const
  {
    if constexpr (!is_ad<Inner>::value) {
      // =============================================================
      //  Pre-extracted path: iterate CSC with flat double derivatives
      //
      //  m_dW_ax was built in factorize().  If n_derivs > nd_W
      //  (RHS has more derivs than W), the extra directions have
      //  dW = 0, so no correction is needed for them.
      // =============================================================

      unsigned nd_W = m_n_derivs_cached;
      if (nd_W == 0) return;

      const int* Ap = m_Ap_cached.data();
      const int* Ai = m_Ai_cached.data();

      for (int col = 0; col < n; ++col) {
        const double b_col = m_b_val[col];
        for (int p = Ap[col]; p < Ap[col + 1]; ++p) {
          int row = Ai[p];
          const double* dw = m_dW_ax.data() + static_cast<size_t>(p) * nd_W;
          for (unsigned j = 0; j < nd_W; ++j)
            m_rhs_all[j * n + row] -= dw[j] * b_col;
        }
      }

    } else {
      // =============================================================
      //  Generic path: nested AD types (Inner = F<...>)
      //
      //  Single pass over W_stored — each F<Inner,N> element is
      //  touched exactly once.
      // =============================================================

      for (int col = 0; col < n; ++col) {
        const Inner b_col = m_b_val[col];
        for (int p = m_W_stored.Ap[col]; p < m_W_stored.Ap[col + 1]; ++p) {
          int row = m_W_stored.Ai[p];
          auto& w_entry = const_cast<F&>(m_W_stored.Ax[p]);
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
  csc_matrix<F>     m_W_stored;       // Full AD matrix (nested AD path only)
  csc_matrix<Inner> m_W_val;          // Persistent scalar extraction buffer
  sparse_lu_solver<Inner> m_inner;

  // Pre-extracted derivative block for IFT (Inner = double only).
  // Built once in factorize(), reused across all solve() calls.
  // Layout: m_dW_ax[p * n_derivs + j] = derivative j of CSC entry p.
  std::vector<double> m_dW_ax;
  unsigned m_n_derivs_cached = 0;
  int m_nnz_cached = 0;
  std::vector<int> m_Ap_cached;       // CSC column pointers (for IFT matvec)
  std::vector<int> m_Ai_cached;       // CSC row indices (for IFT matvec)

  // Persistent solve buffers.
  mutable std::vector<Inner> m_b_val;     // n scalars: extracted value part
  mutable std::vector<Inner> m_rhs_all;   // n × n_derivs: derivative RHS
  mutable std::vector<F>     m_col_buf;   // n entries: column buffer for solve_batch
};

// ============================================================================
//  Sparse Jacobian matrix-vector product: y += W * x
// ============================================================================

template<class T>
void sparse_jac_matvec(const csc_matrix<T>& W,
                       const std::vector<T>& x,
                       std::vector<T>& y)
{
  csc_matvec_add(W, x, y);
}

} // namespace ad_lu
} // namespace cppode

#endif // CPPODE_SPARSE_AD_LU_HPP
