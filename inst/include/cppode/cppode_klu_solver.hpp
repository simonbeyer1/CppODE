/*
 KLU sparse LU solver wrapper for CppODE — raw CSC interface.

 Takes raw int, double pointers (Ap, Ai, Ax) directly.
 Uses klu_refactor() after the first klu_factor() for maximum speed.

 KLU settings (BTF, ordering) are determined at codegen time by
 analyzing the Jacobian sparsity pattern in Python and passed as
 compile-time defines:
 -DKLUBTF=0|1   (0=off, 1=on; default: 1)
 -DKLUAMD=0|1   (0=AMD, 1=COLAMD; default: 0)

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_KLU_SOLVER_HPP
#define CPPODE_KLU_SOLVER_HPP

#ifdef KLU

// KLU is provided by the system SuiteSparse installation; the include
// path is discovered by ./configure at install time and injected via
// PKG_CXXFLAGS when a sparse model is compiled.
#include <klu.h>

#include <vector>
#include <stdexcept>
#include <cstdio>

// Compile-time KLU settings (set by codegen via -D flags).
// Defaults match KLU's own defaults (BTF on, AMD ordering).
#ifndef KLUBTF
#define KLUBTF 1
#endif

#ifndef KLUAMD
#define KLUAMD 0
#endif

namespace cppode {
namespace ad_lu {

class klu_lu_solver
{
public:

  klu_lu_solver()  {
    klu_defaults(&m_common);
    // BTF: determined at codegen time by SCC analysis of the Jacobian graph.
    // nblocks > 1 (e.g. pathway models with causal chains) → BTF on.
    // nblocks == 1 (e.g. PDE stencils, strongly connected) → BTF off.
    m_common.btf = KLUBTF;
    // Ordering: determined at codegen time by row-degree variance heuristic.
    // Low variance (uniform stencil patterns) → AMD (0).
    // High variance (hub nodes, irregular networks) → COLAMD (1).
    m_common.ordering = KLUAMD;
    // Disable row scaling.  BDF iteration matrices W = 1/(γh)I − J
    // are diagonally dominant, so scaling adds O(nnz) work per
    // klu_factor/klu_refactor with no numerical benefit.
    m_common.scale = 0;
  }
  ~klu_lu_solver() { free_numeric(); free_symbolic(); }

  // Non-copyable, moveable
  klu_lu_solver(const klu_lu_solver&) = delete;
  klu_lu_solver& operator=(const klu_lu_solver&) = delete;
  klu_lu_solver(klu_lu_solver&& o) noexcept
    : m_symbolic(o.m_symbolic), m_numeric(o.m_numeric),
      m_common(o.m_common), m_n(o.m_n),
      m_pattern_analyzed(o.m_pattern_analyzed),
      m_has_numeric(o.m_has_numeric)
  {
    o.m_symbolic = nullptr;
    o.m_numeric  = nullptr;
    o.m_pattern_analyzed = false;
    o.m_has_numeric = false;
  }
  klu_lu_solver& operator=(klu_lu_solver&&) = delete;

  // ------------------------------------------------------------------
  //  Analyze sparsity pattern (call once)
  // ------------------------------------------------------------------
  void analyze_pattern(int n, const int* Ap, const int* Ai)
  {
    free_symbolic();
    free_numeric();
    m_n = n;
    m_symbolic = klu_analyze(n,
                             const_cast<int*>(Ap),
                             const_cast<int*>(Ai),
                             &m_common);
    if (!m_symbolic)
      throw std::runtime_error("klu_analyze failed");
    m_pattern_analyzed = true;
    m_has_numeric = false;
  }

  // ------------------------------------------------------------------
  //  Factorize: analyze (if needed) + factor/refactor
  // ------------------------------------------------------------------
  void factorize(int n, const int* Ap, const int* Ai, const double* Ax)
  {
    if (!m_pattern_analyzed)
      analyze_pattern(n, Ap, Ai);

    if (!m_has_numeric) {
      m_numeric = klu_factor(const_cast<int*>(Ap),
                             const_cast<int*>(Ai),
                             const_cast<double*>(Ax),
                             m_symbolic, &m_common);
      if (!m_numeric)
        throw std::runtime_error("klu_factor failed");
      // Sort L and U factors for cache-friendly triangular solves.
      // One-time O(nnz) cost that accelerates every subsequent klu_solve.
      klu_sort(m_symbolic, m_numeric, &m_common);
      m_has_numeric = true;
#ifdef CPPODE_PROFILE
      // Report fill-in and settings (once)
      std::fprintf(stderr, "KLU: n=%d, nnz=%d, L_nnz=%d, U_nnz=%d, fill=%.1fx, "
                     "BTF=%s, ordering=%s\n",
                     m_n, m_symbolic->nz, m_numeric->lnz, m_numeric->unz,
                     (double)(m_numeric->lnz + m_numeric->unz) / (double)m_symbolic->nz,
                     KLUBTF ? "on" : "off",
                     KLUAMD == 0 ? "AMD" : "COLAMD");
#endif
    } else {
      // Fast path: reuse symbolic analysis + numeric structure
      int ok = klu_refactor(const_cast<int*>(Ap),
                            const_cast<int*>(Ai),
                            const_cast<double*>(Ax),
                            m_symbolic, m_numeric, &m_common);
      if (!ok) {
        // Fallback: full re-factor
        free_numeric();
        m_numeric = klu_factor(const_cast<int*>(Ap),
                               const_cast<int*>(Ai),
                               const_cast<double*>(Ax),
                               m_symbolic, &m_common);
        if (!m_numeric)
          throw std::runtime_error("klu_factor failed after refactor fallback");
        klu_sort(m_symbolic, m_numeric, &m_common);
      }
    }
  }

  // ------------------------------------------------------------------
  //  Solve in-place: b ← W⁻¹ b
  // ------------------------------------------------------------------
  void solve(double* b) const
  {
    klu_solve(const_cast<klu_symbolic*>(m_symbolic),
              const_cast<klu_numeric*>(m_numeric),
              m_n, 1, b,
              const_cast<klu_common*>(&m_common));
  }

  void solve(std::vector<double>& b) const
  { solve(b.data()); }

  // ------------------------------------------------------------------
  //  Batched solve: B ← W⁻¹ B  (B is n × nrhs, column-major)
  // ------------------------------------------------------------------
  void solve_batch(double* B, int nrhs) const
  {
    klu_solve(const_cast<klu_symbolic*>(m_symbolic),
              const_cast<klu_numeric*>(m_numeric),
              m_n, nrhs, B,
              const_cast<klu_common*>(&m_common));
  }

  void report_fillin() const {
    if (m_numeric) {
      std::fprintf(stderr, "  KLU fill-in: L_nnz = %d, U_nnz = %d (ratio vs original: %.1fx)\n",
                   m_numeric->lnz, m_numeric->unz,
                   (double)(m_numeric->lnz + m_numeric->unz) / (double)m_n);
    }
  }

  void reset_pattern()
  {
    free_numeric();
    free_symbolic();
    m_pattern_analyzed = false;
    m_has_numeric = false;
  }

private:

  void free_symbolic()
  {
    if (m_symbolic) {
      klu_free_symbolic(&m_symbolic, &m_common);
      m_symbolic = nullptr;
    }
  }

  void free_numeric()
  {
    if (m_numeric) {
      klu_free_numeric(&m_numeric, &m_common);
      m_numeric = nullptr;
    }
  }

  klu_symbolic* m_symbolic = nullptr;
  klu_numeric*  m_numeric  = nullptr;
  klu_common    m_common;
  int           m_n = 0;
  bool          m_pattern_analyzed = false;
  bool          m_has_numeric = false;
};

} // namespace ad_lu
} // namespace cppode

#endif // KLU
#endif // CPPODE_KLU_SOLVER_HPP
