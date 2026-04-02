/*
 CppODE LU — unified Dense/Sparse iteration matrix solver
 ==========================================================

 Single class lu_W<Value, is_sparse> that encapsulates:
 - Jacobian storage (dense_matrix or csc_matrix)
 - W = (1/γh)·I − J construction
 - LU factorization (LAPACK or KLU)
 - Linear solve with AD-aware IFT dispatch
 - Jacobian caching for lagged reuse

 Template parameter is_sparse selects dense vs sparse via
 if constexpr — no SFINAE, no CRTP, no tag dispatch.

 AD types (F<double>, F<F<double>>) are handled transparently
 by the underlying dense_lu_solver / sparse_lu_solver.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_LU_HPP
#define CPPODE_LU_HPP

#include <cstddef>
#include <type_traits>
#include <algorithm>
#include <vector>

#include <cppode/cppode_types.hpp>
#include <cppode/cppode_odeint_compat.hpp>
#include <cppode/cppode_ad_lu.hpp>
#include <cppode/cppode_sparse_ad_lu.hpp>

namespace cppode {

// Tags (still needed for template parameters in generated code)
struct dense_lu_tag {};
struct sparse_lu_tag {};

// ============================================================================
//  Null placeholder for sparse_lu_solver when is_sparse = false.
//  Avoids instantiating the real KLU solver in dense-only builds.
// ============================================================================

namespace ad_lu {
struct detail_null_sparse_lu {
  template<class W> void factorize(const W&) {}
  template<class B> void solve(B&) const {}
  void solve_scalar(std::vector<double>&) const {}
  bool pattern_analyzed() const { return false; }
  void reset_pattern() {}
};
} // namespace ad_lu

// ============================================================================
//  lu_W<Value, is_sparse>
//
//  Self-contained iteration matrix solver.  Steppers hold this as a member
//  and call its methods directly — no CRTP inheritance needed.
//
//  Interface summary:
//    call_jacobian(jac_func, x, t)      — evaluate Jacobian
//    factorize_W(n, inv_gamma_dt)       — build W = 1/(γh)·I − J, factorize
//    refactorize_W_gamma_only(n, val)   — O(n) diagonal update (sparse only)
//    refactorize_W_from_cache(n, val)   — restore cached J, re-factorize
//    cache_jacobian(n)                  — snapshot J for later reuse
//    solve(b)                           — W⁻¹ b  (full AD / IFT)
//    solve_scalar(b)                    — W_val⁻¹ b  (doubles only, no IFT)
//    invalidate() / has_valid_jacobian() / has_valid_lu() — state tracking
//    resize(x)                          — allocate / reallocate buffers
// ============================================================================

template<class Value, bool is_sparse>
class lu_W
{
public:
  typedef Value                            value_type;
  typedef std::vector<value_type>          state_type;
  typedef state_type                       deriv_type;
  typedef value_type                       time_type;
  typedef dense_matrix<value_type>         matrix_type;
  typedef csc_matrix<value_type>           sparse_matrix_type;

  typedef state_wrapper<state_type>        wrapped_state_type;
  typedef state_wrapper<deriv_type>        wrapped_deriv_type;
  typedef state_wrapper<matrix_type>       wrapped_matrix_type;

  // ====================================================================
  //  Jacobian evaluation — writes to internal storage
  // ====================================================================

  template<class JacFunc>
  void call_jacobian(JacFunc& jf, const state_type& x, time_type t)
  {
    if constexpr (is_sparse) {
      jf(x, m_W_sparse, t, m_dfdt.m_v);
    } else {
      jf(x, m_jac.m_v, t, m_dfdt.m_v);
    }
  }

  // ====================================================================
  //  factorize_W: build W = inv_gamma_dt · I − J, then factorize
  // ====================================================================

  void factorize_W(size_t n, value_type inv_gamma_dt)
  {
    if constexpr (is_sparse) {
      factorize_W_sparse(n, inv_gamma_dt);
    } else {
      factorize_W_dense(n, inv_gamma_dt);
    }
  }

  // ====================================================================
  //  refactorize_W_gamma_only: O(n) diagonal update (sparse)
  //  refactorize_W_from_cache: restore cached J, re-factorize
  // ====================================================================

  void refactorize_W_gamma_only(size_t n, value_type inv_gamma_dt_new)
  {
    if constexpr (is_sparse) {
      value_type delta = inv_gamma_dt_new - m_last_inv_gamma_dt;
      for (size_t i = 0; i < n; ++i)
        m_W_work.Ax[m_diag_offsets[i]] += delta;
      m_last_inv_gamma_dt = inv_gamma_dt_new;
      m_sparse_lu.factorize(m_W_work);
    } else {
      // Dense: no gamma-only path, do full refactorize from cache
      refactorize_W_from_cache(n, inv_gamma_dt_new);
    }
  }

  void refactorize_W_from_cache(size_t n, value_type inv_gamma_dt)
  {
    if constexpr (is_sparse) {
      m_dfdt.m_v = m_dfdt_cache.m_v;
      refactorize_W_gamma_only(n, inv_gamma_dt);
    } else {
      m_jac.m_v = m_jac_cache.m_v;
      m_dfdt.m_v = m_dfdt_cache.m_v;
      factorize_W(n, inv_gamma_dt);
    }
  }

  // ====================================================================
  //  cache_jacobian: snapshot for lagged reuse
  // ====================================================================

  void cache_jacobian(size_t)
  {
    if constexpr (is_sparse) {
      // Sparse: Jacobian lives in m_W_sparse (pre-negated by codegen).
      // factorize_W never touches it, so only dfdt needs caching.
      m_dfdt_cache.m_v = m_dfdt.m_v;
    } else {
      m_jac_cache.m_v = m_jac.m_v;
      m_dfdt_cache.m_v = m_dfdt.m_v;
    }
  }

  // ====================================================================
  //  solve: W⁻¹ b  (full AD with IFT for derivative propagation)
  // ====================================================================

  void solve(state_type& b)
  {
    if constexpr (is_sparse) {
      m_sparse_lu.solve(b);
    } else {
      m_dense_lu.solve(b);
    }
  }

  // ====================================================================
  //  solve_scalar: W_val⁻¹ b  (doubles only, no IFT overhead)
  // ====================================================================

  void solve_scalar(std::vector<double>& b)
  {
    if constexpr (is_sparse) {
      m_sparse_lu.solve_scalar(b);
    } else {
      m_dense_lu.solve_scalar(b);
    }
  }

  // ====================================================================
  //  dfdt accessor (needed by Rosenbrock stages)
  // ====================================================================

  const deriv_type& dfdt() const { return m_dfdt.m_v; }
  deriv_type& dfdt_mut() { return m_dfdt.m_v; }

  // ====================================================================
  //  Lagged Jacobian / LU state tracking
  // ====================================================================

  void invalidate()
  {
    m_has_valid_jac = false;
    m_has_valid_lu  = false;
  }

  bool has_valid_jacobian() const { return m_has_valid_jac; }
  bool has_valid_lu()       const { return m_has_valid_lu; }
  time_type last_factorized_dt() const { return m_last_dt; }

  void set_jacobian_valid() { m_has_valid_jac = true; }
  void set_lu_valid(time_type dt)
  {
    m_has_valid_lu = true;
    m_last_dt = dt;
  }

  // ====================================================================
  //  resize: allocate / reallocate all buffers
  // ====================================================================

  template<class StateIn>
  bool resize(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(m_dfdt, x);

    if constexpr (is_sparse) {
      const int n = static_cast<int>(x.size());
      if (m_W_sparse.n != n) {
        m_W_sparse.n = n;
        m_W_sparse.nnz = 0;
        m_W_sparse.pattern_built = false;
        m_W_sparse.Ap.clear();
        m_W_sparse.Ai.clear();
        m_W_sparse.Ax.clear();
        m_W_work.nnz = 0;
        m_diag_offsets.clear();
        m_sparse_lu.reset_pattern();
        resized = true;
      }
      resized |= adjust_size_by_resizeability(m_dfdt_cache, x);
    } else {
      resized |= adjust_size_by_resizeability(m_jac, x);
      resized |= adjust_size_by_resizeability(m_jac_cache, x);
      resized |= adjust_size_by_resizeability(m_dfdt_cache, x);
    }

    return resized;
  }

  // ====================================================================
  //  Direct access to Jacobian storage (for codegen)
  //
  //  Dense:  jac_func writes to  m_jac.m_v  (dense_matrix<T>)
  //  Sparse: jac_func writes to  m_W_sparse (csc_matrix<T>)
  // ====================================================================

  matrix_type&        jac_dense()  { return m_jac.m_v; }
  sparse_matrix_type& jac_sparse() { return m_W_sparse; }

private:

  // ====================================================================
  //  Dense factorize_W implementation
  // ====================================================================

  void factorize_W_dense(size_t n, value_type inv_gamma_dt)
  {
    if (m_W_temp.rows() != static_cast<int>(n))
      m_W_temp.resize(n, n);
    const size_t nn = n * n;

    // Build W = inv_gamma_dt · I − J
    if constexpr (!ad_lu::is_ad<value_type>::value) {
      // BLAS path: dcopy + dscal for W = −J, then diagonal add
      int nn_i = static_cast<int>(nn);
      int inc = 1;
      double neg = -1.0;
      F77_CALL(dcopy)(&nn_i,
               const_cast<double*>(m_jac.m_v.data.data()), &inc,
               m_W_temp.data.data(), &inc);
      F77_CALL(dscal)(&nn_i, &neg, m_W_temp.data.data(), &inc);
      for (size_t i = 0; i < n; ++i)
        m_W_temp(i, i) += inv_gamma_dt;
    } else {
      // Generic path for AD types
      for (size_t k = 0; k < nn; ++k)
        m_W_temp.data[k] = -m_jac.m_v.data[k];
      for (size_t i = 0; i < n; ++i)
        m_W_temp(i, i) += inv_gamma_dt;
    }

    m_dense_lu.factorize(m_W_temp);
  }

  // ====================================================================
  //  Sparse factorize_W implementation (allocation-free after first call)
  // ====================================================================

  void factorize_W_sparse(size_t n, value_type inv_gamma_dt)
  {
    const int nnz = m_W_sparse.nnz;

    // One-time structural setup
    if (m_W_work.nnz != nnz) {
      m_W_work = m_W_sparse;  // deep-copy

      // Precompute diagonal offsets in CSC
      m_diag_offsets.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const int* begin = m_W_work.Ai.data() + m_W_work.Ap[i];
        const int  count = m_W_work.Ap[i + 1] - m_W_work.Ap[i];
        const int* pos = std::lower_bound(begin, begin + count, static_cast<int>(i));
        m_diag_offsets[i] = static_cast<int>(pos - m_W_work.Ai.data());
      }
    }

    // Hot path: copy pre-negated Jacobian + diagonal add
    if constexpr (!ad_lu::is_ad<value_type>::value) {
      int nnz_i = nnz;
      int inc = 1;
      F77_CALL(dcopy)(&nnz_i,
               const_cast<double*>(m_W_sparse.Ax.data()), &inc,
               m_W_work.Ax.data(), &inc);
    } else {
      for (int k = 0; k < nnz; ++k)
        m_W_work.Ax[k] = m_W_sparse.Ax[k];
    }

    for (size_t i = 0; i < n; ++i)
      m_W_work.Ax[m_diag_offsets[i]] += inv_gamma_dt;

    m_last_inv_gamma_dt = inv_gamma_dt;
    m_sparse_lu.factorize(m_W_work);
  }

  // ====================================================================
  //  Storage
  // ====================================================================

  // Dense path
  wrapped_matrix_type  m_jac;                            // J (from codegen)
  ad_lu::dense_lu_solver<value_type> m_dense_lu;
  matrix_type          m_W_temp;                         // persistent W buffer

  // Sparse path
  sparse_matrix_type   m_W_sparse;                       // −J (pre-negated by codegen)
  sparse_matrix_type   m_W_work;                         // scratch: W = −J + diag
  std::vector<int>     m_diag_offsets;                    // precomputed diagonal positions
  value_type           m_last_inv_gamma_dt = value_type(0);

  // Sparse LU solver (only instantiated when is_sparse = true)
  using sparse_lu_type = std::conditional_t<is_sparse,
    ad_lu::sparse_lu_solver<value_type>,
    ad_lu::detail_null_sparse_lu>;
  sparse_lu_type       m_sparse_lu;

  // df/dt vector (shared by both paths)
  wrapped_deriv_type   m_dfdt;

  // Lagged state
  bool      m_has_valid_jac = false;
  bool      m_has_valid_lu  = false;
  time_type m_last_dt       = time_type(0);

  // Caches for Jacobian reuse
  wrapped_matrix_type  m_jac_cache;
  wrapped_deriv_type   m_dfdt_cache;
};

// ============================================================================
//  Tag → bool conversion (for backward compatibility with stepper templates
//  that use dense_lu_tag / sparse_lu_tag)
// ============================================================================

template<class Tag>
struct is_sparse_tag : std::false_type {};

template<>
struct is_sparse_tag<sparse_lu_tag> : std::true_type {};

// Convenience: lu_W from tag
template<class Value, class Tag>
using lu_W_from_tag = lu_W<Value, is_sparse_tag<Tag>::value>;

} // namespace cppode

#endif // CPPODE_LU_HPP
