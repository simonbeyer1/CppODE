/*
 CppODE Rosenbrock4 stepper — dense and sparse LU
 ==================================================

 Two paths, dispatched via JacobianPattern template parameter:

 Dense (JacobianPattern = dense_lu_tag, the default):
 Jacobian in ublas::matrix<T>.  LU via ublas::lu_factorize.
 Identical to the original Boost Rosenbrock4.

 Sparse (JacobianPattern = sparse_lu_tag):
 Jacobian in ublas::compressed_matrix<T>.
 LU via cppode::sparse_lu_cache<T> — symbolic analysis once,
 numerical factorization each step.  No dense n×n matrix ever.

 The Jacobian functor's operator() signature determines the path:
 Dense:  void operator()(vec<T>&, matrix<T>&, T&, vec<T>&)
 Sparse: void operator()(vec<T>&, compressed_matrix<T>&, T&, vec<T>&)

 Codegen produces exactly ONE Jacobian functor matching the path.

 Original: Copyright (C) 2011-2013 Karsten Ahnert, Mario Mulansky, Christoph Koke
 Modified: Copyright (C) 2026 Simon Beyer
 Distributed under the Boost Software License, Version 1.0.
 */

#ifndef CPPODE_ROSENBROCK4_HPP_INCLUDED
#define CPPODE_ROSENBROCK4_HPP_INCLUDED

#include <cstddef>
#include <type_traits>

#include <boost/numeric/odeint/util/bind.hpp>
#include <boost/numeric/odeint/util/unwrap_reference.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <boost/numeric/odeint/stepper/stepper_categories.hpp>
#include <boost/numeric/odeint/util/ublas_wrapper.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resizer.hpp>

#include <cppode/cppode_sparse_lu.hpp>

namespace boost {
namespace numeric {
namespace odeint {

// ============================================================================
//  Rosenbrock4 coefficients (Hairer-Wanner)
// ============================================================================

template<class Value>
struct default_rosenbrock_coefficients
{
  typedef Value value_type;
  typedef unsigned short order_type;

  default_rosenbrock_coefficients( void )
    : gamma ( static_cast<value_type>( 0.25 ) ) ,
      d1 ( static_cast<value_type>( 0.25 ) ) ,
      d2 ( static_cast<value_type>( -0.1043 ) ) ,
      d3 ( static_cast<value_type>( 0.1035 ) ) ,
      d4 ( static_cast<value_type>( 0.3620000000000023e-01 ) ) ,
      c2 ( static_cast<value_type>( 0.386 ) ) ,
      c3 ( static_cast<value_type>( 0.21 ) ) ,
      c4 ( static_cast<value_type>( 0.63 ) ) ,
      c21 ( static_cast<value_type>( -0.5668800000000000e+01 ) ) ,
      a21 ( static_cast<value_type>( 0.1544000000000000e+01 ) ) ,
      c31 ( static_cast<value_type>( -0.2430093356833875e+01 ) ) ,
      c32 ( static_cast<value_type>( -0.2063599157091915e+00 ) ) ,
      a31 ( static_cast<value_type>( 0.9466785280815826e+00 ) ) ,
      a32 ( static_cast<value_type>( 0.2557011698983284e+00 ) ) ,
      c41 ( static_cast<value_type>( -0.1073529058151375e+00 ) ) ,
      c42 ( static_cast<value_type>( -0.9594562251023355e+01 ) ) ,
      c43 ( static_cast<value_type>( -0.2047028614809616e+02 ) ) ,
      a41 ( static_cast<value_type>( 0.3314825187068521e+01 ) ) ,
      a42 ( static_cast<value_type>( 0.2896124015972201e+01 ) ) ,
      a43 ( static_cast<value_type>( 0.9986419139977817e+00 ) ) ,
      c51 ( static_cast<value_type>( 0.7496443313967647e+01 ) ) ,
      c52 ( static_cast<value_type>( -0.1024680431464352e+02 ) ) ,
      c53 ( static_cast<value_type>( -0.3399990352819905e+02 ) ) ,
      c54 ( static_cast<value_type>(  0.1170890893206160e+02 ) ) ,
      a51 ( static_cast<value_type>( 0.1221224509226641e+01 ) ) ,
      a52 ( static_cast<value_type>( 0.6019134481288629e+01 ) ) ,
      a53 ( static_cast<value_type>( 0.1253708332932087e+02 ) ) ,
      a54 ( static_cast<value_type>( -0.6878860361058950e+00 ) ) ,
      c61 ( static_cast<value_type>( 0.8083246795921522e+01 ) ) ,
      c62 ( static_cast<value_type>( -0.7981132988064893e+01 ) ) ,
      c63 ( static_cast<value_type>( -0.3152159432874371e+02 ) ) ,
      c64 ( static_cast<value_type>( 0.1631930543123136e+02 ) ) ,
      c65 ( static_cast<value_type>( -0.6058818238834054e+01 ) ) ,
      d21 ( static_cast<value_type>( 0.1012623508344586e+02 ) ) ,
      d22 ( static_cast<value_type>( -0.7487995877610167e+01 ) ) ,
      d23 ( static_cast<value_type>( -0.3480091861555747e+02 ) ) ,
      d24 ( static_cast<value_type>( -0.7992771707568823e+01 ) ) ,
      d25 ( static_cast<value_type>( 0.1025137723295662e+01 ) ) ,
      d31 ( static_cast<value_type>( -0.6762803392801253e+00 ) ) ,
      d32 ( static_cast<value_type>( 0.6087714651680015e+01 ) ) ,
      d33 ( static_cast<value_type>( 0.1643084320892478e+02 ) ) ,
      d34 ( static_cast<value_type>( 0.2476722511418386e+02 ) ) ,
      d35 ( static_cast<value_type>( -0.6594389125716872e+01 ) )
  {}

  const value_type gamma;
  const value_type d1 , d2 , d3 , d4;
  const value_type c2 , c3 , c4;
  const value_type c21;
  const value_type a21;
  const value_type c31 , c32;
  const value_type a31 , a32;
  const value_type c41 , c42 , c43;
  const value_type a41 , a42 , a43;
  const value_type c51 , c52 , c53 , c54;
  const value_type a51 , a52 , a53 , a54;
  const value_type c61 , c62 , c63 , c64 , c65;
  const value_type d21 , d22 , d23 , d24 , d25;
  const value_type d31 , d32 , d33 , d34 , d35;

  static const order_type stepper_order = 4;
  static const order_type error_order = 3;
};


// ============================================================================
//  Tags
// ============================================================================

struct dense_lu_tag {};
struct sparse_lu_tag {};

// ============================================================================
//  jacobian_hint — tells the stepper what to recompute
//
//  recompute_all:    Fresh Jacobian evaluation + LU factorization (default).
//  reuse_jacobian:   Skip Jacobian evaluation, rebuild W with new dt, re-LU.
//                    Valid when the state hasn't changed much (e.g. after
//                    a rejected step — same x, different dt).
//  reuse_lu:         Skip both Jacobian and LU.  Reuse the existing
//                    factorization from the previous step entirely.
//                    Valid when dt hasn't changed significantly.
//
//  The controller sets the hint; the stepper obeys it.
// ============================================================================

enum class jacobian_hint : unsigned char {
  recompute_all   = 0,
  reuse_jacobian  = 1,
  reuse_lu        = 2
};


// ============================================================================
//  rosenbrock4<Value, JacobianPattern, Coefficients, Resizer>
//
//  JacobianPattern = dense_lu_tag  → dense ublas::matrix LU
//  JacobianPattern = sparse_lu_tag → compressed_matrix + sparse_lu_cache
// ============================================================================

template<
  class Value,
  class JacobianPattern = dense_lu_tag,
  class Coefficients = default_rosenbrock_coefficients<Value>,
  class Resizer = initially_resizer
>
class rosenbrock4
{
public:

  typedef Value value_type;
  typedef boost::numeric::ublas::vector<value_type> state_type;
  typedef state_type deriv_type;
  typedef value_type time_type;
  typedef boost::numeric::ublas::matrix<value_type> matrix_type;
  typedef boost::numeric::ublas::compressed_matrix<value_type> sparse_matrix_type;
  typedef boost::numeric::ublas::permutation_matrix<size_t> pmatrix_type;
  typedef Resizer resizer_type;
  typedef Coefficients rosenbrock_coefficients;
  typedef stepper_tag stepper_category;
  typedef unsigned short order_type;

  typedef state_wrapper<state_type> wrapped_state_type;
  typedef state_wrapper<deriv_type> wrapped_deriv_type;
  typedef state_wrapper<matrix_type> wrapped_matrix_type;
  typedef state_wrapper<pmatrix_type> wrapped_pmatrix_type;

  typedef rosenbrock4<Value, JacobianPattern, Coefficients, Resizer> stepper_type;

  static constexpr bool is_sparse =
    std::is_same<JacobianPattern, sparse_lu_tag>::value;
  static constexpr bool is_dense =
    std::is_same<JacobianPattern, dense_lu_tag>::value;

  const static order_type stepper_order = rosenbrock_coefficients::stepper_order;
  const static order_type error_order = rosenbrock_coefficients::error_order;

  rosenbrock4( void )
    : m_resizer() , m_x_err_resizer() ,
      m_dfdt() , m_dxdt() , m_dxdtnew() ,
      m_g1() , m_g2() , m_g3() , m_g4() , m_g5() ,
      m_cont3() , m_cont4() , m_xtmp() , m_x_err() ,
      m_coef() ,
      m_has_valid_jac( false ) , m_has_valid_lu( false ) ,
      m_last_dt( 0 )
  { }

  order_type order() const { return stepper_order; }

  // ====================================================================
  //  do_step (with error output)
  //
  //  hint controls Jacobian/LU reuse:
  //    recompute_all  — fresh J + fresh LU (always correct)
  //    reuse_jacobian — skip J evaluation, rebuild W with new dt, re-LU
  //    reuse_lu       — skip both J and LU entirely
  // ====================================================================

  template<class Sys>
  void do_step(
      Sys system,
      const state_type& x, time_type t,
      state_type& xout, time_type dt,
      state_type& xerr,
      jacobian_hint hint = jacobian_hint::recompute_all)
  {
    typedef typename odeint::unwrap_reference<Sys>::type system_type;
    typedef typename odeint::unwrap_reference<
      typename system_type::first_type>::type deriv_func_type;
    typedef typename odeint::unwrap_reference<
      typename system_type::second_type>::type jacobi_func_type;

    system_type&      sys         = system;
    deriv_func_type&  deriv_func  = sys.first;
    jacobi_func_type& jacobi_func = sys.second;

    const size_t n = x.size();

    m_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_impl<state_type>(
          std::forward<decltype(arg)>(arg));
    });

    // Always evaluate f(x,t) — it's cheap and needed for the stages
    deriv_func(x, m_dxdt.m_v, t);

    // --- Jacobian + LU, controlled by hint ---
    if (hint == jacobian_hint::reuse_lu && m_has_valid_lu)
    {
      // Reuse everything — skip J and LU
    }
    else if (hint == jacobian_hint::reuse_jacobian && m_has_valid_jac)
    {
      // Jacobian is cached — only rebuild W and re-factorize with new dt
      refactorize_W_from_cache(n, dt);
      m_has_valid_lu = true;
      m_last_dt = dt;
    }
    else
    {
      // Full recompute
      call_jacobian(jacobi_func, x, t);
      cache_jacobian(n);
      factorize_W(n, dt);
      m_has_valid_jac = true;
      m_has_valid_lu = true;
      m_last_dt = dt;
    }

    // --- Stage 1 ---
    for (size_t i = 0; i < n; ++i)
      m_g1.m_v[i] = m_dxdt.m_v[i] + dt * m_coef.d1 * m_dfdt.m_v[i];
    lu_solve(m_g1.m_v);

    // --- Stage 2 ---
    for (size_t i = 0; i < n; ++i)
      m_xtmp.m_v[i] = x[i] + m_coef.a21 * m_g1.m_v[i];
    deriv_func(m_xtmp.m_v, m_dxdtnew.m_v, t + m_coef.c2 * dt);
    for (size_t i = 0; i < n; ++i)
      m_g2.m_v[i] = m_dxdtnew.m_v[i] + dt * m_coef.d2 * m_dfdt.m_v[i]
    + m_coef.c21 * m_g1.m_v[i] / dt;
    lu_solve(m_g2.m_v);

    // --- Stage 3 ---
    for (size_t i = 0; i < n; ++i)
      m_xtmp.m_v[i] = x[i] + m_coef.a31 * m_g1.m_v[i]
    + m_coef.a32 * m_g2.m_v[i];
    deriv_func(m_xtmp.m_v, m_dxdtnew.m_v, t + m_coef.c3 * dt);
    for (size_t i = 0; i < n; ++i)
      m_g3.m_v[i] = m_dxdtnew.m_v[i] + dt * m_coef.d3 * m_dfdt.m_v[i]
    + (m_coef.c31 * m_g1.m_v[i] + m_coef.c32 * m_g2.m_v[i]) / dt;
    lu_solve(m_g3.m_v);

    // --- Stage 4 ---
    for (size_t i = 0; i < n; ++i)
      m_xtmp.m_v[i] = x[i] + m_coef.a41 * m_g1.m_v[i]
    + m_coef.a42 * m_g2.m_v[i] + m_coef.a43 * m_g3.m_v[i];
    deriv_func(m_xtmp.m_v, m_dxdtnew.m_v, t + m_coef.c4 * dt);
    for (size_t i = 0; i < n; ++i)
      m_g4.m_v[i] = m_dxdtnew.m_v[i] + dt * m_coef.d4 * m_dfdt.m_v[i]
    + (m_coef.c41 * m_g1.m_v[i] + m_coef.c42 * m_g2.m_v[i]
         + m_coef.c43 * m_g3.m_v[i]) / dt;
         lu_solve(m_g4.m_v);

         // --- Stage 5 ---
         for (size_t i = 0; i < n; ++i)
           m_xtmp.m_v[i] = x[i] + m_coef.a51 * m_g1.m_v[i]
         + m_coef.a52 * m_g2.m_v[i] + m_coef.a53 * m_g3.m_v[i]
         + m_coef.a54 * m_g4.m_v[i];
         deriv_func(m_xtmp.m_v, m_dxdtnew.m_v, t + dt);
         for (size_t i = 0; i < n; ++i)
           m_g5.m_v[i] = m_dxdtnew.m_v[i]
         + (m_coef.c51 * m_g1.m_v[i] + m_coef.c52 * m_g2.m_v[i]
              + m_coef.c53 * m_g3.m_v[i] + m_coef.c54 * m_g4.m_v[i]) / dt;
              lu_solve(m_g5.m_v);

              // --- Error estimate (stage 6) ---
              for (size_t i = 0; i < n; ++i)
                m_xtmp.m_v[i] += m_g5.m_v[i];
              deriv_func(m_xtmp.m_v, m_dxdtnew.m_v, t + dt);
              for (size_t i = 0; i < n; ++i)
                xerr[i] = m_dxdtnew.m_v[i]
              + (m_coef.c61 * m_g1.m_v[i] + m_coef.c62 * m_g2.m_v[i]
                   + m_coef.c63 * m_g3.m_v[i] + m_coef.c64 * m_g4.m_v[i]
                   + m_coef.c65 * m_g5.m_v[i]) / dt;
                   lu_solve(xerr);

                   // --- Solution ---
                   for (size_t i = 0; i < n; ++i)
                     xout[i] = m_xtmp.m_v[i] + xerr[i];
  }

  // ====================================================================
  //  Convenience overloads
  // ====================================================================

  template<class Sys>
  void do_step(Sys system, state_type& x, time_type t,
               time_type dt, state_type& xerr,
               jacobian_hint hint = jacobian_hint::recompute_all)
  { do_step(system, x, t, x, dt, xerr, hint); }

  template<class Sys>
  void do_step(Sys system, const state_type& x, time_type t,
               state_type& xout, time_type dt,
               jacobian_hint hint = jacobian_hint::recompute_all)
  {
    m_x_err_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_x_err<state_type>(std::forward<decltype(arg)>(arg));
    });
    do_step(system, x, t, xout, dt, m_x_err.m_v, hint);
  }

  template<class Sys>
  void do_step(Sys system, state_type& x, time_type t, time_type dt,
               jacobian_hint hint = jacobian_hint::recompute_all)
  {
    m_x_err_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_x_err<state_type>(std::forward<decltype(arg)>(arg));
    });
    do_step(system, x, t, dt, m_x_err.m_v, hint);
  }

  // ====================================================================
  //  Dense output
  // ====================================================================

  void prepare_dense_output()
  {
    const size_t n = m_g1.m_v.size();
    for (size_t i = 0; i < n; ++i) {
      m_cont3.m_v[i] = m_coef.d21 * m_g1.m_v[i] + m_coef.d22 * m_g2.m_v[i]
      + m_coef.d23 * m_g3.m_v[i] + m_coef.d24 * m_g4.m_v[i]
      + m_coef.d25 * m_g5.m_v[i];
      m_cont4.m_v[i] = m_coef.d31 * m_g1.m_v[i] + m_coef.d32 * m_g2.m_v[i]
      + m_coef.d33 * m_g3.m_v[i] + m_coef.d34 * m_g4.m_v[i]
      + m_coef.d35 * m_g5.m_v[i];
    }
  }

  void calc_state(time_type t, state_type& x,
                  const state_type& x_old, time_type t_old,
                  const state_type& x_new, time_type t_new)
  {
    const size_t n = m_g1.m_v.size();
    time_type dt = t_new - t_old;
    time_type s  = (t - t_old) / dt;
    time_type s1 = 1.0 - s;
    for (size_t i = 0; i < n; ++i)
      x[i] = x_old[i] * s1 + s * (x_new[i] + s1 * (m_cont3.m_v[i] + s * m_cont4.m_v[i]));
  }

  template<class StateType>
  void adjust_size(const StateType& x)
  { resize_impl(x); resize_x_err(x); }

  // ====================================================================
  //  Lagged Jacobian/LU: public interface for controller
  // ====================================================================

  /// Invalidate cached Jacobian and LU (e.g. after events)
  void invalidate_lu()
  {
    m_has_valid_jac = false;
    m_has_valid_lu = false;
  }

  /// Query whether a valid Jacobian is cached
  bool has_valid_jacobian() const { return m_has_valid_jac; }

  /// Query whether a valid LU factorization is cached
  bool has_valid_lu() const { return m_has_valid_lu; }

  /// The dt used for the current cached LU
  time_type last_factorized_dt() const { return m_last_dt; }


protected:

  template<class StateIn>
  bool resize_impl(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(m_dxdt,    x, typename is_resizeable<deriv_type>::type());
    resized |= adjust_size_by_resizeability(m_dfdt,    x, typename is_resizeable<deriv_type>::type());
    resized |= adjust_size_by_resizeability(m_dxdtnew, x, typename is_resizeable<deriv_type>::type());
    resized |= adjust_size_by_resizeability(m_xtmp,    x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(m_g1,      x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(m_g2,      x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(m_g3,      x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(m_g4,      x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(m_g5,      x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(m_cont3,   x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(m_cont4,   x, typename is_resizeable<state_type>::type());
    resized |= resize_W(x);
    resized |= resize_jac_cache(x);
    if (resized) invalidate_lu();
    return resized;
  }

  template<class StateIn>
  bool resize_x_err(const StateIn& x)
  { return adjust_size_by_resizeability(m_x_err, x, typename is_resizeable<state_type>::type()); }


private:

  // ====================================================================
  //  resize_W: dense
  // ====================================================================

  template<class StateIn, class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, dense_lu_tag>::value, bool>::type
  resize_W(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(m_jac, x, typename is_resizeable<matrix_type>::type());
    resized |= adjust_size_by_resizeability(m_pm, x, typename is_resizeable<pmatrix_type>::type());
    return resized;
  }

  // ====================================================================
  //  resize_W: sparse (compressed_matrix is resized on first Jacobian call)
  // ====================================================================

  template<class StateIn, class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, sparse_lu_tag>::value, bool>::type
  resize_W(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(m_pm, x, typename is_resizeable<pmatrix_type>::type());
    const std::size_t n = x.size();
    if (m_W_sparse.size1() != n || m_W_sparse.size2() != n) {
      m_W_sparse.clear();
      m_W_sparse.resize(n, n, false);
      m_W_sparse.reserve(n);
      resized = true;
    }
    return resized;
  }

  // ====================================================================
  //  call_jacobian: dense → writes to ublas::matrix
  // ====================================================================

  template<class JacFunc, class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, dense_lu_tag>::value>::type
  call_jacobian(JacFunc& jf, const state_type& x, time_type t)
  { jf(x, m_jac.m_v, t, m_dfdt.m_v); }

  // ====================================================================
  //  call_jacobian: sparse → writes to compressed_matrix
  // ====================================================================

  template<class JacFunc, class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, sparse_lu_tag>::value>::type
  call_jacobian(JacFunc& jf, const state_type& x, time_type t)
  { jf(x, m_W_sparse, t, m_dfdt.m_v); }

  // ====================================================================
  //  factorize_W: dense
  // ====================================================================

  template<class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, dense_lu_tag>::value>::type
  factorize_W(size_t n, time_type dt)
  {
    for (size_t i = 0; i < n; ++i) m_pm.m_v(i) = i;
    m_jac.m_v *= -1.0;
    m_jac.m_v += 1.0 / m_coef.gamma / dt
    * boost::numeric::ublas::identity_matrix<value_type>(n);
    boost::numeric::ublas::lu_factorize(m_jac.m_v, m_pm.m_v);
  }

  // ====================================================================
  //  factorize_W: sparse
  //
  //  Build W = (1/γh)I − J in compressed_matrix, then sparse LU.
  //  The Jacobian functor already wrote J entries into m_W_sparse.
  //  We negate them and add the diagonal.
  // ====================================================================

  template<class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, sparse_lu_tag>::value>::type
  factorize_W(size_t n, time_type dt)
  {
    const value_type inv_gamma_dt = 1.0 / (m_coef.gamma * dt);

    // Negate all entries and add 1/(γh) to diagonal
    // compressed_matrix allows iteration + modification of existing entries
    for (auto it1 = m_W_sparse.begin1(); it1 != m_W_sparse.end1(); ++it1)
      for (auto it2 = it1.begin(); it2 != it1.end(); ++it2)
        *it2 = -*it2;

    // Add diagonal (compressed_matrix::operator() inserts if missing)
    for (size_t i = 0; i < n; ++i)
      m_W_sparse(i, i) += inv_gamma_dt;

    // Sparse LU with symbolic caching
    for (size_t i = 0; i < n; ++i) m_pm.m_v(i) = i;
    m_lu_cache.factorize(m_W_sparse, m_pm.m_v);
  }

  // ====================================================================
  //  lu_solve: dense
  // ====================================================================

  template<class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, dense_lu_tag>::value>::type
  lu_solve(state_type& b)
  { boost::numeric::ublas::lu_substitute(m_jac.m_v, m_pm.m_v, b); }

  // ====================================================================
  //  lu_solve: sparse
  // ====================================================================

  template<class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, sparse_lu_tag>::value>::type
  lu_solve(state_type& b)
  { m_lu_cache.substitute(b); }

  // ====================================================================
  //  cache_jacobian / refactorize_W_from_cache: dense
  //
  //  Dense factorize_W destroys m_jac.m_v in-place, so we keep a
  //  pristine copy of J in m_jac_cache for lagged re-factorization.
  // ====================================================================

  template<class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, dense_lu_tag>::value>::type
  cache_jacobian(size_t /*n*/)
  {
    m_jac_cache.m_v = m_jac.m_v;    // deep copy of the raw Jacobian
    m_dfdt_cache.m_v = m_dfdt.m_v;  // cache dfdt too
  }

  template<class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, dense_lu_tag>::value>::type
  refactorize_W_from_cache(size_t n, time_type dt)
  {
    // Restore J from cache, then do the standard factorize
    m_jac.m_v = m_jac_cache.m_v;
    m_dfdt.m_v = m_dfdt_cache.m_v;
    factorize_W(n, dt);
  }

  // ====================================================================
  //  cache_jacobian / refactorize_W_from_cache: sparse
  //
  //  Sparse factorize_W also modifies m_W_sparse in-place (negates +
  //  adds diagonal).  We cache the raw Jacobian entries.
  // ====================================================================

  template<class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, sparse_lu_tag>::value>::type
  cache_jacobian(size_t /*n*/)
  {
    m_W_sparse_cache = m_W_sparse;    // deep copy of raw J entries
    m_dfdt_cache.m_v = m_dfdt.m_v;
  }

  template<class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, sparse_lu_tag>::value>::type
  refactorize_W_from_cache(size_t n, time_type dt)
  {
    // Restore raw J from cache, then do the standard factorize
    m_W_sparse = m_W_sparse_cache;
    m_dfdt.m_v = m_dfdt_cache.m_v;
    factorize_W(n, dt);
  }

  // ====================================================================
  //  resize_jac_cache: dense
  // ====================================================================

  template<class StateIn, class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, dense_lu_tag>::value, bool>::type
  resize_jac_cache(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(m_jac_cache, x, typename is_resizeable<matrix_type>::type());
    resized |= adjust_size_by_resizeability(m_dfdt_cache, x, typename is_resizeable<deriv_type>::type());
    return resized;
  }

  // ====================================================================
  //  resize_jac_cache: sparse
  // ====================================================================

  template<class StateIn, class P = JacobianPattern>
  typename std::enable_if<std::is_same<P, sparse_lu_tag>::value, bool>::type
  resize_jac_cache(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(m_dfdt_cache, x, typename is_resizeable<deriv_type>::type());
    // m_W_sparse_cache is resized on first copy
    return resized;
  }


  // ====================================================================
  //  Members
  // ====================================================================

  resizer_type m_resizer;
  resizer_type m_x_err_resizer;

  // Dense path: Jacobian matrix + permutation
  wrapped_matrix_type  m_jac;
  wrapped_pmatrix_type m_pm;

  // Sparse path: compressed_matrix + LU cache
  sparse_matrix_type m_W_sparse;
  cppode::sparse_lu_cache<value_type> m_lu_cache;

  wrapped_deriv_type   m_dfdt, m_dxdt, m_dxdtnew;
  wrapped_state_type   m_g1, m_g2, m_g3, m_g4, m_g5;
  wrapped_state_type   m_cont3, m_cont4;
  wrapped_state_type   m_xtmp;
  wrapped_state_type   m_x_err;

  const rosenbrock_coefficients m_coef;

  // --- Lagged Jacobian/LU state ---
  bool      m_has_valid_jac;    ///< Cached Jacobian is usable
  bool      m_has_valid_lu;     ///< Cached LU factorization is usable
  time_type m_last_dt;          ///< dt used for the current cached W/LU

  // Dense path: pristine copy of J (before factorize_W destroys it)
  wrapped_matrix_type  m_jac_cache;
  // Sparse path: pristine copy of compressed_matrix J
  sparse_matrix_type   m_W_sparse_cache;
  // Both paths: cached dfdt
  wrapped_deriv_type   m_dfdt_cache;
};


} // namespace odeint
} // namespace numeric
} // namespace boost


// ============================================================================
//  Generation helpers
// ============================================================================

namespace boost {
namespace numeric {
namespace odeint {

template<class Stepper>
class rosenbrock4_controller_pi;

template<class ControlledStepper>
class rosenbrock4_dense_output_pi;

template<class Value, class JacobianPattern>
struct get_controller< rosenbrock4<Value, JacobianPattern> >
{
  typedef rosenbrock4_controller_pi<
    rosenbrock4<Value, JacobianPattern>
  > type;
};

template<class Value, class JacobianPattern>
struct get_dense_output< rosenbrock4<Value, JacobianPattern> >
{
  typedef rosenbrock4_dense_output_pi<
    rosenbrock4_controller_pi<
      rosenbrock4<Value, JacobianPattern>
    >
  > type;
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_ROSENBROCK4_HPP_INCLUDED
