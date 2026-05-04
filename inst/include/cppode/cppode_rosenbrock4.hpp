/*
 Rosenbrock4 stepper

 The stepper architecture is derived from Boost.Odeint's rosenbrock4
 by Karsten Ahnert, Mario Mulansky, and Christoph Koke (2011–2013),
 distributed under the Boost Software License, Version 1.0.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_ROSENBROCK4_HPP
#define CPPODE_ROSENBROCK4_HPP

#include <cstddef>
#include <type_traits>

#include <cppode/cppode_types.hpp>
#include <cppode/cppode_odeint_compat.hpp>
#include <cppode/cppode_lu.hpp>
#include <cppode/cppode_ad_lu.hpp>   // for ad_lu::scalar_type_t
#include <cppode/cppode_dual_slab.hpp>
#include <cppode/cppode_stage_matrix.hpp>
#include <cppode/cppode_profiler.hpp>

namespace cppode {

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
//  Tags (re-exported from cppode_lu.hpp for external use)
// ============================================================================

// dense_lu_tag and sparse_lu_tag are defined in cppode_lu.hpp

// ============================================================================
//  jacobian_hint
// ============================================================================

enum class jacobian_hint : unsigned char {
  recompute_all   = 0,
    reuse_jacobian  = 1,
    reuse_lu        = 2
};

// ============================================================================
//  rosenbrock4<Value, JacobianPattern, Coefficients, Resizer>
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

  typedef Value                              value_type;
  typedef std::vector<value_type>            state_type;
  typedef state_type                         deriv_type;
  typedef ad_lu::scalar_type_t<value_type>   time_type;
  typedef dense_matrix<value_type>           matrix_type;
  typedef csc_matrix<value_type>             sparse_matrix_type;
  typedef Resizer                            resizer_type;
  typedef Coefficients                       rosenbrock_coefficients;
  typedef stepper_tag                        stepper_category;
  typedef unsigned short                     order_type;

  typedef state_wrapper<state_type>          wrapped_state_type;
  typedef state_wrapper<deriv_type>          wrapped_deriv_type;
  typedef state_wrapper<matrix_type>         wrapped_matrix_type;

  typedef rosenbrock4<Value, JacobianPattern, Coefficients, Resizer> stepper_type;

  static constexpr bool is_sparse = is_sparse_tag<JacobianPattern>::value;
  static constexpr bool is_dense = !is_sparse;

  using lu_type = lu_W<Value, is_sparse>;

  const static order_type stepper_order = rosenbrock_coefficients::stepper_order;
  const static order_type error_order = rosenbrock_coefficients::error_order;

  rosenbrock4( void )
    : m_lu() ,
      m_resizer() , m_x_err_resizer() ,
      m_dxdt() ,
      m_g1() , m_g2() , m_g3() , m_g4() , m_g5() ,
      m_cont3() , m_cont4() , m_xtmp() , m_x_err() ,
      m_coef() ,
      m_n_fevals( 0 ) , m_n_jevals( 0 ) , m_n_lu_setups( 0 )
  { }

  // Same move/non-copy semantics as the multistepper: copying the slab
  // would build a fresh tangent block while the dual elements still
  // point at the original — UB. Move-assignment preserves data() of
  // both the slab storage and the std::vector<dual> elements.
  rosenbrock4(const rosenbrock4&)            = delete;
  rosenbrock4& operator=(const rosenbrock4&) = delete;
  rosenbrock4(rosenbrock4&&)                 = default;
  rosenbrock4& operator=(rosenbrock4&&)      = default;

  // ====================================================================
  //  prepare_sensitivities
  //
  //  Stores n_sens and primes the SoA tangent slabs for the heap-AD
  //  path (dual<S, 0>). No-op for non-dynamic-dual value_type.
  //
  //  Only the rosenbrock4-owned vectors are primed here; the external
  //  xerr passed to do_step (the controller's m_xerr) is handled
  //  separately by the controller.
  // ====================================================================

  void prepare_sensitivities(unsigned n_sens)
  {
    m_n_sens = n_sens;
    if constexpr (detail::is_dynamic_dual<value_type>::value) {
      if (n_sens == 0) return;
      auto prime = [n_sens](auto& wrapped, auto& slab) {
        if (!wrapped.m_v.empty())
          slab.prime(wrapped.m_v,
                     static_cast<unsigned>(wrapped.m_v.size()), n_sens);
      };
      prime(m_dxdt,  m_dxdt_slab);

      // The five Rosenbrock stages g1..g5 live in one contiguous tangent
      // block via m_G. Per-stage slab views are accessed as m_G.slab(j)
      // for the existing vec_*_with_slab call sites.
      const std::size_t n_g = m_g1.m_v.size();
      bool gs_ready = (n_g > 0)
                   && (m_g2.m_v.size() == n_g) && (m_g3.m_v.size() == n_g)
                   && (m_g4.m_v.size() == n_g) && (m_g5.m_v.size() == n_g);
      if (gs_ready) {
        std::array<std::vector<value_type>*, 5> facades{
          &m_g1.m_v, &m_g2.m_v, &m_g3.m_v, &m_g4.m_v, &m_g5.m_v
        };
        m_G.prime(facades, static_cast<unsigned>(n_g), n_sens);
      }

      prime(m_cont3, m_cont3_slab);
      prime(m_cont4, m_cont4_slab);
      prime(m_xtmp,  m_xtmp_slab);
      prime(m_x_err, m_x_err_slab);
    }
  }

  unsigned n_sens() const noexcept { return m_n_sens; }

  order_type order() const { return stepper_order; }

  int n_fevals() const { return m_n_fevals; }
  int n_jevals() const { return m_n_jevals; }
  int n_setups() const { return m_n_lu_setups; }
  void reset_counters() { m_n_fevals = 0; m_n_jevals = 0; m_n_lu_setups = 0; }

  // ====================================================================
  //  do_step (with error output)
  // ====================================================================

  template<class Sys, class TimeArg>
  void do_step(
      Sys& system,
      const state_type& x, TimeArg t,
      state_type& xout, TimeArg dt, state_type& xerr,
      jacobian_hint hint = jacobian_hint::recompute_all)
  {
    auto& deriv_func = system.first;
    auto& jacobi_func = system.second;

    const size_t n = x.size();

    // Extract scalar time values (strip AD derivatives from incoming time args)
    const time_type t_s = static_cast<time_type>(ad_lu::scalar_value(t));
    const time_type dt_s = static_cast<time_type>(ad_lu::scalar_value(dt));

    m_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_impl(std::forward<decltype(arg)>(arg));
    });

    // --- Initial derivative ---
    { auto _tp = m_prof.timer(prof_cat::f_eval);
      deriv_func(x, m_dxdt.m_v, value_type(t_s)); }
    ++m_n_fevals;

    const value_type inv_gamma_dt =
    static_cast<value_type>(1) / (m_coef.gamma * dt_s);

    // --- Jacobian / LU ---
    if (hint == jacobian_hint::reuse_lu && m_lu.has_valid_lu())
    {
      // Reuse everything
    }
    else if (hint == jacobian_hint::reuse_jacobian && m_lu.has_valid_jacobian())
    {
      { auto _tp = m_prof.timer(prof_cat::lu_factor);
        m_lu.refactorize_W_from_cache(n, inv_gamma_dt); }
      ++m_n_lu_setups;
      m_lu.set_lu_valid(dt_s);
    }
    else
    {
      { auto _tp = m_prof.timer(prof_cat::jac_eval);
        m_lu.call_jacobian(jacobi_func, x, t); }
      ++m_n_jevals;
      m_lu.cache_jacobian(n);
      { auto _tp = m_prof.timer(prof_cat::lu_factor);
        m_lu.factorize_W(n, inv_gamma_dt); }
      ++m_n_lu_setups;
      m_lu.set_jacobian_valid();
      m_lu.set_lu_valid(dt_s);
    }

    // --- Stage 1 ---
    vec_copy_with_slab(m_g1.m_v, m_G.slab(0), m_dxdt.m_v, m_dxdt_slab);
    vec_axpy_with_slab(m_g1.m_v, m_G.slab(0),
                       dt_s * ad_lu::scalar_value(m_coef.d1),
                       m_lu.dfdt_mut(), m_dfdt_unslabbed);
    { auto _tp = m_prof.timer(prof_cat::lu_solve);
      m_lu.solve(m_g1.m_v); }

    // --- Stage 2 ---
    vec_copy_with_slab(m_xtmp.m_v, m_xtmp_slab, x, m_x_in_unslabbed);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a21),
                       m_g1.m_v, m_G.slab(0));
    { auto _tp = m_prof.timer(prof_cat::f_eval);
      deriv_func(m_xtmp.m_v, m_g2.m_v, value_type(t_s + m_coef.c2 * dt_s)); }
    ++m_n_fevals;
    vec_axpy_with_slab(m_g2.m_v, m_G.slab(1),
                       dt_s * ad_lu::scalar_value(m_coef.d2),
                       m_lu.dfdt_mut(), m_dfdt_unslabbed);
    vec_axpy_with_slab(m_g2.m_v, m_G.slab(1),
                       ad_lu::scalar_value(m_coef.c21) / dt_s,
                       m_g1.m_v, m_G.slab(0));
    { auto _tp = m_prof.timer(prof_cat::lu_solve);
      m_lu.solve(m_g2.m_v); }

    // --- Stage 3 ---
    vec_copy_with_slab(m_xtmp.m_v, m_xtmp_slab, x, m_x_in_unslabbed);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a31),
                       m_g1.m_v, m_G.slab(0));
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a32),
                       m_g2.m_v, m_G.slab(1));
    { auto _tp = m_prof.timer(prof_cat::f_eval);
      deriv_func(m_xtmp.m_v, m_g3.m_v, value_type(t_s + m_coef.c3 * dt_s)); }
    ++m_n_fevals;
    vec_axpy_with_slab(m_g3.m_v, m_G.slab(2),
                       dt_s * ad_lu::scalar_value(m_coef.d3),
                       m_lu.dfdt_mut(), m_dfdt_unslabbed);
    vec_axpy_with_slab(m_g3.m_v, m_G.slab(2),
                       ad_lu::scalar_value(m_coef.c31) / dt_s,
                       m_g1.m_v, m_G.slab(0));
    vec_axpy_with_slab(m_g3.m_v, m_G.slab(2),
                       ad_lu::scalar_value(m_coef.c32) / dt_s,
                       m_g2.m_v, m_G.slab(1));
    { auto _tp = m_prof.timer(prof_cat::lu_solve);
      m_lu.solve(m_g3.m_v); }

    // --- Stage 4 ---
    vec_copy_with_slab(m_xtmp.m_v, m_xtmp_slab, x, m_x_in_unslabbed);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a41),
                       m_g1.m_v, m_G.slab(0));
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a42),
                       m_g2.m_v, m_G.slab(1));
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a43),
                       m_g3.m_v, m_G.slab(2));
    { auto _tp = m_prof.timer(prof_cat::f_eval);
      deriv_func(m_xtmp.m_v, m_g4.m_v, value_type(t_s + m_coef.c4 * dt_s)); }
    ++m_n_fevals;
    vec_axpy_with_slab(m_g4.m_v, m_G.slab(3),
                       dt_s * ad_lu::scalar_value(m_coef.d4),
                       m_lu.dfdt_mut(), m_dfdt_unslabbed);
    vec_axpy_with_slab(m_g4.m_v, m_G.slab(3),
                       ad_lu::scalar_value(m_coef.c41) / dt_s,
                       m_g1.m_v, m_G.slab(0));
    vec_axpy_with_slab(m_g4.m_v, m_G.slab(3),
                       ad_lu::scalar_value(m_coef.c42) / dt_s,
                       m_g2.m_v, m_G.slab(1));
    vec_axpy_with_slab(m_g4.m_v, m_G.slab(3),
                       ad_lu::scalar_value(m_coef.c43) / dt_s,
                       m_g3.m_v, m_G.slab(2));
    { auto _tp = m_prof.timer(prof_cat::lu_solve);
      m_lu.solve(m_g4.m_v); }

    // --- Stage 5 ---
    vec_copy_with_slab(m_xtmp.m_v, m_xtmp_slab, x, m_x_in_unslabbed);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a51),
                       m_g1.m_v, m_G.slab(0));
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a52),
                       m_g2.m_v, m_G.slab(1));
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a53),
                       m_g3.m_v, m_G.slab(2));
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       ad_lu::scalar_value(m_coef.a54),
                       m_g4.m_v, m_G.slab(3));
    { auto _tp = m_prof.timer(prof_cat::f_eval);
      deriv_func(m_xtmp.m_v, m_g5.m_v, value_type(t_s + dt_s)); }
    ++m_n_fevals;
    vec_axpy_with_slab(m_g5.m_v, m_G.slab(4),
                       ad_lu::scalar_value(m_coef.c51) / dt_s,
                       m_g1.m_v, m_G.slab(0));
    vec_axpy_with_slab(m_g5.m_v, m_G.slab(4),
                       ad_lu::scalar_value(m_coef.c52) / dt_s,
                       m_g2.m_v, m_G.slab(1));
    vec_axpy_with_slab(m_g5.m_v, m_G.slab(4),
                       ad_lu::scalar_value(m_coef.c53) / dt_s,
                       m_g3.m_v, m_G.slab(2));
    vec_axpy_with_slab(m_g5.m_v, m_G.slab(4),
                       ad_lu::scalar_value(m_coef.c54) / dt_s,
                       m_g4.m_v, m_G.slab(3));
    { auto _tp = m_prof.timer(prof_cat::lu_solve);
      m_lu.solve(m_g5.m_v); }

    // --- Error estimate (stage 6) ---
    // Uses the Hairer-Wanner 6-stage formulation: an additional
    // f-evaluation and W⁻¹ solve to produce the embedded error.
    // The error is added to the solution (local extrapolation).
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       1.0,
                       m_g5.m_v, m_G.slab(4));
    { auto _tp = m_prof.timer(prof_cat::f_eval);
      deriv_func(m_xtmp.m_v, xerr, value_type(t_s + dt_s)); }
    ++m_n_fevals;
    // The error vector xerr is the controller-owned `m_xerr` and is NOT
    // slab-bound. To keep the AXPY hot path free of arena allocations, we
    // pass scalar (double) alphas — `ad_lu::scalar_value` extracts the
    // value component of m_coef.cXX (constants have zero tangents anyway)
    // so vec_axpy reduces to `y[i] += S * x[i]` over plain doubles inside
    // the dual ET path.
    vec_axpy(xerr, ad_lu::scalar_value(m_coef.c61) / dt_s, m_g1.m_v);
    vec_axpy(xerr, ad_lu::scalar_value(m_coef.c62) / dt_s, m_g2.m_v);
    vec_axpy(xerr, ad_lu::scalar_value(m_coef.c63) / dt_s, m_g3.m_v);
    vec_axpy(xerr, ad_lu::scalar_value(m_coef.c64) / dt_s, m_g4.m_v);
    vec_axpy(xerr, ad_lu::scalar_value(m_coef.c65) / dt_s, m_g5.m_v);
    { auto _tp = m_prof.timer(prof_cat::lu_solve);
      m_lu.solve(xerr); }

    // --- Solution ---
    for (size_t i = 0; i < n; ++i)
      xout[i] = m_xtmp.m_v[i] + xerr[i];
  }

  // ====================================================================
  //  Convenience overloads
  // ====================================================================

  template<class Sys, class TimeArg>
  void do_step(Sys& system, state_type& x, TimeArg t,
               TimeArg dt, state_type& xerr,
               jacobian_hint hint = jacobian_hint::recompute_all)
  { do_step(system, x, t, x, dt, xerr, hint); }

  template<class Sys, class TimeArg>
  void do_step(Sys& system, const state_type& x, TimeArg t,
               state_type& xout, TimeArg dt,
               jacobian_hint hint = jacobian_hint::recompute_all)
  {
    m_x_err_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_x_err<state_type>(std::forward<decltype(arg)>(arg));
    });
    do_step(system, x, t, xout, dt, m_x_err.m_v, hint);
  }

  template<class Sys, class TimeArg>
  void do_step(Sys& system, state_type& x, TimeArg t, TimeArg dt,
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
    auto _tp = m_prof.timer(prof_cat::dense_snapshot);
    // cont3 = d21*g1 + d22*g2 + d23*g3 + d24*g4 + d25*g5
    vec_zero_with_slab(m_cont3.m_v, m_cont3_slab);
    vec_axpy_with_slab(m_cont3.m_v, m_cont3_slab,
                       ad_lu::scalar_value(m_coef.d21), m_g1.m_v, m_G.slab(0));
    vec_axpy_with_slab(m_cont3.m_v, m_cont3_slab,
                       ad_lu::scalar_value(m_coef.d22), m_g2.m_v, m_G.slab(1));
    vec_axpy_with_slab(m_cont3.m_v, m_cont3_slab,
                       ad_lu::scalar_value(m_coef.d23), m_g3.m_v, m_G.slab(2));
    vec_axpy_with_slab(m_cont3.m_v, m_cont3_slab,
                       ad_lu::scalar_value(m_coef.d24), m_g4.m_v, m_G.slab(3));
    vec_axpy_with_slab(m_cont3.m_v, m_cont3_slab,
                       ad_lu::scalar_value(m_coef.d25), m_g5.m_v, m_G.slab(4));
    // cont4 = d31*g1 + d32*g2 + d33*g3 + d34*g4 + d35*g5
    vec_zero_with_slab(m_cont4.m_v, m_cont4_slab);
    vec_axpy_with_slab(m_cont4.m_v, m_cont4_slab,
                       ad_lu::scalar_value(m_coef.d31), m_g1.m_v, m_G.slab(0));
    vec_axpy_with_slab(m_cont4.m_v, m_cont4_slab,
                       ad_lu::scalar_value(m_coef.d32), m_g2.m_v, m_G.slab(1));
    vec_axpy_with_slab(m_cont4.m_v, m_cont4_slab,
                       ad_lu::scalar_value(m_coef.d33), m_g3.m_v, m_G.slab(2));
    vec_axpy_with_slab(m_cont4.m_v, m_cont4_slab,
                       ad_lu::scalar_value(m_coef.d34), m_g4.m_v, m_G.slab(3));
    vec_axpy_with_slab(m_cont4.m_v, m_cont4_slab,
                       ad_lu::scalar_value(m_coef.d35), m_g5.m_v, m_G.slab(4));
  }

  template<class TimeArg>
  void calc_state(TimeArg t, state_type& x,
                  const state_type& x_old, TimeArg t_old,
                  const state_type& x_new, TimeArg t_new)
  {
    auto _tp = m_prof.timer(prof_cat::dense_interp);
    const size_t n = m_g1.m_v.size();
    TimeArg dt = t_new - t_old;
    TimeArg s  = (t - t_old) / dt;
    TimeArg s1 = 1.0 - s;
    for (size_t i = 0; i < n; ++i)
      x[i] = x_old[i] * s1 + s * (x_new[i] + s1 * (m_cont3.m_v[i] + s * m_cont4.m_v[i]));
  }

  template<class StateType>
  void adjust_size(const StateType& x)
  { resize_impl(x); resize_x_err(x); }

  // LU access (for controller)
  void invalidate_lu() { m_lu.invalidate(); }
  bool has_valid_jacobian() const { return m_lu.has_valid_jacobian(); }
  bool has_valid_lu() const { return m_lu.has_valid_lu(); }
  double last_factorized_dt() const {
    using ad_lu::scalar_value;
    return static_cast<double>(scalar_value(m_lu.last_factorized_dt()));
  }

  lu_type& lu() { return m_lu; }
  const lu_type& lu() const { return m_lu; }

protected:

  template<class StateIn>
  bool resize_impl(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(m_dxdt,    x);
    resized |= adjust_size_by_resizeability(m_xtmp,    x);
    resized |= adjust_size_by_resizeability(m_g1,      x);
    resized |= adjust_size_by_resizeability(m_g2,      x);
    resized |= adjust_size_by_resizeability(m_g3,      x);
    resized |= adjust_size_by_resizeability(m_g4,      x);
    resized |= adjust_size_by_resizeability(m_g5,      x);
    resized |= adjust_size_by_resizeability(m_cont3,   x);
    resized |= adjust_size_by_resizeability(m_cont4,   x);
    resized |= m_lu.resize(x);
    if (resized) m_lu.invalidate();
    if (resized && m_n_sens != 0) prepare_sensitivities(m_n_sens);
    return resized;
  }

  template<class StateIn>
  bool resize_x_err(const StateIn& x)
  {
    bool resized = adjust_size_by_resizeability(m_x_err, x);
    if (resized && m_n_sens != 0) prepare_sensitivities(m_n_sens);
    return resized;
  }

private:

  lu_type m_lu;

  resizer_type m_resizer;
  resizer_type m_x_err_resizer;

  wrapped_deriv_type   m_dxdt;
  wrapped_state_type   m_g1, m_g2, m_g3, m_g4, m_g5;
  wrapped_state_type   m_cont3, m_cont4;
  wrapped_state_type   m_xtmp;
  wrapped_state_type   m_x_err;

  // SoA tangent storage for the dynamic-dual heap path. Empty stubs
  // for non-dual value_type. The 5 Rosenbrock stages g1..g5 share one
  // contiguous tangent block via m_G; per-stage slab views are
  // m_G.slab(j). Other vectors keep individual slabs.
  detail::tangent_slab<value_type> m_dxdt_slab;
  detail::stage_matrix<value_type, 5> m_G;
  detail::tangent_slab<value_type> m_cont3_slab, m_cont4_slab;
  detail::tangent_slab<value_type> m_xtmp_slab;
  detail::tangent_slab<value_type> m_x_err_slab;
  // Placeholder slabs handed to vec_*_with_slab when the matching state
  // vector lives outside this stepper (controller's xerr, lu_W's m_dfdt,
  // the input x). Always unprimed → helpers fall through to their
  // per-element ET path. Mutable so we can hand out non-const refs to
  // satisfy the helper signature without lying about constness.
  // xerr is not slab-bound here (caller owns it) — we route the 5 xerr
  // axpys through plain vec_axpy below. m_dfdt and the input x are
  // similarly external; m_dfdt_unslabbed / m_x_in_unslabbed are
  // permanently-empty stubs that vec_*_with_slab sees as `primed=false`
  // so it falls through to its per-element loop.
  mutable detail::tangent_slab<value_type> m_dfdt_unslabbed;
  mutable detail::tangent_slab<value_type> m_x_in_unslabbed;
  unsigned m_n_sens = 0;

  const rosenbrock_coefficients m_coef;

  int m_n_fevals;
  int m_n_jevals;
  int m_n_lu_setups;

public:
  mutable cppode::profiler m_prof;
};

} // namespace cppode

#endif // CPPODE_ROSENBROCK4_HPP
