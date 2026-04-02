/*
 Rosenbrock4 controller with PI step-size control.

 Unified version: handles both double and AD types.
 Error norm optionally includes derivative components for AD types,
 equivalent to solving the sensitivity-augmented system.

 The controller architecture is derived from Boost.Odeint's rosenbrock4_controller
 by Karsten Ahnert, Mario Mulansky, and Christoph Koke (2011-2012),
 distributed under the Boost Software License, Version 1.0.

 Modified work:
 Copyright (C) 2026 Simon Beyer

 Modifications:
 - Added reset_after_event() for event-driven integration
 - Implemented Gustafsson–Söderlind PI control algorithm
 - Unified double and AD handling in a single class

 PI controller based on:
 Gustafsson, K., Lundh, M. & Söderlind, G. (1988).
 "A PI stepsize control for the numerical solution of ordinary differential equations".
 BIT 28, 270–287. https://doi.org/10.1007/BF01934091
 */

#ifndef CPPODE_ROSENBROCK4_CONTROLLER_HPP
#define CPPODE_ROSENBROCK4_CONTROLLER_HPP

#include <cmath>
#include <algorithm>
#include <type_traits>
#include <cppode/cppode_rosenbrock4.hpp>
#include <cppode/cppode_ad_lu.hpp>   // for ad_lu::is_ad
#include <cppode/cppode_profiler.hpp>

namespace cppode {

// ============================================================================
//  Scalar value extraction (AD-compatible)
// ============================================================================

namespace controller_detail {

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v) { return static_cast<double>(v); }

template<class T, unsigned int N>
inline double scalar_value(const fadbad::F<T,N>& v) {
  return scalar_value(const_cast<fadbad::F<T,N>&>(v).x());
}

} // namespace controller_detail

// ============================================================================
//  rosenbrock4_controller<Stepper>
//
//  Unified PI controller for both double and AD types.
//  Error norm includes derivative components for AD types.
// ============================================================================

template<class Stepper>
class rosenbrock4_controller
{
public:

  using stepper_type       = Stepper;
  using state_type         = typename stepper_type::state_type;
  using value_type         = typename stepper_type::value_type;
  using time_type          = typename stepper_type::value_type;   // external: value_type for API compat
  using scalar_time_type   = typename stepper_type::time_type;     // internal: scalar for stepper
  using wrapped_state_type = typename stepper_type::wrapped_state_type;
  using resizer_type       = typename stepper_type::resizer_type;

  using stepper_category = controlled_stepper_tag;
  using controller_type  = rosenbrock4_controller<Stepper>;

  /// Method order (used in step-size formula)
  static constexpr double order = 3.0;

  /// Default proportional gain: alpha = 0.7 / (order + 1)
  static constexpr double default_alpha = 0.7 / (order + 1.0);  // ~0.175

  /// Default integral gain: beta = 0.4 / (order + 1)
  static constexpr double default_beta  = 0.4 / (order + 1.0);  // ~0.1

  /// Default safety factor for step-size selection
  static constexpr double default_safety = 0.9;

  /// Default maximum step increase factor
  static constexpr double default_max_factor = 5.0;

  /// Default minimum step decrease factor
  static constexpr double default_min_factor = 0.2;

  // ====================================================================
  //  Constructors
  // ====================================================================

  rosenbrock4_controller(
    double atol = 1.0e-6,
    double rtol = 1.0e-6,
    double alpha = default_alpha,
    double beta = default_beta,
    double safety = default_safety,
    double max_factor = default_max_factor,
    double min_factor = default_min_factor)
    : m_stepper()
    , m_atol(atol), m_rtol(rtol)
    , m_max_dt(0)
    , m_alpha(alpha), m_beta(beta)
    , m_safety(safety)
    , m_max_factor(max_factor), m_min_factor(min_factor)
    , m_first_step(true), m_err_old(1.0), m_dt_old(1.0)
    , m_last_rejected(false)
    , m_n_accepted(0), m_n_rejected(0)
  {}

  rosenbrock4_controller(
    double atol, double rtol, double max_dt,
    double alpha, double beta,
    double safety = default_safety,
    double max_factor = default_max_factor,
    double min_factor = default_min_factor)
    : m_stepper()
    , m_atol(atol), m_rtol(rtol)
    , m_max_dt(max_dt)
    , m_alpha(alpha), m_beta(beta)
    , m_safety(safety)
    , m_max_factor(max_factor), m_min_factor(min_factor)
    , m_first_step(true), m_err_old(1.0), m_dt_old(1.0)
    , m_last_rejected(false)
    , m_n_accepted(0), m_n_rejected(0)
  {}

  // ====================================================================
  //  Jacobian hint (rejection-only reuse)
  // ====================================================================

  jacobian_hint compute_hint() const
  {
    if (m_first_step || !m_stepper.has_valid_jacobian())
      return jacobian_hint::recompute_all;

    // After a rejected step: same (x, t), only dt changed.
    if (m_last_rejected)
      return jacobian_hint::reuse_jacobian;

    return jacobian_hint::recompute_all;
  }

  // ====================================================================
  //  Error norm — AD-aware WRMS norm
  //
  //  Always includes derivative components for AD types.
  //  For double, the derivative loop is empty → identical to non-AD.
  // ====================================================================

  double error(const state_type& x, const state_type& xold, const state_type& xerr)
  {
    auto _tp = m_prof.timer(cppode::prof_cat::error_norm);
    using controller_detail::scalar_value;

    const size_t n = x.size();
    double sumsq = 0.0;
    size_t N_eff = 0;

    for (size_t i = 0; i < n; ++i) {
      double x_old_val = scalar_value(xold[i]);
      double x_new_val = scalar_value(x[i]);
      double scale = m_atol + m_rtol * std::max(std::abs(x_old_val), std::abs(x_new_val));

      // Value component
      double err_val = std::abs(scalar_value(xerr[i]));
      double r = err_val / scale;
      sumsq += r * r;
      ++N_eff;

      // Derivative components (only active for AD types)
      if constexpr (ad_lu::is_ad<value_type>::value) {
        auto& xerr_ad = const_cast<value_type&>(xerr[i]);
        auto& xold_ad = const_cast<value_type&>(xold[i]);
        auto& xnew_ad = const_cast<value_type&>(x[i]);
        unsigned nd = xerr_ad.size();
        for (unsigned j = 0; j < nd; ++j) {
          double err_d = std::abs(scalar_value(xerr_ad.d(j)));
          double xold_d = std::abs(scalar_value(xold_ad.d(j)));
          double xnew_d = std::abs(scalar_value(xnew_ad.d(j)));
          double scale_d = m_atol + m_rtol * std::max(xold_d, xnew_d);
          double rd = err_d / scale_d;
          sumsq += rd * rd;
          ++N_eff;
        }
      }
    }

    return (N_eff > 0) ? std::sqrt(sumsq / N_eff) : 0.0;
  }

  // ====================================================================
  //  try_step: in-place
  // ====================================================================

  template<class System>
  controlled_step_result
  try_step(System sys, state_type& x, time_type& t, time_type& dt)
  {
    m_xnew_resizer.adjust_size(x, [this](const state_type& s){ return this->resize_m_xnew(s); });
    controlled_step_result res = try_step(sys, x, t, m_xnew.m_v, dt);
    if (res == success) {
      x = m_xnew.m_v;
    }
    return res;
  }

  // ====================================================================
  //  try_step: separate input/output — main entry point
  // ====================================================================

  template<class System>
  controlled_step_result
  try_step(System sys, const state_type& x, time_type& t, state_type& xout, time_type& dt)
  {
    if (m_max_dt != 0.0) {
      double dt_val = controller_detail::scalar_value(dt);
      if (std::abs(dt_val) > m_max_dt) {
        dt = time_type(std::copysign(m_max_dt, dt_val));
        return fail;
      }
    }

    m_xerr_resizer.adjust_size(x, [this](const state_type& s){ return this->resize_m_xerr(s); });

    jacobian_hint hint = compute_hint();

    m_stepper.do_step(sys, x, t, xout, dt, m_xerr.m_v, hint);
    double err = error(xout, x, m_xerr.m_v);

    // Prevent division by zero
    err = std::max(err, 1e-15);

    return update_stepsize(err, t, dt);
  }

  // ====================================================================
  //  try_step: mutable input (for dense output compatibility)
  // ====================================================================

  template<class System>
  controlled_step_result
  try_step(System sys, state_type& x_in, time_type& t, state_type& x_out, time_type& dt)
  {
    return try_step(sys, static_cast<const state_type&>(x_in), t, x_out, dt);
  }

  // ====================================================================
  //  Event reset
  // ====================================================================

  void reset_after_event(time_type /*dt_before*/)
  {
    m_first_step = true;
    m_err_old = 1.0;
    m_dt_old = 1.0;
    m_last_rejected = false;
    m_stepper.invalidate_lu();
  }

  // ====================================================================
  //  Accessors
  // ====================================================================

  stepper_type&       stepper()       { return m_stepper; }
  const stepper_type& stepper() const { return m_stepper; }

  template<class StateType>
  void adjust_size(const StateType& x)
  {
    resize_m_xerr(x);
    resize_m_xnew(x);
  }

  double last_error() const { return m_err_old; }
  bool first_step() const { return m_first_step; }
  bool last_rejected() const { return m_last_rejected; }
  double dt_old() const { return m_dt_old; }

  // --- Diagnostics ---
  int n_accepted() const { return m_n_accepted; }
  int n_rejected() const { return m_n_rejected; }
  int n_fevals()   const { return m_stepper.n_fevals(); }
  int n_jevals()   const { return m_stepper.n_jevals(); }
  static constexpr int current_method_order() { return static_cast<int>(stepper_type::stepper_order); }
  void reset_counters() { m_n_accepted = 0; m_n_rejected = 0; m_stepper.reset_counters(); }

  // Tolerance accessors
  double atol() const { return m_atol; }
  double rtol() const { return m_rtol; }
  void set_tolerances(double atol, double rtol) { m_atol = atol; m_rtol = rtol; }

  // PI controller parameter accessors
  double alpha() const { return m_alpha; }
  double beta() const { return m_beta; }
  double safety() const { return m_safety; }
  double max_factor() const { return m_max_factor; }
  double min_factor() const { return m_min_factor; }
  void set_pi_gains(double alpha, double beta) { m_alpha = alpha; m_beta = beta; }
  void set_pi_parameters(double alpha, double beta, double safety,
                         double max_factor, double min_factor)
  {
    m_alpha = alpha; m_beta = beta; m_safety = safety;
    m_max_factor = max_factor; m_min_factor = min_factor;
  }

private:

  // ====================================================================
  //  PI step-size control
  // ====================================================================

  controlled_step_result update_stepsize(double err, time_type& t, time_type& dt)
  {
    if (err <= 1.0) {
      // === Step accepted ===
      double factor;

      if (m_first_step || m_last_rejected) {
        factor = m_safety * std::pow(1.0 / err, 1.0 / (order + 1.0));
      } else {
        factor = m_safety
        * std::pow(m_err_old / err, m_beta)
        * std::pow(1.0 / err, m_alpha);
      }

      if (m_last_rejected) {
        factor = std::min(factor, 1.0);
      }
      factor = std::clamp(factor, m_min_factor, m_max_factor);

      // Update controller state
      m_dt_old = controller_detail::scalar_value(dt);
      m_err_old = std::max(0.01, err);
      m_first_step = false;
      m_last_rejected = false;

      // Advance time and update step size
      t += dt;
      dt *= factor;

      // Apply max_dt limit
      if (m_max_dt != 0.0) {
        double dt_val = controller_detail::scalar_value(dt);
        if (std::abs(dt_val) > m_max_dt) {
          dt = time_type(std::copysign(m_max_dt, dt_val));
        }
      }

      ++m_n_accepted;
      return success;
    }
    else {
      // === Step rejected ===
      ++m_n_rejected;
      double factor = m_safety * std::pow(1.0 / err, 1.0 / (order + 1.0));
      factor = std::clamp(factor, m_min_factor, 0.9);

      m_last_rejected = true;
      dt *= factor;

      return fail;
    }
  }

  // ====================================================================
  //  Resize helpers
  // ====================================================================

  template<class StateIn>
  bool resize_m_xerr(const StateIn& x)
  { return adjust_size_by_resizeability(m_xerr, x); }

  template<class StateIn>
  bool resize_m_xnew(const StateIn& x)
  { return adjust_size_by_resizeability(m_xnew, x); }

  // ====================================================================
  //  Members
  // ====================================================================

  stepper_type       m_stepper;
  resizer_type       m_xerr_resizer;
  resizer_type       m_xnew_resizer;
  wrapped_state_type m_xerr;
  wrapped_state_type m_xnew;

  double m_atol, m_rtol;
  double m_max_dt;

  // PI controller parameters
  double m_alpha, m_beta;
  double m_safety;
  double m_max_factor, m_min_factor;

  // Controller state
  bool   m_first_step;
  double m_err_old, m_dt_old;
  bool   m_last_rejected;

  // Diagnostics
  int m_n_accepted;
  int m_n_rejected;

public:
  mutable cppode::profiler m_prof;
  void finalize_profiler() const { m_prof.merge(m_stepper.m_prof); }
  void report_profiler(const char* label = "CppODE RB4") const {
    finalize_profiler();
    m_prof.report(label);
  }
};

// ============================================================================
//  Backward-compatible type aliases
// ============================================================================

template<class Stepper>
using rosenbrock4_controller_pi = rosenbrock4_controller<Stepper>;

template<class Stepper, bool = false>
using rosenbrock4_controller_pi_ad = rosenbrock4_controller<Stepper>;

template<class Stepper>
using rosenbrock4_controller_pi_ad_val = rosenbrock4_controller<Stepper>;

template<class Stepper>
using rosenbrock4_controller_pi_ad_deriv = rosenbrock4_controller<Stepper>;

} // namespace cppode

#endif // CPPODE_ROSENBROCK4_CONTROLLER_HPP
