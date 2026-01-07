/*
 [begin_description]
 Controller for the Rosenbrock4 method with PI step-size control. Overload for AD types F<T>
 [end_description]

 Based on original work by:
 Copyright 2011-2012 Karsten Ahnert
 Copyright 2011-2012 Mario Mulansky
 Copyright 2012 Christoph Koke

 Extended with PI controller based on:
 - Gustafsson, K. (1991). "Control theoretic techniques for stepsize selection"
 - Söderlind, G. (2002). "Automatic control and adaptive time-stepping"
 */

#ifndef CPPODE_ROSENBROCK4_CONTROLLER_AD_PI_HPP_INCLUDED
#define CPPODE_ROSENBROCK4_CONTROLLER_AD_PI_HPP_INCLUDED

#include <cmath>
#include <algorithm>
#include <type_traits>
#include <boost/numeric/odeint/stepper/controlled_step_result.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <fadbad++/fadiff.h>

namespace boost {
namespace numeric {
namespace odeint {

/*==============================================================================
 Helper: Scalar extraction and derivative-aware error norms
 ==============================================================================*/

namespace controller_detail {

// Scalar extraction (recursive for nested AD types)
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v) { return static_cast<double>(v); }

template<class T>
inline double scalar_value(const fadbad::F<T>& v) {
  return scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

// Maximum absolute value including all derivative components
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
max_abs_with_derivs(const T& v) { return std::abs(v); }

template<class T>
inline double max_abs_with_derivs(const fadbad::F<T>& v) {
  auto& v_mut = const_cast<fadbad::F<T>&>(v);
  double maxv = std::abs(scalar_value(v_mut.x()));
  for (unsigned int i = 0; i < v_mut.size(); ++i) {
    maxv = std::max(maxv, max_abs_with_derivs(v_mut.d(i)));
  }
  return maxv;
}

} // namespace controller_detail


/**
 * @brief AD-aware Rosenbrock4 controller with PI step-size control.
 *
 * Implements the PI controller from Gustafsson (1991) and Söderlind (2002):
 *
 *   dt_new = dt * (err_old / err)^beta * (tol / err)^alpha
 *
 * Features:
 *   - Optional derivative-aware error control (use_derivs_in_error_control)
 *   - Smooth step-size evolution via PI control
 *   - Conservative behavior after step rejection
 *   - Full reset support for discontinuous events
 *
 * Template Parameters:
 *   - Stepper: underlying Rosenbrock4 stepper
 *   - use_derivs_in_error_control: if true, error includes AD derivatives
 */
template<class Stepper, bool use_derivs_in_error_control = false>
class rosenbrock4_controller_pi_ad
{
public:
  using stepper_type       = Stepper;
  using state_type         = typename stepper_type::state_type;
  using value_type         = typename stepper_type::value_type;
  using time_type          = typename stepper_type::time_type;
  using wrapped_state_type = typename stepper_type::wrapped_state_type;
  using resizer_type       = typename stepper_type::resizer_type;

  using stepper_category = controlled_stepper_tag;

  // PI controller parameters for order-4 method
  static constexpr double order = 4.0;
  static constexpr double default_alpha = 0.7 / (order + 1.0);  // ~0.14
  static constexpr double default_beta  = 0.4 / (order + 1.0);  // ~0.08

  // Safety and limiting factors
  static constexpr double default_safety     = 0.9;
  static constexpr double default_max_factor = 5.0;
  static constexpr double default_min_factor = 0.2;

public:

  /**
   * @brief Construct controller with tolerances and optional tuning parameters.
   *
   * @param atol  Absolute tolerance (default: 1e-6)
   * @param rtol  Relative tolerance (default: 1e-6)
   * @param alpha PI proportional gain (default: 0.14)
   * @param beta  PI integral gain (default: 0.08)
   * @param safety Safety factor (default: 0.9)
   * @param max_factor Maximum step increase factor (default: 5.0)
   * @param min_factor Minimum step decrease factor (default: 0.2)
   */
  rosenbrock4_controller_pi_ad(
    double atol = 1.0e-6,
    double rtol = 1.0e-6,
    double alpha = default_alpha,
    double beta = default_beta,
    double safety = default_safety,
    double max_factor = default_max_factor,
    double min_factor = default_min_factor)
    : m_atol(atol)
  , m_rtol(rtol)
  , m_alpha(alpha)
  , m_beta(beta)
  , m_safety(safety)
  , m_max_factor(max_factor)
  , m_min_factor(min_factor)
  , m_err_old(1.0)
  , m_first_step(true)
  , m_last_rejected(false)
  {}

  /*------------------------------------------------------------------------
   Standard try_step: x_in → x_out (separate arrays)
   ------------------------------------------------------------------------*/
  template<class System>
  controlled_step_result try_step(
      System system,
      const state_type& x_in,
      time_type& t,
      state_type& x_out,
      time_type& dt)
  {
    m_xerr_resizer.adjust_size(
      x_in, detail::bind(&rosenbrock4_controller_pi_ad::template resize_xerr<state_type>,
                         detail::ref(*this), detail::_1));

    m_stepper.do_step(system, x_in, t, x_out, dt, m_xerr.m_v);

    double err = compute_error(x_in, x_out);

    return update_stepsize(err, t, dt);
  }

  /*------------------------------------------------------------------------
   In-place try_step: x modified in place
   ------------------------------------------------------------------------*/
  template<class System>
  controlled_step_result try_step(
      System system,
      state_type& x,
      time_type& t,
      time_type& dt)
  {
    m_xtemp_resizer.adjust_size(
      x, detail::bind(&rosenbrock4_controller_pi_ad::template resize_xtemp<state_type>,
                      detail::ref(*this), detail::_1));

    controlled_step_result result = try_step(system, x, t, m_xtemp.m_v, dt);

    if (result == success) {
      x = m_xtemp.m_v;
    }
    return result;
  }

  /*------------------------------------------------------------------------
   Full try_step with x_in, x_out separate (for dense output)
   ------------------------------------------------------------------------*/
  template<class System>
  controlled_step_result try_step(
      System system,
      state_type& x_in,
      time_type& t,
      state_type& x_out,
      time_type& dt)
  {
    m_xerr_resizer.adjust_size(
      x_in, detail::bind(&rosenbrock4_controller_pi_ad::template resize_xerr<state_type>,
                         detail::ref(*this), detail::_1));

    m_stepper.do_step(system, x_in, t, x_out, dt, m_xerr.m_v);

    double err = compute_error(x_in, x_out);

    return update_stepsize(err, t, dt);
  }

  /*------------------------------------------------------------------------
   Event reset — clear controller history
   ------------------------------------------------------------------------*/
  void reset_after_event(time_type /*dt_before*/)
  {
    m_err_old = 1.0;
    m_first_step = true;
    m_last_rejected = false;
  }

  /*------------------------------------------------------------------------
   Accessors
   ------------------------------------------------------------------------*/
  stepper_type& stepper() { return m_stepper; }
  const stepper_type& stepper() const { return m_stepper; }

  template<class StateIn>
  void adjust_size(const StateIn& x)
  {
    resize_xerr(x);
    resize_xtemp(x);
    m_stepper.adjust_size(x);
  }

  // Diagnostics
  double error_old() const { return m_err_old; }
  bool first_step() const { return m_first_step; }
  bool last_rejected() const { return m_last_rejected; }

  // Tuning parameter access
  double atol() const { return m_atol; }
  double rtol() const { return m_rtol; }
  void set_tolerances(double atol, double rtol) { m_atol = atol; m_rtol = rtol; }

  double alpha() const { return m_alpha; }
  double beta() const { return m_beta; }
  void set_pi_gains(double alpha, double beta) { m_alpha = alpha; m_beta = beta; }

private:

  /*------------------------------------------------------------------------
   Compute normalized error (dispatch based on use_derivs_in_error_control)
   ------------------------------------------------------------------------*/
  double compute_error(const state_type& x_old, const state_type& x_new)
  {
    using controller_detail::scalar_value;
    using controller_detail::max_abs_with_derivs;

    double max_err = 0.0;

    for (std::size_t i = 0; i < x_old.size(); ++i) {
      double x_old_val = scalar_value(x_old[i]);
      double x_new_val = scalar_value(x_new[i]);

      double scale = m_atol + m_rtol * std::max(std::abs(x_old_val), std::abs(x_new_val));

      double err_component;
      if constexpr (use_derivs_in_error_control) {
        // Include derivatives in error estimate
        err_component = max_abs_with_derivs(m_xerr.m_v[i]);
      } else {
        // Only use scalar value
        err_component = std::abs(scalar_value(m_xerr.m_v[i]));
      }

      double rel_err = err_component / scale;
      max_err = std::max(max_err, rel_err);
    }

    return max_err;
  }

  /*------------------------------------------------------------------------
   PI Controller step-size update
   ------------------------------------------------------------------------*/
  controlled_step_result update_stepsize(double err, time_type& t, time_type& dt)
  {
    using controller_detail::scalar_value;

    // Prevent division by zero and log of zero
    err = std::max(err, 1e-15);

    if (err <= 1.0) {
      // === Step accepted ===
      double factor;

      if (m_first_step || m_last_rejected) {
        // Pure P-control for first step or after rejection (no history)
        factor = m_safety * std::pow(1.0 / err, 1.0 / (order + 1.0));
      } else {
        // Full PI control
        factor = m_safety
        * std::pow(m_err_old / err, m_beta)
        * std::pow(1.0 / err, m_alpha);
      }

      // Limit step increase
      if (m_last_rejected) {
        // Don't increase step size immediately after rejection
        factor = std::min(factor, 1.0);
      }
      factor = std::clamp(factor, m_min_factor, m_max_factor);

      // Update controller state
      m_err_old = err;
      m_first_step = false;
      m_last_rejected = false;

      // Advance time and update step size
      t += dt;
      dt *= factor;

      return success;
    }
    else {
      // === Step rejected ===
      double factor = m_safety * std::pow(1.0 / err, 1.0 / (order + 1.0));

      // Clamp decrease (not too aggressive, but ensure decrease)
      factor = std::clamp(factor, m_min_factor, 0.9);

      m_last_rejected = true;
      dt *= factor;

      return fail;
    }
  }

  /*------------------------------------------------------------------------
   Resizing
   ------------------------------------------------------------------------*/
  template<class StateIn>
  bool resize_xerr(const StateIn& x)
  {
    return adjust_size_by_resizeability(
      m_xerr, x, typename is_resizeable<state_type>::type());
  }

  template<class StateIn>
  bool resize_xtemp(const StateIn& x)
  {
    return adjust_size_by_resizeability(
      m_xtemp, x, typename is_resizeable<state_type>::type());
  }

private:
  stepper_type       m_stepper;
  resizer_type       m_xerr_resizer;
  resizer_type       m_xtemp_resizer;
  wrapped_state_type m_xerr;
  wrapped_state_type m_xtemp;

  // Tolerances
  double m_atol;
  double m_rtol;

  // PI controller tuning
  double m_alpha;
  double m_beta;
  double m_safety;
  double m_max_factor;
  double m_min_factor;

  // Controller state
  double m_err_old;
  bool   m_first_step;
  bool   m_last_rejected;
};

/*==============================================================================
 Convenience type aliases
 ==============================================================================*/

// Standard: only values in error control
template<class Stepper>
using rosenbrock4_controller_pi_ad_val = rosenbrock4_controller_pi_ad<Stepper, false>;

// With derivatives in error control (more conservative, better sensitivity accuracy)
template<class Stepper>
using rosenbrock4_controller_pi_ad_deriv = rosenbrock4_controller_pi_ad<Stepper, true>;

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_ROSENBROCK4_CONTROLLER_AD_PI_HPP_INCLUDED
