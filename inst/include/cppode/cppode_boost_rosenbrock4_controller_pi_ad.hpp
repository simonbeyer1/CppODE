/*
 Controller for the Rosenbrock4 method with PI step-size control.
 AD-aware version for automatic differentiation types.

 Original work:
 Copyright 2011-2012 Karsten Ahnert
 Copyright 2011-2012 Mario Mulansky
 Copyright 2012 Christoph Koke
 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)

 Modified work:
 Copyright 2026 Simon Beyer

 Modifications:
 - Extended for FADBAD++ automatic differentiation types (fadbad::F<T>)
 - Implemented derivative-aware error control (optional)
 - Added scalar extraction for nested AD types (F<F<...>>)
 - Replaced classical step-size control with PI controller
 - Added reset_after_event() for event-driven integration
 - Improved numerical stability for AD computations

 PI controller based on:
 - Gustafsson, K. (1991). "Control theoretic techniques for stepsize selection
 in explicit Runge-Kutta methods". ACM Trans. Math. Softw. 17(4), 533-554.
 - Söderlind, G. (2002). "Automatic control and adaptive time-stepping".
 Numer. Algorithms 31(1-4), 281-310.
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

//==============================================================================
// Helper Utilities
//==============================================================================

namespace controller_detail {

/**
 * @brief Extract scalar value from arithmetic types
 * @tparam T Arithmetic type (double, float, etc.)
 * @param v Value to extract
 * @return Scalar value as double
 */
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v) { return static_cast<double>(v); }

/**
 * @brief Extract scalar value from FADBAD++ AD types (recursive)
 *
 * Recursively unwraps nested fadbad::F<T> types to extract the
 * underlying scalar value. Supports arbitrary nesting depth
 * (e.g., F<F<F<double>>>).
 *
 * @tparam T Inner type (may be another fadbad::F or a scalar)
 * @param v AD value to unwrap
 * @return Underlying scalar value as double
 */
template<class T>
inline double scalar_value(const fadbad::F<T>& v) {
  return scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

/**
 * @brief Compute maximum absolute value for arithmetic types
 * @tparam T Arithmetic type
 * @param v Value to process
 * @return Absolute value
 */
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
max_abs_with_derivs(const T& v) { return std::abs(v); }

/**
 * @brief Compute maximum absolute value including all derivatives
 *
 * For FADBAD++ types, computes the maximum of:
 * - |value|
 * - |derivative_1|, |derivative_2|, ..., |derivative_n|
 *
 * This ensures that both the function value and all sensitivity
 * information are considered in error control.
 *
 * @tparam T Inner fadbad type
 * @param v AD value
 * @return Maximum absolute value across value and all derivatives
 */
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
 * @class rosenbrock4_controller_pi_ad
 * @brief AD-aware Rosenbrock4 controller with PI step-size control
 *
 * This class extends rosenbrock4_controller_pi to support automatic
 * differentiation types from FADBAD++. It provides two error control modes:
 *
 * **Value-only mode** (use_derivs_in_error_control = false):
 * - Error computed from function values only
 * - Derivatives propagated but not used in step-size control
 * - More aggressive step-size selection
 *
 * **Derivative-aware mode** (use_derivs_in_error_control = true):
 * - Error includes maximum over all derivative components
 * - More conservative (ensures derivative accuracy)
 * - Recommended for sensitivity analysis and optimization
 *
 * @tparam Stepper The underlying Rosenbrock4 stepper (AD-compatible)
 * @tparam use_derivs_in_error_control Include derivatives in error (default: false)
 *
 * @par PI Control Formula
 * @code
 * dt_new = dt * safety * (err_old / err)^beta * (1 / err)^alpha
 * @endcode
 *
 * where alpha ≈ 0.14, beta ≈ 0.08 (tuned for order 4)
 *
 * @par Usage Example
 * @code
 * using ad_type = fadbad::F<double>;
 * using state_t = vector<ad_type>;
 * using stepper_t = rosenbrock4<state_t>;
 *
 * // Value-only error control
 * rosenbrock4_controller_pi_ad<stepper_t, false> ctrl_val(1e-6, 1e-6);
 *
 * // Derivative-aware error control
 * rosenbrock4_controller_pi_ad<stepper_t, true> ctrl_deriv(1e-6, 1e-6);
 * @endcode
 *
 * @see rosenbrock4_controller_pi (non-AD version)
 * @see Gustafsson (1991), Söderlind (2002)
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

  /// Method order (used in step-size formula)
  static constexpr double order = 4.0;

  /// Proportional gain: alpha = 0.7 / (order + 1)
  static constexpr double default_alpha = 0.7 / (order + 1.0);  // ~0.14

  /// Integral gain: beta = 0.4 / (order + 1)
  static constexpr double default_beta  = 0.4 / (order + 1.0);  // ~0.08

  /// Safety factor for step-size selection
  static constexpr double default_safety     = 0.9;

  /// Maximum step increase factor
  static constexpr double default_max_factor = 5.0;

  /// Minimum step decrease factor
  static constexpr double default_min_factor = 0.2;

public:

  /**
   * @brief Construct AD-aware controller with full tuning control
   *
   * @param atol  Absolute tolerance (default: 1e-6)
   * @param rtol  Relative tolerance (default: 1e-6)
   * @param alpha PI proportional gain (default: 0.14)
   * @param beta  PI integral gain (default: 0.08)
   * @param safety Safety factor (default: 0.9)
   * @param max_factor Maximum step increase (default: 5.0)
   * @param min_factor Minimum step decrease (default: 0.2)
   *
   * @par Tuning Guidelines
   * - **Increase alpha/beta**: Smoother step-size evolution, slower adaptation
   * - **Decrease alpha/beta**: Faster adaptation, more oscillations
   * - **Increase safety**: More conservative, smaller steps
   * - **Narrow min/max_factor**: Limits step-size variation
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

  /**
   * @brief Try one integration step (separate input/output)
   *
   * @tparam System ODE system type (callable: void(x, dxdt, t))
   * @param system The ODE system
   * @param x_in Input state (unchanged)
   * @param t Current time (advanced on success)
   * @param x_out Output state (new state on success)
   * @param dt Step size (updated based on error)
   * @return success if step accepted, fail if rejected
   */
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

  /**
   * @brief Try one integration step (in-place modification)
   *
   * @tparam System ODE system type
   * @param system The ODE system
   * @param x State vector (modified on success)
   * @param t Current time (advanced on success)
   * @param dt Step size (updated based on error)
   * @return success if step accepted, fail if rejected
   */
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

  /**
   * @brief Try step with mutable input state (for dense output compatibility)
   *
   * @tparam System ODE system type
   * @param system The ODE system
   * @param x_in Input state (may be modified by stepper internals)
   * @param t Current time (advanced on success)
   * @param x_out Output state
   * @param dt Step size (updated based on error)
   * @return success if step accepted, fail if rejected
   */
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

  /**
   * @brief Reset controller state after discontinuous event
   *
   * Clears error and step-size history to prevent using pre-event
   * information for post-event step-size selection. Critical for
   * event-driven integration with state discontinuities.
   *
   * @param dt_before Step size before event (currently unused)
   *
   * @note Resets to first-step mode (pure P-control)
   */
  void reset_after_event(time_type /*dt_before*/)
  {
    m_err_old = 1.0;
    m_first_step = true;
    m_last_rejected = false;
  }

  /**
   * @brief Access the underlying Rosenbrock4 stepper
   * @return Reference to stepper
   */
  stepper_type& stepper() { return m_stepper; }

  /**
   * @brief Access the underlying Rosenbrock4 stepper (const)
   * @return Const reference to stepper
   */
  const stepper_type& stepper() const { return m_stepper; }

  /**
   * @brief Adjust internal buffer sizes
   * @tparam StateIn State type for sizing
   * @param x Reference state
   */
  template<class StateIn>
  void adjust_size(const StateIn& x)
  {
    resize_xerr(x);
    resize_xtemp(x);
    m_stepper.adjust_size(x);
  }

  // Diagnostic accessors

  /**
   * @brief Get previous step's error
   * @return Normalized error from last successful step
   */
  double error_old() const { return m_err_old; }

  /**
   * @brief Check if this is the first step
   * @return true if no successful steps taken yet
   */
  bool first_step() const { return m_first_step; }

  /**
   * @brief Check if last step was rejected
   * @return true if previous try_step() failed
   */
  bool last_rejected() const { return m_last_rejected; }

  // Tolerance accessors

  /**
   * @brief Get absolute tolerance
   * @return Current absolute tolerance
   */
  double atol() const { return m_atol; }

  /**
   * @brief Get relative tolerance
   * @return Current relative tolerance
   */
  double rtol() const { return m_rtol; }

  /**
   * @brief Update error tolerances
   * @param atol New absolute tolerance
   * @param rtol New relative tolerance
   */
  void set_tolerances(double atol, double rtol) { m_atol = atol; m_rtol = rtol; }

  // PI controller parameter accessors

  /**
   * @brief Get proportional gain
   * @return Current alpha value
   */
  double alpha() const { return m_alpha; }

  /**
   * @brief Get integral gain
   * @return Current beta value
   */
  double beta() const { return m_beta; }

  /**
   * @brief Update PI controller gains
   * @param alpha New proportional gain
   * @param beta New integral gain
   */
  void set_pi_gains(double alpha, double beta) { m_alpha = alpha; m_beta = beta; }

private:

  /**
   * @brief Compute normalized error (mode-dependent)
   *
   * Computes maximum relative error over all state components:
   * @code
   * err = max_i( |xerr[i]| / scale[i] )
   * where scale[i] = atol + rtol * max(|x_old[i]|, |x_new[i]|)
   * @endcode
   *
   * **If use_derivs_in_error_control = true:**
   * - |xerr[i]| = max(|value|, |deriv_1|, ..., |deriv_n|)
   *
   * **If use_derivs_in_error_control = false:**
   * - |xerr[i]| = |value| only
   *
   * @param x_old State before step
   * @param x_new State after step
   * @return Normalized error (accept if <= 1.0)
   */
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

  /**
   * @brief Update step size using PI control
   *
   * Implements the PI controller step-size selection algorithm.
   *
   * @param err Normalized error from current step
   * @param t Time (advanced on success)
   * @param dt Step size (updated)
   * @return success if err <= 1.0, fail otherwise
   *
   * @par Algorithm
   * **On acceptance (err <= 1.0):**
   * - First step or after rejection: factor = safety * (1/err)^(1/5)
   * - Normal: factor = safety * (err_old/err)^beta * (1/err)^alpha
   * - After rejection: limit factor <= 1.0
   * - Clamp: min_factor <= factor <= max_factor
   * - Update: dt *= factor, t += dt_old
   *
   * **On rejection (err > 1.0):**
   * - factor = safety * (1/err)^(1/5)
   * - Clamp: min_factor <= factor <= 0.9
   * - Update: dt *= factor
   */
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

      // Limit step increase after rejection
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

  /**
   * @brief Resize error buffer
   * @tparam StateIn State type
   * @param x Reference state for sizing
   * @return true if resized
   */
  template<class StateIn>
  bool resize_xerr(const StateIn& x)
  {
    return adjust_size_by_resizeability(
      m_xerr, x, typename is_resizeable<state_type>::type());
  }

  /**
   * @brief Resize temporary state buffer
   * @tparam StateIn State type
   * @param x Reference state for sizing
   * @return true if resized
   */
  template<class StateIn>
  bool resize_xtemp(const StateIn& x)
  {
    return adjust_size_by_resizeability(
      m_xtemp, x, typename is_resizeable<state_type>::type());
  }

private:
  stepper_type       m_stepper;        ///< Underlying Rosenbrock4 stepper
  resizer_type       m_xerr_resizer;   ///< Error buffer resizer
  resizer_type       m_xtemp_resizer;  ///< Temp state resizer
  wrapped_state_type m_xerr;           ///< Error estimate buffer
  wrapped_state_type m_xtemp;          ///< Temporary state buffer

  // Tolerances
  double m_atol;  ///< Absolute tolerance
  double m_rtol;  ///< Relative tolerance

  // PI controller tuning
  double m_alpha;       ///< Proportional gain
  double m_beta;        ///< Integral gain
  double m_safety;      ///< Safety factor
  double m_max_factor;  ///< Maximum step increase
  double m_min_factor;  ///< Minimum step decrease

  // Controller state
  double m_err_old;       ///< Previous error (for PI control)
  bool   m_first_step;    ///< First step flag
  bool   m_last_rejected; ///< Rejection flag
};

//==============================================================================
// Type Aliases
//==============================================================================

/**
 * @brief Value-only error control (default)
 * @tparam Stepper Rosenbrock4 stepper type
 */
template<class Stepper>
using rosenbrock4_controller_pi_ad_val = rosenbrock4_controller_pi_ad<Stepper, false>;

/**
 * @brief Derivative-aware error control (conservative, better for sensitivities)
 * @tparam Stepper Rosenbrock4 stepper type
 */
template<class Stepper>
using rosenbrock4_controller_pi_ad_deriv = rosenbrock4_controller_pi_ad<Stepper, true>;

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_ROSENBROCK4_CONTROLLER_AD_PI_HPP_INCLUDED
