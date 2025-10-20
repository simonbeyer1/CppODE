#ifndef ROSENBROCK4_CONTROLLER_AD_HPP
#define ROSENBROCK4_CONTROLLER_AD_HPP

#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/controlled_step_result.hpp>
#include <boost/numeric/odeint/util/copy.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/detail/less_with_sign.hpp>

#include <fadbad++/fadiff.h>

#include <algorithm>
#include <cmath>
#include <functional>

/**
 * ============================================================================
 * @file rosenbrock4_controller_ad.hpp
 * @brief Adaptive controller for the Boost.Odeint Rosenbrock4 stepper
 *        supporting FADBAD++ automatic differentiation types.
 *
 * This controller replaces the default error estimation with one that
 * accounts for derivative components stored in fadbad::F<T> objects.
 * It is compatible with automatic differentiation pipelines where
 * both the function values and their partial derivatives are tracked.
 *
 * @details
 * Key features:
 *  - Handles mixed types: `double` and `fadbad::F<T>`
 *  - Recursively computes error norms across derivative levels
 *  - Fully compatible with Boost.Odeint adaptive stepping framework
 *  - Uses safe const_cast for non-const `.x()` access in fadbad::F<T>
 *
 * @note
 * Designed for use with controlled Rosenbrock4 steppers in Boost.Odeint.
 * Tested on GCC 13.3 with Boost 1.83 and FADBAD++.
 * ============================================================================
 */

namespace boost {
namespace numeric {
namespace odeint {

// ============================================================================
//  detail namespace — helper functions for scalar extraction and norms
// ============================================================================
namespace detail {

/**
 * @brief Extracts a plain scalar double value.
 * This overload is used for built-in numeric types.
 */
inline double get_scalar_value(double v) { return v; }

/**
 * @brief Extracts the scalar value from a fadbad::F<T> object.
 *
 * FADBAD++ stores the "value" and derivatives internally.
 * `.x()` returns a reference to the base value, but it is not const-qualified,
 * so a `const_cast` is required to safely access it from const objects.
 *
 * @tparam T  Base numeric type (typically double)
 * @param v   Input FADBAD++ variable
 * @return Underlying scalar value as double
 */
template<typename T>
inline double get_scalar_value(const fadbad::F<T>& v) {
  return get_scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

/**
 * @brief Computes the maximum absolute value for plain doubles.
 *
 * @note This overload must NOT be templated. It serves as the base case
 *       for recursive overload resolution.
 */
inline double get_max_abs(double v) { return std::abs(v); }

/**
 * @brief Recursively computes the maximum absolute value across
 *        all derivative components of a fadbad::F<T> variable.
 *
 * @tparam T  Base numeric type (e.g. double)
 * @param v   FADBAD++ variable
 * @return Maximum absolute value among base and derivative components
 */
template<typename T>
inline double get_max_abs(const fadbad::F<T>& v) {
  auto& v_mut = const_cast<fadbad::F<T>&>(v);
  double maxv = std::abs(get_scalar_value(v_mut.x()));
  for (int i = 0; i < v_mut.size(); ++i)
    maxv = std::max(maxv, get_max_abs(v_mut.d(i)));
  return maxv;
}

} // namespace detail

// ============================================================================
//  rosenbrock4_controller_ad class definition
// ============================================================================

/**
 * @class rosenbrock4_controller_ad
 * @brief Adaptive error controller for Rosenbrock4 steppers using FADBAD++.
 *
 * This class wraps a Rosenbrock4 stepper and redefines its error estimation
 * to include contributions from derivative information when `value_type`
 * is a fadbad::F<T> type.
 *
 * @tparam Stepper  The underlying Rosenbrock4 stepper type
 */
template<class Stepper>
class rosenbrock4_controller_ad {
public:
  using stepper_type       = Stepper;
  using value_type         = typename stepper_type::value_type;
  using state_type         = typename stepper_type::state_type;
  using wrapped_state_type = typename stepper_type::wrapped_state_type;
  using time_type          = typename stepper_type::time_type;
  using deriv_type         = typename stepper_type::deriv_type;
  using wrapped_deriv_type = typename stepper_type::wrapped_deriv_type;
  using resizer_type       = typename stepper_type::resizer_type;
  using stepper_category   = controlled_stepper_tag;

  using controller_type = rosenbrock4_controller_ad<Stepper>;

  // ------------------------------------------------------------------------
  // Constructors
  // ------------------------------------------------------------------------
  explicit rosenbrock4_controller_ad(
      double atol = 1.0e-6,
      double rtol = 1.0e-6,
      const stepper_type& stepper = stepper_type())
    : m_stepper(stepper),
      m_atol(atol),
      m_rtol(rtol),
      m_max_dt(static_cast<time_type>(0)),
      m_first_step(true),
      m_err_old(0.0),
      m_dt_old(0.0),
      m_last_rejected(false)
  {}

  rosenbrock4_controller_ad(
    double atol,
    double rtol,
    time_type max_dt,
    const stepper_type& stepper = stepper_type())
    : m_stepper(stepper),
      m_atol(atol),
      m_rtol(rtol),
      m_max_dt(max_dt),
      m_first_step(true),
      m_err_old(0.0),
      m_dt_old(0.0),
      m_last_rejected(false)
  {}

  // ------------------------------------------------------------------------
  // Error norm computation
  // ------------------------------------------------------------------------
  /**
   * @brief Computes the weighted RMS error across all state variables.
   *
   * @param x     Current state
   * @param xold  Previous state
   * @param xerr  Estimated local truncation error
   * @return Normalized root-mean-square error
   */
  double error(const state_type& x,
               const state_type& xold,
               const state_type& xerr) {
    const std::size_t n = x.size();
    double err = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
      auto& xi    = const_cast<value_type&>(x[i]);
      auto& xiold = const_cast<value_type&>(xold[i]);
      auto& xerri = const_cast<value_type&>(xerr[i]);

      double xi_val    = detail::get_scalar_value(xi);
      double xiold_val = detail::get_scalar_value(xiold);

      double sk = m_atol + m_rtol * std::max(std::abs(xi_val), std::abs(xiold_val));
      double term = detail::get_max_abs(xerri) / sk;

      err += term * term;
    }
    return std::sqrt(err / static_cast<double>(n));
  }

  double last_error() const { return m_err_old; }

  // ------------------------------------------------------------------------
  // Controlled step interface
  // ------------------------------------------------------------------------

  /**
   * @brief Attempts a single adaptive step and updates x and t if successful.
   */
  template<class System>
  controlled_step_result try_step(System sys, state_type& x, time_type& t, time_type& dt) {
    m_xnew_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_m_xnew<state_type>(std::forward<decltype(arg)>(arg));
    });
    controlled_step_result res = try_step(sys, x, t, m_xnew.m_v, dt);
    if (res == success)
      copy(m_xnew.m_v, x);
    return res;
  }

  /**
   * @brief Core controlled stepper with full adaptive step-size logic.
   */
  template<class System>
  controlled_step_result try_step(System sys,
                                  const state_type& x,
                                  time_type& t,
                                  state_type& xout,
                                  time_type& dt) {
    double dt_val = detail::get_scalar_value(dt);

    // Step-size limit check
    if (m_max_dt != static_cast<time_type>(0)) {
      double max_dt_val = detail::get_scalar_value(m_max_dt);
      if (detail::less_with_sign(max_dt_val, dt_val, dt_val)) {
        dt = m_max_dt;
        return fail;
      }
    }

    using std::pow;
    static const double safe = 0.9, fac1 = 5.0, fac2 = 1.0 / 6.0;

    m_xerr_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_m_xerr<state_type>(std::forward<decltype(arg)>(arg));
    });

    // Perform a Rosenbrock step
    m_stepper.do_step(sys, x, t, xout, dt, m_xerr.m_v);
    double err = error(xout, x, m_xerr.m_v);

    double fac = std::max(fac2, std::min(fac1, pow(err, 0.25) / safe));
    double dt_new = dt_val / fac;

    if (err <= 1.0) {
      if (m_first_step) {
        m_first_step = false;
      } else {
        double fac_pred = (m_dt_old / dt_val) *
          pow(err * err / m_err_old, 0.25) / safe;
        fac_pred = std::max(fac2, std::min(fac1, fac_pred));
        fac = std::max(fac, fac_pred);
        dt_new = dt_val / fac;
      }

      m_dt_old = dt_val;
      m_err_old = std::max(0.01, err);

      if (m_last_rejected)
        dt_new = (dt_val >= 0.0 ? std::min(dt_new, dt_val)
                    : std::max(dt_new, dt_val));

      t += dt;

      if (m_max_dt != static_cast<time_type>(0)) {
        double max_dt_val = detail::get_scalar_value(m_max_dt);
        dt = time_type(std::abs(dt_new) < max_dt_val
                         ? dt_new
                         : (dt_val >= 0.0 ? max_dt_val : -max_dt_val));
      } else {
        dt = time_type(dt_new);
      }

      m_last_rejected = false;
      return success;
    } else {
      dt = time_type(dt_new);
      m_last_rejected = true;
      return fail;
    }
  }

  // ------------------------------------------------------------------------
  // Size management and accessors
  // ------------------------------------------------------------------------
  template<class StateType>
  void adjust_size(const StateType& x) {
    resize_m_xerr(x);
    resize_m_xnew(x);
  }

  stepper_type& stepper() { return m_stepper; }
  const stepper_type& stepper() const { return m_stepper; }

protected:
  template<class StateIn>
  bool resize_m_xerr(const StateIn& x) {
    return adjust_size_by_resizeability(
      m_xerr, x, typename is_resizeable<state_type>::type());
  }

  template<class StateIn>
  bool resize_m_xnew(const StateIn& x) {
    return adjust_size_by_resizeability(
      m_xnew, x, typename is_resizeable<state_type>::type());
  }

private:
  stepper_type m_stepper;
  resizer_type m_xerr_resizer;
  resizer_type m_xnew_resizer;
  wrapped_state_type m_xerr;
  wrapped_state_type m_xnew;

  double m_atol, m_rtol;
  time_type m_max_dt;
  bool m_first_step;
  double m_err_old, m_dt_old;
  bool m_last_rejected;
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // ROSENBROCK4_CONTROLLER_AD_HPP
