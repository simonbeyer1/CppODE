#ifndef BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_AD_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_AD_HPP_INCLUDED

#include <utility>
#include <type_traits>
#include <boost/numeric/odeint/util/bind.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller_fad.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/integrate/max_step_checker.hpp>
#include <fadbad++/fadiff.h>

/**
 * ============================================================================
 * @file rosenbrock4_dense_output_ad.hpp
 * @brief Dense output stepper for Rosenbrock4 with FADBAD++ support
 *
 * Based on the original rosenbrock4_dense_output but with AD-aware
 * interpolation for nested FADBAD types.
 *
 * @author Simon Beyer
 * @date 2025
 * ============================================================================
 */

namespace boost {
namespace numeric {
namespace odeint {

namespace detail {

/**
 * @brief Scalar extraction for plain doubles (base case)
 */
inline double get_scalar_val(double v) {
  return v;
}

/**
 * @brief Recursive scalar extraction for FADBAD types
 */
template<typename T>
inline double get_scalar_val(const fadbad::F<T>& v) {
  return get_scalar_val(const_cast<fadbad::F<T>&>(v).x());
}

/**
 * @brief Linear interpolation for plain doubles
 */
inline double lerp(double a, double b, double alpha) {
  return a + alpha * (b - a);
}

/**
 * @brief Recursive linear interpolation for FADBAD types
 *
 * Uses FADBAD's overloaded operators to automatically and correctly
 * propagate all derivative information through the interpolation.
 * This is CRITICAL - manual assignment would lose derivative tracking!
 *
 * @tparam T Base type (double or F<...>)
 * @param a Start value at t_old
 * @param b End value at t_new
 * @param alpha Interpolation parameter in [0,1]
 * @return Interpolated FADBAD variable with all derivatives correctly propagated
 */
template<typename T>
inline fadbad::F<T> lerp(const fadbad::F<T>& a, const fadbad::F<T>& b, double alpha) {
  // CRITICAL: Use FADBAD's overloaded operators!
  // These automatically propagate derivative information.
  // Manual manipulation of .x() and .d() would lose tracking!
  return a * (1.0 - alpha) + b * alpha;
}

} // namespace detail

/**
 * @class rosenbrock4_dense_output_ad
 * @brief AD-aware dense output stepper for Rosenbrock4
 */
template<class ControlledStepper>
class rosenbrock4_dense_output_ad
{
public:
  typedef ControlledStepper controlled_stepper_type;
  typedef typename unwrap_reference<controlled_stepper_type>::type unwrapped_controlled_stepper_type;
  typedef typename unwrapped_controlled_stepper_type::stepper_type stepper_type;
  typedef typename stepper_type::value_type value_type;
  typedef typename stepper_type::state_type state_type;
  typedef typename stepper_type::wrapped_state_type wrapped_state_type;
  typedef typename stepper_type::time_type time_type;
  typedef typename stepper_type::deriv_type deriv_type;
  typedef typename stepper_type::wrapped_deriv_type wrapped_deriv_type;
  typedef typename stepper_type::resizer_type resizer_type;
  typedef dense_output_stepper_tag stepper_category;
  typedef rosenbrock4_dense_output_ad<ControlledStepper> dense_output_stepper_type;

  // ------------------------------------------------------------------------
  // Constructor
  // ------------------------------------------------------------------------

  rosenbrock4_dense_output_ad(const controlled_stepper_type& stepper = controlled_stepper_type())
    : m_stepper(stepper),
      m_x1(), m_x2(),
      m_current_state_x1(true),
      m_t(), m_t_old(), m_dt()
  {}

  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------

  template<class StateType>
  void initialize(const StateType& x0, time_type t0, time_type dt0)
  {
    m_resizer.adjust_size(x0, detail::bind(
        &dense_output_stepper_type::template resize_impl<StateType>,
        detail::ref(*this), detail::_1));
    get_current_state() = x0;
    m_t = t0;
    m_dt = dt0;
  }

  // ------------------------------------------------------------------------
  // Integration step (same as original)
  // ------------------------------------------------------------------------

  template<class System>
  std::pair<time_type, time_type> do_step(System system)
  {
    unwrapped_controlled_stepper_type& stepper = m_stepper;
    failed_step_checker fail_checker;  // to throw a runtime_error if step size adjustment fails
    controlled_step_result res = fail;
    m_t_old = m_t;
    do
    {
      res = stepper.try_step(system, get_current_state(), m_t, get_old_state(), m_dt);
      fail_checker();  // check for overflow of failed steps
    }
    while(res == fail);

    // NOTE: prepare_dense_output() doesn't work properly with nested AD types
    // so we skip it and do simple linear interpolation in calc_state()
    // stepper.stepper().prepare_dense_output();

    this->toggle_current_state();
    return std::make_pair(m_t_old, m_t);
  }

  // ------------------------------------------------------------------------
  // Dense output - AD-aware interpolation
  // ------------------------------------------------------------------------

  /**
   * @brief Calculate interpolated state at arbitrary time t
   *
   * Uses linear interpolation that properly handles nested AD types.
   * This replaces the stepper's internal dense output which doesn't
   * correctly interpolate derivative components.
   */
  template<class StateOut>
  void calc_state(time_type t, StateOut& x)
  {
    // Ensure output state has correct size
    if (x.size() != get_current_state().size()) {
      x.resize(get_current_state().size());
    }

    // Compute interpolation parameter alpha ∈ [0,1]
    double dt_total = detail::get_scalar_val(m_t - m_t_old);
    double dt_partial = detail::get_scalar_val(t - m_t_old);
    double alpha = dt_partial / dt_total;

    // Linear interpolation for each state component
    // FADBAD's overloaded operators handle derivative propagation automatically
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = detail::lerp(get_old_state()[i], get_current_state()[i], alpha);
    }
  }

  template<class StateOut>
  void calc_state(time_type t, const StateOut& x)
  {
    calc_state(t, const_cast<StateOut&>(x));
  }

  // ------------------------------------------------------------------------
  // Size management
  // ------------------------------------------------------------------------

  template<class StateType>
  void adjust_size(const StateType& x)
  {
    unwrapped_controlled_stepper_type& stepper = m_stepper;
    stepper.adjust_size(x);
    resize_impl(x);
  }

  // ------------------------------------------------------------------------
  // Accessors (same as original)
  // ------------------------------------------------------------------------

  const state_type& current_state() const
  {
    return get_current_state();
  }

  time_type current_time() const
  {
    return m_t;
  }

  const state_type& previous_state() const
  {
    return get_old_state();
  }

  time_type previous_time() const
  {
    return m_t_old;
  }

  time_type current_time_step() const
  {
    return m_dt;
  }

private:
  // ------------------------------------------------------------------------
  // State management (same as original)
  // ------------------------------------------------------------------------

  state_type& get_current_state()
  {
    return m_current_state_x1 ? m_x1.m_v : m_x2.m_v;
  }

  const state_type& get_current_state() const
  {
    return m_current_state_x1 ? m_x1.m_v : m_x2.m_v;
  }

  state_type& get_old_state()
  {
    return m_current_state_x1 ? m_x2.m_v : m_x1.m_v;
  }

  const state_type& get_old_state() const
  {
    return m_current_state_x1 ? m_x2.m_v : m_x1.m_v;
  }

  void toggle_current_state()
  {
    m_current_state_x1 = !m_current_state_x1;
  }

  // ------------------------------------------------------------------------
  // Resize implementation (same as original)
  // ------------------------------------------------------------------------

  template<class StateIn>
  bool resize_impl(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(m_x1, x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(m_x2, x, typename is_resizeable<state_type>::type());
    return resized;
  }

  // ------------------------------------------------------------------------
  // Member variables (same as original)
  // ------------------------------------------------------------------------

  controlled_stepper_type m_stepper;
  resizer_type m_resizer;
  wrapped_state_type m_x1, m_x2;
  bool m_current_state_x1;
  time_type m_t, m_t_old, m_dt;
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_AD_HPP_INCLUDED
