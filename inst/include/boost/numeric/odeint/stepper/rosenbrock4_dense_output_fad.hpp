#ifndef BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_AD_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_AD_HPP_INCLUDED

#include <utility>

#include <boost/numeric/odeint/util/bind.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/integrate/max_step_checker.hpp>

#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include "rosenbrock4_controller_fad.hpp"

/**
 * ============================================================================
 * @file rosenbrock4_dense_output_ad.hpp
 * @brief Dense output wrapper for Rosenbrock4 with AD-aware controller.
 *
 * This class is a thin, AD-friendly variant of the original
 * boost::numeric::odeint::rosenbrock4_dense_output. It assumes that the
 * controlled stepper is a rosenbrock4_controller_ad<rosenbrock4<...>>
 * (or a compatible controller with the same interface).
 *
 * Design goals:
 *   - Retain the original cubic Hermite dense output formulas implemented
 *     inside rosenbrock4::prepare_dense_output() and
 *     rosenbrock4::calc_state(...).
 *   - Avoid any manual interpolation outside the stepper. All dense output
 *     is delegated to rosenbrock4 itself, which is fully templated and
 *     therefore works for AD types as long as the scalar type supports
 *     arithmetic.
 *   - Provide the standard dense_output_stepper_tag interface so that this
 *     class can be used with Boost.Odeint integration utilities and with
 *     your event-aware integrate_times_dense implementation.
 *
 * Typical usage:
 *
 *   using AD   = fadbad::F<double>;
 *   using AD2  = fadbad::F<AD>;
 *   using step = rosenbrock4< AD2 >;
 *   using ctrl = rosenbrock4_controller_ad< step >;
 *   using dense_stepper = rosenbrock4_dense_output_ad< ctrl >;
 *
 *   ctrl controller(abstol, reltol);
 *   dense_stepper dense(controller);
 *
 *   dense.initialize(x0, t0, dt0);
 *   // then call dense.do_step(...) and dense.calc_state(t, x) as usual
 *
 *   Author: Simon Beyer
 * ============================================================================
 */

namespace boost {
namespace numeric {
namespace odeint {

template<class ControlledStepper>
class rosenbrock4_dense_output_ad
{
public:
  using controlled_stepper_type          = ControlledStepper;
  using unwrapped_controlled_stepper_type =
    typename unwrap_reference<controlled_stepper_type>::type;
  using stepper_type       = typename unwrapped_controlled_stepper_type::stepper_type;
  using value_type         = typename stepper_type::value_type;
  using state_type         = typename stepper_type::state_type;
  using wrapped_state_type = typename stepper_type::wrapped_state_type;
  using time_type          = typename stepper_type::time_type;
  using deriv_type         = typename stepper_type::deriv_type;
  using wrapped_deriv_type = typename stepper_type::wrapped_deriv_type;
  using resizer_type       = typename stepper_type::resizer_type;
  using stepper_category   = dense_output_stepper_tag;

  using dense_output_stepper_type = rosenbrock4_dense_output_ad<ControlledStepper>;

  // ------------------------------------------------------------------------
  // Constructor
  // ------------------------------------------------------------------------

  /**
   * @brief Construct dense-output wrapper around a controlled Rosenbrock4.
   *
   * The controlled stepper is typically a rosenbrock4_controller_ad,
   * but any controlled stepper exposing "stepper()" with a compatible API
   * (prepare_dense_output, calc_state) can be used.
   */
  explicit rosenbrock4_dense_output_ad(
      const controlled_stepper_type& stepper = controlled_stepper_type())
    : m_stepper(stepper),
      m_x1(), m_x2(),
      m_current_state_x1(true),
      m_t(), m_t_old(), m_dt()
  {}

  // ------------------------------------------------------------------------
  // Initialization
  // ------------------------------------------------------------------------

  /**
   * @brief Initialize dense output at initial state and time.
   *
   * @param x0  Initial state
   * @param t0  Initial time
   * @param dt0 Initial step size
   */
  template<class StateType>
  void initialize(const StateType& x0, time_type t0, time_type dt0)
  {
    m_resizer.adjust_size(
      x0,
      detail::bind(
        &dense_output_stepper_type::template resize_impl<StateType>,
        detail::ref(*this),
        detail::_1));

    get_current_state() = x0;
    m_t  = t0;
    m_dt = dt0;
  }

  // ------------------------------------------------------------------------
  // Integration step
  // ------------------------------------------------------------------------

  /**
   * @brief Perform one adaptive step and prepare dense output.
   *
   * Returns a pair (t_old, t_new) that brackets the step. After this call,
   * dense output at any intermediate time t in [t_old, t_new] can be
   * obtained via calc_state(t, x).
   */
  template<class System>
  std::pair<time_type, time_type> do_step(System system)
  {
    unwrapped_controlled_stepper_type& stepper = m_stepper;
    failed_step_checker fail_checker;  // throws if too many failed attempts
    controlled_step_result res = fail;

    m_t_old = m_t;

    // Try steps until accepted:
    do {
      res = stepper.try_step(system, get_current_state(), m_t, get_old_state(), m_dt);
      fail_checker();
    } while (res == fail);

    // Ask the underlying rosenbrock4 stepper to set up dense output.
    // This computes the internal coefficients for cubic Hermite interpolation
    // between (m_t_old, get_old_state()) and (m_t, get_current_state()).
    stepper.stepper().prepare_dense_output();

    // Swap old/new state buffers:
    toggle_current_state();

    return std::make_pair(m_t_old, m_t);
  }

  // ------------------------------------------------------------------------
  // Dense output: cubic Hermite via underlying Rosenbrock4
  // ------------------------------------------------------------------------

  /**
   * @brief Compute state at intermediate time t (non-const output).
   *
   * Delegates to rosenbrock4::calc_state(t, x, x_old, t_old, x_new, t_new),
   * which implements a cubic Hermite interpolation using stored stage data.
   * Since rosenbrock4 is templated on value_type, this works for AD types
   * without modification.
   */
  template<class StateOut>
  void calc_state(time_type t, StateOut& x)
  {
    unwrapped_controlled_stepper_type& stepper = m_stepper;
    stepper.stepper().calc_state(
        t,
        x,
        get_old_state(),  m_t_old,
        get_current_state(), m_t);
  }

  /**
   * @brief Const-overload variant to work around forwarding issues.
   */
  template<class StateOut>
  void calc_state(time_type t, const StateOut& x)
  {
    unwrapped_controlled_stepper_type& stepper = m_stepper;
    stepper.stepper().calc_state(
        t,
        x,
        get_old_state(),  m_t_old,
        get_current_state(), m_t);
  }

  // ------------------------------------------------------------------------
  // Size management
  // ------------------------------------------------------------------------

  /**
   * @brief Adjust internal buffers to match the size of a given state.
   */
  template<class StateType>
  void adjust_size(const StateType& x)
  {
    unwrapped_controlled_stepper_type& stepper = m_stepper;
    stepper.adjust_size(x);
    resize_impl(x);
  }

  // ------------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------------

  const state_type& current_state() const  { return get_current_state(); }
  time_type         current_time()  const  { return m_t; }

  const state_type& previous_state() const { return get_old_state(); }
  time_type         previous_time() const  { return m_t_old; }

  time_type         current_time_step() const { return m_dt; }

private:
  // ------------------------------------------------------------------------
  // Internal state switching (double-buffering)
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
  // Resize helper
  // ------------------------------------------------------------------------

  template<class StateIn>
  bool resize_impl(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(
      m_x1, x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(
      m_x2, x, typename is_resizeable<state_type>::type());
    return resized;
  }

  // ------------------------------------------------------------------------
  // Member data
  // ------------------------------------------------------------------------

  controlled_stepper_type m_stepper;
  resizer_type            m_resizer;
  wrapped_state_type      m_x1, m_x2;
  bool                    m_current_state_x1;
  time_type               m_t, m_t_old, m_dt;
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_AD_HPP_INCLUDED
