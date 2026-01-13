/*
 Dense output wrapper for Rosenbrock4 with PI controller and event support.

 Original work:
 Copyright (C) 2011-2012 Karsten Ahnert
 Copyright (C) 2011-2015 Mario Mulansky
 Copyright (C) 2012 Christoph Koke
 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)

 Modified work:
 Copyright (C) 2026 Simon Beyer

 Modifications:
 - Added reinitialize_at_event() for event-driven integration
 - Implemented SFINAE-based controller reset detection
 - Extended double-buffering for stable event handling
 */

#ifndef CPPODE_ROSENBROCK4_DENSE_OUTPUT_PI_HPP_INCLUDED
#define CPPODE_ROSENBROCK4_DENSE_OUTPUT_PI_HPP_INCLUDED

#include <utility>
#include <type_traits>

#include <boost/numeric/odeint/util/bind.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/integrate/max_step_checker.hpp>

#include <cppode/cppode_boost_rosenbrock4_controller_pi.hpp>

namespace boost {
namespace numeric {
namespace odeint {

//==============================================================================
// SFINAE Helper: Safe Controller Reset
//==============================================================================

namespace detail {

/**
 * @brief Type trait to detect reset_after_event() method
 *
 * Uses SFINAE to determine if the controlled stepper provides
 * a reset_after_event(Time) method. This enables safe fallback
 * for controllers without event support.
 *
 * @tparam CS Controlled stepper type
 * @tparam Time Time type
 */
template<class CS, class Time, class = void>
struct has_reset_after_event : std::false_type {};

                             template<class CS, class Time>
                             struct has_reset_after_event<CS, Time,
                                                          std::void_t<decltype(std::declval<CS&>().reset_after_event(std::declval<Time>()))>
                             > : std::true_type {};

                             /**
                              * @brief Call reset_after_event() if available (SFINAE enabled)
                              *
                              * @tparam CS Controlled stepper type
                              * @tparam Time Time type
                              * @param cs Controlled stepper instance
                              * @param dt_before Step size before event
                              */
                             template<class CS, class Time>
                             inline typename std::enable_if<has_reset_after_event<CS, Time>::value>::type
                             try_reset_after_event_impl(CS& cs, Time dt_before)
                             {
                               cs.reset_after_event(dt_before);
                             }

                             /**
                              * @brief Fallback for controllers without reset_after_event() (SFINAE disabled)
                              *
                              * No-op for standard Boost.Odeint controllers that don't implement
                              * event reset functionality.
                              *
                              * @tparam CS Controlled stepper type
                              * @tparam Time Time type
                              */
                             template<class CS, class Time>
                             inline typename std::enable_if<!has_reset_after_event<CS, Time>::value>::type
                             try_reset_after_event_impl(CS& /*cs*/, Time /*dt_before*/)
                             {
                               // No-op for plain Boost steppers without reset API
                             }

} // namespace detail


//==============================================================================
// Dense Output Wrapper
//==============================================================================

/**
 * @class rosenbrock4_dense_output_pi
 * @brief Dense output wrapper for Rosenbrock4 with event-driven integration support
 *
 * This class provides continuous output (dense output) for Rosenbrock methods
 * by wrapping a controlled stepper and managing interpolation data. It extends
 * the standard Boost.Odeint dense output interface with event handling capabilities.
 *
 * **Key Features:**
 * - Hermite cubic interpolation between integration steps
 * - Double-buffered state storage for stable interpolation
 * - Event reinitialization support via reinitialize_at_event()
 * - Compatible with any controlled stepper (PI or classical)
 * - Safe fallback for controllers without event support (SFINAE)
 *
 * **Double-Buffering Strategy:**
 * @code
 * After do_step():
 *   previous_state() → state at t_old (start of interval)
 *   current_state()  → state at t_new (end of interval)
 *
 * Valid interpolation: t_old <= t <= t_new
 * @endcode
 *
 * **Event Handling (Strategy B):**
 * After an event at time t_event, both buffers are reset to the
 * post-event state, ensuring clean restart without stale interpolation data.
 *
 * @tparam ControlledStepper A controlled stepper (e.g., rosenbrock4_controller_pi)
 *
 * @par Usage Example
 * @code
 * using controller_t = rosenbrock4_controller_pi<rosenbrock4<double>>;
 * using dense_output_t = rosenbrock4_dense_output_pi<controller_t>;
 *
 * dense_output_t stepper;
 * stepper.initialize(x0, t0, dt0);
 *
 * while (stepper.current_time() < t_end) {
 *     auto [t_old, t_new] = stepper.do_step(system);
 *
 *     // Interpolate at any t in [t_old, t_new]
 *     state_type x_interp;
 *     stepper.calc_state(t_old + 0.5 * (t_new - t_old), x_interp);
 * }
 * @endcode
 *
 * @see rosenbrock4_controller_pi
 * @see rosenbrock4_dense_output_pi_ad (AD version)
 */
template<class ControlledStepper>
class rosenbrock4_dense_output_pi
{
public:
  typedef ControlledStepper controlled_stepper_type;
  typedef typename unwrap_reference<controlled_stepper_type>::type unwrapped_stepper;

  typedef typename unwrapped_stepper::stepper_type       stepper_type;
  typedef typename unwrapped_stepper::state_type         state_type;
  typedef typename unwrapped_stepper::wrapped_state_type wrapped_state_type;
  typedef typename stepper_type::time_type               time_type;
  typedef typename stepper_type::resizer_type            resizer_type;

  typedef dense_output_stepper_tag stepper_category;

  /**
   * @brief Construct dense output wrapper
   *
   * @param stepper The controlled stepper instance to wrap
   *                (default-constructed if not provided)
   *
   * @note The controlled stepper is copied, so modifications to the
   *       original won't affect this instance
   */
  explicit rosenbrock4_dense_output_pi(
      const controlled_stepper_type &stepper = controlled_stepper_type())
    : m_stepper(stepper)
    , m_x1(), m_x2()
    , m_current_state_x1(true)
    , m_t(), m_t_old(), m_dt()
  {}

  /**
   * @brief Initialize the dense output stepper
   *
   * Sets up initial conditions and internal buffers. Both state buffers
   * are initialized to x0 to ensure valid interpolation from the start.
   *
   * @tparam StateType State vector type (must be compatible with state_type)
   * @param x0 Initial state
   * @param t0 Initial time
   * @param dt0 Initial step size suggestion
   *
   * @note After initialization:
   *       - current_time() = previous_time() = t0
   *       - current_state() = previous_state() = x0
   *       - Valid interpolation: only at t = t0 until first do_step()
   */
  template<class StateType>
  void initialize(const StateType &x0, time_type t0, time_type dt0)
  {
    m_resizer.adjust_size(
      x0,
      detail::bind(
        &rosenbrock4_dense_output_pi::template resize_impl<StateType>,
        detail::ref(*this),
        detail::_1));

    get_current_state() = x0;
    get_old_state()     = x0;

    m_t     = t0;
    m_t_old = t0;
    m_dt    = dt0;
  }

  /**
   * @brief Perform one controlled integration step
   *
   * Attempts steps with adaptive step-size control until one succeeds,
   * then prepares interpolation data and updates the time interval.
   *
   * @tparam System ODE system type (callable: void(x, dxdt, t))
   * @param system The ODE system
   * @return Pair (t_old, t_new) representing the integration interval
   *
   * @throw boost::numeric::odeint::step_adjustment_error if step size
   *        adjustment fails after max attempts (default: 500)
   *
   * @par Post-conditions
   * - previous_time() = t_old (start of interval)
   * - current_time() = t_new (end of interval)
   * - calc_state(t, x) valid for t in [t_old, t_new]
   * - current_time_step() = accepted step size
   *
   * @par Implementation Details
   * After a successful step:
   * 1. Call prepare_dense_output() on the underlying stepper
   * 2. Swap state buffers (previous ↔ current)
   * 3. Return the time interval for user processing
   */
  template<class System>
  std::pair<time_type, time_type> do_step(System system)
  {
    unwrapped_stepper &cs = m_stepper;

    failed_step_checker fail_checker;
    controlled_step_result result = fail;

    m_t_old = m_t;

    do {
      result = cs.try_step(
        system,
        get_current_state(),
        m_t,
        get_old_state(),
        m_dt);

      fail_checker();
    }
    while (result == fail);

    // Prepare Hermite cubic interpolation coefficients
    cs.stepper().prepare_dense_output();

    // Swap buffers: what was "new" becomes "old" for next step
    toggle_current_state();

    return std::make_pair(m_t_old, m_t);
  }

  /**
   * @brief Compute interpolated state at arbitrary time
   *
   * Uses Hermite cubic interpolation to compute the state at any time
   * within the current integration interval [previous_time(), current_time()].
   *
   * @tparam StateOut Output state type (must be compatible with state_type)
   * @param t Time at which to interpolate
   * @param x Output state vector (modified to contain interpolated state)
   *
   * @pre previous_time() <= t <= current_time()
   *
   * @note Interpolation outside the valid interval may produce
   *       inaccurate results or undefined behavior
   *
   * @par Interpolation Quality
   * - Order: 3 (cubic Hermite)
   * - Continuity: C¹ at interval boundaries
   * - Exact at endpoints: calc_state(t_old, x) = previous_state()
   *                       calc_state(t_new, x) = current_state()
   */
  template<class StateOut>
  void calc_state(time_type t, StateOut &x)
  {
    unwrapped_stepper &cs = m_stepper;

    cs.stepper().calc_state(
        t,
        x,
        get_old_state(),  m_t_old,
        get_current_state(), m_t);
  }

  /**
   * @brief Adjust internal buffer sizes
   *
   * Resizes internal state buffers to match the given state size.
   * Automatically called by initialize(), but can be called manually
   * if state size changes during integration.
   *
   * @tparam StateType State vector type
   * @param x Reference state for sizing
   */
  template<class StateType>
  void adjust_size(const StateType& x)
  {
    unwrapped_stepper &cs = m_stepper;
    cs.adjust_size(x);
    resize_impl(x);
  }

  /**
   * @brief Reinitialize after a discontinuous event
   *
   * Performs a complete reset of the dense output stepper after an event
   * that causes a state discontinuity (e.g., dose administration, state
   * replacement, instantaneous change).
   *
   * **Strategy B (Full Reset):**
   * - Both state buffers reset to post-event state
   * - Time reset to event time
   * - Step size restored to pre-event value
   * - Controller state cleared (if supported)
   *
   * @param x_event Post-event state (after discontinuity applied)
   * @param t_event Event time
   * @param dt_before Step size to use after event (typically dt before event)
   *
   * @par Why Full Reset?
   * After a state discontinuity, interpolation data from before the event
   * becomes invalid. By resetting both buffers to the post-event state,
   * we ensure:
   * - No invalid interpolation across the discontinuity
   * - Clean restart of integration
   * - Controller doesn't use pre-event error history
   *
   * @par Controller Compatibility
   * - If controller implements reset_after_event(): Called automatically
   * - If not: Safe no-op (standard Boost controllers)
   *
   * @note After this call, previous_time() = current_time() = t_event
   *       until the next do_step()
   */
  void reinitialize_at_event(const state_type &x_event,
                             time_type t_event,
                             time_type dt_before)
  {
    get_current_state() = x_event;
    get_old_state()     = x_event;

    m_t     = t_event;
    m_t_old = t_event;
    m_dt    = dt_before;

    // Safely reset controller state if supported (SFINAE dispatch)
    detail::try_reset_after_event_impl(m_stepper, dt_before);
  }

  /**
   * @brief Get current state (end of last integration interval)
   * @return State at current_time()
   */
  const state_type& current_state()  const { return get_current_state(); }

  /**
   * @brief Get previous state (start of last integration interval)
   * @return State at previous_time()
   */
  const state_type& previous_state() const { return get_old_state();     }

  /**
   * @brief Get current time (end of last integration interval)
   * @return Current time t_new
   */
  time_type current_time()      const { return m_t;     }

  /**
   * @brief Get previous time (start of last integration interval)
   * @return Previous time t_old
   */
  time_type previous_time()     const { return m_t_old; }

  /**
   * @brief Get current step size
   * @return Step size of last successful step (or initial dt if no steps taken)
   */
  time_type current_time_step() const { return m_dt;    }


private:

  /**
   * @brief Access current state buffer (mutable)
   *
   * Returns reference to whichever buffer currently holds the "new" state.
   * The active buffer alternates after each do_step() via toggle_current_state().
   */
  state_type& get_current_state()
  {
    return m_current_state_x1 ? m_x1.m_v : m_x2.m_v;
  }

  /**
   * @brief Access current state buffer (const)
   */
  const state_type& get_current_state() const
  {
    return m_current_state_x1 ? m_x1.m_v : m_x2.m_v;
  }

  /**
   * @brief Access previous state buffer (mutable)
   *
   * Returns reference to whichever buffer currently holds the "old" state.
   */
  state_type& get_old_state()
  {
    return m_current_state_x1 ? m_x2.m_v : m_x1.m_v;
  }

  /**
   * @brief Access previous state buffer (const)
   */
  const state_type& get_old_state() const
  {
    return m_current_state_x1 ? m_x2.m_v : m_x1.m_v;
  }

  /**
   * @brief Swap current and previous state buffers
   *
   * Toggles the buffer pointer after each successful step.
   * This avoids copying state data between buffers.
   */
  void toggle_current_state()
  {
    m_current_state_x1 = !m_current_state_x1;
  }

  /**
   * @brief Resize both state buffers
   *
   * @tparam StateIn Input state type
   * @param x Reference state for sizing
   * @return true if any buffer was resized
   */
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


private:
  controlled_stepper_type m_stepper;  ///< Wrapped controlled stepper
  resizer_type            m_resizer;  ///< Buffer resizer

  wrapped_state_type      m_x1, m_x2; ///< Double-buffered state storage
  bool                    m_current_state_x1; ///< Buffer toggle flag

  time_type               m_t;        ///< Current time (end of interval)
  time_type               m_t_old;    ///< Previous time (start of interval)
  time_type               m_dt;       ///< Current step size
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_ROSENBROCK4_DENSE_OUTPUT_PI_HPP_INCLUDED
