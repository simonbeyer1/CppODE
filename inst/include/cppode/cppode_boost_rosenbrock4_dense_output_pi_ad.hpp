/*
 Dense output wrapper for AD-aware Rosenbrock4 with PI controller.

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
 - Extended for FADBAD++ automatic differentiation types
 - Added reinitialize_at_event() for event-driven integration
 - Implemented double-buffering for stable AD interpolation
 - Ensured derivative propagation through interpolation
 - Added full controller reset support
 */

#ifndef CPPODE_ROSENBROCK4_DENSE_OUTPUT_PI_AD_HPP_INCLUDED
#define CPPODE_ROSENBROCK4_DENSE_OUTPUT_PI_AD_HPP_INCLUDED

#include <utility>
#include <boost/numeric/odeint/util/bind.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/integrate/max_step_checker.hpp>

#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <cppode/cppode_boost_rosenbrock4_controller_pi_ad.hpp>

namespace boost {
namespace numeric {
namespace odeint {

/**
 * @class rosenbrock4_dense_output_pi_ad
 * @brief AD-aware dense output wrapper for Rosenbrock4 with event support
 *
 * This class extends rosenbrock4_dense_output_pi to support automatic
 * differentiation types from FADBAD++. It provides continuous output with
 * full derivative propagation through interpolation.
 *
 * **Key Features:**
 * - Hermite cubic interpolation for both values and derivatives
 * - Double-buffered state storage (AD-compatible)
 * - Event reinitialization with controller reset
 * - Guaranteed derivative continuity in interpolation
 *
 * **AD-Specific Considerations:**
 * - Derivatives propagate through calc_state() interpolation
 * - State buffers store full AD information (value + derivatives)
 * - Controller reset clears derivative history after events
 *
 * **Double-Buffering with AD:**
 * @code
 * state_type = vector<fadbad::F<double>>
 *
 * After do_step():
 *   previous_state()[i].x()    → value at t_old
 *   previous_state()[i].d(j)   → derivative at t_old
 *   current_state()[i].x()     → value at t_new
 *   current_state()[i].d(j)    → derivative at t_new
 *
 * calc_state(t, x) interpolates both values and derivatives
 * @endcode
 *
 * @tparam ControlledStepper AD-aware controlled stepper
 *                           (e.g., rosenbrock4_controller_pi_ad<stepper_t>)
 *
 * @par Usage Example
 * @code
 * using ad_t = fadbad::F<double>;
 * using state_t = vector<ad_t>;
 * using stepper_t = rosenbrock4<state_t>;
 * using controller_t = rosenbrock4_controller_pi_ad<stepper_t>;
 * using dense_out_t = rosenbrock4_dense_output_pi_ad<controller_t>;
 *
 * dense_out_t stepper;
 * stepper.initialize(x0, t0, dt0);
 *
 * while (stepper.current_time() < t_end) {
 *     auto [t_old, t_new] = stepper.do_step(system);
 *
 *     // Interpolate (preserves derivatives)
 *     state_t x_interp;
 *     stepper.calc_state(t_mid, x_interp);
 *     // x_interp[i].x()   → interpolated value
 *     // x_interp[i].d(j)  → interpolated derivative
 * }
 * @endcode
 *
 * @see rosenbrock4_dense_output_pi (non-AD version)
 * @see rosenbrock4_controller_pi_ad
 */
template<class ControlledStepper>
class rosenbrock4_dense_output_pi_ad
{
public:
  using controlled_stepper_type = ControlledStepper;
  using unwrapped_stepper =
    typename unwrap_reference<controlled_stepper_type>::type;

  using stepper_type       = typename unwrapped_stepper::stepper_type;
  using state_type         = typename unwrapped_stepper::state_type;
  using wrapped_state_type = typename unwrapped_stepper::wrapped_state_type;
  using time_type          = typename stepper_type::time_type;
  using value_type         = typename stepper_type::value_type;
  using resizer_type       = typename stepper_type::resizer_type;

  using stepper_category = dense_output_stepper_tag;

public:

  /**
   * @brief Construct AD-aware dense output wrapper
   *
   * @param stepper The AD-aware controlled stepper to wrap
   *                (default-constructed if not provided)
   *
   * @note The stepper is copied, so modifications to the original
   *       won't affect this instance
   */
  explicit rosenbrock4_dense_output_pi_ad(
      const controlled_stepper_type& stepper = controlled_stepper_type())
    : m_stepper(stepper)
    , m_x1(), m_x2()
    , m_current_state_x1(true)
    , m_t(), m_t_old(), m_dt()
  {}

  /**
   * @brief Initialize the AD-aware dense output stepper
   *
   * Sets up initial conditions including derivative information.
   * Both state buffers are initialized to ensure valid interpolation.
   *
   * @tparam StateType AD-compatible state type (e.g., vector<fadbad::F<T>>)
   * @param x0 Initial state (including derivative information)
   * @param t0 Initial time
   * @param dt0 Initial step size suggestion
   *
   * @par Derivative Handling
   * If x0 contains initialized derivatives (via fadbad::F::diff()),
   * they are preserved through initialization and propagated through
   * subsequent integration steps.
   *
   * @note After initialization:
   *       - current_time() = previous_time() = t0
   *       - current_state() = previous_state() = x0 (including derivatives)
   */
  template<class StateType>
  void initialize(const StateType& x0, time_type t0, time_type dt0)
  {
    m_resizer.adjust_size(
      x0,
      detail::bind(
        &rosenbrock4_dense_output_pi_ad::template resize_impl<StateType>,
        detail::ref(*this),
        detail::_1));

    get_current_state() = x0;
    get_old_state()     = x0;

    m_t     = t0;
    m_t_old = t0;
    m_dt    = dt0;
  }

  /**
   * @brief Perform one controlled integration step with AD
   *
   * Attempts adaptive steps until one succeeds, propagating derivatives
   * through the entire process. Prepares interpolation data for both
   * values and derivatives.
   *
   * @tparam System ODE system type (must accept AD state types)
   * @param system The ODE system
   * @return Pair (t_old, t_new) representing the integration interval
   *
   * @throw boost::numeric::odeint::step_adjustment_error on failure
   *
   * @par Derivative Propagation
   * The Rosenbrock method's Jacobian computation and automatic
   * differentiation work together to propagate derivatives through
   * the integration step. The dense output coefficients include
   * derivative information for accurate interpolation.
   *
   * @post Valid interpolation interval: [previous_time(), current_time()]
   *       for both values and all derivatives
   */
  template<class System>
  std::pair<time_type,time_type> do_step(System system)
  {
    unwrapped_stepper& cs = m_stepper;

    failed_step_checker fail_checker;
    controlled_step_result result = fail;

    m_t_old = m_t;

    do {
      result = cs.try_step(
        system,
        get_current_state(),
        m_t,
        get_old_state(),
        m_dt );

      fail_checker();
    }
    while (result == fail);

    /* Prepare dense interpolation data (includes derivative coefficients) */
    cs.stepper().prepare_dense_output();

    toggle_current_state();
    return std::make_pair(m_t_old, m_t);
  }

  /**
   * @brief Compute interpolated state including derivatives
   *
   * Uses Hermite cubic interpolation to compute both function values
   * and derivatives at any time within the current integration interval.
   *
   * @tparam StateOut AD-compatible output state type
   * @param t Time at which to interpolate
   * @param x Output state vector (contains interpolated values and derivatives)
   *
   * @pre previous_time() <= t <= current_time()
   *
   * @par Derivative Interpolation
   * The interpolation preserves derivative information:
   * @code
   * // After calc_state(t, x):
   * x[i].x()   → value at time t
   * x[i].d(j)  → ∂x[i]/∂p[j] at time t (interpolated)
   * @endcode
   *
   * This ensures that sensitivity trajectories are continuous and
   * accurate between integration points.
   *
   * @par Interpolation Quality
   * - Order: 3 (cubic Hermite) for both values and derivatives
   * - Continuity: C¹ for values, C⁰ for derivatives
   * - Exact at endpoints for both values and derivatives
   */
  template<class StateOut>
  void calc_state(time_type t, StateOut& x)
  {
    unwrapped_stepper& cs = m_stepper;

    cs.stepper().calc_state(
        t,
        x,
        get_old_state(),  m_t_old,
        get_current_state(), m_t);
  }

  /**
   * @brief Adjust internal buffer sizes
   *
   * Resizes AD-aware state buffers to match the given state size.
   *
   * @tparam StateIn AD-compatible state type
   * @param x Reference state for sizing
   */
  template<class StateIn>
  void adjust_size(const StateIn& x)
  {
    unwrapped_stepper& cs = m_stepper;
    cs.adjust_size(x);
    resize_impl(x);
  }

  /**
   * @brief Reinitialize after discontinuous event (AD-aware)
   *
   * Performs complete reset including derivative information after
   * an event that causes state discontinuity.
   *
   * **Strategy B (Full Reset with Derivatives):**
   * - Both state buffers reset to post-event state (including derivatives)
   * - Time reset to event time
   * - Step size restored
   * - Controller state cleared (error history, derivative tracking)
   *
   * @param x_event Post-event state (including updated derivatives)
   * @param t_event Event time
   * @param dt_before Step size to use after event
   *
   * @par Derivative Handling at Events
   * Events may cause derivative discontinuities (e.g., dose → sensitivity jump).
   * By resetting both buffers to x_event, we ensure:
   * - No invalid derivative interpolation across the event
   * - Controller doesn't use pre-event derivative information
   * - Clean restart of sensitivity propagation
   *
   * @par Example: Dose Event with Sensitivities
   * @code
   * // Event: Add dose to compartment 0
   * state_type x_after = x_before;
   * x_after[0].x() += dose_amount;
   *
   * // Derivative w.r.t. dose amount
   * x_after[0].d(dose_param_idx) += 1.0;
   *
   * // Reset stepper with updated derivatives
   * stepper.reinitialize_at_event(x_after, t_event, dt);
   * @endcode
   *
   * @note Controller's reset_after_event() clears error and derivative
   *       history to prevent contamination of post-event step-size control
   */
  void reinitialize_at_event(const state_type& x_event,
                             time_type t_event,
                             time_type dt_before)
  {
    get_current_state() = x_event;
    get_old_state()     = x_event;

    m_t     = t_event;
    m_t_old = t_event;
    m_dt    = dt_before;

    // Reset controller (clears error history and AD state)
    m_stepper.reset_after_event(dt_before);
  }

  /**
   * @brief Get current state including derivatives
   * @return State at current_time() (AD type with derivatives)
   */
  const state_type& current_state()  const { return get_current_state(); }

  /**
   * @brief Get previous state including derivatives
   * @return State at previous_time() (AD type with derivatives)
   */
  const state_type& previous_state() const { return get_old_state();     }

  /**
   * @brief Get current time
   * @return Current time t_new
   */
  time_type current_time()        const { return m_t;     }

  /**
   * @brief Get previous time
   * @return Previous time t_old
   */
  time_type previous_time()       const { return m_t_old; }

  /**
   * @brief Get current step size
   * @return Step size of last successful step
   */
  time_type current_time_step()   const { return m_dt;    }


private:

  /**
   * @brief Access current state buffer (mutable, AD-aware)
   */
  state_type& get_current_state()
  { return m_current_state_x1 ? m_x1.m_v : m_x2.m_v; }

  /**
   * @brief Access current state buffer (const, AD-aware)
   */
  const state_type& get_current_state() const
  { return m_current_state_x1 ? m_x1.m_v : m_x2.m_v; }

  /**
   * @brief Access previous state buffer (mutable, AD-aware)
   */
  state_type& get_old_state()
  { return m_current_state_x1 ? m_x2.m_v : m_x1.m_v; }

  /**
   * @brief Access previous state buffer (const, AD-aware)
   */
  const state_type& get_old_state() const
  { return m_current_state_x1 ? m_x2.m_v : m_x1.m_v; }

  /**
   * @brief Swap current and previous state buffers
   *
   * Toggles the buffer pointer to avoid copying AD state data.
   * More efficient for AD types which may carry significant
   * derivative information.
   */
  void toggle_current_state()
  {
    m_current_state_x1 = !m_current_state_x1;
  }

  /**
   * @brief Resize both AD-aware state buffers
   *
   * @tparam StateIn AD-compatible input state type
   * @param x Reference state for sizing (including derivative count)
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
  controlled_stepper_type m_stepper;  ///< Wrapped AD-aware controlled stepper
  resizer_type            m_resizer;  ///< Buffer resizer

  wrapped_state_type      m_x1, m_x2; ///< Double-buffered AD state storage
  bool                    m_current_state_x1; ///< Buffer toggle flag

  time_type               m_t;        ///< Current time
  time_type               m_t_old;    ///< Previous time
  time_type               m_dt;       ///< Current step size
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_ROSENBROCK4_DENSE_OUTPUT_PI_AD_HPP_INCLUDED
