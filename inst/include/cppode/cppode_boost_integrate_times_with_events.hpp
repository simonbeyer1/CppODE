#ifndef CPPODE_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
#define CPPODE_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED

/**
 * @file integrate_times_with_events.hpp
 * @brief Unified event-aware integration engine for Boost.Odeint
 *
 * This header provides a comprehensive event handling system for ODE integration,
 * supporting both fixed-time events and root-finding events with automatic
 * differentiation (AD) compatibility.
 *
 * @section features Features
 *
 * - **Fixed-time events**: Trigger state modifications at predetermined times
 * - **Root-finding events**: Detect zero-crossings of user-defined functions
 *   and trigger state modifications when roots are found
 * - **Root localization**: Bisection-based refinement to locate roots with
 *   configurable tolerance
 * - **Dual stepper support**: Works with both controlled steppers (try_step)
 *   and dense-output steppers (do_step + interpolation)
 * - **AD compatibility**: Full support for nested automatic differentiation
 *   types (fadbad::F<F<...>>) through scalar_value unwrapping
 *
 * @section usage Usage
 *
 * @code{.cpp}
 * // Define fixed events (e.g., dosing at specific times)
 * std::vector<FixedEvent<double>> fixed_events = {
 *     {10.0, 0, 1.0, EventMethod::Add},  // Add 1.0 to state[0] at t=10
 * };
 *
 * // Define root events (e.g., trigger when state crosses threshold)
 * std::vector<RootEvent<state_type, double>> root_events = {
 *     {[](const state_type& x, double t) { return x[0] - 0.5; },
 *      0, 1.0, EventMethod::Replace}  // Replace state[0] with 1.0 when x[0]=0.5
 * };
 *
 * // Integrate with events
 * integrate_times(stepper, system, x, times.begin(), times.end(), dt,
 *                 observer, fixed_events, root_events, checker);
 * @endcode
 *
 * @section compatibility Stepper Compatibility
 *
 * The engine automatically adapts to different stepper types:
 *
 * - **rosenbrock4_controller_ad**: Uses reset_after_event(dt) for reinitialization
 * - **rosenbrock4_dense_output_ad**: Uses reinitialize_at_event(x,t,dt) for full reset
 * - **Standard Odeint steppers**: Fallback behavior (no special reset needed)
 *
 * @author Simon Beyer
 * @date 22th December 2025
 *
 * @see boost::numeric::odeint for the underlying integration framework
 * @see fadbad++ for automatic differentiation support
 */

#include <vector>
#include <functional>
#include <cmath>
#include <limits>
#include <algorithm>
#include <type_traits>

#include <boost/numeric/odeint/util/detail/less_with_sign.hpp>
#include <boost/numeric/odeint/stepper/stepper_categories.hpp>
#include <cppode/cppode_boost_step_checker.hpp>

#include <fadbad++/fadiff.h>

namespace boost {
namespace numeric {
namespace odeint {
namespace detail {

//==============================================================================
// Section 1: Scalar Value Extraction
//==============================================================================

/**
 * @brief Extract the underlying scalar value from arithmetic types
 *
 * Base case for the scalar_value template recursion. Simply returns
 * the input value when it's already an arithmetic type.
 *
 * @tparam T Arithmetic type (double, float, int, etc.)
 * @param v The value to extract
 * @return The input value as a double
 */
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v)
{
  return static_cast<double>(v);
}

/**
 * @brief Extract the underlying scalar value from FADBAD++ AD types
 *
 * Recursively unwraps nested fadbad::F<T> types to extract the underlying
 * scalar value. This enables seamless comparison and root-finding operations
 * on AD-enhanced state variables.
 *
 * @tparam T The inner type (may be another fadbad::F or a scalar)
 * @param v The AD value to unwrap
 * @return The underlying scalar value as a double
 *
 * @note Uses const_cast to call non-const x() method; this is safe as
 *       we only read the value
 */
template<class T>
inline double scalar_value(const fadbad::F<T>& v)
{
  return scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

//==============================================================================
// Section 2: Event Type Definitions
//==============================================================================

/**
 * @brief Enumeration of event application methods
 *
 * Defines how an event modifies the target state variable when triggered.
 */
enum class EventMethod {
  Replace,   ///< Replace: x[i] = value
  Add,       ///< Add: x[i] += value
  Multiply   ///< Multiply: x[i] *= value
};

/**
 * @brief Fixed-time event specification
 *
 * Represents an event that triggers at a predetermined time point.
 * When the integration reaches the specified time, the event modifies
 * a state variable according to the specified method.
 *
 * @tparam value_type The numeric type for time and value (supports AD types)
 *
 * @par Example
 * @code{.cpp}
 * // Add a dose of 100 units to compartment 0 at time 5.0
 * FixedEvent<double> dose = {5.0, 0, 100.0, EventMethod::Add};
 * @endcode
 */
template<class value_type>
struct FixedEvent {
  value_type  time;         ///< Time at which the event triggers
  int         state_index;  ///< Index of the state variable to modify
  value_type  value;        ///< Value to apply (interpretation depends on method)
  EventMethod method;       ///< How to apply the value (Replace/Add/Multiply)
};

/**
 * @brief Root-finding event specification
 *
 * Represents an event that triggers when a user-defined function crosses zero.
 * The engine monitors the function value during integration and triggers the
 * event when a sign change is detected, using bisection to localize the root.
 *
 * @tparam state_type The ODE state vector type
 * @tparam time_type The time variable type
 *
 * @par Example
 * @code{.cpp}
 * // Trigger when concentration drops below threshold
 * RootEvent<state_type, double> threshold_event = {
 *     [](const state_type& x, double t) { return x[0] - 0.1; },  // Root function
 *     0,                    // State index to modify
 *     1.0,                  // New value
 *     EventMethod::Replace  // Replace state[0] with 1.0
 * };
 * @endcode
 */
template<class state_type, class time_type>
struct RootEvent {
  using value_t = typename state_type::value_type;

  /// Function that returns zero at the event time: f(x,t) = 0
  std::function<value_t(const state_type&, const time_type&)> func;

  int         state_index;  ///< Index of the state variable to modify
  value_t     value;        ///< Value to apply when event triggers
  EventMethod method;       ///< How to apply the value (Replace/Add/Multiply)
};

//==============================================================================
// Section 3: Event Application
//==============================================================================

/**
 * @brief Apply an event modification to a state vector
 *
 * Modifies a single element of the state vector according to the
 * specified method (Replace, Add, or Multiply).
 *
 * @tparam state_type The ODE state vector type
 * @param x The state vector to modify (modified in place)
 * @param idx Index of the state element to modify
 * @param v Value to apply
 * @param method How to apply the value
 */
template<class state_type>
inline void apply_event(
    state_type& x,
    int idx,
    const typename state_type::value_type& v,
    EventMethod method)
{
  switch (method) {
  case EventMethod::Replace:  x[idx]  = v;  break;
  case EventMethod::Add:      x[idx] += v;  break;
  case EventMethod::Multiply: x[idx] *= v;  break;
  }
}

/**
 * @brief Apply all fixed events scheduled for a specific time
 *
 * Iterates through the fixed event list and applies any events whose
 * trigger time matches the current time (using exact floating-point
 * comparison on scalar values).
 *
 * @tparam state_type The ODE state vector type
 * @tparam Time The time type (may be AD-enhanced)
 * @param x The state vector to modify (modified in place)
 * @param t Current integration time
 * @param evs Vector of fixed events to check
 * @return true if at least one event was applied, false otherwise
 */
template<class state_type, class Time>
bool apply_fixed_events_at_time(
    state_type& x,
    const Time& t,
    const std::vector<FixedEvent<typename state_type::value_type>>& evs)
{
  const double tt = scalar_value(t);
  bool fired = false;

  for (const auto& e : evs) {
    if (scalar_value(e.time) == tt) {
      apply_event(x, e.state_index, e.value, e.method);
      fired = true;
    }
  }

  return fired;
}

//==============================================================================
// Section 4: Time Point Management
//==============================================================================

/**
 * @brief Merge user-requested output times with fixed event times
 *
 * Creates a sorted, deduplicated vector of all time points where the
 * integrator needs to stop: both user-requested output times and
 * fixed event trigger times.
 *
 * @tparam Time The time type for output
 * @tparam It Iterator type for user times
 * @tparam V Value type of fixed events
 * @param ubegin Iterator to first user time
 * @param uend Iterator past last user time
 * @param fix Vector of fixed events
 * @return Sorted vector of unique time points
 */
template<class Time, class It, class V>
std::vector<Time> merge_user_and_event_times(
    It ubegin, It uend,
    const std::vector<FixedEvent<V>>& fix)
{
  // Collect all time points
  std::vector<Time> out(ubegin, uend);
  for (const auto& e : fix) {
    out.push_back(e.time);
  }

  // Sort by scalar value (handles AD types correctly)
  std::sort(out.begin(), out.end(),
            [](auto& a, auto& b) {
              return scalar_value(a) < scalar_value(b);
            });

  // Remove duplicates
  out.erase(std::unique(out.begin(), out.end(),
                        [](auto& a, auto& b) {
                          return scalar_value(a) == scalar_value(b);
                        }),
                        out.end());

  return out;
}

//==============================================================================
// Section 5: Stepper Reset Dispatch
//==============================================================================

/**
 * @brief Unified stepper reset after discontinuous events
 *
 * Uses C++20 concepts (via requires expressions) to detect and call
 * the appropriate reset method for different stepper types:
 *
 * - rosenbrock4_controller_ad: reset_after_event(dt)
 * - rosenbrock4_dense_output_ad: reinitialize_at_event(x, t, dt)
 * - Other steppers: no-op (some steppers don't need reset)
 *
 * @tparam S Stepper type
 * @tparam State State vector type
 * @tparam Time Time type
 * @param st The stepper instance
 * @param x Current state (used for dense output reinitialization)
 * @param t Current time
 * @param dt Reference to step size (may be modified by reset)
 */
template<class S, class State, class Time>
inline void reset_stepper_unified(S& st, State& x, Time t, Time& dt)
{
  if constexpr (requires { st.reset_after_event(dt); }) {
    // Controlled stepper with event reset support
    st.reset_after_event(dt);
  }
  else if constexpr (requires { st.reinitialize_at_event(x, t, dt); }) {
    // Dense output stepper with full reinitialization
    st.reinitialize_at_event(x, t, dt);
  }
  else {
    // Fallback: stepper doesn't need special reset
    (void)st; (void)x; (void)t; (void)dt;
  }
}

//==============================================================================
// Section 6: Event Engine Core
//==============================================================================

/**
 * @brief Core event-aware integration engine
 *
 * This class implements the main integration loops for both controlled
 * and dense-output steppers, with full support for fixed-time and
 * root-finding events.
 *
 * @tparam Stepper The ODE stepper type
 * @tparam System The ODE system type (callable with signature void(x, dxdt, t))
 * @tparam State The state vector type
 * @tparam Time The time type
 *
 * @section algorithm Algorithm Overview
 *
 * **Controlled Stepper Mode (process_controlled)**:
 * 1. Take adaptive steps using try_step()
 * 2. After each successful step, evaluate root functions
 * 3. On sign change detection, use bisection with re-integration to localize
 * 4. Apply events and reset stepper state
 *
 * **Dense Output Mode (process_dense)**:
 * 1. Take steps using do_step(), which provides interpolation data
 * 2. For each output time in the step interval, interpolate state
 * 3. Check for root crossings using interpolated states
 * 4. Use bisection with calc_state() for efficient root localization
 * 5. Apply events and reinitialize stepper
 *
 * @note The dense output mode is generally more efficient for root-finding
 *       as it can use interpolation instead of re-integration during bisection.
 */
template<class Stepper, class System, class State, class Time>
class EventEngine {
public:
  using state_type = State;
  using time_type  = Time;

  /**
   * @brief Construct an EventEngine
   *
   * @param st Reference to the stepper (must outlive the engine)
   * @param sys Reference to the ODE system (must outlive the engine)
   * @param fixed Vector of fixed-time events
   * @param root Vector of root-finding events
   */
  EventEngine(
    Stepper& st,
    System& sys,
    const std::vector<FixedEvent<typename State::value_type>>& fixed,
    const std::vector<RootEvent<State, Time>>& root)
    : m_st(st)
    , m_sys(sys)
    , m_fixed(fixed)
    , m_root(root)
  {}

  //--------------------------------------------------------------------------
  // Controlled Stepper Integration
  //--------------------------------------------------------------------------

  /**
   * @brief Integrate using a controlled (adaptive) stepper with event handling
   *
   * This method implements event-aware integration for steppers that provide
   * try_step() interface (e.g., rosenbrock4_controller). Root localization
   * is performed by re-integrating with smaller steps during bisection.
   *
   * @tparam Obs Observer type with signature void(const state_type&, time_type)
   * @tparam Checker Step checker type (e.g., StepChecker for max step limits)
   *
   * @param x Initial state (modified to final state on return)
   * @param times Sorted vector of output times
   * @param dt Initial step size suggestion
   * @param obs Observer called at each output time and event
   * @param checker Step checker for detecting integration failures
   * @param root_tol Tolerance for root localization (bisection stops when
   *                 interval width < root_tol)
   * @param max_trigger Maximum number of times each root event can fire
   *
   * @return Total number of successful integration steps taken
   */
  template<class Obs, class Checker>
  size_t process_controlled(
      state_type& x,
      const std::vector<Time>& times,
      Time dt,
      Obs& obs,
      Checker& checker,
      double root_tol,
      size_t max_trigger)
  {
    using boost::numeric::odeint::detail::less_with_sign;

    size_t steps = 0;
    auto it = times.begin();
    const auto end = times.end();
    Time t = *it;

    // --- Handle events at initial time ---
    if (apply_fixed_events_at_time(x, t, m_fixed)) {
      reset_stepper_unified(m_st, x, t, dt);
    }

    // Output initial state
    obs(x, t);
    ++it;
    if (it == end) return 0;

    // --- Initialize root tracking ---
    // last_val: Previous value of each root function (NaN = not yet evaluated)
    // last_state: State at which last_val was computed
    // last_time: Time at which last_val was computed
    // fired: Number of times each root event has triggered
    std::vector<double> last_val(m_root.size(),
                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<State>  last_state(m_root.size(), x);
    std::vector<Time>   last_time(m_root.size(), t);
    std::vector<size_t> fired(m_root.size(), 0);

    // --- Main integration loop ---
    while (it != end)
    {
      Time t_target = *it;

      // Integrate towards target time
      while (less_with_sign(t, t_target, dt))
      {
        Time dt_step = min_abs(dt, t_target - t);
        auto result = m_st.try_step(m_sys, x, t, dt_step);

        if (result == success) {
          ++steps;
          checker();
          checker.reset();
          dt = dt_step;

          // --- Check for root crossings ---
          for (size_t i = 0; i < m_root.size(); ++i)
          {
            auto fval = m_root[i].func(x, t);
            double f_now = scalar_value(fval);

            // Detect sign change (root crossing)
            if (fired[i] < max_trigger &&
                !std::isnan(last_val[i]) &&
                last_val[i] * f_now < 0.0)
            {
              // Localize root via bisection with re-integration
              localize_root_controlled(
                i,
                last_state[i], last_time[i],
                                        x, t,
                                        last_val[i], f_now,
                                        root_tol,
                                        checker);

              // Output state at root (before event)
              Time t_root = t;
              obs(x, t_root);

              // Apply event
              State x_after = x;
              apply_event(x_after,
                          m_root[i].state_index,
                          m_root[i].value,
                          m_root[i].method);
              fired[i]++;

              // Output state after event (with tiny time offset for plotting)
              obs(x_after, t_root + Time(1e-15));
              x = x_after;

              // Reset stepper for clean restart after discontinuity
              reset_stepper_unified(m_st, x, t, dt);

              // Reset root tracking (keep fired counts)
              for (size_t j = 0; j < m_root.size(); ++j) {
                last_val[j] = std::numeric_limits<double>::quiet_NaN();
                last_state[j] = x;
                last_time[j] = t;
              }

              break;  // Exit root loop, continue integration
            }
            else {
              // Update tracking for next comparison
              last_val[i] = f_now;
              last_state[i] = x;
              last_time[i] = t;
            }
          }
        }
        else {
          // Step rejected - try again with smaller step
          checker();
          dt = dt_step;
        }
      }

      // Reached target time
      t = t_target;

      // Check for fixed events at this time
      if (apply_fixed_events_at_time(x, t, m_fixed)) {
        obs(x, t);
        reset_memory(fired, last_val, last_state, last_time, x, t);
      }
      else {
        obs(x, t);
      }

      ++it;
    }

    return steps;
  }

  //--------------------------------------------------------------------------
  // Dense Output Stepper Integration
  //--------------------------------------------------------------------------

  /**
   * @brief Integrate using a dense-output stepper with event handling
   *
   * This method implements event-aware integration for steppers that provide
   * dense output (interpolation) capability. Root localization uses the
   * efficient calc_state() interpolation instead of re-integration.
   *
   * The dense output approach is particularly efficient when:
   * - Many output times fall within a single integration step
   * - Root localization requires many bisection iterations
   * - The ODE system is expensive to evaluate
   *
   * @tparam Obs Observer type with signature void(const state_type&, time_type)
   * @tparam Checker Step checker type
   *
   * @param x Initial state (modified to final state on return)
   * @param times Sorted vector of output times
   * @param dt Initial step size suggestion
   * @param obs Observer called at each output time and event
   * @param checker Step checker for detecting integration failures
   * @param root_tol Tolerance for root localization
   * @param max_trigger Maximum triggers per root event
   *
   * @return Total number of integration steps taken
   *
   * @note After do_step(), the valid interpolation interval is
   *       [previous_time(), current_time()]. All root localization
   *       must occur within this interval.
   */
  template<class Obs, class Checker>
  size_t process_dense(
      state_type& x,
      const std::vector<Time>& times,
      Time dt,
      Obs& obs,
      Checker& checker,
      double root_tol,
      size_t max_trigger)
  {
    using boost::numeric::odeint::detail::less_eq_with_sign;

    size_t steps = 0;
    auto it = times.begin();
    auto end = times.end();

    // --- Handle events at initial time ---
    if (apply_fixed_events_at_time(x, *it, m_fixed))
      ;  // State modified, continue with modified state

    // Initialize dense output stepper
    m_st.initialize(x, *it, dt);

    // Output initial state
    obs(x, *it);
    ++it;
    if (it == end) return 0;

    // --- Initialize root tracking with actual function values ---
    std::vector<double> last_val(m_root.size());
    std::vector<State>  last_state(m_root.size(), x);
    std::vector<Time>   last_time(m_root.size(), times.front());
    std::vector<size_t> fired(m_root.size(), 0);

    for (size_t i = 0; i < m_root.size(); ++i) {
      last_val[i] = scalar_value(m_root[i].func(x, times.front()));
    }

    // --- Main integration loop ---
    while (it != end)
    {
      // Take one integration step
      m_st.do_step(m_sys);
      ++steps;

      checker();
      checker.reset();

      // Get valid interpolation interval [t_start, t_end]
      Time t_start = m_st.previous_time();
      Time t_end = m_st.current_time();
      dt = m_st.current_time_step();

      // --- Check for roots crossing between intervals ---
      // This handles the case where a root occurs between the last
      // evaluated point and the start of the new interpolation interval
      State x_at_start = x;
      m_st.calc_state(t_start, x_at_start);

      for (size_t i = 0; i < m_root.size(); ++i)
      {
        double f_at_start = scalar_value(m_root[i].func(x_at_start, t_start));

        if (fired[i] < max_trigger &&
            !std::isnan(last_val[i]) &&
            last_val[i] * f_at_start < 0.0 &&
            scalar_value(last_time[i]) < scalar_value(t_start))
        {
          // Root crossed between intervals - trigger at interval start
          // (we cannot interpolate before t_start)
          Time t_root = t_start;
          State x_root = x_at_start;

          obs(x_root, t_root);

          apply_event(x_root, m_root[i].state_index,
                      m_root[i].value, m_root[i].method);
          fired[i]++;

          obs(x_root, t_root + Time(1e-15));

          // Reinitialize stepper from event state
          x = x_root;
          m_st.initialize(x, t_root, dt);
          m_st.do_step(m_sys);
          ++steps;
          checker();
          checker.reset();

          // Update interval bounds
          t_start = m_st.previous_time();
          t_end = m_st.current_time();
          dt = m_st.current_time_step();

          // Reset tracking to new interval start
          m_st.calc_state(t_start, x_at_start);
          for (size_t j = 0; j < m_root.size(); ++j) {
            last_val[j] = scalar_value(m_root[j].func(x_at_start, t_start));
            last_state[j] = x_at_start;
            last_time[j] = t_start;
          }
        }
      }

      // --- Ensure tracking is within current interval ---
      // This is necessary because last_time may be from a previous
      // interval that we can no longer interpolate in
      for (size_t i = 0; i < m_root.size(); ++i) {
        if (scalar_value(last_time[i]) < scalar_value(t_start)) {
          m_st.calc_state(t_start, x_at_start);
          last_val[i] = scalar_value(m_root[i].func(x_at_start, t_start));
          last_state[i] = x_at_start;
          last_time[i] = t_start;
        }
      }

      // --- Process all output times within this interval ---
      while (it != end && less_eq_with_sign(*it, t_end, dt))
      {
        Time t_eval = *it;

        // Skip times before current interval (shouldn't happen normally)
        if (scalar_value(t_eval) < scalar_value(t_start)) {
          ++it;
          continue;
        }

        // Interpolate state at output time
        m_st.calc_state(t_eval, x);

        // --- Check for fixed events ---
        if (apply_fixed_events_at_time(x, t_eval, m_fixed))
        {
          obs(x, t_eval);

          // Reinitialize from event state
          m_st.initialize(x, t_eval, dt);
          m_st.do_step(m_sys);
          ++steps;
          checker();
          checker.reset();

          t_start = m_st.previous_time();
          t_end = m_st.current_time();
          dt = m_st.current_time_step();

          // Reset all root tracking
          State x_new_start = x;
          m_st.calc_state(t_start, x_new_start);
          for (size_t j = 0; j < m_root.size(); ++j) {
            last_val[j] = scalar_value(m_root[j].func(x_new_start, t_start));
            last_state[j] = x_new_start;
            last_time[j] = t_start;
            fired[j] = 0;  // Reset fire counts after fixed event
          }

          ++it;
          continue;
        }

        // --- Check for root events ---
        bool root_fired = false;
        for (size_t i = 0; i < m_root.size(); ++i)
        {
          double f_now = scalar_value(m_root[i].func(x, t_eval));

          if (fired[i] < max_trigger &&
              !std::isnan(last_val[i]) &&
              last_val[i] * f_now < 0.0)
          {
            // Localize root via bisection using interpolation
            // last_time[i] is guaranteed >= t_start at this point
            Time t_root = t_eval;
            State x_root = x;
            localize_root_dense(i, last_state[i], last_time[i],
                                x_root, t_root, last_val[i], f_now, root_tol);

            obs(x_root, t_root);

            apply_event(x_root, m_root[i].state_index,
                        m_root[i].value, m_root[i].method);
            fired[i]++;

            obs(x_root, t_root + Time(1e-15));

            // Reinitialize stepper
            x = x_root;
            m_st.initialize(x, t_root, dt);
            m_st.do_step(m_sys);
            ++steps;
            checker();
            checker.reset();

            t_start = m_st.previous_time();
            t_end = m_st.current_time();
            dt = m_st.current_time_step();

            // Reset root tracking (preserve fired counts)
            State x_new_start = x;
            m_st.calc_state(t_start, x_new_start);
            for (size_t j = 0; j < m_root.size(); ++j) {
              last_val[j] = scalar_value(m_root[j].func(x_new_start, t_start));
              last_state[j] = x_new_start;
              last_time[j] = t_start;
            }

            root_fired = true;
            break;  // Re-evaluate current output time in new interval
          }
          else {
            // Update tracking
            last_val[i] = f_now;
            last_state[i] = x;
            last_time[i] = t_eval;
          }
        }

        if (root_fired) {
          continue;  // Don't advance iterator - reprocess in new interval
        }

        // No event - output interpolated state
        obs(x, t_eval);
        ++it;
      }
    }

    return steps;
  }

private:

  //--------------------------------------------------------------------------
  // Helper Methods
  //--------------------------------------------------------------------------

  /**
   * @brief Reset root tracking state after a fixed event
   *
   * Clears all root tracking data to ensure clean detection of future
   * root crossings. Called after fixed events which may cause discontinuities.
   *
   * @param fired Fire counts (reset to 0)
   * @param last_val Last function values (reset to NaN)
   * @param last_state Last states (reset to current state)
   * @param last_time Last times (reset to current time)
   * @param x Current state
   * @param t Current time
   */
  void reset_memory(
      std::vector<size_t>& fired,
      std::vector<double>& last_val,
      std::vector<State>&  last_state,
      std::vector<Time>&   last_time,
      const State& x, Time t)
  {
    std::fill(last_val.begin(), last_val.end(),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(fired.begin(), fired.end(), 0);

    for (size_t i = 0; i < last_state.size(); ++i) {
      last_state[i] = x;
      last_time[i]  = t;
    }
  }

  /**
   * @brief Localize a root using controlled stepping (re-integration)
   *
   * Performs bisection to find the precise root location. At each bisection
   * step, re-integrates from the lower bound to the midpoint to obtain
   * the state at the midpoint.
   *
   * @tparam Checker Step checker type
   *
   * @param idx Index of the root event being localized
   * @param xa State at lower bound (modified during bisection)
   * @param ta Time at lower bound (modified during bisection)
   * @param xb State at upper bound (output: state at root)
   * @param tb Time at upper bound (output: time at root)
   * @param fa Function value at lower bound
   * @param fb Function value at upper bound
   * @param tol Tolerance (stop when |tb - ta| < tol)
   * @param checker Step checker for sub-integration
   *
   * @pre fa and fb must have opposite signs (fa * fb < 0)
   * @post tb contains the root time to within tol, xb contains state at root
   */
  template<class Checker>
  void localize_root_controlled(
      size_t idx,
      State xa, Time ta,
      State& xb, Time& tb,
      double fa, double fb,
      double tol,
      Checker& checker)
  {
    while (std::abs(scalar_value(tb - ta)) > tol)
    {
      Time tm = (ta + tb) / 2.0;

      // Re-integrate from ta to tm to get state at midpoint
      State xm = xa;
      Time t_tmp = ta;
      Time dt_tmp = (tm - ta) / 10.0;

      while (detail::less_with_sign(t_tmp, tm, dt_tmp))
      {
        Time step_dt = min_abs(dt_tmp, tm - t_tmp);
        auto result = m_st.try_step(m_sys, xm, t_tmp, step_dt);

        if (result == success) {
          checker();
          checker.reset();
        } else {
          checker();
        }
      }

      double fm = scalar_value(m_root[idx].func(xm, tm));

      // Bisection update
      if (fa * fm <= 0.0) {
        tb = tm; xb = xm; fb = fm;
      } else {
        ta = tm; xa = xm; fa = fm;
      }
    }
  }

  /**
   * @brief Localize a root using dense output (interpolation)
   *
   * Performs bisection to find the precise root location using the
   * stepper's interpolation capability. Much more efficient than
   * re-integration when dense output is available.
   *
   * @param idx Index of the root event being localized
   * @param xa State at lower bound (used for sizing only)
   * @param ta Time at lower bound (modified during bisection)
   * @param xb State at upper bound (output: state at root)
   * @param tb Time at upper bound (output: time at root)
   * @param fa Function value at lower bound
   * @param fb Function value at upper bound
   * @param tol Tolerance (stop when |tb - ta| < tol)
   *
   * @pre ta and tb must be within the current interpolation interval
   *      [previous_time(), current_time()]
   * @pre fa and fb must have opposite signs (fa * fb < 0)
   * @post tb contains the root time to within tol, xb contains state at root
   */
  void localize_root_dense(
      size_t idx,
      State xa, Time ta,
      State& xb, Time& tb,
      double fa, double fb,
      double tol)
  {
    while (std::abs(scalar_value(tb - ta)) > tol)
    {
      Time tm = (ta + tb) / 2.0;

      // Use interpolation to get state at midpoint (very efficient!)
      State xm = xa;  // Copy for correct sizing
      m_st.calc_state(tm, xm);

      double fm = scalar_value(m_root[idx].func(xm, tm));

      // Bisection update
      if (fa * fm <= 0.0) {
        tb = tm; xb = xm; fb = fm;
      } else {
        ta = tm; xa = xm; fa = fm;
      }
    }
  }

private:
  Stepper& m_st;   ///< Reference to the ODE stepper
  System&  m_sys;  ///< Reference to the ODE system

  /// Fixed-time events (sorted by time via merge_user_and_event_times)
  const std::vector<FixedEvent<typename State::value_type>>& m_fixed;

  /// Root-finding events
  const std::vector<RootEvent<State, Time>>& m_root;
};

//==============================================================================
// Section 7: Public API - Controlled Stepper Integration
//==============================================================================

/**
 * @brief Integrate ODE at specified times with event handling (controlled stepper)
 *
 * Main entry point for event-aware integration using controlled (adaptive)
 * steppers. Combines user output times with fixed event times, then integrates
 * while monitoring for both fixed-time and root-finding events.
 *
 * @tparam Stepper Controlled stepper type (must provide try_step interface)
 * @tparam System ODE system type
 * @tparam State State vector type
 * @tparam TimeIterator Iterator type for time sequence
 * @tparam Time Time type
 * @tparam Observer Observer callable type
 *
 * @param stepper The ODE stepper instance
 * @param system The ODE system (callable: void(x, dxdt, t))
 * @param x Initial state (modified in place to final state)
 * @param t_begin Iterator to first output time
 * @param t_end Iterator past last output time
 * @param dt Initial step size
 * @param obs Observer called at each output time and event
 * @param fixed Vector of fixed-time events
 * @param root Vector of root-finding events
 * @param checker Step checker for integration monitoring
 * @param root_tol Root localization tolerance (default: 1e-8)
 * @param max_trigger_root Maximum triggers per root event (default: 1)
 * @param tag Stepper category tag (default: controlled_stepper_tag)
 *
 * @return Number of integration steps taken
 */
template<class Stepper, class System, class State,
         class TimeIterator, class Time, class Observer>
size_t integrate_times(
    Stepper stepper,
    System system,
    State& x,
    TimeIterator t_begin,
    TimeIterator t_end,
    Time dt,
    Observer obs,
    const std::vector<FixedEvent<typename State::value_type>>& fixed,
    const std::vector<RootEvent<State, Time>>& root,
    StepChecker& checker,
    double root_tol = 1e-8,
    size_t max_trigger_root = 1,
    controlled_stepper_tag = controlled_stepper_tag())
{
  auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);

  EventEngine<Stepper, System, State, Time> eng(stepper, system, fixed, root);

  return eng.process_controlled(x, times, dt, obs, checker,
                                root_tol, max_trigger_root);
}

//==============================================================================
// Section 8: Public API - Dense Output Stepper Integration
//==============================================================================

/**
 * @brief Integrate ODE at specified times with event handling (dense output)
 *
 * Main entry point for event-aware integration using dense-output steppers.
 * Leverages interpolation for efficient output and root localization.
 *
 * @tparam Stepper Dense output stepper type (must provide initialize, do_step,
 *                 calc_state, previous_time, current_time interfaces)
 * @tparam System ODE system type
 * @tparam State State vector type
 * @tparam TimeIterator Iterator type for time sequence
 * @tparam Time Time type
 * @tparam Observer Observer callable type
 *
 * @param stepper The dense output stepper instance
 * @param system The ODE system (callable: void(x, dxdt, t))
 * @param x Initial state (modified in place to final state)
 * @param t_begin Iterator to first output time
 * @param t_end Iterator past last output time
 * @param dt Initial step size
 * @param obs Observer called at each output time and event
 * @param fixed Vector of fixed-time events
 * @param root Vector of root-finding events
 * @param checker Step checker for integration monitoring
 * @param root_tol Root localization tolerance (default: 1e-8)
 * @param max_trigger_root Maximum triggers per root event (default: 1)
 * @param tag Stepper category tag (default: dense_output_stepper_tag)
 *
 * @return Number of integration steps taken
 *
 * @note This version is preferred when:
 *       - Output times are dense relative to integration steps
 *       - Root events are expected and need precise localization
 *       - The ODE system is expensive to evaluate
 */
template<class Stepper, class System, class State,
         class TimeIterator, class Time, class Observer>
size_t integrate_times_dense(
    Stepper stepper,
    System system,
    State& x,
    TimeIterator t_begin,
    TimeIterator t_end,
    Time dt,
    Observer obs,
    const std::vector<FixedEvent<typename State::value_type>>& fixed,
    const std::vector<RootEvent<State, Time>>& root,
    StepChecker& checker,
    double root_tol = 1e-8,
    size_t max_trigger_root = 1,
    dense_output_stepper_tag = dense_output_stepper_tag())
{
  auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);

  EventEngine<Stepper, System, State, Time> eng(stepper, system, fixed, root);

  return eng.process_dense(x, times, dt, obs, checker,
                           root_tol, max_trigger_root);
}

} // namespace detail

//==============================================================================
// Public Namespace Exports
//==============================================================================

using detail::FixedEvent;
using detail::RootEvent;
using detail::EventMethod;
using detail::integrate_times;
using detail::integrate_times_dense;

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
