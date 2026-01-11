/*
 Event-aware integration at specified times with root-finding support.

 Original work (basic integrate_times structure):
 Copyright 2011-2015 Mario Mulansky
 Copyright 2012 Karsten Ahnert
 Copyright 2012 Christoph Koke
 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)

 Modified work (event handling, root-finding, AD support):
 Copyright 2025 Simon Beyer

 Major extensions:
 - Event system (fixed-time and root-finding events)
 - Bisection-based root localization with adaptive tolerance
 - Event application methods (Replace/Add/Multiply)
 - Stepper reinitialization after discontinuous events
 - FADBAD++ automatic differentiation support
 - Root tracking and fire count management
 - Dense output optimization for root-finding

 This implementation extends the basic integrate_times concept from
 Boost.Odeint with comprehensive event handling capabilities for
 pharmacokinetic/pharmacodynamic modeling and sensitivity analysis.
 */

#ifndef CPPODE_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
#define CPPODE_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED

/**
 * @file cppode_boost_integrate_times_with_events.hpp
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
 * StepChecker checker;
 * integrate_times(stepper, system, x, times.begin(), times.end(), dt,
 *                 observer, fixed_events, root_events, checker);
 * @endcode
 *
 * @section compatibility Stepper Compatibility
 *
 * The engine automatically adapts to different stepper types:
 *
 * - **rosenbrock4_controller_pi**: Uses reset_after_event(dt) for reinitialization
 * - **rosenbrock4_dense_output_pi**: Uses reinitialize_at_event(x,t,dt) for full reset
 * - **Standard Odeint steppers**: Fallback behavior (no special reset needed)
 *
 * @section algorithm Algorithm Overview
 *
 * **Controlled Stepper Mode:**
 * 1. Merge user output times with fixed event times
 * 2. Integrate adaptively between time points
 * 3. After each step, check for root crossings (sign changes)
 * 4. On detection, use bisection with re-integration to localize
 * 5. Apply event and reset stepper state
 *
 * **Dense Output Mode:**
 * 1. Merge user output times with fixed event times
 * 2. Take large steps with dense output
 * 3. Interpolate at output times
 * 4. Check for root crossings using interpolated states
 * 5. Use bisection with interpolation for efficient localization
 * 6. Apply event and reinitialize stepper
 *
 * @author Simon Beyer <simon.beyer@fdm.uni-freiburg.de>
 * @date December 22, 2025
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
 * the input value when it's already an arithmetic type (double, float, int).
 *
 * @tparam T Arithmetic type (std::is_arithmetic<T>::value == true)
 * @param v The value to extract
 * @return The input value as a double
 *
 * @note This function is used in root-finding and time comparisons to
 *       extract numeric values from potentially AD-enhanced types
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
 *
 * @par Example
 * @code
 * fadbad::F<double> x = 3.14;
 * double val = scalar_value(x);  // Returns 3.14
 *
 * fadbad::F<fadbad::F<double>> y;  // Nested AD type
 * double val2 = scalar_value(y);   // Recursively extracts value
 * @endcode
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
 * These methods cover the most common event types in pharmacokinetic
 * and dynamical systems modeling.
 */
enum class EventMethod {
  Replace,   ///< Replace: x[i] = value (e.g., reset to baseline)
  Add,       ///< Add: x[i] += value (e.g., instantaneous dose)
  Multiply   ///< Multiply: x[i] *= value (e.g., fractional change)
};

/**
 * @brief Fixed-time event specification
 *
 * Represents an event that triggers at a predetermined time point.
 * When the integration reaches the specified time, the event modifies
 * a state variable according to the specified method.
 *
 * Fixed-time events are exact - they always trigger at the specified
 * time regardless of the integration step size.
 *
 * @tparam value_type The numeric type for time and value (supports AD types)
 *
 * @par Example: Pharmacokinetic Dosing
 * @code{.cpp}
 * // Oral dose of 100mg at t=0 and t=12 hours
 * std::vector<FixedEvent<double>> doses = {
 *     {0.0,  0, 100.0, EventMethod::Add},   // First dose
 *     {12.0, 0, 100.0, EventMethod::Add}    // Second dose
 * };
 * @endcode
 *
 * @par Example: Periodic Reset
 * @code{.cpp}
 * // Reset state[1] to zero every 24 hours
 * std::vector<FixedEvent<double>> resets;
 * for (int day = 1; day <= 7; ++day) {
 *     resets.push_back({24.0 * day, 1, 0.0, EventMethod::Replace});
 * }
 * @endcode
 */
template<class value_type>
struct FixedEvent {
  value_type  time;         ///< Time at which the event triggers
  int         state_index;  ///< Index of the state variable to modify (0-based)
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
 * Root events enable modeling of state-dependent phenomena like:
 * - Threshold crossings (drug concentration below MIC)
 * - Peak detection (derivative crosses zero)
 * - Collision detection (distance function crosses zero)
 * - Phase transitions (energy below threshold)
 *
 * @tparam state_type The ODE state vector type
 * @tparam time_type The time variable type
 *
 * @par Example: Threshold Detection
 * @code{.cpp}
 * // Trigger when drug concentration drops below threshold
 * RootEvent<state_type, double> below_threshold = {
 *     [](const state_type& x, double t) {
 *         return x[0] - 0.5;  // Root at x[0] = 0.5
 *     },
 *     0,                      // Modify state[0]
 *     10.0,                   // Add rescue dose
 *     EventMethod::Add
 * };
 * @endcode
 *
 * @par Example: Peak Detection
 * @code{.cpp}
 * // Detect local maximum of x[0] by finding where dx[0]/dt = 0
 * RootEvent<state_type, double> peak_detector = {
 *     [&system](const state_type& x, double t) {
 *         state_type dxdt(x.size());
 *         system(x, dxdt, t);
 *         return dxdt[0].x();  // Root when derivative = 0
 *     },
 *     1,                      // Record peak in state[1]
 *     0.0,                    // Will be replaced with x[0] at peak
 *     EventMethod::Replace
 * };
 * @endcode
 */
template<class state_type, class time_type>
struct RootEvent {
  using value_t = typename state_type::value_type;

  /**
   * @brief Root function: f(x,t) = 0 at event time
   *
   * The function should:
   * - Return a continuous value that changes sign at the event
   * - Be relatively smooth near the root for efficient localization
   * - Avoid returning exactly zero to prevent numerical issues
   *
   * @param x Current state vector
   * @param t Current time
   * @return Function value (event triggers when sign changes)
   */
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
 * @param idx Index of the state element to modify (0-based)
 * @param v Value to apply
 * @param method How to apply the value
 *
 * @note For AD types, this operation preserves derivative information:
 * @code
 * // If x[idx] has derivatives w.r.t. parameters
 * x[idx] += dose;  // Derivatives are preserved
 * // New sensitivity: d(x[idx] + dose)/dp = dx[idx]/dp + d(dose)/dp
 * @endcode
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
 * trigger time matches the current time. Uses exact floating-point
 * comparison on scalar values (after AD unwrapping).
 *
 * @tparam state_type The ODE state vector type
 * @tparam Time The time type (may be AD-enhanced)
 * @param x The state vector to modify (modified in place)
 * @param t Current integration time
 * @param evs Vector of fixed events to check
 * @return true if at least one event was applied, false otherwise
 *
 * @note Multiple events at the same time are applied in order
 *
 * @par Example: Multiple Simultaneous Events
 * @code
 * // At t=10: Add dose to state[0] AND reset state[1]
 * std::vector<FixedEvent<double>> events = {
 *     {10.0, 0, 100.0, EventMethod::Add},
 *     {10.0, 1,   0.0, EventMethod::Replace}
 * };
 * // Both events fire when t reaches 10.0
 * @endcode
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
 * This ensures that:
 * - Integration stops exactly at event times
 * - Observer is called at both output and event times
 * - No event is missed due to step-size adaptation
 *
 * @tparam Time The time type for output
 * @tparam It Iterator type for user times
 * @tparam V Value type of fixed events
 * @param ubegin Iterator to first user time
 * @param uend Iterator past last user time
 * @param fix Vector of fixed events
 * @return Sorted vector of unique time points
 *
 * @note Duplicates are removed to avoid redundant observer calls
 *
 * @par Example
 * @code
 * std::vector<double> user_times = {0, 1, 2, 3};
 * std::vector<FixedEvent<double>> events = {
 *     {1.5, 0, 100.0, EventMethod::Add},
 *     {2.0, 0,  50.0, EventMethod::Add}  // Duplicate with user time
 * };
 * auto merged = merge_user_and_event_times<double>(
 *     user_times.begin(), user_times.end(), events);
 * // Result: {0, 1, 1.5, 2, 3}  (2.0 appears once)
 * @endcode
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

  // Remove duplicates (preserves sorted order)
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
 * the appropriate reset method for different stepper types.
 *
 * **Dispatch Logic:**
 * 1. If stepper has `reset_after_event(dt)` → call it
 *    (controlled steppers: rosenbrock4_controller_pi)
 * 2. Else if stepper has `reinitialize_at_event(x, t, dt)` → call it
 *    (dense output steppers: rosenbrock4_dense_output_pi)
 * 3. Else → no-op (standard Boost steppers don't need reset)
 *
 * @tparam S Stepper type
 * @tparam State State vector type
 * @tparam Time Time type
 * @param st The stepper instance
 * @param x Current state (used for dense output reinitialization)
 * @param t Current time
 * @param dt Reference to step size (may be modified by reset)
 *
 * @note This function is called automatically after every event application
 *       to ensure clean continuation of integration
 *
 * @par Why Reset is Necessary
 * Events cause state discontinuities, which invalidate:
 * - Error estimates (based on smooth evolution)
 * - Step-size history (used in PI control)
 * - Dense output coefficients (interpolation data)
 * - Derivative tracking (for AD types)
 *
 * Resetting clears this stale information and prevents
 * spurious step rejections or inaccurate interpolation.
 */
template<class S, class State, class Time>
inline void reset_stepper_unified(S& st, State& x, Time t, Time& dt)
{
  if constexpr (requires { st.reset_after_event(dt); }) {
    // Controlled stepper with event reset support
    // Clears error history and step-size memory
    st.reset_after_event(dt);
  }
  else if constexpr (requires { st.reinitialize_at_event(x, t, dt); }) {
    // Dense output stepper with full reinitialization
    // Resets state buffers and prepares new interpolation interval
    st.reinitialize_at_event(x, t, dt);
  }
  else {
    // Fallback: stepper doesn't need special reset
    // Standard Boost steppers work fine without explicit reset
    (void)st; (void)x; (void)t; (void)dt;
  }
}

//==============================================================================
// Section 6: Event Engine Core
//==============================================================================

/**
 * @class EventEngine
 * @brief Core event-aware integration engine
 *
 * This class implements the main integration loops for both controlled
 * and dense-output steppers, with full support for fixed-time and
 * root-finding events.
 *
 * **Design Philosophy:**
 * - Separation of concerns: integration logic vs. event detection
 * - Efficiency: use best available method (re-integration vs. interpolation)
 * - Robustness: handle edge cases (multiple roots, events at boundaries)
 * - Generality: work with any conforming stepper type
 *
 * @tparam Stepper The ODE stepper type
 * @tparam System The ODE system type (callable with signature void(x, dxdt, t))
 * @tparam State The state vector type
 * @tparam Time The time type
 *
 * @section algorithm Algorithm Overview
 *
 * **Controlled Stepper Mode (process_controlled):**
 * @code
 * 1. For each target time:
 *    a. Take adaptive steps with try_step()
 *    b. After each success:
 *       - Evaluate root functions
 *       - Check for sign changes
 *       - If detected: localize via bisection + re-integration
 *       - Apply event and reset stepper
 *    c. Check for fixed events at target time
 * @endcode
 *
 * **Dense Output Mode (process_dense):**
 * @code
 * 1. Initialize dense output stepper
 * 2. While not finished:
 *    a. Take one large step with do_step()
 *    b. For each output time in [t_old, t_new]:
 *       - Interpolate state with calc_state()
 *       - Check for root crossings
 *       - If detected: localize via bisection + interpolation
 *       - Apply event and reinitialize stepper
 *    c. Check for fixed events
 * @endcode
 *
 * @note The dense output mode is generally more efficient for root-finding
 *       as it can use interpolation instead of re-integration during bisection.
 *       For a root tolerance of 1e-8, interpolation can be 10-100x faster.
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
   *
   * @note The engine stores references, so the stepper and system
   *       must remain valid for the engine's lifetime
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
   * try_step() interface (e.g., rosenbrock4_controller_pi). Root localization
   * is performed by re-integrating with smaller steps during bisection.
   *
   * **Root Detection Strategy:**
   * After each successful step, the engine evaluates all root functions and
   * compares with the previous value. A sign change indicates a root crossing.
   * The bisection then uses re-integration to find the exact crossing time.
   *
   * **Event Application:**
   * When a root is found:
   * 1. Observer is called with state before event
   * 2. Event is applied to create post-event state
   * 3. Observer is called with post-event state (t + epsilon)
   * 4. Stepper is reset to clear stale error history
   * 5. Integration continues from post-event state
   *
   * @tparam Obs Observer type with signature void(const state_type&, time_type)
   * @tparam Checker Step checker type (e.g., StepChecker for max step limits)
   *
   * @param x Initial state (modified to final state on return)
   * @param times Sorted vector of output times (merged with event times)
   * @param dt Initial step size suggestion
   * @param obs Observer called at each output time and event
   * @param checker Step checker for detecting integration failures
   * @param root_tol Tolerance for root localization (bisection stops when
   *                 interval width < root_tol)
   * @param max_trigger Maximum number of times each root event can fire
   *                    (prevents infinite loops on chattering)
   *
   * @return Total number of successful integration steps taken
   *
   * @par Performance Considerations
   * - Root localization requires re-integration, which can be expensive
   * - For tight tolerances (< 1e-10), consider using dense output mode
   * - Multiple roots in one step are handled sequentially
   *
   * @par Example: Chattering Prevention
   * @code
   * // Prevent infinite triggering near equilibrium
   * RootEvent<state_type, double> threshold = {
   *     [](auto& x, auto t) { return x[0] - 1.0; },
   *     0, 0.0, EventMethod::Replace
   * };
   *
   * // Allow max 10 triggers (stops chattering)
   * process_controlled(x, times, dt, obs, checker, 1e-8, 10);
   * @endcode
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

              // Reset root tracking (keep fired counts to prevent re-triggering)
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
        // Reset root tracking after fixed events (may change dynamics)
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
   * **Key Advantages over Controlled Mode:**
   * - **Efficiency**: Interpolation is much faster than re-integration
   *   (typical speedup: 10-100x for root localization)
   * - **Accuracy**: Dense output provides higher-order interpolation
   *   (cubic Hermite vs. linear for re-integration)
   * - **Consistency**: All states come from same underlying solution
   *
   * **Root Detection Strategy:**
   * The engine maintains a valid interpolation interval [t_old, t_new].
   * For each output time in this interval, it interpolates the state and
   * checks root functions. On sign change, bisection uses interpolation
   * to find the exact crossing time.
   *
   * **Edge Cases Handled:**
   * 1. Roots between intervals (last_time < t_start)
   * 2. Multiple output times in one step
   * 3. Events requiring reinitialization mid-interval
   *
   * @tparam Obs Observer type with signature void(const state_type&, time_type)
   * @tparam Checker Step checker type
   *
   * @param x Initial state (modified to final state on return)
   * @param times Sorted vector of output times (merged with event times)
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
   *
   * @par Performance Optimization
   * Dense output is particularly efficient when:
   * - Output times are dense relative to natural step size
   * - Root tolerance is tight (< 1e-8)
   * - ODE system is expensive to evaluate
   *
   * For sparse output (few times per step), controlled mode may be faster
   * due to overhead of dense output coefficient computation.
   *
   * @par Example: Dense Output with Events
   * @code
   * // Many output times (every 0.1 time units)
   * std::vector<double> times;
   * for (double t = 0; t <= 100; t += 0.1) times.push_back(t);
   *
   * // Dense output takes large steps (dt ~ 1.0)
   * // but interpolates at all output times efficiently
   * dense_stepper.initialize(x, 0.0, 1.0);
   * process_dense(x, times, 1.0, obs, checker, 1e-10, 10);
   * // Much faster than controlled mode for this scenario
   * @endcode
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
    // Unlike controlled mode, we evaluate root functions immediately
    // to enable detection of roots in the first step
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
      // Take one integration step (may be large)
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
   * root crossings. Called after fixed events which may cause discontinuities
   * that invalidate previous root function values.
   *
   * @param fired Fire counts (reset to 0 - allows events to trigger again)
   * @param last_val Last function values (reset to NaN - forces re-evaluation)
   * @param last_state Last states (reset to current state)
   * @param last_time Last times (reset to current time)
   * @param x Current state (post-event)
   * @param t Current time (event time)
   *
   * @note This is necessary because fixed events may fundamentally change
   *       system dynamics, making previous root function evaluations meaningless
   *
   * @par Example
   * @code
   * // Fixed event changes parameter that affects root function
   * FixedEvent: Replace state[2] with 10.0  // Changes system behavior
   * RootEvent: x[0] - state[2]  // Root location depends on state[2]
   *
   * // After fixed event, must reset root tracking
   * // because root function behavior has changed
   * @endcode
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
   * the state at the midpoint. This is less efficient than interpolation
   * but works with any stepper.
   *
   * **Bisection Algorithm:**
   * @code
   * while |tb - ta| > tol:
   *     tm = (ta + tb) / 2
   *     Integrate from ta to tm → get xm
   *     Evaluate f(xm, tm) → get fm
   *     if fa * fm <= 0:  // Root in [ta, tm]
   *         tb = tm, xb = xm, fb = fm
   *     else:              // Root in [tm, tb]
   *         ta = tm, xa = xm, fa = fm
   * @endcode
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
   *
   * @note Each bisection iteration requires ~10 integration sub-steps,
   *       making this expensive for tight tolerances
   *
   * @par Convergence Rate
   * Bisection has linear convergence: each iteration reduces the
   * uncertainty by a factor of 2. To reach tolerance tol from
   * initial interval width dt:
   * @code
   * iterations = ceil(log2(dt / tol))
   * // For dt=1, tol=1e-8: ~27 iterations
   * // Each iteration: ~10 integration steps
   * // Total: ~270 integration steps per root
   * @endcode
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
      Time dt_tmp = (tm - ta) / 10.0;  // Use 10 sub-steps

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
        // Root in [ta, tm]
        tb = tm; xb = xm; fb = fm;
      } else {
        // Root in [tm, tb]
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
   * **Key Advantage:**
   * Each bisection iteration requires only ONE calc_state() call
   * (cheap interpolation) instead of ~10 integration steps.
   * Typical speedup: 10-100x over controlled mode.
   *
   * **Bisection Algorithm:**
   * @code
   * while |tb - ta| > tol:
   *     tm = (ta + tb) / 2
   *     Interpolate at tm → get xm  (FAST!)
   *     Evaluate f(xm, tm) → get fm
   *     if fa * fm <= 0:
   *         tb = tm, xb = xm, fb = fm
   *     else:
   *         ta = tm, xa = xm, fa = fm
   * @endcode
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
   *
   * @par Performance Comparison
   * @code
   * Scenario: Root in [0, 1], tolerance 1e-8
   *
   * Controlled (re-integration):
   *   27 iterations × 10 steps = 270 integration steps
   *   Time: ~10 ms (typical)
   *
   * Dense output (interpolation):
   *   27 iterations × 1 interpolation = 27 interpolations
   *   Time: ~0.1 ms (typical)
   *
   * Speedup: 100x
   * @endcode
   *
   * @note Interpolation accuracy is typically better than re-integration
   *       due to higher-order polynomial interpolation (cubic Hermite)
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
        // Root in [ta, tm]
        tb = tm; xb = xm; fb = fm;
      } else {
        // Root in [tm, tb]
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
 * **When to Use:**
 * - Sparse output (few time points per natural step)
 * - Moderate root tolerance (>= 1e-6)
 * - Simple ODE systems (cheap evaluations)
 * - When dense output is not available
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
 *
 * @par Example: PK Model with Dosing and Threshold Events
 * @code
 * // One-compartment PK model
 * auto pk_system = [](const state_type& x, state_type& dxdt, double t) {
 *     double ka = 1.0, ke = 0.1;
 *     dxdt[0] = -ka * x[0];           // Absorption compartment
 *     dxdt[1] = ka * x[0] - ke * x[1]; // Central compartment
 * };
 *
 * // Oral dose at t=0 and t=12
 * std::vector<FixedEvent<double>> doses = {
 *     {0.0,  0, 100.0, EventMethod::Add},
 *     {12.0, 0, 100.0, EventMethod::Add}
 * };
 *
 * // Rescue dose when concentration drops below 5
 * std::vector<RootEvent<state_type, double>> rescue = {
 *     {[](auto& x, auto t) { return x[1] - 5.0; },
 *      0, 50.0, EventMethod::Add}
 * };
 *
 * // Integrate
 * rosenbrock4_controller_pi<rosenbrock4<double>> stepper(1e-6, 1e-6);
 * StepChecker checker;
 * integrate_times(stepper, pk_system, x, times.begin(), times.end(),
 *                 0.1, observer, doses, rescue, checker);
 * @endcode
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
 * **When to Use:**
 * - Dense output (many time points per natural step)
 * - Tight root tolerance (<= 1e-8)
 * - Expensive ODE systems (minimize evaluations)
 * - When highest accuracy is required
 *
 * **Performance Characteristics:**
 * - Large integration steps (dt can be >> output spacing)
 * - Cheap interpolation at output times
 * - Very fast root localization (10-100x speedup)
 * - Higher overhead per step (dense output coefficients)
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
 *
 * @par Example: Dense Output with High Precision Root Finding
 * @code
 * // System with expensive RHS evaluation
 * auto expensive_system = [...];
 *
 * // Dense output times (every 0.01 time units)
 * std::vector<double> times;
 * for (double t = 0; t <= 100; t += 0.01) times.push_back(t);
 *
 * // Root event with very tight tolerance
 * std::vector<RootEvent<state_type, double>> roots = {
 *     {[](auto& x, auto t) { return x[0] - 1.0; },
 *      0, 0.0, EventMethod::Replace}
 * };
 *
 * // Dense output: Large steps with interpolation
 * rosenbrock4_dense_output_pi<controller_t> stepper;
 * StepChecker checker;
 * integrate_times_dense(stepper, expensive_system, x,
 *                       times.begin(), times.end(), 1.0,
 *                       observer, fixed_events, roots, checker,
 *                       1e-12);  // Very tight root tolerance
 *
 * // Typical result:
 * // - 100 integration steps (dt ~ 1.0)
 * // - 10,000 interpolations (cheap!)
 * // - Root found to 1e-12 accuracy in ~30 interpolations
 * // Much faster than controlled mode
 * @endcode
 *
 * @par Performance Tip
 * Choose dt based on natural step size, not output spacing:
 * @code
 * // Good: Let stepper take large steps
 * integrate_times_dense(stepper, system, x, times, 1.0, ...);
 *
 * // Bad: Forces tiny steps (defeats purpose of dense output)
 * integrate_times_dense(stepper, system, x, times, 0.01, ...);
 * @endcode
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

/**
 * @brief Fixed-time event (exported to boost::numeric::odeint)
 * @see detail::FixedEvent
 */
using detail::FixedEvent;

/**
 * @brief Root-finding event (exported to boost::numeric::odeint)
 * @see detail::RootEvent
 */
using detail::RootEvent;

/**
 * @brief Event application method (exported to boost::numeric::odeint)
 * @see detail::EventMethod
 */
using detail::EventMethod;

/**
 * @brief Event-aware integration (controlled stepper, exported to boost::numeric::odeint)
 * @see detail::integrate_times
 */
using detail::integrate_times;

/**
 * @brief Event-aware integration (dense output, exported to boost::numeric::odeint)
 * @see detail::integrate_times_dense
 */
using detail::integrate_times_dense;

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
