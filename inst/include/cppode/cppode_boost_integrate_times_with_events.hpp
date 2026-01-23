/*
 Event-aware integration at specified times with root-finding support.

 Original work (basic integrate_times structure):
 Copyright (C) 2011-2015 Mario Mulansky
 Copyright (C) 2012 Karsten Ahnert
 Copyright (C) 2012 Christoph Koke
 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)

 Modified work (event handling, root-finding, AD support):
 Copyright (C) 2026 Simon Beyer

 Major extensions:
 - Event system (fixed-time and root-finding events)
 - Bisection-based root localization with tolerance
 - Event application methods (Replace/Add/Multiply)
 - Stepper reinitialization after events
 - FADBAD++ automatic differentiation support
 - Root tracking and fire count management
 - Dense output optimization for root-finding
 - **Simultaneous root events**: Multiple events with the same or similar
 root conditions are now detected and applied together efficiently.
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
 * - **Simultaneous root events**: Multiple events triggered by the same root
 *   condition are detected and applied together in a single pass
 * - **Root localization**: Bisection-based refinement to locate roots with
 *   configurable tolerance
 * - **Dual stepper support**: Works with both controlled steppers (try_step)
 *   and dense-output steppers (do_step + interpolation)
 * - **AD compatibility**: Full support for nested automatic differentiation
 *   types (fadbad::F<F<...>>) through scalar_value unwrapping
 *
 * @author Simon Beyer
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
#include <cppode/cppode_fadiff_extensions.hpp>

namespace boost {
namespace numeric {
namespace odeint {
namespace detail {

//==============================================================================
// Section 1: Scalar Value Extraction
//==============================================================================

/**
 * @brief Extract the underlying scalar value from arithmetic types
 */
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v)
{
  return static_cast<double>(v);
}

/**
 * @brief Extract the underlying scalar value from FADBAD++ AD types
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
 */
enum class EventMethod {
  Replace,   ///< Replace: x[i] = value
  Add,       ///< Add: x[i] += value
  Multiply   ///< Multiply: x[i] *= value
};

/**
 * @brief Fixed-time event specification
 */
template<class state_type, class value_type>
struct FixedEvent {
  value_type  time;
  int         state_index;
  std::function<value_type(const state_type&, const value_type&)> value_func;
  EventMethod method;
};

/**
 * @brief Root-finding event specification
 */
template<class state_type, class time_type>
struct RootEvent {
  using value_t = typename state_type::value_type;

  std::function<value_t(const state_type&, const time_type&)> func;
  int         state_index;
  std::function<value_t(const state_type&, const time_type&)> value_func;
  EventMethod method;
  bool terminal = false;
  int direction = 0;
};

//==============================================================================
// Section 3: Helper Structures for Simultaneous Events
//==============================================================================

/**
 * @brief Information about a triggered root event (for batch processing)
 *
 * This structure holds all information needed to process a triggered event,
 * allowing multiple events to be collected and processed together.
 */
struct TriggeredEvent {
  size_t index;        ///< Index in the root events vector
  double last_val;     ///< Function value before crossing
  double curr_val;     ///< Function value after crossing
};

/**
 * @brief Check if a sign change matches the specified direction
 */
inline bool direction_matches(double last_val, double curr_val, int direction)
{
  if (direction == 0) {
    return true;
  }
  else if (direction < 0) {
    return last_val > 0.0 && curr_val < 0.0;
  }
  else {
    return last_val < 0.0 && curr_val > 0.0;
  }
}

//==============================================================================
// Section 4: Steady-State Root Function Factory
//==============================================================================

/**
 * @brief Create a root function for steady-state detection
 */
template<class System, class State, class Time>
auto make_steady_state_root_func(System& sys, double tol)
{
  return [&sys, tol](const State& x, const Time& t) ->
    typename State::value_type
    {
      State dxdt(x.size());
      sys(x, dxdt, t);
      double max_rate = cppode::max_abs_all_levels_vec(dxdt);
      return typename State::value_type(max_rate - tol);
    };
}

//==============================================================================
// Section 5: Event Application
//==============================================================================

/**
 * @brief Apply an event modification to a state vector
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
 */
template<class state_type, class Time>
bool apply_fixed_events_at_time(
    state_type& x,
    const Time& t,
    const std::vector<FixedEvent<state_type, typename state_type::value_type>>& evs)
{
  const double tt = scalar_value(t);
  bool fired = false;

  for (const auto& e : evs) {
    if (scalar_value(e.time) == tt) {
      apply_event(x, e.state_index, e.value_func(x, t), e.method);
      fired = true;
    }
  }

  return fired;
}

//==============================================================================
// Section 6: Time Point Management
//==============================================================================

/**
 * @brief Merge user-requested output times with fixed event times
 */
template<class Time, class It, class state_type, class V>
std::vector<Time> merge_user_and_event_times(
    It ubegin, It uend,
    const std::vector<FixedEvent<state_type, V>>& fix)
{
  std::vector<Time> out(ubegin, uend);
  for (const auto& e : fix) {
    out.push_back(e.time);
  }

  std::sort(out.begin(), out.end(),
            [](auto& a, auto& b) {
              return scalar_value(a) < scalar_value(b);
            });

  out.erase(std::unique(out.begin(), out.end(),
                        [](auto& a, auto& b) {
                          return scalar_value(a) == scalar_value(b);
                        }),
                        out.end());

  return out;
}

//==============================================================================
// Section 7: Stepper Reset Dispatch
//==============================================================================

/**
 * @brief Unified stepper reset after discontinuous events
 */
template<class S, class State, class Time>
inline void reset_stepper_unified(S& st, State& x, Time t, Time& dt)
{
  if constexpr (requires { st.reset_after_event(dt); }) {
    st.reset_after_event(dt);
  }
  else if constexpr (requires { st.reinitialize_at_event(x, t, dt); }) {
    st.reinitialize_at_event(x, t, dt);
  }
  else {
    (void)st; (void)x; (void)t; (void)dt;
  }
}

//==============================================================================
// Section 8: Event Engine Core
//==============================================================================

/**
 * @class EventEngine
 * @brief Core event-aware integration engine with simultaneous event support
 *
 * This class implements the main integration loops for both controlled
 * and dense-output steppers, with full support for fixed-time and
 * root-finding events. **Multiple events with the same root condition
 * are now detected and applied together efficiently.**
 *
 * @section simultaneous Simultaneous Event Handling
 *
 * When multiple root events have the same (or very similar) trigger condition:
 * 1. All root functions are evaluated in a single pass
 * 2. All events with sign changes are collected
 * 3. Root localization is performed once (using the first triggered event)
 * 4. All triggered events are applied at the same root time
 * 5. A single stepper reset is performed
 *
 * This is more efficient than the naive approach of handling one event,
 * resetting, and then detecting the next event in a subsequent step.
 */
template<class Stepper, class System, class State, class Time>
class EventEngine {
public:
  using state_type = State;
  using time_type  = Time;

  EventEngine(
    Stepper& st,
    System& sys,
    const std::vector<FixedEvent<State, typename State::value_type>>& fixed,
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
   * @brief Integrate using a controlled stepper with simultaneous event support
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

    obs(x, t);
    ++it;
    if (it == end) return 0;

    // --- Initialize root tracking ---
    std::vector<double> last_val(m_root.size(),
                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<State>  last_state(m_root.size(), x);
    std::vector<Time>   last_time(m_root.size(), t);
    std::vector<size_t> fired(m_root.size(), 0);

    // Pre-allocate for triggered events (avoids allocations in hot loop)
    std::vector<TriggeredEvent> triggered;
    triggered.reserve(m_root.size());

    // Pre-allocate current values vector
    std::vector<double> curr_val(m_root.size());

    // --- Main integration loop ---
    while (it != end)
    {
      Time t_target = *it;

      while (less_with_sign(t, t_target, dt))
      {
        Time dt_step = min_abs(dt, t_target - t);
        auto result = m_st.try_step(m_sys, x, t, dt_step);

        if (result == success) {
          ++steps;
          checker();
          checker.reset();
          dt = dt_step;

          // === PHASE 1: Evaluate ALL root functions (single pass) ===
          for (size_t i = 0; i < m_root.size(); ++i) {
            curr_val[i] = scalar_value(m_root[i].func(x, t));
          }

          // === PHASE 2: Collect ALL triggered events ===
          triggered.clear();
          for (size_t i = 0; i < m_root.size(); ++i) {
            if (fired[i] < max_trigger &&
                !std::isnan(last_val[i]) &&
                last_val[i] * curr_val[i] < 0.0 &&
                direction_matches(last_val[i], curr_val[i], m_root[i].direction))
            {
              triggered.push_back({i, last_val[i], curr_val[i]});
            }
          }

          // === PHASE 3: Process all triggered events together ===
          if (!triggered.empty()) {
            // Use first triggered event for root localization
            // (all have roots in the same interval)
            size_t ref_idx = triggered[0].index;

            // Localize root once
            localize_root_controlled(
              ref_idx,
              last_state[ref_idx], last_time[ref_idx],
                                            x, t,
                                            triggered[0].last_val, triggered[0].curr_val,
                                            root_tol,
                                            checker);

            Time t_root = t;

            // Output state at root (before any events)
            obs(x, t_root);

            // Check for terminal events first
            bool has_terminal = false;
            for (const auto& te : triggered) {
              if (m_root[te.index].terminal) {
                has_terminal = true;
                break;
              }
            }

            // Apply ALL triggered events to the state
            State x_after = x;
            for (const auto& te : triggered) {
              size_t i = te.index;
              if (!m_root[i].terminal) {
                apply_event(x_after,
                            m_root[i].state_index,
                            m_root[i].value_func(x, t_root),
                            m_root[i].method);
              }
              fired[i]++;
            }

            // Output state after all events
            obs(x_after, t_root + Time(1e-15));
            x = x_after;

            // Terminal check
            if (has_terminal) {
              return steps;
            }

            // Single reset after all events
            reset_stepper_unified(m_st, x, t, dt);

            // Reset root tracking (keep fired counts)
            for (size_t j = 0; j < m_root.size(); ++j) {
              last_val[j] = std::numeric_limits<double>::quiet_NaN();
              last_state[j] = x;
              last_time[j] = t;
            }
          }
          else {
            // No events triggered - update tracking for all
            for (size_t i = 0; i < m_root.size(); ++i) {
              last_val[i] = curr_val[i];
              last_state[i] = x;
              last_time[i] = t;
            }
          }
        }
        else {
          checker();
          dt = dt_step;
        }
      }

      t = t_target;

      if (apply_fixed_events_at_time(x, t, m_fixed)) {
        obs(x, t);
        reset_stepper_unified(m_st, x, t, dt);
        // Reset root tracking after fixed events
        for (size_t j = 0; j < m_root.size(); ++j) {
          last_val[j] = std::numeric_limits<double>::quiet_NaN();
          last_state[j] = x;
          last_time[j] = t;
          fired[j] = 0;
        }
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
   * @brief Integrate using dense output with simultaneous event support
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

    if (apply_fixed_events_at_time(x, *it, m_fixed))
      ;

    m_st.initialize(x, *it, dt);
    obs(x, *it);
    ++it;
    if (it == end) return 0;

    // --- Initialize root tracking ---
    std::vector<double> last_val(m_root.size());
    std::vector<State>  last_state(m_root.size(), x);
    std::vector<Time>   last_time(m_root.size(), times.front());
    std::vector<size_t> fired(m_root.size(), 0);

    for (size_t i = 0; i < m_root.size(); ++i) {
      last_val[i] = scalar_value(m_root[i].func(x, times.front()));
    }

    // Pre-allocate for triggered events
    std::vector<TriggeredEvent> triggered;
    triggered.reserve(m_root.size());

    // Pre-allocate current values
    std::vector<double> curr_val(m_root.size());

    // --- Main integration loop ---
    while (it != end)
    {
      m_st.do_step(m_sys);
      ++steps;
      checker();
      checker.reset();

      Time t_start = m_st.previous_time();
      Time t_end = m_st.current_time();
      dt = m_st.current_time_step();

      // --- Check for roots crossing between intervals ---
      State x_at_start = x;
      m_st.calc_state(t_start, x_at_start);

      // Evaluate all root functions at interval start
      for (size_t i = 0; i < m_root.size(); ++i) {
        curr_val[i] = scalar_value(m_root[i].func(x_at_start, t_start));
      }

      // Collect all events triggered between intervals
      triggered.clear();
      for (size_t i = 0; i < m_root.size(); ++i) {
        if (fired[i] < max_trigger &&
            !std::isnan(last_val[i]) &&
            last_val[i] * curr_val[i] < 0.0 &&
            scalar_value(last_time[i]) < scalar_value(t_start) &&
            direction_matches(last_val[i], curr_val[i], m_root[i].direction))
        {
          triggered.push_back({i, last_val[i], curr_val[i]});
        }
      }

      // Process all triggered events at interval boundary
      if (!triggered.empty()) {
        Time t_root = t_start;
        State x_root = x_at_start;

        obs(x_root, t_root);

        bool has_terminal = false;
        for (const auto& te : triggered) {
          if (m_root[te.index].terminal) {
            has_terminal = true;
            break;
          }
        }

        for (const auto& te : triggered) {
          size_t i = te.index;
          if (!m_root[i].terminal) {
            apply_event(x_root, m_root[i].state_index,
                        m_root[i].value_func(x_root, t_root), m_root[i].method);
          }
          fired[i]++;
        }

        obs(x_root, t_root + Time(1e-15));

        if (has_terminal) {
          x = x_root;
          return steps;
        }

        x = x_root;
        m_st.initialize(x, t_root, dt);
        m_st.do_step(m_sys);
        ++steps;
        checker();
        checker.reset();

        t_start = m_st.previous_time();
        t_end = m_st.current_time();
        dt = m_st.current_time_step();

        m_st.calc_state(t_start, x_at_start);
        for (size_t j = 0; j < m_root.size(); ++j) {
          last_val[j] = scalar_value(m_root[j].func(x_at_start, t_start));
          last_state[j] = x_at_start;
          last_time[j] = t_start;
        }
      }

      // Ensure tracking is within current interval
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

        if (scalar_value(t_eval) < scalar_value(t_start)) {
          ++it;
          continue;
        }

        m_st.calc_state(t_eval, x);

        // Check for fixed events
        if (apply_fixed_events_at_time(x, t_eval, m_fixed))
        {
          obs(x, t_eval);

          m_st.initialize(x, t_eval, dt);
          m_st.do_step(m_sys);
          ++steps;
          checker();
          checker.reset();

          t_start = m_st.previous_time();
          t_end = m_st.current_time();
          dt = m_st.current_time_step();

          State x_new_start = x;
          m_st.calc_state(t_start, x_new_start);
          for (size_t j = 0; j < m_root.size(); ++j) {
            last_val[j] = scalar_value(m_root[j].func(x_new_start, t_start));
            last_state[j] = x_new_start;
            last_time[j] = t_start;
            fired[j] = 0;
          }

          ++it;
          continue;
        }

        // === Evaluate ALL root functions at t_eval ===
        for (size_t i = 0; i < m_root.size(); ++i) {
          curr_val[i] = scalar_value(m_root[i].func(x, t_eval));
        }

        // === Collect ALL triggered events ===
        triggered.clear();
        for (size_t i = 0; i < m_root.size(); ++i) {
          if (fired[i] < max_trigger &&
              !std::isnan(last_val[i]) &&
              last_val[i] * curr_val[i] < 0.0 &&
              direction_matches(last_val[i], curr_val[i], m_root[i].direction))
          {
            triggered.push_back({i, last_val[i], curr_val[i]});
          }
        }

        // === Process all triggered events together ===
        if (!triggered.empty()) {
          // Use first event for root localization
          size_t ref_idx = triggered[0].index;

          Time t_root = t_eval;
          State x_root = x;
          localize_root_dense(ref_idx, last_state[ref_idx], last_time[ref_idx],
                              x_root, t_root, triggered[0].last_val,
                              triggered[0].curr_val, root_tol);

          obs(x_root, t_root);

          bool has_terminal = false;
          for (const auto& te : triggered) {
            if (m_root[te.index].terminal) {
              has_terminal = true;
              break;
            }
          }

          // Apply ALL events at the same root time
          for (const auto& te : triggered) {
            size_t i = te.index;
            if (!m_root[i].terminal) {
              apply_event(x_root, m_root[i].state_index,
                          m_root[i].value_func(x_root, t_root), m_root[i].method);
            }
            fired[i]++;
          }

          obs(x_root, t_root + Time(1e-15));

          if (has_terminal) {
            x = x_root;
            return steps;
          }

          // Single reinitialization after all events
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

          // Don't advance iterator - reprocess in new interval
          continue;
        }

        // No events - update tracking and output
        for (size_t i = 0; i < m_root.size(); ++i) {
          last_val[i] = curr_val[i];
          last_state[i] = x;
          last_time[i] = t_eval;
        }

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
   * @brief Localize a root using controlled stepping (re-integration)
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

      if (fa * fm <= 0.0) {
        tb = tm; xb = xm; fb = fm;
      } else {
        ta = tm; xa = xm; fa = fm;
      }
    }
  }

  /**
   * @brief Localize a root using dense output (interpolation)
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

      State xm = xa;
      m_st.calc_state(tm, xm);

      double fm = scalar_value(m_root[idx].func(xm, tm));

      if (fa * fm <= 0.0) {
        tb = tm; xb = xm; fb = fm;
      } else {
        ta = tm; xa = xm; fa = fm;
      }
    }
  }

private:
  Stepper& m_st;
  System&  m_sys;
  const std::vector<FixedEvent<State, typename State::value_type>>& m_fixed;
  const std::vector<RootEvent<State, Time>>& m_root;
};

//==============================================================================
// Section 9: Public API
//==============================================================================

/**
 * @brief Integrate ODE at specified times with event handling (controlled stepper)
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
    const std::vector<FixedEvent<State, typename State::value_type>>& fixed,
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

/**
 * @brief Integrate ODE at specified times with event handling (dense output)
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
    const std::vector<FixedEvent<State, typename State::value_type>>& fixed,
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
using detail::make_steady_state_root_func;

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
