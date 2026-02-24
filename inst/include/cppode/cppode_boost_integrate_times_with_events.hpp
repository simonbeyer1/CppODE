/*
 * Event-aware integration at specified times with root-finding support.
 *
 * Original work (basic integrate_times structure):
 * Copyright (C) 2011-2015 Mario Mulansky
 * Copyright (C) 2012 Karsten Ahnert
 * Copyright (C) 2012 Christoph Koke
 * Distributed under the Boost Software License, Version 1.0.
 *
 * Modified work (event handling, root-finding, AD support):
 * Copyright (C) 2026 Simon Beyer
 *
 * Major extensions:
 * - Event system (fixed-time and root-finding events)
 * - Bisection-based root localization with tolerance
 * - Event application methods (Replace/Add/Multiply)
 * - Stepper reinitialization after events
 * - FADBAD++ automatic differentiation support
 * - Root tracking and fire count management
 * - Dense output optimization for root-finding
 * - Simultaneous root events
 * - Fully analytical saltation matrix corrections
 */

#ifndef CPPODE_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
#define CPPODE_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED

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

// ============================================================================
// Scalar value extraction
// ============================================================================

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v) {
  return static_cast<double>(v);
}

template<class T>
inline double scalar_value(const fadbad::F<T>& v) {
  return scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

// ============================================================================
// AD depth and sensitivity extraction
// ============================================================================

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, unsigned>::type
get_ad_depth(const T&) { return 0; }

template<class T>
inline unsigned get_ad_depth(const fadbad::F<T>& v) {
  T& inner = const_cast<fadbad::F<T>&>(v).x();
  return 1 + get_ad_depth(inner);
}

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, unsigned>::type
get_n_sens(const T&) { return 0; }

template<class T>
inline unsigned get_n_sens(const fadbad::F<T>& v) {
  return const_cast<fadbad::F<T>&>(v).size();
}

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
get_deriv(const T&, unsigned) { return 0.0; }

template<class T>
inline double get_deriv(const fadbad::F<T>& v, unsigned i) {
  return scalar_value(const_cast<fadbad::F<T>&>(v).d(i));
}

// ============================================================================
// Sensitivity correction helpers
// ============================================================================

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, void>::type
add_sens_correction(T&, unsigned, double) { }

template<class T>
inline void add_sens_correction(fadbad::F<fadbad::F<T>>& x, unsigned i, double correction) {
  if (i < x.size()) {
    x.d(i).x() += correction;
  }
}

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, void>::type
add_sens_correction(fadbad::F<T>& x, unsigned i, double correction) {
  if (i < x.size()) {
    x.d(i) += correction;
  }
}

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, void>::type
add_sens_correction_level(T&, unsigned, unsigned, double) { }

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, void>::type
add_sens_correction_level(fadbad::F<T>& x, unsigned i, unsigned level, double correction) {
  if (level == 0 && i < x.size()) {
    x.d(i) += correction;
  }
}

template<class T>
inline void add_sens_correction_level(fadbad::F<fadbad::F<T>>& x, unsigned i, unsigned level, double correction) {
  if (i >= x.size()) return;
  if (level == 0) {
    x.d(i).x() += correction;
  } else if (level == 1 && i < x.d(i).size()) {
    x.d(i).d(i) += correction;
  }
}

// ============================================================================
// Event application methods
// ============================================================================

enum class EventMethod {
  Replace,
  Add,
  Multiply
};

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

// ============================================================================
// Fixed-time event specification
// ============================================================================

template<class state_type, class value_type>
struct FixedEvent {
  value_type time;
  int state_index;
  std::function<value_type(const state_type&, const value_type&)> value_func;
  EventMethod method;

  // Analytical gradient of value_func w.r.t. state: dh/dx
  // Returns vector of size n_states (sparse: most entries zero)
  std::function<state_type(const state_type&, const value_type&)> dh_dx_func;
};

// ============================================================================
// Root-finding event specification
// ============================================================================

template<class state_type, class time_type>
struct RootEvent {
  using value_t = typename state_type::value_type;

  // Root function g(x, t) - event triggers when g crosses zero
  std::function<value_t(const state_type&, const time_type&)> func;

  // State index affected by event (-1 if terminal only)
  int state_index;

  // Value function h(x, t) - new value or modification
  std::function<value_t(const state_type&, const time_type&)> value_func;

  // Event method (Replace/Add/Multiply)
  EventMethod method;

  // Terminal flag - stop integration if true
  bool terminal = false;

  // Direction: 0 = any, -1 = falling, +1 = rising
  int direction = 0;

  // Analytical gradient of root function: dg/dx (vector of size n_states)
  std::function<state_type(const state_type&, const time_type&)> dg_dx_func;

  // Analytical time derivative of root function: dg/dt (scalar)
  std::function<value_t(const state_type&, const time_type&)> dg_dt_func;

  // Analytical gradient of value function: dh/dx (vector of size n_states)
  std::function<state_type(const state_type&, const time_type&)> dh_dx_func;
};

// ============================================================================
// Event tracking helper
// ============================================================================

struct TriggeredEvent {
  size_t index;
  double last_val;
  double curr_val;
};

inline bool direction_matches(double last_val, double curr_val, int direction) {
  if (direction == 0) return true;
  if (direction < 0) return last_val > 0.0 && curr_val < 0.0;
  return last_val < 0.0 && curr_val > 0.0;
}

// ============================================================================
// Steady-state root function helper
// ============================================================================

template<class System, class State, class Time>
auto make_steady_state_root_func(System& sys, double tol) {
  auto first = std::make_shared<bool>(true);
  return [&sys, tol, first](const State& x, const Time& t) -> typename State::value_type {
    if (*first) {
      *first = false;
      return typename State::value_type(1.0);
    }
    State dxdt(x.size());
    sys(x, dxdt, t);
    double max_rate = cppode::max_abs_all_levels_vec(dxdt);
    return typename State::value_type(max_rate - tol);
  };
}

// ============================================================================
// Analytical Saltation Correction - Core Implementation
// ============================================================================

// Saltation correction formula for state discontinuity at event:
//
//   dx/dp|+ = dx/dp|- + [f(x-, t) - f(x+, t)] * (dt_star/dp)
//
// where t_star is the event time. For root events:
//   t_star satisfies g(x(t_star), t_star) = 0
//
// By implicit function theorem:
//   dt_star/dp = -(dg/dx * dx/dp + dg/dp) / (dg/dx * f + dg/dt)
//              = -(dg/dx * dx/dp) / g_dot   (assuming g has no explicit p-dependence)
//
// For fixed-time events:
//   dt_star/dp = dt_event/dp  (directly from AD on t_event)

template<class state_type, class time_type, class System>
inline void apply_saltation_correction_core(
    state_type& x,
    const state_type& x_before,
    const time_type& dt_star,
    const time_type& t_eval,
    System& sys)
{
  if (x.empty()) return;

  const unsigned ad_depth = get_ad_depth(x[0]);
  if (ad_depth == 0) return;

  const unsigned n_sens = get_n_sens(x[0]);
  if (n_sens == 0) return;

  const size_t n = x.size();

  // Compute f(x_before, t) and f(x_after, t) at the actual event time t_eval.
  // For root events, dt_star = -g/g_dot has scalar value ~0 (since g=0
  // at the root) but carries the AD sensitivities dt*/dp.
  state_type f_before(n), f_after(n);
  sys.first(x_before, f_before, t_eval);
  sys.first(x, f_after, t_eval);

  // Deltaf values (constant across all AD levels)
  std::vector<double> delta_f_val(n);
  for (size_t k = 0; k < n; ++k) {
    delta_f_val[k] = scalar_value(f_before[k]) - scalar_value(f_after[k]);
  }

  // Level 0: First-order sensitivity correction
  for (unsigned j = 0; j < n_sens; ++j) {
    const double dt_dpj = get_deriv(dt_star, j);
    if (std::abs(dt_dpj) < 1e-30) continue;

    for (size_t k = 0; k < n; ++k) {
      add_sens_correction(x[k], j, delta_f_val[k] * dt_dpj);
    }
  }

  // Higher levels: corrections using derivative differences
  for (unsigned level = 1; level < ad_depth; ++level) {
    sys.first(x, f_after, t_eval);

    for (unsigned j = 0; j < n_sens; ++j) {
      const double dt_dpj = get_deriv(dt_star, j);
      if (std::abs(dt_dpj) < 1e-30) continue;

      for (size_t k = 0; k < n; ++k) {
        const double delta_f_deriv = get_deriv(f_before[k], j) - get_deriv(f_after[k], j);
        add_sens_correction_level(x[k], j, level, delta_f_deriv * dt_dpj);
      }
    }
  }
}

// ============================================================================
// Analytical g_dot computation (no numerical differentiation)
// ============================================================================

/*
 * Compute g_dot = dg/dt along the trajectory:
 *   g_dot = (dg/dx) * f(x,t) + dg/dt
 *
 * Both dg/dx and dg/dt are provided analytically via dg_dx_func and dg_dt_func.
 */

template<class state_type, class time_type, class System>
inline typename state_type::value_type compute_g_dot_analytical(
    const state_type& x,
    const time_type& t,
    System& sys,
    const std::function<state_type(const state_type&, const time_type&)>& dg_dx_func,
    const std::function<typename state_type::value_type(const state_type&, const time_type&)>& dg_dt_func)
{
  using value_type = typename state_type::value_type;
  const size_t n = x.size();

  // Get f(x, t)
  state_type f(n);
  sys.first(x, f, t);

  // Get dg/dx
  state_type dg_dx = dg_dx_func(x, t);

  // g_dot = Sum (dg/dx_i * f_i) + dg/dt
  value_type g_dot(0.0);
  for (size_t i = 0; i < n; ++i) {
    g_dot += dg_dx[i] * f[i];
  }

  // Add dg/dt if provided
  if (dg_dt_func) {
    g_dot += dg_dt_func(x, t);
  }

  return g_dot;
}

// ============================================================================
// Saltation correction for root events (fully analytical)
// ============================================================================

// For root events, dt_star is determined implicitly by g(x(t_star), t_star) = 0.
//
// The sensitivity of the event time is:
//   dt_star/dp = -(dg/dx * dx/dp) / g_dot
//
// where g_dot = dg/dx * f + dg/dt

template<class state_type, class time_type, class System>
inline void apply_saltation_correction_root(
    state_type& x,
    const state_type& x_before,
    const time_type& t,
    System& sys,
    const std::function<typename state_type::value_type(const state_type&, const time_type&)>& g_func,
    const std::function<state_type(const state_type&, const time_type&)>& dg_dx_func,
    const std::function<typename state_type::value_type(const state_type&, const time_type&)>& dg_dt_func)
{
  using value_type = typename state_type::value_type;

  if (x.empty()) return;

  const unsigned ad_depth = get_ad_depth(x[0]);
  if (ad_depth == 0) return;

  const unsigned n_sens = get_n_sens(x[0]);
  if (n_sens == 0) return;

  // Compute g_dot analytically
  value_type g_dot = compute_g_dot_analytical(x_before, t, sys, dg_dx_func, dg_dt_func);

  const double g_dot_val = scalar_value(g_dot);
  if (std::abs(g_dot_val) < 1e-15) return;  // Degenerate case

  // Compute dt* = -g / g_dot (as AD type to capture parameter sensitivities)
  value_type g = g_func(x_before, t);
  value_type dt_star = -g / g_dot;

  // Apply core saltation correction with actual event time t
  apply_saltation_correction_core(x, x_before, dt_star, t, sys);
}

// ============================================================================
// Saltation correction for fixed-time events
// ============================================================================

/*
 * For fixed-time events, t_event may depend on parameters.
 * The sensitivity dt_event/dp is captured in the AD type of t_event.
 */

template<class state_type, class time_type, class System>
inline void apply_saltation_correction_fixed(
    state_type& x,
    const state_type& x_before,
    const time_type& t_event,
    System& sys)
{
  apply_saltation_correction_core(x, x_before, t_event, t_event, sys);
}

// ============================================================================
// Apply fixed events with saltation correction
// ============================================================================

template<class state_type, class Time, class System>
bool apply_fixed_events_at_time_with_saltation(
    state_type& x,
    const Time& t,
    const std::vector<FixedEvent<state_type, typename state_type::value_type>>& evs,
    System& sys)
{
  const double tt = scalar_value(t);
  bool fired = false;

  for (const auto& e : evs) {
    if (std::abs(scalar_value(e.time) - tt) < 1e-14) {
      state_type x_before = x;
      apply_event(x, e.state_index, e.value_func(x, e.time), e.method);
      apply_saltation_correction_fixed(x, x_before, e.time, sys);
      fired = true;
    }
  }

  return fired;
}

// ============================================================================
// Merge user times with event times
// ============================================================================

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
            [](const auto& a, const auto& b) {
              return scalar_value(a) < scalar_value(b);
            });

  out.erase(std::unique(out.begin(), out.end(),
                        [](const auto& a, const auto& b) {
                          return std::abs(scalar_value(a) - scalar_value(b)) < 1e-14;
                        }),
                        out.end());

  return out;
}

// ============================================================================
// Unified stepper reset (C++17 SFINAE)
// ============================================================================

// Note: has_reset_after_event is already defined in
// cppode_boost_rosenbrock4_dense_output_pi.hpp (included earlier).
// Only has_reinitialize_at_event needs to be defined here.

template<class S, class State, class Time, class = void>
struct has_reinitialize_at_event : std::false_type {};

template<class S, class State, class Time>
struct has_reinitialize_at_event<S, State, Time,
    std::void_t<decltype(std::declval<S&>().reinitialize_at_event(
        std::declval<State&>(), std::declval<Time>(), std::declval<Time&>()))>
> : std::true_type {};

template<class S, class State, class Time>
inline void reset_stepper_unified(S& st, State& x, Time t, Time& dt) {
  if constexpr (has_reset_after_event<S, Time>::value) {
    st.reset_after_event(dt);
  }
  else if constexpr (has_reinitialize_at_event<S, State, Time>::value) {
    st.reinitialize_at_event(x, t, dt);
  }
  else {
    (void)st; (void)x; (void)t; (void)dt;
  }
}

// ============================================================================
// EventEngine - Core event-aware integration engine
// ============================================================================

template<class Stepper, class System, class State, class Time>
class EventEngine {
public:
  using state_type = State;
  using time_type  = Time;
  using value_type = typename State::value_type;

  EventEngine(
    Stepper& st,
    System& sys,
    const std::vector<FixedEvent<State, value_type>>& fixed,
    const std::vector<RootEvent<State, Time>>& root)
    : m_st(st)
    , m_sys(sys)
    , m_fixed(fixed)
    , m_root(root)
  {}

private:
  // ========================================================================
  // Helper: Check for triggered root events
  // ========================================================================
  void check_root_triggers(
      std::vector<TriggeredEvent>& triggered,
      const std::vector<double>& last_val,
      const std::vector<double>& curr_val,
      const std::vector<size_t>& fired,
      size_t max_trigger)
  {
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
  }

  // ========================================================================
  // Helper: Apply root events and saltation corrections
  // ========================================================================
  bool apply_root_events(
      State& x_root,
      const State& x_before,
      const Time& t_root,
      const std::vector<TriggeredEvent>& triggered,
      std::vector<size_t>& fired)
  {
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
        apply_event(x_root,
                            m_root[i].state_index,
                            m_root[i].value_func(x_before, t_root),
                            m_root[i].method);
      }
      fired[i]++;
    }

    // Apply saltation corrections
    for (const auto& te : triggered) {
      size_t i = te.index;
      if (!m_root[i].terminal && m_root[i].dg_dx_func) {
        apply_saltation_correction_root(
          x_root, x_before, t_root, m_sys,
          m_root[i].func,
          m_root[i].dg_dx_func,
          m_root[i].dg_dt_func);
      }
    }

    return has_terminal;
  }

  // ========================================================================
  // Helper: Update stepper state after event
  // ========================================================================
  template<class Checker>
  void reinit_after_event(
      State& x,
      const Time& t_event,
      Time& dt,
      Time& t_start,
      Time& t_end,
      State& x_at_start,
      std::vector<double>& last_val,
      std::vector<State>& last_state,
      std::vector<Time>& last_time,
      size_t& steps,
      Checker& checker)
  {
    m_st.initialize(x, t_event, dt);
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

  // ========================================================================
  // Helper: Evaluate curr_val at given state and time
  // ========================================================================
  void eval_root_funcs(std::vector<double>& curr_val, const State& x, const Time& t)
  {
    for (size_t i = 0; i < m_root.size(); ++i) {
      curr_val[i] = scalar_value(m_root[i].func(x, t));
    }
  }

public:

  // ========================================================================
  // Controlled Stepper Integration
  // ========================================================================

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
    size_t steps = 0;
    auto it = times.begin();
    const auto end = times.end();
    Time t = *it;

    if (apply_fixed_events_at_time_with_saltation(x, t, m_fixed, m_sys)) {
      reset_stepper_unified(m_st, x, t, dt);
    }

    obs(x, t);
    ++it;
    if (it == end) return 0;

    std::vector<double> last_val(m_root.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<State>  last_state(m_root.size(), x);
    std::vector<Time>   last_time(m_root.size(), t);
    std::vector<size_t> fired(m_root.size(), 0);

    std::vector<TriggeredEvent> triggered;
    triggered.reserve(m_root.size());
    std::vector<double> curr_val(m_root.size());

    while (it != end) {
      Time t_target = *it;
      double t_target_scalar = scalar_value(t_target);

      while (scalar_value(t) < t_target_scalar - 1e-14) {
        double remaining = t_target_scalar - scalar_value(t);
        Time dt_step = (scalar_value(dt) < remaining) ? dt : Time(remaining);

        auto result = m_st.try_step(m_sys, x, t, dt_step);

        if (result == success) {
          ++steps;
          checker();
          checker.reset();
          dt = dt_step;

          eval_root_funcs(curr_val, x, t);
          check_root_triggers(triggered, last_val, curr_val, fired, max_trigger);

          if (!triggered.empty()) {
            localize_root_controlled(
              triggered[0].index,
              last_state[triggered[0].index], last_time[triggered[0].index],
                                                       x, t,
                                                       triggered[0].last_val, triggered[0].curr_val,
                                                       root_tol, checker);

            Time t_root = t;
            State x_before = x;

            // Output state BEFORE event (at t_root - epsilon)
            obs(x_before, t_root - Time(1e-15));

            State x_after = x;
            if (apply_root_events(x_after, x, t_root, triggered, fired)) {
              // Terminal event - output final state at t_root
              obs(x_after, t_root);
              x = x_after;
              return steps;
            }

            // Output state AFTER event (at t_root)
            obs(x_after, t_root);
            x = x_after;

            reset_stepper_unified(m_st, x, t, dt);

            for (size_t j = 0; j < m_root.size(); ++j) {
              last_val[j] = std::numeric_limits<double>::quiet_NaN();
              last_state[j] = x;
              last_time[j] = t;
            }
          }
          else {
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

      if (apply_fixed_events_at_time_with_saltation(x, t, m_fixed, m_sys)) {
        obs(x, t);
        reset_stepper_unified(m_st, x, t, dt);
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

  // ========================================================================
  // Dense Output Stepper Integration
  // ========================================================================

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

    if (apply_fixed_events_at_time_with_saltation(x, *it, m_fixed, m_sys)) {
      m_st.initialize(x, *it, dt);
    }

    obs(x, *it);
    ++it;
    if (it == end) return 0;

    m_st.initialize(x, times.front(), dt);
    m_st.do_step(m_sys);
    ++steps;
    checker();
    checker.reset();

    Time t_start = m_st.previous_time();
    Time t_end = m_st.current_time();
    dt = m_st.current_time_step();

    State x_at_start(x.size());
    m_st.calc_state(t_start, x_at_start);

    std::vector<double> last_val(m_root.size());
    std::vector<State>  last_state(m_root.size(), x_at_start);
    std::vector<Time>   last_time(m_root.size(), t_start);
    std::vector<size_t> fired(m_root.size(), 0);

    for (size_t i = 0; i < m_root.size(); ++i) {
      last_val[i] = scalar_value(m_root[i].func(x_at_start, t_start));
    }

    std::vector<TriggeredEvent> triggered;
    triggered.reserve(m_root.size());
    std::vector<double> curr_val(m_root.size());

    while (it != end) {
      while (!less_eq_with_sign(*it, t_end, dt)) {
        // Before doing next step, check for root crossing in current interval
        m_st.calc_state(t_end, x);
        eval_root_funcs(curr_val, x, t_end);
        check_root_triggers(triggered, last_val, curr_val, fired, max_trigger);

        if (!triggered.empty()) {
          // Root found - handle it before continuing
          State x_root = x_at_start;
          Time t_root = t_start;

          localize_root_dense(
            triggered[0].index,
            x_root, t_root, t_end,
            triggered[0].last_val, triggered[0].curr_val,
            root_tol);

          State x_before = x_root;

          // Output state BEFORE event (at t_root - epsilon)
          obs(x_before, t_root - Time(1e-15));

          if (apply_root_events(x_root, x_before, t_root, triggered, fired)) {
            // Terminal event - output final state at t_root
            obs(x_root, t_root);
            x = x_root;
            return steps;
          }

          // Output state AFTER event (at t_root)
          obs(x_root, t_root);

          x = x_root;
          reinit_after_event(x, t_root, dt, t_start, t_end, x_at_start,
                             last_val, last_state, last_time, steps, checker);
          continue;
        }

        // No root in this interval - update tracking and proceed to next step
        for (size_t i = 0; i < m_root.size(); ++i) {
          last_val[i] = curr_val[i];
          last_state[i] = x;
          last_time[i] = t_end;
        }

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

      // Now t_end >= *it, process all requested output times in current interval
      while (it != end && less_eq_with_sign(*it, t_end, dt)) {
        Time t_eval = *it;

        // Skip times that are before our current interval start
        if (scalar_value(t_eval) < scalar_value(t_start)) {
          // Output at t_start since we can't go back
          m_st.calc_state(t_start, x);
          obs(x, t_eval);
          ++it;
          continue;
        }

        Time t_eval_scalar = Time(scalar_value(t_eval));
        m_st.calc_state(t_eval_scalar, x);

        // Check for root events FIRST (they happen before t_eval)
        eval_root_funcs(curr_val, x, t_eval);
        check_root_triggers(triggered, last_val, curr_val, fired, max_trigger);

        if (!triggered.empty()) {
          State x_root = last_state[triggered[0].index];
          Time t_root = last_time[triggered[0].index];

          localize_root_dense(
            triggered[0].index,
            x_root, t_root, t_eval,
            triggered[0].last_val, triggered[0].curr_val,
            root_tol);

          State x_before = x_root;
          Time t_before = t_root - Time(1e-15);

          // Output state BEFORE event (at t_root - epsilon)
          // Skip if t_eval is very close to t_before (would be duplicate)
          if (std::abs(scalar_value(t_eval) - scalar_value(t_before)) >= 1e-14) {
            obs(x_before, t_before);
          }

          if (apply_root_events(x_root, x_before, t_root, triggered, fired)) {
            // Terminal event - output final state at t_root
            if (std::abs(scalar_value(t_eval) - scalar_value(t_root)) >= 1e-14) {
              obs(x_root, t_root);
            }
            x = x_root;
            return steps;
          }

          // Output state AFTER event (at t_root)
          // Skip if t_eval is very close to t_root (would be duplicate)
          if (std::abs(scalar_value(t_eval) - scalar_value(t_root)) >= 1e-14) {
            obs(x_root, t_root);
          }

          x = x_root;
          reinit_after_event(x, t_root, dt, t_start, t_end, x_at_start,
                             last_val, last_state, last_time, steps, checker);
          // Don't increment it - t_eval still needs to be output
          // Break out to re-check bounds in outer loop
          break;
        }

        // Check for fixed events at this time
        bool fixed_event_fired = apply_fixed_events_at_time_with_saltation(x, t_eval, m_fixed, m_sys);

        // Output the requested time point (with state after any fixed event)
        obs(x, t_eval);
        ++it;

        if (fixed_event_fired) {
          // Reinitialize stepper after the event
          m_st.initialize(x, t_eval_scalar, dt);
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
          // Break out to re-check bounds in outer loop
          break;
        }

        // Update root tracking
        for (size_t i = 0; i < m_root.size(); ++i) {
          last_val[i] = curr_val[i];
          last_state[i] = x;
          last_time[i] = t_eval;
        }
      }
    }

    return steps;
  }

private:
  // ========================================================================
  // Root localization (controlled stepper)
  // ========================================================================

  template<class Checker>
  void localize_root_controlled(
      size_t idx,
      State& x_lo, Time& t_lo,
      State& x_hi, Time& t_hi,
      double g_lo, double g_hi,
      double tol,
      Checker& checker)
  {
    constexpr int max_iter = 50;

    for (int iter = 0; iter < max_iter; ++iter) {
      double dt_interval = scalar_value(t_hi) - scalar_value(t_lo);
      if (dt_interval < tol) break;

      double alpha = -g_lo / (g_hi - g_lo);
      alpha = std::max(0.1, std::min(0.9, alpha));

      Time t_mid = t_lo + Time(alpha * dt_interval);
      State x_mid = x_lo;

      Time t_tmp = t_lo;
      Time dt_tmp = t_mid - t_lo;

      while (scalar_value(t_tmp) < scalar_value(t_mid) - 1e-15) {
        double dt_tmp_scalar = scalar_value(dt_tmp);
        double remaining = scalar_value(t_mid) - scalar_value(t_tmp);
        Time step_dt = (dt_tmp_scalar < remaining) ? dt_tmp : Time(remaining);

        auto result = m_st.try_step(m_sys, x_mid, t_tmp, step_dt);
        if (result == success) {
          checker();
          checker.reset();
          dt_tmp = step_dt;
        } else {
          dt_tmp = step_dt;
        }
      }

      double g_mid = scalar_value(m_root[idx].func(x_mid, t_mid));

      if (g_lo * g_mid < 0.0) {
        x_hi = x_mid;
        t_hi = t_mid;
        g_hi = g_mid;
      } else {
        x_lo = x_mid;
        t_lo = t_mid;
        g_lo = g_mid;
      }
    }

    x_hi = x_lo;
    t_hi = t_lo;
  }

  // ========================================================================
  // Root localization (dense output)
  // ========================================================================

  void localize_root_dense(
      size_t idx,
      State& x_root, Time& t_root, Time t_hi,
      double g_lo, double g_hi,
      double tol)
  {
    constexpr int max_iter = 50;
    Time t_lo = t_root;

    for (int iter = 0; iter < max_iter; ++iter) {
      double dt_interval = scalar_value(t_hi) - scalar_value(t_lo);
      if (dt_interval < tol) break;

      double alpha = -g_lo / (g_hi - g_lo);
      alpha = std::max(0.1, std::min(0.9, alpha));

      Time t_mid = t_lo + Time(alpha * dt_interval);
      State x_mid(x_root.size());
      m_st.calc_state(t_mid, x_mid);

      double g_mid = scalar_value(m_root[idx].func(x_mid, t_mid));

      if (g_lo * g_mid < 0.0) {
        t_hi = t_mid;
        g_hi = g_mid;
      } else {
        t_lo = t_mid;
        g_lo = g_mid;
        x_root = x_mid;
        t_root = t_mid;
      }
    }

    m_st.calc_state(t_lo, x_root);
    t_root = t_lo;
  }

  Stepper& m_st;
  System& m_sys;
  const std::vector<FixedEvent<State, value_type>>& m_fixed;
  const std::vector<RootEvent<State, Time>>& m_root;
};

// ============================================================================
// Public API - integrate_times for controlled stepper
// ============================================================================

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
    boost::numeric::odeint::controlled_stepper_tag = boost::numeric::odeint::controlled_stepper_tag())
{
  auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);
  EventEngine<Stepper, System, State, Time> eng(stepper, system, fixed, root);
  return eng.process_controlled(x, times, dt, obs, checker, root_tol, max_trigger_root);
}

// ============================================================================
// Public API - integrate_times_dense for dense output stepper
// ============================================================================

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
    boost::numeric::odeint::dense_output_stepper_tag = boost::numeric::odeint::dense_output_stepper_tag())
{
  auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);
  EventEngine<Stepper, System, State, Time> eng(stepper, system, fixed, root);
  return eng.process_dense(x, times, dt, obs, checker, root_tol, max_trigger_root);
}

} // namespace detail

// ============================================================================
// Public Namespace Exports
// ============================================================================

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
