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
 - Simultaneous root events
 - Complete saltation matrix corrections
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

// Scalar value extraction for arithmetic types
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v)
{
  return static_cast<double>(v);
}

// Scalar value extraction for FADBAD++ AD types
template<class T>
inline double scalar_value(const fadbad::F<T>& v)
{
  return scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

// Event application methods
enum class EventMethod {
  Replace,
  Add,
  Multiply
};

// Fixed-time event specification
template<class state_type, class value_type>
struct FixedEvent {
  value_type  time;
  int         state_index;
  std::function<value_type(const state_type&, const value_type&)> value_func;
  EventMethod method;
};

// Root-finding event specification
// Field order must match generated code: func, state_index, value_func, method
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

// Helper for simultaneous event tracking
struct TriggeredEvent {
  size_t index;
  double last_val;
  double curr_val;
};

// Direction matching for root events
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

// Helper for steady-state root functions
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

// Get AD nesting depth: double=0, F<double>=1, F<F<double>>=2, etc.
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, unsigned>::type
get_ad_depth(const T&) { return 0; }

template<class T>
inline unsigned get_ad_depth(const fadbad::F<T>& v) {
  // Use const_cast because FADBAD++'s x() is not const-qualified
  T& inner = const_cast<fadbad::F<T>&>(v).x();
  return 1 + get_ad_depth(inner);
}

// Get number of sensitivity directions
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, unsigned>::type
get_n_sens(const T&) { return 0; }

template<class T>
inline unsigned get_n_sens(const fadbad::F<T>& v) {
  return const_cast<fadbad::F<T>&>(v).size();
}

// Get derivative component from AD type
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
get_deriv(const T&, unsigned) { return 0.0; }

template<class T>
inline double get_deriv(const fadbad::F<T>& v, unsigned i) {
  return scalar_value(const_cast<fadbad::F<T>&>(v).d(i));
}

// Add correction to sensitivity component - base case for non-AD types
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, void>::type
add_sens_correction(T&, unsigned, double) { }

// Add correction for F<F<T>> (second-order AD)
// For nested F<F<double>>: x.d(i).x() is the first-order sensitivity dx/dp_i
// (x.x().d(i) would be the inner derivative, used for second-order computation)
template<class T>
inline void add_sens_correction(fadbad::F<fadbad::F<T>>& x, unsigned i, double correction) {
  if (i < x.size()) {
    x.d(i).x() += correction;
  }
}

// Add correction for F<T> where T is arithmetic (first-order AD)
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, void>::type
add_sens_correction(fadbad::F<T>& x, unsigned i, double correction) {
  if (i < x.size()) {
    x.d(i) += correction;
  }
}

// Apply event modification to state vector
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

// Create frozen state (derivatives zeroed)
template<class state_type>
inline state_type create_frozen_state(const state_type& x)
{
  state_type x_frozen(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x_frozen[i] = typename state_type::value_type(scalar_value(x[i]));
  }
  return x_frozen;
}

// Compute time derivative of root function as AD type (for all derivative orders)
// g_dot = dg/dt = ∂g/∂x * f(x,t) + ∂g/∂t
template<class state_type, class time_type, class System, class RootFunc>
inline typename state_type::value_type compute_g_dot_ad(
    const state_type& x,
    const time_type& t,
    System& sys,
    RootFunc& root_func,
    int event_state_idx)
{
  using value_type = typename state_type::value_type;

  state_type f(x.size());
  sys.first(x, f, t);

  // For state-based roots (g = x[idx] - threshold), g_dot = f[idx]
  if (event_state_idx >= 0 && static_cast<size_t>(event_state_idx) < f.size()) {
    if (std::abs(scalar_value(f[event_state_idx])) > 1e-15) {
      return f[event_state_idx];
    }
  }

  // General case: numerical differentiation of g along the flow
  // g_dot ≈ [g(x + eps*f, t + eps) - g(x, t)] / eps
  const double eps = 1e-8;
  value_type g_curr = root_func(x, t);

  state_type x_pert(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x_pert[i] = x[i] + value_type(eps) * f[i];
  }

  value_type g_pert = root_func(x_pert, t + time_type(eps));
  return (g_pert - g_curr) / value_type(eps);
}

// Legacy scalar version for backward compatibility
template<class state_type, class time_type, class System, class RootFunc>
inline double compute_g_dot(
    const state_type& x,
    const time_type& t,
    System& sys,
    RootFunc& root_func,
    int event_state_idx)
{
  return scalar_value(compute_g_dot_ad(x, t, sys, root_func, event_state_idx));
}

// Saltation matrix correction for ROOT events (all AD orders, iterative)
// Formula: x+ = x- + [f(x-) - f(x+)] * dt*
// where dt* = -g / g_dot (implicit function theorem)
//
// For higher-order derivatives, we need iterative application using
// INCREMENTAL differences to avoid adding first-order multiple times.
// Number of iterations = AD nesting depth (F<T>=1, F<F<T>>=2, etc.)
template<class state_type, class time_type, class System, class RootFunc>
inline void apply_saltation_correction_root(
    state_type& x,
    const state_type& x_before,
    const time_type& t,
    System& sys,
    RootFunc& root_func,
    int event_state_idx)
{
  using value_type = typename state_type::value_type;

  if (x.empty()) return;

  // Determine number of iterations from AD depth
  unsigned ad_depth = get_ad_depth(x[0]);
  if (ad_depth == 0) return;  // No AD, no correction needed

  // Compute g and g_dot using x_before (with full AD for parameter derivatives)
  value_type g = root_func(x_before, t);
  value_type g_dot = compute_g_dot_ad(x_before, t, sys, root_func, event_state_idx);

  // Check if g_dot is too small (would cause division issues)
  if (std::abs(scalar_value(g_dot)) < 1e-15) {
    return;
  }

  // dt* = -g / g_dot
  // At the root: g.value() ≈ 0, so dt_star.value() ≈ 0
  value_type dt_star = -g / g_dot;

  // f_prev starts with f_before
  state_type f_prev(x_before.size());
  sys.first(x_before, f_prev, t);

  // Iterative saltation using INCREMENTAL differences
  for (unsigned iter = 0; iter < ad_depth; ++iter) {
    state_type f_curr(x.size());
    sys.first(x, f_curr, t);

    // Apply incremental saltation correction
    for (size_t k = 0; k < x.size(); ++k) {
      x[k] += (f_prev[k] - f_curr[k]) * dt_star;
    }

    // Update f_prev for next iteration
    f_prev = f_curr;
  }
}

// Saltation matrix correction for FIXED-TIME events (all AD orders)
// Formula: dx+/dp = dx-/dp + [f(x-) - f(x+)] * dt_event/dp
//
// For higher-order derivatives, we need iterative application because:
// - First order: x.d1 += Δf.value * dt*.d1
// - Second order: x.d2 += Δf.d1 * dt*.d1 (but Δf.d1 depends on x.d1!)
//
// IMPORTANT: Each iteration must use INCREMENTAL differences to avoid
// adding the first-order contribution multiple times.
// - Iteration 0: x += (f_before - f_after_0) * dt*  → adds 1st order
// - Iteration 1: x += (f_after_0 - f_after_1) * dt* → adds 2nd order only
// - etc.
//
// Number of iterations = AD nesting depth (F<T>=1, F<F<T>>=2, etc.)
template<class state_type, class time_type, class System>
inline void apply_saltation_correction_fixed_iterative(
    state_type& x,                    // x AFTER event (will be corrected in-place)
    const state_type& x_before,       // x BEFORE event (full AD state!)
    const time_type& t_event,
    System& sys)
{
  if (x.empty()) return;

  // Determine number of iterations from AD depth
  unsigned ad_depth = get_ad_depth(x[0]);
  if (ad_depth == 0) return;  // No AD, no correction needed

  // Create a "derivative-only" version of t_event:
  // value = 0, but all derivatives preserved
  time_type t_deriv_only = t_event - time_type(scalar_value(t_event));

  // f_prev starts with f_before
  state_type f_prev(x_before.size());
  sys.first(x_before, f_prev, t_event);

  // Iterative saltation using INCREMENTAL differences:
  // Each iteration only adds the contribution from the CHANGE in f_after
  for (unsigned iter = 0; iter < ad_depth; ++iter) {
    state_type f_curr(x.size());
    sys.first(x, f_curr, t_event);

    // Apply incremental saltation correction
    // Iter 0: (f_before - f_after_0) → first-order contribution
    // Iter 1: (f_after_0 - f_after_1) → second-order contribution only
    for (size_t k = 0; k < x.size(); ++k) {
      x[k] += (f_prev[k] - f_curr[k]) * t_deriv_only;
    }

    // Update f_prev for next iteration
    f_prev = f_curr;
  }
}

// Legacy non-iterative version for backward compatibility
template<class state_type, class time_type>
inline void apply_saltation_correction_fixed(
    state_type& x,
    const time_type& t_event,
    const state_type& f_before,
    const state_type& f_after)
{
  if (x.empty()) return;

  // Create a "derivative-only" version of t_event:
  // value = 0, but all derivatives preserved
  time_type t_deriv_only = t_event - time_type(scalar_value(t_event));

  // Add saltation correction: FADBAD++ propagates all derivative orders
  // Value contribution is 0 (because t_deriv_only.value() == 0)
  // Derivative contributions follow from the product rule
  for (size_t k = 0; k < x.size(); ++k) {
    x[k] += (f_before[k] - f_after[k]) * t_deriv_only;
  }
}

// Apply fixed events with saltation (iterative for all AD orders)
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
      // IMPORTANT: Save x_before BEFORE applying the event!
      // Keep full AD state (parameter derivatives), only freeze later if needed
      state_type x_before = x;

      // Apply the event value change
      apply_event(x, e.state_index, e.value_func(x, e.time), e.method);

      // Apply iterative saltation correction for all AD orders
      apply_saltation_correction_fixed_iterative(x, x_before, e.time, sys);

      fired = true;
    }
  }

  return fired;
}

// Apply fixed events without saltation (legacy)
template<class state_type, class Time>
bool apply_fixed_events_at_time(
    state_type& x,
    const Time& t,
    const std::vector<FixedEvent<state_type, typename state_type::value_type>>& evs)
{
  const double tt = scalar_value(t);
  bool fired = false;

  for (const auto& e : evs) {
    if (std::abs(scalar_value(e.time) - tt) < 1e-14) {
      apply_event(x, e.state_index, e.value_func(x, e.time), e.method);
      fired = true;
    }
  }

  return fired;
}

// Merge user times with event times
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
                          return std::abs(scalar_value(a) - scalar_value(b)) < 1e-14;
                        }),
                        out.end());

  return out;
}

// Unified stepper reset
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

// EventEngine - Core event-aware integration engine
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

  // Controlled Stepper Integration
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

    std::vector<double> last_val(m_root.size(),
                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<State>  last_state(m_root.size(), x);
    std::vector<Time>   last_time(m_root.size(), t);
    std::vector<size_t> fired(m_root.size(), 0);

    std::vector<TriggeredEvent> triggered;
    triggered.reserve(m_root.size());
    std::vector<double> curr_val(m_root.size());

    while (it != end)
    {
      Time t_target = *it;
      double t_target_scalar = scalar_value(t_target);

      while (scalar_value(t) < t_target_scalar - 1e-14)
      {
        double remaining = t_target_scalar - scalar_value(t);
        Time dt_step = (scalar_value(dt) < remaining) ? dt : Time(remaining);

        auto result = m_st.try_step(m_sys, x, t, dt_step);

        if (result == success) {
          ++steps;
          checker();
          checker.reset();
          dt = dt_step;

          for (size_t i = 0; i < m_root.size(); ++i) {
            curr_val[i] = scalar_value(m_root[i].func(x, t));
          }

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

          if (!triggered.empty()) {
            size_t ref_idx = triggered[0].index;

            localize_root_controlled(
              ref_idx,
              last_state[ref_idx], last_time[ref_idx],
                                            x, t,
                                            triggered[0].last_val, triggered[0].curr_val,
                                            root_tol,
                                            checker);

            Time t_root = t;
            obs(x, t_root);

            bool has_terminal = false;
            for (const auto& te : triggered) {
              if (m_root[te.index].terminal) {
                has_terminal = true;
                break;
              }
            }

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

            for (const auto& te : triggered) {
              size_t i = te.index;
              if (!m_root[i].terminal) {
                apply_saltation_correction_root(x_after, x, t_root, m_sys,
                                                m_root[i].func,
                                                m_root[i].state_index);
              }
            }

            obs(x_after, t_root + Time(1e-15));
            x = x_after;

            if (has_terminal) {
              return steps;
            }

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

  // Dense Output Stepper Integration
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

    if (apply_fixed_events_at_time_with_saltation(x, *it, m_fixed, m_sys))
    {
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

    while (it != end)
    {
      while (!less_eq_with_sign(*it, t_end, dt))
      {
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

      m_st.calc_state(t_end, x);
      for (size_t i = 0; i < m_root.size(); ++i) {
        curr_val[i] = scalar_value(m_root[i].func(x, t_end));
      }

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

      if (!triggered.empty()) {
        size_t ref_idx = triggered[0].index;

        State x_root = x_at_start;
        Time t_root = t_start;

        localize_root_dense(
          ref_idx,
          x_root, t_root, t_end,
          triggered[0].last_val, triggered[0].curr_val,
          root_tol);

        State x_before = x_root;

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
            apply_event(x_root,
                        m_root[i].state_index,
                        m_root[i].value_func(x_before, t_root),
                        m_root[i].method);
          }
          fired[i]++;
        }

        for (const auto& te : triggered) {
          size_t i = te.index;
          if (!m_root[i].terminal) {
            apply_saltation_correction_root(x_root, x_before, t_root, m_sys,
                                            m_root[i].func,
                                            m_root[i].state_index);
          }
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

      for (size_t i = 0; i < m_root.size(); ++i) {
        if (scalar_value(last_time[i]) < scalar_value(t_start)) {
          m_st.calc_state(t_start, x_at_start);
          last_val[i] = scalar_value(m_root[i].func(x_at_start, t_start));
          last_state[i] = x_at_start;
          last_time[i] = t_start;
        }
      }

      while (it != end && less_eq_with_sign(*it, t_end, dt))
      {
        Time t_eval = *it;

        if (scalar_value(t_eval) < scalar_value(t_start)) {
          ++it;
          continue;
        }

        // Use scalar time for calc_state to avoid AD propagating d(trajectory)/d(t_event)
        // into sensitivities. The correct parameter dependency is added by saltation correction.
        Time t_eval_scalar = Time(scalar_value(t_eval));
        m_st.calc_state(t_eval_scalar, x);

        if (apply_fixed_events_at_time_with_saltation(x, t_eval, m_fixed, m_sys))
        {
          obs(x, t_eval);

          // Use scalar time for initialize - the state x already has correct sensitivities
          // from saltation correction. Using AD time would pollute subsequent integration.
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

          ++it;
          continue;
        }

        for (size_t i = 0; i < m_root.size(); ++i) {
          curr_val[i] = scalar_value(m_root[i].func(x, t_eval));
        }

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

        if (!triggered.empty()) {
          size_t ref_idx = triggered[0].index;

          State x_root = last_state[ref_idx];
          Time t_root = last_time[ref_idx];

          localize_root_dense(
            ref_idx,
            x_root, t_root, t_eval,
            triggered[0].last_val, triggered[0].curr_val,
            root_tol);

          State x_before = x_root;
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
              apply_event(x_root,
                          m_root[i].state_index,
                          m_root[i].value_func(x_before, t_root),
                          m_root[i].method);
            }
            fired[i]++;
          }

          for (const auto& te : triggered) {
            size_t i = te.index;
            if (!m_root[i].terminal) {
              apply_saltation_correction_root(x_root, x_before, t_root, m_sys,
                                              m_root[i].func,
                                              m_root[i].state_index);
            }
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
          continue;
        }

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
  template<class Checker>
  void localize_root_controlled(
      size_t idx,
      State& x_lo, Time& t_lo,
      State& x_hi, Time& t_hi,
      double g_lo, double g_hi,
      double tol,
      Checker& checker)
  {
    const int max_iter = 50;

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

  void localize_root_dense(
      size_t idx,
      State& x_root, Time& t_root, Time t_hi,
      double g_lo, double g_hi,
      double tol)
  {
    const int max_iter = 50;
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
  const std::vector<FixedEvent<State, typename State::value_type>>& m_fixed;
  const std::vector<RootEvent<State, Time>>& m_root;
};

// Public API - integrate_times for controlled stepper
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
  return eng.process_controlled(x, times, dt, obs, checker,
                                root_tol, max_trigger_root);
}

// Public API - integrate_times_dense for dense output stepper
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
  return eng.process_dense(x, times, dt, obs, checker,
                           root_tol, max_trigger_root);
}

} // namespace detail

// Public Namespace Exports
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
