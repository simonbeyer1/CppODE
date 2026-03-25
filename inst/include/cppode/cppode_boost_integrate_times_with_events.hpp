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
 * - Analytical saltation corrections (IFT-based, codegen-provided gradients)
 *
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
// AD type detection (C++17 SFINAE)
// ============================================================================

template<class T>
struct is_ad_type : std::false_type {};

template<class T>
struct is_ad_type<fadbad::F<T>> : std::true_type {};

// ============================================================================
// Event application methods
// ============================================================================

enum class EventMethod {
  Replace,
  Add,
  Multiply
};

// ============================================================================
// Fixed-time event specification
// ============================================================================

template<class state_type, class value_type>
struct FixedEvent {
  value_type time;
  int state_index;
  std::function<value_type(const state_type&, const value_type&)> value_func;
  EventMethod method;
};

// ============================================================================
// Root-finding event specification
//
// The dg_dx / dg_dt members are codegen-provided analytical partial
// derivatives of the root function g(x,t).  They are required for AD
// models (FADBAD++ types) to compute the analytical saltation correction.
// For pure-double models they are unused and may be left unset.
// ============================================================================

template<class state_type, class time_type>
struct RootEvent {
  using value_t = typename state_type::value_type;

  // Root condition g(x, t) = 0
  std::function<value_t(const state_type&, const time_type&)> func;

  // Affected state index (-1 for none / terminal-only)
  int state_index;

  // Event action value h(x, t)
  std::function<value_t(const state_type&, const time_type&)> value_func;

  EventMethod method;
  bool terminal = false;
  int direction = 0;

  // --- Codegen-provided analytical gradients (required for AD) ---
  //
  //   dg_dx: writes (dg/dx_0, dg/dx_1, ...) into out
  //   dg_dt: returns  dg/dt
  //
  // These are PARTIAL derivatives of g with respect to the state
  // vector and time, operating on the full value_type (AD-compatible).
  // The codegen derives them symbolically from g at model build time.
  std::function<void(const state_type&, const time_type&, state_type&)> dg_dx;
  std::function<value_t(const state_type&, const time_type&)> dg_dt;

  // --- Codegen-provided G_tt for second-order IFT correction ---
  //
  //   g_dot_dot: returns G_tt = d(g_dot)/dt, the total second time
  //              derivative of g along the ODE trajectory.
  //
  //   Symbolically:
  //     G_tt = f^T · H_g · f  +  (dg/dx)^T · (J_f·f + df/dt)
  //          + 2·(d²g/dxdt)^T · f  +  d²g/dt²
  //
  //   where H_g is the Hessian of g w.r.t. x, J_f the ODE Jacobian,
  //   and df/dt the explicit time derivative of the RHS.
  //
  //   For root functions linear in x (the common PK/PD case),
  //   H_g = 0 and the expression simplifies considerably.
  //
  //   This is a SCALAR (double) function: it only contributes to the
  //   second-order IFT correction coefficient and does not need to
  //   carry AD sensitivities.
  //
  //   If nullptr, G_tt is computed via scalar finite difference as a
  //   fallback (still correct, but slightly less accurate).
  std::function<double(const state_type&, const time_type&)> g_dot_dot;
};

// ============================================================================
// Event tracking
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
// Plain event action (no saltation) — works for any value_type
// ============================================================================

template<class state_type, class time_type>
inline void apply_event_action(
    state_type& x,
    const state_type& x_ref,
    const time_type& t,
    const RootEvent<state_type, time_type>& evt)
{
  const int k = evt.state_index;
  if (k >= 0) {
    auto h = evt.value_func(x_ref, t);
    switch (evt.method) {
    case EventMethod::Replace:  x[k] = h; break;
    case EventMethod::Add:      x[k] = x_ref[k] + h; break;
    case EventMethod::Multiply: x[k] = x_ref[k] * h; break;
    }
  }
}

template<class state_type, class value_type>
inline void apply_event_action_fixed(
    state_type& x,
    const state_type& x_ref,
    const FixedEvent<state_type, value_type>& evt)
{
  const int k = evt.state_index;
  if (k >= 0) {
    auto h = evt.value_func(x_ref, evt.time);
    switch (evt.method) {
    case EventMethod::Replace:  x[k] = h; break;
    case EventMethod::Add:      x[k] = x_ref[k] + h; break;
    case EventMethod::Multiply: x[k] = x_ref[k] * h; break;
    }
  }
}

// ============================================================================
// Analytical saltation correction for root events (AD path)
//
// Based on the Implicit Function Theorem:
//
//   g(x(t*(p), p), t*(p)) = 0
//
//   => dt*/dp = -(dg/dx · dx/dp + dg/dp_expl) / g_dot
//
// where g_dot = dg/dx · f(x, t*) + dg/dt is the total time derivative
// of g along the trajectory.
//
// Evaluating g on the AD state x_before gives g_val.d(j) = dg_total/dp_j
// (chain rule through FADBAD).  Computing g_dot from dg_dx · f + dg_dt
// on the AD types gives g_dot as a full AD type, so that the quotient
//
//   dt* = -g_val / g_dot
//
// applies FADBAD's quotient rule and yields correct d(dt*)/dp for
// first-order, and d²(dt*)/dp_i dp_j for second-order AD types.
//
// State transport uses Heun (trapezoid) shifts rather than Euler, because
// the second-order AD product rule applied to x + f·dt* misses the
// curvature term ½·(J·f + df/dt)·dt*².  Although scalar(dt*) = 0,
// dt*² has non-zero second-order AD components:
//
//   (dt*²).dd(a,b) = 2 · d(dt*)/dp_a · d(dt*)/dp_b
//
// The Heun method implicitly captures this curvature via a second RHS
// evaluation at the Euler-shifted state:
//
//   Forward Heun:
//     f1 = f(x, t),  x_euler = x + f1·dt*,  f2 = f(x_euler, t)
//     x_star = x + ½·(f1 + f2)·dt*
//
//   This equals x + f·dt* + ½·(df/dx)·f·dt*² to second order.
//   Since f2 is evaluated on full AD types at x_euler (which inherits
//   all AD sensitivities), FADBAD's chain rule correctly produces
//   d(f2)/dp including the (df/dx)·(dx/dp) cross-terms needed for
//   second-order accuracy.
//
// Summary of the full correction:
//   1. g_dot from codegen-provided dg_dx, dg_dt  (analytical, full AD)
//   2. dt* = -g / g_dot                          (FADBAD quotient rule)
//   3. Forward Heun shift to event surface        (2nd-order AD accurate)
//   4. Apply event action
//   5. Backward Heun shift to grid time           (2nd-order AD accurate)
// ============================================================================

template<class state_type, class time_type, class System>
inline void saltation_root_analytical(
    state_type& x,
    const state_type& x_before,
    const time_type& t_event,
    System& sys,
    const RootEvent<state_type, time_type>& evt)
{
  using value_type = typename state_type::value_type;
  const size_t n = x.size();
  if (n == 0) return;

  // --- 1. RHS before event (full AD) ---
  state_type f_before(n);
  sys.first(x_before, f_before, t_event);

  // --- 2. Analytical g_dot = sum_i(dg/dx_i · f_i) + dg/dt ---
  //     Computed as full AD type: FADBAD chain rule on dg_dx and f
  //     gives d(g_dot)/dp in the .d() components, which is needed
  //     for second-order dt* via the quotient rule.
  state_type grad_g(n);
  evt.dg_dx(x_before, t_event, grad_g);
  value_type g_dot = evt.dg_dt(x_before, t_event);
  for (size_t i = 0; i < n; ++i)
    g_dot += grad_g[i] * f_before[i];

  double g_dot_s = scalar_value(g_dot);
  if (std::abs(g_dot_s) < 1e-15) {
    // Degenerate (tangential crossing): no meaningful dt*, apply event directly
    apply_event_action(x, x_before, t_event, evt);
    return;
  }

  // --- 3. dt* from the IFT (full AD, quotient rule) ---
  //     g_val has scalar ≈ 0 (root localized by bisection).
  //     Force scalar to exactly 0 so only AD components drive dt*.
  value_type g_val = evt.func(x_before, t_event);
  g_val = g_val - value_type(scalar_value(g_val));
  value_type dt_star = -g_val / g_dot;

  // --- 3b. Second-order IFT correction ---
  //
  //   The FADBAD quotient rule on -g/g_dot yields correct first-order
  //   derivatives dt*.d(a) but MISSES a second-order term.  The full
  //   second-order IFT for g(x(t,p), t, p) = 0 gives:
  //
  //     d²t*/dp_a dp_b = -(1/G_t)[G_{pp} + G_{tp}·dt*/dp_b
  //                       + G_{tp}·dt*/dp_a + G_tt·(dt*/dp_a)(dt*/dp_b)]
  //
  //   FADBAD captures the first three terms but MISSES the G_tt term,
  //   where G_tt = d(g_dot)/dt is the total second time derivative of g
  //   along the trajectory.
  //
  //   The correction exploits the F<F<double>> identity:
  //     (dt*²).dd(a,b) = 2 · dt*.d(a) · dt*.d(b)
  //   so that:
  //     dt*_corrected = dt* - 0.5 · (G_tt / G_t) · dt*²
  //   adds exactly -(G_tt/G_t)·dt*.d(a)·dt*.d(b) to dt*.dd(a,b)
  //   while leaving scalar and first-order derivatives unchanged
  //   (since scalar(dt*) = 0 → scalar(dt*²) = 0, dt*².d(a) = 0).
  //
  //   G_tt is computed analytically from the codegen-provided g_dot_dot
  //   function, or via scalar finite difference as a fallback.
  {
    double G_tt;
    if (evt.g_dot_dot) {
      // Analytical G_tt from codegen (exact, no FD error)
      G_tt = evt.g_dot_dot(x_before, t_event);
    } else {
      // Fallback: scalar FD on g_dot
      constexpr double eps = 1e-8;
      state_type x_fwd(n);
      for (size_t i = 0; i < n; ++i)
        x_fwd[i] = x_before[i] + f_before[i] * value_type(eps);

      state_type f_fwd(n);
      sys.first(x_fwd, f_fwd, t_event + time_type(eps));

      state_type grad_g_fwd(n);
      evt.dg_dx(x_fwd, t_event + time_type(eps), grad_g_fwd);
      value_type g_dot_fwd = evt.dg_dt(x_fwd, t_event + time_type(eps));
      for (size_t i = 0; i < n; ++i)
        g_dot_fwd += grad_g_fwd[i] * f_fwd[i];

      G_tt = (scalar_value(g_dot_fwd) - g_dot_s) / eps;
    }

    double corr_coeff = -0.5 * G_tt / g_dot_s;

    // Apply: dt* += corr_coeff · dt*²
    // Since scalar(dt*) = 0, this only affects second-order AD components.
    dt_star = dt_star + value_type(corr_coeff) * dt_star * dt_star;
  }

  // --- 4. Forward Heun shift to event surface ---
  //     x_star = x + ½·(f(x) + f(x + f(x)·dt*))·dt*
  //     Second-order accurate in AD: captures ½·(df/dx)·f·dt*² curvature.
  state_type x_euler(n);
  for (size_t i = 0; i < n; ++i)
    x_euler[i] = x_before[i] + f_before[i] * dt_star;

  state_type f_euler(n);
  sys.first(x_euler, f_euler, t_event);

  value_type half(0.5);
  state_type x_star(n);
  for (size_t i = 0; i < n; ++i)
    x_star[i] = x_before[i] + half * (f_before[i] + f_euler[i]) * dt_star;

  // --- 5. Apply event action at the event surface ---
  state_type x_after(n);
  for (size_t i = 0; i < n; ++i) x_after[i] = x_star[i];
  const int k = evt.state_index;
  if (k >= 0) {
    value_type h = evt.value_func(x_star, t_event);
    switch (evt.method) {
    case EventMethod::Replace:  x_after[k] = h; break;
    case EventMethod::Add:      x_after[k] = x_star[k] + h; break;
    case EventMethod::Multiply: x_after[k] = x_star[k] * h; break;
    }
  }

  // --- 6. Backward Heun shift to grid time ---
  //     x_final = x_after - ½·(f(x_after) + f(x_after - f(x_after)·dt*))·dt*
  state_type f_after(n);
  sys.first(x_after, f_after, t_event);

  state_type x_back(n);
  for (size_t i = 0; i < n; ++i)
    x_back[i] = x_after[i] - f_after[i] * dt_star;

  state_type f_back(n);
  sys.first(x_back, f_back, t_event);

  for (size_t i = 0; i < n; ++i)
    x[i] = x_after[i] - half * (f_after[i] + f_back[i]) * dt_star;
}

// ============================================================================
// Analytical saltation correction for fixed-time events (AD path)
//
// For a fixed event at time t_event(p), the integrator evaluates at
// scalar(t_event).  The AD residual
//
//   dt_corr = t_event - scalar(t_event)
//
// has scalar 0 and carries dt_event/dp_j in its AD components.
// The same Heun forward–event–backward scheme applies for second-order
// accuracy (see saltation_root_analytical for the detailed rationale).
// ============================================================================

template<class state_type, class System>
inline void saltation_fixed_analytical(
    state_type& x,
    const state_type& x_before,
    System& sys,
    const FixedEvent<state_type, typename state_type::value_type>& evt)
{
  using value_type = typename state_type::value_type;
  const size_t n = x.size();
  if (n == 0) return;

  value_type dt_corr = evt.time - value_type(scalar_value(evt.time));
  value_type half(0.5);

  // --- 1. Forward Heun shift to true event time ---
  state_type f1(n);
  sys.first(x_before, f1, evt.time);

  state_type x_euler(n);
  for (size_t i = 0; i < n; ++i)
    x_euler[i] = x_before[i] + f1[i] * dt_corr;

  state_type f2(n);
  sys.first(x_euler, f2, evt.time);

  state_type x_star(n);
  for (size_t i = 0; i < n; ++i)
    x_star[i] = x_before[i] + half * (f1[i] + f2[i]) * dt_corr;

  // --- 2. Apply event action ---
  state_type x_after(n);
  for (size_t i = 0; i < n; ++i) x_after[i] = x_star[i];
  const int k = evt.state_index;
  if (k >= 0) {
    value_type h = evt.value_func(x_star, evt.time);
    switch (evt.method) {
    case EventMethod::Replace:  x_after[k] = h; break;
    case EventMethod::Add:      x_after[k] = x_star[k] + h; break;
    case EventMethod::Multiply: x_after[k] = x_star[k] * h; break;
    }
  }

  // --- 3. Backward Heun shift to grid time ---
  state_type g1(n);
  sys.first(x_after, g1, evt.time);

  state_type x_back(n);
  for (size_t i = 0; i < n; ++i)
    x_back[i] = x_after[i] - g1[i] * dt_corr;

  state_type g2(n);
  sys.first(x_back, g2, evt.time);

  for (size_t i = 0; i < n; ++i)
    x[i] = x_after[i] - half * (g1[i] + g2[i]) * dt_corr;
}

// ============================================================================
// Apply fixed events at a given time — SFINAE dispatch
//
// double path:  plain event action (no saltation needed)
// AD path:      analytical saltation correction
// ============================================================================

template<class state_type, class Time, class System>
bool apply_fixed_events_at_time(
    state_type& x,
    const Time& t,
    const std::vector<FixedEvent<state_type, typename state_type::value_type>>& evs,
    System& sys)
{
  using value_type = typename state_type::value_type;
  const double tt = scalar_value(t);
  bool fired = false;

  for (const auto& e : evs) {
    if (std::abs(scalar_value(e.time) - tt) < 1e-14) {
      if constexpr (std::is_arithmetic_v<value_type>) {
        // double: plain event, no sensitivity correction
        apply_event_action_fixed(x, x, e);
      } else {
        // AD: analytical saltation
        state_type x_before = x;
        saltation_fixed_analytical(x, x_before, sys, e);
      }
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
  for (const auto& e : fix) out.push_back(e.time);

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
 } else if constexpr (has_reinitialize_at_event<S, State, Time>::value) {
   st.reinitialize_at_event(x, t, dt);
 } else {
   (void)st; (void)x; (void)t; (void)dt;
 }
}

// ============================================================================
// EventEngine
// ============================================================================

template<class Stepper, class System, class State, class Time>
class EventEngine {
public:
 using state_type = State;
 using time_type  = Time;
 using value_type = typename State::value_type;

 EventEngine(Stepper& st, System& sys,
             const std::vector<FixedEvent<State, value_type>>& fixed,
             const std::vector<RootEvent<State, Time>>& root)
   : m_st(st), m_sys(sys), m_fixed(fixed), m_root(root) {}

private:
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
       triggered.push_back({i, last_val[i], curr_val[i]});
   }
 }

 // --------------------------------------------------------------------------
 // Apply root events with SFINAE-dispatched saltation correction
 //
 // double:  plain event action, no sensitivity correction
 // AD/AD2:  analytical IFT-based saltation (requires dg_dx, dg_dt)
 // --------------------------------------------------------------------------

 bool apply_root_events(
     State& x_root, const State& x_before, const Time& t_root,
     const std::vector<TriggeredEvent>& triggered,
     std::vector<size_t>& fired)
 {
   bool has_terminal = false;
   for (const auto& te : triggered)
     if (m_root[te.index].terminal) { has_terminal = true; break; }

     for (const auto& te : triggered) {
       size_t i = te.index;

       if (m_root[i].terminal) {
         fired[i]++;
         continue;
       }

       if constexpr (std::is_arithmetic_v<value_type>) {
         // double: plain event action
         apply_event_action(x_root, x_before, t_root, m_root[i]);
       } else {
         // AD path: analytical saltation if gradients are available
         if (m_root[i].dg_dx && m_root[i].dg_dt) {
           saltation_root_analytical(x_root, x_before, t_root, m_sys, m_root[i]);
         } else {
           // Fallback: apply event without saltation correction
           // (e.g. steady-state terminal events, or missing gradients)
           apply_event_action(x_root, x_before, t_root, m_root[i]);
         }
       }

       fired[i]++;
     }
     return has_terminal;
 }

 template<class Checker>
 void reinit_after_event(
     State& x, const Time& t_event, Time& dt,
     Time& t_start, Time& t_end, State& x_at_start,
     std::vector<double>& last_val,
     std::vector<State>& last_state, std::vector<Time>& last_time,
     size_t& steps, Checker& checker)
 {
   m_st.initialize(x, t_event, dt);
   m_st.do_step(m_sys);
   ++steps; checker(); checker.reset();

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

 void eval_root_funcs(std::vector<double>& cv, const State& x, const Time& t) {
   for (size_t i = 0; i < m_root.size(); ++i)
     cv[i] = scalar_value(m_root[i].func(x, t));
 }

public:
 // ========================================================================
 // Controlled stepper
 // ========================================================================
 template<class Obs, class Checker>
 size_t process_controlled(
     state_type& x, const std::vector<Time>& times, Time dt,
     Obs& obs, Checker& checker, double root_tol, size_t max_trigger)
 {
   size_t steps = 0;
   auto it = times.begin();
   const auto end = times.end();
   Time t = *it;

   if (apply_fixed_events_at_time(x, t, m_fixed, m_sys))
     reset_stepper_unified(m_st, x, t, dt);

   obs(x, t); ++it;
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
     double t_target_s = scalar_value(t_target);

     while (scalar_value(t) < t_target_s - 1e-14) {
       double rem = t_target_s - scalar_value(t);
       Time dt_step = (scalar_value(dt) < rem) ? dt : Time(rem);
       auto result = m_st.try_step(m_sys, x, t, dt_step);

       if (result == success) {
         ++steps; checker(); checker.reset(); dt = dt_step;
         eval_root_funcs(curr_val, x, t);
         check_root_triggers(triggered, last_val, curr_val, fired, max_trigger);

         if (!triggered.empty()) {
           localize_root_controlled(
             triggered[0].index,
             last_state[triggered[0].index], last_time[triggered[0].index],
                                                      x, t, triggered[0].last_val, triggered[0].curr_val,
                                                      root_tol, checker);

           State x_before = x;
           Time t_before = t - Time(1e-15);
           obs(x_before, t_before);

           State x_after = x;
           if (apply_root_events(x_after, x, t, triggered, fired)) {
             obs(x_after, t); x = x_after; return steps;
           }
           obs(x_after, t); x = x_after;
           reset_stepper_unified(m_st, x, t, dt);

           for (size_t j = 0; j < m_root.size(); ++j) {
             last_val[j] = std::numeric_limits<double>::quiet_NaN();
             last_state[j] = x; last_time[j] = t;
           }
         } else {
           for (size_t i = 0; i < m_root.size(); ++i) {
             last_val[i] = curr_val[i]; last_state[i] = x; last_time[i] = t;
           }
         }
       } else { checker(); dt = dt_step; }
     }

     t = t_target;
     if (apply_fixed_events_at_time(x, t, m_fixed, m_sys)) {
       obs(x, t);
       reset_stepper_unified(m_st, x, t, dt);
       for (size_t j = 0; j < m_root.size(); ++j) {
         last_val[j] = std::numeric_limits<double>::quiet_NaN();
         last_state[j] = x; last_time[j] = t; fired[j] = 0;
       }
     } else { obs(x, t); }
     ++it;
   }
   return steps;
 }

 // ========================================================================
 // Dense output stepper
 // ========================================================================
 template<class Obs, class Checker>
 size_t process_dense(
     state_type& x, const std::vector<Time>& times, Time dt,
     Obs& obs, Checker& checker, double root_tol, size_t max_trigger)
 {
   using boost::numeric::odeint::detail::less_eq_with_sign;
   size_t steps = 0;
   auto it = times.begin(); auto end = times.end();

   if (apply_fixed_events_at_time(x, *it, m_fixed, m_sys))
     m_st.initialize(x, *it, dt);
   obs(x, *it); ++it;
   if (it == end) return 0;

   m_st.initialize(x, times.front(), dt);
   m_st.do_step(m_sys); ++steps; checker(); checker.reset();

   Time t_start = m_st.previous_time();
   Time t_end = m_st.current_time();
   dt = m_st.current_time_step();

   State x_at_start(x.size());
   m_st.calc_state(t_start, x_at_start);

   std::vector<double> last_val(m_root.size());
   std::vector<State>  last_state(m_root.size(), x_at_start);
   std::vector<Time>   last_time(m_root.size(), t_start);
   std::vector<size_t> fired(m_root.size(), 0);
   for (size_t i = 0; i < m_root.size(); ++i)
     last_val[i] = scalar_value(m_root[i].func(x_at_start, t_start));

   std::vector<TriggeredEvent> triggered;
   triggered.reserve(m_root.size());
   std::vector<double> curr_val(m_root.size());

   while (it != end) {
     while (!less_eq_with_sign(*it, t_end, dt)) {
       m_st.calc_state(t_end, x);
       eval_root_funcs(curr_val, x, t_end);
       check_root_triggers(triggered, last_val, curr_val, fired, max_trigger);

       if (!triggered.empty()) {
         State x_root = x_at_start; Time t_root = t_start;
         localize_root_dense(triggered[0].index, x_root, t_root, t_end,
                             triggered[0].last_val, triggered[0].curr_val, root_tol);
         State x_before = x_root;
         Time t_before = t_root - Time(1e-15);
         obs(x_before, t_before);
         if (apply_root_events(x_root, x_before, t_root, triggered, fired)) {
           obs(x_root, t_root); x = x_root; return steps;
         }
         obs(x_root, t_root); x = x_root;
         reinit_after_event(x, t_root, dt, t_start, t_end, x_at_start,
                            last_val, last_state, last_time, steps, checker);
         continue;
       }

       for (size_t i = 0; i < m_root.size(); ++i) {
         last_val[i] = curr_val[i]; last_state[i] = x; last_time[i] = t_end;
       }
       m_st.do_step(m_sys); ++steps; checker(); checker.reset();
       t_start = m_st.previous_time(); t_end = m_st.current_time();
       dt = m_st.current_time_step();
       m_st.calc_state(t_start, x_at_start);
       for (size_t j = 0; j < m_root.size(); ++j) {
         last_val[j] = scalar_value(m_root[j].func(x_at_start, t_start));
         last_state[j] = x_at_start; last_time[j] = t_start;
       }
     }

     while (it != end && less_eq_with_sign(*it, t_end, dt)) {
       Time t_eval = *it;
       if (scalar_value(t_eval) < scalar_value(t_start)) {
         m_st.calc_state(t_start, x); obs(x, t_eval); ++it; continue;
       }
       Time t_eval_s = Time(scalar_value(t_eval));
       m_st.calc_state(t_eval_s, x);
       eval_root_funcs(curr_val, x, t_eval);
       check_root_triggers(triggered, last_val, curr_val, fired, max_trigger);

       if (!triggered.empty()) {
         State x_root = last_state[triggered[0].index];
         Time t_root = last_time[triggered[0].index];
         localize_root_dense(triggered[0].index, x_root, t_root, t_eval,
                             triggered[0].last_val, triggered[0].curr_val, root_tol);
         State x_before = x_root;
         Time t_before = t_root - Time(1e-15);
         if (std::abs(scalar_value(t_eval) - scalar_value(t_before)) >= 1e-14)
           obs(x_before, t_before);
         if (apply_root_events(x_root, x_before, t_root, triggered, fired)) {
           if (std::abs(scalar_value(t_eval) - scalar_value(t_root)) >= 1e-14)
             obs(x_root, t_root);
           x = x_root; return steps;
         }
         if (std::abs(scalar_value(t_eval) - scalar_value(t_root)) >= 1e-14)
           obs(x_root, t_root);
         x = x_root;
         reinit_after_event(x, t_root, dt, t_start, t_end, x_at_start,
                            last_val, last_state, last_time, steps, checker);
         break;
       }

       bool fef = apply_fixed_events_at_time(x, t_eval, m_fixed, m_sys);
       obs(x, t_eval); ++it;

       if (fef) {
         m_st.initialize(x, t_eval_s, dt);
         m_st.do_step(m_sys); ++steps; checker(); checker.reset();
         t_start = m_st.previous_time(); t_end = m_st.current_time();
         dt = m_st.current_time_step();
         State x_ns = x; m_st.calc_state(t_start, x_ns);
         for (size_t j = 0; j < m_root.size(); ++j) {
           last_val[j] = scalar_value(m_root[j].func(x_ns, t_start));
           last_state[j] = x_ns; last_time[j] = t_start; fired[j] = 0;
         }
         break;
       }
       for (size_t i = 0; i < m_root.size(); ++i) {
         last_val[i] = curr_val[i]; last_state[i] = x; last_time[i] = t_eval;
       }
     }
   }
   return steps;
 }

private:
 template<class Checker>
 void localize_root_controlled(
     size_t idx, State& x_lo, Time& t_lo, State& x_hi, Time& t_hi,
     double g_lo, double g_hi, double tol, Checker& checker)
 {
   for (int iter = 0; iter < 50; ++iter) {
     double dti = scalar_value(t_hi) - scalar_value(t_lo);
     if (dti < tol) break;
     double alpha = std::max(0.1, std::min(0.9, -g_lo / (g_hi - g_lo)));
     Time t_mid = t_lo + Time(alpha * dti);
     State x_mid = x_lo; Time t_tmp = t_lo; Time dt_tmp = t_mid - t_lo;
     while (scalar_value(t_tmp) < scalar_value(t_mid) - 1e-15) {
       double r = scalar_value(t_mid) - scalar_value(t_tmp);
       Time sd = (scalar_value(dt_tmp) < r) ? dt_tmp : Time(r);
       if (m_st.try_step(m_sys, x_mid, t_tmp, sd) == success) {
         checker(); checker.reset();
       }
       dt_tmp = sd;
     }
     double g_mid = scalar_value(m_root[idx].func(x_mid, t_mid));
     if (g_lo * g_mid < 0.0) { x_hi = x_mid; t_hi = t_mid; g_hi = g_mid; }
     else { x_lo = x_mid; t_lo = t_mid; g_lo = g_mid; }
   }
   x_hi = x_lo; t_hi = t_lo;
 }

 void localize_root_dense(
     size_t idx, State& x_root, Time& t_root, Time t_hi,
     double g_lo, double g_hi, double tol)
 {
   Time t_lo = t_root;
   for (int iter = 0; iter < 50; ++iter) {
     double dti = scalar_value(t_hi) - scalar_value(t_lo);
     if (dti < tol) break;
     double alpha = std::max(0.1, std::min(0.9, -g_lo / (g_hi - g_lo)));
     Time t_mid = t_lo + Time(alpha * dti);
     State x_mid(x_root.size()); m_st.calc_state(t_mid, x_mid);
     double g_mid = scalar_value(m_root[idx].func(x_mid, t_mid));
     if (g_lo * g_mid < 0.0) { t_hi = t_mid; g_hi = g_mid; }
     else { t_lo = t_mid; g_lo = g_mid; x_root = x_mid; t_root = t_mid; }
   }
   m_st.calc_state(t_lo, x_root); t_root = t_lo;
 }

 Stepper& m_st; System& m_sys;
 const std::vector<FixedEvent<State, value_type>>& m_fixed;
 const std::vector<RootEvent<State, Time>>& m_root;
};

// ============================================================================
// Public API
// ============================================================================

template<class Stepper, class System, class State,
        class TimeIterator, class Time, class Observer>
size_t integrate_times(
   Stepper stepper, System system, State& x,
   TimeIterator t_begin, TimeIterator t_end, Time dt, Observer obs,
   const std::vector<FixedEvent<State, typename State::value_type>>& fixed,
   const std::vector<RootEvent<State, Time>>& root,
   StepChecker& checker, double root_tol = 1e-8, size_t max_trigger_root = 1,
   boost::numeric::odeint::controlled_stepper_tag = boost::numeric::odeint::controlled_stepper_tag())
{
 auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);
 EventEngine<Stepper, System, State, Time> eng(stepper, system, fixed, root);
 return eng.process_controlled(x, times, dt, obs, checker, root_tol, max_trigger_root);
}

template<class Stepper, class System, class State,
        class TimeIterator, class Time, class Observer>
size_t integrate_times_dense(
   Stepper stepper, System system, State& x,
   TimeIterator t_begin, TimeIterator t_end, Time dt, Observer obs,
   const std::vector<FixedEvent<State, typename State::value_type>>& fixed,
   const std::vector<RootEvent<State, Time>>& root,
   StepChecker& checker, double root_tol = 1e-8, size_t max_trigger_root = 1,
   boost::numeric::odeint::dense_output_stepper_tag = boost::numeric::odeint::dense_output_stepper_tag())
{
 auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);
 EventEngine<Stepper, System, State, Time> eng(stepper, system, fixed, root);
 return eng.process_dense(x, times, dt, obs, checker, root_tol, max_trigger_root);
}

} // namespace detail

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
