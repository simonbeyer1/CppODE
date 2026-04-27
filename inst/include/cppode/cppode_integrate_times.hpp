/*
 * Event-aware integration at specified times with root-finding support.
 *
 * The basic integrate_times structure is derived from Boost.Odeint
 * by Mario Mulansky, Karsten Ahnert, and Christoph Koke (2011-2015),
 * distributed under the Boost Software License, Version 1.0.
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

#ifndef CPPODE_INTEGRATE_TIMES_HPP
#define CPPODE_INTEGRATE_TIMES_HPP

#include <vector>
#include <functional>
#include <cmath>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <memory>

#include <cppode/cppode_step_checker.hpp>
#include <cppode/cppode_stepper_traits.hpp>
#include <cppode/cppode_profiler.hpp>
#include <cppode/cppode_utils.hpp>

#include <fadbad++/fadiff.h>
#include <cppode/cppode_fadiff_extensions.hpp>
#include <cppode/cppode_ad_traits.hpp>

namespace cppode {
namespace detail {

// ============================================================================
// Scalar value extraction & AD type detection — from cppode_ad_traits.hpp
// ============================================================================

using cppode::ad_traits::scalar_value;
template<class T> using is_ad_type = cppode::ad_traits::is_ad<T>;

// ============================================================================
// Diagnostics helpers — extract order and transfer counters from steppers
//
// All stepper/controller/dense-output classes expose a uniform interface:
//   n_accepted(), n_rejected(), n_fevals(), n_jevals(), current_method_order()
//
// get_stepper_order:
//   Extracts the current method order for set_last_order (per-step).
//   BDF: variable (1–5), Rosenbrock4: fixed 4.
//
// transfer_stepper_diagnostics:
//   Called once after integration completes.  Copies the exact counters
//   accumulated inside the stepper hierarchy into the StepChecker.
// ============================================================================

// --- Extract current method order (used per accepted step) ---

template<class T, class = void>
struct has_method_order : std::false_type {};

template<class T>
struct has_method_order<T, std::void_t<
  decltype(std::declval<const T&>().current_method_order())>>
  : std::true_type {};

template<class Stepper>
inline auto get_stepper_order(const Stepper& st)
  -> std::enable_if_t<has_method_order<Stepper>::value, int>
  {
    return st.current_method_order();
  }

// Fallback for unknown steppers
template<class Stepper>
inline auto get_stepper_order(const Stepper&)
  -> std::enable_if_t<!has_method_order<Stepper>::value, int>
  {
    return 0;
  }

// --- Transfer exact counters from stepper into StepChecker (called once) ---

template<class T, class = void>
struct has_diagnostics_counters : std::false_type {};

template<class T>
struct has_diagnostics_counters<T, std::void_t<
  decltype(std::declval<const T&>().n_accepted()),
  decltype(std::declval<const T&>().n_rejected()),
  decltype(std::declval<const T&>().n_fevals()),
  decltype(std::declval<const T&>().n_jevals())>>
    : std::true_type {};

// --- Profiler report dispatch (SFINAE) ---
// Must be declared before transfer_stepper_diagnostics which calls it.
template<class T, class = void>
struct has_controlled_stepper_method : std::false_type {};

template<class T>
struct has_controlled_stepper_method<T, std::void_t<
 decltype(std::declval<const T&>().controlled_stepper())>>
 : std::true_type {};

template<class T, class = void>
struct has_report_profiler_method : std::false_type {};

template<class T>
struct has_report_profiler_method<T, std::void_t<
 decltype(std::declval<const T&>().report_profiler())>>
 : std::true_type {};

template<class Stepper>
inline void report_profiler_if_available(const Stepper& st) {
 if constexpr (has_controlled_stepper_method<Stepper>::value) {
   auto& ctrl = st.controlled_stepper();
   if constexpr (has_report_profiler_method<std::decay_t<decltype(ctrl)>>::value) {
     ctrl.report_profiler();
   }
 } else if constexpr (has_report_profiler_method<Stepper>::value) {
   st.report_profiler();
 }
}

template<class T, class = void>
struct has_n_setups_method : std::false_type {};

template<class T>
struct has_n_setups_method<T, std::void_t<
  decltype(std::declval<const T&>().n_setups())>>
  : std::true_type {};

template<class Stepper, class Checker>
inline auto transfer_stepper_diagnostics(const Stepper& st, Checker& checker)
  -> std::enable_if_t<has_diagnostics_counters<Stepper>::value>
  {
    checker.add_accepted(st.n_accepted());
    checker.add_rejected(st.n_rejected());
    checker.add_fevals(st.n_fevals());
    checker.add_jevals(st.n_jevals());
    if constexpr (has_n_setups_method<Stepper>::value) {
      checker.add_setups(st.n_setups());
    }
    checker.set_last_order(get_stepper_order(st));
    report_profiler_if_available(st);
  }

template<class Stepper, class Checker>
inline auto transfer_stepper_diagnostics(const Stepper&, Checker&)
  -> std::enable_if_t<!has_diagnostics_counters<Stepper>::value>
  {
  }

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
// Steady-state termination helper (threshold check, no root-finding)
// ============================================================================

template<class System, class State, class Time>
std::function<bool(const State&, const Time&)> make_steady_state_termination(System& sys, double tol) {
  return [&sys, tol](const State& x, const Time& t) -> bool {
    State dxdt(x.size());
    sys(x, dxdt, t);
    return cppode::max_abs_all_levels_vec(dxdt) < tol;
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
// evaluation at the Euler-shifted state AND time:
//
//   Forward Heun:
//     f1 = f(x, t),  x_euler = x + f1·dt*,  f2 = f(x_euler, t + dt*)
//     x_star = x + ½·(f1 + f2)·dt*
//
//   This equals x + f·dt* + ½·(df/dx·f + df/dt)·dt*² to second order.
//   The time shift in f2 is essential for non-autonomous systems: without
//   it, the ½·(df/dt)·dt*² term is lost.  While this term has zero
//   first-order AD components (scalar(dt*)=0 => (dt*²).d(j)=0), its
//   second-order AD components are non-trivial:
//     (dt*²).dd(a,b) = 2·d(dt*)/dp_a·d(dt*)/dp_b
//   Omitting the time shift therefore corrupts second-order sensitivities
//   for any ODE with explicit time dependence.
//
// Summary of the full correction:
//   1. g_dot from codegen-provided dg_dx, dg_dt  (analytical, full AD)
//   2. dt* = -g / g_dot                          (FADBAD quotient rule)
//   3. Forward Heun shift to event surface        (2nd-order AD accurate)
//   4. Apply ALL simultaneous event actions
//   5. Backward Heun shift to grid time           (2nd-order AD accurate)
//
// Batch variant: when multiple root events trigger simultaneously (same
// root function or coincident crossings), the Heun forward/backward
// roundtrip is performed ONCE and all event actions are applied at the
// event surface in between.  This is both correct (all events share the
// same dt* and event surface) and efficient (4 RHS evaluations total,
// independent of the number of simultaneous events).
// ============================================================================

// --- Helper: compute dt* for a root event (IFT + 2nd-order correction) ---
template<class state_type, class time_type, class System>
inline typename state_type::value_type compute_dt_star(
    const state_type& x_before,
    const time_type& t_event,
    System& sys,
    const state_type& f_before,
    const RootEvent<state_type, time_type>& evt)
{
  using value_type = typename state_type::value_type;
  const size_t n = x_before.size();

  // Analytical g_dot = sum_i(dg/dx_i · f_i) + dg/dt
  state_type grad_g(n);
  evt.dg_dx(x_before, t_event, grad_g);
  value_type g_dot = evt.dg_dt(x_before, t_event);
  for (size_t i = 0; i < n; ++i)
    g_dot += grad_g[i] * f_before[i];

  double g_dot_s = scalar_value(g_dot);
  if (std::abs(g_dot_s) < 1e-15) {
    return value_type(0.0);
  }

  // dt* from IFT (FADBAD quotient rule)
  value_type g_val = evt.func(x_before, t_event);
  g_val = g_val - value_type(scalar_value(g_val));
  value_type dt_star = -g_val / g_dot;

  // Second-order IFT correction
  {
    double G_tt;
    if (evt.g_dot_dot) {
      G_tt = evt.g_dot_dot(x_before, t_event);
    } else {
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
    dt_star = dt_star + value_type(corr_coeff) * dt_star * dt_star;
  }

  return dt_star;
}

// --- Batch saltation: one Heun roundtrip, N event actions in the middle ---
//
// All triggered events must share the same root function (same dt*).
// The first event with valid dg_dx/dg_dt is used to compute dt*.
// Steps:
//   1. Compute f_before, dt* (once)
//   2. Forward Heun shift to event surface → x_star (once)
//   3. Apply ALL event actions on x_star
//   4. Backward Heun shift to grid time → x (once)
//
// Cost: 4 RHS evaluations, independent of number of events.

template<class state_type, class time_type, class System>
inline void saltation_root_analytical_batch(
    state_type& x,
    const state_type& x_before,
    const time_type& t_event,
    System& sys,
    const std::vector<RootEvent<state_type, time_type>>& root_events,
    const std::vector<TriggeredEvent>& triggered)
{
  using value_type = typename state_type::value_type;
  const size_t n = x.size();
  if (n == 0) return;

  // --- 1. RHS before event (full AD) ---
  state_type f_before(n);
  sys.first(x_before, f_before, t_event);

  // --- 2. Compute dt* from the first event with valid gradients ---
  value_type dt_star(0.0);
  bool have_dt_star = false;
  for (const auto& te : triggered) {
    const auto& evt = root_events[te.index];
    if (evt.terminal) continue;
    if (evt.dg_dx && evt.dg_dt) {
      state_type grad_g(n);
      evt.dg_dx(x_before, t_event, grad_g);
      value_type g_dot = evt.dg_dt(x_before, t_event);
      for (size_t i = 0; i < n; ++i)
        g_dot += grad_g[i] * f_before[i];

      if (std::abs(scalar_value(g_dot)) >= 1e-15) {
        dt_star = compute_dt_star(x_before, t_event, sys, f_before, evt);
        have_dt_star = true;
      }
      break;
    }
  }

  if (!have_dt_star) {
    for (const auto& te : triggered) {
      if (!root_events[te.index].terminal) {
        apply_event_action(x, x_before, t_event, root_events[te.index]);
      }
    }
    return;
  }

  // --- 3. Forward Heun shift to event surface ---
  //     f2 must be evaluated at t_event + dt_star (not t_event) so that
  //     the ½·(df/dt)·dt*² curvature term is captured for non-autonomous
  //     systems.  scalar(dt_star)≈0 so the scalar time is unchanged, but
  //     the AD components carry d(t*)/dp which couple at second order.
  state_type x_euler(n);
  for (size_t i = 0; i < n; ++i)
    x_euler[i] = x_before[i] + f_before[i] * dt_star;

  time_type t_star = t_event + dt_star;
  state_type f_euler(n);
  sys.first(x_euler, f_euler, t_star);

  value_type half(0.5);
  state_type x_star(n);
  for (size_t i = 0; i < n; ++i)
    x_star[i] = x_before[i] + half * (f_before[i] + f_euler[i]) * dt_star;

  // --- 4. Apply ALL event actions at the event surface ---
  state_type x_after(n);
  for (size_t i = 0; i < n; ++i) x_after[i] = x_star[i];

  for (const auto& te : triggered) {
    const auto& evt = root_events[te.index];
    if (evt.terminal) continue;
    const int k = evt.state_index;
    if (k >= 0) {
      value_type h = evt.value_func(x_star, t_event);
      switch (evt.method) {
      case EventMethod::Replace:  x_after[k] = h; break;
      case EventMethod::Add:      x_after[k] = x_star[k] + h; break;
      case EventMethod::Multiply: x_after[k] = x_star[k] * h; break;
      }
    }
  }

  // --- 5. Backward Heun shift to grid time ---
  //     x_after lives at t* = t_event + dt_star.  We transport backward
  //     by -dt_star.  f_after at departure (t_star), f_back at arrival (t_event).
  state_type f_after(n);
  sys.first(x_after, f_after, t_star);

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
  value_type t_grid = value_type(scalar_value(evt.time));

  // --- 1. Forward Heun shift to true event time ---
  //     x_before lives at t_grid.  f1 at departure (t_grid),
  //     f2 at arrival (evt.time).
  state_type f1(n);
  sys.first(x_before, f1, t_grid);

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
  //     x_after lives at evt.time.  g1 at departure (evt.time),
  //     g2 at arrival (t_grid).
  state_type g1(n);
  sys.first(x_after, g1, evt.time);

  state_type x_back(n);
  for (size_t i = 0; i < n; ++i)
    x_back[i] = x_after[i] - g1[i] * dt_corr;

  state_type g2(n);
  sys.first(x_back, g2, t_grid);

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
        apply_event_action_fixed(x, x, e);
      } else {
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
// Unified stepper reset (C++17 if-constexpr dispatch)
//
// Priority:
//   1. Multistep methods (BDF): restart_from_order1() — discard history,
//      restart at order 1.  Detected via cppode::stepper_traits.
//   2. Dense-output steppers: reinitialize_at_event() — reset both
//      state buffers and controller.
//   3. Controlled steppers: reset_after_event() — reset PI controller.
//   4. Fallback: no-op (plain steppers without event support).
// ============================================================================

template<class S, class State, class Time, class = void>
struct has_reinitialize_at_event : std::false_type {};

template<class S, class State, class Time>
struct has_reinitialize_at_event<S, State, Time,
                                std::void_t<decltype(std::declval<S&>().reinitialize_at_event(
                                    std::declval<State&>(), std::declval<Time>(), std::declval<Time&>()))>
> : std::true_type {};

template<class S, class State, class Time, class = void>
struct has_restart_from_order1 : std::false_type {};

template<class S, class State, class Time>
struct has_restart_from_order1<S, State, Time,
                               std::void_t<decltype(std::declval<S&>().restart_from_order1(
                                   std::declval<State&>(), std::declval<Time>(), std::declval<Time&>()))>
> : std::true_type {};

template<class S, class Time, class = void>
struct has_reset_after_event : std::false_type {};

template<class S, class Time>
struct has_reset_after_event<S, Time,
                            std::void_t<decltype(std::declval<S&>().reset_after_event(
                                std::declval<Time>()))>
> : std::true_type {};

template<class S, class State, class Time>
inline void reset_stepper_unified(S& st, State& x, Time t, Time& dt) {
 if constexpr (::cppode::needs_restart_after_event_v<S>) {
   if constexpr (has_restart_from_order1<S, State, Time>::value) {
     st.restart_from_order1(x, t, dt);
   } else if constexpr (has_reinitialize_at_event<S, State, Time>::value) {
     st.reinitialize_at_event(x, t, dt);
   }
 } else if constexpr (has_reset_after_event<S, Time>::value) {
   st.reset_after_event(dt);
 } else if constexpr (has_reinitialize_at_event<S, State, Time>::value) {
   st.reinitialize_at_event(x, t, dt);
 } else {
   (void)st; (void)x; (void)t; (void)dt;
 }
}

// ============================================================================
// no_dt_estimator — sentinel type: "use dt as-is (no re-estimation)"
// ============================================================================

struct no_dt_estimator {};

// ============================================================================
// EventEngine
// ============================================================================

template<class Stepper, class System, class State, class Time,
        class DtEstimator = no_dt_estimator>
class EventEngine {
public:
 using state_type = State;
 using time_type  = Time;
 using value_type = typename State::value_type;

 using TerminationFunc = std::function<bool(const State&, const Time&)>;

 EventEngine(Stepper& st, System& sys,
             const std::vector<FixedEvent<State, value_type>>& fixed,
             const std::vector<RootEvent<State, Time>>& root,
             DtEstimator dt_est = DtEstimator())
   : m_st(st), m_sys(sys), m_fixed(fixed), m_root(root),
     m_dt_estimator(std::move(dt_est)) {}

 void set_termination(TerminationFunc f) { m_termination = std::move(f); }

 cppode::profiler& get_profiler() const {
   if constexpr (has_controlled_stepper_method<Stepper>::value) {
     return m_st.controlled_stepper().m_prof;
   } else {
     return m_st.m_prof;
   }
 }

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
 // --------------------------------------------------------------------------

 bool apply_root_events(
     State& x_root, const State& x_before, const Time& t_root,
     const std::vector<TriggeredEvent>& triggered,
     std::vector<size_t>& fired)
 {
   bool has_terminal = false;
   for (const auto& te : triggered)
     if (m_root[te.index].terminal) { has_terminal = true; break; }

     if constexpr (std::is_arithmetic_v<value_type>) {
       for (const auto& te : triggered) {
         size_t i = te.index;
         if (m_root[i].terminal) { fired[i]++; continue; }
         apply_event_action(x_root, x_before, t_root, m_root[i]);
         fired[i]++;
       }
     } else {
       bool has_gradients = false;
       bool has_non_terminal = false;
       for (const auto& te : triggered) {
         if (m_root[te.index].terminal) continue;
         has_non_terminal = true;
         if (m_root[te.index].dg_dx && m_root[te.index].dg_dt) {
           has_gradients = true;
           break;
         }
       }

       if (has_non_terminal && has_gradients) {
         saltation_root_analytical_batch(x_root, x_before, t_root, m_sys,
                                         m_root, triggered);
       } else if (has_non_terminal) {
         for (const auto& te : triggered) {
           if (!m_root[te.index].terminal) {
             apply_event_action(x_root, x_before, t_root, m_root[te.index]);
           }
         }
       }

       for (const auto& te : triggered) {
         fired[te.index]++;
       }
     }
     return has_terminal;
 }

 // ----------------------------------------------------------------
 // init_stepper_after_event
 //
 // Multistep methods (bdf/adams) discard their Nordsieck history
 // at order 1 after an event, so the pre-event step size is no longer
 // a meaningful starting point — it was tuned to the previous regime
 // and pre-event Nordsieck data, neither of which survives the
 // restart.  Reusing it drives the Newton corrector into repeated
 // convergence failures that collapse h down to the HMIN guard.
 //
 // Re-estimate h0 from scratch via cppode_hin, using the remaining
 // interval (t_now → t_final) as the upper-bound hint — same strategy
 // the multistepper controller uses on its order-1 fallback path.
 // ----------------------------------------------------------------
 void init_stepper_after_event(State& x, Time t, Time& dt) {
   if constexpr (::cppode::needs_restart_after_event_v<Stepper>) {
     if constexpr (!std::is_same_v<DtEstimator, no_dt_estimator>) {
       dt = m_dt_estimator(x, t);
     } else {
       const double atol = m_st.controlled_stepper().atol();
       const double rtol = m_st.controlled_stepper().rtol();
       const double h_est = odeint_utils::cppode_hin<value_type>(
           m_sys.first, x, t, m_t_final, atol, rtol);
       dt = Time(h_est);
     }
     m_st.initialize(x, t, dt);
     State f0(x.size());
     m_sys.first(x, f0, t);
     m_st.controlled_stepper().stepper().initialize(x, t, f0, dt);
     m_st.controlled_stepper().reset_after_event(dt);
   } else {
     m_st.initialize(x, t, dt);
   }
 }

 template<class Checker>
 void reinit_after_event(
     State& x, const Time& t_event, Time& dt,
     Time& t_start, Time& t_end, State& x_at_start,
     std::vector<double>& last_val,
     std::vector<State>& last_state, std::vector<Time>& last_time,
     size_t& steps, Checker& checker)
 {
   init_stepper_after_event(x, t_event, dt);
   m_st.do_step(m_sys);
   ++steps; checker(); checker.reset();
   checker.set_last_order(get_stepper_order(m_st));

   t_start = m_st.previous_time();
   t_end = m_st.current_time();
   dt = m_st.current_time_step();
   checker.set_last_dt(scalar_value(t_end) - scalar_value(t_start));

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
     Obs& obs_raw, Checker& checker, double root_tol, size_t max_trigger)
 {
   auto& prof = get_profiler();
   auto obs = [&](const auto& x_, const auto& t_) {
     auto _tp = prof.timer(cppode::prof_cat::observer);
     obs_raw(x_, t_);
   };

   size_t steps = 0;
   auto it = times.begin();
   const auto end = times.end();
   Time t = *it;
   m_t_final = scalar_value(times.back());

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
         checker.set_last_order(get_stepper_order(m_st));
         checker.set_last_dt(scalar_value(dt_step));

         if (m_termination && m_termination(x, t)) {
           obs(x, t); return steps;
         }

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
       } else {
         checker(); dt = dt_step;
       }
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
     Obs& obs_raw, Checker& checker, double root_tol, size_t max_trigger)
 {
   auto& prof = get_profiler();
   auto obs = [&](const auto& x_, const auto& t_) {
     auto _tp = prof.timer(cppode::prof_cat::observer);
     obs_raw(x_, t_);
   };

   size_t steps = 0;
   auto it = times.begin(); auto end = times.end();
   m_t_final = scalar_value(times.back());

   if (apply_fixed_events_at_time(x, *it, m_fixed, m_sys))
     m_st.initialize(x, *it, dt);
   obs(x, *it); ++it;
   if (it == end) return 0;

   m_st.initialize(x, times.front(), dt);
   m_st.do_step(m_sys); ++steps; checker(); checker.reset();
   checker.set_last_order(get_stepper_order(m_st));

   Time t_start = m_st.previous_time();
   Time t_end = m_st.current_time();
   dt = m_st.current_time_step();
   checker.set_last_dt(scalar_value(t_end) - scalar_value(t_start));

   State x_at_start(x.size());
   m_st.calc_state(t_start, x_at_start);

   if (m_termination) {
     State x_check(x.size());
     m_st.calc_state(t_end, x_check);
     if (m_termination(x_check, t_end)) {
       obs(x_check, t_end); x = x_check; return steps;
     }
   }

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

       if (m_termination) {
         m_st.calc_state(t_end, x);
         if (m_termination(x, t_end)) {
           obs(x, t_end); return steps;
         }
       }

       m_st.do_step(m_sys); ++steps; checker(); checker.reset();
       checker.set_last_order(get_stepper_order(m_st));
       t_start = m_st.previous_time(); t_end = m_st.current_time();
       dt = m_st.current_time_step();
       checker.set_last_dt(scalar_value(t_end) - scalar_value(t_start));
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
         init_stepper_after_event(x, t_eval_s, dt);
         m_st.do_step(m_sys); ++steps; checker(); checker.reset();
         checker.set_last_order(get_stepper_order(m_st));
         t_start = m_st.previous_time(); t_end = m_st.current_time();
         dt = m_st.current_time_step();
         checker.set_last_dt(scalar_value(t_end) - scalar_value(t_start));
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
 DtEstimator m_dt_estimator;
 TerminationFunc m_termination;
 // Integration endpoint, set at the start of process_{controlled,dense}.
 // Used as the upper-bound hint for cppode_hin when re-estimating the
 // initial step size after an event restart on multistep methods.
 double m_t_final = 0.0;
};

// ============================================================================
// Public API
// ============================================================================

template<class Stepper, class System, class State,
        class TimeIterator, class Time, class Observer,
        class DtEstimator = no_dt_estimator>
size_t integrate_times(
   Stepper& stepper, System system, State& x,
   TimeIterator t_begin, TimeIterator t_end, Time dt, Observer obs,
   const std::vector<FixedEvent<State, typename State::value_type>>& fixed,
   const std::vector<RootEvent<State, Time>>& root,
   StepChecker& checker, double root_tol = 1e-8, size_t max_trigger_root = 1,
   DtEstimator dt_est = DtEstimator(),
   std::function<bool(const State&, const Time&)> termination = nullptr,
   cppode::controlled_stepper_tag = cppode::controlled_stepper_tag())
{
 auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);
 EventEngine<Stepper, System, State, Time, DtEstimator> eng(stepper, system, fixed, root, std::move(dt_est));
 if (termination) eng.set_termination(std::move(termination));
 try {
   size_t steps = eng.process_controlled(x, times, dt, obs, checker, root_tol, max_trigger_root);
   transfer_stepper_diagnostics(stepper, checker);
   return steps;
 } catch (...) {
   transfer_stepper_diagnostics(stepper, checker);
   throw;
 }
}

template<class Stepper, class System, class State,
        class TimeIterator, class Time, class Observer,
        class DtEstimator = no_dt_estimator>
size_t integrate_times_dense(
   Stepper& stepper, System system, State& x,
   TimeIterator t_begin, TimeIterator t_end, Time dt, Observer obs,
   const std::vector<FixedEvent<State, typename State::value_type>>& fixed,
   const std::vector<RootEvent<State, Time>>& root,
   StepChecker& checker, double root_tol = 1e-8, size_t max_trigger_root = 1,
   DtEstimator dt_est = DtEstimator(),
   std::function<bool(const State&, const Time&)> termination = nullptr,
   cppode::dense_output_stepper_tag = cppode::dense_output_stepper_tag())
{
 auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);
 EventEngine<Stepper, System, State, Time, DtEstimator> eng(stepper, system, fixed, root, std::move(dt_est));
 if (termination) eng.set_termination(std::move(termination));
 try {
   size_t steps = eng.process_dense(x, times, dt, obs, checker, root_tol, max_trigger_root);
   transfer_stepper_diagnostics(stepper, checker);
   return steps;
 } catch (...) {
   transfer_stepper_diagnostics(stepper, checker);
   throw;
 }
}

} // namespace detail

using detail::FixedEvent;
using detail::RootEvent;
using detail::EventMethod;
using detail::integrate_times;
using detail::integrate_times_dense;
using detail::make_steady_state_termination;

} // namespace cppode

#endif // CPPODE_INTEGRATE_TIMES_HPP
