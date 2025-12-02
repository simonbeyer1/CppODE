/*
 boost/numeric/odeint/integrate/detail/integrate_times_with_events.hpp

 [begin_description]
 Integration of ODEs with observation at user-defined time points,
 extended with fixed-time and root-triggered events.

 This implementation has been extended to support arbitrary levels of
 automatic differentiation (AD) types such as fadbad::F<F<double>>.
 It is part of the CppODE system for event-aware ODE integration in R.

 Author: Simon Beyer
 [end_description]
 */

#ifndef BOOST_NUMERIC_ODEINT_INTEGRATE_DETAIL_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_INTEGRATE_DETAIL_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED

// Standard library
#include <vector>
#include <functional>
#include <cmath>
#include <limits>
#include <algorithm>
#include <type_traits>

// Boost.Odeint internals
#include <boost/numeric/odeint/util/detail/less_with_sign.hpp>
#include <boost/numeric/odeint/stepper/stepper_categories.hpp>
#include <boost/numeric/odeint/integrate/step_checker.hpp>

// FADBAD++ (required for fadbad::F<T> support below)
#include <fadbad++/fadiff.h>

namespace boost {
namespace numeric {
namespace odeint {
namespace detail {

/*==============================================================================
 Scalar extraction utilities
 =============================================================================*/

/**
 * @brief Extract a plain scalar double from an arithmetic type.
 *
 * This is the primary overload used when @c T is a built-in arithmetic type
 * (e.g. @c double, @c float, @c int). It simply casts the value to @c double.
 *
 * @tparam T  Arithmetic type.
 * @param v   Value to convert.
 * @return    The value converted to @c double.
 */
template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v)
{
  return static_cast<double>(v);
}

/**
 * @brief Extract a scalar double from a @c fadbad::F<T> (recursively).
 *
 * This overload recursively unwraps nested @c fadbad::F layers by calling
 * @c .x() until it reaches a plain arithmetic type. The result is then
 * converted to @c double.
 *
 * The @c .x() accessor in FADBAD is non-const, therefore the argument must
 * be temporarily cast away from const to satisfy the interface.
 *
 * @tparam T  Base type of the FADBAD variable (e.g. @c double or @c F<...>).
 * @param v   FADBAD variable.
 * @return    Underlying base value converted to @c double.
 */
template <typename T>
inline double scalar_value(const fadbad::F<T>& v)
{
  return scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

/*==============================================================================
 Event structures
 =============================================================================*/

/**
 * @brief How to modify a state variable when an event triggers.
 *
 * - @c Replace  : overwrite the state variable with a new value
 * - @c Add      : increment the state variable by a given value
 * - @c Multiply : scale the state variable by a given value
 */
enum class EventMethod { Replace, Add, Multiply };

/**
 * @brief Fixed-time event that triggers at a specific simulation time.
 *
 * @tparam value_type Scalar type for time and value (e.g. @c double,
 *         @c fadbad::F<double>, or nested AD types).
 *
 * - @c time        : trigger time (same scalar type as integration time)
 * - @c state_index : index of state component to modify
 * - @c value       : payload value to apply when the event fires
 * - @c method      : modification rule (replace/add/multiply)
 */
template <class value_type>
struct FixedEvent {
  value_type  time;
  int         state_index;
  value_type  value;
  EventMethod method;
};

/**
 * @brief Root-triggered event that fires when a user function crosses zero.
 *
 * @tparam state_type State vector type (e.g. @c ublas::vector<T>).
 * @tparam time_type  Time scalar type (@c double or AD type).
 *
 * - @c func        : root function f(x, t) whose sign change triggers event
 * - @c state_index : index of state component to modify
 * - @c value       : payload value to apply when root is detected
 * - @c method      : modification rule (replace/add/multiply)
 */
template <class state_type, class time_type>
struct RootEvent {
  using value_type = typename state_type::value_type;

  std::function< value_type(const state_type&, const time_type&) > func;
  int         state_index;
  value_type  value;
  EventMethod method;
};

/*==============================================================================
 Event utilities
 =============================================================================*/

/**
 * @brief Apply a single event action to a state vector.
 *
 * @tparam state_type State vector type (e.g. @c ublas::vector<double> or
 *         @c ublas::vector<AD>).
 *
 * @param x      Current state vector (modified in place).
 * @param idx    Index of the affected state variable.
 * @param value  Value to apply (same type as state entries).
 * @param method Event method (Replace, Add, Multiply).
 */
template <class state_type>
inline void apply_event(
    state_type& x,
    int idx,
    const typename state_type::value_type& value,
    EventMethod method)
{
  switch (method) {
  case EventMethod::Replace:
    x[idx] = value;
    break;
  case EventMethod::Add:
    x[idx] += value;
    break;
  case EventMethod::Multiply:
    x[idx] *= value;
    break;
  }
}

/**
 * @brief Apply all fixed-time events scheduled for exactly this time.
 *
 * Since event times are merged into the output time sequence, the integrator
 * stops exactly at each event time. This function applies all events whose
 * time matches the current time exactly (comparing scalar values).
 *
 * @tparam state_type State vector type (e.g. @c ublas::vector<T>).
 * @tparam Time       Scalar time type (e.g. @c double, AD type).
 *
 * @param x            Current state (modified if events trigger).
 * @param t            Current integration time.
 * @param fixed_events Vector of fixed-time events.
 *
 * @return @c true if at least one event was triggered, @c false otherwise.
 */
template <class state_type, class Time>
inline bool apply_fixed_events_at_time(
    state_type& x,
    const Time& t,
    const std::vector< FixedEvent<typename state_type::value_type> >& fixed_events)
{
  const double t_val = scalar_value(t);
  bool triggered = false;

  for (const auto& ev : fixed_events) {
    if (scalar_value(ev.time) == t_val) {
      apply_event(x, ev.state_index, ev.value, ev.method);
      triggered = true;
    }
  }
  return triggered;
}

/**
 * @brief Merge user-defined observation times with all fixed-event times.
 *
 * All event times are appended to the user times. The resulting vector is
 * sorted in ascending order (using @c scalar_value for comparison) and
 * duplicate times are removed.
 *
 * @tparam Time         Scalar time type (e.g. @c double or AD type).
 * @tparam TimeIterator Iterator over user-specified times.
 * @tparam value_type   Value type used for event times.
 *
 * @param user_begin   Iterator to beginning of user times.
 * @param user_end     Iterator past end of user times.
 * @param fixed_events List of fixed events (their times are added).
 *
 * @return Sorted vector of unique times including both user and event times.
 */
template <class Time, class TimeIterator, class value_type>
std::vector<Time> merge_user_and_event_times(
    TimeIterator user_begin, TimeIterator user_end,
    const std::vector< FixedEvent<value_type> >& fixed_events)
{
  std::vector<Time> all_times(user_begin, user_end);

  for (const auto& ev : fixed_events)
    all_times.push_back(ev.time);

  std::sort(
    all_times.begin(), all_times.end(),
    [](const Time& a, const Time& b) {
      return scalar_value(a) < scalar_value(b);
    }
  );

  const auto new_end = std::unique(
    all_times.begin(), all_times.end(),
    [](const Time& a, const Time& b) {
      return scalar_value(a) == scalar_value(b);
    }
  );
  all_times.erase(new_end, all_times.end());
  return all_times;
}

/*==============================================================================
 Controlled-stepper version
 =============================================================================*/

/**
 * @brief Integrate an ODE system with a controlled stepper and event handling.
 *
 * This overload works with controlled steppers (e.g.
 * @c rosenbrock4_controller_ad) and produces output at a prescribed sequence
 * of times. It applies:
 *
 *  - fixed-time events: applied exactly at scheduled times
 *  - root events: detected via sign changes of user-supplied root functions
 *
 * Root events are detected by monitoring the sign of f(x, t) after each
 * successful step. When a sign change is detected, the corresponding event is
 * applied at the *current* time; no additional root localization is performed
 * in the controlled-step mode.
 *
 * @tparam Stepper      Controlled stepper type
 *                      (e.g. @c rosenbrock4_controller_ad<...>).
 * @tparam System       ODE system functor or @c std::pair<system, jacobian>.
 * @tparam state_type   State vector type (e.g. @c ublas::vector<T>).
 * @tparam TimeIterator Iterator over observation times.
 * @tparam Time         Scalar time type (e.g. @c double, AD type).
 * @tparam Observer     Observer with call operator @c obs(x, t).
 *
 * @param stepper       Controlled stepper instance (passed by value).
 * @param system        System dynamics (system or pair<system, jacobian>).
 * @param start_state   Initial state (modified in place).
 * @param start_time    Iterator to first output time.
 * @param end_time      Iterator past last output time.
 * @param dt            Initial step size.
 * @param observer      Observer called at output times.
 * @param fixed_events  Fixed-time events.
 * @param root_events   Root-triggered events.
 * @param checker       StepChecker for step and progress limits.
 * @param max_trigger_root Maximum number of firings allowed per root event
 *                         (use @c std::numeric_limits<size_t>::max() for
 *                         "unlimited").
 * @param controlled_stepper_tag Dispatch tag for controlled steppers.
 *
 * @return The number of successful integration steps performed.
 */
template <class Stepper, class System, class state_type,
          class TimeIterator, class Time, class Observer>
size_t integrate_times(
    Stepper stepper, System system, state_type& start_state,
    TimeIterator start_time, TimeIterator end_time, Time dt,
    Observer observer,
    const std::vector< FixedEvent<typename state_type::value_type> >& fixed_events,
    const std::vector< RootEvent<state_type, Time> >& root_events,
    StepChecker& checker,
    double root_tol = 1e-8,
    size_t max_trigger_root = 1,
    controlled_stepper_tag = controlled_stepper_tag())
{
  using boost::numeric::odeint::detail::less_with_sign;
  typename odeint::unwrap_reference<Observer>::type& obs = observer;
  typename odeint::unwrap_reference<Stepper>::type& st   = stepper;
  failed_step_checker fail_checker;
  size_t steps = 0;

  auto all_times = merge_user_and_event_times<Time>(
    start_time, end_time, fixed_events);
  auto iter = all_times.begin();
  auto end  = all_times.end();
  if (iter == end) return 0;

  // Root event bookkeeping
  std::vector<double> last_vals(
      root_events.size(),
      std::numeric_limits<double>::quiet_NaN()
  );
  std::vector<state_type> last_states(root_events.size(), start_state);
  std::vector<Time> last_times(root_events.size(), *iter);
  std::vector<size_t> root_trigger_count(root_events.size(), 0);

  while (true)
  {
    Time current_time = *iter++;

    apply_fixed_events_at_time(start_state, current_time, fixed_events);

    obs(start_state, current_time);
    if (iter == end) break;

    while (less_with_sign(current_time, *iter, dt))
    {
      Time current_dt = min_abs(dt, *iter - current_time);

      // Store state before step for bisection
      state_type state_before_step = start_state;
      Time time_before_step = current_time;

      if (st.try_step(system, start_state, current_time, current_dt) == success)
      {
        ++steps;
        fail_checker.reset();
        checker();
        dt = max_abs(dt, current_dt);

        // Check root functions for sign changes
        for (size_t i = 0; i < root_events.size(); ++i)
        {
          auto f_val = root_events[i].func(start_state, current_time);
          double f_now = scalar_value(f_val);

          if (root_trigger_count[i] < max_trigger_root &&
              !std::isnan(last_vals[i]) &&
              last_vals[i] * f_now < 0.0)
          {
            // Root detected - localize via bisection
            Time       ta = last_times[i];
            Time       tb = current_time;
            state_type xa = last_states[i];
            state_type xb = start_state;

            double fa = last_vals[i];
            double fb = f_now;

            // Bisection to find root
            while (std::abs(scalar_value(tb - ta)) > root_tol) {
              Time tm = (ta + tb) / 2.0;

              // Integrate from last_times[i] to tm to get state at tm
              state_type xm = xa;
              Time t_temp = ta;
              Time dt_temp = (tm - ta) / 10.0;
              if (scalar_value(dt_temp) == 0.0) break;

              // Simple integration to midpoint
              while (less_with_sign(t_temp, tm, dt_temp)) {
                Time step_dt = min_abs(dt_temp, tm - t_temp);
                st.try_step(system, xm, t_temp, step_dt);
              }

              double fm = scalar_value(root_events[i].func(xm, tm));

              if (fa * fm <= 0.0) {
                tb = tm; xb = xm; fb = fm;
              } else {
                ta = tm; xa = xm; fa = fm;
              }
            }

            Time t_root = tb;

            // Observe before event
            obs(xb, t_root);

            // Apply event
            state_type x_after = xb;
            apply_event(x_after,
                        root_events[i].state_index,
                        root_events[i].value,
                        root_events[i].method);

            root_trigger_count[i]++;

            // Observe after event
            obs(x_after, t_root + Time(1e-15));

            // Update state and continue from root time
            start_state = x_after;
            current_time = t_root;

            // Reset sign memory
            std::fill(last_vals.begin(), last_vals.end(),
                      std::numeric_limits<double>::quiet_NaN());
            std::fill(last_times.begin(), last_times.end(), t_root);
            for (size_t j = 0; j < root_events.size(); ++j) {
              last_states[j] = start_state;
            }

            break;  // Exit root loop, continue integration
          }
          else
          {
            last_vals[i] = f_now;
            last_states[i] = start_state;
            last_times[i] = current_time;
          }
        }
      }
      else
      {
        fail_checker();
        dt = current_dt;
      }
    }
  }

  return steps;
}

/*==============================================================================
 Dense-output stepper version
 =============================================================================*/

/**
 * @brief Integrate an ODE system with dense-output and event handling.
 *
 * This overload works with dense-output steppers (e.g.
 * @c rosenbrock4_dense_output) and produces output at arbitrary times inside
 * the integration interval. It supports:
 *
 *  - fixed-time events: applied exactly at their trigger times
 *  - root events: detected via sign changes and localized by bisection
 *
 * Root events are localized inside the current step interval using bisection
 * in time until @c |t_b - t_a| < root_tol. The event is then applied at the
 * localized root time, the stepper is reinitialized at that time, and the
 * state is observed.
 *
 * @tparam Stepper      Dense-output stepper type
 *                      (e.g. @c rosenbrock4_dense_output<...>).
 * @tparam System       ODE system functor or @c std::pair<system, jacobian>.
 * @tparam state_type   State vector type (e.g. @c ublas::vector<T>).
 * @tparam TimeIterator Iterator over observation times.
 * @tparam Time         Scalar time type (e.g. @c double, AD type).
 * @tparam Observer     Observer with call operator @c obs(x, t).
 *
 * @param stepper       Dense-output stepper instance (passed by value).
 * @param system        System dynamics (system or pair<system, jacobian>).
 * @param start_state   Initial state (modified in place).
 * @param start_time    Iterator to first output time.
 * @param end_time      Iterator past last output time.
 * @param dt            Initial step size.
 * @param observer      Observer called at output times.
 * @param fixed_events  Fixed-time events.
 * @param root_events   Root-triggered events.
 * @param checker       StepChecker for step and progress limits.
 * @param root_tol      Tolerance for root localization via bisection.
 * @param max_trigger_root Maximum number of firings allowed per root event
 *                         (use @c std::numeric_limits<size_t>::max() for
 *                         "unlimited").
 * @param dense_output_stepper_tag Dispatch tag for dense-output steppers.
 *
 * @return The number of real steps performed by the dense stepper.
 */
template <class Stepper, class System, class state_type,
          class TimeIterator, class Time, class Observer>
size_t integrate_times_dense(
    Stepper stepper, System system, state_type& start_state,
    TimeIterator start_time, TimeIterator end_time, Time dt,
    Observer observer,
    const std::vector< FixedEvent<typename state_type::value_type> >& fixed_events,
    const std::vector< RootEvent<state_type, Time> >& root_events,
    StepChecker& checker,
    double root_tol = 1e-8,
    size_t max_trigger_root = 1,
    dense_output_stepper_tag = dense_output_stepper_tag())
{
  using boost::numeric::odeint::detail::less_eq_with_sign;

  typename odeint::unwrap_reference<Observer>::type& obs = observer;
  typename odeint::unwrap_reference<Stepper>::type& st   = stepper;

  auto all_times = merge_user_and_event_times<Time>(
    start_time, end_time, fixed_events);
  if (all_times.empty()) return 0;

  auto iter   = all_times.begin();
  auto end_it = all_times.end();

  st.initialize(start_state, *iter, dt);

  if (apply_fixed_events_at_time(start_state, *iter, fixed_events)) {
    st.reinitialize_at_event(start_state, *iter, dt);
  }

  obs(start_state, *iter);
  ++iter;
  if (iter == end_it) return 0;

  // Root event bookkeeping - now also store last state and time
  std::vector<double> last_vals(root_events.size(),
                                std::numeric_limits<double>::quiet_NaN());
  std::vector<state_type> last_states(root_events.size(), start_state);
  std::vector<Time> last_times(root_events.size(), *all_times.begin());
  std::vector<size_t> root_trigger_count(root_events.size(), 0);

  size_t count = 0;
  Time t_step_start, t_step_end;

  while (iter != end_it)
  {
    t_step_start = st.current_time();
    st.do_step(system);
    t_step_end = st.current_time();
    ++count;
    checker();

    bool event_triggered = false;

    while (iter != end_it && !event_triggered &&
           less_eq_with_sign(*iter, t_step_end, st.current_time_step()))
    {
      if (scalar_value(*iter) < scalar_value(t_step_start)) {
        ++iter;
        continue;
      }

      st.calc_state(*iter, start_state);

      if (apply_fixed_events_at_time(start_state, *iter, fixed_events)) {
        obs(start_state, *iter);
        st.reinitialize_at_event(start_state, *iter, st.current_time_step());
        std::fill(last_vals.begin(), last_vals.end(),
                  std::numeric_limits<double>::quiet_NaN());
        std::fill(root_trigger_count.begin(), root_trigger_count.end(), 0);
        ++iter;
        event_triggered = true;
        continue;
      }

      for (size_t i = 0; i < root_events.size(); ++i)
      {
        auto f_val = root_events[i].func(start_state, *iter);
        double f_now = scalar_value(f_val);

        if (root_trigger_count[i] < max_trigger_root &&
            !std::isnan(last_vals[i]) &&
            last_vals[i] * f_now < 0.0)
        {
          // Root detected - bisection between last known point and current
          Time       ta = last_times[i];
          Time       tb = *iter;
          state_type xa = last_states[i];
          state_type xb = start_state;

          double fa = last_vals[i];
          double fb = f_now;

          state_type xm = xa;

          while (std::abs(scalar_value(tb - ta)) > root_tol) {
            Time tm = (ta + tb) / 2.0;
            st.calc_state(tm, xm);
            double fm = scalar_value(root_events[i].func(xm, tm));

            if (fa * fm <= 0.0) {
              tb = tm; xb = xm; fb = fm;
            } else {
              ta = tm; xa = xm; fa = fm;
            }
          }

          Time t_root = tb;

          // Observe all output times before the root
          while (iter != end_it && scalar_value(*iter) < scalar_value(t_root)) {
            st.calc_state(*iter, start_state);
            obs(start_state, *iter);
            ++iter;
          }

          state_type x_after = xb;

          obs(xb, t_root);

          apply_event(x_after,
                      root_events[i].state_index,
                      root_events[i].value,
                      root_events[i].method);

          root_trigger_count[i]++;

          obs(x_after, t_root + Time(1e-15));

          start_state = x_after;
          st.reinitialize_at_event(start_state, t_root, st.current_time_step());

          std::fill(last_vals.begin(), last_vals.end(),
                    std::numeric_limits<double>::quiet_NaN());

          event_triggered = true;
          break;
        }
        else
        {
          last_vals[i] = f_now;
          last_states[i] = start_state;
          last_times[i] = *iter;
        }
      }

      if (!event_triggered) {
        obs(start_state, *iter);
        ++iter;
      }
    }
  }

  return count;
}

} // namespace detail

// Public type aliases / exports
using detail::FixedEvent;
using detail::RootEvent;
using detail::apply_event;
using detail::EventMethod;
using detail::integrate_times;
using detail::integrate_times_dense;

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // BOOST_NUMERIC_ODEINT_INTEGRATE_DETAIL_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
