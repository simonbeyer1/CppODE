/*
 boost/numeric/odeint/integrate/detail/integrate_times_with_events.hpp

 [begin_description]
 Integration of ODEs with observation at user defined points,
 extended with fixed-time and root events.

 This file contains user-defined extensions for the R package `CppODE`.
 to account for additional functionality (fixed-time and root events)

 Author: Simon Beyer
 With assistance from: ChatGPT5 (OpenAI)
 */

#ifndef BOOST_NUMERIC_ODEINT_INTEGRATE_DETAIL_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_INTEGRATE_DETAIL_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED

#include <vector>
#include <functional>
#include <cmath>
#include <limits>
#include <algorithm>

#include <boost/numeric/odeint/util/detail/less_with_sign.hpp>
#include <boost/numeric/odeint/stepper/stepper_categories.hpp>
#include <boost/numeric/odeint/integrate/step_checker.hpp>

namespace boost {
namespace numeric {
namespace odeint {
namespace detail {

/**
 * @brief Event method type.
 *
 * Determines how an event modifies a state variable:
 *  - Replace : overwrite with a new value
 *  - Add     : increment by a value
 *  - Multiply: scale by a value
 */
enum class EventMethod { Replace, Add, Multiply };

/**
 * @brief Fixed-time event: triggers at a specific integration time.
 *
 * @tparam value_type Scalar type (double, AD, etc.).
 *
 * A fixed-time event modifies one state variable when integration
 * time reaches `time`. Both `time` and `value` may be of automatic
 * differentiation type if sensitivities are required.
 */
template<class value_type>
struct FixedEvent {
  value_type time;
  int state_index;
  value_type value;
  EventMethod method;
};

/**
 * @brief Root event: triggers when a user-supplied root function crosses zero.
 *
 * @tparam state_type State vector type (e.g. ublas::vector<AD>)
 * @tparam time_type  Scalar time type (e.g. double, AD)
 *
 * A root event evaluates `func(x,t)` at each step. When its sign changes,
 * the event fires and modifies a given state variable. Unlike fixed events,
 * root events are triggered dynamically during integration.
 */
template<class state_type , class time_type>
struct RootEvent {
  using value_type = typename state_type::value_type;
  std::function< value_type(const state_type&, const time_type&) > func;
  int state_index;
  value_type value;
  EventMethod method;
};

/**
 * @brief Apply an event action to a state vector.
 *
 * @tparam state_type State vector type (e.g. ublas::vector<double> or ublas::vector<AD>)
 *
 * @param x      Current state vector (modified in place)
 * @param idx    Index of the affected state variable
 * @param value  Value to apply (same type as state entries)
 * @param method Event method (Replace, Add, Multiply)
 *
 * @details
 *  - Replace  : overwrites x[idx] with value
 *  - Add      : increments x[idx] by value
 *  - Multiply : scales x[idx] by value
 */
template<class state_type>
inline void apply_event(
    state_type &x , int idx ,
    const typename state_type::value_type &value ,
    EventMethod method )
{
  switch(method) {
  case EventMethod::Replace:  x[idx] = value; break;
  case EventMethod::Add:      x[idx] += value; break;
  case EventMethod::Multiply: x[idx] *= value; break;
  }
}

/**
 * @brief Check and apply all fixed-time events at the current time.
 *
 * @tparam state_type State vector type (ublas::vector<T>)
 * @tparam Time       Scalar time type (double, AD)
 *
 * @param x            Current state (modified if events trigger)
 * @param t            Current integration time
 * @param fixed_events Vector of fixed-time events
 * @param tol          Absolute tolerance for time matching
 *
 * @return true if at least one event triggered, false otherwise
 *
 * @details
 *  - Each event triggers if |t - event.time| < tol
 *  - Multiple events may fire at the same time
 *  - After triggering, the state vector is updated in-place
 */
template<class state_type , class Time>
inline bool check_and_apply_fixed_events(
    state_type &x , const Time &t ,
    const std::vector< FixedEvent<typename state_type::value_type> > &fixed_events ,
    double tol )
{
  double t_val = static_cast<double>(t);
  bool triggered = false;
  for(const auto& ev : fixed_events) {
    double ev_t = static_cast<double>(ev.time);
    if(std::abs(t_val - ev_t) < tol) {
      apply_event(x, ev.state_index, ev.value, ev.method);
      triggered = true;
    }
  }
  return triggered;
}

/**
 * @brief Merge user-specified times with fixed-event times.
 *
 * @tparam Time         Scalar time type (double, AD)
 * @tparam TimeIterator Iterator over user-specified times
 * @tparam value_type   Value type of event payloads
 *
 * @param user_begin  Iterator to beginning of user times
 * @param user_end    Iterator past end of user times
 * @param fixed_events List of fixed events (their times are added)
 * @param tol          Absolute tolerance for duplicate detection
 *
 * @return Sorted vector of unique times including both user and event times
 *
 * @details
 *  - All event times are appended to the user times
 *  - Resulting vector is sorted ascending
 *  - Times within `tol` are collapsed into a single entry
 */
template<class Time , class TimeIterator , class value_type>
std::vector<Time> merge_user_and_event_times(
    TimeIterator user_begin , TimeIterator user_end ,
    const std::vector< FixedEvent<value_type> > &fixed_events ,
    double tol )
{
  std::vector<Time> all_times(user_begin, user_end);
  for(const auto& ev : fixed_events) {
    all_times.push_back(static_cast<Time>(ev.time));
  }
  std::sort(all_times.begin(), all_times.end(),
            [](const Time &a, const Time &b){
              return static_cast<double>(a) < static_cast<double>(b);
            });
  auto new_end = std::unique(all_times.begin(), all_times.end(),
                             [tol](const Time &a, const Time &b){
                               return std::abs(static_cast<double>(a) - static_cast<double>(b)) < tol;
                             });
  all_times.erase(new_end, all_times.end());
  return all_times;
}

// ============================================================================
// Controlled stepper version
// ============================================================================
/**
 * @brief Integrate an ODE system with controlled stepper and events.
 *
 * @tparam Stepper      Controlled stepper type (e.g. rosenbrock4_controller_ad)
 * @tparam System       ODE system functor, or pair<system,jacobian>
 * @tparam state_type   State vector type (ublas::vector<T>)
 * @tparam TimeIterator Iterator type over observation times
 * @tparam Time         Scalar time type (double, AD)
 * @tparam Observer     Observer functor
 *
 * @param stepper       Controlled stepper instance
 * @param system        System dynamics (sys or pair<sys,jac>)
 * @param start_state   Initial state (modified in place)
 * @param start_time    Iterator to first output time
 * @param end_time      Iterator past last output time
 * @param dt            Initial step size
 * @param observer      Observer called at output times
 * @param fixed_events  Vector of fixed-time events
 * @param root_events   Vector of root events
 * @param checker       StepChecker for step limits
 * @param root_tol      Tolerance for time and root detection
 * @param max_trigger_root Maximum times each root event may fire (default = 1, use std::numeric_limits<size_t>::max() for unlimited)
 * @param controlled_stepper_tag Dispatch tag
 *
 * @return Number of accepted steps
 *
 * @details
 *  - Observes state at all user-specified times and fixed-event times
 *  - Applies fixed events exactly at their trigger times before observation
 *  - Detects zero-crossings for root events and applies corresponding actions
 *  - Root events are limited to at most `max_trigger_root` firings per event
 *  - Stepper is reinitialized after each event modifies the state
 */
template< class Stepper , class System , class state_type ,
          class TimeIterator , class Time , class Observer >
size_t integrate_times(
    Stepper stepper , System system , state_type &start_state ,
    TimeIterator start_time , TimeIterator end_time , Time dt ,
    Observer observer ,
    const std::vector< FixedEvent<typename state_type::value_type> > &fixed_events ,
    const std::vector< RootEvent<state_type, Time> > &root_events ,
    StepChecker &checker ,
    double root_tol = 1e-8 ,
    size_t max_trigger_root = 1 ,
    controlled_stepper_tag = controlled_stepper_tag()
)
{
  using boost::numeric::odeint::detail::less_with_sign;
  typename odeint::unwrap_reference< Observer >::type &obs = observer;
  typename odeint::unwrap_reference< Stepper >::type &st   = stepper;

  failed_step_checker fail_checker;
  size_t steps = 0;

  auto all_times = merge_user_and_event_times<Time>(start_time, end_time, fixed_events, root_tol);
  auto iter = all_times.begin();
  auto end  = all_times.end();
  if (iter == end) return 0;

  std::vector<double> last_vals(root_events.size(),
                                std::numeric_limits<double>::quiet_NaN());
  std::vector<size_t> root_trigger_count(root_events.size(), 0);

  while (true)
  {
    Time current_time = *iter++;
    if (check_and_apply_fixed_events(start_state, current_time, fixed_events, root_tol)) {
      st.initialize(start_state, current_time, dt);
    }
    obs(start_state, current_time);
    if (iter == end) break;

    while (less_with_sign(current_time, *iter, dt))
    {
      Time current_dt = min_abs(dt, *iter - current_time);
      if (st.try_step(system, start_state, current_time, current_dt) == success)
      {
        ++steps; fail_checker.reset(); checker();
        dt = max_abs(dt, current_dt);

        for (size_t i = 0; i < root_events.size(); ++i) {
          auto f_val = root_events[i].func(start_state, current_time);
          double f_now = static_cast<double>(f_val);
          if (root_trigger_count[i] < max_trigger_root &&
              !std::isnan(last_vals[i]) && last_vals[i] * f_now < 0.0) {
            apply_event(start_state, root_events[i].state_index,
                        root_events[i].value, root_events[i].method);
            st.initialize(start_state, current_time, dt);
            root_trigger_count[i]++;
          }
          last_vals[i] = f_now;
        }
      } else {
        fail_checker(); dt = current_dt;
      }
    }
  }
  return steps;
}

// ============================================================================
// Dense-output stepper version
// ============================================================================
/**
 * @brief Integrate an ODE system with dense-output stepper and events.
 *
 * @tparam Stepper      Dense-output stepper type
 * @tparam System       ODE system functor, or pair<system,jacobian>
 * @tparam state_type   State vector type (ublas::vector<T>)
 * @tparam TimeIterator Iterator type over observation times
 * @tparam Time         Scalar time type (double, AD)
 * @tparam Observer     Observer functor
 *
 * @param stepper       Dense-output stepper instance
 * @param system        System dynamics (sys or pair<sys,jac>)
 * @param start_state   Initial state (modified in place)
 * @param start_time    Iterator to first output time
 * @param end_time      Iterator past last output time
 * @param dt            Initial step size
 * @param observer      Observer called at output times
 * @param fixed_events  Vector of fixed-time events
 * @param root_events   Vector of root events
 * @param checker       StepChecker for step limits
 * @param root_tol      Tolerance for time and root detection
 * @param max_trigger_root Maximum times each root event may fire (default = 1, use std::numeric_limits<size_t>::max() for unlimited)
 * @param dense_output_stepper_tag Dispatch tag
 *
 * @return Number of real steps performed by the dense stepper
 *
 * @details
 *  - Produces dense output at all user times and fixed-event times
 *  - Applies fixed events exactly at their trigger times before observation
 *  - Detects zero-crossings for root events and applies corresponding actions
 *  - Root events are limited to at most `max_trigger_root` firings per event
 *  - Stepper is reinitialized after each event modifies the state
 */
template< class Stepper , class System , class state_type ,
          class TimeIterator , class Time , class Observer >
size_t integrate_times_dense(
    Stepper stepper , System system , state_type &start_state ,
    TimeIterator start_time , TimeIterator end_time , Time dt ,
    Observer observer ,
    const std::vector< FixedEvent<typename state_type::value_type> > &fixed_events ,
    const std::vector< RootEvent<state_type, Time> > &root_events ,
    StepChecker &checker ,
    double root_tol = 1e-8 ,
    size_t max_trigger_root = 1 ,
    dense_output_stepper_tag = dense_output_stepper_tag()
)
{
  using boost::numeric::odeint::detail::less_eq_with_sign;
  typename odeint::unwrap_reference< Observer >::type &obs = observer;
  typename odeint::unwrap_reference< Stepper >::type &st   = stepper;

  auto all_times = merge_user_and_event_times<Time>(start_time, end_time, fixed_events, root_tol);
  if (all_times.empty()) return 0;

  auto iter = all_times.begin();
  auto end  = all_times.end();
  Time last_time_point = all_times.back();

  st.initialize(start_state, *iter, dt);
  if (check_and_apply_fixed_events(start_state, st.current_time(), fixed_events, root_tol)) {
    st.initialize(start_state, st.current_time(), dt);
  }
  obs(start_state, *iter++);

  std::vector<double> last_vals(root_events.size(),
                                std::numeric_limits<double>::quiet_NaN());
  std::vector<size_t> root_trigger_count(root_events.size(), 0);

  size_t count = 0;
  while (iter != end)
  {
    while ((iter != end) &&
           less_eq_with_sign(*iter, st.current_time(), st.current_time_step()))
    {
      st.calc_state(*iter, start_state);
      if (check_and_apply_fixed_events(start_state, *iter, fixed_events, root_tol)) {
        st.initialize(start_state, *iter, dt);
      }
      obs(start_state, *iter);

      for (size_t i = 0; i < root_events.size(); ++i) {
        auto f_val = root_events[i].func(start_state, *iter);
        double f_now = static_cast<double>(f_val);
        if (root_trigger_count[i] < max_trigger_root &&
            !std::isnan(last_vals[i]) && last_vals[i] * f_now < 0.0) {
          apply_event(start_state, root_events[i].state_index,
                      root_events[i].value, root_events[i].method);
          st.initialize(start_state, *iter, dt);
          root_trigger_count[i]++;
        }
        last_vals[i] = f_now;
      }
      ++iter;
    }

    if (less_eq_with_sign(st.current_time() + st.current_time_step(),
                          last_time_point, st.current_time_step()))
    {
      st.do_step(system); ++count; checker();
    } else if (iter != end) {
      st.initialize(st.current_state(), st.current_time(),
                    last_time_point - st.current_time());
      st.do_step(system); ++count; checker();
    }
  }
  return count;
}

} // namespace detail

using detail::FixedEvent;
using detail::RootEvent;
using detail::apply_event;
using detail::EventMethod;
using detail::integrate_times;
using detail::integrate_times_dense;

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif

