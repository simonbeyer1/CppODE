/*
 [auto_generated]
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
 * @brief Type of event action.
 *
 * Events may either replace, add to, or multiply a given state variable.
 */
enum class EventMethod { Replace, Add, Multiply };

/**
 * @brief Fixed-time event: triggers when integration time reaches `time`.
 *
 * @tparam value_type The scalar type of the state entries (e.g. double or F<double>).
 */
template<class value_type>
struct FixedEvent {
  double time;        ///< Event trigger time (always stored as double for precision)
  int state_index;    ///< Index of the affected state variable
  value_type value;   ///< Value to apply
  EventMethod method; ///< How to apply (replace, add, multiply)
};

/**
 * @brief Root event: triggers when user-supplied function f(x,t) crosses zero.
 *
 * @tparam state_type The type of the state vector (e.g. ublas::vector<T>)
 * @tparam time_type  The scalar type used for time (e.g. double or F<double>)
 */
template<class state_type , class time_type>
struct RootEvent {
  std::function<double(const state_type&, double)> func; ///< Root function f(x,t)
  int state_index;    ///< Index of the affected state variable
  typename state_type::value_type value; ///< Value to apply
  EventMethod method; ///< How to apply
};

/**
 * @brief Apply an event to a state vector.
 *
 * @param x      State vector
 * @param idx    Index of affected state variable
 * @param value  Value to apply
 * @param method Event method (Replace/Add/Multiply)
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
 * @brief Check and apply all fixed events at the current time.
 *
 * @param x            Current state
 * @param t            Current time
 * @param fixed_events List of fixed events
 * @param tol          Time tolerance (absolute)
 *
 * This function checks whether `t` coincides (within tolerance)
 * with any event times. If so, the corresponding event(s) are applied.
 *
 * @note `t` may be double or an automatic differentiation type.
 *       Conversion to double is attempted via static_cast.
 */
template<class state_type , class Time>
inline bool check_and_apply_fixed_events(
    state_type &x , const Time &t ,
    const std::vector< FixedEvent<typename state_type::value_type> > &fixed_events ,
    double tol )
{
  double t_val = static_cast<double>(t); // works for double and F<double>
  bool triggered = false;

  for(const auto& ev : fixed_events) {
    if(std::abs(t_val - ev.time) < tol) {
      apply_event(x, ev.state_index, ev.value, ev.method);
      triggered = true;
    }
  }
  return triggered;
}

/**
 * @brief Merge user-specified times with fixed-event times.
 *
 * Ensures that all fixed-event times are included in the integration schedule,
 * even if the user did not request them explicitly.
 *
 * @tparam Time        Scalar time type (e.g. double, F<double>)
 * @tparam TimeIterator Iterator type for the user-specified time range
 *
 * @param user_begin   Iterator to begin of user-specified times
 * @param user_end     Iterator to end of user-specified times
 * @param fixed_events List of fixed events
 * @param tol          Tolerance for considering times "equal"
 *
 * @return Vector of merged and sorted time points
 */
template<class Time , class TimeIterator , class value_type>
std::vector<Time> merge_user_and_event_times(
    TimeIterator user_begin , TimeIterator user_end ,
    const std::vector< FixedEvent<value_type> > &fixed_events ,
    double tol )
{
  std::vector<Time> all_times(user_begin, user_end);

  // Add all event times
  for(const auto& ev : fixed_events) {
    all_times.push_back(static_cast<Time>(ev.time));
  }

  // Sort and unique (remove near-duplicates within tol)
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
 * @brief Integrate an ODE with controlled stepper and events.
 *
 * @details
 *  - Observes the state at all user-specified times.
 *  - Additionally observes at all fixed-event times (they are merged into
 *    the observation grid with tolerance-based de-duplication).
 *  - Applies *fixed events exactly at those times* (before calling the observer),
 *    and reinitializes the stepper so integration continues from the updated state.
 *  - Applies root events when zero crossings are detected between samples
 *    (simple sign-change detection).
 *
 * @tparam Stepper      Controlled stepper type
 * @tparam System       System functor (x,t) -> dxdt (or pair<sys,jac> for Rosenbrock)
 * @tparam state_type   State vector type (e.g. ublas::vector<AD>)
 * @tparam TimeIterator Iterator over observation times
 * @tparam Time         Scalar time type (e.g. double or fadbad::F<double>)
 * @tparam Observer     Observer functor (x,t)
 *
 * @param stepper       Controlled stepper instance
 * @param system        System dynamics function
 * @param start_state   Initial state (modified in place)
 * @param start_time    Iterator to first observation time
 * @param end_time      Iterator to one-past-last observation time
 * @param dt            Initial step size
 * @param observer      Observer functor (called at observation times)
 * @param fixed_events  Vector of fixed-time events
 * @param root_events   Vector of root events
 * @param checker       StepChecker (counts steps / no-progress)
 * @param root_tol      Tolerance for time matching & root sign-change de-dup
 *
 * @return Number of accepted steps
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
    controlled_stepper_tag = controlled_stepper_tag()
)
{
  using boost::numeric::odeint::detail::less_with_sign;

  typename odeint::unwrap_reference< Observer >::type &obs = observer;
  typename odeint::unwrap_reference< Stepper >::type &st   = stepper;

  failed_step_checker fail_checker;
  size_t steps = 0;

  // --- Build unified observation grid (user times + fixed-event times)
  auto all_times = merge_user_and_event_times<Time>(start_time, end_time, fixed_events, root_tol);
  auto iter = all_times.begin();
  auto end  = all_times.end();

  if (iter == end) return 0;

  // Root function history for sign-change detection
  std::vector<double> last_vals(root_events.size(),
                                std::numeric_limits<double>::quiet_NaN());

  // Main integration loop over observation grid
  while (true)
  {
    Time current_time = *iter++;

    // >>> Apply fixed events BEFORE observing <<<
    if (check_and_apply_fixed_events(start_state, current_time, fixed_events, root_tol)) {
      // Reinitialize stepper so it continues from updated state
      st.initialize(start_state, current_time, dt);
    }

    obs(start_state, current_time);

    if (iter == end) break; // done

    // Advance integration until next scheduled output time
    while (less_with_sign(current_time, *iter, dt))
    {
      Time current_dt = min_abs(dt, *iter - current_time);

      if (st.try_step(system, start_state, current_time, current_dt) == success)
      {
        ++steps;
        fail_checker.reset();
        checker();

        dt = max_abs(dt, current_dt);

        // Root detection
        for (size_t i = 0; i < root_events.size(); ++i) {
          double f_now = root_events[i].func(start_state, static_cast<double>(current_time));
          if (!std::isnan(last_vals[i]) && last_vals[i] * f_now < 0.0) {
            apply_event(start_state,
                        root_events[i].state_index,
                        root_events[i].value,
                        root_events[i].method);
            // Also reinitialize after root-event application
            st.initialize(start_state, current_time, dt);
          }
          last_vals[i] = f_now;
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


// ============================================================================
// Dense-output stepper version
// ============================================================================
/**
 * @brief Integrate an ODE with dense output stepper and events.
 *
 * @details
 *  - Produces dense output at all user-specified times.
 *  - Additionally forces output at all fixed-event times (they are merged into
 *    the observation grid with tolerance-based de-duplication).
 *  - Applies *fixed events exactly at those times* (before calling the observer),
 *    and reinitializes the stepper so integration continues from the updated state.
 *  - Applies root events when zero crossings are detected between samples
 *    (simple sign-change detection; no bracketing/bisection here).
 *
 * @tparam Stepper      Dense-output stepper type
 * @tparam System       System functor (x,t) -> dxdt (or pair<sys,jac> for Rosenbrock)
 * @tparam state_type   State vector type (e.g. ublas::vector<AD>)
 * @tparam TimeIterator Iterator over observation times
 * @tparam Time         Scalar time type (e.g. double or fadbad::F<double>)
 * @tparam Observer     Observer functor (x,t)
 *
 * @param stepper       Dense-output stepper instance
 * @param system        System dynamics function
 * @param start_state   Initial state (modified in place)
 * @param start_time    Iterator to first observation time
 * @param end_time      Iterator to one-past-last observation time
 * @param dt            Initial step size
 * @param observer      Observer functor (called at observation times)
 * @param fixed_events  Vector of fixed-time events
 * @param root_events   Vector of root events
 * @param checker       StepChecker (counts steps / no-progress)
 * @param root_tol      Tolerance for time matching & root sign-change de-dup
 *
 * @return Number of real steps performed by the dense stepper
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
    dense_output_stepper_tag = dense_output_stepper_tag()
)
{
  using boost::numeric::odeint::detail::less_with_sign;
  using boost::numeric::odeint::detail::less_eq_with_sign;

  typename odeint::unwrap_reference< Observer >::type &obs = observer;
  typename odeint::unwrap_reference< Stepper >::type &st   = stepper;

  // --- Merge user times and fixed-event times
  auto all_times = merge_user_and_event_times<Time>(start_time, end_time, fixed_events, root_tol);
  if (all_times.empty()) return 0;

  auto iter = all_times.begin();
  auto end  = all_times.end();
  Time last_time_point = all_times.back();

  // Initialize at first time
  st.initialize(start_state, *iter, dt);

  // >>> Apply fixed events BEFORE observing <<<
  if (check_and_apply_fixed_events(start_state, st.current_time(), fixed_events, root_tol)) {
    st.initialize(start_state, st.current_time(), dt);
  }

  obs(start_state, *iter++);

  // Root function history
  std::vector<double> last_vals(root_events.size(),
                                std::numeric_limits<double>::quiet_NaN());

  size_t count = 0;
  while (iter != end)
  {
    // Interpolate to each scheduled output time
    while ((iter != end) &&
           less_eq_with_sign(*iter, st.current_time(), st.current_time_step()))
    {
      st.calc_state(*iter, start_state);

      // >>> Apply fixed events BEFORE observing <<<
      if (check_and_apply_fixed_events(start_state, *iter, fixed_events, root_tol)) {
        st.initialize(start_state, *iter, dt);
      }

      obs(start_state, *iter);

      // Root detection
      for (size_t i = 0; i < root_events.size(); ++i) {
        double f_now = root_events[i].func(start_state, static_cast<double>(*iter));
        if (!std::isnan(last_vals[i]) && last_vals[i] * f_now < 0.0) {
          apply_event(start_state,
                      root_events[i].state_index,
                      root_events[i].value,
                      root_events[i].method);
          // Reinitialize after root event
          st.initialize(start_state, *iter, dt);
        }
        last_vals[i] = f_now;
      }

      ++iter;
    }

    // Advance the dense stepper
    if (less_eq_with_sign(st.current_time() + st.current_time_step(),
                          last_time_point, st.current_time_step()))
    {
      st.do_step(system);
      ++count;
      checker();
    }
    else if (iter != end)
    {
      // Final partial step
      st.initialize(st.current_state(), st.current_time(),
                    last_time_point - st.current_time());
      st.do_step(system);
      ++count;
      checker();
    }
  }

  return count;
}



} // namespace detail

// Public re-exports
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
