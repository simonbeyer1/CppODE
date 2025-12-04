#ifndef BOOST_NUMERIC_ODEINT_INTEGRATE_DETAIL_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_INTEGRATE_DETAIL_INTEGRATE_TIMES_WITH_EVENTS_HPP_INCLUDED

/*
 Unified event-aware integration engine for Boost.Odeint.

 Supports:
 • fixed-time events
 • root detection via sign change
 • root localization (bisection)
 • controlled and dense-output steppers
 • arbitrary nested AD types (fadbad::F<F<...>>)

 Exposes:
 • integrate_times()
 • integrate_times_dense()

 Compatible with:
 • rosenbrock4_controller_ad  (reset_after_event(dt))
 • rosenbrock4_dense_output_ad (reinitialize_at_event(x,t,dt))
 • fallback: plain Odeint steppers without event reset capability.
 */

#include <vector>
#include <functional>
#include <cmath>
#include <limits>
#include <algorithm>
#include <type_traits>

#include <boost/numeric/odeint/util/detail/less_with_sign.hpp>
#include <boost/numeric/odeint/stepper/stepper_categories.hpp>
#include <boost/numeric/odeint/integrate/step_checker.hpp>

#include <fadbad++/fadiff.h>

namespace boost {
namespace numeric {
namespace odeint {
namespace detail {

/*==============================================================================
 1. scalar_value — safe unwrapping of nested fadbad::F<T>
 ==============================================================================*/

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value,double>::type
scalar_value(const T& v) { return v; }

template<class T>
inline double scalar_value(const fadbad::F<T>& v)
{
  return scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

/*==============================================================================
 2. Event Types
 ==============================================================================*/

enum class EventMethod { Replace, Add, Multiply };

template<class value_type>
struct FixedEvent {
  value_type time;
  int        state_index;
  value_type value;
  EventMethod method;
};

template<class state_type, class time_type>
struct RootEvent {
  using value_t = typename state_type::value_type;

  std::function<value_t(const state_type&, const time_type&)> func;
  int         state_index;
  value_t     value;
  EventMethod method;
};

/*==============================================================================
 3. Event Application
 ==============================================================================*/

template<class state_type>
inline void apply_event(
    state_type& x,
    int idx,
    const typename state_type::value_type &v,
    EventMethod method)
{
  switch(method){
  case EventMethod::Replace:   x[idx]  = v; break;
  case EventMethod::Add:       x[idx] += v; break;
  case EventMethod::Multiply:  x[idx] *= v; break;
  }
}

template<class state_type,class Time>
bool apply_fixed_events_at_time(
    state_type& x,
    const Time& t,
    const std::vector<FixedEvent<typename state_type::value_type>>& evs)
{
  const double tt = scalar_value(t);
  bool fired = false;
  for(const auto& e : evs){
    if(scalar_value(e.time) == tt){
      apply_event(x, e.state_index, e.value, e.method);
      fired = true;
    }
  }
  return fired;
}

/*==============================================================================
 4. Merge user times + fixed-event times
 ==============================================================================*/

template<class Time,class It,class V>
std::vector<Time> merge_user_and_event_times(
    It ubegin, It uend,
    const std::vector<FixedEvent<V>>& fix)
{
  std::vector<Time> out(ubegin, uend);
  for(const auto& e : fix) out.push_back(e.time);

  std::sort(out.begin(), out.end(),
            [](auto&a,auto&b){ return scalar_value(a) < scalar_value(b); });

  out.erase(std::unique(out.begin(), out.end(),
                        [](auto&a,auto&b){
                          return scalar_value(a)==scalar_value(b);
                        }),
                        out.end());

  return out;
}

/*==============================================================================
 5. Reset dispatcher (supports controller_ad, dense_output_ad, fallback)
 ==============================================================================*/

template<class S,class State,class Time>
inline void reset_stepper_unified(S& st, State& x, Time t, Time& dt)
{
  if constexpr(requires { st.reset_after_event(dt); })
  {
    st.reset_after_event(dt);
  }
  else if constexpr(requires { st.reinitialize_at_event(x,t,dt); })
  {
    st.reinitialize_at_event(x,t,dt);
  }
  else {
    (void)st; (void)x; (void)t; (void)dt;
  }
}

/*==============================================================================
 6. EventEngine
 ==============================================================================*/

template<class Stepper,class System,class State,class Time>
class EventEngine {
public:
  using state_type = State;
  using time_type  = Time;

  EventEngine(
    Stepper& st, System& sys,
    const std::vector<FixedEvent<typename State::value_type>>& fixed,
    const std::vector<RootEvent<State,Time>>& root
  ) : m_st(st), m_sys(sys), m_fixed(fixed), m_root(root) {}

  /*--------------------------------------------------------------------------
   Controlled-stepper mode
   --------------------------------------------------------------------------*/
  template<class Obs,class Checker>
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

    if(apply_fixed_events_at_time(x,t,m_fixed))
      ;   // no reset here; dt is externally chosen before integration

    obs(x,t);
    ++it;
    if(it == end) return 0;

    std::vector<double> last_val(m_root.size(),std::numeric_limits<double>::quiet_NaN());
    std::vector<State>  last_state(m_root.size(), x);
    std::vector<Time>   last_time(m_root.size(), t);
    std::vector<size_t> fired(m_root.size(), 0);

    while(it != end)
    {
      Time t_target = *it;

      while(less_with_sign(t, t_target, dt))
      {
        Time dt_step = min_abs(dt, t_target - t);

        auto result = m_st.try_step(m_sys, x, t, dt_step);

        if(result == success){
          ++steps;
          checker();
          checker.reset();  // Time advanced successfully
          dt = dt_step;

          for(size_t i=0;i<m_root.size();++i)
          {
            auto fval = m_root[i].func(x,t);
            double f_now = scalar_value(fval);

            if(fired[i] < max_trigger &&
               !std::isnan(last_val[i]) &&
               last_val[i] * f_now < 0.0)
            {
              localize_root_controlled(
                i,
                last_state[i], last_time[i],
                                        x, t,
                                        last_val[i], f_now,
                                        root_tol,
                                        checker);

              Time t_root = t;
              obs(x, t_root);

              State x_after = x;
              apply_event(x_after,
                          m_root[i].state_index,
                          m_root[i].value,
                          m_root[i].method);

              fired[i]++;
              obs(x_after, t_root + Time(1e-15));
              x = x_after;

              // Reset stepper after discontinuity
              reset_stepper_unified(m_st, x, t, dt);

              // Reset root tracking but keep fired counts
              for(size_t j = 0; j < m_root.size(); ++j) {
                last_val[j] = std::numeric_limits<double>::quiet_NaN();
                last_state[j] = x;
                last_time[j] = t;
              }

              break;  // exit root loop, continue integration
            }
            else {
              last_val[i] = f_now;
              last_state[i] = x;
              last_time[i] = t;
            }
          }
        }
        else {
          // Step failed - no progress made
          checker();
          dt = dt_step;
        }
      }

      t = t_target;

      if(apply_fixed_events_at_time(x,t,m_fixed))
      {
        obs(x,t);
        reset_memory(fired,last_val,last_state,last_time,x,t);
      }
      else {
        obs(x,t);
      }

      ++it;
    }
    return steps;
  }

  /*--------------------------------------------------------------------------
   Dense-output mode
   --------------------------------------------------------------------------*/
  template<class Obs,class Checker>
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

    m_st.initialize(x, *it, dt);

    if(apply_fixed_events_at_time(x,*it,m_fixed))
      ;

    obs(x,*it);
    ++it;
    if(it == end) return 0;

    std::vector<double> last_val(m_root.size(),std::numeric_limits<double>::quiet_NaN());
    std::vector<State>  last_state(m_root.size(),x);
    std::vector<Time>   last_time(m_root.size(), times.front());
    std::vector<size_t> fired(m_root.size(),0);

    while(it != end)
    {
      Time t_start = m_st.current_time();
      m_st.do_step(m_sys);
      Time t_end   = m_st.current_time();
      ++steps;

      // Dense output do_step always advances time
      checker();
      checker.reset();
      dt = m_st.current_time_step();

      bool event_happened = false;

      while(it != end &&
            !event_happened &&
            less_eq_with_sign(*it, t_end, dt))
      {
        Time t_eval = *it;

        if(scalar_value(t_eval) < scalar_value(t_start)){
          ++it;
          continue;
        }

        m_st.calc_state(t_eval, x);

        if(apply_fixed_events_at_time(x,t_eval,m_fixed))
        {
          obs(x,t_eval);
          reset_memory(fired,last_val,last_state,last_time,x,t_eval);
          ++it;
          event_happened = true;
          continue;
        }

        for(size_t i=0;i<m_root.size();++i)
        {
          double f_now = scalar_value(m_root[i].func(x,t_eval));

          if(fired[i] < max_trigger &&
             !std::isnan(last_val[i]) &&
             last_val[i] * f_now < 0.0)
          {
            localize_root_dense(
              i,
              last_state[i], last_time[i],
                                      x, t_eval,
                                      last_val[i], f_now,
                                      root_tol);

            Time t_root = t_eval;
            obs(x, t_root);

            State x_after = x;
            apply_event(x_after,
                        m_root[i].state_index,
                        m_root[i].value,
                        m_root[i].method);

            fired[i]++;
            obs(x_after, t_root + Time(1e-15));
            x = x_after;

            // Reinitialize dense stepper after discontinuity
            reset_stepper_unified(m_st, x, t_eval, dt);
            m_st.initialize(x, t_eval, dt);

            // Reset root tracking but keep fired counts
            for(size_t j = 0; j < m_root.size(); ++j) {
              last_val[j] = std::numeric_limits<double>::quiet_NaN();
              last_state[j] = x;
              last_time[j] = t_eval;
            }

            ++it;
            event_happened = true;
            break;
          }
          else {
            last_val[i]   = f_now;
            last_state[i] = x;
            last_time[i]  = t_eval;
          }
        }

        if(!event_happened){
          obs(x,t_eval);
          ++it;
        }
      }
    }
    return steps;
  }

private:

  void reset_memory(
      std::vector<size_t>& fired,
      std::vector<double>& last_val,
      std::vector<State>&  last_state,
      std::vector<Time>&   last_time,
      const State& x, Time t)
  {
    std::fill(last_val.begin(), last_val.end(), std::numeric_limits<double>::quiet_NaN());
    std::fill(fired.begin(),   fired.end(),   0);

    for(size_t i=0;i<last_state.size();++i){
      last_state[i] = x;
      last_time[i]  = t;
    }
  }

  /* Root localization via controlled stepping */
  template<class Checker>
  void localize_root_controlled(
      size_t idx,
      State xa, Time ta,
      State& xb, Time& tb,
      double fa, double fb,
      double tol,
      Checker& checker)
  {
    while(std::abs(scalar_value(tb - ta)) > tol)
    {
      Time tm = (ta + tb) / 2.0;

      State xm = xa;
      Time t_tmp = ta;
      Time dt_tmp = (tm - ta) / 10.0;

      while(detail::less_with_sign(t_tmp, tm, dt_tmp))
      {
        Time step_dt = min_abs(dt_tmp, tm - t_tmp);
        auto result = m_st.try_step(m_sys, xm, t_tmp, step_dt);

        if(result == success) {
          checker();
          checker.reset();
        } else {
          checker();
        }
      }

      double fm = scalar_value(m_root[idx].func(xm,tm));

      if(fa * fm <= 0.0){ tb = tm; xb = xm; fb = fm; }
      else             { ta = tm; xa = xm; fa = fm; }
    }
  }

  /* Root localization via dense interpolation */
  void localize_root_dense(
      size_t idx,
      State xa, Time ta,
      State& xb, Time& tb,
      double fa, double fb,
      double tol)
  {
    while(std::abs(scalar_value(tb - ta)) > tol)
    {
      Time tm = (ta + tb) / 2.0;
      State xm;
      m_st.calc_state(tm, xm);

      double fm = scalar_value(m_root[idx].func(xm,tm));

      if(fa * fm <= 0.0){ tb = tm; xb = xm; fb = fm; }
      else             { ta = tm; xa = xm; fa = fm; }
    }
  }

private:
  Stepper& m_st;
  System&  m_sys;
  const std::vector<FixedEvent<typename State::value_type>>& m_fixed;
  const std::vector<RootEvent<State,Time>>&                  m_root;
};

/*==============================================================================
 7. Public API — integrate_times (controlled)
 ==============================================================================*/

template<class Stepper,class System,class State,
         class TimeIterator,class Time,class Observer>
size_t integrate_times(
    Stepper stepper,
    System system,
    State& x,
    TimeIterator t_begin,
    TimeIterator t_end,
    Time dt,
    Observer obs,
    const std::vector<FixedEvent<typename State::value_type>>& fixed,
    const std::vector<RootEvent<State,Time>>& root,
    StepChecker& checker,
    double root_tol = 1e-8,
    size_t max_trigger_root = 1,
    controlled_stepper_tag = controlled_stepper_tag())
{
  auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);

  EventEngine<Stepper,System,State,Time> eng(stepper,system,fixed,root);

  return eng.process_controlled(
    x,times,dt,obs,
    checker,root_tol,max_trigger_root);
}

/*==============================================================================
 8. Public API — integrate_times_dense
 ==============================================================================*/

template<class Stepper,class System,class State,
         class TimeIterator,class Time,class Observer>
size_t integrate_times_dense(
    Stepper stepper,
    System system,
    State& x,
    TimeIterator t_begin,
    TimeIterator t_end,
    Time dt,
    Observer obs,
    const std::vector<FixedEvent<typename State::value_type>>& fixed,
    const std::vector<RootEvent<State,Time>>& root,
    StepChecker& checker,
    double root_tol = 1e-8,
    size_t max_trigger_root = 1,
    dense_output_stepper_tag = dense_output_stepper_tag())
{
  auto times = merge_user_and_event_times<Time>(t_begin, t_end, fixed);

  EventEngine<Stepper,System,State,Time> eng(stepper,system,fixed,root);

  return eng.process_dense(
    x,times,dt,obs,
    checker,root_tol,max_trigger_root);
}

} // namespace detail

using detail::FixedEvent;
using detail::RootEvent;
using detail::EventMethod;
using detail::integrate_times;
using detail::integrate_times_dense;

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
