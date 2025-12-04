/*
 [auto_generated]
 boost/numeric/odeint/stepper/rosenbrock4_dense_output.hpp

 Dense output wrapper for Rosenbrock4 with event-compatible reset semantics.

 This version has been extended to be fully consistent with the AD-aware
 rosenbrock4_dense_output_ad.hpp, allowing both controlled steppers
 (plain double or AD types) to be used interchangeably within an
 event-driven integration framework.

 The wrapper:
 - handles Hermite cubic dense output
 - swaps internal buffers after each successful step
 - supports full reinitialization after fixed-time or root events
 - integrates safely with any ControlledStepper, even if it does not
 implement reset_after_event() (handled via SFINAE)
 */

#ifndef BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_HPP_INCLUDED

#include <utility>

#include <boost/numeric/odeint/util/bind.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/integrate/max_step_checker.hpp>

#include <boost/numeric/odeint/stepper/rosenbrock4_controller.hpp>

namespace boost {
namespace numeric {
namespace odeint {

/*==============================================================================
 SFINAE Helper: call cs.reset_after_event() only if it exists
 ==============================================================================*/

template<class CS, class Time>
auto try_reset_after_event(CS& cs, Time dt_before)
  -> decltype(cs.reset_after_event(dt_before), void())
  {
    // Case 1: ControlledStepper implements reset_after_event()
    cs.reset_after_event(dt_before);
  }

template<class CS, class Time>
void try_reset_after_event(...)
{
  // Case 2: Fallback for plain Boost steppers with NO reset API.
  // No additional action required.
}


/*==============================================================================
 Dense Output Wrapper
 ==============================================================================*/

template<class ControlledStepper>
class rosenbrock4_dense_output
{
public:
  typedef ControlledStepper controlled_stepper_type;
  typedef typename unwrap_reference<controlled_stepper_type>::type unwrapped_stepper;

  typedef typename unwrapped_stepper::stepper_type       stepper_type;
  typedef typename unwrapped_stepper::state_type         state_type;
  typedef typename unwrapped_stepper::wrapped_state_type wrapped_state_type;
  typedef typename stepper_type::time_type               time_type;
  typedef typename stepper_type::resizer_type            resizer_type;

  typedef dense_output_stepper_tag stepper_category;

  /*--------------------------------------------------------------------------
   Constructor
   --------------------------------------------------------------------------*/
  explicit rosenbrock4_dense_output(
      const controlled_stepper_type &stepper = controlled_stepper_type())
    : m_stepper(stepper)
    , m_x1(), m_x2()
    , m_current_state_x1(true)
    , m_t(), m_t_old(), m_dt()
  {}


  /*--------------------------------------------------------------------------
   Initialization
   --------------------------------------------------------------------------*/
  template<class StateType>
  void initialize(const StateType &x0, time_type t0, time_type dt0)
  {
    m_resizer.adjust_size(
      x0,
      detail::bind(
        &rosenbrock4_dense_output::template resize_impl<StateType>,
        detail::ref(*this),
        detail::_1));

    get_current_state() = x0;
    get_old_state()     = x0;

    m_t     = t0;
    m_t_old = t0;
    m_dt    = dt0;
  }


  /*--------------------------------------------------------------------------
   Perform one successful controlled step + generate dense-output coefficients
   --------------------------------------------------------------------------*/
  template<class System>
  std::pair<time_type, time_type> do_step(System system)
  {
    unwrapped_stepper &cs = m_stepper;

    failed_step_checker fail_checker;
    controlled_step_result result = fail;

    m_t_old = m_t;

    do {
      result = cs.try_step(
        system,
        get_current_state(),
        m_t,
        get_old_state(),
        m_dt);

      fail_checker();
    }
    while (result == fail);

    // Prepare Hermite cubic interpolation
    cs.stepper().prepare_dense_output();

    // Swap buffers (old ↔ new)
    toggle_current_state();

    return std::make_pair(m_t_old, m_t);
  }


  /*--------------------------------------------------------------------------
   Dense output evaluation for any t in [t_old, t]
   --------------------------------------------------------------------------*/
  template<class StateOut>
  void calc_state(time_type t, StateOut &x)
  {
    unwrapped_stepper &cs = m_stepper;

    cs.stepper().calc_state(
        t,
        x,
        get_old_state(),  m_t_old,
        get_current_state(), m_t);
  }

  template<class StateType>
  void adjust_size(const StateType& x)
  {
    unwrapped_stepper &cs = m_stepper;
    cs.adjust_size(x);
    resize_impl(x);
  }


  /*--------------------------------------------------------------------------
   Full event reset (fixed-time or root-triggered)
   Strategy B — identical semantics to AD-version
   - both buffers reset to x_event
   - time reset
   - dt restored
   - controller reset via try_reset_after_event()
   --------------------------------------------------------------------------*/
  void reinitialize_at_event(const state_type &x_event,
                             time_type t_event,
                             time_type dt_before)
  {
    get_current_state() = x_event;
    get_old_state()     = x_event;

    m_t     = t_event;
    m_t_old = t_event;
    m_dt    = dt_before;

    // Safely reset controller state if supported
    try_reset_after_event(m_stepper, dt_before);
  }


  /*--------------------------------------------------------------------------
   Accessors
   --------------------------------------------------------------------------*/
  const state_type& current_state()  const { return get_current_state(); }
  const state_type& previous_state() const { return get_old_state();     }

  time_type current_time()      const { return m_t;     }
  time_type previous_time()     const { return m_t_old; }
  time_type current_time_step() const { return m_dt;    }


private:

  /*--------------------------------------------------------------------------
   Internal state buffer access
   --------------------------------------------------------------------------*/
  state_type& get_current_state()
  {
    return m_current_state_x1 ? m_x1.m_v : m_x2.m_v;
  }

  const state_type& get_current_state() const
  {
    return m_current_state_x1 ? m_x1.m_v : m_x2.m_v;
  }

  state_type& get_old_state()
  {
    return m_current_state_x1 ? m_x2.m_v : m_x1.m_v;
  }

  const state_type& get_old_state() const
  {
    return m_current_state_x1 ? m_x2.m_v : m_x1.m_v;
  }


  /*--------------------------------------------------------------------------
   Toggle state buffers
   --------------------------------------------------------------------------*/
  void toggle_current_state()
  {
    m_current_state_x1 = !m_current_state_x1;
  }


  /*--------------------------------------------------------------------------
   Resize internal buffers
   --------------------------------------------------------------------------*/
  template<class StateIn>
  bool resize_impl(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(
      m_x1, x, typename is_resizeable<state_type>::type());
    resized |= adjust_size_by_resizeability(
      m_x2, x, typename is_resizeable<state_type>::type());
    return resized;
  }


  /*--------------------------------------------------------------------------
   Members
   --------------------------------------------------------------------------*/
  controlled_stepper_type m_stepper;
  resizer_type            m_resizer;

  wrapped_state_type      m_x1, m_x2;
  bool                    m_current_state_x1;

  time_type               m_t, m_t_old, m_dt;
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_HPP_INCLUDED
