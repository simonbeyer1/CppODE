#ifndef BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_AD_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_DENSE_OUTPUT_AD_HPP_INCLUDED

#include <utility>

#include <boost/numeric/odeint/util/bind.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/integrate/max_step_checker.hpp>

#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include "rosenbrock4_controller_fad.hpp"

namespace boost {
namespace numeric {
namespace odeint {

/**
 * Dense-output wrapper for Rosenbrock4 with AD-aware controlled stepper.
 *
 * This is a minimal, stable, Boost-compatible wrapper providing:
 *   - cubic Hermite dense output through Rosenbrock4::calc_state()
 *   - reinitialization at events (fixed-time or root events)
 *
 * The wrapper keeps two state buffers:
 *    - current_state()  : state after the accepted step
 *    - previous_state() : state before the accepted step
 *
 * After each successful step:
 *    - prepare_dense_output() is called on the underlying Rosenbrock stepper
 *    - buffer pointers are toggled
 *
 * After an event:
 *    - both buffers are set to the event state
 *    - t_old = t_new = event time
 *    - no dense-output coefficients exist → next do_step() regenerates them
 */
template<class ControlledStepper>
class rosenbrock4_dense_output_ad
{
public:
  using controlled_stepper_type = ControlledStepper;
  using unwrapped_stepper =
    typename unwrap_reference<controlled_stepper_type>::type;

  using stepper_type       = typename unwrapped_stepper::stepper_type;
  using state_type         = typename unwrapped_stepper::state_type;
  using wrapped_state_type = typename unwrapped_stepper::wrapped_state_type;
  using value_type         = typename stepper_type::value_type;
  using time_type          = typename stepper_type::time_type;
  using resizer_type       = typename stepper_type::resizer_type;

  using stepper_category = dense_output_stepper_tag;

  //----------------------------------------------------------------------
  // Constructor
  //----------------------------------------------------------------------
  explicit rosenbrock4_dense_output_ad(
      const controlled_stepper_type& stepper = controlled_stepper_type())
    : m_stepper(stepper)
    , m_x1(), m_x2()
    , m_current_state_x1(true)
    , m_t(), m_t_old(), m_dt()
  {}

  //----------------------------------------------------------------------
  // Initialization at the start of integration
  //----------------------------------------------------------------------
  template<class StateType>
  void initialize(const StateType& x0, time_type t0, time_type dt0)
  {
    // Make sure both internal buffers have the right size
    m_resizer.adjust_size(
      x0,
      detail::bind(
        &rosenbrock4_dense_output_ad::template resize_impl<StateType>,
        detail::ref(*this),
        detail::_1));

    // For the very first step, both buffers contain the same state
    get_current_state() = x0;
    get_old_state()     = x0;

    m_t     = t0;
    m_t_old = t0;
    m_dt    = dt0;
  }

  //----------------------------------------------------------------------
  // Perform one adaptive controlled step + prepare dense output
  //----------------------------------------------------------------------
  template<class System>
  std::pair<time_type,time_type> do_step(System system)
  {
    unwrapped_stepper& cs = m_stepper;

    failed_step_checker fail_checker;
    controlled_step_result result = fail;

    // Save old time before attempting the step
    m_t_old = m_t;

    // Try adaptive step until accepted
    do {
      result = cs.try_step(system,
                           get_current_state(),
                           m_t,
                           get_old_state(),
                           m_dt);
      fail_checker();
    } while(result == fail);

    // Underlying Rosenbrock4 now holds dense-output coefficients
    cs.stepper().prepare_dense_output();

    // Swap old <-> current state buffers
    toggle_current_state();

    return std::make_pair(m_t_old, m_t);
  }

  //----------------------------------------------------------------------
  // Dense output evaluation: Hermite cubic provided by Rosenbrock4
  //----------------------------------------------------------------------
  template<class StateOut>
  void calc_state(time_type t, StateOut& x)
  {
    unwrapped_stepper& cs = m_stepper;

    cs.stepper().calc_state(
        t,
        x,
        get_old_state(),  m_t_old,
        get_current_state(), m_t);
  }

  //----------------------------------------------------------------------
  // Adjust size of internal buffers (if user resizes state)
  //----------------------------------------------------------------------
  template<class StateIn>
  void adjust_size(const StateIn& x)
  {
    unwrapped_stepper& cs = m_stepper;
    cs.adjust_size(x);
    resize_impl(x);
  }

  //----------------------------------------------------------------------
  // Reinitialization after events
  //
  // After a fixed-time event or root event:
  //   - state jumps discontinuously
  //   - there is no valid step bracketing this jump
  //   - both buffers must be set to the same new state
  //   - t_old = t = event time
  //   - dt is restored to previous dt
  //   - NO dense-output coefficients must be used
  //
  // The next do_step() will correctly rebuild coefficients.
  //----------------------------------------------------------------------
  void reinitialize_at_event(const state_type& x_event,
                             time_type t_event,
                             time_type dt_before)
  {
    get_current_state() = x_event;
    get_old_state()     = x_event;

    m_t     = t_event;
    m_t_old = t_event;
    m_dt    = dt_before;
  }

  //----------------------------------------------------------------------
  // Accessors
  //----------------------------------------------------------------------
  const state_type& current_state()  const { return get_current_state(); }
  const state_type& previous_state() const { return get_old_state();     }

  time_type current_time()     const { return m_t;     }
  time_type previous_time()    const { return m_t_old; }
  time_type current_time_step() const { return m_dt;    }

private:

  //----------------------------------------------------------------------
  // Internal buffer access
  //----------------------------------------------------------------------
  state_type& get_current_state()
  { return m_current_state_x1 ? m_x1.m_v : m_x2.m_v; }

  const state_type& get_current_state() const
  { return m_current_state_x1 ? m_x1.m_v : m_x2.m_v; }

  state_type& get_old_state()
  { return m_current_state_x1 ? m_x2.m_v : m_x1.m_v; }

  const state_type& get_old_state() const
  { return m_current_state_x1 ? m_x2.m_v : m_x1.m_v; }

  //----------------------------------------------------------------------
  // Toggle front/back buffers
  //----------------------------------------------------------------------
  void toggle_current_state()
  {
    m_current_state_x1 = !m_current_state_x1;
  }

  //----------------------------------------------------------------------
  // Internal buffer resizing
  //----------------------------------------------------------------------
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

  //----------------------------------------------------------------------
  // Members
  //----------------------------------------------------------------------
  controlled_stepper_type m_stepper;
  resizer_type            m_resizer;

  wrapped_state_type      m_x1, m_x2;   // double-buffering
  bool                    m_current_state_x1;

  time_type               m_t, m_t_old, m_dt;
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif
