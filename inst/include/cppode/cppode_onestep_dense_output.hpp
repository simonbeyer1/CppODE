/*
 Dense output wrapper for single-step methods with PI controller.

 Generic dense output for any single-step stepper + controller pair.
 Provides Hermite interpolation between steps and event reinitialization.

 Originally derived from Boost.Odeint's rosenbrock4_dense_output
 by Karsten Ahnert, Mario Mulansky, and Christoph Koke (2011-2015),
 distributed under the Boost Software License, Version 1.0.

 Modified work:
 Copyright (C) 2026 Simon Beyer

 Modifications:
 - Generalized to any single-step stepper (rosenbrock4, tsit5, ...)
 - Method order derived from stepper
 - Added reinitialize_at_event() for event-driven integration
 - Implemented SFINAE-based controller reset detection
 - Extended double-buffering for stable event handling
 */

#ifndef CPPODE_ONESTEP_DENSE_OUTPUT_HPP
#define CPPODE_ONESTEP_DENSE_OUTPUT_HPP

#include <utility>
#include <type_traits>
#include <cppode/cppode_dual_slab.hpp>
#include <cppode/cppode_onestep_controller.hpp>

namespace cppode {

// ============================================================================
//  SFINAE Helper: detect reset_after_event() on the controller
// ============================================================================

template<class CS, class Time, class = void>
struct has_reset_after_event : std::false_type {};

template<class CS, class Time>
struct has_reset_after_event<CS, Time,
                            std::void_t<decltype(std::declval<CS&>().reset_after_event(std::declval<Time>()))>
> : std::true_type {};

template<class CS, class Time>
inline typename std::enable_if<has_reset_after_event<CS, Time>::value>::type
try_reset_after_event_impl(CS& cs, Time dt_before)
{
  cs.reset_after_event(dt_before);
}

template<class CS, class Time>
inline typename std::enable_if<!has_reset_after_event<CS, Time>::value>::type
try_reset_after_event_impl(CS& /*cs*/, Time /*dt_before*/)
{
  // No-op for controllers without reset API
}

// ============================================================================
//  onestep_dense_output<ControlledStepper>
//
//  Dense output wrapper for single-step methods.  Provides continuous
//  interpolation between integration steps via the stepper's
//  prepare_dense_output() / calc_state() interface.
// ============================================================================

template<class ControlledStepper>
class onestep_dense_output
{
public:
  typedef ControlledStepper controlled_stepper_type;
  typedef typename unwrap_reference<controlled_stepper_type>::type unwrapped_stepper;

  typedef typename unwrapped_stepper::stepper_type       stepper_type;
  typedef typename unwrapped_stepper::state_type         state_type;
  typedef typename unwrapped_stepper::wrapped_state_type wrapped_state_type;
  typedef typename stepper_type::value_type              value_type;
  typedef typename stepper_type::value_type              time_type;
  typedef typename stepper_type::resizer_type            resizer_type;

  typedef dense_output_stepper_tag stepper_category;

  onestep_dense_output()
    : m_stepper()
    , m_x1(), m_x2()
    , m_current_state_x1(true)
    , m_t(), m_t_old(), m_dt()
  {}

  explicit onestep_dense_output(controlled_stepper_type &&stepper)
    : m_stepper(std::move(stepper))
    , m_x1(), m_x2()
    , m_current_state_x1(true)
    , m_t(), m_t_old(), m_dt()
  {}

  // Non-copyable, movable
  onestep_dense_output(const onestep_dense_output&) = delete;
  onestep_dense_output& operator=(const onestep_dense_output&) = delete;
  onestep_dense_output(onestep_dense_output&&) = default;
  onestep_dense_output& operator=(onestep_dense_output&&) = default;

  template<class StateType>
  void initialize(const StateType &x0, time_type t0, time_type dt0)
  {
    m_resizer.adjust_size(x0, [this](const StateType& s){ return this->resize_impl(s); });
    get_current_state() = x0;
    get_old_state()     = x0;
    m_t     = t0;
    m_t_old = t0;
    m_dt    = dt0;
  }

  template<class System>
  std::pair<time_type, time_type> do_step(System& system)
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

    cs.stepper().prepare_dense_output();
    toggle_current_state();
    return std::make_pair(m_t_old, m_t);
  }

  template<class StateOut>
  void calc_state(time_type t, StateOut &x)
  {
    unwrapped_stepper &cs = m_stepper;
    cs.stepper().calc_state(
        t, x,
        get_old_state(),     m_t_old,
        get_current_state(), m_t);
  }

  template<class StateType>
  void adjust_size(const StateType& x)
  {
    unwrapped_stepper &cs = m_stepper;
    cs.adjust_size(x);
    resize_impl(x);
  }

  void reinitialize_at_event(const state_type &x_event,
                             time_type t_event,
                             time_type dt_before)
  {
    get_current_state() = x_event;
    get_old_state()     = x_event;
    m_t     = t_event;
    m_t_old = t_event;
    m_dt    = dt_before;
    try_reset_after_event_impl(m_stepper, dt_before);
  }

  const state_type& current_state()  const { return get_current_state(); }
  const state_type& previous_state() const { return get_old_state(); }
  time_type current_time()      const { return m_t; }
  time_type previous_time()     const { return m_t_old; }
  time_type current_time_step() const { return m_dt; }

  controlled_stepper_type&       controlled_stepper()       { return m_stepper; }
  const controlled_stepper_type& controlled_stepper() const { return m_stepper; }

  // Slab-prime forwarder, mirroring the multistepper-side wrapper. Also
  // primes the dense wrapper's own m_x1 / m_x2 buffers (alternating state
  // slots used by the stepper as in/out across steps). They may still be
  // size 0 here: initialize() / resize_impl re-prime once they grow.
  void prepare_sensitivities(unsigned n_sens)
  {
    m_n_sens = n_sens;
    m_stepper.prepare_sensitivities(n_sens);
    if constexpr (detail::is_dynamic_dual<value_type>::value) {
      if (n_sens == 0) return;
      if (!m_x1.m_v.empty())
        m_x1_slab.prime(m_x1.m_v,
                        static_cast<unsigned>(m_x1.m_v.size()), n_sens);
      if (!m_x2.m_v.empty())
        m_x2_slab.prime(m_x2.m_v,
                        static_cast<unsigned>(m_x2.m_v.size()), n_sens);
    }
  }

  // Diagnostics (delegated to controller/stepper)
  int n_accepted() const { return m_stepper.n_accepted(); }
  int n_rejected() const { return m_stepper.n_rejected(); }
  int n_fevals()   const { return m_stepper.n_fevals(); }
  int n_jevals()   const { return m_stepper.n_jevals(); }
  int n_setups()   const { return m_stepper.n_setups(); }
  static constexpr int current_method_order() { return static_cast<int>(stepper_type::stepper_order); }
  void reset_counters() { m_stepper.reset_counters(); }

private:
  state_type&       get_current_state()       { return m_current_state_x1 ? m_x1.m_v : m_x2.m_v; }
  const state_type& get_current_state() const { return m_current_state_x1 ? m_x1.m_v : m_x2.m_v; }
  state_type&       get_old_state()           { return m_current_state_x1 ? m_x2.m_v : m_x1.m_v; }
  const state_type& get_old_state()     const { return m_current_state_x1 ? m_x2.m_v : m_x1.m_v; }
  void toggle_current_state() { m_current_state_x1 = !m_current_state_x1; }

  template<class StateIn>
  bool resize_impl(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(m_x1, x);
    resized |= adjust_size_by_resizeability(m_x2, x);
    if (resized && m_n_sens != 0) {
      if constexpr (detail::is_dynamic_dual<value_type>::value) {
        m_x1_slab.prime(m_x1.m_v,
                        static_cast<unsigned>(m_x1.m_v.size()), m_n_sens);
        m_x2_slab.prime(m_x2.m_v,
                        static_cast<unsigned>(m_x2.m_v.size()), m_n_sens);
      }
    }
    return resized;
  }

  controlled_stepper_type m_stepper;
  resizer_type            m_resizer;
  wrapped_state_type      m_x1, m_x2;
  // SoA tangent slabs for the dynamic-dual heap path (empty stubs otherwise).
  detail::tangent_slab<value_type> m_x1_slab;
  detail::tangent_slab<value_type> m_x2_slab;
  unsigned                m_n_sens = 0;
  bool                    m_current_state_x1;
  time_type               m_t, m_t_old, m_dt;
};

} // namespace cppode

#endif // CPPODE_ONESTEP_DENSE_OUTPUT_HPP
