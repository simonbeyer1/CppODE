/*
 BDF Dense Output — continuous output via Lagrange interpolation
 ================================================================

 Wraps a BDF controlled stepper (bdf_controller or bdf_controller_ad)
 and provides interpolation at arbitrary times within the last
 integration step.

 The interpolation polynomial has degree k (current BDF order) and is
 exact at all k+1 history points.

 A single template class works for both double and AD types because
 Lagrange interpolation is purely arithmetic (addition + multiplication
 with scalars) — FADBAD++ derivatives propagate automatically.

 References:
 - CVODE/VODE (Brown, Byrne & Hindmarsh, 1989)
 - Hairer & Wanner, "Solving ODEs II", Ch. III
 - SUNDIALS source: sundials/src/cvode/cvode.c

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_BDF_DENSE_OUTPUT_HPP_INCLUDED
#define CPPODE_BDF_DENSE_OUTPUT_HPP_INCLUDED

#include <utility>
#include <cassert>
namespace cppode {
// ============================================================================
//  bdf_dense_output<ControlledStepper>
//
//  ControlledStepper = bdf_controller<bdf_stepper<double, ...>>
//                    or bdf_controller_ad<bdf_stepper<F<double>, ...>>
//                    or bdf_controller_ad<bdf_stepper<F<F<double>>, ...>>
// ============================================================================

template<class ControlledStepper>
class bdf_dense_output
{
public:

  // --- Type aliases ---
  typedef ControlledStepper                                controlled_stepper_type;
  typedef typename controlled_stepper_type::stepper_type   stepper_type;
  typedef typename controlled_stepper_type::state_type     state_type;
  typedef typename controlled_stepper_type::value_type     value_type;
  typedef typename controlled_stepper_type::time_type      time_type;
  typedef typename controlled_stepper_type::resizer_type   resizer_type;

  typedef state_wrapper<state_type>  wrapped_state_type;
  typedef dense_output_stepper_tag   stepper_category;
  // ====================================================================
  //  Constructor
  // ====================================================================

  bdf_dense_output()
    : m_stepper()
    , m_x1(), m_x2()
    , m_current_state_x1(true)
    , m_t(), m_t_old(), m_dt()
  {}

  explicit bdf_dense_output(
      controlled_stepper_type&& stepper)
    : m_stepper(std::move(stepper))
    , m_x1(), m_x2()
    , m_current_state_x1(true)
    , m_t(), m_t_old(), m_dt()
  {}

  // Non-copyable (stepper hierarchy contains non-copyable members)
  bdf_dense_output(const bdf_dense_output&) = delete;
  bdf_dense_output& operator=(const bdf_dense_output&) = delete;

  // Movable
  bdf_dense_output(bdf_dense_output&&) = default;
  bdf_dense_output& operator=(bdf_dense_output&&) = default;
  // ====================================================================
  //  initialize
  //
  //  Both state buffers are set to x0.  No valid interpolation interval
  //  exists until the first do_step().
  // ====================================================================

  template<class StateType>
  void initialize(const StateType& x0, time_type t0, time_type dt0)
  {
    m_resizer.adjust_size(x0, [this](const auto& s) {
      return this->resize_impl(s);
    });

    get_current_state() = x0;
    get_old_state()     = x0;

    m_t     = t0;
    m_t_old = t0;
    m_dt    = dt0;
  }
  // ====================================================================
  //  do_step
  //
  //  Retries with adaptive step-size control until a step is accepted.
  //  Returns (t_old, t_new) — the interval where calc_state is valid.
  //
  //  The BDF stepper's history serves directly as interpolation data,
  //  so no separate prepare_dense_output phase is needed.
  // ====================================================================

  template<class System>
  std::pair<time_type, time_type> do_step(System system)
  {
    failed_step_checker fail_checker;
    controlled_step_result result = fail;

    m_t_old = m_t;

    do {
      result = m_stepper.try_step(
        system,
        get_current_state(),   // input state
        m_t,                   // time (advanced on success)
        get_old_state(),       // output → receives new state
        m_dt);                 // step size (adapted)

      fail_checker();
    }
    while (result == fail);

    // Dense output snapshot is saved inside the controller's try_step,
    // after complete_step but before order/stepsize modifications.
    // Do NOT call prepare_dense_output() here — the Nordsieck array
    // has already been modified for the next step at this point.

    // Swap buffers: "old" buffer now holds the newest state
    toggle_current_state();

    return std::make_pair(m_t_old, m_t);
  }
  // ====================================================================
  //  calc_state: interpolate at arbitrary time
  //
  //  Lagrange interpolation through the BDF history points.
  //  Valid for t in [previous_time(), current_time()].
  //
  //  For AD types, derivatives propagate automatically because
  //  Lagrange interpolation is purely arithmetic.
  // ====================================================================

  template<class StateOut>
  void calc_state(time_type t, StateOut& x)
  {
    auto _tp = m_stepper.stepper().m_prof.timer(cppode::prof_cat::dense_interp);
    // Nordsieck polynomial evaluation from the dense output snapshot.
    // x(t) = sum_{j=0}^{q} zn_dense[j] * s^j
    // where s = (t - tn_dense) / h_dense
    //
    // Evaluated via Horner's method for numerical stability.

    auto& stp = m_stepper.stepper();
    assert(stp.has_dense_output() && "dense output snapshot not available");

    const int    q  = stp.dense_order();
    const auto   h  = stp.dense_h();
    const auto   tn = stp.dense_tn();
    const auto   s  = (t - tn) / h;
    const size_t n  = stp.dense_zn(0).size();

    // Horner: x = zn[q]*s + zn[q-1],  then x = x*s + zn[q-2], ...
    const auto& zq = stp.dense_zn(q);
    for (size_t i = 0; i < n; ++i)
      x[i] = zq[i];

    for (int j = q - 1; j >= 0; --j) {
      const auto& zj = stp.dense_zn(j);
      for (size_t i = 0; i < n; ++i)
        x[i] = x[i] * s + zj[i];
    }
  }
  // ====================================================================
  //  reinitialize_at_event
  //
  //  Full reset after a state discontinuity:
  //  - Both state buffers set to post-event state
  //  - Time set to event time
  //  - Controller PI state cleared
  //  - BDF stepper history discarded, restart at order 1
  // ====================================================================

  void reinitialize_at_event(state_type& x_event,
                             time_type t_event,
                             time_type& dt_before)
  {
    get_current_state() = x_event;
    get_old_state()     = x_event;

    m_t     = t_event;
    m_t_old = t_event;
    m_dt    = dt_before;

    // Reset controller PI state
    m_stepper.reset_after_event(dt_before);

    // Reset BDF history to order 1
    m_stepper.stepper().restart_from_order1(x_event, t_event, dt_before);
  }
  // ====================================================================
  //  Accessors
  // ====================================================================

  const state_type& current_state()  const { return get_current_state(); }
  const state_type& previous_state() const { return get_old_state(); }

  time_type current_time()      const { return m_t; }
  time_type previous_time()     const { return m_t_old; }
  time_type current_time_step() const { return m_dt; }

  controlled_stepper_type&       controlled_stepper()       { return m_stepper; }
  const controlled_stepper_type& controlled_stepper() const { return m_stepper; }

  // --- Diagnostics (delegated to controller/stepper) ---
  int n_accepted() const { return m_stepper.n_accepted(); }
  int n_rejected() const { return m_stepper.n_rejected(); }
  int n_fevals()   const { return m_stepper.n_fevals(); }
  int n_jevals()   const { return m_stepper.n_jevals(); }
  int n_setups()   const { return m_stepper.n_setups(); }
  int current_method_order() const { return m_stepper.current_method_order(); }
  void reset_counters() { m_stepper.reset_counters(); }

  template<class StateType>
  void adjust_size(const StateType& x)
  {
    m_stepper.adjust_size(x);
    resize_impl(x);
  }
private:

  // --- Double-buffered state storage ---

  state_type& get_current_state()
  { return m_current_state_x1 ? m_x1.m_v : m_x2.m_v; }

  const state_type& get_current_state() const
  { return m_current_state_x1 ? m_x1.m_v : m_x2.m_v; }

  state_type& get_old_state()
  { return m_current_state_x1 ? m_x2.m_v : m_x1.m_v; }

  const state_type& get_old_state() const
  { return m_current_state_x1 ? m_x2.m_v : m_x1.m_v; }

  void toggle_current_state()
  { m_current_state_x1 = !m_current_state_x1; }

  template<class StateIn>
  bool resize_impl(const StateIn& x)
  {
    bool resized = false;
    resized |= adjust_size_by_resizeability(
      m_x1, x);
    resized |= adjust_size_by_resizeability(
      m_x2, x);
    return resized;
  }

  // --- Members ---

  controlled_stepper_type m_stepper;
  resizer_type            m_resizer;

  wrapped_state_type      m_x1, m_x2;
  bool                    m_current_state_x1;

  time_type               m_t;
  time_type               m_t_old;
  time_type               m_dt;
};
} // namespace cppode

#endif // CPPODE_BDF_DENSE_OUTPUT_HPP_INCLUDED
