/*
 BDF Controller — CVODE-style step-size and order control
 ==========================================================

 Wraps bdf_stepper (Nordsieck-based) and provides the full CVODE
 step acceptance/rejection pipeline with internal retry loop:

 1. cvAdjustParams: apply pending order/step-size changes
 2. Internal retry loop:
 a. Predict Nordsieck array (cvPredict)
 b. Set BDF coefficients (cvSet)
 c. Newton solve (cvNls) with Jacobian retry on stale J
 d. Error test (cvDoErrorTest) with:
 - nef counter: after MXNEF1 failures, force order reduction
 - q==1 restart: reload zn[1] = h*f(tn, zn[0])
 - LONG_WAIT for qwait after restart
 3. On success: cvCompleteStep, cvPrepareNextStep, cvSetEta

 Works for both double and AD types.  The stepper's WRMS norms
 are already AD-aware (include derivative components), so the
 controller needs no special AD handling.

 References:
 - CVODE/VODE (Brown, Byrne & Hindmarsh, 1989)
 - Hairer & Wanner, "Solving ODEs II", Ch. III
 - SUNDIALS source: sundials/src/cvode/cvode.c

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_BDF_CONTROLLER_HPP_INCLUDED
#define CPPODE_BDF_CONTROLLER_HPP_INCLUDED

#include <cmath>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <cppode/cppode_bdf.hpp>

namespace cppode {

// Scalar value extraction — reuse from bdf_detail
namespace bdf_detail {
// scalar_value: already defined in cppode_bdf.hpp
}

// ============================================================================
//  bdf_controller<Stepper>
//
//  Unified controller for both double and AD types.
//  Error norms come from the stepper and are already AD-aware.
// ============================================================================

template<class Stepper>
class bdf_controller
{
public:

  typedef Stepper                              stepper_type;
  typedef typename stepper_type::value_type    value_type;
  typedef typename stepper_type::state_type    state_type;
  typedef typename stepper_type::deriv_type    deriv_type;
  typedef value_type                           time_type;   // external: matches value_type for API compat
  typedef typename stepper_type::time_type     scalar_time_type; // internal: scalar for stepper calls
  typedef typename stepper_type::resizer_type  resizer_type;

  typedef state_wrapper<state_type>            wrapped_state_type;

  typedef controlled_stepper_tag               stepper_category;
  typedef bdf_controller<Stepper>              controller_type;
  // ====================================================================
  //  Constructor
  // ====================================================================

  bdf_controller(
    double atol = 1.0e-6,
    double rtol = 1.0e-6)
    : m_atol(atol)
    , m_rtol(rtol)
    , m_n_accepted(0)
    , m_n_rejected(0)
  {}
  // ====================================================================
  //  try_step: in-place
  // ====================================================================

  template<class System>
  controlled_step_result try_step(
      System system,
      state_type& x,
      time_type& t,
      time_type& dt)
  {
    m_xnew_resizer.adjust_size(x, [this](const auto& s) {
      return this->resize_xnew(s);
    });

    controlled_step_result result = try_step(system, x, t, m_xnew.m_v, dt);

    if (result == success) {
      std::swap(x, m_xnew.m_v);
    }
    return result;
  }
  // ====================================================================
  //  try_step: separate input/output — CVODE cvStep equivalent
  //
  //  Contains an internal retry loop matching CVODE's cvStep:
  //  - Newton convergence failures: retry with fresh Jacobian (ncf)
  //  - Error test failures: retry with reduced h (nef), with
  //    order reduction after MXNEF1 failures, and full restart
  //    at order 1 if needed.
  //
  //  Returns `success` when a step is accepted, `fail` only for
  //  truly unrecoverable failures (max ncf/nef exceeded).
  //  In normal operation, this function always returns `success`
  //  because it retries internally.
  // ====================================================================

  template<class System>
  controlled_step_result try_step(
      System system,
      const state_type& x,
      time_type& t,
      state_type& x_out,
      time_type& dt)
  {
    using std::pow;
    using std::abs;

    // --- Resize ---
    m_xerr_resizer.adjust_size(x, [this](const auto& s) {
      return this->resize_xerr(s);
    });

    // --- Initialize stepper on first call ---
    if (!m_stepper.is_initialized()) {
      auto& sys = system;
      auto& deriv_func = sys.first;

      m_ftemp_resizer.adjust_size(x, [this](const auto& s) {
        return this->resize_ftemp(s);
      });
      deriv_func(x, m_ftemp.m_v, value_type(t));

      m_stepper.set_tolerances(m_atol, m_rtol);
      m_stepper.initialize(x, t, m_ftemp.m_v, dt);
    }

    // ================================================================
    //  CVODE cvStep equivalent — internal retry loop
    // ================================================================

    int ncf = 0;   // Newton convergence failure count (this step attempt)
    int nef = 0;   // Error test failure count (this step attempt)

    // CVODE nflag equivalent: controls whether we force a Jacobian setup
    // FIRST_CALL or PREV_ERR_FAIL → normal setup decision (convfail = CV_NO_FAILURES)
    // PREV_CONV_FAIL → force setup (convfail = CV_FAIL_OTHER)
    enum class NFlag { first_call, prev_err_fail, prev_conv_fail };
    NFlag nflag = NFlag::first_call;

    // Stepper's convfail type
    using convfail_t = typename std::decay_t<decltype(m_stepper)>::convfail_t;

    for (;;) {

      // --- CVODE: callSetup and convfail decision (cvNlsNewton) ---
      //
      // callSetup = (nflag == PREV_CONV_FAIL) || (nflag == PREV_ERR_FAIL)
      //           || nst==0 || nst >= nstlp+MSBP || |gamrat-1|>DGMAX
      //
      // convfail = (nflag == FIRST_CALL || nflag == PREV_ERR_FAIL)
      //          ? CV_NO_FAILURES : CV_FAIL_OTHER
      //
      // Key insight: PREV_ERR_FAIL forces callSetup but with
      // convfail=CV_NO_FAILURES, so lsetup may reuse the cached
      // Jacobian if gamma hasn't drifted much.  PREV_CONV_FAIL
      // forces both callSetup AND a fresh Jacobian.
      bool force_setup = (nflag == NFlag::prev_conv_fail)
      || (nflag == NFlag::prev_err_fail);

      convfail_t convfail = (nflag == NFlag::first_call || nflag == NFlag::prev_err_fail)
        ? convfail_t::no_failures
      : convfail_t::fail_other;

      // --- Call the BDF stepper: rescale + predict + set + Newton ---
      m_stepper.do_step(system, x, t, x_out, dt, m_xerr.m_v, force_setup, convfail);

      // ============================================================
      //  Newton convergence check (CVODE: cvHandleNFlag)
      // ============================================================
      if (!m_stepper.newton_converged()) {
        ++ncf;
        m_stepper.set_etamax(1.0);   // prevent h increase after recovery

        // Max convergence failures → unrecoverable
        if (ncf >= bdf_constants::MXNCF) {
          ++m_n_rejected;
          return fail;
        }

        // Reduce step size and retry (CVODE: cvHandleNFlag)
        double eta_cf = std::max(
          static_cast<double>(bdf_constants::ETACF),
          static_cast<double>(bdf_constants::ETAMIN));
        m_stepper.invalidate_lu();
        m_stepper.rescale(eta_cf);
        dt = m_stepper.h();

        nflag = NFlag::prev_conv_fail;
        ++m_n_rejected;
        continue;   // retry: predict again
      }

      // ============================================================
      //  Error test (CVODE: cvDoErrorTest)
      // ============================================================
      double dsm = m_stepper.error_norm();

      if (dsm <= 1.0) {
        // === Step accepted — break out of retry loop ===

        time_type t_new = t + m_stepper.h();

        // Complete step: update Nordsieck array (CVODE: cvCompleteStep)
        m_stepper.complete_step();
        m_stepper.set_tn_current(t_new);

        // Snapshot Nordsieck for dense output BEFORE order/stepsize changes
        m_stepper.prepare_dense_output();

        // --- Order/step-size selection (CVODE: cvPrepareNextStep) ---
        cvPrepareNextStep(dsm);

        // --- etamax for next step (CVODE: after cvPrepareNextStep) ---
        if (m_stepper.nst() <= bdf_constants::SMALL_NST) {
          m_stepper.set_etamax(bdf_constants::ETAMX2);
        } else {
          m_stepper.set_etamax(bdf_constants::ETAMX3);
        }

        // Advance time
        t = t_new;

        // Set dt for next step
        dt = m_stepper.h() * m_eta;

        // Rescale Nordsieck if eta != 1
        if (std::abs(m_eta - 1.0) > 1e-14) {
          m_stepper.rescale(m_eta);
        }

        ++m_n_accepted;
        return success;
      }

      // === Error test failed ===
      ++nef;
      nflag = NFlag::prev_err_fail;
      m_stepper.restore();

      // Max error test failures → unrecoverable
      if (nef >= bdf_constants::MXNEF) {
        ++m_n_rejected;
        return fail;
      }

      // Set etamax = 1 to prevent step size increase after recovery
      m_stepper.set_etamax(1.0);

      if (nef <= bdf_constants::MXNEF1) {
        // --- Normal error test retry: reduce h and try again ---
        int L = m_stepper.current_order() + 1;
        double eta = 1.0 / (pow(bdf_constants::BIAS2 * dsm, 1.0 / L)
                              + bdf_constants::ADDON);
        eta = std::max(bdf_constants::ETAMIN, eta);

        // After SMALL_NEF failures, additionally clamp eta
        if (nef >= bdf_constants::SMALL_NEF) {
          eta = std::min(eta, bdf_constants::ETAMXF);
        }

        m_stepper.rescale(eta);
        dt = m_stepper.h();
        ++m_n_rejected;
        continue;   // retry
      }

      // --- After MXNEF1 failures: force order reduction ---
      if (m_stepper.current_order() > 1) {
        double eta = bdf_constants::ETAMIN;
        m_stepper.set_order_for_next_step(m_stepper.current_order() - 1);
        m_stepper.set_qwait(m_stepper.current_order() + 1);  // L = new q + 1
        m_stepper.rescale(eta);
        dt = m_stepper.h();
        ++m_n_rejected;
        continue;   // retry
      }

      // --- At order 1: full restart — reload zn[1] = h*f(tn, zn[0]) ---
      {
        double eta = bdf_constants::ETAMIN;
        // Apply eta to h directly (CVODE: h *= eta; hscale = h)
        time_type h_new = m_stepper.h() * eta;
        m_stepper.set_hscale(h_new);

        m_stepper.set_qwait(bdf_constants::LONG_WAIT);

        // Reload zn[1] = h_new * f(tn, zn[0])
        auto& deriv_func = system.first;
        m_stepper.reload_zn1_from_f(deriv_func);

        dt = h_new;
        ++m_n_rejected;
        continue;   // retry
      }

    } // end retry loop
  }

private:

  // ====================================================================
  //  cvPrepareNextStep — CVODE-faithful order/step-size selection
  // ====================================================================

  void cvPrepareNextStep(double dsm)
  {
    // If etamax == 1, defer step size or order changes (CVODE guard)
    if (m_stepper.etamax() <= 1.0 + 1e-14) {
      m_stepper.set_qwait(std::max(m_stepper.qwait(), 2));
      m_eta = 1.0;
      return;
    }

    // etaq: ratio of new to old h at current order
    double etaq = m_stepper.compute_etaq(dsm);
    int new_q = m_stepper.current_order();
    m_eta = etaq;

    // If qwait != 0, no order change considered
    if (m_stepper.qwait() != 0) {
      // eta = etaq, apply limits and return
      cvSetEta();
      return;
    }

    // qwait == 0: consider order change
    // CVODE: cv_qwait = cv_L after evaluating order change candidates
    // (L = q + 1 of the CURRENT order, before any change is applied)

    double etaqm1 = m_stepper.compute_etaqm1();
    double etaqp1 = m_stepper.compute_etaqp1();

    // cvChooseEta: pick the best among etaqm1, etaq, etaqp1
    double etam = std::max({etaqm1, etaq, etaqp1});

    if (etam < bdf_constants::THRESH) {
      m_eta = 1.0;
      new_q = m_stepper.current_order();
    } else if (etam == etaq) {
      m_eta = etaq;
      new_q = m_stepper.current_order();
    } else if (etam == etaqm1) {
      m_eta = etaqm1;
      new_q = m_stepper.current_order() - 1;
    } else {
      m_eta = etaqp1;
      new_q = m_stepper.current_order() + 1;
      // CVODE: store Delta_n in zn[qmax] for order increase (cvChooseEta)
      m_stepper.save_acor_to_zn_qmax();
    }

    // Apply order change if needed
    if (new_q != m_stepper.current_order()) {
      m_stepper.set_order_for_next_step(new_q);
    }

    // CVODE: qwait = L of the NEW order (set_order_for_next_step updates L)
    m_stepper.set_qwait(m_stepper.current_L());

    cvSetEta();
  }

  // ====================================================================
  //  cvSetEta — apply eta limits (CVODE-faithful)
  // ====================================================================

  void cvSetEta()
  {
    if (m_eta < bdf_constants::THRESH) {
      m_eta = 1.0;
    } else {
      m_eta = std::min(m_eta, m_stepper.etamax());
      // Note: hmax limiting omitted (we don't have hmax)
    }
  }

public:
  // ====================================================================
  //  Event handling
  // ====================================================================

  void reset_after_event(time_type /*dt_before*/)
  {
    m_stepper.invalidate_lu();
  }

  void restart_from_order1(const state_type& x, time_type t, time_type dt)
  {
    m_stepper.restart_from_order1(x, t, dt);
  }
  // ====================================================================
  //  Accessors
  // ====================================================================

  stepper_type&       stepper()       { return m_stepper; }
  const stepper_type& stepper() const { return m_stepper; }

  double atol() const { return m_atol; }
  double rtol() const { return m_rtol; }
  void set_tolerances(double atol, double rtol)
  {
    m_atol = atol; m_rtol = rtol;
    m_stepper.set_tolerances(atol, rtol);
  }

  int  max_order() const { return stepper_type::max_order; }
  int  current_method_order() const { return m_stepper.current_order(); }

  // --- Diagnostics ---
  int n_accepted() const { return m_n_accepted; }
  int n_rejected() const { return m_n_rejected; }
  int n_fevals()   const { return m_stepper.n_fevals(); }
  int n_jevals()   const { return m_stepper.n_jevals(); }
  int n_setups()   const { return m_stepper.n_setups(); }
  void reset_counters() { m_n_accepted = 0; m_n_rejected = 0; m_stepper.reset_counters(); }

  // --- Profiler ---
  mutable cppode::profiler m_prof;
  void finalize_profiler() const { m_prof.merge(m_stepper.m_prof); }
  void report_profiler(const char* label = "CppODE BDF") const {
    finalize_profiler();
    m_prof.report(label);
#ifdef CPPODE_PROFILE
    m_stepper.report_setup_triggers();
#endif
  }

  template<class StateType>
  void adjust_size(const StateType& x)
  {
    resize_xerr(x);
    resize_xnew(x);
    m_stepper.adjust_size(x);
  }
private:

  // ====================================================================
  //  Resize helpers
  // ====================================================================

  template<class StateIn>
  bool resize_xerr(const StateIn& x)
  {
    return adjust_size_by_resizeability(
      m_xerr, x);
  }

  template<class StateIn>
  bool resize_xnew(const StateIn& x)
  {
    return adjust_size_by_resizeability(
      m_xnew, x);
  }

  template<class StateIn>
  bool resize_ftemp(const StateIn& x)
  {
    return adjust_size_by_resizeability(
      m_ftemp, x);
  }
  // ====================================================================
  //  Members
  // ====================================================================

  stepper_type       m_stepper;
  resizer_type       m_xerr_resizer;
  resizer_type       m_xnew_resizer;
  resizer_type       m_ftemp_resizer;
  wrapped_state_type m_xerr;
  wrapped_state_type m_xnew;
  wrapped_state_type m_ftemp;   // for initial f evaluation

  double m_atol, m_rtol;

  double m_eta = 1.0;   // step-size ratio (stored for cvPrepareNextStep)

  int m_n_accepted;
  int m_n_rejected;
};

// ============================================================================
//  Backward-compatible type alias
// ============================================================================
template<class Stepper, bool = false>
using bdf_controller_ad = bdf_controller<Stepper>;

} // namespace cppode

#endif // CPPODE_BDF_CONTROLLER_HPP_INCLUDED
