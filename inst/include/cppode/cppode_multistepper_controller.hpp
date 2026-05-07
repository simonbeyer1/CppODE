/*
 Multistepper Controller: step-size and order control
 =====================================================

 Wraps cppode::multistepper (Nordsieck-based) and provides the step
 acceptance/rejection pipeline with internal retry loop:

 1. Apply pending order/step-size changes
 2. Internal retry loop:
    a. Predict Nordsieck array (ndfPredict)
    b. Set NDF/BDF/Adams coefficients
    c. Corrector solve (Newton for BDF/NDF, PECE for Adams)
       with Jacobian retry on stale J (BDF/NDF side)
    d. Error test with:
       - nef counter: after MXNEF1 failures, force order reduction
       - q==1 restart: reload zn[1] = h*f(tn, zn[0])
       - LONG_WAIT for qwait after restart
 3. On success: complete step, prepare next step, set eta

 Works for any multistepper instantiation: pure BDF (with optional
 NDF kappa) and pure Adams: and for both double and AD types.

 References:
 - Shampine & Reichelt, "The MATLAB ODE Suite",
   SIAM J. Sci. Comput. 18(1), 1997 (NDF kappa modification)
 - Hairer & Wanner, "Solving ODEs II", Ch. III

 Copyright (C) 2026 Simon Beyer

 Portions of this file (step acceptance/rejection logic, eta/qwait
 control, BIAS safety factors, order-change and J-staleness pipeline)
 are derived from SUNDIALS/CVODE(S) (cvode.c), Copyright (c) 2002-2024
 Lawrence Livermore National Security and Southern Methodist
 University, distributed under the BSD-3-Clause license.  See
 inst/COPYRIGHTS for the full license text.
 */

#ifndef CPPODE_MULTISTEPPER_CONTROLLER_HPP_INCLUDED
#define CPPODE_MULTISTEPPER_CONTROLLER_HPP_INCLUDED

#include <cmath>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>
#include <cppode/cppode_multistepper.hpp>
#include <cppode/cppode_utils.hpp>
#include <cppode/cppode_dual_slab.hpp>

namespace cppode {

// (scalar_value lives in cppode::ndf_detail, defined in cppode_multistepper.hpp)

// ============================================================================
//  multistepper_controller<Stepper>
//
//  Unified step-size / order controller for any cppode::multistepper
//  instantiation.  Works for both double and AD types; error norms
//  come from the stepper and are already AD-aware.
// ============================================================================

template<class Stepper>
class multistepper_controller
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
  typedef multistepper_controller<Stepper>     controller_type;
  // ====================================================================
  //  Constructor
  // ====================================================================

  multistepper_controller(
    double atol = 1.0e-6,
    double rtol = 1.0e-6)
    : m_atol(atol)
    , m_rtol(rtol)
    , m_n_accepted(0)
    , m_n_rejected(0)
  {}

  // Slab-bound dual.tan_ pointers reference our own buffer storage. Copying
  // would build a separate buffer with the same address-bound duals: UB.
  // Moves are safe: std::vector::move preserves data() for both the slab
  // storage and the dual-element vectors, so embedded tan_ pointers keep
  // pointing at valid memory in the moved-to instance.
  multistepper_controller(const multistepper_controller&)            = delete;
  multistepper_controller& operator=(const multistepper_controller&) = delete;
  multistepper_controller(multistepper_controller&&)                 = default;
  multistepper_controller& operator=(multistepper_controller&&)      = default;

  // Stores n_sens, primes the inner stepper's slabs, and primes our own
  // m_xerr/m_xnew/m_ftemp slabs (they may still be size 0 here: the
  // resize_* helpers call prime again once the buffers grow). No-op for
  // non-dynamic-dual value_type.
  void prepare_sensitivities(unsigned n_sens)
  {
    m_n_sens = n_sens;
    m_stepper.prepare_sensitivities(n_sens);
    if constexpr (detail::is_dynamic_dual<value_type>::value) {
      if (n_sens == 0) return;
      if (!m_xerr.m_v.empty())
        m_xerr_slab.prime(m_xerr.m_v,
                          static_cast<unsigned>(m_xerr.m_v.size()), n_sens);
      if (!m_xnew.m_v.empty())
        m_xnew_slab.prime(m_xnew.m_v,
                          static_cast<unsigned>(m_xnew.m_v.size()), n_sens);
      if (!m_ftemp.m_v.empty())
        m_ftemp_slab.prime(m_ftemp.m_v,
                           static_cast<unsigned>(m_ftemp.m_v.size()), n_sens);
    }
  }

  // ====================================================================
  //  PID step-size control modes
  //  ===========================
  //
  //  The controller supports three step-size control strategies,
  //  selected via set_pid_mode():
  //
  //  pid_mode::none
  //      Classical CVODE-style I-controller with BIAS2 safety
  //      factor. compute_etaq returns 1/((BIAS2*dsm)^(1/L) + ADDON).
  //      Order selection (cvChooseEta) compares the three classical
  //      candidates etaqm1/etaq/etaqp1 directly.
  //
  //  pid_mode::intermediate
  //      Classical I-controller AND classical order selection: i.e.
  //      everything that pid_mode::none does: and THEN a geometric
  //      low-pass filter is applied to the final m_eta:
  //          log(eta_filtered) = (1-alpha)*log(eta) + alpha*log(eta_prev)
  //      with alpha in [0, 1] (default 0.4). This smooths the
  //      step-size sequence WITHOUT interfering with order selection.
  //      Order changes invalidate the filter history.
  //
  //  pid_mode::full
  //      Söderlind H211b digital filter REPLACES the I-controller
  //      at the current order:
  //          eta = safety * r_n^(-b1) * r_{n-1}^(-b2) * (h_{n-1}/h_n)^(a2)
  //      with H211b coefficients b1 = b2 = a2 = 1/(2L) and an
  //      explicit safety factor (default 0.8). The candidates for
  //      order-down (etaqm1) and order-up (etaqp1) are also lifted
  //      onto the same safety-only scale by removing their BIAS1
  //      and BIAS3 conservatism factors and re-applying m_pid_safety,
  //      so the cvChooseEta max-comparison is fair across all three
  //      candidates and order selection is not biased against the
  //      H211b path. History is reset on order changes, step
  //      failures, and external events.
  //
  //  References:
  //    Söderlind, "Digital Filters in Adaptive Time-Stepping",
  //    ACM TOMS 29(1), 2003, eq. (24)–(25).
  //    Söderlind & Wang, "Adaptive Time-Stepping and Computational
  //    Stability", J. Comput. Appl. Math. 185, 2006.
  // ====================================================================

  enum class pid_mode : int {
    none         = 0,
    intermediate = 1,
    full         = 2
  };

  void     set_pid_mode(pid_mode m)   { m_pid_mode = m; reset_filter_history(); }
  pid_mode get_pid_mode() const       { return m_pid_mode; }

  // Tunables (defaults are reasonable; expose for benchmarking)
  void   set_pid_safety(double s)     { m_pid_safety = s; }
  double pid_safety() const           { return m_pid_safety; }
  void   set_pid_lp_alpha(double a)   { m_pid_lp_alpha = a; }
  double pid_lp_alpha() const         { return m_pid_lp_alpha; }
  // ====================================================================
  //  try_step: in-place
  // ====================================================================

  template<class System>
  controlled_step_result try_step(
      System& system,
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
  //  try_step: separate input/output: step acceptance/rejection pipeline
  //
  //  Contains an internal retry loop matching cvStep:
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
      System& system,
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
    //  internal retry loop
    // ================================================================

    int ncf = 0;   // Newton convergence failure count (this step attempt)
    int nef = 0;   // Error test failure count (this step attempt)

    // nflag: controls whether we force a Jacobian setup
    // FIRST_CALL or PREV_ERR_FAIL → normal setup decision (convfail = CV_NO_FAILURES)
    // PREV_CONV_FAIL → force setup (convfail = CV_FAIL_OTHER)
    enum class NFlag { first_call, prev_err_fail, prev_conv_fail };
    NFlag nflag = NFlag::first_call;

    // Stepper's convfail type
    using convfail_t = typename std::decay_t<decltype(m_stepper)>::convfail_t;

    for (;;) {

      // --- HMIN guard ------------------------------------------------
      // Abort if the step size has collapsed relative to the current
      // time: i.e., `t + h == t` in double precision, so the step
      // cannot make any representable progress.  Without this, a
      // pathological mix of WRMS weights and roundoff-dominated error
      // estimates (e.g. E5 at atol=1e-10 with tiny QSS components) can
      // trap the retry loop in an infinite stall: the controller keeps
      // "accepting" steps that make no real progress.  Matches
      // CV_TOO_MUCH_ACC in CVODE / "step size too small" in radau5/lsoda.
      //
      // The test must use `tn_abs` (not max(tn_abs, 1)) so that very
      // stiff AD-extended systems with huge ||f|| can still take the
      // initial sub-eps-scale step near t=0 (see issue #2).  At t=0
      // the floor is 0 (any h>0 advances representably), so no guard
      // fires.  Once t grows, the floor grows with it, and h collapse
      // is correctly detected.
      {
        using ndf_detail::scalar_value;
        const double tn_abs = std::abs(static_cast<double>(scalar_value(t)));
        const double dt_abs = std::abs(static_cast<double>(scalar_value(dt)));
        const double h_floor = std::numeric_limits<double>::epsilon() * tn_abs;
        if (dt_abs <= h_floor) {
          ++m_n_rejected;
          return fail;
        }
      }

      // --- : callSetup and convfail decision (cvNlsNewton) ---
      //
      // callSetup: triggered on prev_conv_fail, prev_err_fail,
      // first call, MSBP interval, or gamrat drift beyond DGMAX.
      // convfail = no_failures on first call / after error-test
      // failure (lsetup may reuse cached J if gamma is close);
      // convfail = fail_other on prev Newton failure (fresh J).
      bool force_setup = (nflag == NFlag::prev_conv_fail)
      || (nflag == NFlag::prev_err_fail);

      convfail_t convfail = (nflag == NFlag::first_call || nflag == NFlag::prev_err_fail)
        ? convfail_t::no_failures
      : convfail_t::fail_other;

      // --- Call the NDF stepper: rescale + predict + set + Newton ---
      m_stepper.do_step(system, x, t, x_out, dt, m_xerr.m_v, force_setup, convfail);

      // ============================================================
      //  Newton convergence check
      // ============================================================
      if (!m_stepper.newton_converged()) {
        ++ncf;
        m_stepper.set_etamax(1.0);   // prevent h increase after recovery

        // Max convergence failures → unrecoverable
        if (ncf >= ndf_constants::MXNCF) {
          ++m_n_rejected;
          return fail;
        }

        // Reduce step size and retry
        double eta_cf = std::max(
          static_cast<double>(ndf_constants::ETACF),
          static_cast<double>(ndf_constants::ETAMIN));
        m_stepper.invalidate_lu();
        m_stepper.rescale(eta_cf);
        dt = m_stepper.h();

        nflag = NFlag::prev_conv_fail;
        reset_filter_history();
        ++m_n_rejected;
        continue;   // retry: predict again
      }

      // ============================================================
      //  Error test
      // ============================================================
      double dsm = m_stepper.error_norm();

      if (dsm <= 1.0) {
        // === Step accepted: break out of retry loop ===

        time_type t_new = t + m_stepper.h();

        // Complete step: update Nordsieck array
        m_stepper.complete_step();
        m_stepper.set_tn_current(t_new);

        // Snapshot Nordsieck for dense output BEFORE order/stepsize changes
        m_stepper.prepare_dense_output();

        // --- Order/step-size selection ---
        prepare_next_step(dsm);

        // --- etamax for next step ---
        if (m_stepper.nst() <= ndf_constants::SMALL_NST) {
          m_stepper.set_etamax(ndf_constants::ETAMX2);
        } else {
          m_stepper.set_etamax(ndf_constants::ETAMX3);
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
      reset_filter_history();

      // Max error test failures → unrecoverable
      if (nef >= ndf_constants::MXNEF) {
        ++m_n_rejected;
        return fail;
      }

      // Set etamax = 1 to prevent step size increase after recovery
      m_stepper.set_etamax(1.0);

      if (nef <= ndf_constants::MXNEF1) {
        // --- Normal error test retry: reduce h and try again ---
        int L = m_stepper.current_order() + 1;
        double eta = 1.0 / (pow(ndf_constants::BIAS2 * dsm, 1.0 / L)
                              + ndf_constants::ADDON);
        eta = std::max(ndf_constants::ETAMIN, eta);

        // After SMALL_NEF failures, additionally clamp eta
        if (nef >= ndf_constants::SMALL_NEF) {
          eta = std::min(eta, ndf_constants::ETAMXF);
        }

        m_stepper.rescale(eta);
        dt = m_stepper.h();
        ++m_n_rejected;
        continue;   // retry
      }

      // --- After MXNEF1 failures: force order reduction ---
      if (m_stepper.current_order() > 1) {
        double eta = ndf_constants::ETAMIN;
        m_stepper.set_order_for_next_step(m_stepper.current_order() - 1);
        m_stepper.set_qwait(m_stepper.current_order() + 1);  // L = new q + 1
        m_stepper.rescale(eta);
        dt = m_stepper.h();
        ++m_n_rejected;
        continue;   // retry
      }

      // --- At order 1: full restart with fresh initial-dt estimation ---
      //
      // Instead of blindly applying ETAMIN (which may still leave h
      // far from a workable value), re-estimate from scratch using
      // cppode_hin (the CVODES cvHin port).  Finite-difference ÿ is
      // built in, so no Jacobian is needed on this rare-fallback path.
      //
      // The re-estimated h is clamped to not exceed the current h
      // (which just failed), and to stay above a minimum floor.
      {
        using odeint_utils::scalar_value;

        double tn_abs = std::abs(static_cast<double>(scalar_value(t)));
        double h_min = 4.0 * std::numeric_limits<double>::epsilon()
                     * std::max(tn_abs, 1.0);
        double h_cur = std::abs(static_cast<double>(scalar_value(m_stepper.h())));

        auto& deriv_func = system.first;
        state_type& x_cur = const_cast<state_type&>(x);
        double h_est = odeint_utils::cppode_hin<value_type>(
            deriv_func,
            x_cur, t, tn_abs,  // no t_final known here: use |t0| as upper hint
            m_atol, m_rtol);

        // Clamp: don't exceed the h that just failed, stay above floor.
        h_est = std::min(h_est, h_cur);
        h_est = std::max(h_est, h_min);

        time_type h_new = time_type(h_est);
        m_stepper.set_hscale(h_new);
        m_stepper.set_qwait(ndf_constants::LONG_WAIT);
        m_stepper.reload_zn1_from_f(deriv_func);

        dt = h_new;
        ++m_n_rejected;
        continue;   // retry
      }

    } // end retry loop
  }

private:

  // ====================================================================
  //  prepare_next_step: order/step-size selection
  // ====================================================================

  void prepare_next_step(double dsm)
  {
    // If etamax == 1, defer step size or order changes.
    // No real eta proposal happens here, so we must NOT feed the
    // filter: but we also invalidate filter history so that the
    // next real proposal starts fresh (the skipped step breaks
    // the meaning of h_{n-1}).
    if (m_stepper.etamax() <= 1.0 + 1e-14) {
      m_stepper.set_qwait(std::max(m_stepper.qwait(), 2));
      m_eta = 1.0;
      reset_filter_history();
      return;
    }

    // ----------------------------------------------------------------
    //  Step-size and order selection
    //
    //  Three modes (set via set_pid_mode):
    //
    //   pid_mode::none
    //       Pure classical CVODE: etaq from compute_etaq (BIAS2),
    //       cvChooseEta on raw etaqm1/etaq/etaqp1.
    //
    //   pid_mode::intermediate
    //       Same as none, but a geometric low-pass filter is applied
    //       to the FINAL m_eta (after order selection):
    //         log(eta_filt) = (1-alpha)*log(eta) + alpha*log(eta_prev)
    //       Order selection is untouched. Filter history is reset on
    //       order change, failure, and event.
    //
    //   pid_mode::full
    //       Söderlind H211b replaces the I-controller at the current
    //       order. To keep cvChooseEta fair, etaqm1 and etaqp1 are
    //       lifted onto the same safety-only scale by removing their
    //       BIAS1/BIAS3 conservatism factors and re-applying
    //       m_pid_safety. All three candidates then live on the
    //       "no bias, common safety" scale, and the order-selection
    //       max-comparison works as intended.
    // ----------------------------------------------------------------

    const bool order_decision_step = (m_stepper.qwait() == 0);

    // Compute etaq for the CURRENT order: classical or H211b depending on mode.
    double etaq;
    if (m_pid_mode == pid_mode::full) {
      etaq = compute_h211b_eta(dsm);
    } else {
      etaq = m_stepper.compute_etaq(dsm);
    }

    int  new_q        = m_stepper.current_order();
    bool order_changed = false;
    m_eta = etaq;

    if (order_decision_step) {
      // Compute and (if "full") lift the order-down/up candidates.
      double etaqm1 = m_stepper.compute_etaqm1();
      double etaqp1 = m_stepper.compute_etaqp1();

      if (m_pid_mode == pid_mode::full) {
        etaqm1 = lift_to_pid_scale(etaqm1, ndf_constants::BIAS1,
                                   m_stepper.current_order());
        etaqp1 = lift_to_pid_scale(etaqp1, ndf_constants::BIAS3,
                                   m_stepper.current_L() + 1);
      }

      // cvChooseEta on the (possibly lifted) candidates
      double etam = std::max({etaqm1, etaq, etaqp1});

      if (etam < ndf_constants::THRESH) {
        m_eta = 1.0;
        new_q = m_stepper.current_order();
      } else if (etam == etaq) {
        m_eta = etaq;
        new_q = m_stepper.current_order();
      } else if (etam == etaqm1) {
        m_eta = etaqm1;
        new_q = m_stepper.current_order() - 1;
        order_changed = true;
      } else {
        m_eta = etaqp1;
        new_q = m_stepper.current_order() + 1;
        order_changed = true;
        m_stepper.save_acor_to_zn_qmax();
      }

      if (new_q != m_stepper.current_order()) {
        m_stepper.set_order_for_next_step(new_q);
        // Order changed: wait L steps before next order decision
        // (matches CVODE cvAdjustParams: qwait = L after order change)
        m_stepper.set_qwait(m_stepper.current_L());
      } else {
        // Order unchanged: CVODE uses qwait = 2 here (cvPrepareNextStep),
        // allowing more frequent order re-evaluation and faster adaptation.
        m_stepper.set_qwait(2);
      }
    }

    // Reset filter history on order change in the "full" path. The
    // filter operates on (r_{n-1}, h_{n-1}) which were computed at
    // a different order, so they would corrupt the next filter call.
    if (order_changed) {
      reset_filter_history();
    }

    // Intermediate mode: low-pass filter on the final m_eta in log-space.
    // This runs AFTER order selection and respects order changes.
    if (m_pid_mode == pid_mode::intermediate) {
      apply_intermediate_lp_filter(order_changed);
    }

    apply_eta_limits();
  }

  // ====================================================================
  //  apply_intermediate_lp_filter: log-space low-pass on m_eta
  //
  //  Geometric mean with the previous accepted eta:
  //    log(eta_filt) = (1 - alpha) * log(eta) + alpha * log(eta_prev)
  //
  //  alpha = 0  → no filtering (eta_filt = eta)
  //  alpha = 1  → eta is frozen at eta_prev (don't use this!)
  //
  //  On order change or after a reset, we have no usable eta_prev,
  //  so the filter is bypassed and history is seeded.
  // ====================================================================

  void apply_intermediate_lp_filter(bool order_changed)
  {
    // Reset on order change: don't filter across regime boundaries.
    if (order_changed) {
      m_eta_prev = -1.0;
    }

    // Bootstrap: seed history, no filtering this step.
    if (m_eta_prev <= 0.0) {
      m_eta_prev = m_eta;
      ++m_n_filter_skipped;
      return;
    }

    // Skip filter if m_eta is "no change" (THRESH path): the
    // classical solver decided not to touch h, we respect that.
    if (std::abs(m_eta - 1.0) < 1e-14) {
      m_eta_prev = m_eta;
      return;
    }

    const double a = m_pid_lp_alpha;
    const double log_eta = std::log(m_eta);
    const double log_prev = std::log(m_eta_prev);
    const double log_filt = (1.0 - a) * log_eta + a * log_prev;
    m_eta = std::exp(log_filt);

    m_eta_prev = m_eta;
    ++m_n_filter_active;
  }

  // ====================================================================
  //  lift_to_pid_scale: bring a classical etaqX onto the PID safety scale
  //
  //  classical:  etaqX = 1 / ((BIAS * d)^(1/exp) + ADDON)
  //  lifted:     ~ m_pid_safety * (1/d)^(1/exp)
  //
  //  We multiply by pow(BIAS, 1/exp_denom) to undo the BIAS factor
  //  inside the (BIAS*d)^(1/exp) term, then scale by m_pid_safety.
  //  The result lives on the same "no-bias, common safety" scale as
  //  compute_h211b_eta, so the cvChooseEta max-comparison is fair
  //  across all three candidates.
  // ====================================================================

  double lift_to_pid_scale(double etaqX_classical, double bias, double exp_denom) const
  {
    if (etaqX_classical <= 0.0) return 0.0;
    const double bias_factor = std::pow(bias, 1.0 / exp_denom);
    return etaqX_classical * bias_factor * m_pid_safety;
  }

  // ====================================================================
  //  compute_h211b_eta: canonical Söderlind H211b step-size proposal
  //
  //  Used by pid_mode::full. Replaces (not augments) the I-controller's
  //  etaq.  Returns the proposed h_{n+1}/h_n ratio.
  //
  //  On bootstrap / after a reset, falls back to a one-step
  //  I-controller using only the current error (no history available
  //  yet): this is identical to what the I-controller would do, but
  //  without BIAS2 and with our explicit safety factor.  History is
  //  seeded so the next call uses the full filter.
  //
  //  Coefficients: H211b standard, elementwise form
  //      b1 = b2 = 1/(2L),   a2 = 1/(2L)
  //  with L = current_L() = q + 1.
  // ====================================================================

  double compute_h211b_eta(double dsm)
  {
    const double h_cur = static_cast<double>(
      ndf_detail::scalar_value(m_stepper.h()));
    const double L     = static_cast<double>(m_stepper.current_L());
    const double safety = m_pid_safety;

    // Safe r_n: dsm can be essentially zero on trivial problems.
    const double r_n = (dsm > 1e-300) ? dsm : 1e-300;

    // Bootstrap: no history yet → fall back to single-error I-controller
    // (same shape as compute_etaq but without BIAS2, with explicit safety).
    if (m_filter_skip_next || m_filter_r_prev <= 0.0 || m_filter_h_prev <= 0.0) {
      double eta_boot = safety * std::pow(1.0 / r_n, 1.0 / L);
      m_filter_r_prev    = r_n;
      m_filter_h_prev    = (h_cur > 0.0) ? h_cur : 1.0;
      m_filter_skip_next = false;
      ++m_n_filter_skipped;
      return eta_boot;
    }

    const double r_nm1 = m_filter_r_prev;
    const double h_nm1 = m_filter_h_prev;

    // H211b exponents (elementwise form):
    //   b1 = b2 = 1/(2L)  on the two error terms
    //   a2      = 1/(2L)  on the previous step-size ratio
    const double b = 0.5 / L;     // shared exponent for b1 = b2
    const double a = 0.5 / L;     // a2

    // Filter equation:
    //   eta = safety * (1/r_n)^b1 * (1/r_{n-1})^b2 * (h_{n-1}/h_n)^a2
    const double eta = safety
      * std::pow(1.0 / r_n,    b)
      * std::pow(1.0 / r_nm1,  b)
      * std::pow(h_nm1 / h_cur, a);

    // Update history for next step
    m_filter_r_prev = r_n;
    m_filter_h_prev = h_cur;
    ++m_n_filter_active;

    return eta;
  }

  void reset_filter_history()
  {
    m_filter_r_prev    = -1.0;
    m_filter_h_prev    = -1.0;
    m_eta_prev         = -1.0;
    m_filter_skip_next = true;
  }

  // ====================================================================
  //  apply_eta_limits: clamp eta to the allowed range
  // ====================================================================

  void apply_eta_limits()
  {
    if (m_eta < ndf_constants::THRESH) {
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
    reset_filter_history();
  }

  void restart_from_order1(const state_type& x, time_type t, time_type dt)
  {
    m_stepper.restart_from_order1(x, t, dt);
    reset_filter_history();
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
  int n_filter_active()  const { return m_n_filter_active; }
  int n_filter_skipped() const { return m_n_filter_skipped; }
  void reset_counters() {
    m_n_accepted = 0; m_n_rejected = 0;
    m_n_filter_active = 0; m_n_filter_skipped = 0;
    m_stepper.reset_counters();
  }

  // --- Profiler ---
  mutable cppode::profiler m_prof;
  void finalize_profiler() const { m_prof.merge(m_stepper.m_prof); }

  // Compile-time label derived from the stepper's multistep_method so
  // profile output is self-identifying for all instantiations.
  static constexpr const char* method_label() {
    using M = cppode::multistep_method;
    switch (stepper_type::method) {
      case M::bdf:    return "CppODE BDF";
      case M::adams:  return "CppODE Adams";
    }
    return "CppODE multistep";
  }

  void report_profiler(const char* label = nullptr) const {
    finalize_profiler();
    m_prof.report(label ? label : method_label());
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
    bool resized = adjust_size_by_resizeability(m_xerr, x);
    if (resized && m_n_sens != 0)
      m_xerr_slab.prime(m_xerr.m_v,
                        static_cast<unsigned>(m_xerr.m_v.size()), m_n_sens);
    return resized;
  }

  template<class StateIn>
  bool resize_xnew(const StateIn& x)
  {
    bool resized = adjust_size_by_resizeability(m_xnew, x);
    if (resized && m_n_sens != 0)
      m_xnew_slab.prime(m_xnew.m_v,
                        static_cast<unsigned>(m_xnew.m_v.size()), m_n_sens);
    return resized;
  }

  template<class StateIn>
  bool resize_ftemp(const StateIn& x)
  {
    bool resized = adjust_size_by_resizeability(m_ftemp, x);
    if (resized && m_n_sens != 0)
      m_ftemp_slab.prime(m_ftemp.m_v,
                         static_cast<unsigned>(m_ftemp.m_v.size()), m_n_sens);
    return resized;
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

  // SoA tangent slabs for the dynamic-dual heap path (empty stubs otherwise).
  detail::tangent_slab<value_type> m_xerr_slab;
  detail::tangent_slab<value_type> m_xnew_slab;
  detail::tangent_slab<value_type> m_ftemp_slab;
  unsigned m_n_sens = 0;

  double m_atol, m_rtol;

  double m_eta = 1.0;   // step-size ratio (stored for cvPrepareNextStep)

  // --- PID step-size control state ---
  pid_mode m_pid_mode      = pid_mode::none;  // default: classical I-controller
  double   m_pid_safety    = 0.8;     // safety factor for "full" path
  double   m_pid_lp_alpha  = 0.4;     // low-pass pole for "intermediate" path
  double   m_filter_r_prev = -1.0;    // r_{n-1} ; <0 means "uninitialized"
  double   m_filter_h_prev = -1.0;    // h_{n-1} ; <0 means "uninitialized"
  double   m_eta_prev      = -1.0;    // last accepted eta (intermediate path)
  bool     m_filter_skip_next = true; // skip filter on next accepted step
  int      m_n_filter_active  = 0;    // diagnostics: filter fired
  int      m_n_filter_skipped = 0;    // diagnostics: filter bypassed

  int m_n_accepted;
  int m_n_rejected;
};

} // namespace cppode

#endif // CPPODE_MULTISTEPPER_CONTROLLER_HPP_INCLUDED
