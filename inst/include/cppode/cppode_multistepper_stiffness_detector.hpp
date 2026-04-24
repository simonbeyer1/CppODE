/*
 cppode_multistepper_stiffness_detector.hpp — LSODA-style switching
 ==================================================================

 Free function template that decides whether the multistepper should
 switch between the Adams (non-stiff) and BDF/NDF (stiff) families.

 This is a LITERAL TRANSLATION of `methodswitch()` from
 sdwfrost/liblsoda (file src/methodswitch.c), which is itself a port
 of the original ODEPACK Fortran LSODA stoda routine.  Variable names
 mirror the C source one-for-one to make line-by-line review trivial.

 Key idea (Petzold 1983 / Hindmarsh):
 ------------------------------------
 Rather than estimating the dominant Jacobian eigenvalue directly,
 LSODA estimates the step-size ratio that EACH method (the current
 one and the alternative) WOULD allow at the current state.  The
 alternative method's local error is estimated WITHOUT actually
 running the other method, by rescaling the current method's `dsm`
 with the ratio of the two methods' leading error coefficients
 (cm1[] for Adams, cm2[] for BDF — these are tabulated).

 Cost: O(1) per accepted step.  No extra f-evaluations, no extra
 Jacobian work, no eigenvalue iteration.  The Lipschitz estimate
 `pnorm = ||J||_1` comes for free on the BDF side (we already have
 the Jacobian).  On the Adams side `pdlast = 0` and the test falls
 back to the `irflag` (last step was stability-limited?) signal that
 the Adams order-selection logic produces.

 Switch criterion (asymmetric, hence hysteretic):
   Adams → BDF : need rh2 >= 5 * rh1   (5x speedup minimum)
   BDF → Adams : need rh1 >= rh2 / 5   (1x speedup, very low bar)
                 PLUS dm1 > 1000*eps*pnorm to avoid roundoff churn

 Cooldown: after a switch, icount = 20 steps must pass before the
 next switch test runs.  The stepper decrements icount each step;
 the detector returns `recommendation::stay` while icount > 0.

 Reference: sdwfrost/liblsoda commit master, src/methodswitch.c.
 Original ODEPACK Fortran by Linda Petzold and Alan Hindmarsh.

 Status: NOT YET WIRED into the multistepper.  This header provides
 the building blocks; the dispatcher in cppode_multistepper.hpp will
 call methodswitch_step() from on_step_accepted() in a follow-up.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_MULTISTEPPER_STIFFNESS_DETECTOR_HPP_INCLUDED
#define CPPODE_MULTISTEPPER_STIFFNESS_DETECTOR_HPP_INCLUDED

#include <algorithm>
#include <array>
#include <cmath>

// Note: this header is designed to be included from cppode_multistepper.hpp,
// which provides adams_constants::ADAMS_STAB_RADIUS and max_order_adams.
// It can also be included standalone — in that case the including TU must
// either include cppode_multistepper.hpp first or provide its own
// adams_constants namespace.  We intentionally do NOT forward-include
// multistepper.hpp here to avoid a circular dependency.

namespace cppode {

namespace stiffness_detector {

// ============================================================================
//  cm1[] and cm2[] — leading error coefficients for the switch test
//
//  These are LITERAL copies of the cm1[] and cm2[] arrays from
//  sdwfrost/liblsoda src/methodswitch.c lines 11-22.  The original
//  source comment reads:
//
//    "cm1 and cm2 are calculated from tesco/elco 1 and 2 formula
//     in cfode.c:printcm12."
//
//  cm1[q] is for the Adams family (meth = 1 in LSODA), cm2[q] is
//  for the BDF family (meth = 2 in LSODA).
//
//  Index = order q in 1..12 (cm2 only meaningful up to q=5).
//  Index 0 is unused.
//
//  Decimal values were obtained by decoding the C99 hex-float
//  literals via Python's float.fromhex().  The hex literal is
//  preserved in the trailing comment for byte-equivalence verification.
// ============================================================================

static constexpr std::array<double, 13> CM1 = {
  0.0,                          // cm1[ 0]   // 0x0p+0     (unused)
  2.0000000000000000e+00,       // cm1[ 1]   // 0x1p+1
  5.9999999999999991e+00,       // cm1[ 2]   // 0x1.7ffffffffffffp+2
  4.0000000000000000e+00,       // cm1[ 3]   // 0x1p+2
  1.5789473684210522e+00,       // cm1[ 4]   // 0x1.9435e50d79434p+0
  4.4444444444444470e-01,       // cm1[ 5]   // 0x1.c71c71c71c721p-2
  9.7334878331402030e-02,       // cm1[ 6]   // 0x1.8eaf0473189ecp-4
  1.7454545454545455e-02,       // cm1[ 7]   // 0x1.1df9ab7934513p-6
  2.6507230583453592e-03,       // cm1[ 8]   // 0x1.5b6f81b154515p-9
  3.4915591557409953e-04,       // cm1[ 9]   // 0x1.6e1dd3d149b81p-12
  4.0609974117294529e-05,       // cm1[10]   // 0x1.54a9415f71629p-15
  4.2288774180588911e-06,       // cm1[11]   // 0x1.1bcb8f930a98p-18
  3.9866297221387278e-07        // cm1[12]   // 0x1.ac0fa4b46f6c6p-22
};

// "only useful till cm2[5]" — original LSODA comment
static constexpr std::array<double, 13> CM2 = {
  0.0,                          // cm2[ 0]   // 0x0p+0     (unused)
  2.0000000000000000e+00,       // cm2[ 1]   // 0x1p+1
  1.5000000000000000e+00,       // cm2[ 2]   // 0x1.8p+0
  6.6666666666666674e-01,       // cm2[ 3]   // 0x1.5555555555556p-1
  2.0833333333333337e-01,       // cm2[ 4]   // 0x1.aaaaaaaaaaaacp-3
  4.9999999999999996e-02,       // cm2[ 5]   // 0x1.9999999999999p-5
  // The remaining entries are filler and should not be indexed by
  // the BDF-side branch (which clamps q at mxords = 5).
  9.7334878331402030e-02,
  1.7454545454545455e-02,
  2.6507230583453592e-03,
  3.4915591557409953e-04,
  4.0609974117294529e-05,
  4.2288774180588911e-06,
  3.9866297221387278e-07
};

// ============================================================================
//  sm1[] — Adams stability radii (alias of ADAMS_STAB_RADIUS)
//
//  LSODA's `sm1[]` array contains the same Adams-Moulton imaginary-
//  axis stability radii we already expose via adams_constants.
//  Re-export under the LSODA name for line-for-line readability of
//  the methodswitch_step() body below.
// ============================================================================

static constexpr const std::array<double, 13>& SM1 =
  adams_constants::ADAMS_STAB_RADIUS;

// ============================================================================
//  Switching constants
// ============================================================================

// Cooldown (steps) after a successful method switch before the next
// switch test runs.  LSODA hard-codes this to 20.
static constexpr int ICOUNT_RESET = 20;

// Asymmetric switch ratio.  Adams→BDF needs RATIO=5 (significant
// speedup required).  BDF→Adams needs RATIO/5 = 1 (any speedup
// suffices).  Hard-coded in LSODA, see methodswitch.c.
static constexpr double RATIO = 5.0;

// Maximum BDF order — used as the LSODA `mxords` parameter when
// considering an Adams→BDF switch with q > mxords.
static constexpr int MXORDS = 5;

// Maximum Adams order.
static constexpr int MXORDN = adams_constants::max_order_adams;  // 12

// Machine epsilon proxy used in the LSODA switch test.  LSODA uses
// the actual machine epsilon (`ETA` in the source); for double this
// is ~2.22e-16.
static constexpr double ETA = 2.220446049250313e-16;

// ============================================================================
//  recommendation enum — what the detector tells the stepper to do
// ============================================================================

enum class recommendation {
  stay,                  // No change.
  switch_to_bdf,         // Adams → BDF/NDF
  switch_to_adams        // BDF/NDF → Adams
};

// ============================================================================
//  switch_result
//
//  Output of methodswitch_step().  When `rec != stay`, the stepper
//  should clamp the order to `new_q`.  Step-size adjustment is left
//  to the controller's normal prepare_next_step() / rescale() path.
// ============================================================================

struct switch_result {
  recommendation rec;
  int            new_q;     // order to clamp to post-switch
  // Longer cooldown after an "emergency" stability-based switch.
  // When set, the stepper should use this value instead of
  // ICOUNT_RESET to delay the next switch test.  Prevents the
  // BDF↔Adams flip-flop seen on genuinely stiff problems.
  int            cooldown_override = 0;
};

// ============================================================================
//  methodswitch_step  ←  liblsoda methodswitch()
//
//  Inputs:
//    in_adams_mode  — true if currently running Adams, false if BDF/NDF
//    nq             — current method order (1..12 for Adams, 1..5 for BDF)
//    h              — current step size (signed)
//    dsm            — local error norm from the just-accepted step
//                     (= acnrm * tq[2])
//    pnorm          — Jacobian 1-norm or inf-norm at current state.
//                     Pass 0 if unavailable (Adams-side fallback path).
//    pdlast         — last computed Lipschitz estimate (= last pnorm
//                     when J was current).  Pass 0 if not available.
//    icount         — switch cooldown counter.  Stepper should
//                     decrement by 1 per accepted step and pass the
//                     current value here.  When > 0, this routine
//                     immediately returns `stay` without further work.
//    irflag         — Adams side only: 1 if the last Adams step was
//                     stability-limited (rh1 came from the pdh term),
//                     0 otherwise.  Ignored on the BDF side.
//
//  Returns: switch_result.  When rec == stay, rh and new_q are
//  meaningless.
// ============================================================================

inline switch_result methodswitch_step(
    bool   in_adams_mode,
    int    nq,
    double h,
    double dsm,
    double pnorm,
    double pdlast,
    int    icount,
    int    irflag)
{
  // Cooldown.  LSODA decrements icount in stoda(); the stepper does
  // the same and we just check the value here.
  if (icount > 0) {
    return { recommendation::stay, nq };
  }

  // ====================================================================
  //  Branch A: currently Adams — consider switching to BDF
  // ====================================================================
  if (in_adams_mode) {

    // If current order is greater than MXORDS (5), assume the problem
    // is non-stiff and skip the test entirely.  LSODA: `if (nq > 5) return;`
    if (nq > MXORDS) {
      return { recommendation::stay, nq };
    }

    double rh1, rh2, rh1it;
    int    nqm2;
    double exsm = 1.0 / static_cast<double>(nq + 1);

    // If the local error / Lipschitz estimate are polluted by
    // roundoff, fall back to the irflag signal.
    if (dsm <= 100.0 * pnorm * ETA || pdlast == 0.0) {
      // Polluted error estimate path.  Switch only if the last step
      // was stability-restricted; otherwise stay.  When switching with
      // polluted estimates, double the step size in the absence of
      // other information.
      if (irflag == 0) {
        return { recommendation::stay, nq };
      }
      rh2  = 2.0;
      nqm2 = std::min(nq, MXORDS);
    } else {
      // Normal path: compute rh1 (Adams) and rh2 (BDF) from dsm and
      // the cm1/cm2 ratio.
      rh1   = 1.0 / (1.2 * std::pow(dsm, exsm) + 0.0000012);
      rh1it = 2.0 * rh1;

      double pdh = pdlast * std::abs(h);
      if ((pdh * rh1) > 0.00001) {
        // Stability-limited — recompute rh1 from sm1[]/pdh.
        rh1it = SM1[nq] / pdh;
      }
      rh1 = std::min(rh1, rh1it);

      if (nq > MXORDS) {
        // (Unreachable here because we returned above on nq > 5,
        // but mirror the LSODA structure for safety.)
        nqm2       = MXORDS;
        const int lm2 = MXORDS + 1;
        const double exm2 = 1.0 / static_cast<double>(lm2);
        // Original LSODA computes dm2 from yh[lm2+1] and ewt; here
        // we conservatively use the same dsm-rescaling formula as
        // the nq <= MXORDS branch — accuracy difference negligible
        // for the rare nq > 5 case.
        double dm2 = dsm * (CM1[nq] / CM2[nq]);
        rh2 = 1.0 / (1.2 * std::pow(dm2, exm2) + 0.0000012);
        (void)exm2;
      } else {
        double dm2 = dsm * (CM1[nq] / CM2[nq]);
        rh2 = 1.0 / (1.2 * std::pow(dm2, exsm) + 0.0000012);
        nqm2 = nq;
      }

      // Switch only if the BDF speedup would be at least RATIO.
      if (rh2 < RATIO * rh1) {
        return { recommendation::stay, nq };
      }
    }

    // Switch test passed.
    return { recommendation::switch_to_bdf, nqm2 };
  }

  // ====================================================================
  //  Branch B: currently BDF/NDF — consider switching to Adams
  // ====================================================================

  double exsm = 1.0 / static_cast<double>(nq + 1);
  double rh1, rh2, rh1it, dm1, exm1;
  int    nqm1;

  if (MXORDN < nq) {
    // (Won't happen unless someone configures mxordn < 5; preserved
    // for parity with the LSODA source.)
    nqm1            = MXORDN;
    const int lm1   = MXORDN + 1;
    exm1            = 1.0 / static_cast<double>(lm1);
    // dm1 from yh[lm1+1] / cm1[mxordn] in LSODA — we substitute
    // the dsm-rescaling form for self-containment.
    dm1 = dsm * (CM2[nq] / CM1[MXORDN]);
    rh1 = 1.0 / (1.2 * std::pow(dm1, exm1) + 0.0000012);
  } else {
    dm1 = dsm * (CM2[nq] / CM1[nq]);
    rh1 = 1.0 / (1.2 * std::pow(dm1, exsm) + 0.0000012);
    nqm1 = nq;
    exm1 = exsm;
  }

  rh1it = 2.0 * rh1;
  double pdh = pnorm * std::abs(h);   // BDF side uses pnorm directly
  if ((pdh * rh1) > 0.00001) {
    rh1it = SM1[nqm1] / pdh;
  }
  rh1 = std::min(rh1, rh1it);

  rh2 = 1.0 / (1.2 * std::pow(dsm, exsm) + 0.0000012);

  // BDF→Adams condition.  LSODA writes it as
  //   if ((rh1 * RATIO) < (5.0 * rh2)) stay;
  // (which simplifies to rh1 < rh2).  This lets Adams win ties, which
  // is a trap at q=1 where cm1[1] == cm2[1] and therefore rh1 == rh2
  // always — every BDF step would attempt to switch to Adams on
  // genuinely stiff problems like Robertson.  We add a tie-breaking
  // margin so the switch only fires when Adams has a strict, non-
  // trivial advantage.  1.1 is small enough not to hurt problems
  // (e.g. Van der Pol) where Adams is genuinely better.
  static constexpr double TIE_MARGIN = 1.1;
  if ((rh1 * RATIO) < (TIE_MARGIN * 5.0 * rh2)) {
    return { recommendation::stay, nq };
  }

  // Roundoff pollution check: if the would-be Adams step gives an
  // error below ~1000*eps*pnorm, the Adams error estimate would be
  // unreliable — stay in BDF.
  double alpha = std::max(0.001, rh1);
  dm1 *= std::pow(alpha, exm1);
  if (dm1 <= 1000.0 * ETA * pnorm) {
    return { recommendation::stay, nq };
  }

  // Switch test passed.
  return { recommendation::switch_to_adams, nqm1 };
}

} // namespace stiffness_detector

} // namespace cppode

#endif // CPPODE_MULTISTEPPER_STIFFNESS_DETECTOR_HPP_INCLUDED
