/*
 CppODE Multistep Stepper — Nordsieck multistep implementation.

 Unified variable-order, variable-step Nordsieck multistep stepper
 covering two method variants selected at compile time via the
 multistep_method enum:

   bdf     : pure BDF/NDF, max order 5  (default)
   adams   : pure Adams-Moulton PECE,  max order 12

 Whether the stiff side uses NDF (Klopfenstein-Shampine) kappa
 coefficients or classical BDF (kappa = 0) is controlled at runtime
 via set_use_ndf_kappa().  The default is NDF (useNDF = true).

 The stiff-side default formulas are the Klopfenstein-Shampine
 Numerical Differentiation Formulas (NDF) from Shampine & Reichelt
 1997, "The MATLAB ODE Suite", SIAM J. Sci. Comput. 18(1), 1-22.
 Setting use_ndf_kappa to false recovers classical BDF.

 The NDF-kappa modification replaces the BDF correction

 sum_{m=1}^k (1/m) grad^m y - h F = 0

 with

 sum_{m=1}^k (1/m) grad^m y - h F - kappa * gamma_k * (y - y_pred) = 0

 where y_pred is the predictor.  Algebraically this amounts to
 replacing gamma_k by (1 - kappa) * gamma_k in the iteration matrix
 and in the Newton residual, and scaling the leading truncation
 error coefficient from 1/(k+1) to (kappa*gamma_k + 1/(k+1)).

 Klopfenstein-Shampine kappa values (Table 1 in SR97):

 k=1:  kappa = -0.1850     26% larger step vs BDF, same A-stab
 k=2:  kappa = -1/9        26% larger step vs BDF, same A-stab
 k=3:  kappa = -0.0823     26% larger step, stability angle 80 deg
 k=4:  kappa = -0.0415     12% larger step, stability angle 66 deg
 k=5:  kappa =  0          identical to BDF5 (too little margin)

 Architecture:
 - lu_W<Value, is_sparse> member handles all LU operations
 - ndf_newton_solve()        free function — BDF/NDF Newton corrector
                             (declared in cppode_newton.hpp)
 - adams_pece_solve()        free function — Adams PECE corrector
                             (defined inline below)
 - adams_set_coefficients()  free function — Adams Nordsieck coefficients
                             (defined inline below)
 - multistepper              orchestrates the step pipeline

 AD handling is fully transparent: lu_W::solve() dispatches to IFT
 internally, WRMS norms in newton include derivative components.

 This header is the single source of truth for the multistep family:
 the coefficient routines, the Adams PECE corrector, and the stepper
 class itself all live here.

 Copyright (C) 2026 Simon Beyer

 Portions of this file are derived from SUNDIALS/CVODE(S)
 (cvode.c — cvSetAdams, cvAdjustParams, cvNlsFunctional, cvHin,
 error-weight machinery, NDF kappa modification, Adams PECE control),
 Copyright (c) 2002-2024 Lawrence Livermore National Security and
 Southern Methodist University, distributed under the BSD-3-Clause
 license.  See inst/COPYRIGHTS for the full license text.
 */

#ifndef CPPODE_MULTISTEPPER_HPP_INCLUDED
#define CPPODE_MULTISTEPPER_HPP_INCLUDED

#include <cstddef>
#include <cstring>
#include <array>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <type_traits>
#include <cassert>
#include <vector>

#include <cppode/cppode_odeint_compat.hpp>
#include <cppode/cppode_lu.hpp>
#include <cppode/cppode_newton.hpp>
#include <cppode/cppode_stepper_traits.hpp>
#include <cppode/cppode_profiler.hpp>
#include <cppode/cppode_step_trace.hpp>
#include <cppode/cppode_ad_lu.hpp>          // ad_lu::is_ad — used by adams_pece_solve
#include <cppode/cppode_ad_traits.hpp>
#include <cppode/cppode_dual_slab.hpp>
#include <cppode/cppode_nordsieck_block.hpp>

namespace cppode {

// ============================================================================
//  Scalar value extraction — pulled in from cppode_ad_traits.hpp
// ============================================================================

namespace ndf_detail {
using cppode::ad_traits::scalar_value;
} // namespace ndf_detail

// ============================================================================
//  Algorithm constants
// ============================================================================

namespace ndf_constants {

// --- Newton / error-test / step-size parameters ---
static constexpr double CORTES  = 0.1;      // Newton convergence rate threshold (tq4 = CORTES/tq[2])
static constexpr double CRDOWN  = 0.3;      // crate damping: crate = max(CRDOWN*crate, del/delp)
static constexpr double RDIV    = 2.0;      // divergence ratio: fail Newton if del > RDIV*delp (m>=2)
static constexpr double DGMAX   = 0.3;      // |gamrat-1| threshold for forcing LU refactorization
static constexpr double THRESH  = 1.5;      // min eta to bother changing h (eta < THRESH → eta=1)
static constexpr double ADDON   = 1e-6;     // small constant in eta denominator to avoid division by zero
// --- Step-size safety factors (classical CVODE BIAS style) ---
// Formula: eta = 1 / ((BIAS * d)^(1/L) + ADDON)
// The exponent makes the effective safety order-dependent:
//   effective_safety = BIAS^(1/L), which is large at low orders
//   (where error grows fast with h) and approaches 1 at high orders
//   (where error is already well-controlled).
static constexpr double BIAS1   = 6.0;      // safety factor in etaqm1 (order decrease candidate)
static constexpr double BIAS2   = 6.0;      // safety factor in etaq   (current order candidate)
static constexpr double BIAS3   = 10.0;     // safety factor in etaqp1 (order increase candidate)
static constexpr double ETAMX1  = 10000.0;  // max eta on first step (compensate small initial h)
static constexpr double ETAMX2  = 10.0;     // max eta for steps 2..SMALL_NST
static constexpr double ETAMX3  = 10.0;     // max eta after SMALL_NST steps
static constexpr double ETAMXF  = 0.2;      // max eta after SMALL_NEF error test failures
static constexpr double ETAMIN  = 0.1;      // min eta (floor for step size reduction)
static constexpr double ETACF   = 0.25;     // eta on Newton convergence failure
static constexpr double ONEPSM  = 1.000001; // 1 + epsilon for floating-point guard
static constexpr int    SMALL_NST = 10;     // steps before switching etamax from ETAMX2 to ETAMX3
static constexpr int    NLS_MAXCOR = 3;     // max Newton corrector iterations per step
static constexpr int    MSBP    = 20;       // max steps between periodic LU setup calls
static constexpr int    MXNEF   = 7;        // max total error test failures before giving up
static constexpr int    MXNEF1  = 3;        // error test failures before forcing order reduction
static constexpr int    SMALL_NEF = 2;      // error test failures before clamping eta to ETAMXF
static constexpr int    MXNCF   = 10;       // max Newton convergence failures before giving up
static constexpr int    LONG_WAIT = 10;     // qwait after order-1 restart (hold order that many steps)

// --- Klopfenstein-Shampine NDF coefficients (Table 1, SR97) ---
// Indexed by order q in 1..5.  kappa[0] is a dummy (unused).
// Source: Shampine & Reichelt, "The MATLAB ODE Suite",
//         SIAM J. Sci. Comput. 18(1), 1997, Table 1.
//
// These values balance accuracy (smaller leading truncation error
// coefficient than classical BDF) against A(alpha)-stability.  For
// q=5, kappa is set to 0 because BDF5's 51-degree stability angle
// leaves no headroom for further reduction.
static constexpr double NDF_KAPPA[6] = {
  0.0,       // q = 0 (unused)
  -0.1850,    // q = 1
  -1.0/9.0,   // q = 2
  -0.0823,    // q = 3
  -0.0415,    // q = 4
  0.0        // q = 5 (NDF5 == BDF5)
};

} // namespace ndf_constants

namespace adams_constants {

static constexpr int max_order_adams = 12;

} // namespace adams_constants

// ============================================================================
//  Adams coefficient routines  (cvSetAdams port from Sundials/cvode.c)
//
//  Computes the Nordsieck correction coefficients l[0..q] and the
//  error-test constants tq[1..5] for variable-step Adams-Moulton of
//  order q.  Literal translation of cvSetAdams / cvAdamsStart /
//  cvAdamsFinish / cvAltSum — variable names kept identical to the C
//  source for line-by-line review.
//
//  VALIDATED at constant step size against the classical fixed-step
//  Adams-Moulton coefficients (Hairer/Wanner Vol I Table III.1.1) for
//  q = 1..6, and at variable step for q = 1..12 against a reference
//  implementation.  Bit-for-bit equivalent to CVODE at q == 1.
// ============================================================================

namespace adams_detail {

// alt_sum <- cvAltSum :
//   sum_{i=0..iend} (-1)^i * a[i] / (i+k)   (0 if iend < 0)
// Computes the integral over [-1,0] of x^(k-1) * M(x) given the
// coefficients of M(x) — used to normalize the Adams Lambda polynomial.
template<class T, std::size_t N>
inline double alt_sum(int iend, const std::array<T, N>& a, int k)
{
  if (iend < 0) return 0.0;
  double sum  = 0.0;
  double sign = 1.0;
  for (int i = 0; i <= iend; ++i) {
    sum  += sign * static_cast<double>(a[i]) / static_cast<double>(i + k);
    sign  = -sign;
  }
  return sum;
}

} // namespace adams_detail

// adams_set_coefficients <- cvSetAdams
//
// Inputs:
//   q       — current order, 1 <= q <= 12.
//   qwait   — order-change wait counter; tq[1] and tq[3] are only
//             computed when qwait == 1, exactly as in CVODE.
//   h       — current step size.
//   tau     — past step sizes; tau[1] is the most recent (= last
//             accepted h), tau[2] the one before, etc.  Uses indices
//             1..q-1.
//
// Outputs (by reference):
//   l       — l[0..q] written, higher zeroed.  l[0] = 1 by construction.
//   tq      — tq[1..5] written.
//
// Polynomial Lambda(x) of degree q:
//   Lambda(-1) = 0,  Lambda(0) = 1,
//   (d/dx) Lambda(x) = c * PRODUCT_{i=1..q-1} (1 + x/xi_i),
// with xi_i = (t_n - t_{n-i})/h and c chosen so Lambda(0) = 1.  The
// l[i] are the coefficients of Lambda(x).
template<class TimeType, std::size_t L_PLUS_1, std::size_t TAU_LEN>
void adams_set_coefficients(
    int q,
    int qwait,
    TimeType h,
    const std::array<TimeType, TAU_LEN>& tau,    // tau[1..q-1] used
    std::array<TimeType, L_PLUS_1>& l,            // out: l[0..q]
    std::array<double, 6>& tq)                    // out: tq[1..5]
{
  static_assert(L_PLUS_1 >= 14,
    "l[] array must hold at least 14 entries to support Adams order up to 12");
  static_assert(TAU_LEN  >= 13,
    "tau[] array must hold at least 13 entries to support Adams order up to 12");

  // --- q == 1 special case (literal port of cvSetAdams q==1 branch) ---
  if (q == 1) {
    l[0] = TimeType(1);
    l[1] = TimeType(1);
    for (std::size_t i = 2; i < L_PLUS_1; ++i) l[i] = TimeType(0);
    tq[1] = 1.0;
    tq[2] = 0.5;          // HALF
    tq[5] = 1.0;
    tq[3] = 1.0 / 12.0;
    tq[4] = 0.1 / tq[2];  // CORTES / tq[2]
    return;
  }

  // --- q >= 2 path ---
  //
  // Step 1 (cvAdamsStart): build product polynomial m[] of degree q-1:
  //     m(x) = PRODUCT_{i=1..q-1} (1 + x / xi_i)
  // by repeated multiplication, where xi_i = (t_n - t_{n-i})/h.  Also
  // sets tq[1] inside the loop when qwait == 1 and j == q-1.
  // Post-loop hsum = h + tau[1] + ... + tau[q-1].

  std::array<TimeType, 13> m;
  m.fill(TimeType(0));
  m[0] = TimeType(1);

  TimeType hsum = h;
  for (int j = 1; j < q; ++j) {

    // Inside-the-loop tq[1] computation.  At this point m[] holds
    // the product over (1 + x/xi_1)..(1 + x/xi_{q-2}), i.e. degree
    // q-2, which is what cvAltSum(q-2, m, 2) expects.
    if ((j == q - 1) && (qwait == 1)) {
      double sum = adams_detail::alt_sum(q - 2, m, 2);
      tq[1] = static_cast<double>(q) * sum / static_cast<double>(m[q - 2]);
    }

    TimeType xi_inv = h / hsum;

    // Multiply m(x) by (1 + xi_inv * x), in-place high-to-low.
    for (int i = j; i >= 1; --i) {
      m[i] = m[i] + m[i - 1] * xi_inv;
    }

    // Increment hsum AT END of iteration with tau[j], matching CVODE.
    hsum += tau[j];
  }

  // --- M[0], M[1] ---
  const double M0 = adams_detail::alt_sum(q - 1, m, 1);
  const double M1 = adams_detail::alt_sum(q - 1, m, 2);

  // --- cvAdamsFinish: normalize m[] into l[] and compute tq[2..5] ---
  //
  // cvAdamsFinish modifies m[] in-place when computing tq[3]
  // (multiplies by one more (1 + xi_inv*x) factor → degree q).
  // We replicate that.

  const double M0_inv = 1.0 / M0;

  l[0] = TimeType(1);
  for (int i = 1; i <= q; ++i) {
    l[i] = m[i - 1] * TimeType(M0_inv / static_cast<double>(i));
  }
  for (std::size_t i = static_cast<std::size_t>(q + 1); i < L_PLUS_1; ++i) {
    l[i] = TimeType(0);
  }

  const double xi     = static_cast<double>(hsum) / static_cast<double>(h);
  const double xi_inv = 1.0 / xi;

  tq[2] = M1 * M0_inv / xi;
  tq[5] = xi / static_cast<double>(l[q]);

  // tq[3] needs one more multiplication of m by (1 + xi_inv*x), but
  // only when qwait == 1.  After this update m has degree q (was q-1).
  if (qwait == 1) {
    for (int i = q; i >= 1; --i) {
      m[i] = m[i] + m[i - 1] * TimeType(xi_inv);
    }
    const double M2 = adams_detail::alt_sum(q, m, 2);
    const double L_d = static_cast<double>(q + 1);
    tq[3] = M2 * M0_inv / L_d;
  }
  // else: tq[3] left untouched; controller only reads it when qwait == 1.

  tq[4] = 0.1 / tq[2];   // CORTES / tq[2]

  // Sign note: CVODE does NOT take absolute values here.  Adams tq
  // values are structurally positive for q >= 2 when m[] is built
  // correctly.  The BDF/NDF path does call std::abs on tq[2] / tq[5]
  // because alpha0 can be negative in the FLC formulation.
}

// ============================================================================
//  Adams-Moulton PECE corrector  (sibling of ndf_newton_solve)
//
//  Performs one PECE (Predict-Evaluate-Correct-Evaluate) Adams-Moulton
//  step.  Same calling convention as newton_solve, but no LU, no
//  Jacobian, no linear solve — cost per accepted step is 2 f-evals.
//
//  PECE layout:
//    P : y_pred = Pascal predictor on Nordsieck array  (caller's job)
//    E : f_pred = f(t_n, y_pred)
//    C : delta  = l[1] * (h * f_pred - z_pred[1])
//        y_n    = y_pred + delta
//    E : f_new  = f(t_n, y_n)   (stored for next step)
//
//  Stiffness indicator: when the fixed-point iteration diverges,
//  |h * lambda * l[1]| > 1 for the dominant eigenvalue.  Exposed via
//  the `diverged` flag so the stepper can switch to NDF immediately.
//
//  AD handling: corrector is pure arithmetic on f and the Nordsieck
//  slots, so dual types propagate sensitivities transparently.
// ============================================================================

struct pece_result {
  bool   converged;     // Did the PECE fixed-point iteration converge?
  bool   diverged;      // Hard divergence — strong stiffness signal.
  double acnrm;         // WRMS norm of the accumulated correction.
  int    n_fevals;      // Number of f-evaluations used (2 for converged step).
  int    n_iters;       // Number of corrector iterations performed.

  // Maximum convergence rate observed across the corrector iterations,
  // following LSODA correction(): rm = del/delp, rate = max(rate, rm).
  // Used by the stiffness detector to estimate the dominant Lipschitz
  // constant on the Adams side where no Jacobian is available.
  // Zero if the corrector did not run at least 2 iterations (the ideal
  // non-stiff case — no rate signal available).
  double rate_max;
};

// adams_pece_solve: one PECE Adams-Moulton corrector solve.
//
// Inputs:
//   deriv_func   — user RHS, called as deriv_func(y, ydot, t).
//   zn0          — Nordsieck slot 0 BEFORE predict: previous accepted
//                  state (used only for atol/rtol weighting).
//   zn0_pred     — Nordsieck slot 0 AFTER predict: the predictor for y_n.
//   zn1_pred     — Nordsieck slot 1 AFTER predict: h * y' at y_pred.
//   h            — current step size.
//   l1           — l[1] from adams_set_coefficients (leading correction
//                  weight).
//   t_new        — time at the new step, t_n + h.
//   tq4          — Newton-style convergence threshold inherited from
//                  the controller (CORTES / tq[2]).
//   atol, rtol   — error weights for the WRMS norm.
//   max_iter     — maximum corrector iterations (typical: 3-4).
//
// Outputs (by reference):
//   acor         — accumulated correction y_n - y_pred.
//   y            — final corrected y_n.
//   tempv, ftemp — caller-owned scratch space (no allocation).
//   crate        — Nash-style damped convergence rate, persisted across
//                  steps (diagnostic only on the Adams path).
//
// Returns: pece_result.
//   converged=false, diverged=true  → PECE blew up; switch to NDF.
//   converged=false, diverged=false → soft failure; controller retries
//                                     with smaller h (or, for switching
//                                     instantiations, also treated as a
//                                     stiffness signal — see controller).
template<class DerivFunc, class Value, class TimeType>
pece_result adams_pece_solve(
    DerivFunc& deriv_func,
    const std::vector<Value>& zn0,
    const std::vector<Value>& zn0_pred,
    const std::vector<Value>& zn1_pred,
    TimeType h,
    TimeType rl1,                 // rl1 = 1 / l[1], matches CVODE cvNlsFunctional
    TimeType t_new,
    double tq4,
    double atol,
    double rtol,
    int max_iter,
    std::vector<Value>& acor,
    std::vector<Value>& y,
    std::vector<Value>& tempv,
    std::vector<Value>& ftemp,
    double& crate,
    cppode::profiler& prof)
{
  using newton_detail::wrms_norm;

  const size_t n = zn0_pred.size();
  int n_fevals = 0;

  // Initialize: y starts at the predictor, acor = 0.
  { auto _t = prof.timer(prof_cat::newton_overhead);
    for (size_t i = 0; i < n; ++i) {
      acor[i] = Value(0);
      y[i]    = zn0_pred[i];
    }
  }

  // E: first f-evaluation at the predicted point.
  { auto _t = prof.timer(prof_cat::f_eval);
    deriv_func(y, ftemp, t_new); }
  ++n_fevals;

  double del = 0.0, delp = 0.0;
  double rate_max = 0.0;

  // C: PECE fixed-point iteration
  //
  // Adams-Moulton corrector in Nordsieck form:
  //   y_{n,new} = y_pred + l[1] * (h * f(y) - z_pred[1])
  // Fixed point exists iff |h * lambda * l[1]| < 1.
  int m;
  for (m = 0; m < max_iter; ++m) {

    // CVODE cvNlsFunctional:
    //   tempv = rl1 * (h*f(y_prev) - zn[1])        -- full new correction
    //   del   = ||tempv - acor||                    -- increment vs previous
    //   acor  = tempv                                -- overwrite
    // rl1 = 1/l[1], NOT l[1].  Using l[1] here caused the corrector to
    // converge to the wrong fixed point (approximately y_n*(1+h)/(1+2h)
    // instead of AM2) so every step looked like huge local error.
    { auto _t = prof.timer(prof_cat::newton_overhead);
      for (size_t i = 0; i < n; ++i) {
        Value new_corr = rl1 * (h * ftemp[i] - zn1_pred[i]);
        tempv[i] = new_corr - acor[i];   // increment (used for del norm)
        acor[i]  = new_corr;              // full correction overwrites acor
      }
    }

    { auto _t = prof.timer(prof_cat::error_norm);
      del = wrms_norm(tempv, y, n, atol, rtol); }

    // y = zn[0]_pred + acor  (using the fresh full correction in acor)
    { auto _t = prof.timer(prof_cat::newton_overhead);
      for (size_t i = 0; i < n; ++i) {
        y[i] = zn0_pred[i] + acor[i];
      }
    }

    // Rate estimate — needs m >= 1 (need previous delp).  LSODA caps
    // rm at 1024 to guard against pathological growth.
    if (m > 0) {
      const double rm = (delp > 0.0)
        ? std::min(1024.0, del / delp)
        : 1024.0;
      crate    = std::max(ndf_constants::CRDOWN * crate, rm);
      rate_max = std::max(rate_max, rm);
    }
    double dcon = del * std::min(1.0, crate) / tq4;

    if (dcon <= 1.0) {
      // Converged — compute acnrm AD-aware, same convention as Newton.
      double acnrm;
      { auto _t = prof.timer(prof_cat::error_norm);
        acnrm = wrms_norm(acor, zn0, n, atol, rtol); }

      // Final E: re-evaluate at corrected y so next step has fresh data.
      { auto _t = prof.timer(prof_cat::f_eval);
        deriv_func(y, ftemp, t_new); }
      ++n_fevals;

      return { /*converged=*/true, /*diverged=*/false,
               acnrm, n_fevals, m + 1, rate_max };
    }

    // Hard divergence: update growing rapidly → fixed point cannot
    // exist for the current h.  PECE can legitimately wobble so we
    // use 2*delp rather than Newton's RDIV = 2.
    if (m >= 1 && delp > 0.0 && del > 2.0 * delp) {
      return { /*converged=*/false, /*diverged=*/true,
               0.0, n_fevals, m + 1, rate_max };
    }

    delp = del;

    // E: re-evaluate f at updated y for the next iteration.
    { auto _t = prof.timer(prof_cat::f_eval);
      deriv_func(y, ftemp, t_new); }
    ++n_fevals;
  }

  // Ran out of iterations without converging or hard-diverging.
  return { /*converged=*/false, /*diverged=*/false,
           0.0, n_fevals, m, rate_max };
}


// ============================================================================
//  multistep_method — compile-time method selector
//
//  Selects which Nordsieck multistep family the stepper instantiates.
//  Both variants share the same Nordsieck history array, the same
//  step-size controller, and the same resize / dense-output / AD
//  machinery.  They differ in:
//
//    - which xxxSet() coefficient routine is called
//    - whether the corrector is Newton (BDF/NDF) or PECE (Adams)
//    - the maximum order
// ============================================================================

enum class multistep_method {
  bdf,      // pure BDF/NDF (default)
  adams     // pure Adams-Moulton PECE
};

namespace multistep_detail {

constexpr bool method_can_use_adams(multistep_method m) {
  return m == multistep_method::adams;
}
constexpr bool method_can_use_bdf_family(multistep_method m) {
  return m == multistep_method::bdf;
}

// Nordsieck max order: 12 for Adams, 5 for BDF/NDF.
constexpr int method_max_order(multistep_method m) {
  return method_can_use_adams(m) ? 12 : 5;
}

} // namespace multistep_detail


// ============================================================================
//  multistepper<Method, Value, JacobianPattern, Resizer>
// ============================================================================

template<
  multistep_method Method = multistep_method::bdf,
  class Value = double,
  class JacobianPattern = cppode::dense_lu_tag,
  class Resizer = initially_resizer
>
class multistepper
{
public:

  typedef void is_multistepper_tag;   // SFINAE tag for dispatch

  // Compile-time method properties (derived from Method)
  static constexpr multistep_method method            = Method;
  static constexpr bool             can_use_adams     = multistep_detail::method_can_use_adams(Method);
  static constexpr bool             can_use_bdf_family= multistep_detail::method_can_use_bdf_family(Method);

  typedef Value                            value_type;
  typedef std::vector<value_type>          state_type;
  typedef state_type                       deriv_type;
  typedef ad_lu::scalar_type_t<Value>      time_type;
  typedef dense_matrix<value_type>         matrix_type;
  typedef csc_matrix<value_type>           sparse_matrix_type;
  typedef Resizer                          resizer_type;

  typedef state_wrapper<state_type>        wrapped_state_type;
  typedef state_wrapper<deriv_type>        wrapped_deriv_type;
  typedef state_wrapper<matrix_type>       wrapped_matrix_type;

  typedef cppode::stepper_tag              stepper_category;
  typedef multistepper<Method, Value, JacobianPattern, Resizer> stepper_type;

  static constexpr bool is_sparse = is_sparse_tag<JacobianPattern>::value;

  // Maximum order this instantiation supports.  For pure NDF / pure BDF
  // this is 5.  For variants that may run Adams it is 12.  The Nordsieck
  // history array is sized accordingly so that pure NDF / pure BDF users
  // incur no memory cost for the Adams machinery.
  static constexpr int max_order   = multistep_detail::method_max_order(Method);
  static constexpr int L_MAX       = max_order + 1;

  // LU solver type
  using lu_type = lu_W<Value, is_sparse>;

  // ====================================================================
  //  Constructor
  // ====================================================================

  multistepper()
    : m_q(1), m_L(2)
    , m_qwait(2)
    , m_nst(0)
    , m_nscon(0)
    , m_max_newton_iter(ndf_constants::NLS_MAXCOR)
    , m_newton_converged(false)
    , m_initialized(false)
    , m_n_fevals(0)
    , m_n_jevals(0)
    , m_hscale(0)
    , m_eta(1.0)
    , m_etamax(ndf_constants::ETAMX1)
    , m_gamma(0)
    , m_gammap(0)
    , m_gamrat(1.0)
    , m_crate(1.0)
    , m_acnrm(0)
    , m_nstlp(0)
    , m_saved_tq5(0)
    , m_tn_current(0)
  {
    m_l.fill(time_type(0));
    m_tq.fill(0.0);
    m_tau.fill(time_type(0));
  }

  // The slab members embed pointers to their own storage_.data() into the
  // dual elements of the slab-bound vectors (m_zn[j].m_v, m_acor.m_v, …).
  // Copying would build a separate buffer with the same address-bound
  // duals — UB. Moves are safe: std::vector::move preserves data() for
  // both the slab storage and the dual-element vectors, so embedded tan_
  // pointers keep pointing at valid memory in the moved-to instance.
  multistepper(const multistepper&)            = delete;
  multistepper& operator=(const multistepper&) = delete;
  multistepper(multistepper&&)                 = default;
  multistepper& operator=(multistepper&&)      = default;

  // ====================================================================
  //  Sensitivity slab priming
  //
  //  Stores n_sens and primes every internal state-vector slab. Idempotent
  //  and cheap; called from the codegen glue before integrate_times() and
  //  re-called from resize_impl() / prepare_dense_output() whenever any
  //  internal vector may have been reallocated.
  //
  //  No-op for non-dynamic-dual value_type (the slab classes are stubs).
  // ====================================================================

  void prepare_sensitivities(unsigned n_sens)
  {
    m_n_sens = n_sens;
    if constexpr (detail::is_dynamic_dual<value_type>::value) {
      if (n_sens == 0) return;

      // The nordsieck_block needs all K facade vectors to be the same
      // length; bail out (defer until a later prepare_sensitivities call)
      // if any slot is still empty.  This mirrors the per-slot empty()
      // guard the per-slab loop used to do.
      bool zn_ready = true;
      const std::size_t n_zn = m_zn[0].m_v.size();
      for (int j = 0; j <= max_order + 1; ++j) {
        if (m_zn[j].m_v.empty() || m_zn[j].m_v.size() != n_zn) {
          zn_ready = false; break;
        }
      }
      if (zn_ready) {
        std::array<std::vector<value_type>*, max_order + 2> facades{};
        for (int j = 0; j <= max_order + 1; ++j) facades[j] = &m_zn[j].m_v;
        m_zn_block.prime(facades,
                         static_cast<unsigned>(n_zn),
                         n_sens);
      }

      bool znd_ready = true;
      const std::size_t n_znd = m_zn_dense[0].m_v.size();
      for (int j = 0; j <= max_order; ++j) {
        if (m_zn_dense[j].m_v.empty() || m_zn_dense[j].m_v.size() != n_znd) {
          znd_ready = false; break;
        }
      }
      if (znd_ready) {
        std::array<std::vector<value_type>*, max_order + 1> facades{};
        for (int j = 0; j <= max_order; ++j) facades[j] = &m_zn_dense[j].m_v;
        m_zn_dense_block.prime(facades,
                               static_cast<unsigned>(n_znd),
                               n_sens);
      }

      if (!m_acor.m_v.empty())
        m_acor_slab.prime(m_acor.m_v,
                          static_cast<unsigned>(m_acor.m_v.size()), n_sens);
      if (!m_y.m_v.empty())
        m_y_slab.prime(m_y.m_v,
                       static_cast<unsigned>(m_y.m_v.size()), n_sens);
      if (!m_tempv.m_v.empty())
        m_tempv_slab.prime(m_tempv.m_v,
                           static_cast<unsigned>(m_tempv.m_v.size()), n_sens);
      if (!m_ftemp.m_v.empty())
        m_ftemp_slab.prime(m_ftemp.m_v,
                           static_cast<unsigned>(m_ftemp.m_v.size()), n_sens);
      if (!m_dense_x_scratch.m_v.empty())
        m_dense_x_scratch_slab.prime(m_dense_x_scratch.m_v,
                                     static_cast<unsigned>(m_dense_x_scratch.m_v.size()),
                                     n_sens);
    }
  }

  unsigned n_sens() const noexcept { return m_n_sens; }

  // ====================================================================
  //  NDF kappa control (runtime)
  //
  //  When true, uses Klopfenstein-Shampine NDF kappa coefficients in
  //  the BDF-family corrector.  When false, classical BDF (kappa = 0).
  //  Default: true (NDF).  Call before the first step.
  // ====================================================================

  void set_use_ndf_kappa(bool v) { m_use_ndf_kappa = v; }
  bool use_ndf_kappa() const { return m_use_ndf_kappa; }

  // ====================================================================
  //  convfail enum — mirrors convfail parameter to lsetup
  //
  //  CV_NO_FAILURES:  no Newton failures (first call or error-test retry)
  //                   → lsetup may reuse cached Jacobian if gamma is close
  //  CV_FAIL_BAD_J:   Newton failed with stale Jacobian → must recompute J
  //  CV_FAIL_OTHER:   Newton failed after fresh J or other failure
  // ====================================================================

  enum class convfail_t {
    no_failures,
    fail_bad_j,
    fail_other
  };

  // ====================================================================
  //  q_max_current — effective max order for this method.
  //  Adams: 12.  BDF/NDF: 5.  Used by the controller to clamp
  //  order-increase decisions.
  // ====================================================================

  int q_max_current() const
  {
    if constexpr (Method == multistep_method::adams) {
      return adams_constants::max_order_adams;     // = 12
    } else {
      return 5;                                    // pure NDF/BDF
    }
  }

  // ====================================================================
  //  do_step — public dispatcher
  //
  //  Selects between the BDF/NDF Newton corrector path and the Adams
  //  PECE corrector path at compile time.
  // ====================================================================

  template<class System, class TimeArg>
  void do_step(
      System& system,
      const state_type& x,
      TimeArg t,
      state_type& x_out,
      TimeArg dt,
      state_type& xerr,
      bool force_setup = false,
      convfail_t convfail = convfail_t::no_failures)
  {
    if constexpr (Method == multistep_method::adams) {
      step_adams(system, x, t, x_out, dt, xerr);
    } else {
      step_bdf_family(system, x, t, x_out, dt, xerr, force_setup, convfail);
    }
  }

  // ====================================================================
  //  on_step_accepted — called by the controller after a successful step.
  //  Currently a no-op (no mode-switching machinery).  Kept as a hook
  //  in case future per-step bookkeeping is needed.
  // ====================================================================

  template<class System, class TimeArg>
  void on_step_accepted(System& /*system*/, TimeArg /*t*/) {}

  // ====================================================================
  //  step_bdf_family: perform one BDF/NDF step
  //
  //  This is the original do_step body, unchanged in behaviour, just
  //  renamed and made a private member.  Delegates Newton iteration
  //  to ndf_newton_solve().
  // ====================================================================

  template<class System, class TimeArg>
  void step_bdf_family(
      System& system,
      const state_type& x,
      TimeArg t,
      state_type& x_out,
      TimeArg dt,
      state_type& xerr,
      bool force_setup = false,
      convfail_t convfail = convfail_t::no_failures)
  {
    using ndf_detail::scalar_value;
    // Convert incoming time arguments to scalar time_type
    time_type t_s  = static_cast<time_type>(scalar_value(t));
    time_type dt_s = static_cast<time_type>(scalar_value(dt));

    typedef typename unwrap_reference<System>::type system_type;
    typedef typename unwrap_reference<
      typename system_type::first_type>::type deriv_func_type;
    typedef typename unwrap_reference<
      typename system_type::second_type>::type jacobi_func_type;

    system_type&      sys         = system;
    deriv_func_type&  deriv_func  = sys.first;
    jacobi_func_type& jacobi_func = sys.second;

    const size_t n = x.size();

    // --- Resize ---
    resize_impl(x);

    // ================================================================
    //  1. Nordsieck rescale
    // ================================================================
    { auto _tp = m_prof.timer(prof_cat::nordsieck);
      if (m_nst > 0) {
        if (std::abs(dt_s - m_hscale) > 1e-14 * std::max(1.0, std::abs(m_hscale))) {
          m_eta = dt_s / m_hscale;
          ndfRescale(n);
        }
        m_h = m_hscale;
      } else {
        m_h = dt_s;
        if (m_hscale == 0.0) m_hscale = dt_s;
      }
    }

    // ================================================================
    //  1b. Compute error weight vector from ACCEPTED solution
    //      (before prediction modifies zn[0]).  Matches CVODE:
    //      ewt[k] = 1/(atol + rtol*|component_k|), fixed for the step.
    //
    //      Layout: interleaved, matching WRMS norm iteration order.
    //      For each state i: [value_weight, deriv_0_weight, ..., deriv_nd-1_weight].
    //      For non-AD (double): just n value weights.
    // ================================================================
    {
      using ndf_detail::scalar_value;
      m_ewt.clear();
      m_ewt.reserve(n * 2);  // heuristic; grows if AD has many derivs
      for (size_t i = 0; i < n; ++i) {
        double yi = std::abs(scalar_value(m_zn[0].m_v[i]));
        m_ewt.push_back(1.0 / (m_atol + m_rtol * yi));

        if constexpr (ad_lu::is_ad<value_type>::value) {
          auto& y_ad = const_cast<value_type&>(m_zn[0].m_v[i]);
          unsigned nd = y_ad.size();
          for (unsigned j = 0; j < nd; ++j) {
            double yd = std::abs(scalar_value(y_ad.d(j)));
            m_ewt.push_back(1.0 / (m_atol + m_rtol * yd));
          }
        }
      }
    }

    // ================================================================
    //  2. Predict
    // ================================================================
    { auto _tp = m_prof.timer(prof_cat::nordsieck); ndfPredict(n); }

    // ================================================================
    //  3. NDF coefficients
    // ================================================================
    ndfSet();

    // --- NDF kappa modification of the iteration matrix ---
    // Classical BDF uses gamma = h / m_l[1].  NDF replaces this by
    //   gamma_NDF = h / ((1 - kappa_q) * m_l[1])
    // so that W = I - gamma_NDF * J matches Shampine's NDF formulation.
    // The factor is absorbed into rl1 = 1 / ((1 - kappa_q) * m_l[1]).
    // When m_use_ndf_kappa = false, kappa_q = 0 and we recover classical BDF.
    const double kappa_q = m_use_ndf_kappa ? ndf_constants::NDF_KAPPA[m_q] : 0.0;
    time_type rl1 = time_type(1.0) / (m_l[1] * time_type(1.0 - kappa_q));
    m_gamma = m_h * rl1;
    if (m_nst == 0) m_gammap = m_gamma;
    m_gamrat = (m_nst > 0) ? m_gamma / m_gammap : 1.0;

    // ================================================================
    //  4+5. Newton solve with Jacobian retry
    // ================================================================
    {
      time_type inv_gamma_dt = time_type(1.0) / m_gamma;
      time_type t_new_s = t_s + m_h;

      bool callSetup = force_setup
      || !m_lu.has_valid_jacobian()
        || (m_nst == 0)
        || (m_nst >= m_nstlp + ndf_constants::MSBP)
        || (std::abs(m_gamrat - 1.0) > ndf_constants::DGMAX);

        // --- Track which trigger fired (diagnostics) ---
        if (callSetup) {
          ++m_n_setup_total;
          if (force_setup)                                       ++m_n_setup_force;
          else if (!m_lu.has_valid_jacobian())                   ++m_n_setup_nojac;
          else if (m_nst == 0)                                   ++m_n_setup_init;
          else if (m_nst >= m_nstlp + ndf_constants::MSBP)      ++m_n_setup_msbp;
          else if (std::abs(m_gamrat - 1.0) > ndf_constants::DGMAX) ++m_n_setup_dgmax;
        }

// Trace is emitted at the end of step_bdf_family (see trace_step) so we
// get a single uniform line per step (post-outcome).  Stash setup info
// so the trailing trace line can render the reason.  Note that m_gamrat
// is reset to 1.0 inside the setup block below (line 1159) so we stash
// the pre-setup gamrat here for the trace; otherwise every DGMAX-triggered
// setup would show gamrat=1 in the CSV.
#ifdef CPPODE_STEP_TRACE
m_trace_setup_fired  = callSetup;
m_trace_gamrat_pre   = static_cast<double>(m_gamrat);
m_trace_setup_reason =
    !callSetup                                               ? "" :
    force_setup                                              ? "force" :
    !m_lu.has_valid_jacobian()                               ? "no_jac" :
    (m_nst == 0)                                             ? "init" :
    (m_nst >= m_nstlp + ndf_constants::MSBP)                 ? "MSBP" :
    (std::abs(m_gamrat - 1.0) > ndf_constants::DGMAX)        ? "DGMAX" : "?";
#endif

bool jcur = false;
convfail_t nls_convfail = convfail;

for (int nls_attempt = 0; nls_attempt < 2; ++nls_attempt) {

  if (callSetup) {
    // CVODE lsetup jbad logic: recompute the Jacobian (not just
    // refactorize W) when the MSBP periodic trigger fires OR on
    // first step / invalid J / Newton failure classified as stale-J.
    // Without periodic recomputation, Newton residuals accumulate
    // in the Nordsieck array and inflate dsm, preventing step-size
    // growth on long smooth phases (Robertson tight-tol: 3K → 900
    // steps).  CVODE's cvDlsSetup sets jbad = true whenever
    // nst > nstlp + MSBP, ensuring periodic Jacobian refresh.
    bool need_new_jacobian =
      (nls_convfail == convfail_t::fail_bad_j)
    || !m_lu.has_valid_jacobian()
    || (m_nst == 0)
    || (m_nst >= m_nstlp + ndf_constants::MSBP);

    if (need_new_jacobian) {
      { auto _tp = m_prof.timer(prof_cat::jac_eval);
        m_lu.call_jacobian(jacobi_func, m_zn[0].m_v, t_new_s); }
      ++m_n_jevals;
      m_lu.cache_jacobian(n);
      { auto _tp = m_prof.timer(prof_cat::lu_factor);
        m_lu.factorize_W(n, inv_gamma_dt); }
    } else {
      auto _tp = m_prof.timer(prof_cat::lu_factor);
      if constexpr (is_sparse) {
        m_lu.refactorize_W_gamma_only(n, inv_gamma_dt);
      } else {
        m_lu.refactorize_W_from_cache(n, inv_gamma_dt);
      }
    }
    m_lu.set_jacobian_valid();
    m_lu.set_lu_valid(m_gamma);
    callSetup = false;
    jcur = need_new_jacobian;
    m_gammap = m_gamma;
    m_l1_at_setup = static_cast<double>(ndf_detail::scalar_value(m_l[1]));
    m_gamrat = 1.0;
    m_crate = 1.0;
    m_nstlp = m_nst;
  }
  // No else: when callSetup is false, the stepper reuses the stale W
  // factorized at gammap.  The Newton residual uses the CURRENT
  // gamma, so convergence is unaffected as long as gamrat is
  // within DGMAX (which triggers callSetup above).  This avoids
  // expensive refactorizations on every step.

  // Newton iteration — pass ewt for CVODE-style pre-prediction weights
  auto result = ndf_newton_solve(
    m_lu,
    deriv_func,
    m_zn[0].m_v,
    m_zn[1].m_v,
    rl1,
    m_gamma,
    t_new_s,
    m_tq[4],
        m_atol, m_rtol,
        m_max_newton_iter,
        m_acor.m_v,
        m_y.m_v,
        m_tempv.m_v,
        m_ftemp.m_v,
        m_crate,
        m_gamrat,
        m_prof,
        m_ewt
  );

  m_acnrm = result.acnrm;
  m_n_fevals += result.n_fevals;

  if (result.converged) {
    m_newton_converged = true;
    break;
  }

  if (!jcur) {
    callSetup = true;
    // CVODE classification: if gamrat is close to 1, drift cannot
    // explain the failure => J itself must be stale. If gamrat is
    // far from 1, the drift is the likely culprit => keep J, only
    // refactorize W at the current gamma.
    if (std::abs(m_gamrat - 1.0) < 0.2) {
      nls_convfail = convfail_t::fail_bad_j;
    } else {
      nls_convfail = convfail_t::fail_other;
    }
    ++m_n_setup_retry;
    continue;
  }

  m_newton_converged = false;
  break;
}
    }

    // ================================================================
    //  6. Output
    // ================================================================
    { auto _tp = m_prof.timer(prof_cat::step_overhead);
      if (m_newton_converged) {
        for (size_t i = 0; i < n; ++i) {
          x_out[i] = m_y.m_v[i];
          xerr[i] = m_acor.m_v[i];
        }
      } else {
        ndfRestore(n);
        for (size_t i = 0; i < n; ++i) {
          x_out[i] = x[i];
          xerr[i] = value_type(0);
        }
      }
    }

    trace_step(/*is_adams_step=*/false,
               /*t_end=*/static_cast<double>(scalar_value(t_s + m_h)));
  }

  // ====================================================================
  //  step_adams: perform one Adams-Moulton PECE step
  //
  //  Mirrors the structure of step_bdf_family but uses
  //  adams_set_coefficients() instead of ndfSet() and adams_pece_solve()
  //  instead of ndf_newton_solve().  No LU, no Jacobian, no setup
  //  triggers — pure non-stiff path.
  // ====================================================================

  template<class System, class TimeArg>
  void step_adams(
      System& system,
      const state_type& x,
      TimeArg t,
      state_type& x_out,
      TimeArg dt,
      state_type& xerr)
  {
    static_assert(can_use_adams,
      "step_adams() instantiated for a Method that does not support Adams");

    using ndf_detail::scalar_value;
    time_type t_s  = static_cast<time_type>(scalar_value(t));
    time_type dt_s = static_cast<time_type>(scalar_value(dt));

    typedef typename unwrap_reference<System>::type system_type;
    typedef typename unwrap_reference<
      typename system_type::first_type>::type deriv_func_type;

    system_type&     sys        = system;
    deriv_func_type& deriv_func = sys.first;

    const size_t n = x.size();

    // --- Resize ---
    resize_impl(x);

    // ================================================================
    //  1. Nordsieck rescale (identical to BDF/NDF path)
    // ================================================================
    { auto _tp = m_prof.timer(prof_cat::nordsieck);
      if (m_nst > 0) {
        if (std::abs(dt_s - m_hscale) > 1e-14 * std::max(1.0, std::abs(m_hscale))) {
          m_eta = dt_s / m_hscale;
          ndfRescale(n);
        }
        m_h = m_hscale;
      } else {
        m_h = dt_s;
        if (m_hscale == 0.0) m_hscale = dt_s;
      }
    }

    // ================================================================
    //  1b. Compute error weight vector from ACCEPTED solution
    //      (before prediction modifies zn[0]).  Same as BDF path.
    // ================================================================
    {
      using ndf_detail::scalar_value;
      m_ewt.clear();
      m_ewt.reserve(n * 2);
      for (size_t i = 0; i < n; ++i) {
        double yi = std::abs(scalar_value(m_zn[0].m_v[i]));
        m_ewt.push_back(1.0 / (m_atol + m_rtol * yi));

        if constexpr (ad_lu::is_ad<value_type>::value) {
          auto& y_ad = const_cast<value_type&>(m_zn[0].m_v[i]);
          unsigned nd = y_ad.size();
          for (unsigned j = 0; j < nd; ++j) {
            double yd = std::abs(scalar_value(y_ad.d(j)));
            m_ewt.push_back(1.0 / (m_atol + m_rtol * yd));
          }
        }
      }
    }

    // ================================================================
    //  2. Predict (Pascal triangle, methodneutral)
    // ================================================================
    { auto _tp = m_prof.timer(prof_cat::nordsieck); ndfPredict(n); }

    // ================================================================
    //  3. Adams coefficients
    // ================================================================
    adams_set_coefficients(m_q, m_qwait, m_h, m_tau, m_l, m_tq);

    // For Adams, gamma = h / l[1] is still defined and the dense-output
    // / order-selection code reads it; keep it consistent.
    time_type rl1 = time_type(1.0) / m_l[1];
    m_gamma  = m_h * rl1;
    if (m_nst == 0) m_gammap = m_gamma;
    m_gamrat = (m_nst > 0) ? m_gamma / m_gammap : 1.0;

    // ================================================================
    //  4. PECE corrector (no LU, no Jacobian, no setup)
    // ================================================================
    time_type t_new_s = t_s + m_h;

    auto result = adams_pece_solve(
        deriv_func,
        x,                    // zn0 — previous accepted state (for atol weighting)
        m_zn[0].m_v,          // zn0_pred — Nordsieck slot 0 after predict
        m_zn[1].m_v,          // zn1_pred — Nordsieck slot 1 after predict
        m_h,
        rl1,                  // 1 / l[1] — matches cvNlsFunctional
        t_new_s,
        m_tq[4],
        m_atol, m_rtol,
        m_max_newton_iter,
        m_acor.m_v,
        m_y.m_v,
        m_tempv.m_v,
        m_ftemp.m_v,
        m_crate,
        m_prof);

    m_acnrm     = result.acnrm;
    m_n_fevals += result.n_fevals;
    m_newton_converged = result.converged;   // reuse the same flag

#ifdef CPPODE_STEP_TRACE
    // Adams path doesn't go through the BDF setup gate — clear stash.
    m_trace_setup_fired  = false;
    m_trace_setup_reason = "";
    m_trace_pece_iters   = result.n_iters;
    m_trace_pece_diverged = result.diverged;
#endif

    // ================================================================
    //  5. Output
    // ================================================================
    { auto _tp = m_prof.timer(prof_cat::step_overhead);
      if (m_newton_converged) {
        for (size_t i = 0; i < n; ++i) {
          x_out[i] = m_y.m_v[i];
          xerr[i]  = m_acor.m_v[i];
        }
      } else {
        ndfRestore(n);
        for (size_t i = 0; i < n; ++i) {
          x_out[i] = x[i];
          xerr[i]  = value_type(0);
        }
      }
    }

    trace_step(/*is_adams_step=*/true,
               /*t_end=*/static_cast<double>(scalar_value(t_s + m_h)));
  }

  // ====================================================================
  //  Error norm: dsm = acnrm * tq[2]
  // ====================================================================

  double error_norm() const { return m_acnrm * m_tq[2]; }

  // ====================================================================
  //  Order selection (eta computations)
  //
  //  All WRMS norms use ewt (pre-prediction weights), matching CVODE.
  //  This ensures consistency between error test and order selection.
  // ====================================================================

  // WRMS norm using pre-prediction ewt, with AD support.
  // Returns max(state_wrms, max_j sens_wrms[j]) — every sensitivity
  // vector is judged on its own per-vector WRMS, and the controller
  // sees the worst.  Matches CVODES `cvSensUpdateNorm` (CV_STAGGERED).
  double wrms_norm_ewt(const std::vector<value_type>& v) const
  {
    using ndf_detail::scalar_value;
    const size_t n = v.size();
    if (n == 0) return 0.0;

    double state_sumsq = 0.0;
    unsigned nd = 0;
    if constexpr (ad_lu::is_ad<value_type>::value) {
      nd = const_cast<value_type&>(v[0]).size();
    }
    std::vector<double> sens_sumsq(nd, 0.0);

    size_t ew = 0;
    for (size_t i = 0; i < n; ++i) {
      double r = std::abs(scalar_value(v[i])) * m_ewt[ew++];
      state_sumsq += r * r;
      if constexpr (ad_lu::is_ad<value_type>::value) {
        auto& v_ad = const_cast<value_type&>(v[i]);
        for (unsigned j = 0; j < nd; ++j) {
          double rd = std::abs(scalar_value(v_ad.d(j))) * m_ewt[ew++];
          sens_sumsq[j] += rd * rd;
        }
      }
    }

    double max_norm = std::sqrt(state_sumsq / n);
    for (unsigned j = 0; j < nd; ++j) {
      double sens_norm = std::sqrt(sens_sumsq[j] / n);
      if (sens_norm > max_norm) max_norm = sens_norm;
    }
    return max_norm;
  }

  double compute_etaqm1() const
  {
    auto _tp = m_prof.timer(prof_cat::error_norm);
    if (m_q <= 1) return 0.0;
    // Use ewt (pre-prediction weights) for consistency with error test,
    // matching CVODE which uses ewt in all WRMS norms.
    double ddn = wrms_norm_ewt(m_zn[m_q].m_v) * m_tq[1];
    return 1.0 / (std::pow(ndf_constants::BIAS1 * ddn, 1.0 / m_q) + ndf_constants::ADDON);
  }

  double compute_etaq(double dsm) const
  {
    return 1.0 / (std::pow(ndf_constants::BIAS2 * dsm, 1.0 / m_L) + ndf_constants::ADDON);
  }

  double compute_etaqp1() const
  {
    auto _tp = m_prof.timer(prof_cat::error_norm);
    using ndf_detail::scalar_value;
    // Clamp to the current method's max order — for ++-instantiations
    // this is 5 in BDF/NDF mode and 12 in Adams mode.
    if (m_q >= q_max_current()) return 0.0;
    if (m_saved_tq5 == 0.0) return 0.0;

    double cquot = (m_tq[5] / m_saved_tq5)
      * std::pow(scalar_value(m_h) / scalar_value(m_tau[2]), m_L);

    const size_t n = m_zn[0].m_v.size();
    double state_sumsq = 0.0;
    unsigned nd = 0;
    if constexpr (ad_lu::is_ad<value_type>::value) {
      if (n > 0) nd = const_cast<value_type&>(m_acor.m_v[0]).size();
    }
    std::vector<double> sens_sumsq(nd, 0.0);

    size_t ew = 0;  // ewt index (interleaved for AD)
    for (size_t i = 0; i < n; ++i) {
      // Value component — use ewt (pre-prediction weights)
      double vi = scalar_value(m_acor.m_v[i])
      - cquot * scalar_value(m_zn[max_order].m_v[i]);
      double r = std::abs(vi) * m_ewt[ew++];
      state_sumsq += r * r;

      // Derivative components (compiles to nothing for double)
      if constexpr (ad_lu::is_ad<value_type>::value) {
        auto& acor_i = const_cast<value_type&>(m_acor.m_v[i]);
        auto& znqm_i = const_cast<value_type&>(m_zn[max_order].m_v[i]);
        for (unsigned j = 0; j < nd; ++j) {
          double vd = scalar_value(acor_i.d(j))
          - cquot * scalar_value(znqm_i.d(j));
          double rd = std::abs(vd) * m_ewt[ew++];
          sens_sumsq[j] += rd * rd;
        }
      }
    }

    // Max-norm over (state, each sens vector) — CVODES convention.
    double dup = (n > 0) ? std::sqrt(state_sumsq / n) : 0.0;
    for (unsigned j = 0; j < nd; ++j) {
      double sens_norm = std::sqrt(sens_sumsq[j] / n);
      if (sens_norm > dup) dup = sens_norm;
    }
    dup *= m_tq[3];
    return 1.0 / (std::pow(ndf_constants::BIAS3 * dup, 1.0 / (m_L + 1)) + ndf_constants::ADDON);
  }

  // ====================================================================
  //  complete_step, restore, rescale, set_order_for_next_step
  // ====================================================================

  void complete_step()
  {
    auto _tp = m_prof.timer(prof_cat::nordsieck);
    const size_t n = m_zn[0].m_v.size();
    ++m_nst;
    ++m_nscon;

    // Track steps at each order
    if (m_q >= 1 && m_q <= max_order)
      ++m_steps_at_order[m_q];

    for (int i = m_q; i >= 2; --i)
      m_tau[i] = m_tau[i - 1];
    if (m_q == 1 && m_nst > 1) m_tau[2] = m_tau[1];
    m_tau[1] = m_h;

    // Nordsieck rank-1 update: zn[j] += l[j] * acor for j ∈ [0, q].
    // Mathematically Z[:, 0:q+1] += outer(acor, l[0:q+1]) on both the
    // value layer (dual.x() per element) and the tangent layer (the
    // unified n*M × (q+1) tangent matrix in m_zn_block).
    {
      const int qp1 = m_q + 1;

      if constexpr (detail::is_dynamic_dual<value_type>::value) {
        using inner_t = typename value_type::value_type;

        // Value layer: per-element loop on the inline .x() field.
        for (int j = 0; j < qp1; ++j) {
          const inner_t lj = static_cast<inner_t>(m_l[j]);
          auto& zj = m_zn[j].m_v;
          const auto& acor_v = m_acor.m_v;
          for (size_t i = 0; i < n; ++i)
            zj[i].x() += lj * acor_v[i].x();
        }

        // Tangent layer: single dger across the whole [n*slab_cols × (q+1)]
        // block when the slabs are primed and the inner type is double
        // (BLAS only applies to plain doubles; nested-dual deriv2 falls
        // back to per-slot vec_axpy_with_slab).  For static-N AD the
        // slab columns equal N (≥ m_n_sens); the inactive trailing
        // columns are zero so the dger zero-propagates correctly.
        if constexpr (std::is_same_v<inner_t, double>) {
          if (m_zn_block.primed() && m_acor_slab.primed() && m_n_sens > 0
              && m_zn_block.slot_stride() == m_acor_slab.tangent_size())
          {
            std::array<double, max_order + 2> ll{};
            for (int j = 0; j < qp1; ++j) ll[j] = static_cast<double>(m_l[j]);
            int M_ = static_cast<int>(m_zn_block.slot_stride());
            int N_ = qp1;
            int inc = 1;
            double one = 1.0;
            F77_CALL(dger)(&M_, &N_, &one,
                           m_acor_slab.tangent_data(), &inc,
                           ll.data(), &inc,
                           m_zn_block.tangent_block_data(), &M_);
          }
        } else {
          for (int j = 0; j < qp1; ++j)
            vec_axpy_with_slab(m_zn[j].m_v, m_zn_block.slab(j),
                               static_cast<double>(m_l[j]),
                               m_acor.m_v, m_acor_slab);
        }
      } else {
        // Plain double value_type: existing per-slot path.
        for (int j = 0; j < qp1; ++j)
          vec_axpy_with_slab(m_zn[j].m_v, m_zn_block.slab(j),
                             static_cast<double>(m_l[j]),
                             m_acor.m_v, m_acor_slab);
      }
    }

    m_qwait--;
    if (m_qwait == 1 && m_q != q_max_current()) {
      for (size_t i = 0; i < n; ++i)
        m_zn[max_order].m_v[i] = m_acor.m_v[i];
      m_saved_tq5 = m_tq[5];
    }
  }

  void restore()
  {
    auto _tp = m_prof.timer(prof_cat::nordsieck);
    ndfRestore(m_zn[0].m_v.size());
  }

  void rescale(time_type eta)
  {
    auto _tp = m_prof.timer(prof_cat::nordsieck);
    m_eta = eta;
    ndfRescale(m_zn[0].m_v.size());
  }

  void set_order_for_next_step(int new_q)
  {
    // BDF/NDF and Adams use different Nordsieck adjustment formulas
    // (cvAdjustBDF vs cvAdjustAdams in CVODE).  Selected at compile time.
    constexpr bool in_adams = (Method == multistep_method::adams);

    if (new_q == m_q + 1) {
      ++m_n_order_up;
      if constexpr (in_adams) adamsIncreaseOrder();
      else                    ndfIncreaseOrder();
      m_q = new_q;
      m_L = new_q + 1;
    } else if (new_q == m_q - 1) {
      ++m_n_order_down;
      if constexpr (in_adams) adamsDecreaseOrder();
      else                    ndfDecreaseOrder();
      m_q = new_q;
      m_L = new_q + 1;
    }
    m_nscon = 0;
  }

  // ====================================================================
  //  Initialize
  // ====================================================================

  template<class StateIn, class DerivIn, class TimeArg>
  void initialize(const StateIn& x0, TimeArg t0,
                  const DerivIn& f0, TimeArg dt)
  {
    using ndf_detail::scalar_value;
    time_type t0_s = static_cast<time_type>(scalar_value(t0));
    time_type dt_s = static_cast<time_type>(scalar_value(dt));
    const size_t n = x0.size();
    resize_impl(x0);

    for (size_t i = 0; i < n; ++i) {
      m_zn[0].m_v[i] = x0[i];
      m_zn[1].m_v[i] = dt_s * f0[i];
    }
    for (int j = 2; j <= max_order + 1; ++j)
      for (size_t i = 0; i < n; ++i)
        m_zn[j].m_v[i] = value_type(0);

    m_q = 1; m_L = 2; m_qwait = 2;
    m_nst = 0; m_nscon = 0;
    m_h = dt_s; m_hscale = dt_s;
    m_eta = time_type(1);
    m_etamax = ndf_constants::ETAMX1;
    m_gammap = time_type(0);
    m_gamrat = 1.0; m_crate = 1.0;
    m_acnrm = 0; m_nstlp = 0;
    m_saved_tq5 = 0; m_tn_current = t0_s;

    m_tau.fill(time_type(0));
    m_tau[1] = dt_s;
    m_initialized = true;
    m_lu.invalidate();
  }

  template<class TimeArg>
  void restart_from_order1(const state_type& x, TimeArg t, TimeArg dt)
  {
    const size_t n = x.size();
    m_q = 1; m_L = 2; m_qwait = 2; m_nscon = 0;
    for (size_t i = 0; i < n; ++i)
      m_zn[0].m_v[i] = x[i];
    m_h = static_cast<time_type>(ndf_detail::scalar_value(dt)); m_hscale = m_h;
    m_eta = time_type(1);
    m_etamax = ndf_constants::ETAMX1;
    m_gammap = time_type(0);
    m_gamrat = 1.0; m_crate = 1.0;
    m_saved_tq5 = 0; m_tn_current = static_cast<time_type>(ndf_detail::scalar_value(t));
    m_lu.invalidate();
  }

  // ====================================================================
  //  Dense output
  // ====================================================================

  void prepare_dense_output()
  {
    auto _tp = m_prof.timer(prof_cat::dense_snapshot);
    const size_t n = m_zn[0].m_v.size();
    // m_zn_dense is sized lazily per-order. If any slot was just resized,
    // its new dual elements have size_=0 and the subsequent vector copy at
    // m_v = m_zn[j].m_v would arena-allocate per element. Re-prime first.
    bool dense_resized = false;
    for (int j = 0; j <= m_q; ++j) {
      if (m_zn_dense[j].m_v.size() != n) {
        m_zn_dense[j].m_v.resize(n);
        dense_resized = true;
      }
    }
    if (dense_resized && m_n_sens != 0)
      prepare_sensitivities(m_n_sens);

    // Value layer: per-element dual = dual copy.  Only the first (q+1)
    // slots are alive — leave the rest untouched (consistent with the
    // legacy per-slot loop, which only ran for j ∈ [0, q]).
    for (int j = 0; j <= m_q; ++j) {
      auto& dst = m_zn_dense[j].m_v;
      const auto& src = m_zn[j].m_v;
      for (size_t i = 0; i < n; ++i) dst[i] = src[i];
    }

    // Tangent layer: when both blocks are primed and the active block
    // shape matches, a single std::memcpy of the (q+1) leading slots'
    // tangent storage replaces the (q+1) per-slot vec_copy_with_slab
    // calls. Slots [q+1, max_order] hold stale tangents but they're
    // never read — eval_dense_into() only touches slots [0, m_q_dense].
    if constexpr (detail::is_dynamic_dual<value_type>::value) {
      using inner_t = typename value_type::value_type;
      if (m_zn_block.primed() && m_zn_dense_block.primed() &&
          m_zn_block.n_rows() == m_zn_dense_block.n_rows() &&
          m_zn_block.n_cols() == m_zn_dense_block.n_cols())
      {
        const std::size_t copy_slots = static_cast<std::size_t>(m_q + 1);
        const std::size_t per_slot   = m_zn_block.slot_stride();
        std::memcpy(m_zn_dense_block.tangent_block_data(),
                    m_zn_block.tangent_block_data(),
                    copy_slots * per_slot * sizeof(inner_t));
      }
    }

    m_q_dense = m_q;
    m_h_dense = m_h;
    m_tn_dense = m_tn_current;
    m_dense_valid = true;
  }

  // ====================================================================
  //  Accessors
  // ====================================================================

  int current_order() const { return m_q; }
  int steps_at_current_order() const { return m_nscon; }
  bool newton_converged() const { return m_newton_converged; }
  bool is_initialized() const { return m_initialized; }
  double acnrm() const { return m_acnrm; }
  const state_type& acor() const { return m_acor.m_v; }
  const std::array<double, 6>& tq() const { return m_tq; }
  int qwait() const { return m_qwait; }
  int nst() const { return m_nst; }
  double etamax() const { return m_etamax; }
  void set_etamax(double em) { m_etamax = em; }
  time_type h() const { return m_h; }
  time_type hscale() const { return m_hscale; }
  template<class TimeArg> void set_hscale(TimeArg hs) { m_hscale = static_cast<time_type>(ndf_detail::scalar_value(hs)); }
  double saved_tq5() const { return m_saved_tq5; }
  const state_type& zn(int j) const { return m_zn[j].m_v; }
  int current_L() const { return m_L; }
  const std::array<time_type, max_order + 1>& tau() const { return m_tau; }

  template<class TimeArg> void set_tn_current(TimeArg tn) { m_tn_current = static_cast<time_type>(ndf_detail::scalar_value(tn)); }
  void set_tolerances(double atol, double rtol) { m_atol = atol; m_rtol = rtol; }
  void set_qwait(int qw) { m_qwait = qw; }

  int n_fevals() const { return m_n_fevals; }
  int n_jevals() const { return m_n_jevals; }
  int n_setups() const { return m_n_setup_total; }
  void reset_counters() { m_n_fevals = 0; m_n_jevals = 0; }

  void report_setup_triggers() const {
    // Setup-trigger breakdown: only meaningful for methods that call
    // lsetup (BDF / NDF / BDF++ / NDF++).  Pure Adams has no Jacobian.
    if constexpr (can_use_bdf_family) {
      std::fprintf(stderr, "\n=== BDF/NDF setup trigger breakdown (total = %d) ===\n", m_n_setup_total);
      std::fprintf(stderr, "  force_setup (err/conv fail): %d\n", m_n_setup_force);
      std::fprintf(stderr, "  !has_valid_jacobian:         %d\n", m_n_setup_nojac);
      std::fprintf(stderr, "  nst == 0 (initial):          %d\n", m_n_setup_init);
      std::fprintf(stderr, "  MSBP periodic (>%d steps):   %d\n", ndf_constants::MSBP, m_n_setup_msbp);
      std::fprintf(stderr, "  DGMAX (|gamrat-1| > %.2f):   %d\n", ndf_constants::DGMAX, m_n_setup_dgmax);
      std::fprintf(stderr, "  Newton retry (stale J):      %d\n", m_n_setup_retry);
      std::fprintf(stderr, "================================================\n");
    }

    std::fprintf(stderr, "\n=== Order change statistics ===\n");
    std::fprintf(stderr, "  Order increases (q→q+1):  %d\n", m_n_order_up);
    std::fprintf(stderr, "  Order decreases (q→q-1):  %d\n", m_n_order_down);
    std::fprintf(stderr, "  Steps per order:\n");
    for (int q = 1; q <= max_order; ++q) {
      if (m_steps_at_order[q] > 0)
        std::fprintf(stderr, "    q=%d: %d steps\n", q, m_steps_at_order[q]);
    }
    std::fprintf(stderr, "===================================\n\n");
  }

  // ====================================================================
  //  trace_step — unified per-step trace row
  //
  //  One entry per invocation of step_* (regardless of whether the step
  //  is ultimately accepted by the controller).  When CPPODE_STEP_TRACE
  //  is not defined this is a no-op — the compiler eliminates the body.
  //
  //  The row is appended to the process-wide TraceBuffer declared in
  //  cppode_step_trace.hpp.  The generated entry point marshals the
  //  buffer into an R list after integration and clears it.  No file
  //  I/O happens here — `solveODE()` on the R side decides what to do
  //  with the trace.
  //
  //  dsm ~ acnrm * tq[2] is the post-step error norm the controller's
  //  error test will use.  Stored pre-error-test, so "dsm > 1" rows
  //  correspond to steps that the controller will reject.  `t` is the
  //  time at the end of the attempted step (t_old + h).
  // ====================================================================
  void trace_step(bool is_adams_step, double t_end) const {
#ifdef CPPODE_STEP_TRACE
    using ndf_detail::scalar_value;
    const double h_d      = static_cast<double>(scalar_value(m_h));
    const double gamma_d  = static_cast<double>(scalar_value(m_gamma));
    const double tq2_d    = m_tq[2];
    const double dsm_est  = m_acnrm * (tq2_d != 0.0 ? tq2_d : 1.0);

    // acnrm_state: WRMS computed from the state (value) slots only,
    // directly comparable to CVODE's reconstructed `acnrm` (which the
    // CVODES public API exposes only over the state block).  At
    // sens=FALSE this equals m_acnrm; at sens=TRUE it is the non-sens
    // slice of it.
    double acnrm_state_sq = 0.0;
    const size_t n_state = m_acor.m_v.size();
    for (size_t i = 0; i < n_state; ++i) {
      const double acor_v = std::abs(scalar_value(m_acor.m_v[i]));
      const double y_v    = std::abs(scalar_value(m_zn[0].m_v[i]));
      const double denom  = m_atol + m_rtol * y_v;
      if (denom > 0.0) {
        const double r = acor_v / denom;
        acnrm_state_sq += r * r;
      }
    }
    const double acnrm_state =
      (n_state > 0) ? std::sqrt(acnrm_state_sq / static_cast<double>(n_state)) : 0.0;

    // Resolve the mode label for the trace line — known at compile time.
    const char* mode_label;
    if constexpr (Method == multistep_method::adams) {
      mode_label = "ADAMS";
    } else {
      mode_label = m_use_ndf_kappa ? "NDF" : "BDF";
    }

    const char* setup_reason =
      (!is_adams_step && m_trace_setup_fired) ? m_trace_setup_reason : "";

    // m_gamrat is reset to 1.0 by the setup block before trace_step runs.
    // Report the pre-setup value stashed in m_trace_gamrat_pre so the trace
    // reflects the value that actually triggered (or did not trigger) the
    // DGMAX setup path.
    const double gamrat_out =
      (!is_adams_step && m_trace_setup_fired) ? m_trace_gamrat_pre : m_gamrat;

    auto& tb = ndf_detail::get_trace_buffer();
    tb.nst.push_back(m_nst);
    tb.t.push_back(t_end);
    tb.h.push_back(h_d);
    tb.q.push_back(m_q);
    tb.dsm.push_back(dsm_est);
    tb.acnrm.push_back(m_acnrm);
    tb.acnrm_state.push_back(acnrm_state);
    tb.tq2.push_back(tq2_d);
    tb.gamma.push_back(gamma_d);
    tb.gamrat.push_back(gamrat_out);
    tb.newton_conv.push_back(static_cast<int>(m_newton_converged));
    tb.mode.emplace_back(mode_label);
    tb.nfe.push_back(m_n_fevals);
    tb.njev.push_back(m_n_jevals);
    tb.nsetups.push_back(m_n_setup_total);
    tb.setup_reason.emplace_back(setup_reason);
    tb.pece_iters.push_back(is_adams_step ? m_trace_pece_iters : 0);
    tb.pece_diverged.push_back(
      is_adams_step ? static_cast<int>(m_trace_pece_diverged) : 0);
#else
    (void)is_adams_step; (void)t_end;
#endif
  }

  int max_achievable_order() const { return max_order; }
  void increment_step_counter() { }
  template<class TimeArg> void push_history(const state_type&, TimeArg, TimeArg) { }
  value_type error_constant() const { return value_type(m_tq[2]); }
  value_type error_constant_km1() const { return value_type(m_tq[1]); }
  value_type error_constant_kp1() const { return value_type(m_tq[3]); }

  // LU access (for controller invalidation)
  void invalidate_lu() { m_lu.invalidate(); }
  bool has_valid_jacobian() const { return m_lu.has_valid_jacobian(); }
  bool has_valid_lu() const { return m_lu.has_valid_lu(); }
  double last_factorized_dt() const {
    return static_cast<double>(ndf_detail::scalar_value(m_lu.last_factorized_dt()));
  }

  // Save acor → zn[qmax]
  void save_acor_to_zn_qmax()
  {
    const size_t n = m_zn[0].m_v.size();
    for (size_t i = 0; i < n; ++i)
      m_zn[max_order].m_v[i] = m_acor.m_v[i];
  }

  // Reload zn[1] = h * f(tn, zn[0]) for order-1 restart
  template<class DerivFunc>
  void reload_zn1_from_f(DerivFunc& deriv_func)
  {
    const size_t n = m_zn[0].m_v.size();
    deriv_func(m_zn[0].m_v, m_ftemp.m_v, m_tn_current);
    ++m_n_fevals;
    for (size_t i = 0; i < n; ++i)
      m_zn[1].m_v[i] = m_h * m_ftemp.m_v[i];
  }

  template<class StateType>
  void adjust_size(const StateType& x) { resize_impl(x); }

  // Dense output accessors
  bool has_dense_output() const { return m_dense_valid; }
  int dense_order() const { return m_q_dense; }
  time_type dense_h() const { return m_h_dense; }
  time_type dense_tn() const { return m_tn_dense; }
  const state_type& dense_zn(int j) const { return m_zn_dense[j].m_v; }

  // ====================================================================
  //  eval_dense_into
  //
  //  Slab-aware Horner evaluation of the Nordsieck dense-output polynomial:
  //
  //      x(t) = sum_{j=0}^{q_dense} zn_dense[j] * s^j,   s = (t - t_n) / h
  //
  //  evaluated bottom-up:
  //      x = zn[q]
  //      for j = q-1 .. 0:  x = x * s + zn[j]
  //
  //  For dual<S, 0> with primed slabs (the heap-AD path), the Horner runs
  //  on a slab-bound scratch buffer so the tangent half is a single tight
  //  loop over the contiguous [n × n_sens] block instead of n separate
  //  per-element ET materialisations whose MulOp.tangent path computes
  //  the symmetric `bv*at + av*bt` even when the scalar side has zero
  //  tangent. The scratch is then copied into the caller-supplied
  //  StateOut& x in one pass.
  //
  //  For non-dual / static-N value types (no slab) the routine collapses
  //  to the same per-element Horner the previous calc_state had — no
  //  scratch copy, no overhead.
  // ====================================================================

  template<class TimeArg, class StateOut>
  void eval_dense_into(TimeArg t, StateOut& x)
  {
    using ndf_detail::scalar_value;
    assert(m_dense_valid && "dense output snapshot not available");

    const int    q  = m_q_dense;
    const auto   h  = m_h_dense;
    const auto   tn = m_tn_dense;
    const auto   s  = (static_cast<time_type>(scalar_value(t)) - tn) / h;
    const size_t n  = m_zn_dense[0].m_v.size();

    if constexpr (detail::is_dynamic_dual<value_type>::value) {
      using S = typename value_type::value_type;

      const bool slab_path = m_dense_x_scratch_slab.primed()
                          && m_zn_dense_block.slab(0).primed();

      if (slab_path) {
        // --- Values: scalar Horner directly into scratch.x() ---
        const auto& zq_v = m_zn_dense[q].m_v;
        for (size_t i = 0; i < n; ++i)
          m_dense_x_scratch.m_v[i].x() = zq_v[i].x();
        for (int j = q - 1; j >= 0; --j) {
          const auto& zj_v = m_zn_dense[j].m_v;
          for (size_t i = 0; i < n; ++i) {
            S& xv = m_dense_x_scratch.m_v[i].x();
            xv = xv * static_cast<S>(s) + zj_v[i].x();
          }
        }

        // --- Tangents: SoA on the contiguous slab. ---
        S* xp = m_dense_x_scratch_slab.tangent_data();
        const std::size_t total = m_dense_x_scratch_slab.tangent_size();
        if (total > 0) {
          if constexpr (std::is_same_v<S, double>) {
            // x_slab = zq_slab
            int len = static_cast<int>(total);
            int inc = 1;
            F77_CALL(dcopy)(&len,
                            const_cast<double*>(m_zn_dense_block.slab(q).tangent_data()), &inc,
                            xp, &inc);
            // for j: x_slab = x_slab * s + zj_slab
            const double s_d = static_cast<double>(s);
            const double one = 1.0;
            for (int j = q - 1; j >= 0; --j) {
              double a = s_d;
              F77_CALL(dscal)(&len, &a, xp, &inc);
              F77_CALL(daxpy)(&len, const_cast<double*>(&one),
                              const_cast<double*>(m_zn_dense_block.slab(j).tangent_data()),
                              &inc, xp, &inc);
            }
          } else {
            const S* zp = m_zn_dense_block.slab(q).tangent_data();
            for (std::size_t k = 0; k < total; ++k) xp[k] = zp[k];
            const S s_v = static_cast<S>(s);
            for (int j = q - 1; j >= 0; --j) {
              const S* zjp = m_zn_dense_block.slab(j).tangent_data();
              for (std::size_t k = 0; k < total; ++k)
                xp[k] = xp[k] * s_v + zjp[k];
            }
          }
        }

        // --- Copy scratch -> caller's x (per-element dual = dual; in-place
        //     tangent copy when x[i] already has a buffer of size n_sens). ---
        for (size_t i = 0; i < n; ++i)
          x[i] = m_dense_x_scratch.m_v[i];
        return;
      }
      // Slab-less path falls through to the generic Horner below.
    }

    // Generic (per-element ET) Horner — used for non-dual value types
    // and for the dynamic-dual case before slabs are primed.
    const auto& zq = m_zn_dense[q].m_v;
    for (size_t i = 0; i < n; ++i)
      x[i] = zq[i];
    for (int j = q - 1; j >= 0; --j) {
      const auto& zj = m_zn_dense[j].m_v;
      for (size_t i = 0; i < n; ++i)
        x[i] = x[i] * s + zj[i];
    }
  }

  // Direct LU access (for advanced use / testing)
  lu_type& lu() { return m_lu; }
  const lu_type& lu() const { return m_lu; }

private:

  // ====================================================================
  //  ndfPredict / ndfRestore / ndfSet / ndfRescale
  //
  //  ndfPredict applies the Nordsieck Pascal shift. As a matrix op the
  //  loop `for k ∈ [1, q]: for j ∈ [q, k]: zn[j-1] += zn[j]` is
  //  Z := Z @ P where P is the (q+1) × (q+1) lower-triangular unit-Pascal
  //  matrix P[i, j] = C(i, j) for i >= j (viewing Z as a
  //  states×slots matrix with slot j as column j). ndfRestore is the
  //  inverse, P_inv[i, j] = (-1)^(i-j) * C(i, j).
  //
  //  When the tangent block is primed and the inner type is plain
  //  double, the (n*M) × (q+1) tangent matrix is updated in one dtrmm
  //  call. Value layer stays per-element (the .x() fields aren't
  //  contiguous across slots — they live inside the dual struct).
  //  Nested-dual deriv2 falls back to the legacy per-slot loop.
  // ====================================================================

private:
  static constexpr int LM = max_order + 2;

  static const std::array<double, LM * LM>& pascal_predict_table_() {
    // Column-major (i, j) -> P[j*LM + i] = C(i, j) for i >= j else 0.
    static const std::array<double, LM * LM> P = []() {
      std::array<double, LM * LM> p{};
      long long binom[LM][LM] = {};
      for (int i = 0; i < LM; ++i) {
        binom[i][0] = 1;
        for (int j = 1; j <= i; ++j)
          binom[i][j] = binom[i-1][j-1] + (j <= i-1 ? binom[i-1][j] : 0);
      }
      for (int j = 0; j < LM; ++j)
        for (int i = j; i < LM; ++i)
          p[static_cast<std::size_t>(j) * LM + i] = static_cast<double>(binom[i][j]);
      return p;
    }();
    return P;
  }

  static const std::array<double, LM * LM>& pascal_restore_table_() {
    static const std::array<double, LM * LM> P = []() {
      std::array<double, LM * LM> p{};
      long long binom[LM][LM] = {};
      for (int i = 0; i < LM; ++i) {
        binom[i][0] = 1;
        for (int j = 1; j <= i; ++j)
          binom[i][j] = binom[i-1][j-1] + (j <= i-1 ? binom[i-1][j] : 0);
      }
      for (int j = 0; j < LM; ++j)
        for (int i = j; i < LM; ++i)
          p[static_cast<std::size_t>(j) * LM + i]
            = ((i - j) % 2 == 0 ? 1.0 : -1.0) * static_cast<double>(binom[i][j]);
      return p;
    }();
    return P;
  }

  // Pascal shift driver for both predict (sign=+1) and restore (sign=-1).
  void apply_pascal_shift_(size_t n, bool restore)
  {
    if (m_q < 1) return;

    if constexpr (detail::is_dynamic_dual<value_type>::value) {
      using inner_t = typename value_type::value_type;
      const double sign = restore ? -1.0 : 1.0;

      // Value layer: per-element legacy loop on .x().
      for (int k = 1; k <= m_q; ++k)
        for (int j = m_q; j >= k; --j) {
          auto& zlow = m_zn[j - 1].m_v;
          const auto& zup = m_zn[j].m_v;
          for (size_t i = 0; i < n; ++i)
            zlow[i].x() += static_cast<inner_t>(sign) * zup[i].x();
        }

      // Tangent layer: single dtrmm across the (n*slab_cols) × (q+1)
      // block. Static-N: slab cols = N (≥ m_n_sens); dynamic-N: = M.
      // The Pascal P is unit lower-triangular so trailing inactive
      // tangent columns (which are zero by construction) stay zero.
      if constexpr (std::is_same_v<inner_t, double>) {
        if (m_zn_block.primed() && m_n_sens > 0) {
          const auto& P = restore ? pascal_restore_table_()
                                  : pascal_predict_table_();
          char side = 'R', uplo = 'L', transa = 'N', diag = 'U';
          int M_ = static_cast<int>(m_zn_block.slot_stride());
          int N_ = m_q + 1;
          int lda = LM;
          int ldb = M_;
          double one = 1.0;
          F77_CALL(dtrmm)(&side, &uplo, &transa, &diag,
                          &M_, &N_, &one,
                          const_cast<double*>(P.data()), &lda,
                          m_zn_block.tangent_block_data(), &ldb
                          FCONE FCONE FCONE FCONE);
        }
      } else {
        // Nested-dual / non-double inner: legacy per-slot path.
        for (int k = 1; k <= m_q; ++k)
          for (int j = m_q; j >= k; --j)
            vec_axpy_with_slab(m_zn[j - 1].m_v, m_zn_block.slab(j - 1),
                               sign,
                               m_zn[j].m_v, m_zn_block.slab(j));
      }
    } else {
      const double sign = restore ? -1.0 : 1.0;
      for (int k = 1; k <= m_q; ++k)
        for (int j = m_q; j >= k; --j)
          vec_axpy_with_slab(m_zn[j - 1].m_v, m_zn_block.slab(j - 1),
                             sign,
                             m_zn[j].m_v, m_zn_block.slab(j));
    }
  }

public:
  void ndfPredict(size_t n) { apply_pascal_shift_(n, /*restore=*/false); }
  void ndfRestore(size_t n) { apply_pascal_shift_(n, /*restore=*/true);  }

  // ====================================================================
  //  ndfSet
  //
  //  Builds the Nordsieck correction coefficients m_l[0..q] for the
  //  current order q and step size m_h, given the past step history
  //  m_tau[1..q-1].  At constant h this reduces to the classical BDF_q
  //  coefficients with m_l[1] = gamma_k (= sum_{j=1}^k 1/j).
  //
  //  NDF modification (uses_ndf_kappa = true):
  //  --------------------------------------
  //  IMPORTANT: m_l[0..q] is NOT modified here.  The Nordsieck update
  //  rule in complete_step uses m_l[j] to map the correction acor
  //  (= h^{k+1}/(k+1)! * nabla^{k+1} y) back to the history array:
  //
  //      zn[j] += m_l[j] * acor     for j = 0..q.
  //
  //  Scaling m_l[1] here would corrupt zn[1] (the h*y' Nordsieck slot)
  //  and destroy the history for subsequent steps.
  //
  //  Instead, the NDF kappa factor is applied in do_step, where
  //  gamma = h / m_l[1] is converted to the NDF gamma
  //
  //      gamma_NDF = h / ((1 - kappa) * m_l[1]).
  //
  //  ndfSetTq below does rescale m_tq[2] (the error-test constant),
  //  which is a stateless per-step quantity and safe to modify.
  // ====================================================================

  void ndfSet()
  {
    m_l.fill(time_type(0));
    time_type xi_inv(1), xistar_inv(1);
    m_l[0] = m_l[1] = time_type(1);
    time_type alpha0 = time_type(-1);
    time_type alpha0_hat = time_type(-1);
    time_type hsum = m_h;

    if (m_q > 1) {
      for (int j = 2; j < m_q; ++j) {
        hsum += m_tau[j - 1];
        xi_inv = m_h / hsum;
        alpha0 -= time_type(1) / time_type(j);
        for (int i = j; i >= 1; --i)
          m_l[i] += m_l[i - 1] * xi_inv;
      }
      alpha0 -= time_type(1) / time_type(m_q);
      xistar_inv = -m_l[1] - alpha0;
      hsum += m_tau[m_q - 1];
      xi_inv = m_h / hsum;
      alpha0_hat = -m_l[1] - xi_inv;
      for (int i = m_q; i >= 1; --i)
        m_l[i] += m_l[i - 1] * xistar_inv;
    }

    // NDF kappa for current order (0 when m_use_ndf_kappa = false)
    const double kappa_q = m_use_ndf_kappa ? ndf_constants::NDF_KAPPA[m_q] : 0.0;

    ndfSetTq(hsum, static_cast<double>(alpha0), static_cast<double>(alpha0_hat),
             static_cast<double>(xi_inv), static_cast<double>(xistar_inv), kappa_q);
  }

  // ====================================================================
  //  ndfSetTq
  //
  //  Builds the error-test constants m_tq[1..5] used by the controller.
  //
  //  NDF modification (kappa != 0):
  //  ------------------------------
  //  For classical BDF_k the leading truncation error is
  //      LTE_BDF = (1 / (k+1)) * h^{k+1} y^{(k+1)}.
  //  For the Klopfenstein-Shampine NDF_k it becomes
  //      LTE_NDF = (kappa * gamma_k + 1/(k+1)) * h^{k+1} y^{(k+1)}
  //              = (1 + kappa * gamma_k * (k+1)) * LTE_BDF.
  //  We rescale m_tq[2] (current-order error constant) by this factor
  //  so that the error test dsm = acnrm * tq[2] remains calibrated.
  //  gamma_k = -alpha0 in the variable-step Nordsieck formulation.
  // ====================================================================

  void ndfSetTq(time_type hsum, double alpha0, double alpha0_hat,
                double xi_inv, double xistar_inv, double kappa_q)
  {
    double A1 = 1.0 - alpha0_hat + alpha0;
    double A2 = 1.0 + m_q * A1;

    m_tq[2] = std::abs(A1 / (alpha0 * A2));
    m_tq[5] = std::abs(A2 * xistar_inv / (static_cast<double>(m_l[m_q]) * xi_inv));

    // Newton convergence tolerance: based on the BDF error constant
    // (before NDF scaling) so that the Newton residual stays small
    // relative to the BDF truncation error.  Using the NDF-scaled
    // tq[2] here would loosen the tolerance by 1/|ndf_factor| (up to
    // 1.76× at q=4), causing Newton residual drift in the Nordsieck
    // array that prevents step-size growth on long integrations.
    m_tq[4] = ndf_constants::CORTES / m_tq[2];

    // NDF: scale current-order error constant by (1 + kappa * gamma_k * (k+1))
    if (kappa_q != 0.0) {
      double gamma_k = -alpha0;              // variable-step gamma_k
      double ndf_factor = 1.0 + kappa_q * gamma_k * (m_q + 1);
      m_tq[2] *= std::abs(ndf_factor);
    }

    if (m_qwait == 1) {
      if (m_q > 1) {
        double C = xistar_inv / static_cast<double>(m_l[m_q]);
        double A3 = alpha0 + 1.0 / m_q;
        double A4 = alpha0_hat + xi_inv;
        double Cpinv = (1.0 - A4 + A3) / A3;
        m_tq[1] = std::abs(C * Cpinv);
      } else {
        m_tq[1] = 1.0;
      }
      hsum += m_tau[m_q];
      double xi_inv_qp1 = static_cast<double>(m_h) / static_cast<double>(hsum);
      double A5 = alpha0 - 1.0 / (m_q + 1);
      double A6 = alpha0_hat - xi_inv_qp1;
      double Cppinv = (1.0 - A6 + A5) / A2;
      m_tq[3] = std::abs(Cppinv / (xi_inv_qp1 * (m_q + 2) * A5));
    }
  }

  void ndfRescale(size_t n)
  {
    time_type factor = m_eta;
    for (int j = 1; j <= m_q; ++j) {
      vec_scale_with_slab(m_zn[j].m_v, m_zn_block.slab(j),
                          static_cast<double>(factor));
      factor *= m_eta;
    }
    m_h = m_hscale * m_eta;
    m_hscale = m_h;
    m_nscon = 0;
  }

  void ndfIncreaseOrder()
  {
    const size_t n = m_zn[0].m_v.size();
    std::array<time_type, L_MAX + 1> ll;
    ll.fill(time_type(0));
    time_type alpha1(1), prod(1), xiold(1);
    double alpha0_inc = -1.0;
    ll[2] = time_type(1);
    time_type hsum_v = m_hscale;
    if (m_q > 1) {
      for (int j = 1; j < m_q; ++j) {
        hsum_v += m_tau[j + 1];
        time_type xi = hsum_v / m_hscale;
        prod *= xi;
        alpha0_inc -= 1.0 / (j + 1);
        alpha1 += time_type(1) / xi;
        for (int i = j + 2; i >= 2; --i) ll[i] = ll[i] * xiold + ll[i - 1];
        xiold = xi;
      }
    }
    double A1 = (-alpha0_inc - static_cast<double>(alpha1)) / static_cast<double>(prod);
    for (size_t i = 0; i < n; ++i)
      m_zn[m_L].m_v[i] = value_type(A1) * m_zn[max_order].m_v[i];
    for (int j = 2; j <= m_q; ++j)
      vec_axpy_with_slab(m_zn[j].m_v, m_zn_block.slab(j),
                         static_cast<double>(ll[j]),
                         m_zn[m_L].m_v, m_zn_block.slab(m_L));
  }

  void ndfDecreaseOrder()
  {
    const size_t n = m_zn[0].m_v.size();
    std::array<time_type, L_MAX + 1> ll;
    ll.fill(time_type(0));
    ll[2] = time_type(1);
    time_type hsum_v(0);
    for (int j = 1; j <= m_q - 2; ++j) {
      hsum_v += m_tau[j];
      time_type xi = hsum_v / m_hscale;
      for (int i = j + 2; i >= 2; --i)
        ll[i] = ll[i] * xi + ll[i - 1];
    }
    for (int j = 2; j < m_q; ++j)
      vec_axpy_with_slab(m_zn[j].m_v, m_zn_block.slab(j),
                         -static_cast<double>(ll[j]),
                         m_zn[m_q].m_v, m_zn_block.slab(m_q));
  }

  // ====================================================================
  //  adamsIncreaseOrder / adamsDecreaseOrder  (cvAdjustAdams port)
  //
  //  CVODE uses a different Nordsieck history adjustment for Adams
  //  than for BDF.  On an order increase the new high-order slot
  //  zn[L] (old L = new q) is simply set to zero — Adams does not
  //  consume a saved derivative estimate the way BDF does through
  //  zn[qmax].  On an order decrease each zn[j] is adjusted by a
  //  multiple of zn[q] derived from the polynomial
  //      q * ∫_0^x u*(u+xi_1)*...*(u+xi_{q-2}) du
  //  (see cvAdjustAdams in cvode.c).  Using the BDF routine instead
  //  corrupts the history array and causes the corrector to compute
  //  huge errors immediately after any order change.
  // ====================================================================

  void adamsIncreaseOrder()
  {
    // Called BEFORE m_q / m_L are updated, so m_L == old q + 1 == new q.
    // cvAdjustAdams: N_VConst(ZERO, zn[L]).
    const size_t n = m_zn[0].m_v.size();
    for (size_t i = 0; i < n; ++i)
      m_zn[m_L].m_v[i] = value_type(0);
  }

  void adamsDecreaseOrder()
  {
    // Literal port of cvAdjustAdams (deltaq = -1).  Uses m_l[] as
    // scratch for the polynomial coefficients — this is safe because
    // set_order_for_next_step runs between complete_step and the next
    // do_step, and do_step rebuilds m_l via adams_set_coefficients().
    if (m_q == 2) return;   // matches cvAdjustOrder q==2 early-out

    const int qmax_local = max_order;
    for (int i = 0; i <= qmax_local; ++i) m_l[i] = time_type(0);
    m_l[1] = time_type(1);

    time_type hsum_v(0);
    for (int j = 1; j <= m_q - 2; ++j) {
      hsum_v += m_tau[j];
      time_type xi = hsum_v / m_hscale;
      for (int i = j + 1; i >= 1; --i)
        m_l[i] = m_l[i] * xi + m_l[i - 1];
    }

    for (int j = 1; j <= m_q - 2; ++j)
      m_l[j + 1] = time_type(m_q) * (m_l[j] / time_type(j + 1));

    for (int j = 2; j < m_q; ++j)
      vec_axpy_with_slab(m_zn[j].m_v, m_zn_block.slab(j),
                         -static_cast<double>(m_l[j]),
                         m_zn[m_q].m_v, m_zn_block.slab(m_q));
  }

  // ====================================================================
  //  Resize
  // ====================================================================

  template<class StateIn>
  void resize_impl(const StateIn& x)
  {
    bool resized = false;
    for (int j = 0; j <= max_order + 1; ++j)
      resized |= adjust_size_by_resizeability(m_zn[j], x);
    for (int j = 0; j <= max_order; ++j)
      resized |= adjust_size_by_resizeability(m_zn_dense[j], x);
    resized |= adjust_size_by_resizeability(m_acor, x);
    resized |= adjust_size_by_resizeability(m_y, x);
    resized |= adjust_size_by_resizeability(m_tempv, x);
    resized |= adjust_size_by_resizeability(m_ftemp, x);
    resized |= adjust_size_by_resizeability(m_dense_x_scratch, x);
    if (m_ewt.size() != x.size()) m_ewt.resize(x.size());
    resized |= m_lu.resize(x);
    if (resized) m_lu.invalidate();
    // After any resize, re-prime the sensitivity slabs so each slab-bound
    // dual.tan_ points into the (possibly reallocated) vector's element row.
    // No-op when value_type isn't a dynamic dual or n_sens hasn't been set.
    if (resized && m_n_sens != 0)
      prepare_sensitivities(m_n_sens);
  }

  // ====================================================================
  //  Members
  // ====================================================================

  lu_type m_lu;

  std::array<wrapped_state_type, max_order + 2> m_zn;
  std::array<wrapped_state_type, max_order + 1> m_zn_dense;
  int m_q_dense = 1;
  time_type m_h_dense = time_type(0);
  time_type m_tn_dense = time_type(0);
  bool m_dense_valid = false;

  wrapped_state_type m_acor, m_y, m_tempv;
  wrapped_deriv_type m_ftemp;
  // Scratch used by eval_dense_into() for slab-aware Horner evaluation. The
  // user's StateOut x is arena-backed (per-element tangent buffers), so we
  // run the Horner on the slab-bound scratch first and copy out at the end.
  wrapped_state_type m_dense_x_scratch;
  std::vector<double> m_ewt;  // CVODE-style error weights from pre-prediction zn[0]

  // SoA tangent storage for the dynamic-dual heap path (empty stubs
  // otherwise). The Nordsieck history (m_zn) and dense-output snapshots
  // (m_zn_dense) each live in one contiguous [(K) × n_states × n_sens]
  // block via nordsieck_block, so BLAS-3 (dtrmm for Pascal shifts, dger
  // for the rank-1 complete_step update) can operate across all K slots
  // in one call. Per-slot ops (vec_axpy_with_slab, vec_copy_with_slab,
  // …) keep working unchanged via the slab views exposed by .slab(j).
  // Sized once per solve via prepare_sensitivities(); never grown.
  detail::nordsieck_block<value_type, max_order + 2> m_zn_block;
  detail::nordsieck_block<value_type, max_order + 1> m_zn_dense_block;
  detail::tangent_slab<value_type> m_acor_slab, m_y_slab, m_tempv_slab, m_ftemp_slab;
  detail::tangent_slab<value_type> m_dense_x_scratch_slab;
  unsigned m_n_sens = 0;

  std::array<time_type, L_MAX + 1> m_l;
  std::array<double, 6> m_tq;
  std::array<time_type, max_order + 1> m_tau;

  int m_q, m_L, m_qwait, m_nst, m_nscon;
  time_type m_h, m_hscale, m_eta;
  int m_max_newton_iter;
  bool m_newton_converged;
  double m_crate, m_acnrm;
  time_type m_gamma, m_gammap;
  double m_gamrat;
  double m_l1_at_setup = 1.0;  // l[1] at last LU setup, for diagnostic decomposition
  int m_nstlp;
  double m_etamax, m_saved_tq5;
  double m_atol = 1e-6, m_rtol = 1e-6;
  time_type m_tn_current;
  bool m_initialized;
  int m_n_fevals, m_n_jevals;

  // Setup trigger diagnostics
  int m_n_setup_total = 0, m_n_setup_force = 0, m_n_setup_nojac = 0;
  int m_n_setup_init = 0, m_n_setup_msbp = 0, m_n_setup_dgmax = 0;
  int m_n_setup_retry = 0;

  // Order change diagnostics
  int m_n_order_up = 0, m_n_order_down = 0, m_n_order_same = 0;
  // Per-order step counts: m_steps_at_order[q] = # steps completed at order q
  std::array<int, max_order + 1> m_steps_at_order = {};

  bool m_use_ndf_kappa = true;   // NDF kappa coefficients (runtime, default: NDF)

#ifdef CPPODE_STEP_TRACE
  // Scratch state populated by step_bdf_family / step_adams and read by
  // trace_step() at the end of the respective step body.
  bool        m_trace_setup_fired   = false;
  const char* m_trace_setup_reason  = "";
  double      m_trace_gamrat_pre    = 1.0;
  int         m_trace_pece_iters    = 0;
  bool        m_trace_pece_diverged = false;
#endif

public:
  mutable cppode::profiler m_prof;
};

// ============================================================================
//  User-facing type aliases
//
//  Templated aliases over <Value, JacobianPattern, Resizer> that fix
//  the multistep_method.  These are the names user code and codegen
//  output should reference.
// ============================================================================

template<class Value, class JacobianPattern = cppode::dense_lu_tag, class Resizer = initially_resizer>
using bdf_stepper_t   = multistepper<multistep_method::bdf,   Value, JacobianPattern, Resizer>;

template<class Value, class JacobianPattern = cppode::dense_lu_tag, class Resizer = initially_resizer>
using adams_stepper_t = multistepper<multistep_method::adams, Value, JacobianPattern, Resizer>;

} // namespace cppode

#endif // CPPODE_MULTISTEPPER_HPP_INCLUDED
