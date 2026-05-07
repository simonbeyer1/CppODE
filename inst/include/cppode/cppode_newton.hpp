/*
 CppODE Newton Solver: simplified Newton iteration for NDF/BDF
 ================================================================

 Free function template that performs one chord (simplified Newton)
 solve for a single NDF/BDF step.  Takes the LU solver by reference --
 works with any lu_W<Value, is_sparse>.

 Unified code path for both AD and non-AD types:

 For AD types (F<double,N>, F<F<double,N>,M>, ...):
 lu.solve(tempv) performs full IFT at every iteration, giving correct
 value AND derivative corrections each time.  Newton convergence is
 checked AD-aware (derivative components included) so that sensitivity
 blow-ups trigger fresh Jacobian evaluations.

 For non-AD types (double):
 lu.solve() is a standard forward/back-substitution.
 wrms_norm_correction == wrms_norm (no derivative components).

 Separated from the NDF stepper for testability.

 Copyright (C) 2026 Simon Beyer

 Portions of this file (chord-Newton convergence test, Nordsieck
 correction update, sensitivity-aware staggered WRMS norm matching
 CVODES CV_STAGGERED) are derived from SUNDIALS/CVODE(S),
 Copyright (c) 2002-2024 Lawrence Livermore National Security and
 Southern Methodist University, distributed under the BSD-3-Clause
 license.  See inst/COPYRIGHTS for the full license text.
 */

#ifndef CPPODE_NEWTON_HPP
#define CPPODE_NEWTON_HPP

#include <cstddef>
#include <cmath>
#include <vector>

#include <cppode/cppode_ad_lu.hpp>   // for ad_lu::is_ad
#include <cppode/cppode_ad_traits.hpp>
#include <cppode/cppode_profiler.hpp>

namespace cppode {

// ============================================================================
//  Scalar value extraction: pulled in from cppode_ad_traits.hpp
// ============================================================================

namespace newton_detail {
using cppode::ad_traits::scalar_value;
} // namespace newton_detail

// ============================================================================
//  Newton result
// ============================================================================

struct newton_result {
  bool   converged;   // Did Newton converge?
  double acnrm;       // WRMS norm of accumulated correction (for error test)
  int    n_fevals;    // Number of f-evaluations used
};

// ============================================================================
//  WRMS norm helpers
//
//  Three variants:
//
//  1. wrms_norm_scalar: double vectors (for scalar Newton convergence)
//
//  2. wrms_norm_correction: SCALAR ONLY from AD vectors
//     Measures only the value components.  Used for Newton convergence.
//
//  3. wrms_norm: AD-AWARE (for error control / order selection)
//     Includes all derivative components.
// ============================================================================

namespace newton_detail {

// WRMS norm on plain double vectors.  Used in the scalar Newton phase.
inline double wrms_norm_scalar(const std::vector<double>& b,
                               const std::vector<double>& y,
                               size_t n, double atol, double rtol)
{
  double sumsq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double bi = std::abs(b[i]);
    double yi = std::abs(y[i]);
    double r = bi / (atol + rtol * yi);
    sumsq += r * r;
  }
  return (n > 0) ? std::sqrt(sumsq / n) : 0.0;
}

// WRMS norm of correction b, weighted against iterate y: SCALAR ONLY.
// Used for Newton convergence test (del, crate, dcon).
template<class T>
double wrms_norm_correction(const std::vector<T>& b,
                            const std::vector<T>& y,
                            size_t n, double atol, double rtol)
{
  double sumsq = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double bi = std::abs(scalar_value(b[i]));
    double yi = std::abs(scalar_value(y[i]));
    double r = bi / (atol + rtol * yi);
    sumsq += r * r;
  }
  return (n > 0) ? std::sqrt(sumsq / n) : 0.0;
}

// WRMS norm of vector v, weighted against reference y0: AD-AWARE.
// Returns max(state_wrms, max_j sens_wrms[j]) over the state and
// each sensitivity-parameter slice.  This matches CVODES's
// `cvSensUpdateNorm` (CV_STAGGERED) convention: every sensitivity
// vector is held individually below tolerance, rather than letting
// large per-parameter errors hide inside an averaged WRMS.
// For non-AD types the sens loop compiles to nothing → returns the
// pure state WRMS.
template<class T>
double wrms_norm(const std::vector<T>& v,
                 const std::vector<T>& y0,
                 size_t n, double atol, double rtol)
{
  if (n == 0) return 0.0;

  double state_sumsq = 0.0;
  unsigned nd = 0;
  if constexpr (ad_lu::is_ad<T>::value) {
    nd = const_cast<T&>(v[0]).size();
  }
  std::vector<double> sens_sumsq(nd, 0.0);

  for (size_t i = 0; i < n; ++i) {
    double vi = std::abs(scalar_value(v[i]));
    double yi = std::abs(scalar_value(y0[i]));
    double r = vi / (atol + rtol * yi);
    state_sumsq += r * r;

    if constexpr (ad_lu::is_ad<T>::value) {
      auto& v_ad = const_cast<T&>(v[i]);
      auto& y_ad = const_cast<T&>(y0[i]);
      for (unsigned j = 0; j < nd; ++j) {
        double vd = std::abs(scalar_value(v_ad.d(j)));
        double yd = std::abs(scalar_value(y_ad.d(j)));
        double rd = vd / (atol + rtol * yd);
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

} // namespace newton_detail

// ============================================================================
//  ndf_newton_solve
//
//  Performs one Newton iteration for one NDF/BDF step.
//
//  For AD types: adaptive scalar-solve / IFT split.
//  For double:   standard Newton (no AD overhead).
// ============================================================================

template<class LU, class DerivFunc, class Value, class TimeType>
newton_result ndf_newton_solve(
    LU& lu,
    DerivFunc& deriv_func,
    const std::vector<Value>& zn0,
    const std::vector<Value>& zn1,
    TimeType rl1,
    TimeType gamma,
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
    double gamrat,
    cppode::profiler& prof,
    const std::vector<double>& ewt = {})
{
  using newton_detail::wrms_norm_correction;
  using newton_detail::wrms_norm;
  using newton_detail::scalar_value;

  const bool use_ewt = !ewt.empty();
  const size_t n = zn0.size();
  int n_fevals = 0;

  // Initialize
  { auto _t = prof.timer(prof_cat::newton_overhead);
    for (size_t i = 0; i < n; ++i) {
      acor[i] = Value(0);
      y[i] = zn0[i];
    }
  }

  // Initial f-eval at predicted point
  { auto _t = prof.timer(prof_cat::f_eval);
    deriv_func(y, ftemp, t_new); }
  ++n_fevals;

  double del = 0.0, delp = 0.0;

  for (int m = 0; m < max_iter; ++m) {
    // Residual: tempv = f(y) - (rl1*zn1 + acor) / gamma
    { auto _t = prof.timer(prof_cat::newton_overhead);
      for (size_t i = 0; i < n; ++i) {
        tempv[i] = ftemp[i] - (rl1 * zn1[i] + acor[i]) / gamma;
      }
    }

    // Solve W * delta = tempv
    // For AD types, lu.solve() performs full IFT at every iteration,
    // giving correct value AND derivative corrections each time.
    { auto _t = prof.timer(prof_cat::lu_solve);
      lu.solve(tempv); }

    // Linear solution scaling for gamma drift
    if (std::abs(gamrat - 1.0) > 1e-14) {
      double scale = 2.0 / (1.0 + gamrat);
      for (size_t i = 0; i < n; ++i)
        tempv[i] *= scale;
    }

    // WRMS norm of correction: AD-aware, max over (state, sens[j]).
    // ewt is interleaved: [val_0, d0_0, d1_0, ..., val_1, d0_1, ...].
    // Each sensitivity vector contributes its own per-vector WRMS;
    // the controller sees the worst: matching CVODES cvSensUpdateNorm.
    { auto _t = prof.timer(prof_cat::error_norm);
      if (use_ewt) {
        double state_sumsq = 0.0;
        unsigned nd = 0;
        if constexpr (ad_lu::is_ad<Value>::value) {
          if (n > 0) nd = const_cast<Value&>(tempv[0]).size();
        }
        std::vector<double> sens_sumsq(nd, 0.0);
        size_t ew = 0;
        for (size_t i = 0; i < n; ++i) {
          double vi = std::abs(scalar_value(tempv[i]));
          double r = vi * ewt[ew++];
          state_sumsq += r * r;
          if constexpr (ad_lu::is_ad<Value>::value) {
            auto& v_ad = const_cast<Value&>(tempv[i]);
            for (unsigned j = 0; j < nd; ++j) {
              double vd = std::abs(scalar_value(v_ad.d(j)));
              double rd = vd * ewt[ew++];
              sens_sumsq[j] += rd * rd;
            }
          }
        }
        double max_norm = (n > 0) ? std::sqrt(state_sumsq / n) : 0.0;
        for (unsigned j = 0; j < nd; ++j) {
          double sens_norm = std::sqrt(sens_sumsq[j] / n);
          if (sens_norm > max_norm) max_norm = sens_norm;
        }
        del = max_norm;
      } else {
        del = wrms_norm(tempv, y, n, atol, rtol);
      }
    }

    // Update
      { auto _t = prof.timer(prof_cat::newton_overhead);
        for (size_t i = 0; i < n; ++i) {
          acor[i] += tempv[i];
          y[i] = zn0[i] + acor[i];
        }
      }

    // Convergence test
    if (m > 0) {
      crate = std::max(0.3 * crate, (delp > 0.0) ? del / delp : del);
    }
    double dcon = del * std::min(1.0, crate) / tq4;

    if (dcon <= 1.0) {
      // AD-aware acnrm for error control: max over (state_wrms,
      // each sens-vector_wrms).  The step-size controller therefore
      // limits the worst-case local truncation error across the
      // augmented system, including each sensitivity parameter
      // individually: matching CVODES `cvSensUpdateNorm`
      // (CV_STAGGERED).  A flat WRMS over all components averages
      // per-parameter errors and can hide one bad sensitivity behind
      // many small ones; the max-norm prevents that.
      double acnrm;
      { auto _t = prof.timer(prof_cat::error_norm);
        if (use_ewt) {
          double state_sumsq = 0.0;
          unsigned nd = 0;
          if constexpr (ad_lu::is_ad<Value>::value) {
            if (n > 0) nd = const_cast<Value&>(acor[0]).size();
          }
          std::vector<double> sens_sumsq(nd, 0.0);
          size_t ew = 0;
          for (size_t i = 0; i < n; ++i) {
            double vi = std::abs(scalar_value(acor[i]));
            double r = vi * ewt[ew++];
            state_sumsq += r * r;
            if constexpr (ad_lu::is_ad<Value>::value) {
              auto& a_ad = const_cast<Value&>(acor[i]);
              for (unsigned j = 0; j < nd; ++j) {
                double vd = std::abs(scalar_value(a_ad.d(j)));
                double rd = vd * ewt[ew++];
                sens_sumsq[j] += rd * rd;
              }
            }
          }
          double max_norm = (n > 0) ? std::sqrt(state_sumsq / n) : 0.0;
          for (unsigned j = 0; j < nd; ++j) {
            double sens_norm = std::sqrt(sens_sumsq[j] / n);
            if (sens_norm > max_norm) max_norm = sens_norm;
          }
          acnrm = max_norm;
        } else {
          acnrm = wrms_norm(acor, zn0, n, atol, rtol);
        }
      }
      return { true, acnrm, n_fevals };
    }

    if (m >= 2 && del > 2.0 * delp) {
      return { false, 0.0, n_fevals };
    }

    delp = del;

    // Re-evaluate f at updated y
    { auto _t = prof.timer(prof_cat::f_eval);
      deriv_func(y, ftemp, t_new); }
    ++n_fevals;
  }

  return { false, 0.0, n_fevals };
}

} // namespace cppode

#endif // CPPODE_NEWTON_HPP
