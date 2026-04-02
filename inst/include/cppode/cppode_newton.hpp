/*
 CppODE Newton Solver — CVODE-style Newton iteration
 =====================================================

 Free function template that performs Newton iteration for BDF steps.
 Takes the LU solver by reference — works with any lu_W<Value, is_sparse>.

 AD optimization — adaptive scalar-solve / IFT split:

 For AD types (F<double,N>, F<F<double,N>,M>, ...):

 Iteration m=0: FULL IFT solve (identical to non-optimized path).
 If Newton converges on the first try, there is ZERO overhead
 compared to the original code.  This is the common case for
 well-behaved problems.

 Iterations m>=1: SCALAR-ONLY solve (no IFT, no batched deriv solves).
 Only the value components are iterated.  Derivative components
 in acor stay at the values from m=0's IFT solve.

 After convergence at m>=1: single IFT sensitivity correction.
 One full AD f-eval + full IFT solve to correct the derivative
 components of acor for the additional value corrections from
 iterations 1..m.

 Cost model:
 m=0 convergence: identical to original (most common case)
 m=1 convergence: 1 IFT + 1 scalar + 1 IFT = 2 IFT + 1 scalar
 (vs original: 2 IFT — slightly MORE expensive)
 m=2 convergence: 1 IFT + 2 scalar + 1 IFT = 2 IFT + 2 scalar
 (vs original: 3 IFT — saves 1 IFT solve)
 m=3 convergence: 1 IFT + 3 scalar + 1 IFT = 2 IFT + 3 scalar
 (vs original: 4 IFT — saves 2 IFT solves)

 Break-even at m=2, net savings for m>=2.
 No penalty for the common m=0 case.

 For non-AD types (double):
 Falls through to the standard Newton — no overhead, no change.

 Separated from the BDF stepper for testability.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_NEWTON_HPP
#define CPPODE_NEWTON_HPP

#include <cstddef>
#include <cmath>
#include <vector>

#include <cppode/cppode_ad_lu.hpp>   // for ad_lu::is_ad
#include <cppode/cppode_profiler.hpp>

namespace cppode {

// ============================================================================
//  Scalar value extraction (duplicated here for self-containment)
// ============================================================================

namespace newton_detail {

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v) { return static_cast<double>(v); }

template<class T, unsigned int N>
inline double scalar_value(const fadbad::F<T,N>& v) {
  return scalar_value(const_cast<fadbad::F<T,N>&>(v).x());
}

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
//  1. wrms_norm_scalar — double vectors (for scalar Newton convergence)
//
//  2. wrms_norm_correction — SCALAR ONLY from AD vectors
//     Measures only the value components.  Used for Newton convergence.
//
//  3. wrms_norm — AD-AWARE (for error control / order selection)
//     Includes all derivative components.
// ============================================================================

namespace newton_detail {

/// WRMS norm on plain double vectors.  Used in the scalar Newton phase.
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

/// WRMS norm of correction b, weighted against iterate y — SCALAR ONLY.
/// Used for Newton convergence test (del, crate, dcon).
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

/// WRMS norm of vector v, weighted against reference y0 — AD-AWARE.
/// Includes derivative components for AD types, equivalent to the
/// augmented sensitivity system.  Used for error control (acnrm,
/// dsm, order selection eta computations).
/// For double, the derivative loop compiles to nothing.
template<class T>
double wrms_norm(const std::vector<T>& v,
                 const std::vector<T>& y0,
                 size_t n, double atol, double rtol)
{
  double sumsq = 0.0;
  size_t N_eff = 0;

  for (size_t i = 0; i < n; ++i) {
    double vi = std::abs(scalar_value(v[i]));
    double yi = std::abs(scalar_value(y0[i]));
    double r = vi / (atol + rtol * yi);
    sumsq += r * r;
    ++N_eff;

    if constexpr (ad_lu::is_ad<T>::value) {
      auto& v_ad = const_cast<T&>(v[i]);
      auto& y_ad = const_cast<T&>(y0[i]);
      unsigned nd = v_ad.size();
      for (unsigned j = 0; j < nd; ++j) {
        double vd = std::abs(scalar_value(v_ad.d(j)));
        double yd = std::abs(scalar_value(y_ad.d(j)));
        double rd = vd / (atol + rtol * yd);
        sumsq += rd * rd;
        ++N_eff;
      }
    }
  }
  return (N_eff > 0) ? std::sqrt(sumsq / N_eff) : 0.0;
}

} // namespace newton_detail

// ============================================================================
//  bdf_newton_solve
//
//  Performs CVODE-style Newton iteration for one BDF step.
//
//  For AD types: adaptive scalar-solve / IFT split.
//  For double:   standard Newton (no AD overhead).
// ============================================================================

template<class LU, class DerivFunc, class Value, class TimeType>
newton_result bdf_newton_solve(
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
    cppode::profiler& prof)
{
  using newton_detail::scalar_value;
  using newton_detail::wrms_norm_correction;
  using newton_detail::wrms_norm;

  const size_t n = zn0.size();

  // ========================================================================
  //  AD path: adaptive scalar-solve / IFT split
  //
  //  m=0: full IFT solve (identical to original — zero overhead)
  //  m>=1: scalar-only solves + single IFT correction at convergence
  // ========================================================================
  if constexpr (ad_lu::is_ad<Value>::value) {

    // Thread-local scalar work vector for solve_scalar (m>=1 only)
    static thread_local std::vector<double> s_tempv_val;

    int n_fevals = 0;

    // Initialize (full AD)
    { auto _t = prof.timer(prof_cat::newton_overhead);
      for (size_t i = 0; i < n; ++i) {
        acor[i] = Value(0);
        y[i] = zn0[i];
      }
    }

    // Initial f-eval at predicted point (full AD)
    { auto _t = prof.timer(prof_cat::f_eval);
      deriv_func(y, ftemp, t_new); }
    ++n_fevals;

    double del = 0.0, delp = 0.0;

    for (int m = 0; m < max_iter; ++m) {

      if (m == 0) {
        // ==============================================================
        //  First iteration (m=0): FULL IFT solve
        //
        //  Identical to the original code path.  If Newton converges
        //  here, there is zero overhead — acor gets correct values
        //  AND derivatives in one shot.
        // ==============================================================

        // Full AD residual
        { auto _t = prof.timer(prof_cat::newton_overhead);
          for (size_t i = 0; i < n; ++i) {
            tempv[i] = ftemp[i] - (rl1 * zn1[i] + acor[i]) / gamma;
          }
        }

        // Full IFT solve (value + all derivative directions)
        { auto _t = prof.timer(prof_cat::lu_solve);
          lu.solve(tempv); }

        // Linear solution scaling
        if (std::abs(gamrat - 1.0) > 1e-14) {
          double scale = 2.0 / (1.0 + gamrat);
          for (size_t i = 0; i < n; ++i)
            tempv[i] *= scale;
        }

        // WRMS norm of correction — scalar only for convergence
        { auto _t = prof.timer(prof_cat::error_norm);
          del = wrms_norm_correction(tempv, y, n, atol, rtol); }

        // Update (full AD — acor gets both values and derivatives)
          { auto _t = prof.timer(prof_cat::newton_overhead);
            for (size_t i = 0; i < n; ++i) {
              acor[i] += tempv[i];
              y[i] = zn0[i] + acor[i];
            }
          }

        // Convergence test
        double dcon = del * std::min(1.0, crate) / tq4;

        if (dcon <= 1.0) {
          // Converged at m=0 — acor already has correct derivs from IFT.
          // This is the fast path: identical to original code.
          double acnrm;
          { auto _t = prof.timer(prof_cat::error_norm);
            acnrm = wrms_norm(acor, zn0, n, atol, rtol); }
          return { true, acnrm, n_fevals };
        }

        // Not converged — continue to m=1 with scalar-only solves.
        // acor now has derivatives from the m=0 IFT solve, but they
        // will be stale after further scalar-only value corrections.
        // We'll fix them in Phase 2 if/when we converge.

        delp = del;

        // Re-evaluate f at updated y (full AD)
        { auto _t = prof.timer(prof_cat::f_eval);
          deriv_func(y, ftemp, t_new); }
        ++n_fevals;

      } else {
        // ==============================================================
        //  Iterations m>=1: SCALAR-ONLY solve
        //
        //  Save expensive IFT solves.  Only the value components are
        //  iterated.  Derivative components in acor are stale (from m=0).
        //  They will be corrected in Phase 2 after convergence.
        // ==============================================================

        // Lazy-init scalar work vector
        s_tempv_val.resize(n);

        // Extract scalar residual
        { auto _t = prof.timer(prof_cat::newton_overhead);
          for (size_t i = 0; i < n; ++i) {
            s_tempv_val[i] = scalar_value(ftemp[i])
            - (static_cast<double>(rl1) * scalar_value(zn1[i])
                 + scalar_value(acor[i])) / static_cast<double>(gamma);
          }
        }

        // Scalar solve (no IFT, no batched derivative solves)
        { auto _t = prof.timer(prof_cat::lu_solve);
          lu.solve_scalar(s_tempv_val); }

        // Linear solution scaling
        if (std::abs(gamrat - 1.0) > 1e-14) {
          double scale = 2.0 / (1.0 + gamrat);
          for (size_t i = 0; i < n; ++i)
            s_tempv_val[i] *= scale;
        }

        // WRMS norm of scalar correction
        { auto _t = prof.timer(prof_cat::error_norm);
          static thread_local std::vector<double> s_y_val;
          s_y_val.resize(n);
          for (size_t i = 0; i < n; ++i)
            s_y_val[i] = scalar_value(y[i]);
          del = newton_detail::wrms_norm_scalar(
            s_tempv_val, s_y_val, n, atol, rtol);
        }

        // Update value components only
        { auto _t = prof.timer(prof_cat::newton_overhead);
          for (size_t i = 0; i < n; ++i) {
            acor[i].x() += s_tempv_val[i];
            y[i] = zn0[i] + acor[i];
          }
        }

        // Convergence test
        crate = std::max(0.3 * crate, (delp > 0.0) ? del / delp : del);
        double dcon = del * std::min(1.0, crate) / tq4;

        if (dcon <= 1.0) {
          // ----------------------------------------------------------------
          //  Phase 2: Single IFT sensitivity correction
          //
          //  Value iteration converged at m>=1.  The derivative components
          //  in acor are stale (from the m=0 IFT solve — they don't
          //  account for the scalar corrections from iterations 1..m).
          //
          //  Fix: one full AD f-eval + IFT solve at the converged point
          //  to recompute the correct derivative components.
          //
          //  y.d(j) = zn0.d(j) + acor.d(j)  where acor.d(j) is from m=0.
          //  The f-eval sees these (partially correct) derivatives.
          //  The IFT solve gives the incremental derivative correction.
          // ----------------------------------------------------------------

          // Full AD f-eval at converged y*
          { auto _t = prof.timer(prof_cat::f_eval);
            deriv_func(y, ftemp, t_new); }
          ++n_fevals;

          // Full AD residual
          { auto _t = prof.timer(prof_cat::newton_overhead);
            for (size_t i = 0; i < n; ++i) {
              tempv[i] = ftemp[i] - (rl1 * zn1[i] + acor[i]) / gamma;
            }
          }

          // Full IFT solve
          { auto _t = prof.timer(prof_cat::lu_solve);
            lu.solve(tempv); }

          // Gamrat scaling
          if (std::abs(gamrat - 1.0) > 1e-14) {
            double scale = 2.0 / (1.0 + gamrat);
            for (size_t i = 0; i < n; ++i)
              tempv[i] *= scale;
          }

          // Update acor and y with the full correction
          { auto _t = prof.timer(prof_cat::newton_overhead);
            for (size_t i = 0; i < n; ++i) {
              acor[i] += tempv[i];
              y[i] = zn0[i] + acor[i];
            }
          }

          // AD-aware acnrm for error control
          double acnrm;
          { auto _t = prof.timer(prof_cat::error_norm);
            acnrm = wrms_norm(acor, zn0, n, atol, rtol); }

          return { true, acnrm, n_fevals };
        }

        if (m >= 2 && del > 2.0 * delp) {
          return { false, 0.0, n_fevals };
        }

        delp = del;

        // Re-evaluate f at updated y (full AD)
        { auto _t = prof.timer(prof_cat::f_eval);
          deriv_func(y, ftemp, t_new); }
        ++n_fevals;
      }
    }

    return { false, 0.0, n_fevals };

  } else {
    // ========================================================================
    //  Non-AD path: standard Newton iteration (unchanged)
    // ========================================================================

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
      { auto _t = prof.timer(prof_cat::lu_solve);
        lu.solve(tempv); }

      // CVODE linear solution scaling
      if (std::abs(gamrat - 1.0) > 1e-14) {
        double scale = 2.0 / (1.0 + gamrat);
        for (size_t i = 0; i < n; ++i)
          tempv[i] *= scale;
      }

      // WRMS norm of correction
      { auto _t = prof.timer(prof_cat::error_norm);
        del = wrms_norm_correction(tempv, y, n, atol, rtol); }

      // Update
        { auto _t = prof.timer(prof_cat::newton_overhead);
          for (size_t i = 0; i < n; ++i) {
            acor[i] += tempv[i];
            y[i] = zn0[i] + acor[i];
          }
        }

      // Convergence test (CVODE-style)
      if (m > 0) {
        crate = std::max(0.3 * crate, (delp > 0.0) ? del / delp : del);
      }
      double dcon = del * std::min(1.0, crate) / tq4;

      if (dcon <= 1.0) {
        double acnrm;
        { auto _t = prof.timer(prof_cat::error_norm);
          acnrm = wrms_norm(acor, zn0, n, atol, rtol); }
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
}

} // namespace cppode

#endif // CPPODE_NEWTON_HPP
