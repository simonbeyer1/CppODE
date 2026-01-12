#ifndef CPPODE_ODEINT_UTILS_HPP
#define CPPODE_ODEINT_UTILS_HPP

/**
 * @file cppode_odeint_utils.hpp
 * @brief Utility functions for ODE integration and sensitivity calculation with automatic differentiation
 *
 * This header provides:
 *   - Safe scalar extraction from nested FADBAD types (F<F<T>>)
 *   - Recursive maximum absolute value extraction (including derivatives)
 *   - Weighted norms for adaptive step-size control
 *   - Initial step-size estimation for stiff solvers (Hairer-Nørsett-Wanner heuristic)
 *   - uBLAS identity_matrix multiplication operators for F<T>
 *
 * @note Requires cppode_fadiff_extensions.hpp to be included first.
 *
 * @author Simon Beyer <simon.beyer@fdm.uni-freiburg.de>
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <fadbad++/fadiff.h>

// ============================================================================
//  Boost.uBLAS extensions for fadbad::F<T> types
// ============================================================================

namespace boost { namespace numeric { namespace ublas {

/**
 * @brief Enables multiplication between a fadbad::F<T> scalar and an identity matrix.
 *
 * This operator allows expressions of the form:
 * @code
 * fadbad::F<double> a = 2.0;
 * identity_matrix<fadbad::F<double>> I(3);
 * matrix<fadbad::F<double>> M = a * I;
 * @endcode
 *
 * @tparam T  Inner type of fadbad::F<T> (usually `double` or another `F<>`).
 * @param a   Scalar to multiply.
 * @param I   Identity matrix.
 * @return Matrix with all diagonal elements set to `a` and zeros elsewhere.
 */
template<class T>
inline matrix< fadbad::F<T> >
operator*(const fadbad::F<T>& a,
          const identity_matrix< fadbad::F<T> >& I)
{
  matrix< fadbad::F<T> > M(I.size1(), I.size2(), fadbad::F<T>(T(0)));
  for (std::size_t i = 0; i < I.size1(); ++i)
    M(i, i) = a;
  return M;
}

/**
 * @brief Commutative overload: identity matrix multiplied by fadbad::F<T> scalar.
 * @tparam T  Inner fadbad scalar type.
 * @param I   Identity matrix.
 * @param a   Scalar to multiply.
 * @return Same as `a * I`.
 */
template<class T>
inline matrix< fadbad::F<T> >
operator*(const identity_matrix< fadbad::F<T> >& I,
          const fadbad::F<T>& a)
{
  return a * I;
}

}}} // namespace boost::numeric::ublas

// ============================================================================
//  odeint_utils namespace — helper functions for AD and initial step-size selection
// ============================================================================

namespace odeint_utils {

using boost::numeric::ublas::vector;
using boost::numeric::ublas::matrix;
using fadbad::F;

// ============================================================================
//  Scalar extraction — full recursive unwrapping of nested F<F<...>>
// ============================================================================

/**
 * @brief Base case: extract scalar from double.
 */
inline double scalar_value(double v) { return v; }

/**
 * @brief Recursive case: extract innermost scalar from F<T>.
 * @tparam T Inner type (may be another F<> or double).
 */
template<typename T>
inline double scalar_value(const F<T>& v)
{
  // FADBAD's .x() is non-const → remove constness
  return scalar_value(const_cast<F<T>&>(v).x());
}

// ============================================================================
//  Maximum absolute value including all derivative components
// ============================================================================

/**
 * @brief Base case: max abs for double is just abs.
 */
inline double max_abs_with_derivatives(double v)
{
  return std::abs(v);
}

/**
 * @brief Recursive case: max abs over value and all derivative components.
 *
 * For F<T>, this computes max(|val|, max over all |d[i]|) recursively.
 *
 * @tparam T Inner type.
 */
template<typename T>
inline double max_abs_with_derivatives(const F<T>& v)
{
  auto& v_mut = const_cast<F<T>&>(v);

  double maxv = std::abs(scalar_value(v_mut.x()));
  for (unsigned int i = 0; i < v_mut.size(); ++i)
    maxv = std::max(maxv, max_abs_with_derivatives(v_mut.d(i)));

  return maxv;
}

// ============================================================================
//  Weighted infinity norms
// ============================================================================

/**
 * @brief Weighted sup-norm for double vectors.
 *
 * Computes max_i |v[i]| / (atol + rtol * |x0[i]|)
 *
 * @param v    Vector to compute norm of.
 * @param x0   Reference state for scaling.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return Weighted infinity norm.
 */
inline double weighted_sup_norm(
    vector<double>& v,
    vector<double>& x0,
    double atol,
    double rtol)
{
  double nrm = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    double scale = atol + rtol * std::abs(x0[i]);
    nrm = std::max(nrm, std::abs(v[i]) / scale);
  }
  return nrm;
}

/**
 * @brief Weighted sup-norm for F<T> vectors (AD-aware).
 *
 * Uses max_abs_with_derivatives to include derivative components.
 *
 * @tparam T Inner AD type.
 */
template<typename T>
inline double weighted_sup_norm(
    vector<F<T>>& v,
    vector<F<T>>& x0,
    double atol,
    double rtol)
{
  double nrm = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    double scale = atol + rtol * std::abs(scalar_value(x0[i]));
    double comp  = max_abs_with_derivatives(v[i]);
    nrm = std::max(nrm, comp / scale);
  }
  return nrm;
}

// ============================================================================
//  Initial step-size estimation (Hairer-Nørsett-Wanner heuristic)
// ============================================================================

/**
 * @brief Estimates the initial step size for an ODE solver (e.g., Rosenbrock-4).
 *
 * This implementation follows the heuristic approach by Hairer, Nørsett, and Wanner.
 * It uses the first derivative (velocity) and second derivative (curvature) to
 * find a dimensionally consistent time scale that respects the error tolerances.
 *
 * Ref: Hairer, Nørsett, Wanner - "Solving Ordinary Differential Equations I", p. 169.
 *
 * @param system   ODE right-hand side function.
 * @param jacobian Jacobian function (also returns df/dt).
 * @param x0       Initial state.
 * @param t0       Initial time.
 * @param atol     Absolute tolerance.
 * @param rtol     Relative tolerance.
 * @param eta      Safety factor (default 1e-2).
 * @return Estimated initial step size.
 */
inline double estimate_initial_dt_local(
    const std::function<void(const vector<double>&, vector<double>&, const double&)>& system,
    const std::function<void(const vector<double>&, matrix<double>&, const double&, vector<double>&)>& jacobian,
    vector<double>& x0,
    double t0,
    double atol,
    double rtol,
    double eta = 1e-2)
{
  const std::size_t n = x0.size();

  // First derivative
  vector<double> dxdt(n);
  system(x0, dxdt, t0);

  // Jacobian + explicit df/dt
  matrix<double> J(n, n);
  vector<double> dfdt(n);
  jacobian(x0, J, t0, dfdt);

  // Second derivative x'' = df/dt + J*dxdt
  vector<double> xdd(n);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = dfdt[i];
    for (std::size_t j = 0; j < n; ++j)
      sum += J(i,j) * dxdt[j];
    xdd[i] = sum;
  }

  // Norms
  double norm_x   = weighted_sup_norm(x0, x0, atol, rtol);
  double norm_dx  = weighted_sup_norm(dxdt, x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  // Heuristics
  // dt1: Based on velocity (Order 1) -> dt ~ eta * ||x|| / ||x'||
  // dt2: Based on curvature (Order 2) -> dt ~ sqrt(eta * ||x|| / ||x''||)
  double dt1 = std::numeric_limits<double>::infinity();
  double dt2 = std::numeric_limits<double>::infinity();

  if (norm_dx > 0.0) {
    dt1 = norm_x / norm_dx;
  }

  if (norm_xdd > 0.0) {
    dt2 = std::sqrt(norm_x / norm_xdd);
  }

  double dt = eta * std::min(dt1, dt2);

  if (!std::isfinite(dt) || dt <= 0.0)
    dt = atol;  // fallback

  return dt;
}

/**
 * @brief Convenience overload for arbitrary functor-style systems (double version).
 */
template<typename System, typename Jacobian>
inline double estimate_initial_dt_local(
    System system,
    Jacobian jacobian,
    vector<double>& x0,
    double t0,
    double atol,
    double rtol,
    double eta = 1e-2)
{
  std::function<void(const vector<double>&, vector<double>&, const double&)> sys_f =
    [&system](const vector<double>& x, vector<double>& dxdt, const double& t) {
      system(x, dxdt, t);
    };

    std::function<void(const vector<double>&, matrix<double>&, const double&, vector<double>&)> jac_f =
      [&jacobian](const vector<double>& x, matrix<double>& J, const double& t, vector<double>& dfdt) {
        jacobian(x, J, t, dfdt);
      };

      return estimate_initial_dt_local(sys_f, jac_f, x0, t0, atol, rtol, eta);
}

/**
 * @brief Templated version of estimate_initial_dt_local() for Dual number types F<T>.
 *
 * @tparam T Inner AD type.
 */
template<typename T>
inline F<T> estimate_initial_dt_local(
    const std::function<void(vector<F<T>>&, vector<F<T>>&, const F<T>&)>& system,
    const std::function<void(vector<F<T>>&, matrix<F<T>>&, const F<T>&, vector<F<T>>&)>& jacobian,
    vector<F<T>>& x0,
    const F<T> t0,
    double atol,
    double rtol,
    double eta = 1e-2)
{
  const std::size_t n = x0.size();

  // First derivative
  vector<F<T>> dxdt(n);
  system(x0, dxdt, t0);

  // Jacobian + explicit df/dt
  matrix<F<T>> J(n, n);
  vector<F<T>> dfdt(n);
  jacobian(x0, J, t0, dfdt);

  // Second derivative x'' = df/dt + J*dxdt
  vector<F<T>> xdd(n);
  for (std::size_t i = 0; i < n; ++i) {
    F<T> sum = dfdt[i];
    for (std::size_t j = 0; j < n; ++j)
      sum += J(i,j) * dxdt[j];
    xdd[i] = sum;
  }

  // Norms (AD-aware)
  double norm_x   = weighted_sup_norm(x0, x0, atol, rtol);
  double norm_dx  = weighted_sup_norm(dxdt, x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  // Heuristics
  double dt1 = std::numeric_limits<double>::infinity();
  double dt2 = std::numeric_limits<double>::infinity();

  if (norm_dx > 0.0) {
    dt1 = norm_x / norm_dx;
  }

  if (norm_xdd > 0.0) {
    dt2 = std::sqrt(norm_x / norm_xdd);
  }

  double dt = eta * std::min(dt1, dt2);

  if (!std::isfinite(dt) || dt <= 0.0)
    dt = atol;  // fallback

  return dt;
}

/**
 * @brief Convenience overload for arbitrary functor-style systems (AD version).
 */
template<typename System, typename Jacobian, typename T>
inline F<T> estimate_initial_dt_local(
    System system,
    Jacobian jacobian,
    vector<F<T>>& x0,
    const F<T> t0,
    double atol,
    double rtol,
    double eta = 5e-2)
{
  std::function<void(vector<F<T>>&, vector<F<T>>&, const F<T>&)> sys_f =
    [&system](vector<F<T>>& x, vector<F<T>>& dxdt, const F<T>& t) {
      system(x, dxdt, t);
    };

    std::function<void(vector<F<T>>&, matrix<F<T>>&, const F<T>&, vector<F<T>>&)> jac_f =
      [&jacobian](vector<F<T>>& x, matrix<F<T>>& J, const F<T>& t, vector<F<T>>& dfdt) {
        jacobian(x, J, t, dfdt);
      };

      return estimate_initial_dt_local<T>(sys_f, jac_f, x0, t0, atol, rtol, eta);
}

} // namespace odeint_utils

#endif // CPPODE_ODEINT_UTILS_HPP
