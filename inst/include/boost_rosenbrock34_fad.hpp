#ifndef BOOST_ROSENBROCK34_FAD_HPP
#define BOOST_ROSENBROCK34_FAD_HPP

#include <algorithm>
#include <vector>
#include <cmath>
#include <functional>
#include <fadbad++/fadiff.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller_fad.hpp>
#include <boost/numeric/odeint/integrate/detail/integrate_times_with_events.hpp>
#include <boost/numeric/odeint/integrate/step_checker.hpp>

/**
 * @file boost_rosenbrock34_fad.hpp
 * @brief Extended utilities enabling Rosenbrock-type ODE integration
 *        for FADBAD++ automatic differentiation types within Boost.Odeint.
 *
 * This header bridges the gap between Boost.Odeint and the FADBAD++
 * automatic differentiation library. It provides:
 *   - Safe scalar extraction from nested FADBAD types (F<F<T>>)
 *   - Recursive maximum absolute value extraction
 *   - Weighted norms for adaptive step-size control
 *   - Initial step-size estimation for stiff solvers
 *   - Operator and abs() overloads to prevent ambiguity in nested FADBAD usage
 *
 * @note All functions are defined as `inline` for header-only inclusion.
 *
 * @author Simon Beyer <simon.beyer@fdm.uni-freiburg.de>
 *
 */

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
//  odeint_utils namespace — helper functions for AD support
// ============================================================================
namespace odeint_utils {

using boost::numeric::ublas::vector;
using boost::numeric::ublas::matrix;
using fadbad::F;

/**
 * @brief Extracts a scalar double value from nested fadbad::F<T> expressions.
 *
 * This function is designed to handle arbitrarily nested fadbad types:
 * - `F<double>`
 * - `F<F<double>>`
 * - and deeper (recursively).
 *
 * The recursion continues by calling `.x()` until a raw arithmetic type is reached.
 *
 * @tparam T Inner fadbad or arithmetic type.
 * @param v  The fadbad object or arithmetic value.
 * @return The contained scalar as a `double`.
 */
inline double scalar_value(double v) { return v; }

template<typename T>
inline double scalar_value(const F<T>& v) {
  return scalar_value(const_cast<F<T>&>(v).x());
}

/**
 * @brief Computes the maximum absolute value across a fadbad variable and all its derivatives.
 *
 * For a given fadbad::F<T> object, this function traverses all derivative
 * components recursively and returns the largest absolute value.
 *
 * @param v Value to inspect.
 * @return Maximum absolute value found in the variable or its derivatives.
 */
inline double max_abs_with_derivatives(double v) { return std::abs(v); }

template<typename T>
inline double max_abs_with_derivatives(const F<T>& v) {
  auto& v_mut = const_cast<F<T>&>(v);
  double maxv = std::abs(scalar_value(v_mut.x()));
  for (int i = 0; i < v_mut.size(); ++i)
    maxv = std::max(maxv, max_abs_with_derivatives(v_mut.d(i)));
  return maxv;
}

/**
 * @brief Computes a weighted infinity norm for vectors of doubles.
 *
 * This is used in adaptive step-size control to scale the tolerance
 * according to the magnitude of each variable.
 *
 * @param v    Vector of residuals or differences.
 * @param x0   Reference vector for scaling.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return Weighted infinity norm (maximum scaled component).
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
 * @brief Weighted infinity norm for fadbad::F<T> vectors.
 *
 * Unlike the plain double version, this recursively includes derivative
 * magnitudes to ensure step-size control respects derivative sensitivities.
 *
 * @tparam T Inner type (usually `double`).
 * @param v    Vector of FADBAD variables.
 * @param x0   Reference vector for scaling.
 * @param atol Absolute tolerance.
 * @param rtol Relative tolerance.
 * @return Weighted infinity norm including derivatives.
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
    F<T>& vi = v[i];
    F<T>& xi = x0[i];
    double scale = atol + rtol * std::abs(scalar_value(xi));
    double comp_max = max_abs_with_derivatives(vi);
    nrm = std::max(nrm, comp_max / scale);
  }
  return nrm;
}

/**
 * @brief Estimates a suitable initial integration step size for non-stiff problems (double precision).
 *
 * The heuristic is based on the curvature criterion:
 * \f[
 *   \Delta t \approx \sqrt{\eta \, \frac{|x|}{|x''|}}
 * \f]
 *
 * where `η` is a small safety factor (default 1e-3).
 *
 * @param system    Right-hand-side function f(x,t).
 * @param jacobian  Jacobian function ∂f/∂x.
 * @param x0        Initial state vector.
 * @param t0        Initial time.
 * @param te        Target time.
 * @param atol      Absolute tolerance.
 * @param rtol      Relative tolerance.
 * @param eta       Safety factor (default 1e-3).
 * @return Suggested initial time step.
 */
inline double estimate_initial_dt(
    const std::function<void(vector<double>&, vector<double>&, double)>& system,
    const std::function<void(vector<double>&, matrix<double>&, double, vector<double>&)>& jacobian,
    vector<double>& x0,
    double t0,
    double te,
    double atol,
    double rtol,
    double eta = 1e-3)
{
  vector<double> dxdt(x0.size());
  system(x0, dxdt, t0);

  matrix<double> J(x0.size(), x0.size());
  vector<double> dfdt(x0.size());
  jacobian(x0, J, t0, dfdt);

  vector<double> xdd(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    double sum = dfdt[i];
    for (std::size_t j = 0; j < x0.size(); ++j)
      sum += J(i, j) * dxdt[j];
    xdd[i] = sum;
  }

  double norm_x   = weighted_sup_norm(x0, x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  double dt_curv = (norm_xdd > 0.0)
    ? std::sqrt(eta * norm_x / norm_xdd)
      : atol * (te - t0);

  return dt_curv;
}

/**
 * @brief Estimates initial step size for FADBAD-based ODEs.
 *
 * Equivalent to the double-precision version but supports AD variables
 * to allow fully differentiable solver initialization.
 *
 * @tparam T Inner type (e.g. `double`).
 * @param system    RHS function `dx/dt = f(x, t)`.
 * @param jacobian  Jacobian function ∂f/∂x.
 * @param x0        Initial state vector.
 * @param t0        Initial time.
 * @param te        Target time.
 * @param atol      Absolute tolerance.
 * @param rtol      Relative tolerance.
 * @param eta       Safety factor (default 1e-3).
 * @return Suggested initial time step as `F<T>`.
 */
template<typename T>
inline F<T> estimate_initial_dt(
    const std::function<void(vector<F<T>>&, vector<F<T>>&, const F<T>&)>& system,
    const std::function<void(vector<F<T>>&, matrix<F<T>>&, const F<T>&, vector<F<T>>&)> &jacobian,
    vector<F<T>>& x0,
    const F<T> t0,
    const F<T> te,
    double atol,
    double rtol,
    double eta = 1e-3)
{
  vector<F<T>> dxdt(x0.size());
  system(x0, dxdt, t0);

  matrix<F<T>> J(x0.size(), x0.size());
  vector<F<T>> dfdt(x0.size());
  jacobian(x0, J, t0, dfdt);

  vector<F<T>> xdd(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    F<T> sum = dfdt[i];
    for (std::size_t j = 0; j < x0.size(); ++j)
      sum += J(i, j) * dxdt[j];
    xdd[i] = sum;
  }

  double norm_x   = weighted_sup_norm(x0, x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  double dt_curv = (norm_xdd > 0.0)
    ? std::sqrt(eta * norm_x / norm_xdd)
      : atol * scalar_value(te - t0);

  return F<T>(dt_curv);
}

/**
 * @brief Generic overload for arbitrary functor-style systems (non-std::function).
 *
 * Wraps callable objects (like structs or lambdas) into std::function
 * before calling the AD-compatible version.
 */
template<typename System, typename Jacobian, typename T>
inline F<T> estimate_initial_dt(
    System system, Jacobian jacobian,
    vector<F<T>>& x0,
    const F<T> t0,
    const F<T> te,
    double atol, double rtol, double eta = 1e-3)
{
  std::function<void(vector<F<T>>&, vector<F<T>>&, const F<T>&)> sys_f =
    [&system](vector<F<T>>& x, vector<F<T>>& dxdt, const F<T>& t) { system(x, dxdt, t); };

    std::function<void(vector<F<T>>&, matrix<F<T>>&, const F<T>&, vector<F<T>>&)> jac_f =
      [&jacobian](vector<F<T>>& x, matrix<F<T>>& J, const F<T>& t, vector<F<T>>& dfdt) {
        jacobian(x, J, t, dfdt);
      };

      return estimate_initial_dt<T>(sys_f, jac_f, x0, t0, te, atol, rtol, eta);
}

} // namespace odeint_utils

// ============================================================================
//  FADBAD namespace patching: safe operator< and abs() overloads
// ============================================================================
namespace fadbad {

/**
 * @brief Safe less-than operator for comparing nested fadbad types.
 *
 * This overload prevents ambiguous template resolution between
 * fadbad::F<F<T>> and fadbad::F<T> comparisons by delegating
 * scalar extraction to Boost.Odeint’s scalar_value().
 *
 * @tparam Inner  Inner fadbad or arithmetic type.
 * @param a First operand (possibly nested FAD type).
 * @param b Second operand.
 * @return True if scalar_value(a) < scalar_value(b).
 */
template <typename Inner>
inline bool operator<(const F<F<Inner>>& a, const F<Inner>& b) {
  using boost::numeric::odeint::detail::scalar_value;
  return scalar_value(a) < scalar_value(b);
}

/**
 * @brief Symmetric overload for comparing F<T> with F<F<T>>.
 * @see operator<(const F<F<Inner>>&, const F<Inner>&)
 */
template <typename Inner>
inline bool operator<(const F<Inner>& a, const F<F<Inner>>& b) {
  using boost::numeric::odeint::detail::scalar_value;
  return scalar_value(a) < scalar_value(b);
}

/**
 * @brief Safe absolute value function for nested fadbad::F types.
 *
 * This replaces fadbad's internal abs() implementation, which fails
 * for nested types due to ambiguous `operator<` overloads.
 *
 * @tparam Inner Inner fadbad or arithmetic type.
 * @param x Value to take the absolute of.
 * @return `|x|` evaluated safely using scalar_value().
 */
template <typename Inner>
inline F<F<Inner>> abs(const F<F<Inner>>& x) {
  using boost::numeric::odeint::detail::scalar_value;
  double val = scalar_value(x);
  return (val < 0.0) ? -x : x;
}

} // namespace fadbad

#endif // BOOST_ROSENBROCK34_FAD_HPP
