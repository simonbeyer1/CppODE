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
#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller_ad.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_dense_output_ad.hpp>
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
//  odeint_utils namespace — helper functions for AD and initial step-size selection
// ============================================================================

namespace odeint_utils {

using boost::numeric::ublas::vector;
using boost::numeric::ublas::matrix;
using fadbad::F;

/*==============================================================================
 Scalar extraction — full recursive unwrapping of nested F<F<...>>
 ==============================================================================*/

inline double scalar_value(double v) { return v; }

template<typename T>
inline double scalar_value(const F<T>& v)
{
  // FADBAD's .x() is non-const → remove constness
  return scalar_value(const_cast<F<T>&>(v).x());
}

/*==============================================================================
 Maximum absolute value including all derivative components
 ==============================================================================*/

inline double max_abs_with_derivatives(double v)
{
  return std::abs(v);
}

template<typename T>
inline double max_abs_with_derivatives(const F<T>& v)
{
  auto& v_mut = const_cast<F<T>&>(v);

  double maxv = std::abs(scalar_value(v_mut.x()));
  for (int i = 0; i < v_mut.size(); ++i)
    maxv = std::max(maxv, max_abs_with_derivatives(v_mut.d(i)));

  return maxv;
}

/*==============================================================================
 Weighted infinity norms
 ==============================================================================*/

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

/*==============================================================================
 Initial step-size selection for plain double (non-AD)
 Based on Hairer–Wanner NDF heuristics:
 dt1 = sqrt( η * |x| / |x'| )
 dt2 = cbrt( η * |x| / |x''| )
 ==============================================================================*/

inline double estimate_initial_dt_local(
    const std::function<void(const vector<double>&, vector<double>&, const double&)>& system,
    const std::function<void(const vector<double>&, matrix<double>&, const double&, vector<double>&)>& jacobian,
    vector<double>& x0,
    double t0,
    double atol,
    double rtol,
    double eta = 1e-3)
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
  double dt1 = std::numeric_limits<double>::infinity();
  double dt2 = std::numeric_limits<double>::infinity();

  if (norm_dx > 0.0)
    dt1 = std::sqrt(eta * norm_x / norm_dx);

  if (norm_xdd > 0.0)
    dt2 = std::cbrt(eta * norm_x / norm_xdd);

  double dt = std::min(dt1, dt2);

  if (!std::isfinite(dt) || dt <= 0.0)
    dt = 1e-6;  // fallback

  return dt;
}

/**
 * @brief Convenience overload for arbitrary functor-style systems (double version)
 */
template<typename System, typename Jacobian>
inline double estimate_initial_dt_local(
    System system,
    Jacobian jacobian,
    vector<double>& x0,
    double t0,
    double atol,
    double rtol,
    double eta = 1e-3)
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

/*==============================================================================
 Initial step-size selection for AD types (F<T>)
 Based on Hairer–Wanner NDF heuristics:
 dt1 = sqrt( η * |x| / |x'| )
 dt2 = cbrt( η * |x| / |x''| )
 ==============================================================================*/

template<typename T>
inline F<T> estimate_initial_dt_local(
    const std::function<void(vector<F<T>>&, vector<F<T>>&, const F<T>&)>& system,
    const std::function<void(vector<F<T>>&, matrix<F<T>>&, const F<T>&, vector<F<T>>&)> &jacobian,
    vector<F<T>>& x0,
    const F<T> t0,
    double atol,
    double rtol,
    double eta = 1e-3)
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

  if (norm_dx > 0.0)
    dt1 = std::sqrt(eta * norm_x / norm_dx);

  if (norm_xdd > 0.0)
    dt2 = std::cbrt(eta * norm_x / norm_xdd);

  double dt = std::min(dt1, dt2);

  if (!std::isfinite(dt) || dt <= 0.0)
    dt = 1e-6;  // fallback

  return F<T>(dt);
}

/**
 * @brief Convenience overload for arbitrary functor-style systems (AD version)
 */
template<typename System, typename Jacobian, typename T>
inline F<T> estimate_initial_dt_local(
    System system,
    Jacobian jacobian,
    vector<F<T>>& x0,
    const F<T> t0,
    double atol,
    double rtol,
    double eta = 1e-3)
{
  // Wrap into std::function so both AD and non-AD versions work
  std::function<void(vector<F<T>>&, vector<F<T>>&, const F<T>&)> sys_f =
    [&system](vector<F<T>>& x, vector<F<T>>& dxdt, const F<T>& t) {
      system(x, dxdt, t);
    };

    std::function<void(vector<F<T>>&, matrix<F<T>>&, const F<T>&, vector<F<T>>&)> jac_f =
      [&jacobian](vector<F<T>>& x, matrix<F<T>>& J, const F<T>& t, vector<F<T>>& dfdt) {
        jacobian(x, J, t, dfdt);
      };

      return estimate_initial_dt_local<T>(
        sys_f, jac_f, x0, t0,
        atol, rtol, eta);
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
 * scalar extraction to Boost.Odeint's scalar_value().
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
