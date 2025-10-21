#ifndef BOOST_ROSENBROCK34_FAD_HPP
#define BOOST_ROSENBROCK34_FAD_HPP

#include <algorithm>
#include <vector>
#include <cmath>
#include <fadbad++/fadiff.h>

namespace fadbad {
template<class T>
inline F<T> abs(const F<T>& x) { return (x < Op<T>::myZero()) ? -x : x; }

template<class T>
inline F<T> min(const F<T>& a, const F<T>& b) { return (a < b) ? a : b; }

template<class T>
inline F<T> max(const F<T>& a, const F<T>& b) { return (a > b) ? a : b; }
}



namespace std {

  // Import FADBAD math overloads so std::sin, std::sqrt, ... work
  using fadbad::abs;
  using fadbad::sin;
  using fadbad::cos;
  using fadbad::tan;
  using fadbad::asin;
  using fadbad::acos;
  using fadbad::atan;
  using fadbad::exp;
  using fadbad::log;
  using fadbad::sqrt;
  using fadbad::pow;
  using fadbad::sqr;
  using fadbad::abs;
  using fadbad::min;
  using fadbad::max;

} // namespace std


#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller_fad.hpp>
#include <boost/numeric/odeint/integrate/detail/integrate_times_with_events.hpp>
#include <boost/numeric/odeint/integrate/step_checker.hpp>


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

namespace odeint_utils {

using boost::numeric::ublas::vector;
using boost::numeric::ublas::matrix;
using fadbad::F;

// ============================================================================
//  Weighted infinity norms
// ============================================================================

/**
 * @brief Computes the weighted infinity norm (supremum norm) for a vector of plain types.
 *
 * The norm is defined as:
 * \f[
 *   \|v\|_{\infty,w} = \max_i \frac{|v_i|}{a_{\text{tol}} + r_{\text{tol}} |x_{0,i}|}
 * \f]
 *
 * @tparam T Scalar type (e.g., double)
 * @param v    Input vector
 * @param x0   Reference state (used for relative scaling)
 * @param atol Absolute tolerance
 * @param rtol Relative tolerance
 * @return Weighted supremum norm as a double
 */
template<typename T>
inline double weighted_sup_norm(
    vector<T> &v,
    vector<T> &x0,
    double atol,
    double rtol
) {
  double nrm = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    double scale = atol + rtol * std::abs(static_cast<double>(x0[i]));
    double comp  = std::abs(static_cast<double>(v[i]));
    nrm = std::max(nrm, comp / scale);
  }
  return nrm;
}

/**
 * @brief Computes the weighted infinity norm for FADBAD++ variables,
 *        including all derivative components recursively.
 *
 * The computation includes the base value and all sensitivities stored in F<T>:
 * \f[
 *   \|v\|_{\infty,w} = \max_i \frac{\max\big(|v_i|, |dv_i/dp_j|\big)}{a_{\text{tol}} + r_{\text{tol}} |x_{0,i}|}
 * \f]
 *
 * @tparam T Inner scalar type (e.g., double or F<double>)
 * @param v    Input vector of FADBAD variables
 * @param x0   Reference state (for relative scaling)
 * @param atol Absolute tolerance
 * @param rtol Relative tolerance
 * @return Weighted supremum norm including derivatives
 */
template<typename T>
inline double weighted_sup_norm(
    vector<F<T>> &v,
    vector<F<T>> &x0,
    double atol,
    double rtol
) {
  double nrm = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    F<T> &vi = v[i];
    F<T> &xi = x0[i];

    double scale = atol + rtol * std::abs(static_cast<double>(xi));
    double comp_max = std::abs(static_cast<double>(vi));

    // Include sensitivities recursively
    for (int j = 0; j < vi.size(); ++j)
      comp_max = std::max(comp_max, std::abs(static_cast<double>(vi.d(j))));

    nrm = std::max(nrm, comp_max / scale);
  }
  return nrm;
}

// ============================================================================
//  Matrix infinity norm
// ============================================================================

/**
 * @brief Computes the infinity norm (max row sum) of a dense matrix.
 *
 * \f[
 *   \|A\|_{\infty} = \max_i \sum_j |A_{ij}|
 * \f]
 *
 * @tparam T Scalar type (e.g., double)
 * @param J Input matrix
 * @return Matrix infinity norm as a double
 */
template<typename T>
inline double matrix_norm_inf(const matrix<T> &J) {
  double max_row_sum = 0.0;
  for (std::size_t i = 0; i < J.size1(); ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < J.size2(); ++j)
      sum += std::abs(static_cast<double>(J(i, j)));
    max_row_sum = std::max(max_row_sum, sum);
  }
  return max_row_sum;
}

/**
 * @brief Computes the infinity norm for a matrix of FADBAD++ values.
 *
 * The inner values are cast to double recursively via static_cast<double>(),
 * extracting the base value of each F<T> entry.
 *
 * @tparam T Inner scalar type (e.g., double or F<double>)
 * @param J Input matrix with FADBAD entries
 * @return Matrix infinity norm as a double
 */
template<typename T>
inline double matrix_norm_inf(const matrix<F<T>> &J) {
  double max_row_sum = 0.0;
  for (std::size_t i = 0; i < J.size1(); ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < J.size2(); ++j)
      sum += std::abs(static_cast<double>(J(i, j)));
    max_row_sum = std::max(max_row_sum, sum);
  }
  return max_row_sum;
}

// ============================================================================
//  Initial time step estimation (curvature- and stiffness-based)
// ============================================================================

/**
 * @brief Estimates an initial time step for stiff ODE integration using curvature and stiffness.
 *
 * This function combines two physically motivated criteria to determine a stable and
 * conservative initial step size for Rosenbrock-type (stiff) solvers.
 *
 * 1. **Curvature-based time scale**:
 *    \f[
 *      h_\text{curv} = \sqrt{ \eta \frac{ \|x_0\|_\infty }{ \|x''(t_0)\|_\infty } }
 *    \f]
 *    where \( x''(t_0) = \frac{df}{dt} + J f \) approximates the local curvature.
 *
 * 2. **Stiffness-based time scale**:
 *    \f[
 *      h_J = \frac{c_J}{ \|J(t_0)\|_\infty }
 *    \f]
 *    which limits the step size based on the Jacobian magnitude.
 *
 * The final step size is the smaller of both estimates, scaled by a safety margin.
 * The computed value is independent of user-specified tolerances.
 *
 * @tparam T         Inner scalar type (e.g., double or F<double>)
 * @tparam System    Callable object implementing f(x, dxdt, t)
 * @tparam Jacobian  Callable object implementing jacobian(x, J, t, dfdt)
 *
 * @param system   ODE right-hand side function
 * @param jacobian Jacobian and df/dt evaluator
 * @param x0       Initial state vector
 * @param t0       Start time
 * @param eta      Curvature scaling factor (default 1e-2)
 * @param c_J      Safety factor for stiffness-based limit (default 0.1)
 * @return F<T> Estimated initial step size (same AD nesting level as x0 elements)
 */
template<typename T, typename System, typename Jacobian>
inline F<T> estimate_initial_dt(
    System&& system,
    Jacobian&& jacobian,
    vector<F<T>>& x0,
    const F<T> t0,
    double eta = 1e-2,
    double c_J = 0.1
) {
  // Evaluate f(x0, t0)
  vector<F<T>> dxdt(x0.size());
  system(x0, dxdt, t0);

  // Compute Jacobian and df/dt
  matrix<F<T>> J(x0.size(), x0.size());
  vector<F<T>> dfdt(x0.size());
  jacobian(x0, J, t0, dfdt);

  // Approximate curvature term: x'' = df/dt + J * f
  vector<F<T>> xdd(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    F<T> sum = dfdt[i];
    for (std::size_t j = 0; j < x0.size(); ++j)
      sum += J(i, j) * dxdt[j];
    xdd[i] = sum;
  }

  // Unweighted infinity norms (physically scaled)
  auto norm_inf = [](const auto& v) {
    double n = 0.0;
    for (const auto& vi : v)
      n = std::max(n, std::abs(static_cast<double>(vi)));
    return n;
  };

  double norm_x   = norm_inf(x0);
  double norm_xdd = norm_inf(xdd);
  double normJ    = matrix_norm_inf(J);

  // Curvature-based step
  double dt_curv = (norm_xdd > 1e-15)
    ? std::sqrt(std::max(1e-16, eta * norm_x / norm_xdd))
      : 0.0;

  // Stiffness-based limit
  double dt_J = (normJ > 1e-15)
    ? c_J / normJ
  : 1.0;

  // Combine conservatively
  double dt_final = (dt_curv > 0.0)
    ? std::min(dt_curv, dt_J)
      : dt_J;

  // Safety margin
  dt_final *= 0.9;

  // Lower limit to avoid underflow
  dt_final = std::max(dt_final, 1e-14);

  return F<T>(dt_final);
}

/**
 * @brief Estimates an initial time step for stiff ODE integration (plain double version).
 *
 * Combines curvature-based and stiffness-based estimates to choose a stable initial
 * step size for Rosenbrock or other implicit methods.
 *
 * 1. **Curvature-based time scale**:
 *    \f[
 *      h_\text{curv} = \sqrt{ \eta \frac{ \|x_0\|_\infty }{ \|x''(t_0)\|_\infty } }
 *    \f]
 *
 * 2. **Stiffness-based time scale**:
 *    \f[
 *      h_J = \frac{c_J}{ \|J(t_0)\|_\infty }
 *    \f]
 *
 * The smaller of both values is chosen, multiplied by a safety factor (0.9).
 * This estimation is independent of user-specified tolerances and aims to capture
 * the natural dynamical and stiffness scales of the system.
 *
 * @param system   ODE right-hand side function: f(x, dxdt, t)
 * @param jacobian Jacobian function: jacobian(x, J, t, dfdt)
 * @param x0       Initial state vector
 * @param t0       Start time
 * @param eta      Curvature scaling factor (default 1e-2)
 * @param c_J      Safety factor for 1/||J|| (default 0.1)
 * @return double  Estimated initial time step
 */
inline double estimate_initial_dt(
    const std::function<void(vector<double>&, vector<double>&, double)>& system,
    const std::function<void(vector<double>&, matrix<double>&, double, vector<double>&)>& jacobian,
    vector<double>& x0,
    double t0,
    double eta = 1e-2,
    double c_J = 0.1
) {
  // Evaluate f(x0, t0)
  vector<double> dxdt(x0.size());
  system(x0, dxdt, t0);

  // Compute Jacobian and df/dt
  matrix<double> J(x0.size(), x0.size());
  vector<double> dfdt(x0.size());
  jacobian(x0, J, t0, dfdt);

  // Approximate curvature term: x'' = df/dt + J * f
  vector<double> xdd(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    double sum = dfdt[i];
    for (std::size_t j = 0; j < x0.size(); ++j)
      sum += J(i, j) * dxdt[j];
    xdd[i] = sum;
  }

  // Unweighted infinity norms
  auto norm_inf = [](const auto& v) {
    double n = 0.0;
    for (const auto& vi : v)
      n = std::max(n, std::abs(vi));
    return n;
  };

  double norm_x   = norm_inf(x0);
  double norm_xdd = norm_inf(xdd);
  double normJ    = matrix_norm_inf(J);

  // Curvature-based estimate
  double dt_curv = (norm_xdd > 1e-15)
    ? std::sqrt(std::max(1e-16, eta * norm_x / norm_xdd))
      : 0.0;

  // Stiffness-based limit
  double dt_J = (normJ > 1e-15)
    ? c_J / normJ
  : 1.0;

  // Combine
  double dt_final = (dt_curv > 0.0)
    ? std::min(dt_curv, dt_J)
      : dt_J;

  // Safety scaling
  dt_final *= 0.9;
  dt_final = std::max(dt_final, 1e-14);

  return dt_final;
}


} // namespace odeint_utils

#endif // BOOST_ROSENBROCK34_FAD_HPP
