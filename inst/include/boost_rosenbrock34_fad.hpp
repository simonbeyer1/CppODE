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

/**
 * @brief Weighted infinity-norm (supremum norm) for vectors.
 *
 * Computes the weighted sup norm of vector v with respect to x0:
 *     max_i |v_i| / (atol + rtol * |x0_i|)
 *
 * Supports both plain double and FADBAD types (F<double>, F<F<double>>, ...).
 *
 * @tparam T scalar or FADBAD type
 * @param v vector of values
 * @param x0 reference state vector (for scaling)
 * @param atol absolute tolerance
 * @param rtol relative tolerance
 * @return double weighted supremum norm
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
    // convert to plain double using .x() or operator double()
    double scale = atol + rtol * std::abs(static_cast<double>(x0[i]));
    double comp  = std::abs(static_cast<double>(v[i]));
    nrm = std::max(nrm, comp / scale);
  }
  return nrm;
}

/**
 * @brief Weighted infinity-norm with sensitivities (first or higher-order AD).
 *
 * For FADBAD F<T> values, this function includes all derivatives in the norm:
 *     max( |x|, |dx/dp_i| ) for all i.
 *
 * It works recursively: if T itself is F<U>, the derivatives d(j) may also
 * contain nested AD values (F<F<U>>).
 *
 * @tparam T inner scalar type (e.g. double or F<double>)
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

    // scaling uses the base value (cast works recursively)
    double scale = atol + rtol * std::abs(static_cast<double>(xi));

    // start with function value
    double comp_max = std::abs(static_cast<double>(vi));

    // include all sensitivities
    for (int j = 0; j < vi.size(); ++j) {
      comp_max = std::max(comp_max,
                          std::abs(static_cast<double>(vi.d(j))));
    }

    nrm = std::max(nrm, comp_max / scale);
  }
  return nrm;
}

/**
 * @brief Estimate an initial time step for ODE integration (AD version).
 *
 * This template operates on FADBAD types (e.g. F<double>, F<F<double>>) and
 * includes sensitivities in the curvature-based step size estimation.
 * It works for first- and higher-order automatic differentiation.
 *
 * @tparam T         Inner scalar type (e.g. double or F<double>)
 * @tparam System    Callable type for the ODE system: void(x, dxdt, t)
 * @tparam Jacobian  Callable type for the Jacobian: void(x, J, t, dfdt)
 *
 * @param system   Callable object or lambda computing the ODE RHS f(x, dxdt, t)
 * @param jacobian Callable object or lambda computing the Jacobian and df/dt
 * @param x0       Initial state vector
 * @param t0       Start time
 * @param te       End time
 * @param atol     Absolute tolerance
 * @param rtol     Relative tolerance
 * @param eta      Safety scaling factor (default 1e-3)
 *
 * @return F<T> — estimated initial time step (same nesting level as x0 elements)
 */
template<typename T, typename System, typename Jacobian>
inline F<T> estimate_initial_dt(
    System&& system,
    Jacobian&& jacobian,
    vector<F<T>> &x0,
    const F<T> t0,
    const F<T> te,
    double atol,
    double rtol,
    double eta = 1e-3
) {
  // Evaluate RHS f(t0, x0)
  vector<F<T>> dxdt(x0.size());
  system(x0, dxdt, t0);

  // Jacobian and df/dt
  matrix<F<T>> J(x0.size(), x0.size());
  vector<F<T>> dfdt(x0.size());
  jacobian(x0, J, t0, dfdt);

  // compute x'' = dfdt + J * f
  vector<F<T>> xdd(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    F<T> sum = dfdt[i];
    for (std::size_t j = 0; j < x0.size(); ++j) {
      sum += J(i, j) * dxdt[j];
    }
    xdd[i] = sum;
  }

  double norm_x   = weighted_sup_norm(x0,  x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  double dt_curv = (norm_xdd > 0.0)
    ? std::sqrt(eta * norm_x / norm_xdd)
      : atol * static_cast<double>(te - t0);

  return F<T>(dt_curv);
}


/**
 * @brief Estimate an initial time step for ODE integration (plain double version).
 *
 * Computes the curvature-based step size using:
 *   dt = sqrt(eta * norm(x) / norm(xdd))
 *
 * @param system   ODE RHS function
 * @param jacobian Jacobian and df/dt function
 * @param x0       initial state
 * @param t0       start time
 * @param te       end time
 * @param atol     absolute tolerance
 * @param rtol     relative tolerance
 * @param eta      safety scaling (default 1e-3)
 * @return double suggested initial step size
 */
inline double estimate_initial_dt(
    const std::function<void(vector<double>&, vector<double>&, double)>& system,
    const std::function<void(vector<double>&, matrix<double>&, double, vector<double>&)> &jacobian,
    vector<double> &x0,
    double t0,
    double te,
    double atol,
    double rtol,
    double eta = 1e-3
) {
  // Evaluate RHS f(t0, x0)
  vector<double> dxdt(x0.size());
  system(x0, dxdt, t0);

  // Jacobian and df/dt
  matrix<double> J(x0.size(), x0.size());
  vector<double> dfdt(x0.size());
  jacobian(x0, J, t0, dfdt);

  // compute x'' = dfdt + J * f
  vector<double> xdd(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    double sum = dfdt[i];
    for (std::size_t j = 0; j < x0.size(); ++j) {
      sum += J(i, j) * dxdt[j];
    }
    xdd[i] = sum;
  }

  // norms
  double norm_x   = weighted_sup_norm(x0,  x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  double dt_curv = (norm_xdd > 0.0)
    ? std::sqrt(eta * norm_x / norm_xdd)
      : atol * (te - t0);

  return dt_curv;
}

} // namespace odeint_utils

#endif // BOOST_ROSENBROCK34_FAD_HPP
