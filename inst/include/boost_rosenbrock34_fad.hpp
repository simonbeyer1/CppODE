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

/*
 * ============================================================================
 *  boost_rosenbrock34_fad.hpp
 *  -----------------------------------------
 *  Helper utilities for Boost.Odeint Rosenbrock steppers with FADBAD++ types.
 *  Fixes issues with constness, template deduction, and callable overloading.
 * ============================================================================
 */

namespace boost { namespace numeric { namespace ublas {

// --------------------------------------------------------------------------
//  Identity matrix multiplications for fadbad::F<T>
// --------------------------------------------------------------------------
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

template<class T>
inline matrix< fadbad::F<T> >
operator*(const identity_matrix< fadbad::F<T> >& I,
          const fadbad::F<T>& a)
{
  return a * I;
}

}}} // namespace boost::numeric::ublas

// ============================================================================
//  odeint_utils namespace: FADBAD helpers and dt-estimation
// ============================================================================
namespace odeint_utils {

using boost::numeric::ublas::vector;
using boost::numeric::ublas::matrix;
using fadbad::F;

// ---------------------------------------------------------------------------
//  scalar_value: safely extract double from fadbad::F<T>
// ---------------------------------------------------------------------------
inline double scalar_value(double v) { return v; }

template<typename T>
inline double scalar_value(const F<T>& v) {
  // F<T>::x() is non-const, so we must const_cast
  return scalar_value(const_cast<F<T>&>(v).x());
}

// ---------------------------------------------------------------------------
//  max_abs_with_derivatives: recursive max(|x|, |dx_i|, ...)
// ---------------------------------------------------------------------------
inline double max_abs_with_derivatives(double v) { return std::abs(v); }

template<typename T>
inline double max_abs_with_derivatives(const F<T>& v) {
  auto& v_mut = const_cast<F<T>&>(v);
  double maxv = std::abs(scalar_value(v_mut.x()));
  for (int i = 0; i < v_mut.size(); ++i)
    maxv = std::max(maxv, max_abs_with_derivatives(v_mut.d(i)));
  return maxv;
}

// ---------------------------------------------------------------------------
//  weighted_sup_norm: elementwise scaling for error control
// ---------------------------------------------------------------------------
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
    F<T>& vi = v[i];
    F<T>& xi = x0[i];
    double scale = atol + rtol * std::abs(scalar_value(xi));
    double comp_max = max_abs_with_derivatives(vi);
    nrm = std::max(nrm, comp_max / scale);
  }
  return nrm;
}

// ---------------------------------------------------------------------------
//  estimate_initial_dt for plain double types
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
//  estimate_initial_dt for FADBAD types (std::function interface)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
//  estimate_initial_dt overload for arbitrary callable functors
// ---------------------------------------------------------------------------
//  This allows direct passing of structs/lambdas instead of std::function.
// ---------------------------------------------------------------------------
template<typename System, typename Jacobian, typename T>
inline F<T> estimate_initial_dt(
    System system, Jacobian jacobian,
    vector<F<T>>& x0,
    const F<T> t0,
    const F<T> te,
    double atol, double rtol, double eta = 1e-3)
{
  // Wrap functors into std::function with correct signatures
  std::function<void(vector<F<T>>&, vector<F<T>>&, const F<T>&)> sys_f =
    [&system](vector<F<T>>& x, vector<F<T>>& dxdt, const F<T>& t) {
      system(x, dxdt, t);
    };

    std::function<void(vector<F<T>>&, matrix<F<T>>&, const F<T>&, vector<F<T>>&)> jac_f =
      [&jacobian](vector<F<T>>& x, matrix<F<T>>& J, const F<T>& t, vector<F<T>>& dfdt) {
        jacobian(x, J, t, dfdt);
      };

      // Correct order of arguments (system first, then jacobian)
      return estimate_initial_dt<T>(sys_f, jac_f, x0, t0, te, atol, rtol, eta);
}

} // namespace odeint_utils

namespace fadbad {

/**
 * @brief Custom operator< overload to avoid ambiguity when comparing nested fadbad::F types.
 *
 * This overload explicitly handles comparisons between F<F<double>> and F<double>
 * (and more generally F<F<T>> and F<T>) by comparing their scalar base values.
 */
template <typename Inner>
inline bool operator<(const F<F<Inner>>& a, const F<Inner>& b) {
  return scalar_value(a) < scalar_value(b);
}

template <typename Inner>
inline bool operator<(const F<Inner>& a, const F<F<Inner>>& b) {
  return scalar_value(a) < scalar_value(b);
}

/**
 * @brief Safe absolute value for nested fadbad::F types.
 *
 * FADBAD’s default abs(x) uses (x < 0) ? -x : x, which is ambiguous
 * for nested types. This overload compares scalar base values directly.
 */
template <typename Inner>
inline F<F<Inner>> abs(const F<F<Inner>>& x) {
  return (scalar_value(x) < 0.0) ? -x : x;
}

} // namespace fadbad

#endif // BOOST_ROSENBROCK34_FAD_HPP
