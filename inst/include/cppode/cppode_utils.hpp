/*
 Utility functions for ODE integration and sensitivity calculation with automatic differentiation

 This header provides:
 - Safe scalar extraction from nested FADBAD types (F<F<T>>)
 - Recursive maximum absolute value extraction (including derivatives)
 - Weighted norms for adaptive step-size control
 - Initial step-size estimation for stiff solvers (Hairer-Nørsett-Wanner heuristic)

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_UTILS_HPP
#define CPPODE_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <fadbad++/fadiff.h>
#include <cppode/cppode_types.hpp>
#include <cppode/cppode_sparse_ad_lu.hpp>

namespace odeint_utils {

using fadbad::F;

// =========================================================================================
//  Scalar extraction — full recursive unwrapping of nested F<F<...>>
// =========================================================================================

inline double scalar_value(double v) { return v; }

template<typename T, unsigned int N>
inline double scalar_value(const F<T,N>& v)
{
  return scalar_value(const_cast<F<T,N>&>(v).x());
}

// =========================================================================================
//  Maximum absolute value including all derivative components
// =========================================================================================

inline double max_abs_with_derivatives(double v)
{
  return std::abs(v);
}

template<typename T, unsigned int N>
inline double max_abs_with_derivatives(const F<T,N>& v)
{
  auto& v_mut = const_cast<F<T,N>&>(v);
  double maxv = std::abs(scalar_value(v_mut.x()));
  for (unsigned int i = 0; i < v_mut.size(); ++i)
    maxv = std::max(maxv, max_abs_with_derivatives(v_mut.d(i)));
  return maxv;
}

// =========================================================================================
//  Weighted infinity norms
// =========================================================================================

inline double weighted_sup_norm(
    std::vector<double>& v,
    std::vector<double>& x0,
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

template<typename T, unsigned int N>
inline double weighted_sup_norm(
    std::vector<F<T,N>>& v,
    std::vector<F<T,N>>& x0,
    double atol,
    double rtol)
{
  double nrm = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    auto& vi = const_cast<F<T,N>&>(v[i]);
    auto& yi = const_cast<F<T,N>&>(x0[i]);

    // Value component only: |v_i| / (atol + rtol * |y_i|)
    // Derivative components are excluded — the IFT guarantees that
    // derivative accuracy tracks value accuracy, so controlling
    // step size via derivative norms is redundant and can cause
    // step size collapse when derivative magnitudes >> value magnitudes.
    double vi_val = std::abs(scalar_value(vi));
    double yi_val = std::abs(scalar_value(yi));
    nrm = std::max(nrm, vi_val / (atol + rtol * yi_val));
  }
  return nrm;
}

// =========================================================================================
//  Initial step-size estimation (Hairer-Nørsett-Wanner heuristic)
// =========================================================================================

inline double estimate_initial_dt_local(
    const std::function<void(const std::vector<double>&, std::vector<double>&, const double&)>& system,
    const std::function<void(const std::vector<double>&, cppode::dense_matrix<double>&, const double&, std::vector<double>&)>& jacobian,
    std::vector<double>& x0,
    double t0,
    double atol,
    double rtol,
    double eta = 5e-2)
{
  const std::size_t n = x0.size();

  std::vector<double> dxdt(n);
  system(x0, dxdt, t0);

  cppode::dense_matrix<double> J(n, n);
  std::vector<double> dfdt(n);
  jacobian(x0, J, t0, dfdt);

  // x'' = df/dt + J*dxdt
  std::vector<double> xdd(n);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = dfdt[i];
    for (std::size_t j = 0; j < n; ++j)
      sum += J(i,j) * dxdt[j];
    xdd[i] = sum;
  }

  double norm_x   = weighted_sup_norm(x0, x0, atol, rtol);
  double norm_dx  = weighted_sup_norm(dxdt, x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  double dt1 = std::numeric_limits<double>::infinity();
  double dt2 = std::numeric_limits<double>::infinity();

  if (norm_dx > 0.0) dt1 = norm_x / norm_dx;
  if (norm_xdd > 0.0) dt2 = std::sqrt(norm_x / norm_xdd);

  double dt = eta * std::min(dt1, dt2);
  if (!std::isfinite(dt) || dt <= 0.0) dt = atol;

  return dt;
}

template<typename System, typename Jacobian>
inline double estimate_initial_dt_local(
    System system,
    Jacobian jacobian,
    std::vector<double>& x0,
    double t0,
    double atol,
    double rtol,
    double eta = 5e-2)
{
  std::function<void(const std::vector<double>&, std::vector<double>&, const double&)> sys_f =
    [&system](const std::vector<double>& x, std::vector<double>& dxdt, const double& t) {
      system(x, dxdt, t);
    };

    std::function<void(const std::vector<double>&, cppode::dense_matrix<double>&, const double&, std::vector<double>&)> jac_f =
      [&jacobian](const std::vector<double>& x, cppode::dense_matrix<double>& J, const double& t, std::vector<double>& dfdt) {
        jacobian(x, J, t, dfdt);
      };

      return estimate_initial_dt_local(sys_f, jac_f, x0, t0, atol, rtol, eta);
}

template<typename T, unsigned int N>
inline double estimate_initial_dt_local(
    const std::function<void(std::vector<F<T,N>>&, std::vector<F<T,N>>&, const F<T,N>&)>& system,
    const std::function<void(std::vector<F<T,N>>&, cppode::dense_matrix<F<T,N>>&, const F<T,N>&, std::vector<F<T,N>>&)>& jacobian,
    std::vector<F<T,N>>& x0,
    const F<T,N> t0,
    double atol,
    double rtol,
    double eta = 5e-2)
{
  const std::size_t n = x0.size();

  std::vector<F<T,N>> dxdt(n);
  system(x0, dxdt, t0);

  cppode::dense_matrix<F<T,N>> J(n, n);
  std::vector<F<T,N>> dfdt(n);
  jacobian(x0, J, t0, dfdt);

  std::vector<F<T,N>> xdd(n);
  for (std::size_t i = 0; i < n; ++i) {
    F<T,N> sum = dfdt[i];
    for (std::size_t j = 0; j < n; ++j)
      sum += J(i,j) * dxdt[j];
    xdd[i] = sum;
  }

  double norm_x   = weighted_sup_norm(x0, x0, atol, rtol);
  double norm_dx  = weighted_sup_norm(dxdt, x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  double dt1 = std::numeric_limits<double>::infinity();
  double dt2 = std::numeric_limits<double>::infinity();

  if (norm_dx > 0.0) dt1 = norm_x / norm_dx;
  if (norm_xdd > 0.0) dt2 = std::sqrt(norm_x / norm_xdd);

  double dt = eta * std::min(dt1, dt2);
  if (!std::isfinite(dt) || dt <= 0.0) dt = atol;

  return dt;
}

template<typename System, typename Jacobian, typename T, unsigned int N>
inline double estimate_initial_dt_local(
    System system,
    Jacobian jacobian,
    std::vector<F<T,N>>& x0,
    const F<T,N> t0,
    double atol,
    double rtol,
    double eta = 5e-2)
{
  std::function<void(std::vector<F<T,N>>&, std::vector<F<T,N>>&, const F<T,N>&)> sys_f =
    [&system](std::vector<F<T,N>>& x, std::vector<F<T,N>>& dxdt, const F<T,N>& t) {
      system(x, dxdt, t);
    };

    std::function<void(std::vector<F<T,N>>&, cppode::dense_matrix<F<T,N>>&, const F<T,N>&, std::vector<F<T,N>>&)> jac_f =
      [&jacobian](std::vector<F<T,N>>& x, cppode::dense_matrix<F<T,N>>& J, const F<T,N>& t, std::vector<F<T,N>>& dfdt) {
        jacobian(x, J, t, dfdt);
      };

      return estimate_initial_dt_local<T,N>(sys_f, jac_f, x0, t0, atol, rtol, eta);
}

// ============================================================================
//  Sparse estimate_initial_dt_local
//
//  The Jacobian functor writes into csc_matrix (pattern built on first call).
//  J·dxdt uses csc_matvec_add which iterates only over non-zeros.
// ============================================================================

template<typename System, typename Jacobian>
inline double estimate_initial_dt_local_sparse(
    System system,
    Jacobian jacobian,
    std::vector<double>& x0,
    double t0,
    double atol,
    double rtol,
    double eta = 5e-2)
{
  const std::size_t n = x0.size();

  std::vector<double> dxdt(n);
  system(x0, dxdt, t0);

  cppode::csc_matrix<double> J;
  std::vector<double> dfdt(n, 0.0);
  jacobian(x0, J, t0, dfdt);

  // x'' = df/dt + J·dxdt (sparse product)
  std::vector<double> xdd(dfdt);
  cppode::csc_matvec_add(J, dxdt, xdd);

  double norm_x   = weighted_sup_norm(x0, x0, atol, rtol);
  double norm_dx  = weighted_sup_norm(dxdt, x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  double dt1 = std::numeric_limits<double>::infinity();
  double dt2 = std::numeric_limits<double>::infinity();

  if (norm_dx > 0.0) dt1 = norm_x / norm_dx;
  if (norm_xdd > 0.0) dt2 = std::sqrt(norm_x / norm_xdd);

  double dt = eta * std::min(dt1, dt2);
  if (!std::isfinite(dt) || dt <= 0.0) dt = atol;
  return dt;
}

// AD version — returns double (step size is a scalar, not an AD type)
template<typename T, unsigned int N, typename System, typename Jacobian>
inline double estimate_initial_dt_local_sparse(
    System system,
    Jacobian jacobian,
    std::vector<fadbad::F<T,N>>& x0,
    const fadbad::F<T,N> t0,
    double atol,
    double rtol,
    double eta = 5e-2)
{
  using AD = fadbad::F<T,N>;
  const std::size_t n = x0.size();

  std::vector<AD> dxdt(n);
  system(x0, dxdt, t0);

  cppode::csc_matrix<AD> J;
  std::vector<AD> dfdt(n);
  jacobian(x0, J, t0, dfdt);

  std::vector<AD> xdd(dfdt);
  cppode::csc_matvec_add(J, dxdt, xdd);

  double norm_x   = weighted_sup_norm(x0, x0, atol, rtol);
  double norm_dx  = weighted_sup_norm(dxdt, x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  double dt1 = std::numeric_limits<double>::infinity();
  double dt2 = std::numeric_limits<double>::infinity();

  if (norm_dx > 0.0) dt1 = norm_x / norm_dx;
  if (norm_xdd > 0.0) dt2 = std::sqrt(norm_x / norm_xdd);

  double dt = eta * std::min(dt1, dt2);
  if (!std::isfinite(dt) || dt <= 0.0) dt = atol;
  return dt;
}

} // namespace odeint_utils

// ============================================================================
//  BDF-specific initial step size estimation
// ============================================================================

namespace bdf_utils {

template<typename System, typename Jacobian>
inline double estimate_initial_dt_bdf(
    System system, Jacobian jacobian,
    std::vector<double>& x0, double t0,
    double atol, double rtol)
{
  return odeint_utils::estimate_initial_dt_local(system, jacobian, x0, t0, atol, rtol, 1e-2);
}

template<typename System, typename Jacobian, typename T, unsigned int N>
inline double estimate_initial_dt_bdf(
    System system, Jacobian jacobian,
    std::vector<fadbad::F<T,N>>& x0, const fadbad::F<T,N> t0,
    double atol, double rtol)
{
  return odeint_utils::estimate_initial_dt_local<System, Jacobian, T, N>(system, jacobian, x0, t0, atol, rtol, 1e-2);
}

template<typename System, typename Jacobian>
inline double estimate_initial_dt_bdf_sparse(
    System system, Jacobian jacobian,
    std::vector<double>& x0, double t0,
    double atol, double rtol)
{
  return odeint_utils::estimate_initial_dt_local_sparse(system, jacobian, x0, t0, atol, rtol, 1e-2);
}

template<typename T, unsigned int N, typename System, typename Jacobian>
inline double estimate_initial_dt_bdf_sparse(
    System system, Jacobian jacobian,
    std::vector<fadbad::F<T,N>>& x0, const fadbad::F<T,N> t0,
    double atol, double rtol)
{
  return odeint_utils::estimate_initial_dt_local_sparse<T,N>(system, jacobian, x0, t0, atol, rtol, 1e-2);
}

} // namespace bdf_utils

#endif // CPPODE_UTILS_HPP
