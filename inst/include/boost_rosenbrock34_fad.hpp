#ifndef BOOST_ROSENBROCK34_FAD_HPP
#define BOOST_ROSENBROCK34_FAD_HPP

#include <algorithm>
#include <vector>
#include <cmath>
#include <fadbad++/fadiff.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller_fad.hpp>
#include <boost/numeric/odeint/integrate/detail/integrate_times_with_events.hpp>
#include <boost/numeric/odeint/integrate/step_checker.hpp>

namespace boost { namespace numeric { namespace ublas {

template<class T>
inline matrix< fadbad::F<T> >
operator*(const fadbad::F<T>& a,
          const identity_matrix< fadbad::F<T> >& I)
{
  matrix< fadbad::F<T> > M(I.size1(), I.size2(), fadbad::F<T>(T(0)));
  for (std::size_t i = 0; i < I.size1(); ++i)
    M(i,i) = a;
  return M;
}

template<class T>
inline matrix< fadbad::F<T> >
operator*(const identity_matrix< fadbad::F<T> >& I,
          const fadbad::F<T>& a)
{
  return a * I;
}

} // namespace ublas
} // namespace numeric
} // namespace boost

namespace odeint_utils {

using boost::numeric::ublas::vector;
using boost::numeric::ublas::matrix;
using fadbad::F;

// ----------- weighted_sup_norm Spezialisierungen -----------

// double-Spezialisierung
inline double weighted_sup_norm(
    vector<double> &v,
    vector<double> &x0,
    double atol,
    double rtol
) {
  double nrm = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    double scale = atol + rtol * std::abs(x0[i]);
    double comp  = std::abs(v[i]);
    nrm = std::max(nrm, comp / scale);
  }
  return nrm;
}

// F<double>-Spezialisierung (mit Sensitivitäten)
inline double weighted_sup_norm(
    vector<F<double>> &v,
    vector<F<double>> &x0,
    double atol,
    double rtol
) {
  double nrm = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    F<double> &vi = v[i];
    F<double> &xi = x0[i];
    double scale = atol + rtol * std::abs(xi.x());

    // max über Funktionswert + alle Sensitivitäten
    double comp_max = std::abs(vi.x());
    for (int j = 0; j < vi.size(); ++j) {
      comp_max = std::max(comp_max, std::abs(vi.d(j)));
    }

    nrm = std::max(nrm, comp_max / scale);
  }
  return nrm;
}

// ----------- Version für AD -----------
// nutzt F<double> und berücksichtigt Sensitivitäten
inline F<double> estimate_initial_dt(
    const std::function<void(vector<F<double>>&, vector<F<double>>&, const F<double>&)>& system,
    const std::function<void(vector<F<double>>&, matrix<F<double>>&, const F<double>&, vector<F<double>>&)> &jacobian,
    vector<F<double>> &x0,
    const F<double> t0,
    const F<double> te,
    double atol,
    double rtol,
    double eta  = 1e-3
) {
  // Evaluate RHS f(t0, x0)
  vector<F<double>> dxdt(x0.size());
  system(x0, dxdt, t0);

  // Jacobian and df/dt
  matrix<F<double>> J(x0.size(), x0.size());
  vector<F<double>> dfdt(x0.size());
  jacobian(x0, J, t0, dfdt);

  // compute x'' = dfdt + J * f
  vector<F<double>> xdd(x0.size());
  for (std::size_t i = 0; i < x0.size(); ++i) {
    F<double> sum = dfdt[i];
    for (std::size_t j = 0; j < x0.size(); ++j) {
      sum += J(i, j) * dxdt[j];
    }
    xdd[i] = sum;
  }

  // norms with sensitivities
  double norm_x   = weighted_sup_norm(x0,  x0, atol, rtol);
  double norm_xdd = weighted_sup_norm(xdd, x0, atol, rtol);

  double dt_curv = (norm_xdd > 0.0)
    ? std::sqrt(eta * norm_x / norm_xdd)
      : atol * (te - t0).x();

  return F<double>(dt_curv);
}

// ----------- Version für double -----------
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
