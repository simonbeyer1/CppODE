#define R_NO_REMAP  // Prevent R headers from remapping common function names like 'length' to avoid conflicts with C++ STL
#include <R.h>
#include <Rinternals.h>

#include <algorithm>
#include <vector>
#include <cmath>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <cppad/cppad.hpp>

using namespace boost::numeric::odeint;
using boost::numeric::ublas::vector;
using boost::numeric::ublas::matrix;

using AD = CppAD::AD<double>;

// Van-der-Pol: wir brauchen nur mu als Parameter auf dem Tape
struct ode_system {
  AD mu;
  explicit ode_system(const AD& mu_) : mu(mu_) {}
  void operator()(const vector<AD>& x, vector<AD>& dxdt, const AD& t) {
    dxdt[0] = x[1];
    dxdt[1] = mu * (1.0 - x[0] * x[0]) * x[1] - x[0];
  }
};

struct jacobian {
  AD mu;
  explicit jacobian(const AD& mu_) : mu(mu_) {}
  void operator()(const vector<AD>& x, matrix<AD>& J, const AD& t, vector<AD>& dfdt) {
    J(0,0) = 0.0;                           J(0,1) = 1.0;
    J(1,0) = -2.0 * mu * x[0] * x[1] - 1.0; J(1,1) = mu * (1.0 - x[0] * x[0]);
    dfdt[0] = 0.0; dfdt[1] = 0.0;
  }
};

// Oberes Dreieck (mit Diagonale)
inline void make_upper_triangle_indices(size_t n,
                                        CppAD::vector<size_t>& row,
                                        CppAD::vector<size_t>& col) {
  row.clear(); col.clear();
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i; j < n; ++j) { row.push_back(i); col.push_back(j); }
}

// Observer—schreibt nur Zeiten (AD) und y=[x,v] (AD) fortlaufend
struct observer {
  std::vector<AD>& times;
  std::vector<AD>& y;
  int& event_count;
  double last_x0;
  bool event_pending;
  AD& t_event;
  vector<AD>& x_at_event;

  observer(std::vector<AD>& t, std::vector<AD>& y_, int& ec, AD& t_evt, vector<AD>& x_evt)
    : times(t), y(y_), event_count(ec), last_x0(NAN),
      event_pending(false), t_event(t_evt), x_at_event(x_evt) {}

  void operator()(vector<AD>& x, const AD& t) {
    double curr_x0 = CppAD::Value(x[0]);

    if (!std::isnan(last_x0)) {
      if (last_x0 < 0.0 && curr_x0 >= 0.0 && event_count < 1) {
        event_pending = true;
        t_event = t;
        x_at_event = x;  // Speichere Zustand
        x_at_event[0] = 2.0;  // Simuliere Reset
        event_count++;
        Rprintf("Event bei t = %.3f erkannt.\n", CppAD::Value(t));
      }
    }

    last_x0 = curr_x0;

    times.push_back(t);
    y.push_back(x[0]);
    y.push_back(x[1]);
  }
};



extern "C" SEXP solve(SEXP timesSEXP, SEXP paramsSEXP, SEXP abstolSEXP, SEXP reltolSEXP) {
  if (!Rf_isReal(timesSEXP) || Rf_length(timesSEXP) < 2) Rf_error("times must be numeric, length >= 2");
  if (!Rf_isReal(paramsSEXP) || Rf_length(paramsSEXP) != 3) Rf_error("params must be numeric length 3: x0, v0, mu");
  if (!Rf_isReal(abstolSEXP) || !Rf_isReal(reltolSEXP)) Rf_error("abstol/reltol must be numeric scalars");

  const int N = Rf_length(timesSEXP);
  const double* times  = REAL(timesSEXP);
  const double* params = REAL(paramsSEXP);
  const double abstol  = REAL(abstolSEXP)[0];
  const double reltol  = REAL(reltolSEXP)[0];

  // Unabhängige Variablen auf dem Tape (std::vector für ADFun-Konsistenz)
  // x0, v0, mu — wir nutzen x0/v0 nur als Initialzustand, mu im System
  std::vector<AD> indep(3);
  indep[0] = params[0];
  indep[1] = params[1];
  indep[2] = params[2];
  CppAD::Independent(indep);

  // Initialzustand (uBLAS-State für Odeint)
  vector<AD> x(2);
  x[0] = indep[0];
  x[1] = indep[1];

  // Zeiten sortieren + 5.0 einfügen
  std::vector<double> t_d(times, times + N);
  t_d.push_back(5.0);
  std::sort(t_d.begin(), t_d.end());
  std::vector<AD> t_ad(t_d.begin(), t_d.end());

  ode_system sys(indep[2]);
  jacobian   jac(indep[2]);
  rosenbrock4_controller< rosenbrock4<AD> > stepper(abstol, reltol);

  AD dt = (t_ad.back() - t_ad.front()) / AD(100.0);

  int event_count = 0;
  std::vector<AD> result_times; result_times.reserve(t_ad.size() + 16);
  std::vector<AD> y;            y.reserve(2 * (t_ad.size() + 16));
  observer obs(result_times, y, event_count);

  integrate_times(stepper, std::make_pair(sys, jac), x,
                  t_ad.begin(), t_ad.end(), dt, obs);

  const int n_out = static_cast<int>(result_times.size());
  if (n_out <= 0) Rf_error("Integration produced no output");

  // ADFun — beide Argumente std::vector<AD> → keine Typkonflikte
  CppAD::ADFun<double> f(indep, y);
  f.optimize();

  // Jacobian wrt (x0, v0, mu)
  CppAD::vector<double> xval(3);
  xval[0] = params[0]; xval[1] = params[1]; xval[2] = params[2];
  CppAD::vector<double> J = f.Jacobian(xval); // Länge: (2*n_out)*3

  // R-Matrix allozieren (Spalten-major), n_out x 21
  const int nrow = n_out, ncol = 21;
  SEXP ans = PROTECT(Rf_allocMatrix(REALSXP, nrow, ncol));
  double* out = REAL(ans);
  auto IDX = [nrow](int r, int c){ return r + c * nrow; };

  // time, x, v
  for (int i = 0; i < n_out; ++i) {
    out[IDX(i,0)] = CppAD::Value(result_times[i]);
    out[IDX(i,1)] = CppAD::Value(y[2*i + 0]);
    out[IDX(i,2)] = CppAD::Value(y[2*i + 1]);
  }
  // Jacobian
  for (int i = 0; i < n_out; ++i) {
    out[IDX(i,3)] = J[(2*i)*3 + 0]; // x.x
    out[IDX(i,4)] = J[(2*i)*3 + 1]; // x.v
    out[IDX(i,5)] = J[(2*i)*3 + 2]; // x.mu
    out[IDX(i,6)] = J[(2*i+1)*3 + 0]; // v.x
    out[IDX(i,7)] = J[(2*i+1)*3 + 1]; // v.v
    out[IDX(i,8)] = J[(2*i+1)*3 + 2]; // v.mu
  }

  // Sparse-Hessian (oberes Dreieck) direkt in die Matrix
  CppAD::vector<size_t> row, col; make_upper_triangle_indices(3, row, col); // 6
  CppAD::vector<double> w(f.Range()), h( row.size() );
  CppAD::vector<bool> p(9); for (size_t i=0;i<p.size();++i) p[i] = true;
  CppAD::sparse_hessian_work work;

  for (int i = 0; i < n_out; ++i) {
    // x(t_i)
    for (size_t k=0;k<w.size();++k) w[k]=0.0; w[2*i]=1.0;
    f.SparseHessian(xval, w, p, row, col, h, work);
    out[IDX(i, 9)] = h[0];  out[IDX(i,10)] = h[1];  out[IDX(i,11)] = h[2];
    out[IDX(i,12)] = h[3];  out[IDX(i,13)] = h[4];  out[IDX(i,14)] = h[5];

    // v(t_i)
    for (size_t k=0;k<w.size();++k) w[k]=0.0; w[2*i+1]=1.0;
    f.SparseHessian(xval, w, p, row, col, h, work);
    out[IDX(i,15)] = h[0];  out[IDX(i,16)] = h[1];  out[IDX(i,17)] = h[2];
    out[IDX(i,18)] = h[3];  out[IDX(i,19)] = h[4];  out[IDX(i,20)] = h[5];
  }

  // Spaltennamen
  SEXP coln = PROTECT(Rf_allocVector(STRSXP, ncol));
  const char* names[] = {
    "time","x","v","x.x","x.v","x.mu","v.x","v.v","v.mu",
    "xx.xx","xx.xv","xx.xmu","xx.vv","xx.vmu","xx.mumu",
    "vv.xx","vv.xv","vv.xmu","vv.vv","vv.vmu","vv.mumu"
  };
  for (int j=0;j<ncol;++j) SET_STRING_ELT(coln, j, Rf_mkChar(names[j]));
  SEXP dimn = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(dimn, 0, R_NilValue);
  SET_VECTOR_ELT(dimn, 1, coln);
  Rf_setAttrib(ans, R_DimNamesSymbol, dimn);

  // Wir haben 3x PROTECT benutzt: ans, coln, dimn
  UNPROTECT(3);
  return ans;
}
