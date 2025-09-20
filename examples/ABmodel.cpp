/** Minimal FADBAD++ + uBLAS ODE Beispiel **/
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fadiff.h>
#include <badiff.h>
#include <stiff_solver_fad.hpp>


using namespace boost::numeric::odeint;
namespace ublas = boost::numeric::ublas;
using AD = fadbad::F<double>;


// ODE-System
struct ode_system {
  ublas::vector<AD> params;
  explicit ode_system(const ublas::vector<AD>& p_) : params(p_) {}

  void operator()(const ublas::vector<AD>& x,
                ublas::vector<AD>& dxdt,
                const AD& t)
  {
    dxdt[0] = -(x[0] * x[0]) * params[2] * t;
    dxdt[1] = (x[0] * x[0]) * params[2] * t - params[3] * x[1];
  }
};

// Jacobian
struct jacobian {
  ublas::vector<AD> params;
  explicit jacobian(const ublas::vector<AD>& p_) : params(p_) {}

  void operator()(const ublas::vector<AD>& x,
                ublas::matrix<AD>& J,
                const AD& t,
                ublas::vector<AD>& dfdt)
  {
    J.resize(2,2,false);
    dfdt.resize(2,false);

    J(0,0) = -2 * x[0] * params[2] * t;
    J(0,1) = 0;

    J(1,0) =  2 * x[0] * params[2] * t;
    J(1,1) = -params[3];

    dfdt[0] = -(x[0] * x[0]) * params[2];
    dfdt[1] =  (x[0] * x[0]) * params[2];
  }
};


// Observer zum Speichern von Trajektorie
struct observer {
  std::vector<AD>& times;
  std::vector<AD>& y;

  explicit observer(std::vector<AD>& t, std::vector<AD>& y_)
    : times(t), y(y_) {}

  void operator()(const ublas::vector<AD>& x, const AD& t) {
    times.push_back(t);
    for (size_t i = 0; i < x.size(); ++i)
      y.push_back(x[i]);
  }
};

extern "C" SEXP solve(SEXP timesSEXP, SEXP paramsSEXP,
                     SEXP abstolSEXP, SEXP reltolSEXP,
                     SEXP maxprogressSEXP, SEXP maxstepsSEXP) {
  try {
    const int x_N = 2;
    const int p_N = 2;

    if (!Rf_isReal(timesSEXP) || Rf_length(timesSEXP) < 2)
      Rf_error("times must be numeric, length >= 2");
    if (!Rf_isReal(paramsSEXP) || Rf_length(paramsSEXP) != x_N + p_N)
      Rf_error("params must be numeric length %d", x_N + p_N);
    if (!Rf_isReal(abstolSEXP) || !Rf_isReal(reltolSEXP))
      Rf_error("abstol/reltol must be numeric scalars");

    const int T_N = Rf_length(timesSEXP);
    const double* times  = REAL(timesSEXP);
    const double* params = REAL(paramsSEXP);
    const double abstol  = REAL(abstolSEXP)[0];
    const double reltol  = REAL(reltolSEXP)[0];

    if (!Rf_isInteger(maxprogressSEXP) || Rf_length(maxprogressSEXP) != 1)
      Rf_error("maxprogress must be a single integer");
    if (!Rf_isInteger(maxstepsSEXP) || Rf_length(maxstepsSEXP) != 1)
      Rf_error("maxsteps must be a single integer");

    int tmp_progress = INTEGER(maxprogressSEXP)[0];
    int tmp_steps    = INTEGER(maxstepsSEXP)[0];
    if (tmp_progress <= 0) Rf_error("maxprogress must be > 0");
    if (tmp_steps    <= 0) Rf_error("maxsteps must be > 0");

    StepChecker checker(tmp_progress, tmp_steps);

    const int sens_xN = 2;
    const int sens_pN = 2;
    const int dom_N = sens_xN + sens_pN;
    const int sens_state_idx[2] = {0,1};
    const int sens_param_idx[2] = {0,1};
    const int state_sens_pos[2] = {0,1};
    const int param_sens_pos[2] = {0,1};
    std::vector<AD> indep(dom_N);

    for (int j = 0; j < sens_xN; ++j)
      indep[j] = params[sens_state_idx[j]];
    for (int j = 0; j < sens_pN; ++j)
      indep[sens_xN + j] = params[x_N + sens_param_idx[j]];

    for (int j = 0; j < dom_N; ++j)
      indep[j].diff(j, dom_N);

    ublas::vector<AD> x(x_N);
    for (int i = 0; i < x_N; ++i) {
      if (state_sens_pos[i] >= 0) x[i] = indep[state_sens_pos[i]];
      else x[i] = params[i];
    }

    ublas::vector<AD> full_params(x_N + p_N);
    for (int i = 0; i < x_N; ++i) full_params[i] = x[i];
    for (int j = 0; j < p_N; ++j) {
      if (param_sens_pos[j] >= 0)
        full_params[x_N + j] = indep[sens_xN + param_sens_pos[j]];
      else
        full_params[x_N + j] = params[x_N + j];
    }

    std::vector<AD> t_ad;
    for (int i = 0; i < T_N; ++i) t_ad.push_back(times[i]);

    ode_system sys(full_params);
    jacobian jac(full_params);
    rosenbrock4_controller_ad<rosenbrock4<AD>> stepper(abstol, reltol);
    AD dt = (t_ad.back() - t_ad.front()) * abstol;

    std::vector<AD> result_times;
    std::vector<AD> y;
    observer obs(result_times, y);
    integrate_times(stepper, std::make_pair(sys, jac),
                    x, t_ad.begin(), t_ad.end(), dt, obs, checker);

    const int n_out = static_cast<int>(result_times.size());
    if (n_out <= 0) Rf_error("Integration produced no output");

    const int ncol = 1 + x_N + x_N * dom_N;
    SEXP ans = PROTECT(Rf_allocMatrix(REALSXP, n_out, ncol));
    double* out = REAL(ans);
    auto IDX = [n_out](int r, int c){ return r + c * n_out; };

    for (int i = 0; i < n_out; ++i) {
      out[IDX(i, 0)] = result_times[i].x();
      for (int s = 0; s < x_N; ++s) {
        AD& xi = y[i * x_N + s];
        out[IDX(i, 1 + s)] = xi.x();
        int base_col = 1 + x_N + s * dom_N;
        for (int v = 0; v < dom_N; ++v) {
          out[IDX(i, base_col + v)] = xi.d(v);
        }
      }
    }

    // Spaltennamen
    SEXP coln = PROTECT(Rf_allocVector(STRSXP, ncol));
    int col = 0;
    SET_STRING_ELT(coln, col++, Rf_mkChar("time"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("A"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("B"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("A.A"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("A.B"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("A.k1"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("A.k2"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("B.A"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("B.B"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("B.k1"));
    SET_STRING_ELT(coln, col++, Rf_mkChar("B.k2"));

    SEXP dimn = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(dimn, 0, R_NilValue);
    SET_VECTOR_ELT(dimn, 1, coln);
    Rf_setAttrib(ans, R_DimNamesSymbol, dimn);

    UNPROTECT(3);
    return ans;

  } catch (const std::exception& e) {
    Rf_error("ODE solver failed: %s", e.what());
  } catch (...) {
    Rf_error("ODE solver failed: unknown C++ exception! Good Luck :P");
  }
}
