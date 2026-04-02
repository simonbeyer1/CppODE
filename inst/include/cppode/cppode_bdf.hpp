/*
 CppODE BDF Stepper — CVODE-style Nordsieck implementation
 ==========================================================

 Variable-order (1–5), variable-step BDF stepper using the Nordsieck
 history array representation, following the CVODE/VODE algorithm.

 Architecture:
 - lu_W<Value, is_sparse> member handles all LU operations
 - bdf_newton_solve() free function handles Newton iteration
 - BDF stepper orchestrates the CVODE pipeline

 AD handling is fully transparent: lu_W::solve() dispatches to IFT
 internally, WRMS norms in newton include derivative components.

 References:
 - CVODE/VODE (Brown, Byrne & Hindmarsh, 1989)
 - SUNDIALS source: sundials/src/cvode/cvode.c

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_BDF_HPP_INCLUDED
#define CPPODE_BDF_HPP_INCLUDED

#include <cstddef>
#include <array>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <type_traits>
#include <cassert>

#include <cppode/cppode_odeint_compat.hpp>
#include <cppode/cppode_lu.hpp>
#include <cppode/cppode_newton.hpp>
#include <cppode/cppode_stepper_traits.hpp>
#include <cppode/cppode_profiler.hpp>

namespace cppode {

// ============================================================================
//  Scalar value extraction (AD-compatible)
// ============================================================================

namespace bdf_detail {

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v) { return static_cast<double>(v); }

template<class T, unsigned int N>
inline double scalar_value(const fadbad::F<T,N>& v) {
  return scalar_value(const_cast<fadbad::F<T,N>&>(v).x());
}

} // namespace bdf_detail

// ============================================================================
//  CVODE-style constants
// ============================================================================

namespace bdf_constants {

// --- CVODE constants (sundials/src/cvode/cvode.c) ---
// --- CVODE constants (sundials/src/cvode/cvode.c) ---
static constexpr double CORTES  = 0.1;      // Newton convergence rate threshold (tq4 = CORTES/tq[2])
static constexpr double CRDOWN  = 0.3;      // crate damping: crate = max(CRDOWN*crate, del/delp)
static constexpr double RDIV    = 2.0;      // divergence ratio: fail Newton if del > RDIV*delp (m>=2)
static constexpr double DGMAX   = 0.3;      // |gamrat-1| threshold for forcing LU refactorization
static constexpr double THRESH  = 1.5;      // min eta to bother changing h (eta < THRESH → eta=1)
static constexpr double ADDON   = 1e-6;     // small constant in eta denominator to avoid division by zero
static constexpr double BIAS1   = 6.0;      // safety factor in etaqm1 (order decrease candidate)
static constexpr double BIAS2   = 6.0;      // safety factor in etaq   (current order candidate)
static constexpr double BIAS3   = 10.0;     // safety factor in etaqp1 (order increase candidate)
static constexpr double ETAMX1  = 10000.0;  // max eta on first step (compensate small initial h)
static constexpr double ETAMX2  = 10.0;     // max eta for steps 2..SMALL_NST
static constexpr double ETAMX3  = 10.0;     // max eta after SMALL_NST steps
static constexpr double ETAMXF  = 0.2;      // max eta after SMALL_NEF error test failures
static constexpr double ETAMIN  = 0.1;      // min eta (floor for step size reduction)
static constexpr double ETACF   = 0.25;     // eta on Newton convergence failure
static constexpr double ONEPSM  = 1.000001; // 1 + epsilon for floating-point guard
static constexpr int    SMALL_NST = 10;     // steps before switching etamax from ETAMX2 to ETAMX3
static constexpr int    NLS_MAXCOR = 3;     // max Newton corrector iterations per step
static constexpr int    MSBP    = 20;       // max steps between periodic LU setup calls
static constexpr int    MXNEF   = 7;        // max total error test failures before giving up
static constexpr int    MXNEF1  = 3;        // error test failures before forcing order reduction
static constexpr int    SMALL_NEF = 2;      // error test failures before clamping eta to ETAMXF
static constexpr int    MXNCF   = 10;       // max Newton convergence failures before giving up
static constexpr int    LONG_WAIT = 10;     // qwait after order-1 restart (hold order that many steps)

} // namespace bdf_constants

// ============================================================================
//  bdf_stepper<Value, JacobianPattern, Resizer>
// ============================================================================

template<
  class Value,
  class JacobianPattern = cppode::dense_lu_tag,
                                  class Resizer = initially_resizer
>
class bdf_stepper
{
public:

  typedef void is_bdf_stepper_tag;

  typedef Value                            value_type;
  typedef std::vector<value_type>          state_type;
  typedef state_type                       deriv_type;
  typedef ad_lu::scalar_type_t<Value>      time_type;
  typedef dense_matrix<value_type>         matrix_type;
  typedef csc_matrix<value_type>           sparse_matrix_type;
  typedef Resizer                          resizer_type;

  typedef state_wrapper<state_type>        wrapped_state_type;
  typedef state_wrapper<deriv_type>        wrapped_deriv_type;
  typedef state_wrapper<matrix_type>       wrapped_matrix_type;

  typedef cppode::stepper_tag              stepper_category;
  typedef bdf_stepper<Value, JacobianPattern, Resizer> stepper_type;

  static constexpr bool is_sparse = is_sparse_tag<JacobianPattern>::value;
  static constexpr int max_order = 5;
  static constexpr int L_MAX = max_order + 1;

  // LU solver type
  using lu_type = lu_W<Value, is_sparse>;

  // ====================================================================
  //  Constructor
  // ====================================================================

  bdf_stepper()
    : m_q(1), m_L(2)
    , m_qwait(2)
    , m_nst(0)
    , m_nscon(0)
    , m_max_newton_iter(bdf_constants::NLS_MAXCOR)
    , m_newton_converged(false)
    , m_initialized(false)
    , m_n_fevals(0)
    , m_n_jevals(0)
    , m_hscale(0)
    , m_eta(1.0)
    , m_etamax(bdf_constants::ETAMX1)
    , m_gamma(0)
    , m_gammap(0)
    , m_gamrat(1.0)
    , m_crate(1.0)
    , m_acnrm(0)
    , m_nstlp(0)
    , m_saved_tq5(0)
    , m_tn_current(0)
  {
    m_l.fill(time_type(0));
    m_tq.fill(0.0);
    m_tau.fill(time_type(0));
  }

  // ====================================================================
  //  convfail enum — mirrors CVODE's convfail parameter to lsetup
  //
  //  CV_NO_FAILURES:  no Newton failures (first call or error-test retry)
  //                   → lsetup may reuse cached Jacobian if gamma is close
  //  CV_FAIL_BAD_J:   Newton failed with stale Jacobian → must recompute J
  //  CV_FAIL_OTHER:   Newton failed after fresh J or other failure
  // ====================================================================

  enum class convfail_t {
    no_failures,   // CVODE: CV_NO_FAILURES
    fail_bad_j,    // CVODE: CV_FAIL_BAD_J
    fail_other     // CVODE: CV_FAIL_OTHER
  };

  // ====================================================================
  //  do_step: perform one BDF step
  //
  //  Delegates Newton iteration to bdf_newton_solve().
  // ====================================================================

  template<class System, class TimeArg>
  void do_step(
      System system,
      const state_type& x,
      TimeArg t,
      state_type& x_out,
      TimeArg dt,
      state_type& xerr,
      bool force_setup = false,
      convfail_t convfail = convfail_t::no_failures)
  {
    using bdf_detail::scalar_value;
    // Convert incoming time arguments to scalar time_type
    time_type t_s  = static_cast<time_type>(scalar_value(t));
    time_type dt_s = static_cast<time_type>(scalar_value(dt));

    typedef typename unwrap_reference<System>::type system_type;
    typedef typename unwrap_reference<
      typename system_type::first_type>::type deriv_func_type;
    typedef typename unwrap_reference<
      typename system_type::second_type>::type jacobi_func_type;

    system_type&      sys         = system;
    deriv_func_type&  deriv_func  = sys.first;
    jacobi_func_type& jacobi_func = sys.second;

    const size_t n = x.size();

    // --- Resize ---
    resize_impl(x);

    // ================================================================
    //  1. Nordsieck rescale
    // ================================================================
    { auto _tp = m_prof.timer(prof_cat::nordsieck);
      if (m_nst > 0) {
        if (std::abs(dt_s - m_hscale) > 1e-14 * std::max(1.0, std::abs(m_hscale))) {
          m_eta = dt_s / m_hscale;
          cvRescale(n);
        }
        m_h = m_hscale;
      } else {
        m_h = dt_s;
        if (m_hscale == 0.0) m_hscale = dt_s;
      }
    }

    // ================================================================
    //  2. Predict
    // ================================================================
    { auto _tp = m_prof.timer(prof_cat::nordsieck); cvPredict(n); }

    // ================================================================
    //  3. BDF coefficients
    // ================================================================
    cvSetBDF();

    time_type rl1 = time_type(1.0) / m_l[1];
    m_gamma = m_h * rl1;
    if (m_nst == 0) m_gammap = m_gamma;
    m_gamrat = (m_nst > 0) ? m_gamma / m_gammap : 1.0;

    // ================================================================
    //  4+5. Newton solve with Jacobian retry (CVODE: cvNlsNewton)
    // ================================================================
    {
      time_type inv_gamma_dt = time_type(1.0) / m_gamma;
      time_type t_new_s = t_s + m_h;

      bool callSetup = force_setup
      || !m_lu.has_valid_jacobian()
        || (m_nst == 0)
        || (m_nst >= m_nstlp + bdf_constants::MSBP)
        || (std::abs(m_gamrat - 1.0) > bdf_constants::DGMAX);

        // --- Track which trigger fired (diagnostics) ---
        if (callSetup) {
          ++m_n_setup_total;
          if (force_setup)                                       ++m_n_setup_force;
          else if (!m_lu.has_valid_jacobian())                   ++m_n_setup_nojac;
          else if (m_nst == 0)                                   ++m_n_setup_init;
          else if (m_nst >= m_nstlp + bdf_constants::MSBP)      ++m_n_setup_msbp;
          else if (std::abs(m_gamrat - 1.0) > bdf_constants::DGMAX) ++m_n_setup_dgmax;
        }

#ifdef CPPODE_STEP_TRACE
        std::fprintf(stderr, "STEP %4d: q=%d h=%.6e gamma=%.6e gamrat=%.6f setup=%d",
                     m_nst, m_q, static_cast<double>(bdf_detail::scalar_value(m_h)),
                     static_cast<double>(bdf_detail::scalar_value(m_gamma)),
                     m_gamrat, callSetup ? 1 : 0);
        if (callSetup) {
          const char* reason = force_setup ? "force" :
          !m_lu.has_valid_jacobian() ? "no_jac" :
          (m_nst == 0) ? "init" :
          (m_nst >= m_nstlp + bdf_constants::MSBP) ? "MSBP" :
          (std::abs(m_gamrat - 1.0) > bdf_constants::DGMAX) ? "DGMAX" : "?";
          std::fprintf(stderr, " [%s]", reason);
        }
        std::fprintf(stderr, "\n");
#endif

        bool jcur = false;
        convfail_t nls_convfail = convfail;

        for (int nls_attempt = 0; nls_attempt < 2; ++nls_attempt) {

          if (callSetup) {
            bool need_new_jacobian =
              (nls_convfail != convfail_t::no_failures)
            || !m_lu.has_valid_jacobian()
            || (m_nst == 0);

            if (need_new_jacobian) {
              { auto _tp = m_prof.timer(prof_cat::jac_eval);
                m_lu.call_jacobian(jacobi_func, m_zn[0].m_v, t_new_s); }
              ++m_n_jevals;
              m_lu.cache_jacobian(n);
              { auto _tp = m_prof.timer(prof_cat::lu_factor);
                m_lu.factorize_W(n, inv_gamma_dt); }
            } else {
              auto _tp = m_prof.timer(prof_cat::lu_factor);
              if constexpr (is_sparse) {
                m_lu.refactorize_W_gamma_only(n, inv_gamma_dt);
              } else {
                m_lu.refactorize_W_from_cache(n, inv_gamma_dt);
              }
            }
            m_lu.set_jacobian_valid();
            m_lu.set_lu_valid(m_gamma);
            callSetup = false;
            jcur = need_new_jacobian;
            m_gammap = m_gamma;
            m_gamrat = 1.0;
            m_crate = 1.0;
            m_nstlp = m_nst;
          }
          // No else: when callSetup is false, CVODE reuses the stale W
          // factorized at gammap.  The Newton residual uses the CURRENT
          // gamma, so convergence is unaffected as long as gamrat is
          // within DGMAX (which triggers callSetup above).  This avoids
          // expensive refactorizations on every step.

          // Newton iteration (CVODE: cvNewtonIteration)
          auto result = bdf_newton_solve(
            m_lu,
            deriv_func,
            m_zn[0].m_v,
            m_zn[1].m_v,
            rl1,
            m_gamma,
            t_new_s,
            m_tq[4],
                m_atol, m_rtol,
                m_max_newton_iter,
                m_acor.m_v,
                m_y.m_v,
                m_tempv.m_v,
                m_ftemp.m_v,
                m_crate,
                m_gamrat,
                m_prof
          );

          m_acnrm = result.acnrm;
          m_n_fevals += result.n_fevals;

          if (result.converged) {
            m_newton_converged = true;
            break;
          }

          if (!jcur) {
            callSetup = true;
            nls_convfail = convfail_t::fail_bad_j;
            ++m_n_setup_retry;
            continue;
          }

          m_newton_converged = false;
          break;
        }
    }

    // ================================================================
    //  6. Output
    // ================================================================
    { auto _tp = m_prof.timer(prof_cat::step_overhead);
      if (m_newton_converged) {
        for (size_t i = 0; i < n; ++i) {
          x_out[i] = m_y.m_v[i];
          xerr[i] = m_acor.m_v[i];
        }
      } else {
        cvRestore(n);
        for (size_t i = 0; i < n; ++i) {
          x_out[i] = x[i];
          xerr[i] = value_type(0);
        }
      }
    }
  }

  // ====================================================================
  //  Error norm: dsm = acnrm * tq[2]
  // ====================================================================

  double error_norm() const { return m_acnrm * m_tq[2]; }

  // ====================================================================
  //  Order selection (eta computations)
  // ====================================================================

  double compute_etaqm1() const
  {
    auto _tp = m_prof.timer(prof_cat::error_norm);
    if (m_q <= 1) return 0.0;
    double ddn = newton_detail::wrms_norm(
      m_zn[m_q].m_v, m_zn[0].m_v, m_zn[0].m_v.size(), m_atol, m_rtol) * m_tq[1];
    return 1.0 / (std::pow(bdf_constants::BIAS1 * ddn, 1.0 / m_q) + bdf_constants::ADDON);
  }

  double compute_etaq(double dsm) const
  {
    return 1.0 / (std::pow(bdf_constants::BIAS2 * dsm, 1.0 / m_L) + bdf_constants::ADDON);
  }

  double compute_etaqp1() const
  {
    auto _tp = m_prof.timer(prof_cat::error_norm);
    using bdf_detail::scalar_value;
    if (m_q >= max_order) return 0.0;
    if (m_saved_tq5 == 0.0) return 0.0;

    double cquot = (m_tq[5] / m_saved_tq5)
      * std::pow(scalar_value(m_h) / scalar_value(m_tau[2]), m_L);

    const size_t n = m_zn[0].m_v.size();
    double sumsq = 0.0;
    size_t N_eff = 0;
    for (size_t i = 0; i < n; ++i) {
      double vi = scalar_value(m_acor.m_v[i])
      - cquot * scalar_value(m_zn[max_order].m_v[i]);
      double yi = std::abs(scalar_value(m_zn[0].m_v[i]));
      double r = std::abs(vi) / (m_atol + m_rtol * yi);
      sumsq += r * r;
      ++N_eff;

      if constexpr (ad_lu::is_ad<value_type>::value) {
        auto& acor_ad = const_cast<value_type&>(m_acor.m_v[i]);
        auto& zqmax_ad = const_cast<value_type&>(m_zn[max_order].m_v[i]);
        auto& y0_ad = const_cast<value_type&>(m_zn[0].m_v[i]);
        unsigned nd = acor_ad.size();
        for (unsigned j = 0; j < nd; ++j) {
          double vd = scalar_value(acor_ad.d(j))
          - cquot * scalar_value(zqmax_ad.d(j));
          double yd = std::abs(scalar_value(y0_ad.d(j)));
          double rd = std::abs(vd) / (m_atol + m_rtol * yd);
          sumsq += rd * rd;
          ++N_eff;
        }
      }
    }
    double dup = (N_eff > 0) ? std::sqrt(sumsq / N_eff) : 0.0;
    dup *= m_tq[3];
    return 1.0 / (std::pow(bdf_constants::BIAS3 * dup, 1.0 / (m_L + 1)) + bdf_constants::ADDON);
  }

  // ====================================================================
  //  complete_step, restore, rescale, set_order_for_next_step
  // ====================================================================

  void complete_step()
  {
    auto _tp = m_prof.timer(prof_cat::nordsieck);
    const size_t n = m_zn[0].m_v.size();
    ++m_nst;
    ++m_nscon;

    // Track steps at each order
    if (m_q >= 1 && m_q <= max_order)
      ++m_steps_at_order[m_q];

    for (int i = m_q; i >= 2; --i)
      m_tau[i] = m_tau[i - 1];
    if (m_q == 1 && m_nst > 1) m_tau[2] = m_tau[1];
    m_tau[1] = m_h;

    for (int j = 0; j <= m_q; ++j)
      vec_axpy(m_zn[j].m_v, m_l[j], m_acor.m_v);

    m_qwait--;
    if (m_qwait == 1 && m_q != max_order) {
      for (size_t i = 0; i < n; ++i)
        m_zn[max_order].m_v[i] = m_acor.m_v[i];
      m_saved_tq5 = m_tq[5];
    }
  }

  void restore()
  {
    auto _tp = m_prof.timer(prof_cat::nordsieck);
    cvRestore(m_zn[0].m_v.size());
  }

  void rescale(time_type eta)
  {
    auto _tp = m_prof.timer(prof_cat::nordsieck);
    m_eta = eta;
    cvRescale(m_zn[0].m_v.size());
  }

  void set_order_for_next_step(int new_q)
  {
    if (new_q == m_q + 1) {
      ++m_n_order_up;
      cvIncreaseBDF();
      m_q = new_q;
      m_L = new_q + 1;
    } else if (new_q == m_q - 1) {
      ++m_n_order_down;
      cvDecreaseBDF();
      m_q = new_q;
      m_L = new_q + 1;
    }
    m_nscon = 0;
  }

  // ====================================================================
  //  Initialize
  // ====================================================================

  template<class StateIn, class DerivIn, class TimeArg>
  void initialize(const StateIn& x0, TimeArg t0,
                  const DerivIn& f0, TimeArg dt)
  {
    using bdf_detail::scalar_value;
    time_type t0_s = static_cast<time_type>(scalar_value(t0));
    time_type dt_s = static_cast<time_type>(scalar_value(dt));
    const size_t n = x0.size();
    resize_impl(x0);

    for (size_t i = 0; i < n; ++i) {
      m_zn[0].m_v[i] = x0[i];
      m_zn[1].m_v[i] = dt_s * f0[i];
    }
    for (int j = 2; j <= max_order + 1; ++j)
      for (size_t i = 0; i < n; ++i)
        m_zn[j].m_v[i] = value_type(0);

    m_q = 1; m_L = 2; m_qwait = 2;
    m_nst = 0; m_nscon = 0;
    m_h = dt_s; m_hscale = dt_s;
    m_eta = time_type(1);
    m_etamax = bdf_constants::ETAMX1;
    m_gammap = time_type(0);
    m_gamrat = 1.0; m_crate = 1.0;
    m_acnrm = 0; m_nstlp = 0;
    m_saved_tq5 = 0; m_tn_current = t0_s;

    m_tau.fill(time_type(0));
    m_tau[1] = dt_s;
    m_initialized = true;
    m_lu.invalidate();
  }

  template<class TimeArg>
  void restart_from_order1(const state_type& x, TimeArg t, TimeArg dt)
  {
    const size_t n = x.size();
    m_q = 1; m_L = 2; m_qwait = 2; m_nscon = 0;
    for (size_t i = 0; i < n; ++i)
      m_zn[0].m_v[i] = x[i];
    m_h = static_cast<time_type>(bdf_detail::scalar_value(dt)); m_hscale = m_h;
    m_eta = time_type(1);
    m_etamax = bdf_constants::ETAMX1;
    m_gammap = time_type(0);
    m_gamrat = 1.0; m_crate = 1.0;
    m_saved_tq5 = 0; m_tn_current = static_cast<time_type>(bdf_detail::scalar_value(t));
    m_lu.invalidate();
  }

  // ====================================================================
  //  Dense output
  // ====================================================================

  void prepare_dense_output()
  {
    auto _tp = m_prof.timer(prof_cat::dense_snapshot);
    const size_t n = m_zn[0].m_v.size();
    for (int j = 0; j <= m_q; ++j) {
      if (m_zn_dense[j].m_v.size() != n)
        m_zn_dense[j].m_v.resize(n);
      m_zn_dense[j].m_v = m_zn[j].m_v;
    }
    m_q_dense = m_q;
    m_h_dense = m_h;
    m_tn_dense = m_tn_current;
    m_dense_valid = true;
  }

  // ====================================================================
  //  Accessors
  // ====================================================================

  int current_order() const { return m_q; }
  int steps_at_current_order() const { return m_nscon; }
  bool newton_converged() const { return m_newton_converged; }
  bool is_initialized() const { return m_initialized; }
  double acnrm() const { return m_acnrm; }
  const state_type& acor() const { return m_acor.m_v; }
  const std::array<double, 6>& tq() const { return m_tq; }
  int qwait() const { return m_qwait; }
  int nst() const { return m_nst; }
  double etamax() const { return m_etamax; }
  void set_etamax(double em) { m_etamax = em; }
  time_type h() const { return m_h; }
  time_type hscale() const { return m_hscale; }
  template<class TimeArg> void set_hscale(TimeArg hs) { m_hscale = static_cast<time_type>(bdf_detail::scalar_value(hs)); }
  double saved_tq5() const { return m_saved_tq5; }
  const state_type& zn(int j) const { return m_zn[j].m_v; }
  int current_L() const { return m_L; }
  const std::array<time_type, max_order + 1>& tau() const { return m_tau; }

  template<class TimeArg> void set_tn_current(TimeArg tn) { m_tn_current = static_cast<time_type>(bdf_detail::scalar_value(tn)); }
  void set_tolerances(double atol, double rtol) { m_atol = atol; m_rtol = rtol; }
  void set_qwait(int qw) { m_qwait = qw; }

  int n_fevals() const { return m_n_fevals; }
  int n_jevals() const { return m_n_jevals; }
  int n_setups() const { return m_n_setup_total; }
  void reset_counters() { m_n_fevals = 0; m_n_jevals = 0; }

  void report_setup_triggers() const {
    std::fprintf(stderr, "\n=== BDF setup trigger breakdown (total = %d) ===\n", m_n_setup_total);
    std::fprintf(stderr, "  force_setup (err/conv fail): %d\n", m_n_setup_force);
    std::fprintf(stderr, "  !has_valid_jacobian:         %d\n", m_n_setup_nojac);
    std::fprintf(stderr, "  nst == 0 (initial):          %d\n", m_n_setup_init);
    std::fprintf(stderr, "  MSBP periodic (>%d steps):   %d\n", bdf_constants::MSBP, m_n_setup_msbp);
    std::fprintf(stderr, "  DGMAX (|gamrat-1| > %.2f):   %d\n", bdf_constants::DGMAX, m_n_setup_dgmax);
    std::fprintf(stderr, "  Newton retry (stale J):      %d\n", m_n_setup_retry);
    std::fprintf(stderr, "================================================\n");

    std::fprintf(stderr, "\n=== BDF order change statistics ===\n");
    std::fprintf(stderr, "  Order increases (q→q+1):  %d\n", m_n_order_up);
    std::fprintf(stderr, "  Order decreases (q→q-1):  %d\n", m_n_order_down);
    std::fprintf(stderr, "  Steps per order:\n");
    for (int q = 1; q <= max_order; ++q) {
      if (m_steps_at_order[q] > 0)
        std::fprintf(stderr, "    q=%d: %d steps\n", q, m_steps_at_order[q]);
    }
    std::fprintf(stderr, "===================================\n\n");
  }

  int max_achievable_order() const { return max_order; }
  void increment_step_counter() { }
  template<class TimeArg> void push_history(const state_type&, TimeArg, TimeArg) { }
  value_type error_constant() const { return value_type(m_tq[2]); }
  value_type error_constant_km1() const { return value_type(m_tq[1]); }
  value_type error_constant_kp1() const { return value_type(m_tq[3]); }

  // LU access (for controller invalidation)
  void invalidate_lu() { m_lu.invalidate(); }
  bool has_valid_jacobian() const { return m_lu.has_valid_jacobian(); }
  bool has_valid_lu() const { return m_lu.has_valid_lu(); }
  double last_factorized_dt() const {
    return static_cast<double>(bdf_detail::scalar_value(m_lu.last_factorized_dt()));
  }

  // Save acor → zn[qmax] (CVODE: cvChooseEta when choosing q+1)
  void save_acor_to_zn_qmax()
  {
    const size_t n = m_zn[0].m_v.size();
    for (size_t i = 0; i < n; ++i)
      m_zn[max_order].m_v[i] = m_acor.m_v[i];
  }

  // Reload zn[1] = h * f(tn, zn[0]) for order-1 restart (CVODE: cvDoErrorTest)
  template<class DerivFunc>
  void reload_zn1_from_f(DerivFunc& deriv_func)
  {
    const size_t n = m_zn[0].m_v.size();
    deriv_func(m_zn[0].m_v, m_ftemp.m_v, m_tn_current);
    ++m_n_fevals;
    for (size_t i = 0; i < n; ++i)
      m_zn[1].m_v[i] = m_h * m_ftemp.m_v[i];
  }

  template<class StateType>
  void adjust_size(const StateType& x) { resize_impl(x); }

  // Dense output accessors
  bool has_dense_output() const { return m_dense_valid; }
  int dense_order() const { return m_q_dense; }
  time_type dense_h() const { return m_h_dense; }
  time_type dense_tn() const { return m_tn_dense; }
  const state_type& dense_zn(int j) const { return m_zn_dense[j].m_v; }

  // Direct LU access (for advanced use / testing)
  lu_type& lu() { return m_lu; }
  const lu_type& lu() const { return m_lu; }

private:

  // ====================================================================
  //  cvPredict / cvRestore / cvSetBDF / cvRescale
  // ====================================================================

  void cvPredict(size_t n)
  {
    for (int k = 1; k <= m_q; ++k)
      for (int j = m_q; j >= k; --j)
        vec_axpy(m_zn[j - 1].m_v, value_type(1), m_zn[j].m_v);
  }

  void cvRestore(size_t n)
  {
    for (int k = 1; k <= m_q; ++k)
      for (int j = m_q; j >= k; --j)
        vec_axpy(m_zn[j - 1].m_v, value_type(-1), m_zn[j].m_v);
  }

  void cvSetBDF()
  {
    m_l.fill(time_type(0));
    time_type xi_inv(1), xistar_inv(1);
    m_l[0] = m_l[1] = time_type(1);
    time_type alpha0 = time_type(-1);
    time_type alpha0_hat = time_type(-1);
    time_type hsum = m_h;

    if (m_q > 1) {
      for (int j = 2; j < m_q; ++j) {
        hsum += m_tau[j - 1];
        xi_inv = m_h / hsum;
        alpha0 -= time_type(1) / time_type(j);
        for (int i = j; i >= 1; --i)
          m_l[i] += m_l[i - 1] * xi_inv;
      }
      alpha0 -= time_type(1) / time_type(m_q);
      xistar_inv = -m_l[1] - alpha0;
      hsum += m_tau[m_q - 1];
      xi_inv = m_h / hsum;
      alpha0_hat = -m_l[1] - xi_inv;
      for (int i = m_q; i >= 1; --i)
        m_l[i] += m_l[i - 1] * xistar_inv;
    }

    cvSetTqBDF(hsum, static_cast<double>(alpha0), static_cast<double>(alpha0_hat),
               static_cast<double>(xi_inv), static_cast<double>(xistar_inv));
  }

  void cvSetTqBDF(time_type hsum, double alpha0, double alpha0_hat,
                  double xi_inv, double xistar_inv)
  {
    double A1 = 1.0 - alpha0_hat + alpha0;
    double A2 = 1.0 + m_q * A1;

    m_tq[2] = std::abs(A1 / (alpha0 * A2));
    m_tq[5] = std::abs(A2 * xistar_inv / (static_cast<double>(m_l[m_q]) * xi_inv));

    if (m_qwait == 1) {
      if (m_q > 1) {
        double C = xistar_inv / static_cast<double>(m_l[m_q]);
        double A3 = alpha0 + 1.0 / m_q;
        double A4 = alpha0_hat + xi_inv;
        double Cpinv = (1.0 - A4 + A3) / A3;
        m_tq[1] = std::abs(C * Cpinv);
      } else {
        m_tq[1] = 1.0;
      }
      hsum += m_tau[m_q];
      double xi_inv_qp1 = static_cast<double>(m_h) / static_cast<double>(hsum);
      double A5 = alpha0 - 1.0 / (m_q + 1);
      double A6 = alpha0_hat - xi_inv_qp1;
      double Cppinv = (1.0 - A6 + A5) / A2;
      m_tq[3] = std::abs(Cppinv / (xi_inv_qp1 * (m_q + 2) * A5));
    }
    m_tq[4] = bdf_constants::CORTES / m_tq[2];
  }

  void cvRescale(size_t n)
  {
    time_type factor = m_eta;
    for (int j = 1; j <= m_q; ++j) {
      vec_scale(m_zn[j].m_v, factor);
      factor *= m_eta;
    }
    m_h = m_hscale * m_eta;
    m_hscale = m_h;
    m_nscon = 0;
  }

  void cvIncreaseBDF()
  {
    const size_t n = m_zn[0].m_v.size();
    std::array<time_type, L_MAX + 1> ll;
    ll.fill(time_type(0));
    time_type alpha1(1), prod(1), xiold(1);
    double alpha0_inc = -1.0;
    ll[2] = time_type(1);
    time_type hsum_v = m_hscale;
    if (m_q > 1) {
      for (int j = 1; j < m_q; ++j) {
        hsum_v += m_tau[j + 1];
        time_type xi = hsum_v / m_hscale;
        prod *= xi;
        alpha0_inc -= 1.0 / (j + 1);
        alpha1 += time_type(1) / xi;
        for (int i = j + 2; i >= 2; --i) ll[i] = ll[i] * xiold + ll[i - 1];
        xiold = xi;
      }
    }
    double A1 = (-alpha0_inc - static_cast<double>(alpha1)) / static_cast<double>(prod);
    for (size_t i = 0; i < n; ++i)
      m_zn[m_L].m_v[i] = value_type(A1) * m_zn[max_order].m_v[i];
    for (int j = 2; j <= m_q; ++j)
      vec_axpy(m_zn[j].m_v, ll[j], m_zn[m_L].m_v);
  }

  void cvDecreaseBDF()
  {
    const size_t n = m_zn[0].m_v.size();
    std::array<time_type, L_MAX + 1> ll;
    ll.fill(time_type(0));
    ll[2] = time_type(1);
    time_type hsum_v(0);
    for (int j = 1; j <= m_q - 2; ++j) {
      hsum_v += m_tau[j];
      time_type xi = hsum_v / m_hscale;
      for (int i = j + 2; i >= 2; --i)
        ll[i] = ll[i] * xi + ll[i - 1];
    }
    for (int j = 2; j < m_q; ++j)
      vec_axpy(m_zn[j].m_v, -ll[j], m_zn[m_q].m_v);
  }

  // ====================================================================
  //  Resize
  // ====================================================================

  template<class StateIn>
  void resize_impl(const StateIn& x)
  {
    bool resized = false;
    for (int j = 0; j <= max_order + 1; ++j)
      resized |= adjust_size_by_resizeability(m_zn[j], x);
    for (int j = 0; j <= max_order; ++j)
      resized |= adjust_size_by_resizeability(m_zn_dense[j], x);
    resized |= adjust_size_by_resizeability(m_acor, x);
    resized |= adjust_size_by_resizeability(m_y, x);
    resized |= adjust_size_by_resizeability(m_tempv, x);
    resized |= adjust_size_by_resizeability(m_ftemp, x);
    resized |= m_lu.resize(x);
    if (resized) m_lu.invalidate();
  }

  // ====================================================================
  //  Members
  // ====================================================================

  lu_type m_lu;

  std::array<wrapped_state_type, max_order + 2> m_zn;
  std::array<wrapped_state_type, max_order + 1> m_zn_dense;
  int m_q_dense = 1;
  time_type m_h_dense = time_type(0);
  time_type m_tn_dense = time_type(0);
  bool m_dense_valid = false;

  wrapped_state_type m_acor, m_y, m_tempv;
  wrapped_deriv_type m_ftemp;

  std::array<time_type, L_MAX + 1> m_l;
  std::array<double, 6> m_tq;
  std::array<time_type, max_order + 1> m_tau;

  int m_q, m_L, m_qwait, m_nst, m_nscon;
  time_type m_h, m_hscale, m_eta;
  int m_max_newton_iter;
  bool m_newton_converged;
  double m_crate, m_acnrm;
  time_type m_gamma, m_gammap;
  double m_gamrat;
  int m_nstlp;
  double m_etamax, m_saved_tq5;
  double m_atol = 1e-6, m_rtol = 1e-6;
  time_type m_tn_current;
  bool m_initialized;
  int m_n_fevals, m_n_jevals;

  // Setup trigger diagnostics
  int m_n_setup_total = 0, m_n_setup_force = 0, m_n_setup_nojac = 0;
  int m_n_setup_init = 0, m_n_setup_msbp = 0, m_n_setup_dgmax = 0;
  int m_n_setup_retry = 0;

  // Order change diagnostics
  int m_n_order_up = 0, m_n_order_down = 0, m_n_order_same = 0;
  // Per-order step counts: m_steps_at_order[q] = # steps completed at order q
  std::array<int, max_order + 1> m_steps_at_order = {};

public:
  mutable cppode::profiler m_prof;
};

} // namespace cppode

#endif // CPPODE_BDF_HPP_INCLUDED
