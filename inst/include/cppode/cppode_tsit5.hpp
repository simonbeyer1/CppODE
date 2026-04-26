/*
 Tsitouras 5(4) explicit Runge-Kutta stepper.

 A 7-stage, 5th-order explicit method with embedded 4th-order error
 estimator and FSAL (First Same As Last) property.  Widely used as
 the default non-stiff solver in DifferentialEquations.jl.

 Reference:
 Tsitouras, Ch. (2011). "Runge-Kutta pairs of order 5(4) satisfying
 only the first column simplifying assumption."  Computers &
 Mathematics with Applications, 62(2), 770-775.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_TSIT5_HPP
#define CPPODE_TSIT5_HPP

#include <cstddef>
#include <vector>
#include <cmath>
#include <type_traits>

#include <cppode/cppode_types.hpp>
#include <cppode/cppode_odeint_compat.hpp>
#include <cppode/cppode_ad_lu.hpp>       // for ad_lu::scalar_type_t
#include <cppode/cppode_dual_slab.hpp>
#include <cppode/cppode_profiler.hpp>

namespace cppode {

// ============================================================================
//  tsit5<Value, Resizer>
//
//  Explicit Runge-Kutta 5(4) method with FSAL.
//  No Jacobian / linear algebra needed.
// ============================================================================

template<
  class Value,
  class Resizer = initially_resizer
>
class tsit5
{
public:

  typedef Value                              value_type;
  typedef std::vector<value_type>            state_type;
  typedef state_type                         deriv_type;
  typedef ad_lu::scalar_type_t<value_type>   time_type;
  typedef Resizer                            resizer_type;
  typedef stepper_tag                        stepper_category;
  typedef unsigned short                     order_type;

  typedef state_wrapper<state_type>          wrapped_state_type;
  typedef state_wrapper<deriv_type>          wrapped_deriv_type;

  typedef tsit5<Value, Resizer>              stepper_type;

  static constexpr order_type stepper_order = 5;
  static constexpr order_type error_order   = 4;

  tsit5()
    : m_n_fevals(0)
    , m_fsal_valid(false)
  {}

  // Same move-only semantics as the multistep family: copying the slab
  // members would build a fresh tangent block while the dual elements
  // still point at the original — UB.
  tsit5(const tsit5&)            = delete;
  tsit5& operator=(const tsit5&) = delete;
  tsit5(tsit5&&)                 = default;
  tsit5& operator=(tsit5&&)      = default;

  order_type order() const { return stepper_order; }

  int n_fevals() const { return m_n_fevals; }
  int n_jevals() const { return 0; }
  int n_setups() const { return 0; }
  void reset_counters() { m_n_fevals = 0; }

  // No-op Jacobian/LU interface for controller compatibility
  void invalidate_lu() { m_fsal_valid = false; }
  bool has_valid_lu() const { return false; }
  double last_factorized_dt() const { return 0.0; }

  // ====================================================================
  //  prepare_sensitivities
  //
  //  Primes the SoA tangent slabs for the heap-AD path so the per-stage
  //  ET assignments hit the in-place reuse branch from the very first
  //  step instead of arena-allocating each m_k*[i].tan_ on first write.
  //  No-op for non-dynamic-dual value_type.
  // ====================================================================

  void prepare_sensitivities(unsigned n_sens)
  {
    m_n_sens = n_sens;
    if constexpr (detail::is_dynamic_dual<value_type>::value) {
      if (n_sens == 0) return;
      auto prime = [n_sens](auto& wrapped, auto& slab) {
        if (!wrapped.m_v.empty())
          slab.prime(wrapped.m_v,
                     static_cast<unsigned>(wrapped.m_v.size()), n_sens);
      };
      prime(m_k1,   m_k1_slab);
      prime(m_k2,   m_k2_slab);
      prime(m_k3,   m_k3_slab);
      prime(m_k4,   m_k4_slab);
      prime(m_k5,   m_k5_slab);
      prime(m_k6,   m_k6_slab);
      prime(m_k7,   m_k7_slab);
      prime(m_xtmp, m_xtmp_slab);
    }
  }

  unsigned n_sens() const noexcept { return m_n_sens; }

  // ====================================================================
  //  do_step (with error output)
  //
  //  Computes one Tsit5 step from (x, t) to (xout, t+dt) and fills
  //  xerr with the embedded error estimate.
  //
  //  The jacobian_hint parameter is accepted for API compatibility
  //  with the onestep_controller but is ignored.
  // ====================================================================

  template<class Sys, class TimeArg>
  void do_step(
      Sys& system,
      const state_type& x, TimeArg t,
      state_type& xout, TimeArg dt, state_type& xerr,
      jacobian_hint /*hint*/ = jacobian_hint::recompute_all)
  {
    auto _tp = m_prof.timer(prof_cat::step_overhead);
    auto& deriv_func = system.first;

    const size_t n = x.size();

    // --- Resize scratch vectors ---
    resize_if_needed(n);

    // --- Stage 1 (FSAL: reuse k7 from previous accepted step) ---
    if (m_fsal_valid && m_k1.m_v.size() == n) {
      // k1 already holds f(x, t) from the previous step's k7
    } else {
      deriv_func(x, m_k1.m_v, t);
      ++m_n_fevals;
    }

    // Use TimeArg (not time_type) so AD derivative components propagate
    // through the deriv_func t arguments below.
    const TimeArg h    = dt;
    // Stage AXPY alphas extracted to scalar — matches rosenbrock4
    // convention. AD-time propagation in stage assembly is dropped here
    // (the deriv_func calls below still receive a TimeArg-typed t so AD
    // tangents propagate where they matter).
    const double h_s = static_cast<double>(ad_lu::scalar_value(dt));

    // --- Stage 2:  xtmp = x + h*a21 * k1 ---
    vec_copy_with_slab(m_xtmp.m_v, m_xtmp_slab, x, m_x_in_unslabbed);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab,
                       h_s * a21, m_k1.m_v, m_k1_slab);
    deriv_func(m_xtmp.m_v, m_k2.m_v, t + c2 * h);
    ++m_n_fevals;

    // --- Stage 3:  xtmp = x + h*(a31*k1 + a32*k2) ---
    vec_copy_with_slab(m_xtmp.m_v, m_xtmp_slab, x, m_x_in_unslabbed);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a31, m_k1.m_v, m_k1_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a32, m_k2.m_v, m_k2_slab);
    deriv_func(m_xtmp.m_v, m_k3.m_v, t + c3 * h);
    ++m_n_fevals;

    // --- Stage 4 ---
    vec_copy_with_slab(m_xtmp.m_v, m_xtmp_slab, x, m_x_in_unslabbed);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a41, m_k1.m_v, m_k1_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a42, m_k2.m_v, m_k2_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a43, m_k3.m_v, m_k3_slab);
    deriv_func(m_xtmp.m_v, m_k4.m_v, t + c4 * h);
    ++m_n_fevals;

    // --- Stage 5 ---
    vec_copy_with_slab(m_xtmp.m_v, m_xtmp_slab, x, m_x_in_unslabbed);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a51, m_k1.m_v, m_k1_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a52, m_k2.m_v, m_k2_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a53, m_k3.m_v, m_k3_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a54, m_k4.m_v, m_k4_slab);
    deriv_func(m_xtmp.m_v, m_k5.m_v, t + c5 * h);
    ++m_n_fevals;

    // --- Stage 6 ---
    vec_copy_with_slab(m_xtmp.m_v, m_xtmp_slab, x, m_x_in_unslabbed);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a61, m_k1.m_v, m_k1_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a62, m_k2.m_v, m_k2_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a63, m_k3.m_v, m_k3_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a64, m_k4.m_v, m_k4_slab);
    vec_axpy_with_slab(m_xtmp.m_v, m_xtmp_slab, h_s * a65, m_k5.m_v, m_k5_slab);
    deriv_func(m_xtmp.m_v, m_k6.m_v, t + c6 * h);
    ++m_n_fevals;

    // --- Solution (5th order) and Stage 7 (FSAL) ---
    // xout is controller-owned (m_xnew) and not slab-bound — we use plain
    // vec_copy / vec_axpy here. Each axpy iteration is a single ScalarLeaf *
    // DualLeaf binop, which the ET path materialises as one fused N-element
    // tangent loop without nested template depth.
    vec_copy(xout, x);
    vec_axpy(xout, h_s * b1, m_k1.m_v);
    vec_axpy(xout, h_s * b2, m_k2.m_v);
    vec_axpy(xout, h_s * b3, m_k3.m_v);
    vec_axpy(xout, h_s * b4, m_k4.m_v);
    vec_axpy(xout, h_s * b5, m_k5.m_v);
    vec_axpy(xout, h_s * b6, m_k6.m_v);

    deriv_func(xout, m_k7.m_v, t + h);
    ++m_n_fevals;

    // --- Error estimate: xerr = h * sum(e_i * k_i)  (e_i = bhat_i - b_i) ---
    vec_zero(xerr);
    vec_axpy(xerr, h_s * e1, m_k1.m_v);
    vec_axpy(xerr, h_s * e2, m_k2.m_v);
    vec_axpy(xerr, h_s * e3, m_k3.m_v);
    vec_axpy(xerr, h_s * e4, m_k4.m_v);
    vec_axpy(xerr, h_s * e5, m_k5.m_v);
    vec_axpy(xerr, h_s * e6, m_k6.m_v);
    vec_axpy(xerr, h_s * e7, m_k7.m_v);

    // FSAL: k7 becomes k1 for the next step (if accepted)
    m_fsal_valid = false;  // will be set by prepare_dense_output / accept
  }

  // ====================================================================
  //  Dense output
  //
  //  The Tsit5 continuous extension uses the 7 FSAL stages to build
  //  a 4th-order interpolant.  Coefficients from Tsitouras (2011),
  //  Section 3.
  // ====================================================================

  void prepare_dense_output()
  {
    // After an accepted step, mark FSAL as valid:
    // k7 from this step becomes k1 for the next step.
    // We swap k7 → k1 so the next do_step picks it up.
    //
    // For the slab-aware heap path we must swap the slabs alongside the
    // vectors. The dual elements keep their tan_ pointers under std::swap,
    // so post-vector-swap m_k1.m_v[i].tan_ still points into m_k7_slab's
    // storage. By also swapping the slabs themselves (which transfers the
    // backing std::vector buffer pointer), m_k1_slab's storage now sits
    // exactly where the swapped duals' tan_ pointers expect it — keeping
    // vec_*_with_slab BLAS paths reading the correct memory.
    std::swap(m_k1.m_v, m_k7.m_v);
    if constexpr (detail::is_dynamic_dual<value_type>::value) {
      std::swap(m_k1_slab, m_k7_slab);
    }
    m_fsal_valid = true;

    // Dense output coefficients are computed lazily in calc_state
    // from the stored k1..k7 stages.  The stages must not be
    // overwritten until the next do_step call.
  }

  template<class TimeArg, class StateOut>
  void calc_state(
      TimeArg t, StateOut& x,
      const state_type& x_old, TimeArg t_old,
      const state_type& x_new, TimeArg t_new)
  {
    // Note: after prepare_dense_output, k1 and k7 have been swapped.
    // So m_k1 holds the OLD k7 (= f(x_new, t_new)) and m_k7 holds
    // the old k1 (= f(x_old, t_old)).  For the interpolant we need
    // f_old = old k1 = m_k7  and  f_new = old k7 = m_k1.

    const size_t n = x_old.size();
    // Use TimeArg (not time_type) so AD derivative components propagate.
    const TimeArg h = t_new - t_old;
    const TimeArg s = (t - t_old) / h;   // theta in [0, 1]

    // Hermite cubic interpolation using endpoint values and derivatives.
    // After prepare_dense_output(), m_k7 = old k1 = f(x_old, t_old)
    // and m_k1 = old k7 = f(x_new, t_new).

    const TimeArg s1 = TimeArg(1) - s;
    const TimeArg s2 = s * s;
    const TimeArg s1_2 = s1 * s1;

    for (size_t i = 0; i < n; ++i) {
      // Hermite basis functions:
      //   H00 = (1 + 2s)(1-s)^2 = 1 - 3s^2 + 2s^3
      //   H10 = s(1-s)^2         = s - 2s^2 + s^3
      //   H01 = s^2(3 - 2s)      = 3s^2 - 2s^3
      //   H11 = s^2(s - 1)       = s^3 - s^2
      value_type f_old = m_k7.m_v[i];   // after swap: old k1
      value_type f_new = m_k1.m_v[i];   // after swap: old k7

      x[i] = x_old[i] * (s1_2 * (TimeArg(1) + TimeArg(2) * s))
           + x_new[i] * (s2 * (TimeArg(3) - TimeArg(2) * s))
           + h * f_old * (s * s1_2)
           + h * f_new * (s2 * (s - TimeArg(1)));
    }
  }

  // ====================================================================
  //  Tsitouras 5(4) coefficients
  //
  //  From Table 1 and eq. (7)-(8) of Tsitouras (2011).
  // ====================================================================

private:

  // Nodes c_i
  static constexpr double c2 = 0.161;
  static constexpr double c3 = 0.327;
  static constexpr double c4 = 0.9;
  static constexpr double c5 = 0.9800255409045097;
  static constexpr double c6 = 1.0;

  // Runge-Kutta matrix a_ij (lower triangular, row by row)
  static constexpr double a21 = 0.161;

  static constexpr double a31 = -0.008480655492356989;
  static constexpr double a32 =  0.335480655492357;

  static constexpr double a41 =  2.8971530571054935;
  static constexpr double a42 = -6.359448489975075;
  static constexpr double a43 =  4.3622954328695815;

  static constexpr double a51 =  5.325864828439257;
  static constexpr double a52 = -11.748883564062828;
  static constexpr double a53 =  7.4955393428898365;
  static constexpr double a54 = -0.09249506636175525;

  static constexpr double a61 =  5.86145544294642;
  static constexpr double a62 = -12.92096931784711;
  static constexpr double a63 =  8.159367898576159;
  static constexpr double a64 = -0.071584973281401;
  static constexpr double a65 = -0.028269050394068383;

  // 5th-order weights b_i (= a_{7,i}, FSAL)
  static constexpr double b1 =  0.09646076681806523;
  static constexpr double b2 =  0.01;
  static constexpr double b3 =  0.4798896504144996;
  static constexpr double b4 =  1.379008574103742;
  static constexpr double b5 = -3.290069515436081;
  static constexpr double b6 =  2.324710524099774;
  // b7 = 0 (FSAL: k7 = f(x_new, t_new), weight absorbed)

  // Error coefficients e_i = bhat_i - b_i  (4th order minus 5th order)
  // These are the btilde values from Tsitouras (2011); sum(e_i) ≈ 0.
  static constexpr double e1 = -1.78001105222577727e-03;
  static constexpr double e2 = -8.16434459656747991e-04;
  static constexpr double e3 =  7.88087801026199353e-03;
  static constexpr double e4 = -1.44711007173262907e-01;
  static constexpr double e5 =  5.82357165452555670e-01;
  static constexpr double e6 = -4.58082105929186856e-01;
  static constexpr double e7 =  1.51515151515151519e-02;

  // ====================================================================
  //  Scratch storage
  // ====================================================================

  void resize_if_needed(size_t n)
  {
    bool resized = false;
    auto resize_one = [n, &resized](wrapped_deriv_type& w) {
      if (w.m_v.size() != n) { w.m_v.resize(n); resized = true; }
    };
    resize_one(m_k1);
    resize_one(m_k2);
    resize_one(m_k3);
    resize_one(m_k4);
    resize_one(m_k5);
    resize_one(m_k6);
    resize_one(m_k7);
    resize_one(m_xtmp);
    if (resized && m_n_sens != 0) prepare_sensitivities(m_n_sens);
  }

  wrapped_deriv_type m_k1, m_k2, m_k3, m_k4, m_k5, m_k6, m_k7;
  wrapped_deriv_type m_xtmp;   // scratch for stage evaluation

  // SoA tangent slabs for the dynamic-dual heap path.
  detail::tangent_slab<value_type> m_k1_slab, m_k2_slab, m_k3_slab,
                                   m_k4_slab, m_k5_slab, m_k6_slab,
                                   m_k7_slab;
  detail::tangent_slab<value_type> m_xtmp_slab;
  // Permanently-empty stub for the externally-owned input state x (controller
  // owns it; it is not slab-bound here). vec_*_with_slab sees primed=false
  // and falls through to the per-element loop. mutable so we can hand out
  // non-const refs to the helper signature without lying about constness.
  mutable detail::tangent_slab<value_type> m_x_in_unslabbed;
  unsigned m_n_sens = 0;

  int  m_n_fevals;
  bool m_fsal_valid;

public:
  mutable cppode::profiler m_prof;
};

} // namespace cppode

#endif // CPPODE_TSIT5_HPP
