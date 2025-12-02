#ifndef ROSENBROCK4_CONTROLLER_AD_HPP
#define ROSENBROCK4_CONTROLLER_AD_HPP

#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/controlled_step_result.hpp>
#include <boost/numeric/odeint/util/copy.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/detail/less_with_sign.hpp>

#include <fadbad++/fadiff.h>

#include <algorithm>
#include <cmath>
#include <functional>

/**
 * ============================================================================
 *  @file rosenbrock4_controller_ad.hpp
 *
 *  @brief Adaptive error controller for Boost.Odeint's Rosenbrock4 stepper,
 *         for FADBAD++ AD types (including nested F<F<T>>).
 *
 *  This controller performs complete error estimation across:
 *     - The base value of each state error component
 *     - All derivative components
 *     - All nested AD levels (F<F<T>>, F<F<F<T>>> …)
 *
 *  It is *guaranteed* never to read invalid FADBAD derivative slots:
 *     - `F<T>::size()` is respected
 *     - No derivative index beyond valid range is accessed
 *
 * ============================================================================
 */

namespace boost {
namespace numeric {
namespace odeint {

namespace detail {

/**
 * @brief Base case: extract scalar from arithmetic type.
 */
template<typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
get_scalar_value(const T& v)
{
  return static_cast<double>(v);
}

/**
 * @brief Recursive extraction of underlying scalar from fadbad::F<T>.
 *
 *  Uses `.x()` repeatedly until a plain value is reached.
 *  `.x()` is non-const, therefore const_cast is required (safe here).
 */
template<typename T>
inline double get_scalar_value(const fadbad::F<T>& v)
{
  return get_scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

/**
 * ============================================================================
 *  FULLY SAFE RECURSIVE DERIVATIVE SCANNER
 * ============================================================================
 *
 *  Computes:
 *      max( |value|, |d0|, |d1|, …, recursively )
 *
 * ============================================================================
 */

/** Arithmetic base case */
template<typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
max_abs_all(const T& v)
{
  return std::abs(v);
}

/** Recursive case for fadbad::F<T> */
template<typename T>
inline double max_abs_all(const fadbad::F<T>& v)
{
  auto& node = const_cast<fadbad::F<T>&>(v);

  // Start with base value
  double m = max_abs_all(node.x());

  // If derivative array is empty: STOP
  const int n = node.size();
  if (n <= 0)
    return m;

  // Only evaluate valid derivative entries
  for (int i = 0; i < n; ++i)
  {
    m = std::max(m, max_abs_all(node.d(i)));
  }

  return m;
}

} // namespace detail

/**
 * ============================================================================
 *  @class rosenbrock4_controller_ad
 *  @brief Fully AD-aware Rosenbrock4 step-size controller.
 * ============================================================================
 */
template<class Stepper>
class rosenbrock4_controller_ad
{
public:
  using stepper_type       = Stepper;
  using value_type         = typename stepper_type::value_type;
  using state_type         = typename stepper_type::state_type;
  using wrapped_state_type = typename stepper_type::wrapped_state_type;
  using time_type          = typename stepper_type::time_type;
  using deriv_type         = typename stepper_type::deriv_type;
  using wrapped_deriv_type = typename stepper_type::wrapped_deriv_type;
  using resizer_type       = typename stepper_type::resizer_type;
  using stepper_category   = controlled_stepper_tag;

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------

  rosenbrock4_controller_ad(
    double atol = 1e-6,
    double rtol = 1e-6,
    const stepper_type& stepper = stepper_type())
    : m_stepper(stepper)
    , m_xerr_resizer()
    , m_xnew_resizer()
    , m_xerr()
    , m_xnew()
    , m_atol(atol)
    , m_rtol(rtol)
    , m_max_dt(static_cast<time_type>(0))
    , m_first_step(true)
    , m_err_old(0.0)
    , m_dt_old(0.0)
    , m_last_rejected(false)
  {}

  rosenbrock4_controller_ad(
    double atol,
    double rtol,
    time_type maxdt,
    const stepper_type& stepper = stepper_type())
    : m_stepper(stepper)
    , m_xerr_resizer()
    , m_xnew_resizer()
    , m_xerr()
    , m_xnew()
    , m_atol(atol)
    , m_rtol(rtol)
    , m_max_dt(maxdt)
    , m_first_step(true)
    , m_err_old(0.0)
    , m_dt_old(0.0)
    , m_last_rejected(false)
  {}

  // -------------------------------------------------------------------------
  // Error computation (FULL AD-AWARE)
  // -------------------------------------------------------------------------

  /**
   * @brief Computes weighted RMS error across entire AD trees.
   *
   *  For every component:
   *     sk = atol + rtol * max(|x|, |xold|)
   *     term = max_abs_all(xerr[i]) / sk
   *
   *  where max_abs_all recursively inspects:
   *      value, d0, d1, …, nested derivatives
   */
  double error(const state_type& x,
               const state_type& xold,
               const state_type& xerr)
  {
    const std::size_t n = x.size();
    double sum = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
      double xi    = detail::get_scalar_value(x[i]);
      double xiold = detail::get_scalar_value(xold[i]);

      double sk = m_atol + m_rtol * std::max(std::abs(xi), std::abs(xiold));

      double term = detail::max_abs_all(xerr[i]) / sk;

      sum += term * term;
    }
    return std::sqrt(sum / static_cast<double>(n));
  }

  // -------------------------------------------------------------------------
  // Controlled step attempt
  // -------------------------------------------------------------------------

  template<class System>
  controlled_step_result try_step(System sys, state_type& x, time_type& t, time_type& dt)
  {
    m_xnew_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_m_xnew<state_type>(std::forward<decltype(arg)>(arg));
    });

    controlled_step_result res = try_step(sys, x, t, m_xnew.m_v, dt);

    if (res == success)
      copy(m_xnew.m_v, x);

    return res;
  }

  template<class System>
  controlled_step_result try_step(System sys,
                                  const state_type& x,
                                  time_type& t,
                                  state_type& xout,
                                  time_type& dt)
  {
    double dt_val = detail::get_scalar_value(dt);

    // Hard max_dt limit
    if (m_max_dt != static_cast<time_type>(0))
    {
      double maxdt_val = detail::get_scalar_value(m_max_dt);
      if (detail::less_with_sign(maxdt_val, dt_val, dt_val))
      {
        dt = m_max_dt;
        return fail;
      }
    }

    using std::pow;
    static constexpr double safe = 0.9;
    static constexpr double fac1 = 5.0;
    static constexpr double fac2 = 1.0 / 6.0;

    // Resize error buffer
    m_xerr_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_m_xerr<state_type>(std::forward<decltype(arg)>(arg));
    });

    // Perform Rosenbrock step
    m_stepper.do_step(sys, x, t, xout, dt, m_xerr.m_v);

    double err = error(xout, x, m_xerr.m_v);

    double fac = std::max(fac2, std::min(fac1, pow(err, 0.25) / safe));
    double dt_new = dt_val / fac;

    if (err <= 1.0)
    {
      // Accepted step
      if (!m_first_step)
      {
        double fac_pred = (m_dt_old / dt_val) *
          pow(err * err / m_err_old, 0.25) / safe;

        fac_pred = std::max(fac2, std::min(fac1, fac_pred));
        fac = std::max(fac, fac_pred);

        dt_new = dt_val / fac;
      }

      m_first_step = false;
      m_dt_old = dt_val;
      m_err_old = std::max(0.01, err);

      if (m_last_rejected)
      {
        dt_new = (dt_val >= 0.0
                    ? std::min(dt_new, dt_val)
                      : std::max(dt_new, dt_val));
      }

      t += dt;

      if (m_max_dt != static_cast<time_type>(0))
      {
        double maxdt_val = detail::get_scalar_value(m_max_dt);
        dt = time_type(std::abs(dt_new) < maxdt_val
                         ? dt_new
                         : (dt_val >= 0.0 ? maxdt_val : -maxdt_val));
      }
      else
      {
        dt = time_type(dt_new);
      }

      m_last_rejected = false;
      return success;
    }
    else
    {
      // Rejected step
      dt = time_type(dt_new);
      m_last_rejected = true;
      return fail;
    }
  }

  // -------------------------------------------------------------------------
  // Resize
  // -------------------------------------------------------------------------

  template<class StateType>
  void adjust_size(const StateType& x)
  {
    resize_m_xerr(x);
    resize_m_xnew(x);
  }

  stepper_type& stepper() { return m_stepper; }
  const stepper_type& stepper() const { return m_stepper; }

private:

  template<class StateIn>
  bool resize_m_xerr(const StateIn& x)
  {
    return adjust_size_by_resizeability(
      m_xerr, x, typename is_resizeable<state_type>::type());
  }

  template<class StateIn>
  bool resize_m_xnew(const StateIn& x)
  {
    return adjust_size_by_resizeability(
      m_xnew, x, typename is_resizeable<state_type>::type());
  }

private:
  stepper_type m_stepper;
  resizer_type m_xerr_resizer, m_xnew_resizer;
  wrapped_state_type m_xerr, m_xnew;

  double m_atol, m_rtol;
  time_type m_max_dt;

  bool   m_first_step;
  double m_err_old;
  double m_dt_old;
  bool   m_last_rejected;
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // ROSENBROCK4_CONTROLLER_AD_HPP
