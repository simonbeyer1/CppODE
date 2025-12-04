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

/**
 * @file rosenbrock4_controller_ad.hpp
 *
 * @brief Adaptive error controller for Boost.Odeint's Rosenbrock4 stepper,
 *        fully compatible with FADBAD++ AD types (including nested F<F<T>>).
 *
 * Features:
 *   - Recursive AD-safe error estimation across values and all derivatives.
 *   - Balanced adaptive step-size update (variant 3).
 *   - Full reset capabilities for event-driven integration (Strategy B).
 *   - Identical external behavior to Boost’s standard rosenbrock4_controller.
 */

namespace boost {
namespace numeric {
namespace odeint {

namespace detail {

/*------------------------------------------------------------------------------
 * Extract scalar values from arithmetic or FADBAD types
 *----------------------------------------------------------------------------*/

template<typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
get_scalar_value(const T& v)
{
  return static_cast<double>(v);
}

template<typename T>
inline double get_scalar_value(const fadbad::F<T>& v)
{
  return get_scalar_value(const_cast<fadbad::F<T>&>(v).x());
}

/*------------------------------------------------------------------------------
 * Recursively compute max absolute value of AD tree (value + derivatives)
 *----------------------------------------------------------------------------*/

template<typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
max_abs_all(const T& v)
{
  return std::abs(v);
}

template<typename T>
inline double max_abs_all(const fadbad::F<T>& v)
{
  auto& node = const_cast<fadbad::F<T>&>(v);

  double m = max_abs_all(node.x());
  int n = node.size();
  for (int i = 0; i < n; ++i)
    m = std::max(m, max_abs_all(node.d(i)));

  return m;
}

} // namespace detail



/*==============================================================================
 * Rosenbrock4 Adaptive Controller (AD-aware)
 *============================================================================*/

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

public:

  /*--------------------------------------------------------------------------
   * Constructor
   *--------------------------------------------------------------------------*/
  rosenbrock4_controller_ad(
    double atol = 1e-6,
    double rtol = 1e-6,
    const stepper_type& stepper = stepper_type())
    : m_stepper(stepper)
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
  , m_atol(atol)
  , m_rtol(rtol)
  , m_max_dt(maxdt)
  , m_first_step(true)
  , m_err_old(0.0)
  , m_dt_old(0.0)
  , m_last_rejected(false)
  {}



  /*--------------------------------------------------------------------------
   * Full reset after events (Strategy B)
   *
   * This is called by dense-output and controlled integration when a jump
   * occurs (fixed event or root event).
   *--------------------------------------------------------------------------*/
  void reset_after_event(time_type dt_before)
  {
    m_first_step    = true;
    m_last_rejected = false;
    m_err_old       = 0.0;
    m_dt_old        = detail::get_scalar_value(dt_before);
  }



  /*--------------------------------------------------------------------------
   * Compute AD-aware weighted RMS error
   *--------------------------------------------------------------------------*/
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



  /*--------------------------------------------------------------------------
   * try_step overload #1 (in-place)
   *--------------------------------------------------------------------------*/
  template<class System>
  controlled_step_result try_step(System sys,
                                  state_type& x,
                                  time_type& t,
                                  time_type& dt)
  {
    m_xnew_resizer.adjust_size(x,
                               [this](auto&& arg){ return this->resize_m_xnew<state_type>(arg); });

    controlled_step_result res =
      try_step(sys, x, t, m_xnew.m_v, dt);

    if (res == success)
      copy(m_xnew.m_v, x);

    return res;
  }



  /*--------------------------------------------------------------------------
   * try_step overload #2 (separate output buffer)
   *--------------------------------------------------------------------------*/
  template<class System>
  controlled_step_result try_step(System sys,
                                  const state_type& x,
                                  time_type& t,
                                  state_type& xout,
                                  time_type& dt)
  {
    double dt_val = detail::get_scalar_value(dt);

    /*--- enforce max_dt constraint ---*/
    if (m_max_dt != static_cast<time_type>(0))
    {
      double maxdt_val = detail::get_scalar_value(m_max_dt);
      if (detail::less_with_sign(maxdt_val, dt_val, dt_val))
      {
        dt = m_max_dt;
        return fail;
      }
    }

    /*--- step size control parameters (balanced) ---*/
    static constexpr double safe = 0.9;
    static constexpr double fac_max = 7.0;
    static constexpr double fac_min = 1.0 / 7.0;

    /*--- resize error buffer ---*/
    m_xerr_resizer.adjust_size(x,
                               [this](auto&& arg){ return this->resize_m_xerr<state_type>(arg); });

    /*--- perform Rosenbrock step ---*/
    m_stepper.do_step(sys, x, t, xout, dt, m_xerr.m_v);

    double err = error(xout, x, m_xerr.m_v);

    /*--- primary dt update factor ---*/
    double fac = std::pow(err, 0.25) / safe;
    fac = std::max(fac_min, std::min(fac_max, fac));
    double dt_new = dt_val / fac;

    /*--- accept step ---*/
    if (err <= 1.0)
    {
      if (!m_first_step)
      {
        double fac_pred =
          (m_dt_old / dt_val) *
          std::pow(err * err / m_err_old, 0.25) / safe;

        fac_pred = std::max(fac_min, std::min(fac_max, fac_pred));
        fac = std::max(fac, fac_pred);

        dt_new = dt_val / fac;
      }

      m_first_step = false;
      m_dt_old = dt_val;
      m_err_old = std::max(0.01, err);

      if (m_last_rejected)
      {
        dt_new = (dt_val >= 0.0)
        ? std::min(dt_new, dt_val)
          : std::max(dt_new, dt_val);
      }

      t += dt;

      /*--- clamp to max_dt ---*/
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

    /*--- reject step ---*/
    dt = time_type(dt_new);
    m_last_rejected = true;
    return fail;
  }



  /*--------------------------------------------------------------------------
   * Resize helpers
   *--------------------------------------------------------------------------*/
  template<class StateType>
  void adjust_size(const StateType& x)
  {
    resize_m_xerr(x);
    resize_m_xnew(x);
  }

  stepper_type& stepper()       { return m_stepper; }
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

  double    m_atol, m_rtol;
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
