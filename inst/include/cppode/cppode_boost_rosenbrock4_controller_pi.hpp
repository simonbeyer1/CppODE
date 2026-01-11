/*
 Controller for the Rosenbrock4 method with PI step-size control.

 Original work:
 Copyright 2011-2012 Karsten Ahnert
 Copyright 2011-2012 Mario Mulansky
 Copyright 2012 Christoph Koke
 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)

 Modified work:
 Copyright 2026 Simon Beyer

 Modifications:
 - Replaced classical step-size control with PI controller
 - Added reset_after_event() for event-driven integration
 - Implemented Gustafsson-Söderlind PI control algorithm
 - Added diagnostic accessors (first_step, last_rejected, dt_old)
 - Improved step-size stability after rejections

 PI controller based on:
 - Gustafsson, K. (1991). "Control theoretic techniques for stepsize selection
 in explicit Runge-Kutta methods". ACM Trans. Math. Softw. 17(4), 533-554.
 - Söderlind, G. (2002). "Automatic control and adaptive time-stepping".
 Numer. Algorithms 31(1-4), 281-310.
 */

#ifndef CPPODE_ROSENBROCK4_CONTROLLER_PI_HPP_INCLUDED
#define CPPODE_ROSENBROCK4_CONTROLLER_PI_HPP_INCLUDED

#include <boost/config.hpp>
#include <boost/numeric/odeint/util/bind.hpp>

#include <boost/numeric/odeint/stepper/controlled_step_result.hpp>
#include <boost/numeric/odeint/stepper/stepper_categories.hpp>

#include <boost/numeric/odeint/util/copy.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/detail/less_with_sign.hpp>

#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>

namespace boost {
namespace numeric {
namespace odeint {

/**
 * @class rosenbrock4_controller_pi
 * @brief Controlled Rosenbrock4 stepper with PI step-size control
 *
 * This class implements a proportional-integral (PI) controller for
 * adaptive step-size selection in Rosenbrock methods. The PI controller
 * provides smoother step-size evolution compared to classical I-controllers
 * by incorporating information from both current and previous errors.
 *
 * **PI Control Formula:**
 * @code
 * dt_new = dt * safety * (err_old / err)^beta * (1 / err)^alpha
 * @endcode
 *
 * where:
 * - alpha = 0.14 (proportional gain, tuned for order 4)
 * - beta = 0.08 (integral gain, provides smoothness)
 * - safety = 0.9 (safety factor)
 *
 * **Features:**
 * - Smooth step-size evolution (reduces oscillations)
 * - Conservative behavior after step rejections
 * - Event reset support via reset_after_event()
 * - Compatible with standard Boost.Odeint integration routines
 *
 * @tparam Stepper The underlying Rosenbrock4 stepper type
 *
 * @par Usage Example
 * @code
 * using stepper_t = rosenbrock4<double>;
 * rosenbrock4_controller_pi<stepper_t> controlled(1e-6, 1e-6);
 *
 * integrate_adaptive(controlled, system, x, t0, t1, dt, observer);
 * @endcode
 *
 * @see Gustafsson (1991), Söderlind (2002)
 * @see boost::numeric::odeint::rosenbrock4
 */
template< class Stepper >
class rosenbrock4_controller_pi
{
public:

  typedef Stepper stepper_type;
  typedef typename stepper_type::value_type value_type;
  typedef typename stepper_type::state_type state_type;
  typedef typename stepper_type::wrapped_state_type wrapped_state_type;
  typedef typename stepper_type::time_type time_type;
  typedef typename stepper_type::deriv_type deriv_type;
  typedef typename stepper_type::wrapped_deriv_type wrapped_deriv_type;
  typedef typename stepper_type::resizer_type resizer_type;
  typedef controlled_stepper_tag stepper_category;

  typedef rosenbrock4_controller_pi< Stepper > controller_type;

  /// Proportional gain for PI controller (tuned for order 4)
  static constexpr value_type pi_alpha = 0.7 / 5.0;  // ~0.14

  /// Integral gain for PI controller (provides smoothness)
  static constexpr value_type pi_beta  = 0.4 / 5.0;  // ~0.08

  /**
   * @brief Construct controller with tolerances
   *
   * @param atol Absolute tolerance (default: 1e-6)
   * @param rtol Relative tolerance (default: 1e-6)
   * @param stepper The underlying Rosenbrock4 stepper instance
   *
   * @note Initial error and dt history are set to 1.0 to ensure
   *       conservative first step
   */
  rosenbrock4_controller_pi( value_type atol = 1.0e-6 , value_type rtol = 1.0e-6 ,
                             const stepper_type &stepper = stepper_type() )
    : m_stepper( stepper ) , m_atol( atol ) , m_rtol( rtol ) ,
      m_max_dt( static_cast<time_type>(0) ) ,
      m_first_step( true ) , m_err_old( 1.0 ) , m_dt_old( 1.0 ) ,
      m_last_rejected( false )
  { }

  /**
   * @brief Construct controller with tolerances and maximum step size
   *
   * @param atol Absolute tolerance
   * @param rtol Relative tolerance
   * @param max_dt Maximum allowed step size (0 = unlimited)
   * @param stepper The underlying Rosenbrock4 stepper instance
   */
  rosenbrock4_controller_pi( value_type atol, value_type rtol, time_type max_dt,
                             const stepper_type &stepper = stepper_type() )
    : m_stepper( stepper ) , m_atol( atol ) , m_rtol( rtol ) , m_max_dt( max_dt ) ,
      m_first_step( true ) , m_err_old( 1.0 ) , m_dt_old( 1.0 ) ,
      m_last_rejected( false )
  { }

  /**
   * @brief Compute normalized error using L2 norm
   *
   * Computes the weighted RMS error:
   * @code
   * err = sqrt(mean((xerr[i] / scale[i])^2))
   * where scale[i] = atol + rtol * max(|xold[i]|, |x[i]|)
   * @endcode
   *
   * @param x New state after step
   * @param xold Old state before step
   * @param xerr Error estimate from embedded method
   * @return Normalized error (accept step if err <= 1.0)
   */
  value_type error( const state_type &x , const state_type &xold , const state_type &xerr )
  {
    BOOST_USING_STD_MAX();
    using std::abs;
    using std::sqrt;

    const size_t n = x.size();
    value_type err = 0.0 , sk = 0.0;
    for( size_t i=0 ; i<n ; ++i )
    {
      sk = m_atol + m_rtol * max BOOST_PREVENT_MACRO_SUBSTITUTION ( abs( xold[i] ) , abs( x[i] ) );
      err += xerr[i] * xerr[i] / sk / sk;
    }
    return sqrt( err / value_type( n ) );
  }

  /**
   * @brief Get the error from the previous step
   *
   * @return Previous step's normalized error (used in PI control)
   */
  value_type last_error( void ) const
  {
    return m_err_old;
  }

  /**
   * @brief Reset controller state after a discontinuous event
   *
   * Clears the step-size controller's memory to prevent using
   * pre-event error history for post-event step-size selection.
   * This ensures clean restart after state discontinuities.
   *
   * @param dt_before Step size before the event (currently unused)
   *
   * @note Call this after applying fixed-time or root-finding events
   *       that modify the state discontinuously
   *
   * @par Implementation Details
   * - Resets to first-step mode (uses P-control only)
   * - Clears error and step-size history
   * - Resets rejection flag
   */
  void reset_after_event( time_type /*dt_before*/ )
  {
    m_first_step = true;
    m_err_old = 1.0;
    m_dt_old = 1.0;
    m_last_rejected = false;
  }

  /**
   * @brief In-place try_step (modifies state vector directly)
   *
   * Attempts one integration step, modifying the state in place
   * on success. Automatically manages internal buffers.
   *
   * @tparam System ODE system type (callable: void(x, dxdt, t))
   * @param sys The ODE system
   * @param x State vector (modified on success)
   * @param t Current time (advanced on success)
   * @param dt Step size (updated based on error estimate)
   * @return success if step accepted, fail if rejected
   */
  template< class System >
  boost::numeric::odeint::controlled_step_result
  try_step( System sys , state_type &x , time_type &t , time_type &dt )
  {
    m_xnew_resizer.adjust_size( x , detail::bind( &controller_type::template resize_m_xnew< state_type > , detail::ref( *this ) , detail::_1 ) );
    boost::numeric::odeint::controlled_step_result res = try_step( sys , x , t , m_xnew.m_v , dt );
    if( res == success )
    {
      boost::numeric::odeint::copy( m_xnew.m_v , x );
    }
    return res;
  }

  /**
   * @brief Main try_step with separate input/output states
   *
   * Attempts one integration step using PI control for step-size adaptation.
   * This is the core implementation of the adaptive controller.
   *
   * @tparam System ODE system type
   * @param sys The ODE system
   * @param x Input state (unchanged)
   * @param t Current time (advanced on success)
   * @param xout Output state (new state on success, undefined on fail)
   * @param dt Step size (updated based on error estimate)
   * @return success if step accepted, fail if rejected
   *
   * @par PI Control Strategy
   * - **First step or after rejection**: Pure P-control (no history)
   *   - factor = 0.9 * (1/err)^0.25
   * - **Normal steps**: Full PI control
   *   - factor = 0.9 * (err_old/err)^0.08 * (1/err)^0.14
   * - **After rejection**: Limit growth to prevent oscillations
   *   - factor <= 1.0
   *
   * @par Step Size Limits
   * - Minimum decrease: factor >= 0.2 (max 80% reduction)
   * - Maximum increase: factor <= 5.0 (max 400% growth)
   * - Absolute limit: dt <= max_dt (if set)
   */
  template< class System >
  boost::numeric::odeint::controlled_step_result
  try_step( System sys , const state_type &x , time_type &t , state_type &xout , time_type &dt )
  {
    if( m_max_dt != static_cast<time_type>(0) && detail::less_with_sign(m_max_dt, dt, dt) )
    {
      dt = m_max_dt;
      return fail;
    }

    BOOST_USING_STD_MIN();
    BOOST_USING_STD_MAX();
    using std::pow;
    using std::abs;

    // Safety and limiting factors
    static const value_type safe = 0.9;
    static const value_type max_factor = 5.0;
    static const value_type min_factor = 0.2;

    m_xerr_resizer.adjust_size( x , detail::bind( &controller_type::template resize_m_xerr< state_type > , detail::ref( *this ) , detail::_1 ) );

    m_stepper.do_step( sys , x , t , xout , dt , m_xerr.m_v );
    value_type err = error( xout , x , m_xerr.m_v );

    // Prevent division by zero
    err = max BOOST_PREVENT_MACRO_SUBSTITUTION ( err, static_cast<value_type>(1e-15) );

    if ( err <= 1.0 )
    {
      // === Step accepted ===
      value_type factor;

      if( m_first_step || m_last_rejected )
      {
        // Pure P-control for first step or after rejection
        factor = safe * pow( 1.0 / err, 0.25 );
      }
      else
      {
        // PI control: use both current and previous error
        factor = safe
        * pow( m_err_old / err, pi_beta )
        * pow( 1.0 / err, pi_alpha );
      }

      // Limit growth after rejection
      if( m_last_rejected )
      {
        factor = min BOOST_PREVENT_MACRO_SUBSTITUTION ( factor, static_cast<value_type>(1.0) );
      }

      // Clamp factor
      factor = max BOOST_PREVENT_MACRO_SUBSTITUTION ( min_factor,
                                                      min BOOST_PREVENT_MACRO_SUBSTITUTION ( max_factor, factor ) );

      // Update controller state
      m_dt_old = dt;
      m_err_old = max BOOST_PREVENT_MACRO_SUBSTITUTION ( static_cast<value_type>(0.01), err );
      m_first_step = false;
      m_last_rejected = false;

      // Advance time
      t += dt;

      // Update step size
      value_type dt_new = dt * factor;

      // Apply max_dt limit
      if( m_max_dt != static_cast<time_type>(0) )
      {
        dt = detail::min_abs( m_max_dt, dt_new );
      }
      else
      {
        dt = dt_new;
      }

      return success;
    }
    else
    {
      // === Step rejected ===
      value_type factor = safe * pow( 1.0 / err, 0.25 );

      // Clamp decrease (ensure we decrease but not too aggressively)
      factor = max BOOST_PREVENT_MACRO_SUBSTITUTION ( min_factor,
                                                      min BOOST_PREVENT_MACRO_SUBSTITUTION ( static_cast<value_type>(0.9), factor ) );

      m_last_rejected = true;
      dt *= factor;

      return fail;
    }
  }

  /**
   * @brief Adjust internal buffer sizes to match state
   *
   * @tparam StateType State vector type
   * @param x Reference state for sizing
   */
  template< class StateType >
  void adjust_size( const StateType &x )
  {
    resize_m_xerr( x );
    resize_m_xnew( x );
  }

  /**
   * @brief Access the underlying Rosenbrock4 stepper
   * @return Reference to the stepper
   */
  stepper_type& stepper( void )
  {
    return m_stepper;
  }

  /**
   * @brief Access the underlying Rosenbrock4 stepper (const)
   * @return Const reference to the stepper
   */
  const stepper_type& stepper( void ) const
  {
    return m_stepper;
  }

  /**
   * @brief Check if this is the first step
   * @return true if no successful steps have been taken yet
   */
  bool first_step() const { return m_first_step; }

  /**
   * @brief Check if the last step was rejected
   * @return true if the previous try_step() returned fail
   */
  bool last_rejected() const { return m_last_rejected; }

  /**
   * @brief Get the previous step size
   * @return Step size of the last successful step
   */
  value_type dt_old() const { return m_dt_old; }

protected:

  template< class StateIn >
  bool resize_m_xerr( const StateIn &x )
  {
    return adjust_size_by_resizeability( m_xerr , x , typename is_resizeable<state_type>::type() );
  }

  template< class StateIn >
  bool resize_m_xnew( const StateIn &x )
  {
    return adjust_size_by_resizeability( m_xnew , x , typename is_resizeable<state_type>::type() );
  }

  stepper_type m_stepper;             ///< Underlying Rosenbrock4 stepper
  resizer_type m_xerr_resizer;        ///< Resizer for error buffer
  resizer_type m_xnew_resizer;        ///< Resizer for new state buffer
  wrapped_state_type m_xerr;          ///< Error estimate buffer
  wrapped_state_type m_xnew;          ///< New state buffer
  value_type m_atol , m_rtol;         ///< Absolute and relative tolerances
  time_type m_max_dt;                 ///< Maximum step size (0 = unlimited)
  bool m_first_step;                  ///< First step flag
  value_type m_err_old , m_dt_old;    ///< Error and step-size history for PI control
  bool m_last_rejected;               ///< Rejection flag
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_ROSENBROCK4_CONTROLLER_PI_HPP_INCLUDED
