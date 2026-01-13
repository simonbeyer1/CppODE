/*
 Controller for the Rosenbrock4 method with PI step-size control.

 Original work:
 Copyright (C) 2011–2012 Karsten Ahnert
 Copyright (C) 2011–2012 Mario Mulansky
 Copyright (C) 2012 Christoph Koke
 Distributed under the Boost Software License, Version 1.0.

 Modified work:
 Copyright (C) 2026 Simon Beyer

 Modifications:
 - Replaced classical step-size control with PI controller
 - Added reset_after_event() for event-driven integration
 - Implemented Gustafsson–Söderlind PI control algorithm
 - Added diagnostic accessors (first_step, last_rejected, dt_old)

 This file remains distributed under the Boost Software License, Version 1.0.

 PI controller based on:
 Gustafsson, K., Lundh, M. & Söderlind, G. (1988).
 "A PI stepsize control for the numerical solution of ordinary differential equations".
 BIT 28, 270–287. https://doi.org/10.1007/BF01934091
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
 * where (defaults):
 * - alpha = 0.175 (proportional gain)
 * - beta = 0.1 (integral gain)
 * - safety = 0.9 (safety factor)
 *
 * **Features:**
 * - Smooth step-size evolution
 * - Conservative behavior after step rejections
 * - Event reset support via reset_after_event()
 * - Configurable PI controller parameters
 *
 * @tparam Stepper The underlying Rosenbrock4 stepper type
 *
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

  /// Method order (used in step-size formula)
  static constexpr value_type order = 3.0;

  /// Default proportional gain: alpha = 0.7 / (order + 1)
  static constexpr value_type default_alpha = 0.7 / (order + 1.0);  // ~0.175

  /// Default integral gain: beta = 0.4 / (order + 1)
  static constexpr value_type default_beta  = 0.4 / (order + 1.0);  // ~0.1

  /// Default safety factor for step-size selection
  static constexpr value_type default_safety = 0.9;

  /// Default maximum step increase factor
  static constexpr value_type default_max_factor = 5.0;

  /// Default minimum step decrease factor
  static constexpr value_type default_min_factor = 0.2;

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
      m_alpha( default_alpha ) , m_beta( default_beta ) ,
      m_safety( default_safety ) , m_max_factor( default_max_factor ) ,
      m_min_factor( default_min_factor ) ,
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
      m_alpha( default_alpha ) , m_beta( default_beta ) ,
      m_safety( default_safety ) , m_max_factor( default_max_factor ) ,
      m_min_factor( default_min_factor ) ,
      m_first_step( true ) , m_err_old( 1.0 ) , m_dt_old( 1.0 ) ,
      m_last_rejected( false )
  { }

  /**
   * @brief Construct controller with full tuning control
   *
   * @param atol Absolute tolerance
   * @param rtol Relative tolerance
   * @param max_dt Maximum allowed step size (0 = unlimited)
   * @param alpha PI proportional gain (default: ~0.175)
   * @param beta PI integral gain (default: ~0.1)
   * @param safety Safety factor (default: 0.9)
   * @param max_factor Maximum step increase (default: 5.0)
   * @param min_factor Minimum step decrease (default: 0.2)
   * @param stepper The underlying Rosenbrock4 stepper instance
   *
   * @par Tuning Guidelines
   * - **Increase alpha/beta**: Smoother step-size evolution, slower adaptation
   * - **Decrease alpha/beta**: Faster adaptation, more oscillations
   * - **Increase safety**: More conservative, smaller steps
   * - **Narrow min/max_factor**: Limits step-size variation
   */
  rosenbrock4_controller_pi( value_type atol, value_type rtol, time_type max_dt,
                             value_type alpha, value_type beta,
                             value_type safety = default_safety,
                             value_type max_factor = default_max_factor,
                             value_type min_factor = default_min_factor,
                             const stepper_type &stepper = stepper_type() )
    : m_stepper( stepper ) , m_atol( atol ) , m_rtol( rtol ) , m_max_dt( max_dt ) ,
      m_alpha( alpha ) , m_beta( beta ) ,
      m_safety( safety ) , m_max_factor( max_factor ) , m_min_factor( min_factor ) ,
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
   *   - factor = safety * (1/err)^(1/(order+1))
   * - **Normal steps**: Full PI control
   *   - factor = safety * (err_old/err)^beta * (1/err)^alpha
   * - **After rejection**: Limit growth to prevent oscillations
   *   - factor <= 1.0
   *
   * @par Step Size Limits
   * - Minimum decrease: factor >= min_factor (default: max 80% reduction)
   * - Maximum increase: factor <= max_factor (default: max 400% growth)
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
        factor = m_safety * pow( 1.0 / err, 1.0 / (order + 1.0) );
      }
      else
      {
        // PI control: use both current and previous error
        factor = m_safety
        * pow( m_err_old / err, m_beta )
        * pow( 1.0 / err, m_alpha );
      }

      // Limit growth after rejection
      if( m_last_rejected )
      {
        factor = min BOOST_PREVENT_MACRO_SUBSTITUTION ( factor, static_cast<value_type>(1.0) );
      }

      // Clamp factor
      factor = max BOOST_PREVENT_MACRO_SUBSTITUTION ( m_min_factor,
                                                      min BOOST_PREVENT_MACRO_SUBSTITUTION ( m_max_factor, factor ) );

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
      value_type factor = m_safety * pow( 1.0 / err, 1.0 / (order + 1.0) );

      // Clamp decrease (ensure we decrease but not too aggressively)
      factor = max BOOST_PREVENT_MACRO_SUBSTITUTION ( m_min_factor,
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

  // Tolerance accessors

  /**
   * @brief Get absolute tolerance
   * @return Current absolute tolerance
   */
  value_type atol() const { return m_atol; }

  /**
   * @brief Get relative tolerance
   * @return Current relative tolerance
   */
  value_type rtol() const { return m_rtol; }

  /**
   * @brief Update error tolerances
   * @param atol New absolute tolerance
   * @param rtol New relative tolerance
   */
  void set_tolerances(value_type atol, value_type rtol) { m_atol = atol; m_rtol = rtol; }

  // PI controller parameter accessors

  /**
   * @brief Get proportional gain
   * @return Current alpha value
   */
  value_type alpha() const { return m_alpha; }

  /**
   * @brief Get integral gain
   * @return Current beta value
   */
  value_type beta() const { return m_beta; }

  /**
   * @brief Get safety factor
   * @return Current safety value
   */
  value_type safety() const { return m_safety; }

  /**
   * @brief Get maximum step increase factor
   * @return Current max_factor value
   */
  value_type max_factor() const { return m_max_factor; }

  /**
   * @brief Get minimum step decrease factor
   * @return Current min_factor value
   */
  value_type min_factor() const { return m_min_factor; }

  /**
   * @brief Update PI controller gains
   * @param alpha New proportional gain
   * @param beta New integral gain
   */
  void set_pi_gains(value_type alpha, value_type beta) { m_alpha = alpha; m_beta = beta; }

  /**
   * @brief Update all PI controller parameters
   * @param alpha Proportional gain
   * @param beta Integral gain
   * @param safety Safety factor
   * @param max_factor Maximum step increase
   * @param min_factor Minimum step decrease
   */
  void set_pi_parameters(value_type alpha, value_type beta, value_type safety,
                         value_type max_factor, value_type min_factor)
  {
    m_alpha = alpha;
    m_beta = beta;
    m_safety = safety;
    m_max_factor = max_factor;
    m_min_factor = min_factor;
  }

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

  // PI controller parameters (configurable)
  value_type m_alpha;                 ///< Proportional gain
  value_type m_beta;                  ///< Integral gain
  value_type m_safety;                ///< Safety factor
  value_type m_max_factor;            ///< Maximum step increase factor
  value_type m_min_factor;            ///< Minimum step decrease factor

  // Controller state
  bool m_first_step;                  ///< First step flag
  value_type m_err_old , m_dt_old;    ///< Error and step-size history for PI control
  bool m_last_rejected;               ///< Rejection flag
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_ROSENBROCK4_CONTROLLER_PI_HPP_INCLUDED
