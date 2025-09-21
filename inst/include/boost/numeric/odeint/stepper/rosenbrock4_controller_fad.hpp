#ifndef ROSENBROCK4_CONTROLLER_AD_HPP
#define ROSENBROCK4_CONTROLLER_AD_HPP

#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/controlled_step_result.hpp>
#include <boost/numeric/odeint/util/copy.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/detail/less_with_sign.hpp>
#include <cmath>
#include <algorithm>

namespace boost {
namespace numeric {
namespace odeint {

/**
 * Custom controller for rosenbrock4 that is aware of automatic differentiation.
 *
 * Modified error estimate: it accounts for all derivative components
 * stored inside fadbad::F<T> (or fadbad::B<T>), not only the value part.
 *
 * Use ONLY with AD types (fadbad::F<T>, fadbad::B<T>).
 */
template< class Stepper >
class rosenbrock4_controller_ad
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

  typedef rosenbrock4_controller_ad< Stepper > controller_type;

  rosenbrock4_controller_ad( double atol = 1.0e-6 , double rtol = 1.0e-6 ,
                             const stepper_type &stepper = stepper_type() )
    : m_stepper( stepper ) , m_atol( atol ) , m_rtol( rtol ) ,
      m_max_dt( static_cast<time_type>(0) ) ,
      m_first_step( true ) , m_err_old( 0.0 ) , m_dt_old( 0.0 ) ,
      m_last_rejected( false )
  { }

  rosenbrock4_controller_ad( double atol, double rtol, time_type max_dt,
                             const stepper_type &stepper = stepper_type() )
    : m_stepper( stepper ) , m_atol( atol ) , m_rtol( rtol ) , m_max_dt( max_dt ) ,
      m_first_step( true ) , m_err_old( 0.0 ) , m_dt_old( 0.0 ) ,
      m_last_rejected( false )
  { }

  /**
   * Modified error estimate:
   * For each component, compute normalized error based on both the
   * value and all derivative components (sensitivities).
   */
  double error( const state_type &x , const state_type &xold , const state_type &xerr )
  {
    const size_t n = x.size();
    double err = 0.0;
    for( size_t i=0 ; i<n ; ++i )
    {
      // remove constness to access F<T>::x() and F<T>::d()
      auto &xi    = const_cast<value_type&>(x[i]);
      auto &xiold = const_cast<value_type&>(xold[i]);
      auto &xerri = const_cast<value_type&>(xerr[i]);

      double sk = m_atol + m_rtol * std::max(std::abs(xiold.x()), std::abs(xi.x()));

      // start with value part
      double term = std::abs(xerri.x()) / sk;

      // add all derivative parts
      for (int j = 0; j < xerri.size(); ++j) {
        term = std::max(term, std::abs(xerri.d(j)) / sk);
      }

      err += term * term;
    }
    return std::sqrt( err / double(n) );
  }

  double last_error( void ) const { return m_err_old; }

  // In-place stepping
  template< class System >
  controlled_step_result try_step( System sys , state_type &x , time_type &t , time_type &dt )
  {
    m_xnew_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_m_xnew<state_type>(std::forward<decltype(arg)>(arg)); });
    controlled_step_result res = try_step( sys , x , t , m_xnew.m_v , dt );
    if( res == success )
    {
      copy( m_xnew.m_v , x );
    }
    return res;
  }

  // Out-of-place stepping
  template< class System >
  controlled_step_result try_step( System sys , const state_type &x , time_type &t ,
                                   state_type &xout , time_type &dt )
  {
    if( m_max_dt != static_cast<time_type>(0) && detail::less_with_sign(m_max_dt, dt, dt) )
    {
      dt = m_max_dt;
      return fail;
    }

    using std::pow;
    static const double safe = 0.9 , fac1 = 5.0 , fac2 = 1.0 / 6.0;

    m_xerr_resizer.adjust_size(x, [this](auto&& arg) {
      return this->resize_m_xerr<state_type>(std::forward<decltype(arg)>(arg)); });

    m_stepper.do_step( sys , x , t , xout , dt , m_xerr.m_v );
    double err = error( xout , x , m_xerr.m_v );

    double fac = std::max( fac2 , std::min( fac1 , pow( err , 0.25 ) / safe ) );
    double dt_new = dt.x() / fac;   // use value part of dt for scaling

    if ( err <= 1.0 )
    {
      if( m_first_step ) { m_first_step = false; }
      else
      {
        double fac_pred = ( m_dt_old / dt.x() ) * pow( err * err / m_err_old , 0.25 ) / safe;
        fac_pred = std::max( fac2 , std::min( fac1 , fac_pred ) );
        fac = std::max( fac , fac_pred );
        dt_new = dt.x() / fac;
      }

      m_dt_old = dt.x();
      m_err_old = std::max( 0.01 , err );

      if( m_last_rejected )
        dt_new = ( dt.x() >= 0.0 ? std::min( dt_new , dt.x() ) : std::max( dt_new , dt.x() ) );

      t += dt;
      if( m_max_dt != static_cast<time_type>(0) )
        dt = detail::min_abs(m_max_dt, time_type(dt_new));
      else
        dt = time_type(dt_new);

      m_last_rejected = false;
      return success;
    }
    else
    {
      dt = time_type(dt_new);
      m_last_rejected = true;
      return fail;
    }
  }

  template< class StateType >
  void adjust_size( const StateType &x )
  {
    resize_m_xerr( x );
    resize_m_xnew( x );
  }

  stepper_type& stepper( void ) { return m_stepper; }
  const stepper_type& stepper( void ) const { return m_stepper; }

protected:
  template< class StateIn >
  bool resize_m_xerr( const StateIn &x )
  { return adjust_size_by_resizeability( m_xerr , x , typename is_resizeable<state_type>::type() ); }

  template< class StateIn >
  bool resize_m_xnew( const StateIn &x )
  { return adjust_size_by_resizeability( m_xnew , x , typename is_resizeable<state_type>::type() ); }

  stepper_type m_stepper;
  resizer_type m_xerr_resizer;
  resizer_type m_xnew_resizer;
  wrapped_state_type m_xerr;
  wrapped_state_type m_xnew;

  double m_atol , m_rtol;
  time_type m_max_dt;
  bool m_first_step;
  double m_err_old , m_dt_old;
  bool m_last_rejected;
};


} } } // namespace boost::numeric::odeint

#endif // ROSENBROCK4_CONTROLLER_AD_HPP
