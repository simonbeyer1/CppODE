#ifndef STIFF_SOLVER_FAD_HPP
#define STIFF_SOLVER_FAD_HPP

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <rosenbrock4_controller_fad.hpp>
#include <StepChecker.hpp>

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

}}} // namespace boost::numeric::ublas


#endif // STIFF_SOLVER_FAD_HPP
