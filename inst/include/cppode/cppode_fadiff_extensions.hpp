#ifndef CPPODE_FADIFF_EXTENSIONS_HPP
#define CPPODE_FADIFF_EXTENSIONS_HPP

/**
 * @file cppode_fadiff_extensions.hpp
 * @brief Extensions for FADBAD++ fadiff.h to work with Boost.Odeint
 *
 * This header provides additional functionality for fadbad::F<T> types:
 *   - abs() function for F<T> types
 *   - Safe comparison operators for nested F<F<T>> types
 *   - std namespace imports for math functions
 *   - std::min/max overloads for F<T>
 *
 * @author Simon Beyer <simon.beyer@fdm.uni-freiburg.de>
 */

#include <fadbad++/fadiff.h>
#include <type_traits>

// ============================================================================
//  FADBAD namespace extensions
// ============================================================================

namespace fadbad {

/**
 * @brief Absolute value function for fadbad::F types.
 *
 * FADBAD++ does not provide abs() by default. This implementation
 * uses the comparison operators already defined for F<T>.
 *
 * @tparam T Inner type (double, F<double>, etc.)
 * @param x Value to take the absolute of.
 * @return |x|
 */
template<class T>
inline F<T> abs(const F<T>& x) {
  return (x < T(0)) ? -x : x;
}

/**
 * @brief Safe less-than operator for comparing nested fadbad types.
 *
 * This overload prevents ambiguous template resolution between
 * fadbad::F<F<T>> and fadbad::F<T> comparisons by delegating
 * to scalar value comparison.
 *
 * @tparam Inner Inner fadbad or arithmetic type.
 * @param a First operand (F<F<Inner>>).
 * @param b Second operand (F<Inner>).
 * @return True if scalar value of a < scalar value of b.
 */
template <typename Inner>
inline bool operator<(const F<F<Inner>>& a, const F<Inner>& b) {
  return a.val().val() < b.val();
}

/**
 * @brief Symmetric overload for comparing F<T> with F<F<T>>.
 */
template <typename Inner>
inline bool operator<(const F<Inner>& a, const F<F<Inner>>& b) {
  return a.val() < b.val().val();
}

/**
 * @brief Safe absolute value function for nested fadbad::F types.
 *
 * This handles the F<F<T>> case which requires special comparison.
 *
 * @tparam Inner Inner fadbad or arithmetic type.
 * @param x Value to take the absolute of.
 * @return |x|
 */
template <typename Inner>
inline F<F<Inner>> abs(const F<F<Inner>>& x) {
  double val = x.val().val();
  return (val < 0.0) ? -x : x;
}

} // namespace fadbad

// ============================================================================
//  std namespace: import FADBAD math functions and provide min/max
// ============================================================================

namespace std {

// Import FADBAD math overloads so std::sin, std::sqrt, ... work with F<T>
using fadbad::abs;
using fadbad::sin;
using fadbad::cos;
using fadbad::tan;
using fadbad::asin;
using fadbad::acos;
using fadbad::atan;
using fadbad::exp;
using fadbad::log;
using fadbad::sqrt;
using fadbad::pow;
using fadbad::sqr;

/**
 * @brief std::min for two fadbad::F types
 */
template<class T>
inline fadbad::F<T> min(const fadbad::F<T>& a, const fadbad::F<T>& b) {
  return (a < b) ? a : b;
}

/**
 * @brief std::min for fadbad::F and scalar (F on left)
 */
template<class T, class U>
inline typename std::enable_if<std::is_arithmetic<U>::value, fadbad::F<T>>::type
min(const fadbad::F<T>& a, const U& b) {
  return (a < b) ? a : fadbad::F<T>(b);
}

/**
 * @brief std::min for scalar and fadbad::F (F on right)
 */
template<class T, class U>
inline typename std::enable_if<std::is_arithmetic<U>::value, fadbad::F<T>>::type
min(const U& a, const fadbad::F<T>& b) {
  return (a < b) ? fadbad::F<T>(a) : b;
}

/**
 * @brief std::max for two fadbad::F types
 */
template<class T>
inline fadbad::F<T> max(const fadbad::F<T>& a, const fadbad::F<T>& b) {
  return (a > b) ? a : b;
}

/**
 * @brief std::max for fadbad::F and scalar (F on left)
 */
template<class T, class U>
inline typename std::enable_if<std::is_arithmetic<U>::value, fadbad::F<T>>::type
max(const fadbad::F<T>& a, const U& b) {
  return (a > b) ? a : fadbad::F<T>(b);
}

/**
 * @brief std::max for scalar and fadbad::F (F on right)
 */
template<class T, class U>
inline typename std::enable_if<std::is_arithmetic<U>::value, fadbad::F<T>>::type
max(const U& a, const fadbad::F<T>& b) {
  return (a > b) ? fadbad::F<T>(a) : b;
}

} // namespace std

#endif // CPPODE_FADIFF_EXTENSIONS_HPP
