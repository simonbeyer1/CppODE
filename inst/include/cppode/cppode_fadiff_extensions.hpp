/*
 Extensions for FADBAD++ fadiff.h to work with Boost.Odeint

 This header provides additional functionality for fadbad::F<T> types:
 - abs() function for F<T> types
 - Hyperbolic functions (sinh, cosh, tanh, asinh, acosh, atanh)
 - Safe comparison operators for nested F<F<T>> types
 - std::min/max overloads for F<T>

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_FADIFF_EXTENSIONS_HPP
#define CPPODE_FADIFF_EXTENSIONS_HPP

#include <fadbad++/fadiff.h>
#include <type_traits>
#include <cmath>

// =========================================================================================
//  FADBAD namespace extensions
// =========================================================================================
namespace fadbad {

// ----------------------------------------------------------------------------
//  Hyperbolic functions for F<T>
// ----------------------------------------------------------------------------
// FADBAD++ provides exp() but not hyperbolic functions.
// We implement them using the identities:
//   sinh(x) = (exp(x) - exp(-x)) / 2
//   cosh(x) = (exp(x) + exp(-x)) / 2
//   tanh(x) = sinh(x) / cosh(x)
// For the inverse functions:
//   asinh(x) = log(x + sqrt(x^2 + 1))
//   acosh(x) = log(x + sqrt(x^2 - 1))  [x >= 1]
//   atanh(x) = 0.5 * log((1 + x) / (1 - x))  [|x| < 1]

/**
 * @brief Hyperbolic sine for fadbad::F types.
 * @tparam T Inner type (double, F<double>, etc.)
 * @param x Argument
 * @return sinh(x) = (exp(x) - exp(-x)) / 2
 */
template<class T>
inline F<T> sinh(const F<T>& x) {
  F<T> ep = exp(x);
  F<T> em = exp(-x);
  return (ep - em) * T(0.5);
}

/**
 * @brief Hyperbolic cosine for fadbad::F types.
 * @tparam T Inner type (double, F<double>, etc.)
 * @param x Argument
 * @return cosh(x) = (exp(x) + exp(-x)) / 2
 */
template<class T>
inline F<T> cosh(const F<T>& x) {
  F<T> ep = exp(x);
  F<T> em = exp(-x);
  return (ep + em) * T(0.5);
}

/**
 * @brief Hyperbolic tangent for fadbad::F types.
 * @tparam T Inner type (double, F<double>, etc.)
 * @param x Argument
 * @return tanh(x) = sinh(x) / cosh(x)
 */
template<class T>
inline F<T> tanh(const F<T>& x) {
  F<T> ep = exp(x);
  F<T> em = exp(-x);
  return (ep - em) / (ep + em);
}

/**
 * @brief Inverse hyperbolic sine for fadbad::F types.
 * @tparam T Inner type (double, F<double>, etc.)
 * @param x Argument
 * @return asinh(x) = log(x + sqrt(x^2 + 1))
 */
template<class T>
inline F<T> asinh(const F<T>& x) {
  return log(x + sqrt(x * x + T(1)));
}

/**
 * @brief Inverse hyperbolic cosine for fadbad::F types.
 * @tparam T Inner type (double, F<double>, etc.)
 * @param x Argument (must be >= 1)
 * @return acosh(x) = log(x + sqrt(x^2 - 1))
 */
template<class T>
inline F<T> acosh(const F<T>& x) {
  return log(x + sqrt(x * x - T(1)));
}

/**
 * @brief Inverse hyperbolic tangent for fadbad::F types.
 * @tparam T Inner type (double, F<double>, etc.)
 * @param x Argument (must satisfy |x| < 1)
 * @return atanh(x) = 0.5 * log((1 + x) / (1 - x))
 */
template<class T>
inline F<T> atanh(const F<T>& x) {
  return T(0.5) * log((T(1) + x) / (T(1) - x));
}

// ----------------------------------------------------------------------------
//  Absolute value
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
//  Safe comparison operators for nested types
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
//  Min/Max functions
// ----------------------------------------------------------------------------

/**
 * @brief min for two fadbad::F types
 */
template<class T>
inline fadbad::F<T> min(const fadbad::F<T>& a, const fadbad::F<T>& b) {
  return (a < b) ? a : b;
}

/**
 * @brief min for fadbad::F and scalar (F on left)
 */
template<class T, class U>
inline typename std::enable_if<std::is_arithmetic<U>::value, fadbad::F<T>>::type
min(const fadbad::F<T>& a, const U& b) {
  return (a < b) ? a : fadbad::F<T>(b);
}

/**
 * @brief min for scalar and fadbad::F (F on right)
 */
template<class T, class U>
inline typename std::enable_if<std::is_arithmetic<U>::value, fadbad::F<T>>::type
min(const U& a, const fadbad::F<T>& b) {
  return (a < b) ? fadbad::F<T>(a) : b;
}

/**
 * @brief max for two fadbad::F types
 */
template<class T>
inline fadbad::F<T> max(const fadbad::F<T>& a, const fadbad::F<T>& b) {
  return (a > b) ? a : b;
}

/**
 * @brief max for fadbad::F and scalar (F on left)
 */
template<class T, class U>
inline typename std::enable_if<std::is_arithmetic<U>::value, fadbad::F<T>>::type
max(const fadbad::F<T>& a, const U& b) {
  return (a > b) ? a : fadbad::F<T>(b);
}

/**
 * @brief max for scalar and fadbad::F (F on right)
 */
template<class T, class U>
inline typename std::enable_if<std::is_arithmetic<U>::value, fadbad::F<T>>::type
max(const U& a, const fadbad::F<T>& b) {
  return (a > b) ? fadbad::F<T>(a) : b;
}

} // namespace fadbad

// =========================================================================================
//  Steady-State Detection Utilities
// =========================================================================================
namespace cppode {
namespace detail {

/**
 * @brief Recursively extract max absolute value from all AD levels
 *
 * This function traverses all levels of nested fadbad::F types and returns
 * the maximum absolute value found at any level. This is essential for
 * steady-state detection where we need ALL derivatives (including sensitivities)
 * to be near zero.
 *
 * For F<F<double>> (second-order sensitivities):
 *   - Level 0: x.x().x()           (the scalar value)
 *   - Level 1: x.x().d(j)          (first-order sensitivities of value)
 *   - Level 2: x.d(i).x()          (second-order: d/dp_i of value)
 *   - Level 3: x.d(i).d(j)         (second-order: d²/dp_i dp_j)
 *
 * @section algorithm Algorithm
 * Base case (arithmetic type): return |v|
 * Recursive case (F<T>): return max(recurse(x()), max_i(recurse(d(i))))
 */

// Base case: arithmetic types (double, float, int, ...)
template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
max_abs_all_levels(const T& v)
{
  return std::abs(static_cast<double>(v));
}

// Recursive case: fadbad::F<T>
template<class T>
inline double max_abs_all_levels(const fadbad::F<T>& v)
{
  // Get max from the main value
  double m = max_abs_all_levels(const_cast<fadbad::F<T>&>(v).x());

  // Get max from all derivative components
  unsigned n = const_cast<fadbad::F<T>&>(v).size();
  for (unsigned i = 0; i < n; ++i) {
    m = std::max(m, max_abs_all_levels(const_cast<fadbad::F<T>&>(v).d(i)));
  }

  return m;
}

/**
 * @brief Compute max absolute value across all elements and all AD levels of a vector
 *
 * Applies max_abs_all_levels to each element of a state vector and returns
 * the overall maximum. This gives the infinity norm (sup norm) over both
 * state dimensions and AD derivative levels.
 *
 * @tparam State Vector type (e.g., ublas::vector<F<F<double>>>)
 * @param v The state vector
 * @return max_{i,levels} |v[i]|
 *
 * @par Example
 * @code
 * ublas::vector<F<F<double>>> dxdt(3);
 * // ... compute derivatives ...
 * double max_rate = max_abs_all_levels_vec(dxdt);
 * // max_rate contains the largest |value| across:
 * //   - all 3 state components
 * //   - all first-order sensitivities
 * //   - all second-order sensitivities
 * @endcode
 */
template<class State>
inline double max_abs_all_levels_vec(const State& v)
{
  double m = 0.0;
  for (size_t i = 0; i < v.size(); ++i) {
    m = std::max(m, max_abs_all_levels(v[i]));
  }
  return m;
}

} // namespace detail

// Export to cppode namespace
using detail::max_abs_all_levels;
using detail::max_abs_all_levels_vec;

} // namespace cppode

#endif // CPPODE_FADIFF_EXTENSIONS_HPP
