/*
 Non-arithmetic free functions and comparisons for cppode::dual2nd<T, N>.

 Arithmetic / transcendental operators have moved to cppode_dual2nd_expr.hpp
 where they build expression-template trees materialised on assignment.
 This header keeps:
   - min, max, clamp (selection operations: return one of the operands by
     value, no tangent propagation; ETs cannot easily express these because
     the result type is data-dependent on a runtime comparison).
   - Comparison operators (==, !=, <, <=, >, >=) which return bool and act
     on the innermost scalar.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_DUAL2ND_MATH_HPP
#define CPPODE_DUAL2ND_MATH_HPP

#include <cppode/cppode_dual2nd.hpp>

#include <cmath>
#include <type_traits>

namespace cppode {

// ============================================================================
//  min / max / clamp
// ============================================================================
template<class T, unsigned N>
inline dual2nd<T, N> min(const dual2nd<T, N>& a, const dual2nd<T, N>& b) {
  return (a.scalar() < b.scalar()) ? a : b;
}
template<class T, unsigned N>
inline dual2nd<T, N> max(const dual2nd<T, N>& a, const dual2nd<T, N>& b) {
  return (a.scalar() < b.scalar()) ? b : a;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline dual2nd<T, N> min(const dual2nd<T, N>& a, const U& b) {
  return (a.scalar() < static_cast<T>(b)) ? a : dual2nd<T, N>(b);
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline dual2nd<T, N> min(const U& a, const dual2nd<T, N>& b) { return min(b, a); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline dual2nd<T, N> max(const dual2nd<T, N>& a, const U& b) {
  return (a.scalar() < static_cast<T>(b)) ? dual2nd<T, N>(b) : a;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline dual2nd<T, N> max(const U& a, const dual2nd<T, N>& b) { return max(b, a); }

template<class T, unsigned N, class L, class H>
inline dual2nd<T, N> clamp(const dual2nd<T, N>& a, const L& lo, const H& hi) {
  return min(max(a, lo), hi);
}

// ============================================================================
//  Comparisons (act on the innermost scalar)
// ============================================================================
template<class T, unsigned N>
inline bool operator==(const dual2nd<T, N>& a, const dual2nd<T, N>& b) { return a.scalar() == b.scalar(); }
template<class T, unsigned N>
inline bool operator!=(const dual2nd<T, N>& a, const dual2nd<T, N>& b) { return a.scalar() != b.scalar(); }
template<class T, unsigned N>
inline bool operator< (const dual2nd<T, N>& a, const dual2nd<T, N>& b) { return a.scalar() <  b.scalar(); }
template<class T, unsigned N>
inline bool operator<=(const dual2nd<T, N>& a, const dual2nd<T, N>& b) { return a.scalar() <= b.scalar(); }
template<class T, unsigned N>
inline bool operator> (const dual2nd<T, N>& a, const dual2nd<T, N>& b) { return a.scalar() >  b.scalar(); }
template<class T, unsigned N>
inline bool operator>=(const dual2nd<T, N>& a, const dual2nd<T, N>& b) { return a.scalar() >= b.scalar(); }

template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator==(const dual2nd<T, N>& a, const U& b) { return a.scalar() == static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator==(const U& a, const dual2nd<T, N>& b) { return static_cast<T>(a) == b.scalar(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator!=(const dual2nd<T, N>& a, const U& b) { return a.scalar() != static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator!=(const U& a, const dual2nd<T, N>& b) { return static_cast<T>(a) != b.scalar(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator< (const dual2nd<T, N>& a, const U& b) { return a.scalar() <  static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator< (const U& a, const dual2nd<T, N>& b) { return static_cast<T>(a) <  b.scalar(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator<=(const dual2nd<T, N>& a, const U& b) { return a.scalar() <= static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator<=(const U& a, const dual2nd<T, N>& b) { return static_cast<T>(a) <= b.scalar(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator> (const dual2nd<T, N>& a, const U& b) { return a.scalar() >  static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator> (const U& a, const dual2nd<T, N>& b) { return static_cast<T>(a) >  b.scalar(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator>=(const dual2nd<T, N>& a, const U& b) { return a.scalar() >= static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator>=(const U& a, const dual2nd<T, N>& b) { return static_cast<T>(a) >= b.scalar(); }

} // namespace cppode

#endif // CPPODE_DUAL2ND_MATH_HPP
