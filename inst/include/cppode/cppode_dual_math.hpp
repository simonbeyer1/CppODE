/*
 Arithmetic operators, math functions, and comparisons for cppode::dual<T, N>.

 All defined in namespace cppode so generated codegen output `cppode::exp(x)`
 etc resolves via ADL or qualified call. Math functions use the `using std::fn`
 idiom to dispatch to std (T=double) or to user-namespace overloads (T=mpfr).

 Convention:
 - Result tangent for unary y = f(x):    y.tan[i] = f'(x.val) * x.tan[i]
 - Result tangent for binary y = f(a,b): y.tan[i] = f_a * a.tan[i] + f_b * b.tan[i]
 - Comparisons fall back to .x() (FADBAD-compatible).

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_DUAL_MATH_HPP
#define CPPODE_DUAL_MATH_HPP

#include <cppode/cppode_dual.hpp>
#include <cppode/cppode_ad_traits.hpp>

#include <cmath>
#include <type_traits>

namespace cppode {

// =============================================================================
// Eager-vs-ET routing helper. The expression-template overlay
// (cppode_dual_expr.hpp) covers the (non-AD T, N == 0) slice where heap
// allocations dominate cost. The eager operators below are SFINAE-gated to
// avoid that slice so the two paths don't ambiguously overlap.
//
// Triggered as:    template<class T, unsigned N, EAGER_GATE(T, N)> ...
// =============================================================================
namespace detail {
template<class T, unsigned N>
struct eager_dual_active
  : std::bool_constant<(ad_traits::is_ad<T>::value || N != 0)> {};
} // namespace detail

#define CPPODE_EAGER_GATE(T, N) \
  std::enable_if_t<::cppode::detail::eager_dual_active<T, N>::value, int> = 0

// =============================================================================
// Arithmetic operators (dual op dual)
// =============================================================================

template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> operator+(const dual<T, N>& a, const dual<T, N>& b) {
  dual<T, N> r(a.x() + b.x());
  if (!a.depend() && !b.depend()) return r;
  if (a.depend() && b.depend()) {
    r.set_depend_from(a, b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i] + b[i];
  } else if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i];
  } else {
    r.set_depend_from(b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = b[i];
  }
  return r;
}

template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> operator-(const dual<T, N>& a, const dual<T, N>& b) {
  dual<T, N> r(a.x() - b.x());
  if (!a.depend() && !b.depend()) return r;
  if (a.depend() && b.depend()) {
    r.set_depend_from(a, b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i] - b[i];
  } else if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i];
  } else {
    r.set_depend_from(b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = -b[i];
  }
  return r;
}

template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> operator*(const dual<T, N>& a, const dual<T, N>& b) {
  dual<T, N> r(a.x() * b.x());
  if (!a.depend() && !b.depend()) return r;
  if (a.depend() && b.depend()) {
    r.set_depend_from(a, b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i] * b.x() + a.x() * b[i];
  } else if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i] * b.x();
  } else {
    r.set_depend_from(b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a.x() * b[i];
  }
  return r;
}

template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> operator/(const dual<T, N>& a, const dual<T, N>& b) {
  T inv = T(1) / b.x();
  dual<T, N> r(a.x() * inv);
  if (!a.depend() && !b.depend()) return r;
  if (a.depend() && b.depend()) {
    r.set_depend_from(a, b);
    for (unsigned i = 0; i < r.loop_size(); ++i)
      r[i] = (a[i] - r.x() * b[i]) * inv;
  } else if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i] * inv;
  } else {
    r.set_depend_from(b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = -r.x() * b[i] * inv;
  }
  return r;
}

// Unary
template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> operator+(const dual<T, N>& a) { return a; }

template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> operator-(const dual<T, N>& a) {
  dual<T, N> r;
  r.x() = -a.x();
  if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = -a[i];
  }
  return r;
}

// =============================================================================
// Arithmetic operators (dual op scalar U / U op dual)
// =============================================================================

template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> operator+(const dual<T, N>& a, const U& b) {
  dual<T, N> r;
  r.x() = a.x() + static_cast<T>(b);
  if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i];
  }
  return r;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> operator+(const U& a, const dual<T, N>& b) { return b + a; }

template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> operator-(const dual<T, N>& a, const U& b) {
  dual<T, N> r;
  r.x() = a.x() - static_cast<T>(b);
  if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i];
  }
  return r;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> operator-(const U& a, const dual<T, N>& b) {
  dual<T, N> r;
  r.x() = static_cast<T>(a) - b.x();
  if (b.depend()) {
    r.set_depend_from(b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = -b[i];
  }
  return r;
}

template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> operator*(const dual<T, N>& a, const U& b) {
  dual<T, N> r;
  T tb = static_cast<T>(b);
  r.x() = a.x() * tb;
  if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i] * tb;
  }
  return r;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> operator*(const U& a, const dual<T, N>& b) { return b * a; }

template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> operator/(const dual<T, N>& a, const U& b) {
  dual<T, N> r;
  T inv = T(1) / static_cast<T>(b);
  r.x() = a.x() * inv;
  if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = a[i] * inv;
  }
  return r;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> operator/(const U& a, const dual<T, N>& b) {
  dual<T, N> r;
  T inv = T(1) / b.x();
  r.x() = static_cast<T>(a) * inv;
  if (b.depend()) {
    r.set_depend_from(b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = -r.x() * b[i] * inv;
  }
  return r;
}

// =============================================================================
// Compound assignment (member declarations resolve here)
// =============================================================================

template<class T, unsigned N>
inline dual<T, N>& dual<T, N>::operator+=(const dual<T, N>& o) {
  *this = *this + o; return *this;
}
template<class T, unsigned N>
inline dual<T, N>& dual<T, N>::operator-=(const dual<T, N>& o) {
  *this = *this - o; return *this;
}
template<class T, unsigned N>
inline dual<T, N>& dual<T, N>::operator*=(const dual<T, N>& o) {
  *this = *this * o; return *this;
}
template<class T, unsigned N>
inline dual<T, N>& dual<T, N>::operator/=(const dual<T, N>& o) {
  *this = *this / o; return *this;
}

template<class T>
inline dual<T, 0>& dual<T, 0>::operator+=(const dual<T, 0>& o) {
  *this = *this + o; return *this;
}
template<class T>
inline dual<T, 0>& dual<T, 0>::operator-=(const dual<T, 0>& o) {
  *this = *this - o; return *this;
}
template<class T>
inline dual<T, 0>& dual<T, 0>::operator*=(const dual<T, 0>& o) {
  *this = *this * o; return *this;
}
template<class T>
inline dual<T, 0>& dual<T, 0>::operator/=(const dual<T, 0>& o) {
  *this = *this / o; return *this;
}

// =============================================================================
// Math: unary functions  y = f(x), y.tan[i] = f'(x.val) * x.tan[i]
// =============================================================================

// VAL_EXPR and DERIV_EXPR may invoke any std math function (e.g. asin's
// derivative needs sqrt). Bringing the whole std-math suite into scope keeps
// ADL working for both T=double (resolves to std::) and T=user-namespace
// (resolves to that namespace via ADL).
#define CPPODE_DEFINE_UNARY(NAME, VAL_EXPR, DERIV_EXPR)                       \
  template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>                      \
  inline dual<T, N> NAME(const dual<T, N>& a) {                               \
    using std::exp;   using std::log;   using std::sqrt;                      \
    using std::sin;   using std::cos;   using std::tan;                       \
    using std::asin;  using std::acos;  using std::atan;                      \
    using std::sinh;  using std::cosh;  using std::tanh;                      \
    using std::asinh; using std::acosh; using std::atanh;                     \
    dual<T, N> r;                                                             \
    const T xv = a.x();                                                       \
    r.x() = (VAL_EXPR);                                                       \
    if (a.depend()) {                                                         \
      const T fp = (DERIV_EXPR);                                              \
      r.set_depend_from(a);                                                   \
      for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fp * a[i];               \
    }                                                                         \
    return r;                                                                 \
  }

CPPODE_DEFINE_UNARY(exp,  exp(xv),    r.x())
CPPODE_DEFINE_UNARY(log,  log(xv),    T(1) / xv)
CPPODE_DEFINE_UNARY(sqrt, sqrt(xv),   T(1) / (T(2) * r.x()))

// trig
template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> sin(const dual<T, N>& a) {
  using std::sin; using std::cos;
  dual<T, N> r;
  const T xv = a.x();
  r.x() = sin(xv);
  if (a.depend()) {
    const T fp = cos(xv);
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fp * a[i];
  }
  return r;
}
template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> cos(const dual<T, N>& a) {
  using std::sin; using std::cos;
  dual<T, N> r;
  const T xv = a.x();
  r.x() = cos(xv);
  if (a.depend()) {
    const T fp = -sin(xv);
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fp * a[i];
  }
  return r;
}
template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> tan(const dual<T, N>& a) {
  using std::tan;
  dual<T, N> r;
  const T xv = a.x();
  const T tv = tan(xv);
  r.x() = tv;
  if (a.depend()) {
    const T fp = T(1) + tv * tv;  // sec^2 = 1 + tan^2
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fp * a[i];
  }
  return r;
}

CPPODE_DEFINE_UNARY(asin, asin(xv),  T(1) / sqrt(T(1) - xv * xv))
CPPODE_DEFINE_UNARY(acos, acos(xv), -T(1) / sqrt(T(1) - xv * xv))
CPPODE_DEFINE_UNARY(atan, atan(xv),  T(1) / (T(1) + xv * xv))

// hyperbolic
template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> sinh(const dual<T, N>& a) {
  using std::sinh; using std::cosh;
  dual<T, N> r;
  const T xv = a.x();
  r.x() = sinh(xv);
  if (a.depend()) {
    const T fp = cosh(xv);
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fp * a[i];
  }
  return r;
}
template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> cosh(const dual<T, N>& a) {
  using std::sinh; using std::cosh;
  dual<T, N> r;
  const T xv = a.x();
  r.x() = cosh(xv);
  if (a.depend()) {
    const T fp = sinh(xv);
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fp * a[i];
  }
  return r;
}
template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> tanh(const dual<T, N>& a) {
  using std::tanh;
  dual<T, N> r;
  const T xv = a.x();
  const T tv = tanh(xv);
  r.x() = tv;
  if (a.depend()) {
    const T fp = T(1) - tv * tv;
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fp * a[i];
  }
  return r;
}

CPPODE_DEFINE_UNARY(asinh, asinh(xv), T(1) / sqrt(xv * xv + T(1)))
CPPODE_DEFINE_UNARY(acosh, acosh(xv), T(1) / sqrt(xv * xv - T(1)))
CPPODE_DEFINE_UNARY(atanh, atanh(xv), T(1) / (T(1) - xv * xv))

#undef CPPODE_DEFINE_UNARY

// =============================================================================
// abs: piecewise linear, derivative sign(x); at x=0 we return 0 (FADBAD parity)
// =============================================================================
template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> abs(const dual<T, N>& a) {
  using std::abs;
  dual<T, N> r;
  const T xv = a.x();
  r.x() = abs(xv);
  if (a.depend()) {
    const T fp = (xv > T(0)) ? T(1) : ((xv < T(0)) ? T(-1) : T(0));
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fp * a[i];
  }
  return r;
}

// =============================================================================
// pow: dual^dual, dual^scalar, scalar^dual
// y = a^b ⇒  dy = b * a^(b-1) * da + log(a) * a^b * db
// =============================================================================
template<class T, unsigned N, CPPODE_EAGER_GATE(T, N)>
inline dual<T, N> pow(const dual<T, N>& a, const dual<T, N>& b) {
  using std::pow; using std::log;
  dual<T, N> r;
  const T av = a.x();
  const T bv = b.x();
  const T y  = pow(av, bv);
  r.x() = y;
  if (!a.depend() && !b.depend()) return r;
  const T fa = (a.depend()) ? bv * pow(av, bv - T(1)) : T(0);
  const T fb = (b.depend()) ? log(av) * y              : T(0);
  if (a.depend() && b.depend()) {
    r.set_depend_from(a, b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fa * a[i] + fb * b[i];
  } else if (a.depend()) {
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fa * a[i];
  } else {
    r.set_depend_from(b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fb * b[i];
  }
  return r;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> pow(const dual<T, N>& a, const U& b) {
  using std::pow;
  dual<T, N> r;
  const T av = a.x();
  const T bv = static_cast<T>(b);
  r.x() = pow(av, bv);
  if (a.depend()) {
    const T fa = bv * pow(av, bv - T(1));
    r.set_depend_from(a);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fa * a[i];
  }
  return r;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>
                          && detail::eager_dual_active<T, N>::value, int> = 0>
inline dual<T, N> pow(const U& a, const dual<T, N>& b) {
  using std::pow; using std::log;
  dual<T, N> r;
  const T av = static_cast<T>(a);
  const T bv = b.x();
  const T y  = pow(av, bv);
  r.x() = y;
  if (b.depend()) {
    const T fb = log(av) * y;
    r.set_depend_from(b);
    for (unsigned i = 0; i < r.loop_size(); ++i) r[i] = fb * b[i];
  }
  return r;
}

// =============================================================================
// min / max / clamp: select-by-value, derivatives = winner's derivatives
// =============================================================================
template<class T, unsigned N>
inline dual<T, N> min(const dual<T, N>& a, const dual<T, N>& b) {
  return (a.x() < b.x()) ? a : b;
}
template<class T, unsigned N>
inline dual<T, N> max(const dual<T, N>& a, const dual<T, N>& b) {
  return (a.x() < b.x()) ? b : a;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline dual<T, N> min(const dual<T, N>& a, const U& b) {
  return (a.x() < static_cast<T>(b)) ? a : dual<T, N>(b);
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline dual<T, N> min(const U& a, const dual<T, N>& b) { return min(b, a); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline dual<T, N> max(const dual<T, N>& a, const U& b) {
  return (a.x() < static_cast<T>(b)) ? dual<T, N>(b) : a;
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline dual<T, N> max(const U& a, const dual<T, N>& b) { return max(b, a); }

template<class T, unsigned N, class L, class H>
inline dual<T, N> clamp(const dual<T, N>& a, const L& lo, const H& hi) {
  return min(max(a, lo), hi);
}

// =============================================================================
// Comparisons (always on .x() — FADBAD parity)
// =============================================================================
template<class T, unsigned N>
inline bool operator==(const dual<T, N>& a, const dual<T, N>& b) { return a.x() == b.x(); }
template<class T, unsigned N>
inline bool operator!=(const dual<T, N>& a, const dual<T, N>& b) { return a.x() != b.x(); }
template<class T, unsigned N>
inline bool operator< (const dual<T, N>& a, const dual<T, N>& b) { return a.x() <  b.x(); }
template<class T, unsigned N>
inline bool operator<=(const dual<T, N>& a, const dual<T, N>& b) { return a.x() <= b.x(); }
template<class T, unsigned N>
inline bool operator> (const dual<T, N>& a, const dual<T, N>& b) { return a.x() >  b.x(); }
template<class T, unsigned N>
inline bool operator>=(const dual<T, N>& a, const dual<T, N>& b) { return a.x() >= b.x(); }

template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator==(const dual<T, N>& a, const U& b) { return a.x() == static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator==(const U& a, const dual<T, N>& b) { return static_cast<T>(a) == b.x(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator!=(const dual<T, N>& a, const U& b) { return a.x() != static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator!=(const U& a, const dual<T, N>& b) { return static_cast<T>(a) != b.x(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator< (const dual<T, N>& a, const U& b) { return a.x() <  static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator< (const U& a, const dual<T, N>& b) { return static_cast<T>(a) <  b.x(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator<=(const dual<T, N>& a, const U& b) { return a.x() <= static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator<=(const U& a, const dual<T, N>& b) { return static_cast<T>(a) <= b.x(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator> (const dual<T, N>& a, const U& b) { return a.x() >  static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator> (const U& a, const dual<T, N>& b) { return static_cast<T>(a) >  b.x(); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator>=(const dual<T, N>& a, const U& b) { return a.x() >= static_cast<T>(b); }
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
inline bool operator>=(const U& a, const dual<T, N>& b) { return static_cast<T>(a) >= b.x(); }

} // namespace cppode

#endif // CPPODE_DUAL_MATH_HPP
