/*
 Expression Templates for cppode::dual2nd<T, N>.

 Mirrors the structure of cppode_dual_expr.hpp (first-order ETs) lifted to
 second order. Each ET node exposes:
   - eval_value()        -> T (innermost scalar)
   - eval_d1(i)          -> T  (gradient component i)
   - eval_d2(i, j)       -> T  (Hessian component, j <= i in callers)
   - any_depend()        -> bool
   - width()             -> unsigned (= N for static, runtime else)

 Operators on dual2nd<T,N> values, scalar mixes, and Expr2nd<D> trees all
 build BinExpr2 / BinScalarRight2 / BinScalarLeft2 / UnaryExpr2 / pow*
 nodes, never materialising an intermediate dual2nd. The full tree is
 materialised exactly once on assignment into a dual2nd<T,N> via
 dual2nd::operator=(const Expr2nd<D>&) (declared in cppode_dual2nd.hpp,
 defined here below).

 Aliasing safety for `*this = *this + o`:
 1. Leaves snapshot the scalar value at ctor and hold const-pointers /
    const-refs to the source storage. Tangent reads go through the source.
 2. The materialiser orders writes as:
       (a) scalar layer
       (b) lower-triangle Hessian (reads source's d1 + d2; writes inner
           tangent slots which do NOT alias the d1 storage)
       (c) gradient (writes outer.tan_[i].x() = d1 slots; reads source's d1
           at the SAME index, which has not yet been written this iteration)
       (d) mirror upper triangle from lower (intra-Hessian, no cross-leak)
       (e) sync redundant gradient (outer.val_.tan_[i] <- outer.tan_[i].x())
    Cross-iteration reads in (c) only touch source.d1 slots that have not
    yet been overwritten because we iterate i monotonically and write at
    index i after reading at index i.

 Coverage: binary +, -, *, /; scalar mixes; unary -; transcendentals exp,
 log, sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh,
 acosh, atanh; abs; pow (dual^dual, dual^scalar, scalar^dual). Comparisons
 stay in cppode_dual2nd_math.hpp (return bool, no tangent propagation).

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_DUAL2ND_EXPR_HPP
#define CPPODE_DUAL2ND_EXPR_HPP

#include <cppode/cppode_dual2nd.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

namespace cppode {
namespace expr2 {

#if defined(__GNUC__) || defined(__clang__)
  #define CPPODE_ET2_INLINE __attribute__((always_inline)) inline
#else
  #define CPPODE_ET2_INLINE inline
#endif

// ===========================================================================
// CRTP base for second-order expression nodes.
// ===========================================================================
template<class Derived>
struct Expr2nd {
  CPPODE_ET2_INLINE const Derived& self() const { return *static_cast<const Derived*>(this); }
};

template<class X, class = void>
struct is_expr2nd : std::false_type {};
template<class X>
struct is_expr2nd<X, std::void_t<typename std::enable_if<
    std::is_base_of<Expr2nd<X>, X>::value
>::type>> : std::true_type {};
template<class X>
constexpr bool is_expr2nd_v = is_expr2nd<X>::value;

template<class X> struct is_dual2nd_t : std::false_type {};
template<class T, unsigned N>
struct is_dual2nd_t<dual2nd<T, N>> : std::true_type {};

// ===========================================================================
// Leaf: dual2nd<T, N> reference. Snapshots scalar at ctor, reads tangents
// through the source on demand.
// ===========================================================================
template<class T, unsigned N>
struct Dual2ndLeaf : Expr2nd<Dual2ndLeaf<T, N>> {
  using value_type = T;
  static constexpr unsigned static_N = N;

  const dual2nd<T, N>& src_;
  T x_;
  bool depend_;
  unsigned m_;

  CPPODE_ET2_INLINE explicit Dual2ndLeaf(const dual2nd<T, N>& d)
    : src_(d), x_(d.scalar()), depend_(d.depend()),
      m_(d.depend() ? d.width() : 0u) {}

  CPPODE_ET2_INLINE T eval_value() const { return x_; }
  CPPODE_ET2_INLINE T eval_d1(unsigned i) const {
    return depend_ ? src_.d1_at(i) : T(0);
  }
  CPPODE_ET2_INLINE T eval_d2(unsigned i, unsigned j) const {
    return depend_ ? src_.dd_at(i, j) : T(0);
  }
  CPPODE_ET2_INLINE bool any_depend() const { return depend_; }
  CPPODE_ET2_INLINE unsigned width() const {
    if constexpr (N > 0) return N;
    else                  return m_;
  }
};

// Scalar leaf: arithmetic constant.
template<class T>
struct ScalarLeaf2 : Expr2nd<ScalarLeaf2<T>> {
  using value_type = T;
  static constexpr unsigned static_N = 0;
  T v_;
  CPPODE_ET2_INLINE explicit ScalarLeaf2(T v) : v_(v) {}
  CPPODE_ET2_INLINE T eval_value() const { return v_; }
  CPPODE_ET2_INLINE T eval_d1(unsigned)             const { return T(0); }
  CPPODE_ET2_INLINE T eval_d2(unsigned, unsigned)   const { return T(0); }
  CPPODE_ET2_INLINE bool any_depend() const { return false; }
  CPPODE_ET2_INLINE unsigned width() const { return 0u; }
};

// ===========================================================================
// Op tags for binary arithmetic. Each provides static methods:
//   value(av, bv)                       -> y
//   d1(av, bv, ai, bi, y)               -> dy/dt at index i
//   d2(av, bv, ai, bi, aj, bj, addij, bdij, y)  -> d2 y at (i, j)
// ===========================================================================
struct AddOp2 {
  template<class T> static T value(const T& a, const T& b) { return a + b; }
  template<class T> static T d1(const T&, const T&, const T& ai, const T& bi, const T&) {
    return ai + bi;
  }
  template<class T> static T d2(const T&, const T&,
                                const T&, const T&, const T&, const T&,
                                const T& addij, const T& bddij, const T&) {
    return addij + bddij;
  }
};
struct SubOp2 {
  template<class T> static T value(const T& a, const T& b) { return a - b; }
  template<class T> static T d1(const T&, const T&, const T& ai, const T& bi, const T&) {
    return ai - bi;
  }
  template<class T> static T d2(const T&, const T&,
                                const T&, const T&, const T&, const T&,
                                const T& addij, const T& bddij, const T&) {
    return addij - bddij;
  }
};
struct MulOp2 {
  template<class T> static T value(const T& a, const T& b) { return a * b; }
  template<class T> static T d1(const T& av, const T& bv,
                                const T& ai, const T& bi, const T&) {
    return ai * bv + av * bi;
  }
  template<class T> static T d2(const T& av, const T& bv,
                                const T& ai, const T& bi, const T& aj, const T& bj,
                                const T& addij, const T& bddij, const T&) {
    return addij * bv + av * bddij + ai * bj + aj * bi;
  }
};
struct DivOp2 {
  template<class T> static T value(const T& a, const T& b) { return a / b; }
  template<class T> static T d1(const T&, const T& bv,
                                const T& ai, const T& bi, const T& y) {
    return (ai - y * bi) / bv;
  }
  template<class T> static T d2(const T& av, const T& bv,
                                const T& ai, const T& bi, const T& aj, const T& bj,
                                const T& addij, const T& bddij, const T& y) {
    // y = a/b => y'' = (a'' - 2*y'*b' - y*b'') / b
    // For (i,j): use the symmetric variant
    //   yi = (ai - y*bi)/bv
    //   yj = (aj - y*bj)/bv
    //   y_ddij = (addij - yi*bj - yj*bi - y*bddij) / bv
    const T inv = T(1) / bv;
    const T yi = (ai - y * bi) * inv;
    const T yj = (aj - y * bj) * inv;
    return (addij - yi * bj - yj * bi - y * bddij) * inv;
  }
};

// ===========================================================================
// BinExpr2: dual2nd op dual2nd. Caches result scalar y_ at ctor.
// ===========================================================================
template<class L, class R, class Op>
struct BinExpr2 : Expr2nd<BinExpr2<L, R, Op>> {
  using value_type = typename L::value_type;
  static constexpr unsigned static_N =
    (L::static_N > R::static_N) ? L::static_N : R::static_N;

  L l_;
  R r_;
  value_type y_;
  unsigned   m_;
  bool       dep_;

  CPPODE_ET2_INLINE BinExpr2(L l, R r)
    : l_(std::move(l)), r_(std::move(r)),
      y_(Op::value(l_.eval_value(), r_.eval_value())),
      m_(std::max(l_.width(), r_.width())),
      dep_(l_.any_depend() || r_.any_depend()) {}

  CPPODE_ET2_INLINE value_type eval_value() const { return y_; }
  CPPODE_ET2_INLINE value_type eval_d1(unsigned i) const {
    return Op::d1(l_.eval_value(), r_.eval_value(),
                  l_.eval_d1(i), r_.eval_d1(i), y_);
  }
  CPPODE_ET2_INLINE value_type eval_d2(unsigned i, unsigned j) const {
    return Op::d2(l_.eval_value(), r_.eval_value(),
                  l_.eval_d1(i), r_.eval_d1(i),
                  l_.eval_d1(j), r_.eval_d1(j),
                  l_.eval_d2(i, j), r_.eval_d2(i, j), y_);
  }
  CPPODE_ET2_INLINE bool any_depend() const { return dep_; }
  CPPODE_ET2_INLINE unsigned width() const { return m_; }
};

// ===========================================================================
// Binary scalar-on-the-right: dual2nd op U. Tangents come only from the dual2nd.
// ===========================================================================
template<class L, class Op, class T>
struct BinScalarRight2 : Expr2nd<BinScalarRight2<L, Op, T>> {
  using value_type = T;
  static constexpr unsigned static_N = L::static_N;
  L l_;
  T s_;
  T y_;
  T inv_;     // only used by DivOp2 (1/s_)

  CPPODE_ET2_INLINE BinScalarRight2(L l, T s)
    : l_(std::move(l)), s_(s), y_(T()), inv_(T())
  {
    const T lv = l_.eval_value();
    if constexpr (std::is_same_v<Op, AddOp2>)      y_ = lv + s_;
    else if constexpr (std::is_same_v<Op, SubOp2>) y_ = lv - s_;
    else if constexpr (std::is_same_v<Op, MulOp2>) y_ = lv * s_;
    else if constexpr (std::is_same_v<Op, DivOp2>) { inv_ = T(1) / s_; y_ = lv * inv_; }
  }
  CPPODE_ET2_INLINE T eval_value() const { return y_; }
  CPPODE_ET2_INLINE T eval_d1(unsigned i) const {
    if constexpr (std::is_same_v<Op, AddOp2> || std::is_same_v<Op, SubOp2>)
      return l_.eval_d1(i);
    else if constexpr (std::is_same_v<Op, MulOp2>)
      return l_.eval_d1(i) * s_;
    else
      return l_.eval_d1(i) * inv_;
  }
  CPPODE_ET2_INLINE T eval_d2(unsigned i, unsigned j) const {
    if constexpr (std::is_same_v<Op, AddOp2> || std::is_same_v<Op, SubOp2>)
      return l_.eval_d2(i, j);
    else if constexpr (std::is_same_v<Op, MulOp2>)
      return l_.eval_d2(i, j) * s_;
    else
      return l_.eval_d2(i, j) * inv_;
  }
  CPPODE_ET2_INLINE bool any_depend() const { return l_.any_depend(); }
  CPPODE_ET2_INLINE unsigned width() const { return l_.width(); }
};

// Binary scalar-on-the-left: U op dual2nd. Tangents come from the dual2nd.
template<class R, class Op, class T>
struct BinScalarLeft2 : Expr2nd<BinScalarLeft2<R, Op, T>> {
  using value_type = T;
  static constexpr unsigned static_N = R::static_N;
  R r_;
  T s_;
  T y_;
  T inv_;     // only used by DivOp2 (1/r.value)

  CPPODE_ET2_INLINE BinScalarLeft2(T s, R r)
    : r_(std::move(r)), s_(s), y_(T()), inv_(T())
  {
    const T rv = r_.eval_value();
    if constexpr (std::is_same_v<Op, AddOp2>)      y_ = s_ + rv;
    else if constexpr (std::is_same_v<Op, SubOp2>) y_ = s_ - rv;
    else if constexpr (std::is_same_v<Op, MulOp2>) y_ = s_ * rv;
    else if constexpr (std::is_same_v<Op, DivOp2>) { inv_ = T(1) / rv; y_ = s_ * inv_; }
  }
  CPPODE_ET2_INLINE T eval_value() const { return y_; }
  CPPODE_ET2_INLINE T eval_d1(unsigned i) const {
    if constexpr (std::is_same_v<Op, AddOp2>) return r_.eval_d1(i);
    else if constexpr (std::is_same_v<Op, SubOp2>) return -r_.eval_d1(i);
    else if constexpr (std::is_same_v<Op, MulOp2>) return s_ * r_.eval_d1(i);
    else /* Div: y = s/r, y' = -y/r * r' */
      return -y_ * r_.eval_d1(i) * inv_;
  }
  CPPODE_ET2_INLINE T eval_d2(unsigned i, unsigned j) const {
    if constexpr (std::is_same_v<Op, AddOp2>) return r_.eval_d2(i, j);
    else if constexpr (std::is_same_v<Op, SubOp2>) return -r_.eval_d2(i, j);
    else if constexpr (std::is_same_v<Op, MulOp2>) return s_ * r_.eval_d2(i, j);
    else {
      // y = s/r, pb = -y/r, pbb = 2 y / r^2
      const T ri = r_.eval_d1(i);
      const T rj = r_.eval_d1(j);
      const T rd2 = r_.eval_d2(i, j);
      const T pb  = -y_ * inv_;
      const T pbb = T(2) * y_ * inv_ * inv_;
      return pb * rd2 + pbb * (ri * rj);
    }
  }
  CPPODE_ET2_INLINE bool any_depend() const { return r_.any_depend(); }
  CPPODE_ET2_INLINE unsigned width() const { return r_.width(); }
};

// ===========================================================================
// Unary node: caches y_ and derivative coefficients fp_, fpp_ at ctor.
// d1(i) = fp_ * a.d1(i), d2(i,j) = fp_ * a.d2(i,j) + fpp_ * a.d1(i) * a.d1(j).
// ===========================================================================
struct NegOp2 {
  template<class T> static void compute(const T& xv, T& y, T& fp, T& fpp) {
    y = -xv; fp = T(-1); fpp = T(0); (void)xv;
  }
};

#define CPPODE_DEFINE_ET2_UNARY(NAME, VAL_EXPR, FP_EXPR, FPP_EXPR)            \
  struct NAME ## Op2 {                                                        \
    template<class T> static void compute(const T& xv, T& y, T& fp, T& fpp) {  \
      using std::exp;   using std::log;   using std::sqrt;                    \
      using std::sin;   using std::cos;   using std::tan;                     \
      using std::asin;  using std::acos;  using std::atan;                    \
      using std::sinh;  using std::cosh;  using std::tanh;                    \
      using std::asinh; using std::acosh; using std::atanh;                   \
      y   = (VAL_EXPR);                                                       \
      fp  = (FP_EXPR);                                                        \
      fpp = (FPP_EXPR);                                                       \
    }                                                                         \
  }

CPPODE_DEFINE_ET2_UNARY(Exp,  exp(xv),  y,  y);
CPPODE_DEFINE_ET2_UNARY(Log,  log(xv),  T(1) / xv,  -T(1) / (xv * xv));
CPPODE_DEFINE_ET2_UNARY(Sqrt, sqrt(xv), T(1) / (T(2) * y),
                              -T(1) / (T(4) * y * y * y));
CPPODE_DEFINE_ET2_UNARY(Sin,  sin(xv),  cos(xv), -y);
CPPODE_DEFINE_ET2_UNARY(Cos,  cos(xv), -sin(xv), -y);
CPPODE_DEFINE_ET2_UNARY(Tan,  tan(xv),  T(1) + y * y,
                              T(2) * y * (T(1) + y * y));
CPPODE_DEFINE_ET2_UNARY(Asin, asin(xv),
                              T(1) / sqrt(T(1) - xv * xv),
                              xv / ((T(1) - xv * xv) * sqrt(T(1) - xv * xv)));
CPPODE_DEFINE_ET2_UNARY(Acos, acos(xv),
                             -T(1) / sqrt(T(1) - xv * xv),
                             -xv / ((T(1) - xv * xv) * sqrt(T(1) - xv * xv)));
CPPODE_DEFINE_ET2_UNARY(Atan, atan(xv),
                              T(1) / (T(1) + xv * xv),
                             -T(2) * xv / ((T(1) + xv * xv) * (T(1) + xv * xv)));
CPPODE_DEFINE_ET2_UNARY(Sinh, sinh(xv), cosh(xv), y);
CPPODE_DEFINE_ET2_UNARY(Cosh, cosh(xv), sinh(xv), y);
CPPODE_DEFINE_ET2_UNARY(Tanh, tanh(xv), T(1) - y * y,
                             -T(2) * y * (T(1) - y * y));
CPPODE_DEFINE_ET2_UNARY(Asinh, asinh(xv),
                               T(1) / sqrt(xv * xv + T(1)),
                              -xv / ((xv * xv + T(1)) * sqrt(xv * xv + T(1))));
CPPODE_DEFINE_ET2_UNARY(Acosh, acosh(xv),
                               T(1) / sqrt(xv * xv - T(1)),
                              -xv / ((xv * xv - T(1)) * sqrt(xv * xv - T(1))));
CPPODE_DEFINE_ET2_UNARY(Atanh, atanh(xv),
                               T(1) / (T(1) - xv * xv),
                               T(2) * xv / ((T(1) - xv * xv) * (T(1) - xv * xv)));

// abs: piecewise; at x=0 set fp=fpp=0.
struct AbsOp2 {
  template<class T> static void compute(const T& xv, T& y, T& fp, T& fpp) {
    using std::abs;
    y = abs(xv);
    fp = (xv > T(0)) ? T(1) : ((xv < T(0)) ? T(-1) : T(0));
    fpp = T(0);
  }
};

#undef CPPODE_DEFINE_ET2_UNARY

template<class A, class Op>
struct UnaryExpr2 : Expr2nd<UnaryExpr2<A, Op>> {
  using value_type = typename A::value_type;
  static constexpr unsigned static_N = A::static_N;
  A a_;
  value_type y_;
  value_type fp_;
  value_type fpp_;

  CPPODE_ET2_INLINE explicit UnaryExpr2(A a) : a_(std::move(a)) {
    Op::compute(a_.eval_value(), y_, fp_, fpp_);
  }
  CPPODE_ET2_INLINE value_type eval_value() const { return y_; }
  CPPODE_ET2_INLINE value_type eval_d1(unsigned i) const {
    return fp_ * a_.eval_d1(i);
  }
  CPPODE_ET2_INLINE value_type eval_d2(unsigned i, unsigned j) const {
    return fp_ * a_.eval_d2(i, j) + fpp_ * (a_.eval_d1(i) * a_.eval_d1(j));
  }
  CPPODE_ET2_INLINE bool any_depend() const { return a_.any_depend(); }
  CPPODE_ET2_INLINE unsigned width() const { return a_.width(); }
};

// ===========================================================================
// pow: three forms (dual^dual, dual^scalar, scalar^dual).
// y = a^b
// pa = b a^(b-1), pb = log(a) y
// paa = b(b-1) a^(b-2), pbb = log(a)^2 y, pab = a^(b-1) (1 + b log a)
// ===========================================================================
template<class L, class R>
struct PowExpr2DD : Expr2nd<PowExpr2DD<L, R>> {
  using value_type = typename L::value_type;
  static constexpr unsigned static_N =
    (L::static_N > R::static_N) ? L::static_N : R::static_N;

  L l_; R r_;
  value_type y_, pa_, pb_, paa_, pbb_, pab_;
  unsigned m_;
  bool dep_;

  CPPODE_ET2_INLINE PowExpr2DD(L l, R r)
    : l_(std::move(l)), r_(std::move(r))
  {
    using std::pow; using std::log;
    using T = value_type;
    const T av = l_.eval_value();
    const T bv = r_.eval_value();
    y_   = pow(av, bv);
    const T amB = pow(av, bv - T(1));
    const T la  = log(av);
    pa_  = l_.any_depend() ? bv * amB : T(0);
    pb_  = r_.any_depend() ? la * y_ : T(0);
    paa_ = l_.any_depend() ? bv * (bv - T(1)) * pow(av, bv - T(2)) : T(0);
    pbb_ = r_.any_depend() ? la * la * y_ : T(0);
    pab_ = (l_.any_depend() && r_.any_depend()) ? amB * (T(1) + bv * la) : T(0);
    m_   = std::max(l_.width(), r_.width());
    dep_ = l_.any_depend() || r_.any_depend();
  }
  CPPODE_ET2_INLINE value_type eval_value() const { return y_; }
  CPPODE_ET2_INLINE value_type eval_d1(unsigned i) const {
    return pa_ * l_.eval_d1(i) + pb_ * r_.eval_d1(i);
  }
  CPPODE_ET2_INLINE value_type eval_d2(unsigned i, unsigned j) const {
    const value_type li = l_.eval_d1(i), lj = l_.eval_d1(j);
    const value_type ri = r_.eval_d1(i), rj = r_.eval_d1(j);
    return pa_ * l_.eval_d2(i, j)
         + pb_ * r_.eval_d2(i, j)
         + paa_ * (li * lj)
         + pbb_ * (ri * rj)
         + pab_ * (li * rj + lj * ri);
  }
  CPPODE_ET2_INLINE bool any_depend() const { return dep_; }
  CPPODE_ET2_INLINE unsigned width() const { return m_; }
};

template<class L>
struct PowExpr2DS : Expr2nd<PowExpr2DS<L>> {
  using value_type = typename L::value_type;
  static constexpr unsigned static_N = L::static_N;
  L l_;
  value_type bv_, y_, pa_, paa_;

  CPPODE_ET2_INLINE PowExpr2DS(L l, value_type b)
    : l_(std::move(l)), bv_(b)
  {
    using std::pow;
    using T = value_type;
    const T av = l_.eval_value();
    y_   = pow(av, bv_);
    pa_  = bv_ * pow(av, bv_ - T(1));
    paa_ = bv_ * (bv_ - T(1)) * pow(av, bv_ - T(2));
  }
  CPPODE_ET2_INLINE value_type eval_value() const { return y_; }
  CPPODE_ET2_INLINE value_type eval_d1(unsigned i) const { return pa_ * l_.eval_d1(i); }
  CPPODE_ET2_INLINE value_type eval_d2(unsigned i, unsigned j) const {
    return pa_ * l_.eval_d2(i, j) + paa_ * (l_.eval_d1(i) * l_.eval_d1(j));
  }
  CPPODE_ET2_INLINE bool any_depend() const { return l_.any_depend(); }
  CPPODE_ET2_INLINE unsigned width() const { return l_.width(); }
};

template<class R>
struct PowExpr2SD : Expr2nd<PowExpr2SD<R>> {
  using value_type = typename R::value_type;
  static constexpr unsigned static_N = R::static_N;
  R r_;
  value_type av_, y_, pb_, pbb_;

  CPPODE_ET2_INLINE PowExpr2SD(value_type a, R r)
    : r_(std::move(r)), av_(a)
  {
    using std::pow; using std::log;
    using T = value_type;
    const T bv = r_.eval_value();
    y_   = pow(av_, bv);
    const T la = log(av_);
    pb_  = la * y_;
    pbb_ = la * la * y_;
  }
  CPPODE_ET2_INLINE value_type eval_value() const { return y_; }
  CPPODE_ET2_INLINE value_type eval_d1(unsigned i) const { return pb_ * r_.eval_d1(i); }
  CPPODE_ET2_INLINE value_type eval_d2(unsigned i, unsigned j) const {
    return pb_ * r_.eval_d2(i, j) + pbb_ * (r_.eval_d1(i) * r_.eval_d1(j));
  }
  CPPODE_ET2_INLINE bool any_depend() const { return r_.any_depend(); }
  CPPODE_ET2_INLINE unsigned width() const { return r_.width(); }
};

// ===========================================================================
// Wrap operands into Expr2nd nodes.
// ===========================================================================
template<class D>
CPPODE_ET2_INLINE auto to_expr2(const Expr2nd<D>& e) -> D { return e.self(); }

template<class T, unsigned N>
CPPODE_ET2_INLINE Dual2ndLeaf<T, N> to_expr2(const dual2nd<T, N>& d) {
  return Dual2ndLeaf<T, N>(d);
}

template<class S, std::enable_if_t<std::is_arithmetic<S>::value, int> = 0>
CPPODE_ET2_INLINE ScalarLeaf2<S> to_expr2(S s) { return ScalarLeaf2<S>(s); }

// expr2_value_type<X>: deduce scalar T from any wrappable operand.
template<class X, class = void>
struct expr2_value_type { using type = void; };
template<class X>
struct expr2_value_type<X, std::enable_if_t<is_expr2nd_v<X>>> {
  using type = typename X::value_type;
};
template<class T, unsigned N>
struct expr2_value_type<dual2nd<T, N>> { using type = T; };
template<class S>
struct expr2_value_type<S, std::enable_if_t<std::is_arithmetic<S>::value>> {
  using type = S;
};
template<class X> using expr2_value_type_t = typename expr2_value_type<X>::type;

// Wrappable: dual2nd, Expr2nd, or arithmetic.
template<class X>
struct is_et2_operand
  : std::bool_constant<
      is_expr2nd_v<X> || is_dual2nd_t<X>::value
    > {};

// make_leaf2<TT>: wrap with target T, so scalar gets static_cast.
template<class TT, class X,
         std::enable_if_t<is_expr2nd_v<X>, int> = 0>
CPPODE_ET2_INLINE X make_leaf2(const X& x, TT* = nullptr) { return x; }

template<class TT, class T, unsigned N>
CPPODE_ET2_INLINE Dual2ndLeaf<T, N> make_leaf2(const dual2nd<T, N>& d, TT* = nullptr) {
  return Dual2ndLeaf<T, N>(d);
}

template<class TT, class S,
         std::enable_if_t<std::is_arithmetic<S>::value, int> = 0>
CPPODE_ET2_INLINE ScalarLeaf2<TT> make_leaf2(S s, TT* = nullptr) {
  return ScalarLeaf2<TT>(static_cast<TT>(s));
}

// Pair-enabled SFINAE: at least one operand triggers the ET2 path AND there
// is a meaningful scalar T to align to.
template<class A, class B>
struct et2_pair_target {
  using TA = expr2_value_type_t<A>;
  using TB = expr2_value_type_t<B>;
  using type =
    std::conditional_t<is_et2_operand<A>::value, TA,
    std::conditional_t<is_et2_operand<B>::value, TB, void>>;
};

template<class A, class B>
struct et2_pair_enabled
  : std::bool_constant<
      (is_et2_operand<A>::value || is_et2_operand<B>::value)
      && !std::is_void<typename et2_pair_target<A, B>::type>::value
    > {};

// ===========================================================================
// Binary operator overloads. Each binary op has 8 overloads covering all
// combinations of {dual2nd, Expr2nd, scalar} on each side. We use explicit
// concrete-type signatures (dual2nd<T,N> and Expr2nd<D>) rather than fully
// generic templates so partial ordering picks these over the eager nested
// dual<dual<T,N>,N> operators in cppode_dual_math.hpp (which would otherwise
// match dual2nd via base-class deduction since dual2nd inherits from
// dual<dual<T,N>,N>).
// ===========================================================================
#define CPPODE_DEFINE_ET2_BINOP(SYM, OPTAG)                                   \
  /* dual2nd op dual2nd */                                                    \
  template<class T, unsigned N>                                               \
  CPPODE_ET2_INLINE auto operator SYM(const dual2nd<T, N>& a,                 \
                                      const dual2nd<T, N>& b) {               \
    return BinExpr2<Dual2ndLeaf<T, N>, Dual2ndLeaf<T, N>, OPTAG>(              \
                Dual2ndLeaf<T, N>(a), Dual2ndLeaf<T, N>(b));                  \
  }                                                                            \
  /* dual2nd op Expr2nd */                                                    \
  template<class T, unsigned N, class D>                                      \
  CPPODE_ET2_INLINE auto operator SYM(const dual2nd<T, N>& a,                 \
                                      const Expr2nd<D>& b) {                  \
    return BinExpr2<Dual2ndLeaf<T, N>, D, OPTAG>(                             \
                Dual2ndLeaf<T, N>(a), b.self());                              \
  }                                                                            \
  /* Expr2nd op dual2nd */                                                    \
  template<class D, class T, unsigned N>                                      \
  CPPODE_ET2_INLINE auto operator SYM(const Expr2nd<D>& a,                    \
                                      const dual2nd<T, N>& b) {               \
    return BinExpr2<D, Dual2ndLeaf<T, N>, OPTAG>(                             \
                a.self(), Dual2ndLeaf<T, N>(b));                              \
  }                                                                            \
  /* Expr2nd op Expr2nd */                                                    \
  template<class L, class R>                                                  \
  CPPODE_ET2_INLINE auto operator SYM(const Expr2nd<L>& a,                    \
                                      const Expr2nd<R>& b) {                  \
    return BinExpr2<L, R, OPTAG>(a.self(), b.self());                         \
  }                                                                            \
  /* dual2nd op scalar */                                                     \
  template<class T, unsigned N, class U,                                      \
           std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>           \
  CPPODE_ET2_INLINE auto operator SYM(const dual2nd<T, N>& a, U b) {          \
    return BinScalarRight2<Dual2ndLeaf<T, N>, OPTAG, T>(                      \
                Dual2ndLeaf<T, N>(a), static_cast<T>(b));                     \
  }                                                                            \
  /* scalar op dual2nd */                                                     \
  template<class T, unsigned N, class U,                                      \
           std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>           \
  CPPODE_ET2_INLINE auto operator SYM(U a, const dual2nd<T, N>& b) {          \
    return BinScalarLeft2<Dual2ndLeaf<T, N>, OPTAG, T>(                       \
                static_cast<T>(a), Dual2ndLeaf<T, N>(b));                     \
  }                                                                            \
  /* Expr2nd op scalar */                                                     \
  template<class D, class U,                                                  \
           std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>           \
  CPPODE_ET2_INLINE auto operator SYM(const Expr2nd<D>& a, U b) {             \
    using T = typename D::value_type;                                         \
    return BinScalarRight2<D, OPTAG, T>(a.self(), static_cast<T>(b));         \
  }                                                                            \
  /* scalar op Expr2nd */                                                     \
  template<class U, class D,                                                  \
           std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>           \
  CPPODE_ET2_INLINE auto operator SYM(U a, const Expr2nd<D>& b) {             \
    using T = typename D::value_type;                                         \
    return BinScalarLeft2<D, OPTAG, T>(static_cast<T>(a), b.self());          \
  }

CPPODE_DEFINE_ET2_BINOP(+, AddOp2)
CPPODE_DEFINE_ET2_BINOP(-, SubOp2)
CPPODE_DEFINE_ET2_BINOP(*, MulOp2)
CPPODE_DEFINE_ET2_BINOP(/, DivOp2)

#undef CPPODE_DEFINE_ET2_BINOP

// ===========================================================================
// Unary +/- on dual2nd / Expr2nd.
// ===========================================================================
template<class T, unsigned N>
CPPODE_ET2_INLINE auto operator+(const dual2nd<T, N>& a) {
  return Dual2ndLeaf<T, N>(a);
}
template<class D>
CPPODE_ET2_INLINE auto operator+(const Expr2nd<D>& a) {
  return a.self();
}

template<class T, unsigned N>
CPPODE_ET2_INLINE auto operator-(const dual2nd<T, N>& a) {
  return UnaryExpr2<Dual2ndLeaf<T, N>, NegOp2>(Dual2ndLeaf<T, N>(a));
}
template<class D>
CPPODE_ET2_INLINE auto operator-(const Expr2nd<D>& a) {
  return UnaryExpr2<D, NegOp2>(a.self());
}

#define CPPODE_DEFINE_ET2_UNARY_FN(NAME, OPTAG)                               \
  template<class T, unsigned N>                                               \
  CPPODE_ET2_INLINE auto NAME(const dual2nd<T, N>& a) {                       \
    return UnaryExpr2<Dual2ndLeaf<T, N>, OPTAG>(Dual2ndLeaf<T, N>(a));        \
  }                                                                            \
  template<class D>                                                           \
  CPPODE_ET2_INLINE auto NAME(const Expr2nd<D>& a) {                          \
    return UnaryExpr2<D, OPTAG>(a.self());                                    \
  }

CPPODE_DEFINE_ET2_UNARY_FN(exp,   ExpOp2)
CPPODE_DEFINE_ET2_UNARY_FN(log,   LogOp2)
CPPODE_DEFINE_ET2_UNARY_FN(sqrt,  SqrtOp2)
CPPODE_DEFINE_ET2_UNARY_FN(sin,   SinOp2)
CPPODE_DEFINE_ET2_UNARY_FN(cos,   CosOp2)
CPPODE_DEFINE_ET2_UNARY_FN(tan,   TanOp2)
CPPODE_DEFINE_ET2_UNARY_FN(asin,  AsinOp2)
CPPODE_DEFINE_ET2_UNARY_FN(acos,  AcosOp2)
CPPODE_DEFINE_ET2_UNARY_FN(atan,  AtanOp2)
CPPODE_DEFINE_ET2_UNARY_FN(sinh,  SinhOp2)
CPPODE_DEFINE_ET2_UNARY_FN(cosh,  CoshOp2)
CPPODE_DEFINE_ET2_UNARY_FN(tanh,  TanhOp2)
CPPODE_DEFINE_ET2_UNARY_FN(asinh, AsinhOp2)
CPPODE_DEFINE_ET2_UNARY_FN(acosh, AcoshOp2)
CPPODE_DEFINE_ET2_UNARY_FN(atanh, AtanhOp2)
CPPODE_DEFINE_ET2_UNARY_FN(abs,   AbsOp2)

#undef CPPODE_DEFINE_ET2_UNARY_FN

// ===========================================================================
// pow: three forms with explicit dual2nd/Expr2nd parameters.
// ===========================================================================
template<class T, unsigned N>
CPPODE_ET2_INLINE auto pow(const dual2nd<T, N>& a, const dual2nd<T, N>& b) {
  return PowExpr2DD<Dual2ndLeaf<T, N>, Dual2ndLeaf<T, N>>(
              Dual2ndLeaf<T, N>(a), Dual2ndLeaf<T, N>(b));
}
template<class T, unsigned N, class D>
CPPODE_ET2_INLINE auto pow(const dual2nd<T, N>& a, const Expr2nd<D>& b) {
  return PowExpr2DD<Dual2ndLeaf<T, N>, D>(Dual2ndLeaf<T, N>(a), b.self());
}
template<class D, class T, unsigned N>
CPPODE_ET2_INLINE auto pow(const Expr2nd<D>& a, const dual2nd<T, N>& b) {
  return PowExpr2DD<D, Dual2ndLeaf<T, N>>(a.self(), Dual2ndLeaf<T, N>(b));
}
template<class L, class R>
CPPODE_ET2_INLINE auto pow(const Expr2nd<L>& a, const Expr2nd<R>& b) {
  return PowExpr2DD<L, R>(a.self(), b.self());
}
template<class T, unsigned N, class U,
         std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>
CPPODE_ET2_INLINE auto pow(const dual2nd<T, N>& a, U b) {
  return PowExpr2DS<Dual2ndLeaf<T, N>>(Dual2ndLeaf<T, N>(a), static_cast<T>(b));
}
template<class D, class U,
         std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>
CPPODE_ET2_INLINE auto pow(const Expr2nd<D>& a, U b) {
  using T = typename D::value_type;
  return PowExpr2DS<D>(a.self(), static_cast<T>(b));
}
template<class U, class T, unsigned N,
         std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>
CPPODE_ET2_INLINE auto pow(U a, const dual2nd<T, N>& b) {
  return PowExpr2SD<Dual2ndLeaf<T, N>>(static_cast<T>(a), Dual2ndLeaf<T, N>(b));
}
template<class U, class D,
         std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>
CPPODE_ET2_INLINE auto pow(U a, const Expr2nd<D>& b) {
  using T = typename D::value_type;
  return PowExpr2SD<D>(static_cast<T>(a), b.self());
}

} // namespace expr2

// Hoist operators and math functions into namespace cppode for ADL pickup.
using expr2::operator+;
using expr2::operator-;
using expr2::operator*;
using expr2::operator/;
using expr2::exp;
using expr2::log;
using expr2::sqrt;
using expr2::sin;
using expr2::cos;
using expr2::tan;
using expr2::asin;
using expr2::acos;
using expr2::atan;
using expr2::sinh;
using expr2::cosh;
using expr2::tanh;
using expr2::asinh;
using expr2::acosh;
using expr2::atanh;
using expr2::abs;
using expr2::pow;

// ===========================================================================
// Materialise an Expr2nd<D> tree into a dual2nd<T, N> at assignment.
//
// Aliasing-safe order:
//   1. scalar layer
//   2. lower-triangle Hessian (reads source d1 + d2; writes inner tan slots)
//   3. gradient (writes outer.tan_[i].x()) — d1 sources read at index i
//      before being overwritten at index i; cross-iteration safe
//   4. mirror upper triangle from lower
//   5. sync redundant gradient (outer.val_.tan_[i] <- d1)
// ===========================================================================
namespace expr2 {

template<class T, unsigned N, class E,
         std::enable_if_t<is_expr2nd_v<E>, int> = 0>
CPPODE_ET2_INLINE void materialise(dual2nd<T, N>& lhs, const E& e) {
  // 1. Scalar.
  lhs.scalar() = e.eval_value();

  if (!e.any_depend()) {
    // No tangents: leave lhs's tangent buffers untouched if previously armed
    // (the caller may want them zero; the eager primitive convention is to
    // leave them alone). For consistency with the eager dual2nd constructors
    // we also zero out d1 if lhs was already armed; but the eager copy ctor
    // does not so we follow base semantics.
    return;
  }

  unsigned m;
  if constexpr (N > 0) m = N;
  else                  m = e.width();
  if (m == 0) return;

  lhs.arm_full(m);

  // 2. Lower triangle Hessian (must come before d1 writes in case of self-aliasing).
  for (unsigned i = 0; i < m; ++i) {
    for (unsigned j = 0; j <= i; ++j)
      lhs.dd_at(i, j) = e.eval_d2(i, j);
  }
  // 3. Gradient.
  for (unsigned i = 0; i < m; ++i)
    lhs.d1_at(i) = e.eval_d1(i);
  // 4. Mirror upper triangle.
  for (unsigned i = 1; i < m; ++i)
    for (unsigned j = 0; j < i; ++j)
      lhs.dd_raw(j, i) = lhs.dd_raw(i, j);
  // (sync_d1_redundant DROPPED: LU now reads gradient from inline_d1 via
  // first_order_view. val_tan_block is no longer maintained in sync.)
}

} // namespace expr2

// ===========================================================================
// Out-of-line definitions for dual2nd's ET assignment / ctor (declared in
// cppode_dual2nd.hpp). The argument type `expr2::Expr2nd<D>&` only matches
// CRTP derivations of Expr2nd; dual2nd-from-dual2nd or scalar assignments
// take the inherited base class operators via overload resolution.
// ===========================================================================

template<class T, unsigned N>
template<class D>
CPPODE_ET2_INLINE
dual2nd<T, N>&
dual2nd<T, N>::operator=(const expr2::Expr2nd<D>& e) {
  expr2::materialise(*this, e.self());
  return *this;
}

template<class T, unsigned N>
template<class D>
CPPODE_ET2_INLINE
dual2nd<T, N>::dual2nd(const expr2::Expr2nd<D>& e) : base() {
  expr2::materialise(*this, e.self());
}

// Compound assignment helpers: build (*this op other) as a BinExpr2 and
// materialise the whole tree back into *this.
template<class T, unsigned N>
template<class D>
CPPODE_ET2_INLINE dual2nd<T, N>&
dual2nd<T, N>::operator+=(const expr2::Expr2nd<D>& e) {
  *this = *this + e;
  return *this;
}
template<class T, unsigned N>
template<class D>
CPPODE_ET2_INLINE dual2nd<T, N>&
dual2nd<T, N>::operator-=(const expr2::Expr2nd<D>& e) {
  *this = *this - e;
  return *this;
}
template<class T, unsigned N>
template<class D>
CPPODE_ET2_INLINE dual2nd<T, N>&
dual2nd<T, N>::operator*=(const expr2::Expr2nd<D>& e) {
  *this = *this * e;
  return *this;
}
template<class T, unsigned N>
template<class D>
CPPODE_ET2_INLINE dual2nd<T, N>&
dual2nd<T, N>::operator/=(const expr2::Expr2nd<D>& e) {
  *this = *this / e;
  return *this;
}

} // namespace cppode

#endif // CPPODE_DUAL2ND_EXPR_HPP
