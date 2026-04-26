/*
 Expression Templates for cppode::dual<T, N>  (any N, T non-AD).

 Eliminates intermediate temporaries / allocations in deep RHS expressions like
   dxdt[i] = a*b + c*d - e/f;
 by making operators return tiny CRTP proxy nodes (BinExpr, UnaryExpr, ...).
 The whole tree is materialized once on assignment into the LHS:
   1. lhs.val_ = root.val();                 // single value-pass (cached
                                             //   during ctor of each node)
   2. (N == 0) lhs.set_depend_size(...);     // one arena alloc, or in-place
      (N >  0) lhs.depend_ = true;           // (statically-sized; no alloc)
   3. for (i)  lhs.tan_[i] = root.tan(i);    // one fused chain-rule loop

 Coverage:
   - dual<T, 0> (heap, arena-backed) — eliminates the per-binary-op arena
     bump + the per-element `*this = *this + o` synthesis-temp leak.
   - dual<T, N> with N > 0 (stack, inline tan_[N]) — eliminates per-binary-op
     352-byte temp duals and replaces N separate eager tangent loops with one
     fused, compile-time-bounded loop the optimiser unrolls / vectorises.
   - Nested AD (T = dual<U, M>): SKIPPED. The eager_dual_active gate
     (cppode_dual_math.hpp) keeps eager active when T itself is AD, so the
     outer layer of dual2nd uses the recursive eager path while the inner
     layer (T = double) routes through ETs.

 Aliasing safety in compound assignment:
   The pattern `*this = *this + o;` (cppode_dual_math.hpp operator+= path) binds
   `*this + o` to an Expr that holds a const-ref to *this. Two layers protect
   against aliasing during materialisation:
     (1) DualLeaf snapshots `.x()` and the `tan_` pointer at construction so a
         later set_depend_size() on the LHS (N==0 case, which allocates a
         fresh tan_ buffer when transitioning from non-depend to depend)
         doesn't make the leaf observe freshly-allocated, uninitialised arena
         memory. For N>0 there is no allocation, but the snapshot is still
         valid: the inline tan_[N] address is stable for the assignment.
     (2) BinExpr/UnaryExpr cache `y_` (and `fp_` for unary) at ctor, so
         val_ = root.val() in the materialiser uses the OLD value of *this.
   Tangent writes still happen at index i AFTER the read of leaf.tan(i) for
   the same i, so per-i in-place writes remain safe (forward-mode tangents
   are index-independent).

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_DUAL_EXPR_HPP
#define CPPODE_DUAL_EXPR_HPP

#include <cppode/cppode_dual.hpp>
#include <cppode/cppode_ad_traits.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

namespace cppode {
namespace expr {

// =============================================================================
// CRTP base
// =============================================================================
template<class Derived>
struct Expr {
  const Derived& self() const { return *static_cast<const Derived*>(this); }
};

// =============================================================================
// Trait: is X a CRTP-Expr node?
// =============================================================================
template<class X, class = void>
struct is_dual_expr : std::false_type {};

template<class X>
struct is_dual_expr<X, std::void_t<typename std::enable_if<
    std::is_base_of<Expr<X>, X>::value
>::type>> : std::true_type {};

template<class X>
constexpr bool is_dual_expr_v = is_dual_expr<X>::value;

// Always-inline marker for ET hot methods. Deep BinExpr trees (e.g. TSIT5's
// 7-term error-estimate axpy) bottom out as 10+ nested template calls per
// tan(i); without this hint, gcc -O2's inlining budget can back off and emit
// real call/return on inner nodes, which destroys the fused-loop win.
#if defined(__GNUC__) || defined(__clang__)
  #define CPPODE_ET_INLINE __attribute__((always_inline)) inline
#else
  #define CPPODE_ET_INLINE inline
#endif

// =============================================================================
// Trait: is X a dual<T, N> with non-AD T (= ET-eligible operand)?
// =============================================================================
template<class X> struct is_dual_nonad : std::false_type {};
template<class T, unsigned N> struct is_dual_nonad<dual<T, N>>
  : std::bool_constant<!ad_traits::is_ad<T>::value> {};

// =============================================================================
// Trait: does X trigger the ET path? (operand is an Expr or an eligible dual)
// =============================================================================
template<class X> struct is_et_operand
  : std::bool_constant<is_dual_expr_v<X> || is_dual_nonad<X>::value> {};

// =============================================================================
// Leaf: dual<T, N> reference. Snapshots .x() and the tan_ pointer at ctor so
// compound assignments like  *this = *this + o  see the OLD val/tan even after
// materialisation has set this->val_. For N == 0 the snapshot also pins the
// pre-allocation tan_ pointer (so leaf.tan(i) reads the OLD arena buffer
// even after set_depend_size has swapped *this->tan_ to a fresh allocation —
// without this, the bug was "first call OK, calls 2..N drift" because arena
// memory served on call ≥ 2 contained stale data while call 1 saw zeros from
// the very first malloc). For N > 0 there is no reallocation, but pinning
// the pointer at ctor lets the same code path serve both regimes.
// =============================================================================
template<class T, unsigned N = 0>
struct DualLeaf : Expr<DualLeaf<T, N>> {
  using value_type = T;
  T        val_;
  const T* tan_;     // nullptr if the source dual was non-depend at ctor

  CPPODE_ET_INLINE explicit DualLeaf(const dual<T, N>& d)
    : val_(d.x()),
      tan_(d.depend() ? &d[0] : nullptr)
  {}

  CPPODE_ET_INLINE T val() const { return val_; }
  static constexpr unsigned tan_size() { return N; }
  CPPODE_ET_INLINE bool depends() const { return tan_ != nullptr; }
  CPPODE_ET_INLINE T tan(unsigned i) const { return tan_ ? tan_[i] : T(); }
};

// Partial specialisation for N == 0: tan_size is runtime, captured at ctor.
template<class T>
struct DualLeaf<T, 0> : Expr<DualLeaf<T, 0>> {
  using value_type = T;
  T        val_;
  const T* tan_;     // nullptr if the source dual was non-depend at ctor
  unsigned sz_;

  CPPODE_ET_INLINE explicit DualLeaf(const dual<T, 0>& d)
    : val_(d.x()),
      tan_(d.size() > 0 ? &d[0] : nullptr),
      sz_(d.size())
  {}

  CPPODE_ET_INLINE T val() const { return val_; }
  CPPODE_ET_INLINE unsigned tan_size() const { return sz_; }
  CPPODE_ET_INLINE bool depends() const { return sz_ > 0; }
  CPPODE_ET_INLINE T tan(unsigned i) const { return tan_ ? tan_[i] : T(); }
};

// =============================================================================
// Leaf: scalar (captured by value)
// =============================================================================
template<class T>
struct ScalarLeaf : Expr<ScalarLeaf<T>> {
  using value_type = T;
  T s_;

  CPPODE_ET_INLINE explicit ScalarLeaf(T s) : s_(s) {}

  CPPODE_ET_INLINE T val() const { return s_; }
  static constexpr unsigned tan_size() { return 0u; }
  static constexpr bool depends() { return false; }
  CPPODE_ET_INLINE T tan(unsigned) const { return T(0); }
};

// =============================================================================
// Op tags — value() and tangent() static methods.
// tangent() takes the cached operand vals + per-index operand tangents + the
// node's own cached y_ where useful (Div).
// =============================================================================
struct AddOp {
  template<class T> static T value(const T& a, const T& b) { return a + b; }
  template<class T> static T tangent(const T&, const T&,
                                     const T& at, const T& bt, const T&) {
    return at + bt;
  }
};
struct SubOp {
  template<class T> static T value(const T& a, const T& b) { return a - b; }
  template<class T> static T tangent(const T&, const T&,
                                     const T& at, const T& bt, const T&) {
    return at - bt;
  }
};
struct MulOp {
  template<class T> static T value(const T& a, const T& b) { return a * b; }
  template<class T> static T tangent(const T& av, const T& bv,
                                     const T& at, const T& bt, const T&) {
    return av * bt + at * bv;
  }
};
struct DivOp {
  template<class T> static T value(const T& a, const T& b) { return a / b; }
  template<class T> static T tangent(const T&, const T& bv,
                                     const T& at, const T& bt, const T& y) {
    return (at - y * bt) / bv;
  }
};

// =============================================================================
// Binary node — captures operands by value (small proxies, cheap to copy).
// Caches result value y_ in ctor: turns N nested vals into one pass through
// the tree at construction, then per-tangent calls use cached y_ + recursive
// tan(i) into children. Mirrors the eager path's "compute fp once, loop
// tangents" structure (cppode_dual_math.hpp:256).
// =============================================================================
template<class L, class R, class Op>
struct BinExpr : Expr<BinExpr<L, R, Op>> {
  using value_type = typename L::value_type;
  L l_;
  R r_;
  value_type y_;
  unsigned   sz_;
  bool       dep_;

  CPPODE_ET_INLINE BinExpr(L l, R r)
    : l_(std::move(l)), r_(std::move(r)),
      y_(Op::value(l_.val(), r_.val())),
      sz_(std::max(l_.tan_size(), r_.tan_size())),
      dep_(l_.depends() || r_.depends())
  {}

  CPPODE_ET_INLINE value_type val()      const { return y_; }
  CPPODE_ET_INLINE unsigned   tan_size() const { return sz_; }
  CPPODE_ET_INLINE bool       depends()  const { return dep_; }
  CPPODE_ET_INLINE value_type tan(unsigned i) const {
    return Op::tangent(l_.val(), r_.val(), l_.tan(i), r_.tan(i), y_);
  }
};

// =============================================================================
// Unary node — caches both result value y_ and derivative coefficient fp_ in
// ctor. tan(i) is then a single multiply: fp_ * x_.tan(i). This mirrors the
// eager unary path's structure (cppode_dual_math.hpp:256: "const T fp = ...").
// Op tags implement: static void compute(xv, y_out, fp_out).
// =============================================================================
template<class X, class Op>
struct UnaryExpr : Expr<UnaryExpr<X, Op>> {
  using value_type = typename X::value_type;
  X x_;
  value_type y_;
  value_type fp_;

  CPPODE_ET_INLINE explicit UnaryExpr(X x) : x_(std::move(x)) {
    Op::compute(x_.val(), y_, fp_);
  }

  CPPODE_ET_INLINE value_type val()      const { return y_; }
  CPPODE_ET_INLINE unsigned   tan_size() const { return x_.tan_size(); }
  CPPODE_ET_INLINE bool       depends()  const { return x_.depends(); }
  CPPODE_ET_INLINE value_type tan(unsigned i) const { return fp_ * x_.tan(i); }
};

// =============================================================================
// Unary op tags — each defines compute(x_val, y_out, fp_out).
//
// Math functions are reached via ADL: bring the std overloads into scope so
// for T=double they resolve to std::, and for T=user-namespace they resolve
// via the user's overload. Same idiom as the eager path.
// =============================================================================
struct NegOp {
  template<class T> static void compute(const T& xv, T& y, T& fp) {
    y = -xv;
    fp = T(-1);
  }
};

#define CPPODE_DEFINE_ET_UNARY_OP(NAME, VAL_EXPR, FP_EXPR)                    \
  struct NAME ## Op {                                                         \
    template<class T> static void compute(const T& xv, T& y, T& fp) {         \
      using std::exp;   using std::log;   using std::sqrt;                    \
      using std::sin;   using std::cos;   using std::tan;                     \
      using std::asin;  using std::acos;  using std::atan;                    \
      using std::sinh;  using std::cosh;  using std::tanh;                    \
      using std::asinh; using std::acosh; using std::atanh;                   \
      y  = (VAL_EXPR);                                                        \
      fp = (FP_EXPR);                                                         \
    }                                                                         \
  }

CPPODE_DEFINE_ET_UNARY_OP(Exp,   exp(xv),    y);
CPPODE_DEFINE_ET_UNARY_OP(Log,   log(xv),    T(1) / xv);
CPPODE_DEFINE_ET_UNARY_OP(Sqrt,  sqrt(xv),   T(1) / (T(2) * y));
CPPODE_DEFINE_ET_UNARY_OP(Sin,   sin(xv),    cos(xv));
CPPODE_DEFINE_ET_UNARY_OP(Cos,   cos(xv),   -sin(xv));
CPPODE_DEFINE_ET_UNARY_OP(Tan,   tan(xv),    T(1) + y * y);
CPPODE_DEFINE_ET_UNARY_OP(Asin,  asin(xv),   T(1) / sqrt(T(1) - xv * xv));
CPPODE_DEFINE_ET_UNARY_OP(Acos,  acos(xv),  -T(1) / sqrt(T(1) - xv * xv));
CPPODE_DEFINE_ET_UNARY_OP(Atan,  atan(xv),   T(1) / (T(1) + xv * xv));
CPPODE_DEFINE_ET_UNARY_OP(Sinh,  sinh(xv),   cosh(xv));
CPPODE_DEFINE_ET_UNARY_OP(Cosh,  cosh(xv),   sinh(xv));
CPPODE_DEFINE_ET_UNARY_OP(Tanh,  tanh(xv),   T(1) - y * y);
CPPODE_DEFINE_ET_UNARY_OP(Asinh, asinh(xv),  T(1) / sqrt(xv * xv + T(1)));
CPPODE_DEFINE_ET_UNARY_OP(Acosh, acosh(xv),  T(1) / sqrt(xv * xv - T(1)));
CPPODE_DEFINE_ET_UNARY_OP(Atanh, atanh(xv),  T(1) / (T(1) - xv * xv));

// abs: derivative sign(x); 0 at x=0 (FADBAD parity, cppode_dual_math.hpp:371).
struct AbsOp {
  template<class T> static void compute(const T& xv, T& y, T& fp) {
    using std::abs;
    y  = abs(xv);
    fp = (xv > T(0)) ? T(1) : ((xv < T(0)) ? T(-1) : T(0));
  }
};

#undef CPPODE_DEFINE_ET_UNARY_OP

// =============================================================================
// pow nodes — three forms: pow(dual,dual), pow(dual,scalar), pow(scalar,dual).
// Cache fa, fb, y in ctor; tan(i) = fa*l_t + fb*r_t. For the mixed-scalar
// forms the missing operand contributes 0 to its tangent factor.
// =============================================================================

// pow(dual, dual)  →  y = a^b
//   ∂y/∂a = b * a^(b-1)
//   ∂y/∂b = log(a) * a^b = log(a) * y
template<class L, class R>
struct PowExprDD : Expr<PowExprDD<L, R>> {
  using value_type = typename L::value_type;
  L l_;
  R r_;
  value_type y_;
  value_type fa_;
  value_type fb_;
  unsigned   sz_;
  bool       dep_;

  PowExprDD(L l, R r) : l_(std::move(l)), r_(std::move(r)) {
    using std::pow; using std::log;
    const value_type av = l_.val();
    const value_type bv = r_.val();
    y_  = pow(av, bv);
    fa_ = l_.depends() ? bv * pow(av, bv - value_type(1)) : value_type(0);
    fb_ = r_.depends() ? log(av) * y_                     : value_type(0);
    sz_  = std::max(l_.tan_size(), r_.tan_size());
    dep_ = l_.depends() || r_.depends();
  }
  value_type val()      const { return y_; }
  unsigned   tan_size() const { return sz_; }
  bool       depends()  const { return dep_; }
  value_type tan(unsigned i) const {
    return fa_ * l_.tan(i) + fb_ * r_.tan(i);
  }
};

// pow(dual, scalar) — scalar exponent
template<class L>
struct PowExprDS : Expr<PowExprDS<L>> {
  using value_type = typename L::value_type;
  L l_;
  value_type bv_;
  value_type y_;
  value_type fa_;

  PowExprDS(L l, value_type b) : l_(std::move(l)), bv_(b) {
    using std::pow;
    const value_type av = l_.val();
    y_  = pow(av, bv_);
    fa_ = bv_ * pow(av, bv_ - value_type(1));
  }
  value_type val()      const { return y_; }
  unsigned   tan_size() const { return l_.tan_size(); }
  bool       depends()  const { return l_.depends(); }
  value_type tan(unsigned i) const { return fa_ * l_.tan(i); }
};

// pow(scalar, dual) — scalar base
template<class R>
struct PowExprSD : Expr<PowExprSD<R>> {
  using value_type = typename R::value_type;
  R r_;
  value_type av_;
  value_type y_;
  value_type fb_;

  PowExprSD(value_type a, R r) : r_(std::move(r)), av_(a) {
    using std::pow; using std::log;
    const value_type bv = r_.val();
    y_  = pow(av_, bv);
    fb_ = log(av_) * y_;
  }
  value_type val()      const { return y_; }
  unsigned   tan_size() const { return r_.tan_size(); }
  bool       depends()  const { return r_.depends(); }
  value_type tan(unsigned i) const { return fb_ * r_.tan(i); }
};

// (pow free-function overloads are defined after the make_leaf / et_pair_target
// helpers — see further below in this header.)

// =============================================================================
// to_expr() — wrap any operand into an ET node.
// =============================================================================
// Already an Expr: pass through (return by value of the derived type).
template<class D>
inline auto to_expr(const Expr<D>& e) -> D { return e.self(); }

// dual<T, N> with non-AD T -> DualLeaf<T, N>.  Covers both heap (N == 0) and
// the static-N stack path through the unified leaf template above.
template<class T, unsigned N,
         std::enable_if_t<!ad_traits::is_ad<T>::value, int> = 0>
inline DualLeaf<T, N> to_expr(const dual<T, N>& d) { return DualLeaf<T, N>(d); }

// Arithmetic scalar -> ScalarLeaf<S>.
template<class S,
         std::enable_if_t<std::is_arithmetic<S>::value, int> = 0>
inline ScalarLeaf<S> to_expr(S s) { return ScalarLeaf<S>(s); }

// =============================================================================
// expr_value_type<X> — determine the result T from any wrappable operand.
// Used to constrain mixed dual+scalar so ScalarLeaf has a matching value_type.
// =============================================================================
template<class X, class = void>
struct expr_value_type { using type = void; };

template<class X>
struct expr_value_type<X, std::enable_if_t<is_dual_expr_v<X>>> {
  using type = typename X::value_type;
};

template<class T, unsigned N>
struct expr_value_type<dual<T, N>> { using type = T; };

template<class S>
struct expr_value_type<S, std::enable_if_t<std::is_arithmetic<S>::value>> {
  using type = S;
};

template<class X> using expr_value_type_t = typename expr_value_type<X>::type;

// =============================================================================
// Result type of to_expr<X> (for use in BinExpr template instantiations).
// =============================================================================
template<class X> struct to_expr_t_helper;
template<class D> struct to_expr_t_helper<Expr<D>> { using type = D; };
template<class T, unsigned N>
struct to_expr_t_helper<dual<T, N>> { using type = DualLeaf<T, N>; };
// scalar: caller picks ScalarLeaf<T> directly with the dual operand's T.

// Convenience: wrap a non-Expr operand in a leaf with a TARGET T (so the
// scalar's leaf type matches the dual's value_type when mixing). For an
// already-Expr operand we just return it.
template<class TT, class X,
         std::enable_if_t<is_dual_expr_v<X>, int> = 0>
inline X make_leaf(const X& x, TT* /*target_tag*/ = nullptr) { return x; }

template<class TT, class T, unsigned N>
inline DualLeaf<T, N> make_leaf(const dual<T, N>& d, TT* /*tag*/ = nullptr) {
  return DualLeaf<T, N>(d);
}

template<class TT, class S,
         std::enable_if_t<std::is_arithmetic<S>::value, int> = 0>
inline ScalarLeaf<TT> make_leaf(S s, TT* /*tag*/ = nullptr) {
  return ScalarLeaf<TT>(static_cast<TT>(s));
}

// =============================================================================
// Trait helpers for the operator overload SFINAE
// =============================================================================
// Both operands wrappable, AND at least one triggers the ET path,
// AND there's a meaningful T to align scalars to (deduced from the
// non-scalar operand's value_type).
// =============================================================================
template<class A, class B>
struct et_pair_target {
  using TA = expr_value_type_t<A>;
  using TB = expr_value_type_t<B>;
  // pick the non-arithmetic side's T as the target (scalar gets static_cast)
  using type =
    std::conditional_t<is_et_operand<A>::value, TA,
    std::conditional_t<is_et_operand<B>::value, TB, void>>;
};

template<class A, class B>
struct et_pair_enabled
  : std::bool_constant<
      (is_et_operand<A>::value || is_et_operand<B>::value)
      && !std::is_void<typename et_pair_target<A, B>::type>::value
    > {};

// =============================================================================
// Binary operator overloads — one template covers (dual op dual), (Expr op
// dual), (dual op Expr), (Expr op Expr), (dual op scalar), (scalar op dual),
// (Expr op scalar), (scalar op Expr) — eight cases per arithmetic op.
// =============================================================================
#define CPPODE_DEFINE_ET_BINOP(SYM, OPTAG)                                     \
  template<class A, class B,                                                   \
           std::enable_if_t<et_pair_enabled<A, B>::value, int> = 0>            \
  inline auto operator SYM(const A& a, const B& b) {                           \
    using TT = typename et_pair_target<A, B>::type;                            \
    auto la = make_leaf<TT>(a);                                                \
    auto lb = make_leaf<TT>(b);                                                \
    return BinExpr<decltype(la), decltype(lb), OPTAG>(std::move(la),           \
                                                       std::move(lb));         \
  }

CPPODE_DEFINE_ET_BINOP(+, AddOp)
CPPODE_DEFINE_ET_BINOP(-, SubOp)
CPPODE_DEFINE_ET_BINOP(*, MulOp)
CPPODE_DEFINE_ET_BINOP(/, DivOp)

#undef CPPODE_DEFINE_ET_BINOP

// =============================================================================
// Unary +/- on ET-eligible operands.
// =============================================================================
template<class A,
         std::enable_if_t<is_et_operand<A>::value, int> = 0>
inline auto operator+(const A& a) {
  using TT = expr_value_type_t<A>;
  return make_leaf<TT>(a);  // identity: pass through wrapped
}

template<class A,
         std::enable_if_t<is_et_operand<A>::value, int> = 0>
inline auto operator-(const A& a) {
  using TT = expr_value_type_t<A>;
  auto la = make_leaf<TT>(a);
  return UnaryExpr<decltype(la), NegOp>(std::move(la));
}

// =============================================================================
// Math functions — same shape: wrap operand via to_expr / make_leaf and return
// a UnaryExpr templated on the matching Op tag.
// =============================================================================
#define CPPODE_DEFINE_ET_MATH_FN(NAME, OPTAG)                                 \
  template<class A,                                                           \
           std::enable_if_t<is_et_operand<A>::value, int> = 0>                \
  inline auto NAME(const A& a) {                                              \
    using TT = expr_value_type_t<A>;                                          \
    auto la = make_leaf<TT>(a);                                               \
    return UnaryExpr<decltype(la), OPTAG>(std::move(la));                     \
  }

CPPODE_DEFINE_ET_MATH_FN(exp,   ExpOp)
CPPODE_DEFINE_ET_MATH_FN(log,   LogOp)
CPPODE_DEFINE_ET_MATH_FN(sqrt,  SqrtOp)
CPPODE_DEFINE_ET_MATH_FN(sin,   SinOp)
CPPODE_DEFINE_ET_MATH_FN(cos,   CosOp)
CPPODE_DEFINE_ET_MATH_FN(tan,   TanOp)
CPPODE_DEFINE_ET_MATH_FN(asin,  AsinOp)
CPPODE_DEFINE_ET_MATH_FN(acos,  AcosOp)
CPPODE_DEFINE_ET_MATH_FN(atan,  AtanOp)
CPPODE_DEFINE_ET_MATH_FN(sinh,  SinhOp)
CPPODE_DEFINE_ET_MATH_FN(cosh,  CoshOp)
CPPODE_DEFINE_ET_MATH_FN(tanh,  TanhOp)
CPPODE_DEFINE_ET_MATH_FN(asinh, AsinhOp)
CPPODE_DEFINE_ET_MATH_FN(acosh, AcoshOp)
CPPODE_DEFINE_ET_MATH_FN(atanh, AtanhOp)
CPPODE_DEFINE_ET_MATH_FN(abs,   AbsOp)

#undef CPPODE_DEFINE_ET_MATH_FN

// =============================================================================
// pow free-function overloads — three SFINAE-gated forms.
// (Definitions deferred to here so make_leaf / et_pair_target are visible.)
// =============================================================================

// pow(A, B) where both are ET-eligible operands (covers dual^dual, Expr^Expr,
// Expr^dual, dual^Expr).
template<class A, class B,
         std::enable_if_t<is_et_operand<A>::value && is_et_operand<B>::value, int> = 0>
inline auto pow(const A& a, const B& b) {
  using TT = typename et_pair_target<A, B>::type;
  auto la = make_leaf<TT>(a);
  auto lb = make_leaf<TT>(b);
  return PowExprDD<decltype(la), decltype(lb)>(std::move(la), std::move(lb));
}

// pow(ET-eligible, scalar)
template<class A, class S,
         std::enable_if_t<is_et_operand<A>::value
                          && std::is_arithmetic<S>::value, int> = 0>
inline auto pow(const A& a, const S& s) {
  using TT = expr_value_type_t<A>;
  auto la = make_leaf<TT>(a);
  return PowExprDS<decltype(la)>(std::move(la), static_cast<TT>(s));
}

// pow(scalar, ET-eligible)
template<class S, class B,
         std::enable_if_t<std::is_arithmetic<S>::value
                          && is_et_operand<B>::value, int> = 0>
inline auto pow(const S& s, const B& b) {
  using TT = expr_value_type_t<B>;
  auto lb = make_leaf<TT>(b);
  return PowExprSD<decltype(lb)>(static_cast<TT>(s), std::move(lb));
}

} // namespace expr
} // namespace cppode

// =============================================================================
// Hoist the ET operators into namespace cppode so codegen-emitted
// cppode::operator+ etc resolve them via ADL (the duals live in cppode).
// =============================================================================
namespace cppode {
using expr::operator+;
using expr::operator-;
using expr::operator*;
using expr::operator/;

// Math functions — codegen emits cppode::exp(...) etc., so make the ET
// overloads visible at namespace cppode (alongside the eager overloads in
// cppode_dual_math.hpp; SFINAE on the eager side keeps the (non-AD T, N=0)
// slice unambiguous).
using expr::exp;
using expr::log;
using expr::sqrt;
using expr::sin;
using expr::cos;
using expr::tan;
using expr::asin;
using expr::acos;
using expr::atan;
using expr::sinh;
using expr::cosh;
using expr::tanh;
using expr::asinh;
using expr::acosh;
using expr::atanh;
using expr::abs;
using expr::pow;
} // namespace cppode

// =============================================================================
// Out-of-line definitions for dual<T, 0>'s ET assignment / copy-ctor (declared
// in cppode_dual.hpp, defined here so Expr<D> is complete).
//
// These are only ever called when D is a CRTP Expr<D> derivative — for
// dual-to-dual or scalar assignment the existing non-template overloads in
// cppode_dual.hpp win the overload resolution because template arg deduction
// against `const Expr<D>&` fails for non-Expr operands.
// =============================================================================
namespace cppode {

template<class T>
template<class D>
inline dual<T, 0>& dual<T, 0>::operator=(const expr::Expr<D>& e) {
  const D& d = e.self();
  val_ = d.val();
  if (d.depends()) {
    set_depend_size(d.tan_size());
    for (unsigned i = 0; i < size_; ++i) tan_[i] = d.tan(i);
  } else if (size_ > 0) {
    for (unsigned i = 0; i < size_; ++i) tan_[i] = T();
  }
  return *this;
}

template<class T>
template<class D>
inline dual<T, 0>::dual(const expr::Expr<D>& e)
  : tan_(nullptr), size_(0), val_()
{
  *this = e;
}

// -- compound assignment from Expr -------------------------------------------
// Update val_ and tan_ in place; no temporary dual, no arena allocation.
// If *this has no tangent yet (size_==0) but the rhs depends on tangents,
// allocate once via set_depend_size; subsequent calls hit the in-place path.

template<class T>
template<class D>
inline dual<T, 0>& dual<T, 0>::operator+=(const expr::Expr<D>& e) {
  const D& d = e.self();
  val_ += d.val();
  if (d.depends()) {
    if (size_ == 0) {
      set_depend_size(d.tan_size());
      for (unsigned i = 0; i < size_; ++i) tan_[i] = d.tan(i);
    } else {
      assert(size_ == d.tan_size() && "dual<T,0>::operator+=(Expr): tan size mismatch");
      for (unsigned i = 0; i < size_; ++i) tan_[i] += d.tan(i);
    }
  }
  return *this;
}

template<class T>
template<class D>
inline dual<T, 0>& dual<T, 0>::operator-=(const expr::Expr<D>& e) {
  const D& d = e.self();
  val_ -= d.val();
  if (d.depends()) {
    if (size_ == 0) {
      set_depend_size(d.tan_size());
      for (unsigned i = 0; i < size_; ++i) tan_[i] = -d.tan(i);
    } else {
      assert(size_ == d.tan_size() && "dual<T,0>::operator-=(Expr): tan size mismatch");
      for (unsigned i = 0; i < size_; ++i) tan_[i] -= d.tan(i);
    }
  }
  return *this;
}

template<class T>
template<class D>
inline dual<T, 0>& dual<T, 0>::operator*=(const expr::Expr<D>& e) {
  const D& d = e.self();
  const T new_val = val_ * d.val();
  if (size_ > 0 && d.depends()) {
    assert(size_ == d.tan_size() && "dual<T,0>::operator*=(Expr): tan size mismatch");
    // (a*b)' = a'*b + a*b'  — uses CURRENT val_ in tan_, then update val_.
    for (unsigned i = 0; i < size_; ++i)
      tan_[i] = tan_[i] * d.val() + val_ * d.tan(i);
  } else if (size_ > 0) {
    for (unsigned i = 0; i < size_; ++i) tan_[i] *= d.val();
  } else if (d.depends()) {
    set_depend_size(d.tan_size());
    for (unsigned i = 0; i < size_; ++i) tan_[i] = val_ * d.tan(i);
  }
  val_ = new_val;
  return *this;
}

template<class T>
template<class D>
inline dual<T, 0>& dual<T, 0>::operator/=(const expr::Expr<D>& e) {
  const D& d = e.self();
  const T inv = T(1) / d.val();
  const T new_val = val_ * inv;
  if (size_ > 0 && d.depends()) {
    assert(size_ == d.tan_size() && "dual<T,0>::operator/=(Expr): tan size mismatch");
    // (a/b)' = (a' - (a/b)*b') / b
    for (unsigned i = 0; i < size_; ++i)
      tan_[i] = (tan_[i] - new_val * d.tan(i)) * inv;
  } else if (size_ > 0) {
    for (unsigned i = 0; i < size_; ++i) tan_[i] *= inv;
  } else if (d.depends()) {
    set_depend_size(d.tan_size());
    for (unsigned i = 0; i < size_; ++i) tan_[i] = -new_val * d.tan(i) * inv;
  }
  val_ = new_val;
  return *this;
}

// =============================================================================
// Out-of-line definitions for the static-N dual<T, N> (N > 0) ET assignment /
// ctor / compound-assignment templates (declared in cppode_dual.hpp).
//
// Loop bound is the compile-time constant N — the optimiser unrolls and SIMD-
// vectorises this for the typical n_sens = 32..64 stack widths. No allocation,
// no temp dual: the entire RHS Expr<D> tree materialises directly into the
// inline tan_[N] slots.
// =============================================================================

template<class T, unsigned N>
template<class D>
inline dual<T, N>& dual<T, N>::operator=(const expr::Expr<D>& e) {
  const D& d = e.self();
  val_ = d.val();
  if (d.depends()) {
    depend_ = true;
    for (unsigned i = 0; i < N; ++i) tan_[i] = d.tan(i);
  } else {
    depend_ = false;
  }
  return *this;
}

template<class T, unsigned N>
template<class D>
inline dual<T, N>::dual(const expr::Expr<D>& e) : val_(), depend_(false) {
  *this = e;
}

template<class T, unsigned N>
template<class D>
inline dual<T, N>& dual<T, N>::operator+=(const expr::Expr<D>& e) {
  const D& d = e.self();
  val_ += d.val();
  if (d.depends()) {
    if (depend_) {
      for (unsigned i = 0; i < N; ++i) tan_[i] += d.tan(i);
    } else {
      depend_ = true;
      for (unsigned i = 0; i < N; ++i) tan_[i] = d.tan(i);
    }
  }
  return *this;
}

template<class T, unsigned N>
template<class D>
inline dual<T, N>& dual<T, N>::operator-=(const expr::Expr<D>& e) {
  const D& d = e.self();
  val_ -= d.val();
  if (d.depends()) {
    if (depend_) {
      for (unsigned i = 0; i < N; ++i) tan_[i] -= d.tan(i);
    } else {
      depend_ = true;
      for (unsigned i = 0; i < N; ++i) tan_[i] = -d.tan(i);
    }
  }
  return *this;
}

template<class T, unsigned N>
template<class D>
inline dual<T, N>& dual<T, N>::operator*=(const expr::Expr<D>& e) {
  const D& d = e.self();
  const T new_val = val_ * d.val();
  if (depend_ && d.depends()) {
    // (a*b)' = a'*b + a*b'  — uses CURRENT val_ in tan_, then update val_.
    for (unsigned i = 0; i < N; ++i)
      tan_[i] = tan_[i] * d.val() + val_ * d.tan(i);
  } else if (depend_) {
    for (unsigned i = 0; i < N; ++i) tan_[i] *= d.val();
  } else if (d.depends()) {
    depend_ = true;
    for (unsigned i = 0; i < N; ++i) tan_[i] = val_ * d.tan(i);
  }
  val_ = new_val;
  return *this;
}

template<class T, unsigned N>
template<class D>
inline dual<T, N>& dual<T, N>::operator/=(const expr::Expr<D>& e) {
  const D& d = e.self();
  const T inv = T(1) / d.val();
  const T new_val = val_ * inv;
  if (depend_ && d.depends()) {
    // (a/b)' = (a' - (a/b)*b') / b
    for (unsigned i = 0; i < N; ++i)
      tan_[i] = (tan_[i] - new_val * d.tan(i)) * inv;
  } else if (depend_) {
    for (unsigned i = 0; i < N; ++i) tan_[i] *= inv;
  } else if (d.depends()) {
    depend_ = true;
    for (unsigned i = 0; i < N; ++i) tan_[i] = -new_val * d.tan(i) * inv;
  }
  val_ = new_val;
  return *this;
}

} // namespace cppode

#endif // CPPODE_DUAL_EXPR_HPP
