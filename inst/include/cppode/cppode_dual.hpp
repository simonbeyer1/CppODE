/*
 cppode::dual<T, N> — first-order forward-mode AD type.

 Both static-N (N > 0) and dynamic-N (N == 0) specs share a single storage
 strategy: `T* tan_;` pointing either at a row of an externally-owned
 cppode::detail::tangent_slab block (set via rebind_storage), or at a buffer
 obtained from the thread-local cppode::dual_arena. Per-RHS-eval allocation
 cost for temporaries is a bump-pointer increment; rolled back LIFO via a
 cppode::dual_arena::scope guard at the RHS body level. Storage layout in
 std::vector<dual<T, N>> is therefore SoA at the tangent level (rows of one
 contiguous slab), enabling BLAS daxpy/dscal across the tangent block.

 The only structural difference between the two specs is that the static-N
 primary template has no runtime size_ field — N is a compile-time constant
 used as the loop bound (constexpr-foldable for unrolling / SIMD).

 Provides the standard accessor surface used by codegen output:
 `.x()`, `.diff(idx)`, `.diff(idx, n)`, `.d(j)`, `.size()`, `.depend()`,
 `operator[]`.

 This header defines the data class only. Arithmetic operators and math
 functions (eager path, gated to nested-AD T only) live in
 cppode_dual_math.hpp; the expression-template overlay (active for non-AD T)
 lives in cppode_dual_expr.hpp.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_DUAL_HPP
#define CPPODE_DUAL_HPP

#include <cppode/cppode_dual_arena.hpp>

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace cppode {

// Forward declaration of the ET CRTP base (cppode_dual_expr.hpp). dual<T, 0>
// declares a templated assignment / copy-ctor from any ET node so
// `dxdt[i] = a*b + c*d;` materialises the whole tree directly into dxdt[i]
// without intermediate arena allocations. Out-of-line definitions live in
// cppode_dual_expr.hpp.
namespace expr {
  template<class Derived> struct Expr;
}

// -----------------------------------------------------------------------------
// Internal helper: allocate T[n] from the TLS arena, dispatching trivial vs
// non-trivial destructibility AND trivial vs non-trivial default-construction.
//
// Three regimes:
//   1. Trivially destructible AND trivially default-constructible
//      (e.g. double, int): bump-alloc only, caller writes — caller MUST
//      write before reading (no zero-init of returned memory).
//   2. Trivially destructible BUT non-trivially default-constructible
//      (e.g. dual<double, N>: user-provided ctor zeroes tan_ pointer):
//      bump-alloc + per-element placement-new of default ctor. No dtor
//      tracking. Critical for nested AD: the outer layer of
//      dual<dual<double, N>, N> alloc-bumps an array of N inner duals;
//      each must have tan_ = nullptr after construction so that
//      subsequent set_depend_size() correctly detects "not yet bound" and
//      allocates a fresh tangent buffer.
//   3. Non-trivially destructible (e.g. boost::multiprecision::cpp_dec_float
//      — with a real dtor): full alloc + ctor + dtor-tracking path.
// -----------------------------------------------------------------------------
namespace detail {

template<class T>
inline T* arena_alloc_t(std::size_t n) {
  if constexpr (std::is_trivially_destructible_v<T>
                && std::is_trivially_default_constructible_v<T>) {
    return dual_arena::arena().alloc_trivial<T>(n);
  } else if constexpr (std::is_trivially_destructible_v<T>) {
    T* arr = dual_arena::arena().alloc_trivial<T>(n);
    for (std::size_t i = 0; i < n; ++i) ::new (static_cast<void*>(arr + i)) T();
    return arr;
  } else {
    return dual_arena::arena().template alloc<T>(n);
  }
}

} // namespace detail

// =============================================================================
// Primary template: static N (compile-time tangent width). N must be > 0;
// N == 0 uses the partial specialization below.
//
// Storage: T* tan_ pointing at either an externally-owned tangent_slab row
// (after rebind_storage), or a buffer allocated from cppode::dual_arena for
// temporaries. The compile-time N is the loop bound (constexpr-foldable),
// not a struct-inline storage size — that's what unifies this spec with the
// dynamic spec. std::vector<dual<T, N>> is then SoA at the tangent level,
// matching the dual<T, 0> heap path.
// =============================================================================
template<class T = double, unsigned N = 0>
class dual {
  static_assert(N > 0, "primary template requires N > 0; N == 0 is specialized");

  T    val_;
  T*   tan_;
  bool depend_;

public:
  using value_type   = T;
  static constexpr unsigned static_size = N;

  // -- constructors -----------------------------------------------------------
  dual() : val_(), tan_(nullptr), depend_(false) {}

  template<class U,
           std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
  dual(const U& v) : val_(static_cast<T>(v)), tan_(nullptr), depend_(false) {}

  dual(const dual& o) : val_(o.val_), tan_(nullptr), depend_(false) {
    if (o.depend_) {
      depend_ = true;
      tan_    = detail::arena_alloc_t<T>(N);
      for (unsigned i = 0; i < N; ++i) tan_[i] = o.tan_[i];
    }
  }

  dual& operator=(const dual& o) {
    if (this == &o) return *this;
    val_ = o.val_;
    if (o.depend_) {
      if (tan_ == nullptr) tan_ = detail::arena_alloc_t<T>(N);
      depend_ = true;
      for (unsigned i = 0; i < N; ++i) tan_[i] = o.tan_[i];
    } else {
      // Non-depend source: zero any existing tangents (preserves slab- or
      // arena-binding so subsequent .diff() / set_depend_size() / ET assigns
      // can reuse the buffer). depend_ goes to false — same semantics as
      // operator=(const U&): a non-depend assignment yields a non-depend
      // dual, even if tan_ stays allocated.
      if (tan_ != nullptr) {
        for (unsigned i = 0; i < N; ++i) tan_[i] = T();
      }
      depend_ = false;
    }
    return *this;
  }

  template<class U,
           std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
  dual& operator=(const U& v) {
    val_ = static_cast<T>(v);
    // Preserve any existing tan_ buffer (slab- or arena-bound) and zero the
    // tangent values — same semantics as dual<T,0>::operator=(const U&)
    // (cppode_dual.hpp dynamic-spec). Without this, slab-bound duals would
    // lose their binding on every `dual = scalar` assignment (e.g. codegen
    // state re-seeding `x[i] = paramsSEXP[i]` after the slab has been
    // primed).
    if (tan_ != nullptr) {
      for (unsigned i = 0; i < N; ++i) tan_[i] = T();
    }
    depend_ = false;
    return *this;
  }

  // Move ctor / move assignment: STEAL the tan_ pointer instead of allocating
  // a fresh buffer + copying. Critical for `dxdt[i] = a + b + c;` style
  // expressions where the rvalue chain on the rhs already owns a buffer.
  // (Implicit move ctor is suppressed because we declared a copy assignment
  // operator; declare both moves explicitly.)
  dual(dual&& o) noexcept
    : val_(std::move(o.val_)), tan_(o.tan_), depend_(o.depend_)
  {
    o.tan_    = nullptr;
    o.depend_ = false;
  }
  dual& operator=(dual&& o) noexcept {
    if (this == &o) return *this;
    val_ = std::move(o.val_);
    if (o.depend_) {
      if (tan_ == nullptr) {
        // Steal: cheap path, no allocation, no copy.
        tan_      = o.tan_;
        depend_   = true;
        o.tan_    = nullptr;
        o.depend_ = false;
      } else {
        // Pre-existing tangent buffer (likely slab-bound): copy values into
        // it. This preserves address stability for callers that hold
        // references to our tangent storage (slab is the canonical case).
        for (unsigned i = 0; i < N; ++i) tan_[i] = std::move(o.tan_[i]);
        depend_ = true;
      }
    } else if (tan_ != nullptr) {
      // RHS is non-depend, we have a buffer: zero the tangents (mirror
      // copy-assignment behaviour to avoid stale derivative values).
      for (unsigned i = 0; i < N; ++i) tan_[i] = T();
      depend_ = false;
    } else {
      depend_ = false;
    }
    return *this;
  }

  // -- accessors --------------------------------------------------------------
  const T& x()   const { return val_; }
  T&       x()         { return val_; }
  const T& val() const { return val_; }   // .x() alias
  T&       val()       { return val_; }

  unsigned size()   const { return depend_ ? N : 0u; }
  bool     depend() const { return depend_; }
  // Loop bound for use INSIDE arithmetic operators after set_depend_from*().
  // Returns the compile-time constant N (constexpr-foldable for unrolling /
  // SIMD vectorisation). The runtime-checked size() is for callers that
  // need to honour the depend_ state.
  static constexpr unsigned loop_size() { return N; }

  // Returns mutable reference, even when out-of-bounds or !depend
  // (returns ref to a thread-local zero). Codegen sometimes does
  // `dual.d(j) = value` after .diff(idx) seeding.
  T& d(unsigned j) {
    if (!depend_ || j >= N) {
      thread_local T zero{};
      zero = T();
      return zero;
    }
    return tan_[j];
  }
  const T& d(unsigned j) const {
    if (!depend_ || j >= N) {
      thread_local T zero{};
      zero = T();
      return zero;
    }
    return tan_[j];
  }

  // Direct tangent access (caller must have called .diff() / set_depend_size()
  // first to ensure tan_ is allocated; otherwise this dereferences nullptr).
  const T& operator[](unsigned j) const { return tan_[j]; }
  T&       operator[](unsigned j)       { return tan_[j]; }

  // -- seeding ---------------------------------------------------------------
  // Activate this dual as the idx-th independent variable: tan_[idx]=1, rest=0.
  T& diff(unsigned idx) {
    assert(idx < N && "diff(idx): idx out of compile-time bound N");
    if (tan_ == nullptr) tan_ = detail::arena_alloc_t<T>(N);
    depend_ = true;
    for (unsigned i = 0; i < N; ++i) tan_[i] = T();
    tan_[idx] = T(1);
    return tan_[idx];
  }
  // 2-arg form (kept for API uniformity with the dynamic spec): n must
  // equal N at compile time.
  T& diff(unsigned idx, unsigned n) {
    assert(n == N && "diff(idx, n): n must equal compile-time N");
    (void)n;
    return diff(idx);
  }

  // Internal: mark this dual as having an active tangent vector without
  // initializing values (operators / ET fill the tangents themselves). If
  // tan_ has not been bound yet, allocate a fresh buffer from the arena.
  void set_depend() {
    if (tan_ == nullptr) tan_ = detail::arena_alloc_t<T>(N);
    depend_ = true;
  }

  // Allocate-without-init helper used by the ET path (mirrors the dual<T,0>
  // member of the same name). The 0-arg form is the natural one for static-N
  // (size is the template parameter); the 1-arg form keeps API parity with
  // the dynamic spec for shared ET callers — n must equal N.
  void set_depend_size() { set_depend(); }
  void set_depend_size(unsigned n) {
    assert(n == N && "dual<T,N>::set_depend_size(n): n must equal compile-time N");
    (void)n;
    set_depend();
  }

  // Non-allocating bind: point tan_ at an externally-owned buffer of length N
  // (typically a row of cppode::detail::tangent_slab). The dual does not own
  // the buffer (it never frees tan_), so rebinding is safe as long as the
  // external owner keeps the buffer alive for the dual's remaining lifetime.
  // Sets depend_ = true: a slab-bound dual is always considered active, so
  // subsequent eager ops take the depending branch. Pre-prime fill of the
  // slab block is the slab owner's responsibility (tangent_slab::prime
  // zero-fills the storage on size).
  void rebind_storage(T* p, unsigned n) noexcept {
    assert(n == N && "dual<T,N>::rebind_storage: n must equal compile-time N");
    (void)n;
    tan_    = p;
    depend_ = true;
  }

  void set_depend_from(const dual&)              { set_depend(); }
  void set_depend_from(const dual&, const dual&) { set_depend(); }

  // Compound assignment operators (defined inline via free + / - / * / /).
  dual& operator+=(const dual& o);
  dual& operator-=(const dual& o);
  dual& operator*=(const dual& o);
  dual& operator/=(const dual& o);
  template<class U,
           std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  dual& operator+=(const U& v) { val_ += static_cast<T>(v); return *this; }
  template<class U,
           std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  dual& operator-=(const U& v) { val_ -= static_cast<T>(v); return *this; }
  template<class U,
           std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  dual& operator*=(const U& v) {
    val_ *= static_cast<T>(v);
    if (depend_) for (unsigned i = 0; i < N; ++i) tan_[i] *= static_cast<T>(v);
    return *this;
  }
  template<class U,
           std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  dual& operator/=(const U& v) {
    val_ /= static_cast<T>(v);
    if (depend_) for (unsigned i = 0; i < N; ++i) tan_[i] /= static_cast<T>(v);
    return *this;
  }

  // -- expression-template assignment / construction --------------------------
  // Materialises any CRTP Expr<D> tree directly into our tan_ slots:
  //   1) val_ = root.val();
  //   2) set_depend_size();                    (alloc once if not yet bound)
  //   3) for (i) tan_[i] = root.tan(i);        (constant N → unrolled / vec'd)
  // Definitions live in cppode_dual_expr.hpp (after Expr<D> is complete).
  template<class D>
  dual& operator=(const expr::Expr<D>& e);

  template<class D>
  dual(const expr::Expr<D>& e);

  template<class D>
  dual& operator+=(const expr::Expr<D>& e);
  template<class D>
  dual& operator-=(const expr::Expr<D>& e);
  template<class D>
  dual& operator*=(const expr::Expr<D>& e);
  template<class D>
  dual& operator/=(const expr::Expr<D>& e);
};

// =============================================================================
// Partial specialization: dynamic N (arena-backed storage).
// =============================================================================
template<class T>
class dual<T, 0> {
  T*       tan_;
  unsigned size_;
  T        val_;

public:
  using value_type   = T;
  static constexpr unsigned static_size = 0;

  // -- constructors -----------------------------------------------------------
  dual() : tan_(nullptr), size_(0), val_() {}

  template<class U,
           std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
  dual(const U& v) : tan_(nullptr), size_(0), val_(static_cast<T>(v)) {}

  dual(const dual& o) : tan_(nullptr), size_(0), val_(o.val_) {
    if (o.size_ > 0) {
      size_ = o.size_;
      tan_  = detail::arena_alloc_t<T>(size_);
      for (unsigned i = 0; i < size_; ++i) tan_[i] = o.tan_[i];
    }
  }

  dual& operator=(const dual& o) {
    if (this == &o) return *this;
    val_ = o.val_;
    if (o.size_ > 0) {
      if (size_ == 0) {
        size_ = o.size_;
        tan_  = detail::arena_alloc_t<T>(size_);
      }
      assert(size_ == o.size_ && "dual<T,0>: tangent-vector size mismatch");
      for (unsigned i = 0; i < size_; ++i) tan_[i] = o.tan_[i];
    } else if (size_ > 0) {
      for (unsigned i = 0; i < size_; ++i) tan_[i] = T();
    }
    return *this;
  }

  template<class U,
           std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
  dual& operator=(const U& v) {
    val_ = static_cast<T>(v);
    // Preserve any existing tan_ buffer (slab-bound or arena-bound) and just
    // zero the tangent values. Without this, slab-bound duals would lose
    // their binding on every `dual = scalar` assignment (e.g. codegen state
    // re-seeding `x[i] = paramsSEXP[i]` after the slab has been primed).
    // The semantic difference vs. setting size_=0 is negligible: a fully-
    // zero tangent vector contributes 0 to downstream chain rules, same as
    // a non-depend dual; subsequent ops do redundant zero-multiplies, but
    // the slab-bound code paths take the BLAS / fused-loop fast path anyway.
    if (size_ > 0) {
      for (unsigned i = 0; i < size_; ++i) tan_[i] = T();
    }
    return *this;
  }

  // Move ctor / move assignment: STEAL the arena pointer instead of allocating
  // a fresh buffer + copying. Critical for `dxdt[i] = a + b + c;` style
  // expressions, where the rvalue chain on the rhs already owns an arena
  // buffer. Saves one allocation per top-level expression.
  // (Implicit move ctor is suppressed because we declared a copy assignment
  // operator; declare both moves explicitly.)
  dual(dual&& o) noexcept : tan_(o.tan_), size_(o.size_), val_(std::move(o.val_)) {
    o.tan_ = nullptr;
    o.size_ = 0;
  }
  dual& operator=(dual&& o) noexcept {
    if (this == &o) return *this;
    val_ = std::move(o.val_);
    if (o.size_ > 0) {
      if (size_ == 0) {
        // Steal: cheap path, no allocation, no copy.
        tan_   = o.tan_;
        size_  = o.size_;
        o.tan_  = nullptr;
        o.size_ = 0;
      } else {
        // Pre-existing tangent buffer: copy values into it (preserves
        // address stability for callers that hold references).
        assert(size_ == o.size_ && "dual<T,0>: tangent-vector size mismatch in move-assign");
        for (unsigned i = 0; i < size_; ++i) tan_[i] = std::move(o.tan_[i]);
      }
    } else if (size_ > 0) {
      // RHS is non-depend, we have a buffer: zero the tangents (mirror
      // the copy-assignment behaviour to avoid stale derivative values).
      for (unsigned i = 0; i < size_; ++i) tan_[i] = T();
    }
    return *this;
  }

  // -- expression-template assignment / construction --------------------------
  // Materialises any CRTP Expr<D> tree directly into this dual:
  //   1) val_ = root.val();                    (one scalar pass through tree)
  //   2) set_depend_size(root.tan_size());     (one arena alloc, or in-place reuse)
  //   3) for (i) tan_[i] = root.tan(i);        (one fused chain-rule loop)
  // Definitions live in cppode_dual_expr.hpp (after Expr<D> is complete).
  template<class D>
  dual& operator=(const expr::Expr<D>& e);

  template<class D>
  dual(const expr::Expr<D>& e);

  // In-place compound assignment from any Expr<D>. Without these the
  // compiler picks operator+=(const dual&) etc. and synthesises a
  // BinExpr→dual temporary via the Expr ctor above — that temp allocates
  // a fresh tan_ buffer from the arena per element, which is the dominant
  // per-step leak in `vec_axpy(y, alpha, x)` style kernels (y[i] += alpha*x[i]).
  // These overloads update val_ and tan_ in place; no temp dual, no alloc.
  template<class D>
  dual& operator+=(const expr::Expr<D>& e);
  template<class D>
  dual& operator-=(const expr::Expr<D>& e);
  template<class D>
  dual& operator*=(const expr::Expr<D>& e);
  template<class D>
  dual& operator/=(const expr::Expr<D>& e);

  // -- accessors --------------------------------------------------------------
  const T& x()   const { return val_; }
  T&       x()         { return val_; }
  const T& val() const { return val_; }
  T&       val()       { return val_; }

  unsigned size()   const { return size_; }
  bool     depend() const { return size_ != 0; }
  // Mirrors dual<T,N>::loop_size() for the dynamic spec — see comment there.
  unsigned loop_size() const { return size_; }

  // Mutable + const overloads, both returning a reference.
  // Out-of-bounds returns ref to a thread-local zero.
  T& d(unsigned j) {
    if (j >= size_) {
      thread_local T zero{};
      zero = T();
      return zero;
    }
    return tan_[j];
  }
  const T& d(unsigned j) const {
    if (j >= size_) {
      thread_local T zero{};
      zero = T();
      return zero;
    }
    return tan_[j];
  }

  const T& operator[](unsigned j) const { return tan_[j]; }
  T&       operator[](unsigned j)       { return tan_[j]; }

  // -- seeding ----------------------------------------------------------------
  T& diff(unsigned idx, unsigned n) {
    assert(idx < n && "diff(idx, n): idx out of bounds");
    if (size_ == 0) {
      size_ = n;
      tan_  = detail::arena_alloc_t<T>(n);
    } else {
      assert(size_ == n && "diff(idx, n): n must match existing size");
    }
    for (unsigned i = 0; i < size_; ++i) tan_[i] = T();
    tan_[idx] = T(1);
    return tan_[idx];
  }

  // Allocate-without-init helper used by operators (fills tangents themselves).
  void set_depend_size(unsigned n) {
    if (size_ == 0) {
      size_ = n;
      tan_  = detail::arena_alloc_t<T>(n);
    } else {
      assert(size_ == n && "dual<T,0>: tangent size mismatch");
    }
  }

  // Non-allocating bind: point tan_ at an externally-owned buffer of length n
  // (typically a row of cppode::detail::tangent_slab). The dual does not own
  // the buffer (just like the arena-backed case — it never frees tan_), so
  // rebinding is safe as long as the external owner keeps the buffer alive
  // for the dual's remaining lifetime.
  void rebind_storage(T* p, unsigned n) noexcept {
    tan_  = p;
    size_ = n;
  }
  void set_depend_from(const dual& a) {
    set_depend_size(a.size_);
  }
  void set_depend_from(const dual& a, const dual& b) {
    assert(a.size_ == b.size_ && "operands have different tangent sizes");
    set_depend_size(a.size_);
  }

  // Compound assignment operators (free + / - / * / / defined in math header).
  dual& operator+=(const dual& o);
  dual& operator-=(const dual& o);
  dual& operator*=(const dual& o);
  dual& operator/=(const dual& o);
  template<class U,
           std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  dual& operator+=(const U& v) { val_ += static_cast<T>(v); return *this; }
  template<class U,
           std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  dual& operator-=(const U& v) { val_ -= static_cast<T>(v); return *this; }
  template<class U,
           std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  dual& operator*=(const U& v) {
    val_ *= static_cast<T>(v);
    if (size_) for (unsigned i = 0; i < size_; ++i) tan_[i] *= static_cast<T>(v);
    return *this;
  }
  template<class U,
           std::enable_if_t<std::is_arithmetic_v<U>, int> = 0>
  dual& operator/=(const U& v) {
    val_ /= static_cast<T>(v);
    if (size_) for (unsigned i = 0; i < size_; ++i) tan_[i] /= static_cast<T>(v);
    return *this;
  }
};

// =============================================================================
// dual2nd<T, N> — second-order forward AD via nested dual.
//
// Layout: dual<dual<T, N>, N>. The outer layer's tangent vector holds N copies
// of dual<T, N>, each of which carries its own gradient. Composition through
// arithmetic / math operators recursively propagates first-order chain rules
// at both layers, yielding a Hessian via H[i, j] = result.d(i).d(j).
//
// Memory cost: per number, N + N*(1+N) values of T (gradient duplicated across
// outer-tangent entries). For dynamic
// width (N == 0) all storage is arena-backed.
//
// Naming-only alias — no new code; the recursive AD machinery (traits, LU,
// math) already handles nesting through the inner-layer template instantiation.
// =============================================================================
template<class T = double, unsigned N = 0>
using dual2nd = dual<dual<T, N>, N>;

} // namespace cppode

#endif // CPPODE_DUAL_HPP
