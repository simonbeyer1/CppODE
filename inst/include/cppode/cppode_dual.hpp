/*
 cppode::dual<T, N> — first-order forward-mode AD type.

 Two storage strategies, selected by N:
   N >  0  : inline storage `T tan_[N]` — zero-allocation, compile-time width.
   N == 0  : arena-backed pointer `T* tan_; unsigned size_;` — width fixed
             at runtime via .diff(idx, n). Memory comes from the thread-local
             cppode::dual_arena (cppode_dual_arena.hpp). Per-RHS-eval
             allocation cost is a single bump-pointer increment.

 API surface mirrors fadbad::F<T, N> so generated codegen output (`.x()`,
 `.diff(idx)`, `.diff(idx, n)`, `.d(j)`, `.size()`, `.depend()`,
 `operator[]`) compiles unchanged when the codegen swaps the type alias.

 This header defines the data class only. Arithmetic operators and math
 functions live in cppode_dual_math.hpp.

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
// non-trivial destructibility.
// -----------------------------------------------------------------------------
namespace detail {

template<class T>
inline T* arena_alloc_t(std::size_t n) {
  if constexpr (std::is_trivially_destructible_v<T>) {
    return dual_arena::arena().alloc_trivial<T>(n);
  } else {
    return dual_arena::arena().template alloc<T>(n);
  }
}

} // namespace detail

// =============================================================================
// Primary template: static N (inline storage). N must be > 0; N == 0 uses
// the partial specialization below.
// =============================================================================
template<class T = double, unsigned N = 0>
class dual {
  static_assert(N > 0, "primary template requires N > 0; N == 0 is specialized");

  T    val_;
  T    tan_[N];
  bool depend_;

public:
  using value_type   = T;
  static constexpr unsigned static_size = N;

  // -- constructors -----------------------------------------------------------
  dual() : val_(), depend_(false) {}

  template<class U,
           std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
  dual(const U& v) : val_(static_cast<T>(v)), depend_(false) {}

  dual(const dual& o) : val_(o.val_), depend_(o.depend_) {
    if (depend_) {
      for (unsigned i = 0; i < N; ++i) tan_[i] = o.tan_[i];
    }
  }

  dual& operator=(const dual& o) {
    if (this == &o) return *this;
    val_ = o.val_;
    if (o.depend_) {
      depend_ = true;
      for (unsigned i = 0; i < N; ++i) tan_[i] = o.tan_[i];
    } else {
      depend_ = false;
    }
    return *this;
  }

  template<class U,
           std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
  dual& operator=(const U& v) {
    val_ = static_cast<T>(v);
    depend_ = false;
    return *this;
  }

  // -- accessors --------------------------------------------------------------
  const T& x()   const { return val_; }
  T&       x()         { return val_; }
  const T& val() const { return val_; }   // FADBAD alias
  T&       val()       { return val_; }

  unsigned size()   const { return depend_ ? N : 0u; }
  bool     depend() const { return depend_; }
  // Loop bound for use INSIDE arithmetic operators after set_depend_from*().
  // Returns the compile-time constant N (constexpr-foldable for unrolling /
  // SIMD vectorisation). The runtime-checked size() is for callers that
  // need to honour the depend_ state.
  static constexpr unsigned loop_size() { return N; }

  // FADBAD-compatible: returns mutable reference, even when out-of-bounds
  // or !depend (returns ref to a thread-local zero, mirroring fadiff.h:179).
  // Codegen sometimes does `dual.d(j) = value` after .diff(idx) seeding.
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

  // Direct tangent access (caller must have called .diff() first).
  const T& operator[](unsigned j) const { return tan_[j]; }
  T&       operator[](unsigned j)       { return tan_[j]; }

  // -- seeding ---------------------------------------------------------------
  // Activate this dual as the idx-th independent variable: tan_[idx]=1, rest=0.
  T& diff(unsigned idx) {
    assert(idx < N && "diff(idx): idx out of compile-time bound N");
    depend_ = true;
    for (unsigned i = 0; i < N; ++i) tan_[i] = T();
    tan_[idx] = T(1);
    return tan_[idx];
  }
  // FADBAD-compatible 2-arg form: n must equal N at compile time.
  T& diff(unsigned idx, unsigned n) {
    assert(n == N && "diff(idx, n): n must equal compile-time N");
    return diff(idx);
  }

  // Internal: mark this dual as having an active tangent vector without
  // initializing it (operators fill the tangents themselves). Mirrors
  // FADBAD's setDepend(...) but with no size cross-checks.
  void set_depend()            { depend_ = true; }
  void set_depend_from(const dual&)               { depend_ = true; }
  void set_depend_from(const dual&, const dual&)  { depend_ = true; }

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
    val_  = static_cast<T>(v);
    tan_  = nullptr;
    size_ = 0;
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

  // -- accessors --------------------------------------------------------------
  const T& x()   const { return val_; }
  T&       x()         { return val_; }
  const T& val() const { return val_; }
  T&       val()       { return val_; }

  unsigned size()   const { return size_; }
  bool     depend() const { return size_ != 0; }
  // Mirrors dual<T,N>::loop_size() for the dynamic spec — see comment there.
  unsigned loop_size() const { return size_; }

  // FADBAD-compatible: mutable + const overloads, both returning a reference.
  // Out-of-bounds returns ref to a thread-local zero (FADBAD parity).
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
// outer-tangent entries — same as fadbad::F<fadbad::F<T,N>,N>). For dynamic
// width (N == 0) all storage is arena-backed.
//
// Naming-only alias — no new code; the recursive AD machinery (traits, LU,
// math) already handles nesting through the inner-layer template instantiation.
// =============================================================================
template<class T = double, unsigned N = 0>
using dual2nd = dual<dual<T, N>, N>;

} // namespace cppode

#endif // CPPODE_DUAL_HPP
