/*
 cppode::dual2nd<T, N>: second-order forward-mode AD type.

 dual2nd publicly inherits from dual<dual<T, N>, N>. Inheritance preserves the
 storage layout and the full accessor surface (.x(), .d(j), .size(), .depend(),
 .diff(), operator[]) so that the generic AD machinery (cppode_ad_lu.hpp,
 cppode_newton.hpp, cppode_integrate_times.hpp, the LU IFT recursion in
 cppode_ad_traits.hpp) continues to work unchanged: a dual2nd is-a nested
 dual.

 The reason for a distinct type is dispatch. Math primitives in
 cppode_dual2nd_math.hpp are templated on dual2nd<T, N> specifically and
 exploit Hessian symmetry by computing only the lower triangle (j <= i) and
 mirroring to the upper. Function-template argument deduction is strict: the
 dual2nd-specific operators match dual2nd exactly and do NOT match the base
 dual<dual<T, N>, N> via subclass slicing. Conversely, the eager nested-dual
 operators in cppode_dual_math.hpp deduce on dual<dual<T, N>, N> and do NOT
 match dual2nd. The two operator sets are unambiguous.

 Convenience accessors d1_at(i) / dd_at(i, j) translate to the underlying
 nested-dual storage:
   - d1_at(i) -> outer.tan_[i].x()  (gradient slot, also redundantly stored
                                     in outer.val_.tan_[i] for LU-IFT
                                     correctness; primitives mirror writes)
   - dd_at(i, j) -> outer.tan_[max(i,j)].tan_[min(i,j)]  (canonical lower
                                     triangle, fed by symmetric computation)

 Storage layout for now matches dual<dual<T, N>, N> (full N x N inner-tangent
 block, gradient redundantly stored across both outer-val and outer-tan).
 The compute saving comes from math primitives only filling the lower
 triangle and mirroring at the end of each operation. Storage compaction to
 a packed N(N+1)/2 Hessian is a follow-up that requires reworking the LU
 IFT extraction pipeline (the recursive layer-by-layer model assumes .x()
 peels exactly one AD layer; a packed dual2nd peels two).

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_DUAL2ND_HPP
#define CPPODE_DUAL2ND_HPP

#include <cppode/cppode_dual.hpp>

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace cppode {

// Forward-declare the ET CRTP base so dual2nd can declare templated
// assignment / construction from any Expr2nd<D> tree. Definitions live in
// cppode_dual2nd_expr.hpp.
namespace expr2 {
  template<class Derived> struct Expr2nd;
}

// ----------------------------------------------------------------------------
// Triangular indexing helpers. tri_idx canonicalises (i, j) to the lower
// triangle and returns the packed offset; tri_size(N) is the total entry
// count in the lower triangle. Used by math primitives that iterate the
// symmetric Hessian in canonical order (j <= i).
// ----------------------------------------------------------------------------
namespace detail {
constexpr inline unsigned tri_idx(unsigned i, unsigned j) noexcept {
  return (j <= i) ? (i * (i + 1u) / 2u + j) : (j * (j + 1u) / 2u + i);
}
constexpr inline unsigned tri_size(unsigned n) noexcept {
  return n * (n + 1u) / 2u;
}
} // namespace detail

// =============================================================================
// dual2nd<T, N>: distinct type, public-inheritance refinement of nested dual.
// All storage and standard accessors are inherited; the new type adds
// triangular convenience accessors used by math primitives.
// =============================================================================
template<class T = double, unsigned N = 0>
class dual2nd : public dual<dual<T, N>, N> {
public:
  using base         = dual<dual<T, N>, N>;
  using inner_dual_t = dual<T, N>;
  using value_type   = inner_dual_t;
  static constexpr unsigned static_size = N;

  // Inherit base ctors and assignments. base provides:
  //   - default ctor                     dual2nd()
  //   - converting ctor from arithmetic  dual2nd(const U&)
  //   - copy / move ctors and assigns
  //   - operator=(arithmetic U)
  //   - operator=(const dual<U,M>&) via the nested instantiation
  //   - operator+= / -= / *= / /= for plain dual + arithmetic operands
  using base::base;
  using base::operator=;
  using base::operator+=;
  using base::operator-=;
  using base::operator*=;
  using base::operator/=;

  // Default ctor must be redeclared explicitly because using-base-ctors
  // does not import the implicit default ctor in all compiler versions.
  dual2nd() : base() {}

  // Implicit copy / move are generated and copy/move the base subobject.
  // No new data members, so default copy/move are correct.

  // ---------------------------------------------------------------------------
  // Triangular accessors. Implementation detail: the underlying nested-dual
  // storage holds the gradient redundantly in two places:
  //
  //   outer.val_.tan_[i]   (the first-order tangent of the value layer)
  //   outer.tan_[i].x()    (the value of the i-th outer tangent)
  //
  // For a function f(theta), both equal df/dtheta_i. The LU IFT recursion
  // reads outer.val_ when extracting "the value layer" (peeling one AD
  // layer), so primitives must keep both copies in sync to remain correct
  // under LU.
  //
  // Math primitives use d1_at(i) (the tan_-side copy) for iteration and
  // call sync_d1_redundant() at the end of each operation to mirror into
  // the val_-side copy. dd_at(i, j) is the canonical lower-triangle Hessian
  // slot; primitives compute j <= i and rely on dd_at to canonicalise on
  // unordered queries.
  // ---------------------------------------------------------------------------

  // First-order tangent g_i = df/dtheta_i. Returns the tan_-side copy
  // (outer.tan_[i].x()).
  //
  // Mutable overload: caller has armed the storage (math primitives call
  // arm_full before writing via d1_at). Returns a reference into the
  // outer.tan_[i] cell directly.
  //
  // Const overload: routes through base::d(i) which is bounds-safe and
  // returns a thread-local zero if the outer layer is not yet armed (e.g.
  // a freshly seeded operand on which only base::diff(0) was called and
  // whose inner tangents are nullptr).
  T& d1_at(unsigned i) {
    return base::operator[](i).x();
  }
  const T& d1_at(unsigned i) const {
    return static_cast<const base*>(this)->d(i).x();
  }

  // Second-order tangent H_ij = d^2 f / dtheta_i dtheta_j. Stored at
  // outer.tan_[max(i,j)].tan_[min(i,j)] (the canonical lower-triangle slot).
  //
  // Mutable overload: armed storage required, canonicalises (i, j) -> lower.
  // Const overload: bounds-safe via the inherited .d(j).d(k) accessors,
  // canonicalises (i, j) -> lower.
  T& dd_at(unsigned i, unsigned j) {
    const unsigned r = (j <= i) ? i : j;
    const unsigned c = (j <= i) ? j : i;
    return base::operator[](r)[c];
  }
  const T& dd_at(unsigned i, unsigned j) const {
    const unsigned r = (j <= i) ? i : j;
    const unsigned c = (j <= i) ? j : i;
    return static_cast<const base*>(this)->d(r).d(c);
  }

  // Raw access to the (i, j) inner-tangent slot WITHOUT canonicalisation.
  // Reads/writes operator[](i)[j] directly. Used by mirror_upper_dd to
  // copy the lower-triangle Hessian into the upper triangle of the
  // underlying N x N inner-tangent block, so downstream LU IFT extraction
  // (which reads the natural [j][i] slot for j > i) sees a populated cell.
  T& dd_raw(unsigned i, unsigned j) {
    return base::operator[](i)[j];
  }
  const T& dd_raw(unsigned i, unsigned j) const {
    return base::operator[](i)[j];
  }

  // sync_d1_redundant: NO-OP STUB. Historically this mirrored the inline
  // gradient (outer.tan_[k].x()) into outer.val_.tan_[k] (val_tan_block)
  // so the LU value-layer extraction could read .x().d(k) and get the
  // gradient. The LU now reads gradient via first_order_view from inline
  // outer.tan_[k].x() directly (cppode_ad_lu.hpp / cppode_sparse_ad_lu.hpp
  // dual2nd dispatch), so no mirroring is needed. Kept as a no-op for
  // back-compat with old call sites; can be removed once those are gone.
  void sync_d1_redundant() noexcept {}

  // Convenience: arm the outer tangent slots so d1_at and dd_at writes go
  // into pre-allocated storage. After this call, base::depend() is true and
  // base::operator[](i).depend() is true for all i in [0, m).
  //
  // The outer.val_ layer is no longer armed: with the val_tan_block
  // dropped, outer.val_.tan_ stays nullptr and the LU reads gradient via
  // first_order_view from inline outer.tan_[k].x() instead.
  void arm_full(unsigned m) {
    if constexpr (N > 0) {
      (void)m;
      if (!this->depend()) base::set_depend();
      for (unsigned i = 0; i < N; ++i) {
        auto& ti = base::operator[](i);
        if (!ti.depend()) ti.set_depend();
      }
    } else {
      if (!this->depend()) base::set_depend_size(m);
      for (unsigned i = 0; i < m; ++i) {
        auto& ti = base::operator[](i);
        if (!ti.depend()) ti.set_depend_size(m);
      }
    }
  }

  // Width helper. For static-N this is the compile-time N; for dynamic-N
  // it is the runtime size. Math primitives need this as the loop bound.
  unsigned width() const {
    if constexpr (N > 0) return N;
    else                  return this->size();
  }

  // Innermost scalar f(theta). Equivalent to base::x().x(), peeling both AD
  // layers. Math primitives read/write this directly to avoid going through
  // the nested-dual eager operators (which would re-evaluate the value layer
  // as a dual<T,N> expression instead of a single T).
  T& scalar()             { return base::x().x(); }
  const T& scalar() const { return base::x().x(); }

  // -- Expression-template assignment / construction -------------------------
  // Materialises any Expr2nd<D> tree directly into this dual2nd's storage.
  // Definitions live in cppode_dual2nd_expr.hpp (after Expr2nd<D> is complete).
  // Argument deduction against `const expr2::Expr2nd<D>&` only matches CRTP
  // derivations; dual2nd-to-dual2nd or dual2nd-from-scalar paths fall back
  // to the base class operators inherited via using base::operator=.
  template<class D>
  inline dual2nd& operator=(const expr2::Expr2nd<D>& e);

  template<class D>
  inline dual2nd(const expr2::Expr2nd<D>& e);

  // Compound assignment from Expr2nd: route through (*this op other) which
  // builds a BinExpr2 and materialises into *this. The eager nested base
  // operator+=(const dual<dual<T,N>,N>&) is preserved for plain dual2nd
  // operands via inheritance.
  template<class D>
  inline dual2nd& operator+=(const expr2::Expr2nd<D>& e);
  template<class D>
  inline dual2nd& operator-=(const expr2::Expr2nd<D>& e);
  template<class D>
  inline dual2nd& operator*=(const expr2::Expr2nd<D>& e);
  template<class D>
  inline dual2nd& operator/=(const expr2::Expr2nd<D>& e);
};

// ----------------------------------------------------------------------------
// Synthesise a first-order dual<S, N> from a dual2nd's scalar + inline gradient
// (outer.tan_[k].x()), bypassing the redundant val_tan_block. Used by the LU
// dual2nd dispatch to extract the value layer without requiring val_tan to be
// kept in sync via sync_d1_redundant.
// ----------------------------------------------------------------------------
template<class S, unsigned N>
inline cppode::dual<S, N> first_order_view(const cppode::dual2nd<S, N>& v) {
  cppode::dual<S, N> r;
  r.x() = v.scalar();
  if (v.depend()) {
    if constexpr (N > 0) {
      r.set_depend_size();
      for (unsigned k = 0; k < N; ++k) r[k] = v.d1_at(k);
    } else {
      const unsigned m = v.size();
      if (m > 0) {
        r.set_depend_size(m);
        for (unsigned k = 0; k < m; ++k) r[k] = v.d1_at(k);
      }
    }
  }
  return r;
}

} // namespace cppode

#endif // CPPODE_DUAL2ND_HPP
