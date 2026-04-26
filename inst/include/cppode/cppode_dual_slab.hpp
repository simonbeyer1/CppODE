/*
 Contiguous tangent storage for std::vector<dual<T, 0>>.

 Replaces the per-element arena allocations of dual<T, 0>::tan_ with a single
 [n_rows × n_cols] block owned by the multistepper / controller. Each
 dual<T,0>::tan_ then points into one row of the block (set via
 rebind_storage). Subsequent dual = expr materialisations hit the in-place
 reuse branch (size_ matches), so the hot path makes zero arena allocations.

 The slab is sized once per solve via prepare_sensitivities(n_sens) and never
 grown afterward, which keeps the embedded tan_ pointers stable.

 For non-dynamic-dual T (double, static-N dual<T,N!=0>, …) tangent_slab is
 specialised as an empty stub so multistepper<double, …> instances pay no
 size or codegen cost for the slab machinery.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_DUAL_SLAB_HPP
#define CPPODE_DUAL_SLAB_HPP

#include <cassert>
#include <type_traits>
#include <vector>

#include <cppode/cppode_dual.hpp>

namespace cppode {
namespace detail {

// =============================================================================
//  is_dynamic_dual<T>
//
//  True when T is dual<S, 0> with S a non-AD scalar (the heap path that
//  needs slabbing). Nested duals (dual<dual<...,0>,0>) are intentionally
//  excluded — heap deriv2 is forbidden upstream and the inner duals would
//  still need their own per-row buffers.
// =============================================================================

template<class T>
struct is_dynamic_dual : std::false_type {};

template<class S>
struct is_dynamic_dual<cppode::dual<S, 0>>
  : std::bool_constant<!std::is_class_v<S> || std::is_arithmetic_v<S>>
{};

// =============================================================================
//  tangent_slab<T> — contiguous [n_rows × n_cols] storage for dual<S,0>.
//
//  Primary template handles dual<S,0>; non-dynamic-dual T uses the empty
//  stub specialisation below.
// =============================================================================

template<class T, bool = is_dynamic_dual<T>::value>
class tangent_slab; // primary template (active path)

template<class T>
class tangent_slab<T, true> {
public:
  using inner_type = typename T::value_type;  // dual<S,0>::value_type == S

  tangent_slab() = default;
  tangent_slab(const tangent_slab&)            = delete;
  tangent_slab& operator=(const tangent_slab&) = delete;
  tangent_slab(tangent_slab&&) noexcept        = default;
  tangent_slab& operator=(tangent_slab&&) noexcept = default;

  unsigned n_rows() const noexcept { return n_rows_; }
  unsigned n_cols() const noexcept { return n_cols_; }
  bool     primed() const noexcept { return n_cols_ != 0; }

  // Size the slab and rebind every v[i].tan_ into row i. Idempotent: if the
  // (n_rows, n_cols) shape already matches, only the rebind step runs.
  void prime(std::vector<T>& v, unsigned n_rows, unsigned n_cols) {
    assert(static_cast<unsigned>(v.size()) == n_rows
           && "tangent_slab::prime: vector size mismatch");
    if (n_rows == n_rows_ && n_cols == n_cols_) {
      rebind_only(v);
      return;
    }
    n_rows_ = n_rows;
    n_cols_ = n_cols;
    storage_.assign(static_cast<std::size_t>(n_rows) * n_cols, inner_type{});
    rebind_only(v);
  }

  // Re-bind v[i].tan_ pointers without resizing storage. Used after a
  // std::vector::resize that may have moved the dual elements (their tan_
  // pointers remain valid as long as storage_ hasn't moved).
  void rebind_only(std::vector<T>& v) {
    if (n_cols_ == 0) return;  // not yet primed; nothing to bind
    assert(static_cast<unsigned>(v.size()) == n_rows_
           && "tangent_slab::rebind_only: vector size changed");
    inner_type* base = storage_.data();
    for (unsigned i = 0; i < n_rows_; ++i) {
      v[i].rebind_storage(base + static_cast<std::size_t>(i) * n_cols_, n_cols_);
    }
  }

private:
  std::vector<inner_type> storage_;
  unsigned                n_rows_ = 0;
  unsigned                n_cols_ = 0;
};

// Empty stub for non-dynamic-dual T: zero size, all methods no-op.
template<class T>
class tangent_slab<T, false> {
public:
  unsigned n_rows() const noexcept { return 0; }
  unsigned n_cols() const noexcept { return 0; }
  bool     primed() const noexcept { return false; }

  void prime(std::vector<T>&, unsigned, unsigned) noexcept {}
  void rebind_only(std::vector<T>&) noexcept {}
};

} // namespace detail
} // namespace cppode

#endif // CPPODE_DUAL_SLAB_HPP
