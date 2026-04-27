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
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cppode/cppode_dual.hpp>
#include <cppode/cppode_types.hpp>

namespace cppode {
namespace detail {

// =============================================================================
//  is_dynamic_dual<T>
//
//  True when T is cppode::dual<S, N> (any N >= 0) with S a non-AD scalar
//  (i.e. the slab-eligible dual specs — both heap dual<S, 0> and static-N
//  dual<S, N>). Nested duals (dual<dual<...>,...>) are intentionally
//  excluded — deriv2 inner-tangent slabs would need per-outer-tangent
//  buffers, which the current slab layout doesn't model.
//
//  Name is historical (originally only the dynamic-N path needed slabbing);
//  after the static-N migration to pointer storage the predicate covers all
//  non-nested dual specs uniformly.
// =============================================================================

template<class T>
struct is_dynamic_dual : std::false_type {};

template<class S, unsigned N>
struct is_dynamic_dual<cppode::dual<S, N>>
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

  // Raw access to the contiguous tangent block. Layout is row-major:
  // tangents of v[i] live at storage_[i * n_cols_ .. i * n_cols_ + n_cols_).
  // Free helpers (vec_axpy_with_slab / vec_scale_with_slab) call BLAS over
  // this whole block; per-element pointers are still reachable via v[i].
  inner_type*       tangent_data()       noexcept { return storage_.data(); }
  const inner_type* tangent_data() const noexcept { return storage_.data(); }
  std::size_t       tangent_size() const noexcept { return storage_.size(); }

  // Size the slab and rebind every v[i].tan_ into row i. Idempotent: if the
  // (n_rows, n_cols) shape already matches, only the rebind step runs.
  //
  // For static-N duals (T::static_size > 0): pin n_cols to N. Stepper-side
  // calls pass the *active* sensitivity width (n_sens, which under reparam
  // or runtime-fixed can be < N), but each dual must always span N tangent
  // slots — eager / ET loops bound on loop_size() = N, and reading past
  // the slab row otherwise lands on the next row's tangents (UB). Inactive
  // slots remain zero, contribute nothing to chain rules.
  void prime(std::vector<T>& v, unsigned n_rows, unsigned n_cols) {
    assert(static_cast<unsigned>(v.size()) == n_rows
           && "tangent_slab::prime: vector size mismatch");
    if constexpr (T::static_size > 0) {
      n_cols = T::static_size;
    }
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

// =============================================================================
//  Slab-aware AXPY / SCAL on std::vector<dual<S, 0>>.
//
//  When the slab is primed, the per-dual tangent buffers are slices of one
//  contiguous [n_rows × n_cols] block. That lets the tangent half of the
//  operation collapse into one BLAS call (daxpy / dscal) instead of n
//  per-element ET evaluations whose MulOp.tangent path computes the
//  symmetric `bv*at + av*bt` even when the scalar side has zero tangent.
//
//  For non-dynamic-dual T (double, static-N dual<T,N!=0>) these helpers
//  forward to plain vec_axpy / vec_scale — the empty-stub slab is ignored.
//  Callers can therefore pass `m_zn[j].m_v` + `m_zn_slab[j]` uniformly,
//  regardless of whether the stepper was instantiated over double, dual,
//  or nested dual.
// =============================================================================

template<class T>
inline void vec_axpy_with_slab(
    std::vector<T>& y, detail::tangent_slab<T>& y_slab,
    double alpha,
    const std::vector<T>& x, const detail::tangent_slab<T>& x_slab)
{
  if constexpr (detail::is_dynamic_dual<T>::value) {
    using S = typename T::value_type;
    const std::size_t n = y.size();
    if (y_slab.primed() && x_slab.primed()) {
      // BLAS hot path: per-element val + one BLAS pass over tangent block.
      // The two passes are disjoint (val ≠ tangents), no double-counting.
      for (std::size_t i = 0; i < n; ++i)
        y[i].x() += static_cast<S>(alpha) * x[i].x();
      assert(y_slab.tangent_size() == x_slab.tangent_size()
             && "vec_axpy_with_slab: slab size mismatch");
      const std::size_t total = y_slab.tangent_size();
      if (total > 0) {
        if constexpr (std::is_same_v<S, double>) {
          int len = static_cast<int>(total);
          int inc = 1;
          double a = alpha;
          F77_CALL(daxpy)(&len, &a,
                          const_cast<double*>(x_slab.tangent_data()), &inc,
                          y_slab.tangent_data(), &inc);
        } else {
          S* yp = y_slab.tangent_data();
          const S* xp = x_slab.tangent_data();
          const S a_s = static_cast<S>(alpha);
          for (std::size_t k = 0; k < total; ++k)
            yp[k] += a_s * xp[k];
        }
      }
    } else {
      // Slab unprimed (sensitivities-off solve, e.g. M=0 reparam path):
      // per-element ET handles val AND tangents in one shot. Wrap alpha
      // in a dual so dual::operator+=(Expr<>) fires.
      const T a_t = T(static_cast<S>(alpha));
      for (std::size_t i = 0; i < n; ++i) y[i] += a_t * x[i];
    }
  } else {
    // Non-dual scalar (double): ordinary vec_axpy.
    vec_axpy(y, alpha, x);
  }
}

// Slab-aware vector zero:
//   y.values = 0;  y.tangents (slab block) = 0
// Crucially this does NOT call `y[i] = T(0)` on the dual elements — that
// would invoke dual<T,0>::operator=(const U&), which sets tan_ = nullptr
// and size_ = 0, undoing the slab binding. Using vec_zero on a slab-bound
// vector therefore breaks the next vec_axpy_with_slab call. The
// fallback path (slab unprimed) still uses dual<T,0>::operator*=(0)
// to preserve any pre-allocated tangent buffers.
template<class T>
inline void vec_zero_with_slab(
    std::vector<T>& y, detail::tangent_slab<T>& y_slab)
{
  if constexpr (detail::is_dynamic_dual<T>::value) {
    using S = typename T::value_type;
    const std::size_t n = y.size();
    for (std::size_t i = 0; i < n; ++i) y[i].x() = S(0);
    if (y_slab.primed()) {
      const std::size_t total = y_slab.tangent_size();
      if (total > 0)
        std::memset(y_slab.tangent_data(), 0, total * sizeof(S));
    } else {
      // Slab unprimed: zero values via *=0 to preserve any arena tan_
      // buffers (so they stay live for the next vec_axpy materialisation).
      const S zero = S(0);
      for (std::size_t i = 0; i < n; ++i) y[i] *= zero;
    }
  } else {
    vec_zero(y);
  }
}

// Slab-aware vector copy: replaces the per-element dual::operator=
// loop (which iterates val_ + N tangents per dual) with a flat
// std::memcpy over the contiguous slab block. Falls back to plain
// per-element assignment for non-dynamic-dual T or unprimed slabs.
template<class T>
inline void vec_copy_with_slab(
    std::vector<T>& y, detail::tangent_slab<T>& y_slab,
    const std::vector<T>& x, const detail::tangent_slab<T>& x_slab)
{
  if constexpr (detail::is_dynamic_dual<T>::value) {
    using S = typename T::value_type;
    const std::size_t n = y.size();
    for (std::size_t i = 0; i < n; ++i)
      y[i].x() = x[i].x();
    if (y_slab.primed() && x_slab.primed()) {
      assert(y_slab.tangent_size() == x_slab.tangent_size()
             && "vec_copy_with_slab: slab size mismatch");
      const std::size_t total = y_slab.tangent_size();
      if (total > 0)
        std::memcpy(y_slab.tangent_data(), x_slab.tangent_data(),
                    total * sizeof(S));
    } else {
      // Slab unprimed: per-element copy through dual::operator=.
      for (std::size_t i = 0; i < n; ++i) y[i] = x[i];
    }
  } else {
    // Non-dual / static-N: std::vector copy assignment is fine.
    y = x;
  }
}

template<class T>
inline void vec_scale_with_slab(
    std::vector<T>& y, detail::tangent_slab<T>& y_slab, double alpha)
{
  if constexpr (detail::is_dynamic_dual<T>::value) {
    using S = typename T::value_type;
    const std::size_t n = y.size();
    if (y_slab.primed()) {
      // BLAS hot path: per-element val * scalar + BLAS dscal on tangent block.
      for (std::size_t i = 0; i < n; ++i)
        y[i].x() *= static_cast<S>(alpha);
      const std::size_t total = y_slab.tangent_size();
      if (total > 0) {
        if constexpr (std::is_same_v<S, double>) {
          int len = static_cast<int>(total);
          int inc = 1;
          double a = alpha;
          F77_CALL(dscal)(&len, &a, y_slab.tangent_data(), &inc);
        } else {
          S* yp = y_slab.tangent_data();
          const S a_s = static_cast<S>(alpha);
          for (std::size_t k = 0; k < total; ++k)
            yp[k] *= a_s;
        }
      }
    } else {
      // Sensitivities off: per-element scalar *= goes through the inline
      // dual<T,0>::operator*=(U) path (no allocations).
      const S a_s = static_cast<S>(alpha);
      for (std::size_t i = 0; i < n; ++i) y[i] *= a_s;
    }
  } else {
    vec_scale(y, alpha);
  }
}

} // namespace cppode

#endif // CPPODE_DUAL_SLAB_HPP
