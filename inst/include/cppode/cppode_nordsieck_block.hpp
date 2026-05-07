/*
 nordsieck_block<T, K>: contiguous tangent storage for K Nordsieck slots.

 Owns one [(K) × n_rows × n_cols] inner_type buffer; provides K
 tangent_slab views (each one [n_rows × n_cols] block) bound via
 prime_external() onto the corresponding contiguous slice. The unified
 buffer enables BLAS-3 across the slot axis: ndfPredict / ndfRestore
 collapse to one dtrmm; complete_step's rank-1 update collapses to one
 dger. Per-slot operations (vec_axpy_with_slab, vec_copy_with_slab, …)
 keep working unchanged via the per-slot slab views.

 Storage layout: column-major over (slot j, n_rows*n_cols flat). Slot j
 occupies indices [j * n_rows * n_cols, (j+1) * n_rows * n_cols). Within
 a slot the layout matches the legacy tangent_slab: row-major over
 (state i, deriv k), so v[i].tan_ -> base_j + i * n_cols + [0, n_cols).
 With BLAS this is column-major from the slot perspective: each slot is
 one column of a [(n_rows*n_cols) × K] matrix with leading dimension
 n_rows*n_cols. dtrmm("R", ...) and dger right-multiply onto this
 matrix to apply (q+1) × (q+1) Pascal / outer-product updates on the
 slot axis.

 Empty stub for non-dynamic-dual T (plain double, nested dual<dual>,…)
 so that multistepper<double, …> instances pay no size or codegen cost
 and nested-dual paths fall back to the existing per-element loops.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_NORDSIECK_BLOCK_HPP
#define CPPODE_NORDSIECK_BLOCK_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include <cppode/cppode_dual_slab.hpp>

namespace cppode {
namespace detail {

template<class T, unsigned K, bool = is_dynamic_dual<T>::value>
class nordsieck_block;

// =============================================================================
//  Active path: T is a slab-eligible dual type.
// =============================================================================
template<class T, unsigned K>
class nordsieck_block<T, K, true> {
public:
  using inner_type = typename T::value_type;

  static constexpr unsigned K_slots = K;

  nordsieck_block() = default;
  nordsieck_block(const nordsieck_block&)            = delete;
  nordsieck_block& operator=(const nordsieck_block&) = delete;
  nordsieck_block(nordsieck_block&&) noexcept        = default;
  nordsieck_block& operator=(nordsieck_block&&) noexcept = default;

  unsigned n_rows() const noexcept { return n_rows_; }
  unsigned n_cols() const noexcept { return n_cols_; }
  bool     primed() const noexcept { return n_cols_ != 0; }

  // Per-slot tangent block size in inner_type units.
  std::size_t slot_stride() const noexcept {
    return static_cast<std::size_t>(n_rows_) * n_cols_;
  }
  std::size_t total_size() const noexcept {
    return slot_stride() * K;
  }

  inner_type*       tangent_block_data()       noexcept { return storage_.data(); }
  const inner_type* tangent_block_data() const noexcept { return storage_.data(); }

  // Pointer to slot j's tangent block (n_rows × n_cols).
  inner_type* slot_data(unsigned j) noexcept {
    assert(j < K);
    return storage_.data() + j * slot_stride();
  }
  const inner_type* slot_data(unsigned j) const noexcept {
    assert(j < K);
    return storage_.data() + j * slot_stride();
  }

  // Slot j's slab view (a tangent_slab bound to the slice via
  // prime_external()).  vec_*_with_slab helpers operate on this exactly
  // like they did on the legacy per-slot owned slabs.
  tangent_slab<T>&       slab(unsigned j)       noexcept { assert(j < K); return slabs_[j]; }
  const tangent_slab<T>& slab(unsigned j) const noexcept { assert(j < K); return slabs_[j]; }

  // Allocate the unified [K × n_rows × n_cols] buffer (or just rebind every
  // facade vector if the shape is unchanged) and bind every slab view onto
  // the corresponding slice.  Idempotent.  facades[j] must match the slot
  // j vector with v[i].tan_ to be rebound into the buffer.
  void prime(std::array<std::vector<T>*, K> facades,
             unsigned n_rows, unsigned n_cols) {
    if constexpr (T::static_size > 0) {
      n_cols = T::static_size;
    }
    const bool reshape = (n_rows != n_rows_ || n_cols != n_cols_);
    if (reshape) {
      n_rows_ = n_rows;
      n_cols_ = n_cols;
      storage_.assign(total_size(), inner_type{});
    }
    // Bind each slab onto its slice and rebind its facade's tan_ pointers.
    for (unsigned j = 0; j < K; ++j) {
      if (facades[j] == nullptr) continue;
      slabs_[j].prime_external(*facades[j],
                               storage_.data() + j * slot_stride(),
                               n_rows_, n_cols_);
    }
  }

  // Re-bind the K facade tan_ pointers without reallocating. Used after a
  // resize that may have moved the dual elements; the unified storage is
  // unchanged.
  void rebind(std::array<std::vector<T>*, K> facades) {
    if (n_cols_ == 0) return;
    for (unsigned j = 0; j < K; ++j) {
      if (facades[j] == nullptr) continue;
      slabs_[j].rebind_only(*facades[j]);
    }
  }

private:
  std::vector<inner_type>          storage_;
  std::array<tangent_slab<T>, K>   slabs_;
  unsigned                         n_rows_ = 0;
  unsigned                         n_cols_ = 0;
};

// =============================================================================
//  Empty stub for non-dynamic-dual T.
// =============================================================================
template<class T, unsigned K>
class nordsieck_block<T, K, false> {
public:
  static constexpr unsigned K_slots = K;

  unsigned n_rows() const noexcept { return 0; }
  unsigned n_cols() const noexcept { return 0; }
  bool     primed() const noexcept { return false; }
  std::size_t slot_stride() const noexcept { return 0; }
  std::size_t total_size() const noexcept  { return 0; }

  void prime(std::array<std::vector<T>*, K>, unsigned, unsigned) noexcept {}
  void rebind(std::array<std::vector<T>*, K>) noexcept {}

  tangent_slab<T>&       slab(unsigned)       noexcept { return stub_; }
  const tangent_slab<T>& slab(unsigned) const noexcept { return stub_; }

private:
  // One stub instance shared across slot accesses (empty stub class anyway).
  mutable tangent_slab<T> stub_;
};

// =============================================================================
//  nordsieck_block specialisation for dual2nd<S, N>: K-slot two-block storage
//
//  Owns two concatenated buffers (val_tan_block dropped after the LU dual2nd
//  dispatch started reading gradient via first_order_view from inline_d1):
//    outer_block_   : K * n_rows * n_cols  dual<S, N>      (outer.tan_ values)
//    hess_block_    : K * n_rows * n_cols * n_cols  S      (Hessian rows)
// =============================================================================
template<class S, unsigned N, unsigned K>
class nordsieck_block<cppode::dual2nd<S, N>, K, true> {
public:
  using value_type    = cppode::dual2nd<S, N>;
  using outer_inner_t = cppode::dual<S, N>;
  using inner_type    = outer_inner_t;

  static constexpr unsigned K_slots = K;

  nordsieck_block() = default;
  nordsieck_block(const nordsieck_block&)            = delete;
  nordsieck_block& operator=(const nordsieck_block&) = delete;
  nordsieck_block(nordsieck_block&&) noexcept        = default;
  nordsieck_block& operator=(nordsieck_block&&) noexcept = default;

  unsigned n_rows() const noexcept { return n_rows_; }
  unsigned n_cols() const noexcept { return n_cols_; }
  bool     primed() const noexcept { return n_cols_ != 0; }

  std::size_t slot_stride() const noexcept {
    return static_cast<std::size_t>(n_rows_) * n_cols_;
  }
  std::size_t total_size() const noexcept { return slot_stride() * K; }

  outer_inner_t*       tangent_block_data()       noexcept { return outer_block_.data(); }
  const outer_inner_t* tangent_block_data() const noexcept { return outer_block_.data(); }

  S* hess_block_data() noexcept { return hess_block_.data(); }
  const S* hess_block_data() const noexcept { return hess_block_.data(); }
  std::size_t hess_slot_stride() const noexcept {
    return static_cast<std::size_t>(n_rows_) * n_cols_ * n_cols_;
  }

  outer_inner_t* slot_data(unsigned j) noexcept {
    assert(j < K);
    return outer_block_.data() + j * slot_stride();
  }
  const outer_inner_t* slot_data(unsigned j) const noexcept {
    assert(j < K);
    return outer_block_.data() + j * slot_stride();
  }

  tangent_slab<value_type>&       slab(unsigned j)       noexcept { assert(j < K); return slabs_[j]; }
  const tangent_slab<value_type>& slab(unsigned j) const noexcept { assert(j < K); return slabs_[j]; }

  void prime(std::array<std::vector<value_type>*, K> facades,
             unsigned n_rows, unsigned n_cols) {
    if constexpr (N > 0) n_cols = N;
    const bool reshape = (n_rows != n_rows_ || n_cols != n_cols_);
    if (reshape) {
      n_rows_ = n_rows;
      n_cols_ = n_cols;
      outer_block_.clear();
      outer_block_.resize(slot_stride() * K);
      hess_block_.assign(hess_slot_stride() * K, S(0));
    }
    for (unsigned j = 0; j < K; ++j) {
      if (facades[j] == nullptr) continue;
      slabs_[j].prime_external(*facades[j],
                               outer_block_.data() + j * slot_stride(),
                               static_cast<S*>(nullptr),  // no val_tan
                               hess_block_.data()  + j * hess_slot_stride(),
                               n_rows_, n_cols_);
    }
  }

  void rebind(std::array<std::vector<value_type>*, K> facades) {
    if (n_cols_ == 0) return;
    for (unsigned j = 0; j < K; ++j) {
      if (facades[j] == nullptr) continue;
      slabs_[j].rebind_only(*facades[j]);
    }
  }

private:
  std::vector<outer_inner_t>                outer_block_;
  std::vector<S>                            hess_block_;
  std::array<tangent_slab<value_type>, K>   slabs_;
  unsigned                                  n_rows_ = 0;
  unsigned                                  n_cols_ = 0;
};

} // namespace detail
} // namespace cppode

#endif // CPPODE_NORDSIECK_BLOCK_HPP
