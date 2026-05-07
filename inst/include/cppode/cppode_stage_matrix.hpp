/*
 stage_matrix<T, S>: contiguous tangent storage for S RK stages.

 Owns one [(S) × n_rows × n_cols] inner_type buffer; provides S
 tangent_slab views (each one [n_rows × n_cols] block) bound via
 prime_external() onto the corresponding contiguous slice. The unified
 buffer enables BLAS-2 (dgemv) across the stage axis: per-stage assembly
 `xtmp = x + h * sum_{j<s} a_{s,j} * k_j` collapses to one dgemv on the
 (n*M) × s matrix instead of s separate daxpys.

 Storage layout matches nordsieck_block: column-major over (stage j,
 n_rows*n_cols flat), so the unified buffer is effectively a
 (n_rows*n_cols) × S matrix with leading dimension n_rows*n_cols and
 column j = stage j. Per-slot views (slab(j)) keep the row-major
 (state, deriv) layout the legacy tangent_slab uses.

 Empty stub for non-dynamic-dual T so non-AD steppers pay no size or
 codegen cost.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_STAGE_MATRIX_HPP
#define CPPODE_STAGE_MATRIX_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include <cppode/cppode_dual_slab.hpp>

namespace cppode {
namespace detail {

template<class T, unsigned S, bool = is_dynamic_dual<T>::value>
class stage_matrix;

template<class T, unsigned S>
class stage_matrix<T, S, true> {
public:
  using inner_type = typename T::value_type;

  static constexpr unsigned n_stages = S;

  stage_matrix() = default;
  stage_matrix(const stage_matrix&)            = delete;
  stage_matrix& operator=(const stage_matrix&) = delete;
  stage_matrix(stage_matrix&&) noexcept        = default;
  stage_matrix& operator=(stage_matrix&&) noexcept = default;

  unsigned n_rows() const noexcept { return n_rows_; }
  unsigned n_cols() const noexcept { return n_cols_; }
  bool     primed() const noexcept { return n_cols_ != 0; }

  std::size_t slot_stride() const noexcept {
    return static_cast<std::size_t>(n_rows_) * n_cols_;
  }
  std::size_t total_size() const noexcept { return slot_stride() * S; }

  inner_type*       tangent_block_data()       noexcept { return storage_.data(); }
  const inner_type* tangent_block_data() const noexcept { return storage_.data(); }

  inner_type* slot_data(unsigned j) noexcept {
    assert(j < S);
    return storage_.data() + j * slot_stride();
  }
  const inner_type* slot_data(unsigned j) const noexcept {
    assert(j < S);
    return storage_.data() + j * slot_stride();
  }

  tangent_slab<T>&       slab(unsigned j)       noexcept { assert(j < S); return slabs_[j]; }
  const tangent_slab<T>& slab(unsigned j) const noexcept { assert(j < S); return slabs_[j]; }

  void prime(std::array<std::vector<T>*, S> facades,
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
    for (unsigned j = 0; j < S; ++j) {
      if (facades[j] == nullptr) continue;
      slabs_[j].prime_external(*facades[j],
                               storage_.data() + j * slot_stride(),
                               n_rows_, n_cols_);
    }
  }

  void rebind(std::array<std::vector<T>*, S> facades) {
    if (n_cols_ == 0) return;
    for (unsigned j = 0; j < S; ++j) {
      if (facades[j] == nullptr) continue;
      slabs_[j].rebind_only(*facades[j]);
    }
  }

  // Swap stage columns i and j (used for FSAL: stage 0 <-> stage S-1
  // after step acceptance).  Swaps both the values inside the contiguous
  // tangent buffer; the dual elements' tan_ pointers continue to point
  // at the same slots so caller-side facade vectors (with their .x()
  // values) must be std::swap'd separately.
  void swap_stage_tangents(unsigned i, unsigned j) {
    assert(i < S && j < S);
    if (i == j || n_cols_ == 0) return;
    const std::size_t per = slot_stride();
    if (per == 0) return;
    inner_type* a = storage_.data() + i * per;
    inner_type* b = storage_.data() + j * per;
    for (std::size_t k = 0; k < per; ++k) {
      inner_type tmp = a[k];
      a[k] = b[k];
      b[k] = tmp;
    }
  }

private:
  std::vector<inner_type>          storage_;
  std::array<tangent_slab<T>, S>   slabs_;
  unsigned                         n_rows_ = 0;
  unsigned                         n_cols_ = 0;
};

template<class T, unsigned S>
class stage_matrix<T, S, false> {
public:
  static constexpr unsigned n_stages = S;

  unsigned n_rows() const noexcept { return 0; }
  unsigned n_cols() const noexcept { return 0; }
  bool     primed() const noexcept { return false; }
  std::size_t slot_stride() const noexcept { return 0; }
  std::size_t total_size() const noexcept  { return 0; }

  void prime(std::array<std::vector<T>*, S>, unsigned, unsigned) noexcept {}
  void rebind(std::array<std::vector<T>*, S>) noexcept {}
  void swap_stage_tangents(unsigned, unsigned) noexcept {}

  tangent_slab<T>&       slab(unsigned)       noexcept { return stub_; }
  const tangent_slab<T>& slab(unsigned) const noexcept { return stub_; }

private:
  mutable tangent_slab<T> stub_;
};

} // namespace detail
} // namespace cppode

#endif // CPPODE_STAGE_MATRIX_HPP
