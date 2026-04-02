/*
 CppODE core type definitions — Eigen-free.

 Provides lightweight replacements for Eigen types:
 vector_t<T>       = std::vector<T>
 dense_matrix<T>   = flat column-major array with (i,j) access
 csc_matrix<T>     = compressed sparse column (Ap, Ai, Ax)

 Dense LU uses LAPACK (dgetrf/dgetrs) via R's bundled BLAS/LAPACK.
 Sparse LU uses KLU (bundled SuiteSparse).

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_TYPES_HPP
#define CPPODE_TYPES_HPP

#include <vector>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <cmath>
#include <numeric>

// BLAS/LAPACK declarations via R's headers
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

namespace cppode {

// ============================================================================
//  vector_t<T> — dynamic vector (drop-in for Eigen::VectorXd)
// ============================================================================

template<class T>
using vector_t = std::vector<T>;

// ============================================================================
//  dense_matrix<T> — column-major flat array
//
//  Layout: element (row, col) lives at data[col * n_rows + row].
//  This matches Fortran/LAPACK column-major convention.
// ============================================================================

template<class T>
struct dense_matrix
{
  std::vector<T> data;
  int n_rows = 0;
  int n_cols = 0;

  dense_matrix() = default;
  dense_matrix(int r, int c) : data(r * c, T(0)), n_rows(r), n_cols(c) {}

  void resize(int r, int c)
  {
    n_rows = r;
    n_cols = c;
    data.assign(static_cast<size_t>(r) * c, T(0));
  }

  void set_zero()
  {
    std::fill(data.begin(), data.end(), T(0));
  }

  T& operator()(int i, int j)             { return data[j * n_rows + i]; }
  const T& operator()(int i, int j) const { return data[j * n_rows + i]; }

  int rows() const { return n_rows; }
  int cols() const { return n_cols; }
  int size() const { return n_rows * n_cols; }

  T*       ptr()       { return data.data(); }
  const T* ptr() const { return data.data(); }
};

// ============================================================================
//  csc_matrix<T> — compressed sparse column format
//
//  Ap[0..n]:   column pointers (Ap[j] = start of column j in Ai/Ax)
//  Ai[0..nnz]: row indices
//  Ax[0..nnz]: values
//
//  Pattern is built once at runtime (first Jacobian call).
//  Subsequent calls only overwrite Ax.
// ============================================================================

template<class T>
struct csc_matrix
{
  std::vector<int> Ap;   // size n+1
  std::vector<int> Ai;   // size nnz
  std::vector<T>   Ax;   // size nnz
  int n   = 0;
  int nnz = 0;
  bool pattern_built = false;

  // Build CSC pattern from pre-sorted (row, col) pairs.
  // Entries must be sorted by column, then by row within column.
  // This is called once; after that, only Ax is overwritten.
  void build_pattern(int dim, int n_entries,
                     const int* rows, const int* cols)
  {
    n   = dim;
    nnz = n_entries;
    Ap.assign(n + 1, 0);
    Ai.resize(nnz);
    Ax.resize(nnz);

    // Count entries per column
    for (int k = 0; k < nnz; ++k)
      Ap[cols[k] + 1]++;

    // Cumulative sum
    for (int j = 0; j < n; ++j)
      Ap[j + 1] += Ap[j];

    // Fill Ai (rows are already sorted within each column)
    std::vector<int> cursor(Ap.begin(), Ap.begin() + n);
    for (int k = 0; k < nnz; ++k) {
      int pos = cursor[cols[k]]++;
      Ai[pos] = rows[k];
    }

    // Zero-initialize values
    std::fill(Ax.begin(), Ax.end(), T(0));
    pattern_built = true;
  }

  // Direct access to value array (for codegen: Ax[k] = expr)
  T&       val(int k)       { return Ax[k]; }
  const T& val(int k) const { return Ax[k]; }

  // Raw pointer access (for KLU)
  int*     ap_ptr()       { return Ap.data(); }
  int*     ai_ptr()       { return Ai.data(); }
  T*       ax_ptr()       { return Ax.data(); }
  const int* ap_ptr() const { return Ap.data(); }
  const int* ai_ptr() const { return Ai.data(); }
  const T*   ax_ptr() const { return Ax.data(); }

  // Iterate over column j: entries Ai[Ap[j]..Ap[j+1]), Ax[Ap[j]..Ap[j+1])
  int col_begin(int j) const { return Ap[j]; }
  int col_end(int j) const   { return Ap[j + 1]; }
};

// ============================================================================
//  Vector arithmetic helpers (replacing Eigen operator overloads)
//
//  These are used by the steppers for stage computation.
//  All operate element-wise on std::vector<T>.
// ============================================================================

/// v = 0
template<class T>
inline void vec_zero(std::vector<T>& v)
{
  std::fill(v.begin(), v.end(), T(0));
}

/// dst = src (resize dst if needed)
template<class T>
inline void vec_copy(std::vector<T>& dst, const std::vector<T>& src)
{
  dst = src;
}

/// v *= scalar
template<class T, class S>
inline void vec_scale(std::vector<T>& v, S alpha)
{
  for (auto& x : v) x *= alpha;
}

/// v *= scalar — BLAS specialization for double
inline void vec_scale(std::vector<double>& v, double alpha)
{
  int n = static_cast<int>(v.size());
  int inc = 1;
  F77_CALL(dscal)(&n, &alpha, v.data(), &inc);
}

/// y += alpha * x  (AXPY)
template<class T, class S>
inline void vec_axpy(std::vector<T>& y, S alpha, const std::vector<T>& x)
{
  const size_t n = y.size();
  for (size_t i = 0; i < n; ++i)
    y[i] += alpha * x[i];
}

/// y += alpha * x — BLAS specialization for double
inline void vec_axpy(std::vector<double>& y, double alpha, const std::vector<double>& x)
{
  int n = static_cast<int>(y.size());
  int inc = 1;
  F77_CALL(daxpy)(&n, &alpha, const_cast<double*>(x.data()), &inc,
           y.data(), &inc);
}

/// ||v||_∞
template<class T>
inline double vec_norm_inf(const std::vector<T>& v)
{
  double mx = 0;
  for (const auto& x : v) {
    double a = std::abs(static_cast<double>(x));
    if (a > mx) mx = a;
  }
  return mx;
}

/// Weighted RMS norm: sqrt(mean((v[i]/w[i])^2))
template<class T>
inline double vec_wrms_norm(const std::vector<T>& v, const std::vector<T>& w)
{
  const size_t n = v.size();
  if (n == 0) return 0.0;
  double sum = 0;
  for (size_t i = 0; i < n; ++i) {
    double r = static_cast<double>(v[i]) / static_cast<double>(w[i]);
    sum += r * r;
  }
  return std::sqrt(sum / static_cast<double>(n));
}

// ============================================================================
//  Dense matrix-vector product: y = A * x  (column-major)
// ============================================================================

template<class T>
inline void matvec(const dense_matrix<T>& A,
                   const std::vector<T>& x,
                   std::vector<T>& y)
{
  const int m = A.rows(), k = A.cols();
  y.assign(m, T(0));
  for (int j = 0; j < k; ++j) {
    const T xj = x[j];
    for (int i = 0; i < m; ++i)
      y[i] += A(i, j) * xj;
  }
}

/// y = A * x — BLAS dgemv specialization for double
inline void matvec(const dense_matrix<double>& A,
                   const std::vector<double>& x,
                   std::vector<double>& y)
{
  int m = A.rows(), n = A.cols();
  y.assign(m, 0.0);
  if (m == 0 || n == 0) return;
  double alpha = 1.0, beta = 0.0;
  int inc = 1;
  char trans = 'N';
  F77_CALL(dgemv)(&trans, &m, &n, &alpha,
           const_cast<double*>(A.ptr()), &m,
           const_cast<double*>(x.data()), &inc,
           &beta, y.data(), &inc  FCONE);
}

/// y -= A * x  (for IFT: rhs -= dW · x_val)
template<class T>
inline void matvec_sub(const dense_matrix<T>& A,
                       const std::vector<T>& x,
                       std::vector<T>& y)
{
  const int m = A.rows(), k = A.cols();
  for (int j = 0; j < k; ++j) {
    const T xj = x[j];
    for (int i = 0; i < m; ++i)
      y[i] -= A(i, j) * xj;
  }
}

/// y -= A * x — BLAS dgemv specialization for double
inline void matvec_sub(const dense_matrix<double>& A,
                       const std::vector<double>& x,
                       std::vector<double>& y)
{
  int m = A.rows(), n = A.cols();
  if (m == 0 || n == 0) return;
  double alpha = -1.0, beta = 1.0;
  int inc = 1;
  char trans = 'N';
  F77_CALL(dgemv)(&trans, &m, &n, &alpha,
           const_cast<double*>(A.ptr()), &m,
           const_cast<double*>(x.data()), &inc,
           &beta, y.data(), &inc  FCONE);
}

// ============================================================================
//  CSC matrix-vector product: y += W * x
// ============================================================================

template<class T>
inline void csc_matvec_add(const csc_matrix<T>& W,
                           const std::vector<T>& x,
                           std::vector<T>& y)
{
  for (int j = 0; j < W.n; ++j) {
    const T xj = x[j];
    for (int p = W.Ap[j]; p < W.Ap[j + 1]; ++p)
      y[W.Ai[p]] += W.Ax[p] * xj;
  }
}

/// y -= W * x
template<class T>
inline void csc_matvec_sub(const csc_matrix<T>& W,
                           const std::vector<T>& x,
                           std::vector<T>& y)
{
  for (int j = 0; j < W.n; ++j) {
    const T xj = x[j];
    for (int p = W.Ap[j]; p < W.Ap[j + 1]; ++p)
      y[W.Ai[p]] -= W.Ax[p] * xj;
  }
}

} // namespace cppode

#endif // CPPODE_TYPES_HPP
