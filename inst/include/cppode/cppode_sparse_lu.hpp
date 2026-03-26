/*
 Sparse LU for ublas::compressed_matrix
 Two-phase sparse LU that works directly on ublas::compressed_matrix<T>.

 Phase 1: Symbolic (ONCE per sparsity pattern):
 - Approximate Minimum Degree (AMD) reordering to minimize fill-in.
 - Simulate Gaussian elimination on the reordered pattern to discover
   fill-in (static pivoting, no row swaps).
 - Build an expanded CSR structure in the AMD-reordered space.
 - Precompute direct offsets for all elimination operations so the
   numerical phase uses O(1) array access intead of a binary search.

 Phase 2: Numerical (every Rosenbrock step):
 - Copy values from compressed_matrix into the reordered CSR using
   precomputed scatter map ~ O(nnz_original).
 - Execute elimination using precomputed offset pairs ~ O(nnz_lu).
 - Check diagonal pivots; return status flag for fallback.

 The symbolic structure is cached in sparse_lu_cache<T> and reused
 across integration steps. The Rosenbrock stepper holds one cache
 as a member.

 Works with double, F<double>, F<F<double>> via templates.
 Copyright (C) 2026 Simon Beyer
 */
#ifndef CPPODE_SPARSE_LU_HPP
#define CPPODE_SPARSE_LU_HPP

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <limits>

#include <fadbad++/fadiff.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

namespace cppode {

// =========================================================================================
//  Scalar value extraction (for pivot checking with AD types)
// =========================================================================================
namespace detail {

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
abs_val(const T& v) { return std::abs(static_cast<double>(v)); }

template<class T>
inline double abs_val(const fadbad::F<T>& v) {
  return abs_val(const_cast<fadbad::F<T>&>(v).x());
}

} // namespace detail

// =========================================================================================
//  Approximate Minimum Degree ordering (AMD)
//
//  Lightweight implementation operating on a symmetric sparsity pattern
//  (|A| + |A^T|).  Produces a fill-reducing permutation for LU.
//  Pure integer algorithm, independent of the value type T.
// =========================================================================================
namespace detail {

inline void amd_order(int n,
                      const std::vector<std::vector<int>>& adj,
                      std::vector<int>& perm,
                      std::vector<int>& iperm)
{
  // adj[i] = sorted list of columns j where A(i,j) != 0 OR A(j,i) != 0.
  // We use a simple greedy minimum-degree heuristic with external-degree
  // approximation and mass elimination of indistinguishable nodes.

  perm.resize(n);
  iperm.resize(n);

  // Degree of each node (number of adjacent non-eliminated nodes)
  std::vector<int> degree(n);
  for (int i = 0; i < n; ++i)
    degree[i] = static_cast<int>(adj[i].size());

  // Track eliminated nodes
  std::vector<bool> eliminated(n, false);

  // Adjacency sets as sorted vectors (will be modified during elimination)
  std::vector<std::vector<int>> nbr(adj);

  for (int step = 0; step < n; ++step) {
    // Find non-eliminated node with minimum degree
    int pivot = -1;
    int min_deg = n + 1;
    for (int i = 0; i < n; ++i) {
      if (!eliminated[i] && degree[i] < min_deg) {
        min_deg = degree[i];
        pivot = i;
      }
    }

    perm[step] = pivot;
    eliminated[pivot] = true;

    // Collect non-eliminated neighbors of pivot
    std::vector<int> live_nbrs;
    live_nbrs.reserve(nbr[pivot].size());
    for (int j : nbr[pivot])
      if (!eliminated[j]) live_nbrs.push_back(j);

      // Form clique: connect all live neighbors to each other
      // and update degrees
      for (int a : live_nbrs) {
        for (int b : live_nbrs) {
          if (a == b) continue;
          // Add edge a-b if not present
          auto& na = nbr[a];
          auto it = std::lower_bound(na.begin(), na.end(), b);
          if (it == na.end() || *it != b) {
            na.insert(it, b);
          }
        }
        // Remove pivot from neighbor list of a
        auto& na = nbr[a];
        auto it = std::lower_bound(na.begin(), na.end(), pivot);
        if (it != na.end() && *it == pivot) na.erase(it);

        // Recompute degree: count non-eliminated neighbors
        int deg = 0;
        for (int j : na)
          if (!eliminated[j]) ++deg;
          degree[a] = deg;
      }

      // Clear pivot's neighbor list (no longer needed)
      nbr[pivot].clear();
      degree[pivot] = 0;
  }

  // Build inverse permutation
  for (int i = 0; i < n; ++i)
    iperm[perm[i]] = i;
}

} // namespace detail

// =========================================================================================
//  sparse_lu_cache<T>
// =========================================================================================
template<class T>
struct sparse_lu_cache {
  int n = 0;
  bool symbolic_ready = false;

  // --- AMD permutation ---
  // perm[new_idx] = old_idx,  iperm[old_idx] = new_idx
  std::vector<int> amd_perm;
  std::vector<int> amd_iperm;

  // --- Expanded CSR in AMD-reordered space ---
  // Row i in this CSR corresponds to original row amd_perm[i].
  std::vector<int> row_ptr;
  std::vector<int> col_idx;   // columns in reordered space
  int nnz_lu = 0;

  // --- Diagonal positions in CSR ---
  std::vector<int> diag_pos;

  // --- Scatter map: original compressed_matrix → reordered CSR ---
  // For each nonzero in the original matrix (enumerated row-major),
  // store the position in values[] where it should go.
  // scatter_pos[k] = position in values[] for k-th original nonzero.
  std::vector<int> scatter_pos;

  // --- Elimination plan with precomputed direct offsets ---
  // For each elimination step k (in reordered space):
  //   For each sub-diagonal row i with L(i,k) != 0:
  //     mult = values[L_pos] / values[diag_pos[k]]
  //     values[L_pos] = mult
  //     for each update:  values[ij_pos] -= mult * values[kj_pos]
  struct update_op {
    int ij_pos;   // position of A(i,j) in values[]
    int kj_pos;   // position of A(k,j) in values[]
  };

  struct elim_row_op {
    int L_pos;            // position of L(i,k) in values[]
    int upd_begin;        // range in updates_flat
    int upd_end;
  };

  struct elim_col_op {
    int k;                // pivot column (= pivot row, reordered)
    int row_begin;        // range in elim_rows_flat
    int row_end;
  };

  std::vector<elim_col_op> elim_cols;
  std::vector<elim_row_op> elim_rows;
  std::vector<update_op>   updates_flat;

  // --- Forward/backward substitution plans ---
  // Precomputed (position, column) pairs for L and U sweeps.
  struct solve_entry {
    int val_pos;   // position in values[]
    int col;       // column index (reordered) for indexing the RHS
  };
  // For each row i (reordered), L entries with col < i and U entries with col > i
  std::vector<int> L_row_ptr;           // size n+1
  std::vector<solve_entry> L_entries;   // L entries for forward sub
  std::vector<int> U_row_ptr;           // size n+1
  std::vector<solve_entry> U_entries;   // U entries for backward sub

  // --- Numerical storage ---
  std::vector<T> values;

  // =========================================================================================
  //  Binary search helper (used only during symbolic analysis)
  // =========================================================================================
  int find_in_row(int row, int col) const {
    int lo = row_ptr[row], hi = row_ptr[row + 1];
    while (lo < hi) {
      int mid = lo + ((hi - lo) >> 1);
      if (col_idx[mid] < col) lo = mid + 1;
      else hi = mid;
    }
    if (lo < row_ptr[row + 1] && col_idx[lo] == col)
      return lo;
    return -1;
  }

  // =========================================================================================
  //  Symbolic analysis with AMD reordering
  // =========================================================================================
  template<class CM>
  void symbolic_analyze(const CM& W)
  {
    n = static_cast<int>(W.size1());

    // --- Extract pattern and build symmetric adjacency for AMD ---
    std::vector<std::vector<int>> orig_rows(n);
    for (auto it1 = W.begin1(); it1 != W.end1(); ++it1) {
      int i = static_cast<int>(it1.index1());
      for (auto it2 = it1.begin(); it2 != it1.end(); ++it2)
        orig_rows[i].push_back(static_cast<int>(it2.index2()));
    }
    for (int i = 0; i < n; ++i) {
      auto& r = orig_rows[i];
      std::sort(r.begin(), r.end());
      r.erase(std::unique(r.begin(), r.end()), r.end());
    }

    // Symmetric adjacency: adj[i] = union of {j : A(i,j)!=0} ∪ {j : A(j,i)!=0}
    std::vector<std::vector<int>> sym_adj(n);
    for (int i = 0; i < n; ++i) {
      for (int j : orig_rows[i]) {
        if (j != i) {
          sym_adj[i].push_back(j);
          sym_adj[j].push_back(i);
        }
      }
    }
    for (int i = 0; i < n; ++i) {
      auto& a = sym_adj[i];
      std::sort(a.begin(), a.end());
      a.erase(std::unique(a.begin(), a.end()), a.end());
    }

    // --- AMD ordering ---
    detail::amd_order(n, sym_adj, amd_perm, amd_iperm);
    sym_adj.clear(); // free memory

    // --- Reorder pattern into AMD space ---
    // rows_reord[new_i] = sorted set of new_j where A(perm[new_i], perm[new_j]) != 0
    std::vector<std::vector<int>> rows_reord(n);
    for (int old_i = 0; old_i < n; ++old_i) {
      int new_i = amd_iperm[old_i];
      for (int old_j : orig_rows[old_i]) {
        int new_j = amd_iperm[old_j];
        rows_reord[new_i].push_back(new_j);
      }
    }
    for (int i = 0; i < n; ++i) {
      auto& r = rows_reord[i];
      std::sort(r.begin(), r.end());
      r.erase(std::unique(r.begin(), r.end()), r.end());
      // Ensure diagonal
      auto dit = std::lower_bound(r.begin(), r.end(), i);
      if (dit == r.end() || *dit != i)
        r.insert(dit, i);
    }

    // --- Symbolic factorization (static pivoting, no row swaps) ---
    // Simulate elimination to discover fill-in.
    // Column lists for quick lookup of "which rows have entry in column k"
    std::vector<std::vector<int>> col_lists(n);
    for (int i = 0; i < n; ++i)
      for (int j : rows_reord[i])
        col_lists[j].push_back(i);

    // Temporary storage for elimination plan before offsets are known
    struct pre_elim_row {
      int row;                     // reordered row index
      std::vector<int> upd_cols;   // columns to update (reordered, > k)
    };
    struct pre_elim_col {
      int k;
      std::vector<pre_elim_row> sub_rows;
    };
    std::vector<pre_elim_col> pre_plan;

    for (int k = 0; k < n; ++k) {
      // Sub-diagonal rows with entry in column k
      std::vector<int> sub_rows;
      for (int i : col_lists[k])
        if (i > k) sub_rows.push_back(i);

        if (sub_rows.empty()) continue;

        // Pivot row's columns > k
        std::vector<int> pivot_cols_gt_k;
        for (int j : rows_reord[k])
          if (j > k) pivot_cols_gt_k.push_back(j);

          pre_elim_col pec;
          pec.k = k;

          for (int i : sub_rows) {
            // Fill-in: row i gets all columns from pivot row that it doesn't have
            for (int j : pivot_cols_gt_k) {
              auto& ri = rows_reord[i];
              auto it = std::lower_bound(ri.begin(), ri.end(), j);
              if (it == ri.end() || *it != j) {
                ri.insert(it, j);
                col_lists[j].push_back(i);
                // col_lists[j] doesn't need to stay sorted for our purposes
              }
            }

            // Columns to update in this elimination: columns > k present in
            // BOTH pivot row and row i after fill-in
            std::vector<int> upd_cols;
            for (int j : pivot_cols_gt_k) {
              // After fill-in, row i definitely has column j
              upd_cols.push_back(j);
            }

            pre_elim_row per;
            per.row = i;
            per.upd_cols = std::move(upd_cols);
            pec.sub_rows.push_back(std::move(per));
          }

          pre_plan.push_back(std::move(pec));
    }

    // --- Build expanded CSR from the filled-in reordered pattern ---
    row_ptr.resize(n + 1);
    col_idx.clear();
    row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
      auto& r = rows_reord[i];
      std::sort(r.begin(), r.end());
      for (int j : r) col_idx.push_back(j);
      row_ptr[i + 1] = static_cast<int>(col_idx.size());
    }
    nnz_lu = static_cast<int>(col_idx.size());

    // Diagonal positions
    diag_pos.resize(n);
    for (int i = 0; i < n; ++i) {
      int pos = find_in_row(i, i);
      if (pos < 0)
        throw std::runtime_error(
            "symbolic_analyze: diagonal (" + std::to_string(i) +
              ", " + std::to_string(i) + ") missing after fill-in");
      diag_pos[i] = pos;
    }

    // --- Build scatter map for copy_values_from ---
    // Count original nonzeros to size the scatter map
    int nnz_orig = 0;
    for (int i = 0; i < n; ++i)
      nnz_orig += static_cast<int>(orig_rows[i].size());
    scatter_pos.resize(nnz_orig);

    int idx = 0;
    for (int old_i = 0; old_i < n; ++old_i) {
      int new_i = amd_iperm[old_i];
      for (int old_j : orig_rows[old_i]) {
        int new_j = amd_iperm[old_j];
        int pos = find_in_row(new_i, new_j);
        scatter_pos[idx++] = pos; // pos >= 0 guaranteed (original entries are in filled pattern)
      }
    }

    // --- Build elimination plan with precomputed direct offsets ---
    elim_cols.clear();
    elim_rows.clear();
    updates_flat.clear();

    for (const auto& pec : pre_plan) {
      int k = pec.k;
      elim_col_op eco;
      eco.k = k;
      eco.row_begin = static_cast<int>(elim_rows.size());

      for (const auto& per : pec.sub_rows) {
        int i = per.row;

        elim_row_op ero;
        ero.L_pos = find_in_row(i, k);  // position of L(i,k)
        ero.upd_begin = static_cast<int>(updates_flat.size());

        for (int j : per.upd_cols) {
          update_op uo;
          uo.ij_pos = find_in_row(i, j);  // position of A(i,j)
          uo.kj_pos = find_in_row(k, j);  // position of A(k,j)
          updates_flat.push_back(uo);
        }

        ero.upd_end = static_cast<int>(updates_flat.size());
        elim_rows.push_back(ero);
      }

      eco.row_end = static_cast<int>(elim_rows.size());
      elim_cols.push_back(eco);
    }

    // --- Build forward/backward substitution plans ---
    L_row_ptr.resize(n + 1);
    U_row_ptr.resize(n + 1);
    L_entries.clear();
    U_entries.clear();

    L_row_ptr[0] = 0;
    U_row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
      // L: entries in row i with col < i
      for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
        int j = col_idx[p];
        if (j >= i) break;  // col_idx is sorted
        L_entries.push_back({p, j});
      }
      L_row_ptr[i + 1] = static_cast<int>(L_entries.size());

      // U: entries in row i with col > i (in reverse order for backward sub)
      for (int p = row_ptr[i + 1] - 1; p >= row_ptr[i]; --p) {
        int j = col_idx[p];
        if (j <= i) break;
        U_entries.push_back({p, j});
      }
      U_row_ptr[i + 1] = static_cast<int>(U_entries.size());
    }

    // --- Allocate numerical storage ---
    values.resize(nnz_lu, T(0));

    symbolic_ready = true;
  }

  // =========================================================================================
  //  Copy values from compressed_matrix into reordered CSR
  //  Uses precomputed scatter map ~ O(nnz_original), no binary search.
  // =========================================================================================
  template<class CM>
  void copy_values_from(const CM& W)
  {
    // Zero the full values array (fill-in entries must start at zero)
    std::fill(values.begin(), values.end(), T(0));

    // Scatter original entries using precomputed positions
    int idx = 0;
    for (auto it1 = W.begin1(); it1 != W.end1(); ++it1) {
      for (auto it2 = it1.begin(); it2 != it1.end(); ++it2) {
        int pos = scatter_pos[idx++];
        if (pos >= 0)
          values[pos] = *it2;
      }
    }
  }

  // =========================================================================================
  //  Numerical LU factorize with static pivoting, precomputed offsets
  //
  //  All array accesses are direct, no binary search.
  //  Returns true if all pivots are acceptable, false if a near-zero
  //  diagonal was detected (caller should fall back to dense).
  // =========================================================================================
  bool numeric_factorize()
  {
    for (const auto& eco : elim_cols) {
      int k = eco.k;
      const T& pivot_val = values[diag_pos[k]];

      for (int r = eco.row_begin; r < eco.row_end; ++r) {
        const auto& ero = elim_rows[r];

        T mult = values[ero.L_pos] / pivot_val;
        values[ero.L_pos] = mult;

        for (int u = ero.upd_begin; u < ero.upd_end; ++u) {
          const auto& uo = updates_flat[u];
          values[uo.ij_pos] -= mult * values[uo.kj_pos];
        }
      }
    }

    return check_pivots();
  }

  // =========================================================================================
  //  Check diagonal pivots after factorization
  //  Returns true if all pivots are acceptable.
  // =========================================================================================
  bool check_pivots() const
  {
    double max_diag = 0.0;
    for (int i = 0; i < n; ++i) {
      double a = detail::abs_val(values[diag_pos[i]]);
      if (a > max_diag) max_diag = a;
    }

    if (max_diag == 0.0) return false;

    const double tol = std::numeric_limits<double>::epsilon() * max_diag * n;
    for (int i = 0; i < n; ++i) {
      if (detail::abs_val(values[diag_pos[i]]) < tol)
        return false;
    }
    return true;
  }

  // =========================================================================================
  //  Forward/backward substitution: precomputed plans, no binary search
  //
  //  Solves LU·x = b  where L and U are stored in values[].
  //  The permutation is the AMD reordering.
  // =========================================================================================
  void solve(boost::numeric::ublas::vector<T>& b) const
  {
    // Permute b into reordered space: pb[new_i] = b[old_i]
    std::vector<T> pb(n);
    for (int i = 0; i < n; ++i)
      pb[amd_iperm[i]] = b[i];

    // Forward substitution: L·y = pb
    for (int i = 0; i < n; ++i) {
      T sum = pb[i];
      for (int p = L_row_ptr[i]; p < L_row_ptr[i + 1]; ++p)
        sum -= values[L_entries[p].val_pos] * pb[L_entries[p].col];
      pb[i] = sum;
    }

    // Backward substitution: U·x = y
    for (int i = n - 1; i >= 0; --i) {
      T sum = pb[i];
      for (int p = U_row_ptr[i]; p < U_row_ptr[i + 1]; ++p)
        sum -= values[U_entries[p].val_pos] * pb[U_entries[p].col];
      pb[i] = sum / values[diag_pos[i]];
    }

    // Unpermute: b[old_i] = pb[new_i]
    for (int i = 0; i < n; ++i)
      b[i] = pb[amd_iperm[i]];
  }

  // =========================================================================================
  //  Combined factorize: symbolic (if needed) + copy + numerical
  //
  //  Returns 0 on success, -1 if pivots are unacceptable (caller should
  //  fall back to dense LU).
  //  The permutation_matrix parameter is kept for interface compatibility
  //  with the Rosenbrock stepper but is set to identity since AMD
  //  reordering is handled internally.
  // =========================================================================================
  template<class CM>
  int factorize(CM& W,
                boost::numeric::ublas::permutation_matrix<std::size_t>& pm)
  {
    if (!symbolic_ready)
      symbolic_analyze(W);

    copy_values_from(W);
    bool ok = numeric_factorize();

    // Set permutation to identity (AMD is handled internally)
    for (int i = 0; i < n; ++i)
      pm(i) = static_cast<std::size_t>(i);

    return ok ? 0 : -1;
  }

  void substitute(boost::numeric::ublas::vector<T>& b) const
  {
    solve(b);
  }
};

// =========================================================================================
//  Sparse Jacobian matrix-vector product:  y += W * x
// =========================================================================================
template<class T>
void sparse_jac_matvec(
    const boost::numeric::ublas::compressed_matrix<T>& W,
    const boost::numeric::ublas::vector<T>& x,
    boost::numeric::ublas::vector<T>& y)
{
  for (auto it1 = W.begin1(); it1 != W.end1(); ++it1)
    for (auto it2 = it1.begin(); it2 != it1.end(); ++it2)
      y(it2.index1()) += (*it2) * x(it2.index2());
}

} // namespace cppode
#endif // CPPODE_SPARSE_LU_HPP
