/*
 Centralized AD type traits for CppODE.

 Consolidates is_ad / inner_type / scalar_type / scalar_value plus the bulk
 value/derivative extraction helpers (extract_values, extract_derivs,
 max_deriv_size, bulk_extract_derivs, bulk_inject_results) into a single
 header. Specialized for fadbad::F<T,N> (legacy backend) and cppode::dual<T,N>
 (in-tree backend). Second-order via cppode::dual2nd<T,N> = dual<dual<T,N>,N>
 falls out of the recursive specialisations — no separate trait entries.

 The bulk helpers are written generically over any AD type that exposes the
 FADBAD-compatible accessor surface: `.x()`, `.d(j)`, `.size()`, `.depend()`,
 `.diff(idx[, n])`, `operator[](j)`. Both fadbad::F and cppode::dual satisfy
 this surface, so the same code instantiates for either backend.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_AD_TRAITS_HPP
#define CPPODE_AD_TRAITS_HPP

#include <type_traits>
#include <vector>

#include <fadbad++/fadiff.h>
#include <cppode/cppode_types.hpp>

// Forward declaration for the dual backend so traits can specialise without
// pulling in the full headers from every call site.
namespace cppode {
  template<class T, unsigned N> class dual;
}

namespace cppode {
namespace ad_traits {

// ============================================================================
//  is_ad<T>
// ============================================================================

template<class T>           struct is_ad : std::false_type {};
template<class T, unsigned int N> struct is_ad<fadbad::F<T, N>> : std::true_type {};
template<class T, unsigned N>     struct is_ad<cppode::dual<T, N>> : std::true_type {};

// ============================================================================
//  inner_type<T> — strip ONE layer of AD wrapping
// ============================================================================

template<class T>           struct inner_type       { using type = T; };
template<class T, unsigned int N> struct inner_type<fadbad::F<T, N>> { using type = T; };
template<class T, unsigned N>     struct inner_type<cppode::dual<T, N>> { using type = T; };
template<class T> using inner_type_t = typename inner_type<T>::type;

// ============================================================================
//  scalar_type<T> — recursively unwrap to innermost non-AD scalar
// ============================================================================

template<class T>           struct scalar_type       { using type = T; };
template<class T, unsigned int N> struct scalar_type<fadbad::F<T, N>>     : scalar_type<T> {};
template<class T, unsigned N>     struct scalar_type<cppode::dual<T, N>>    : scalar_type<T> {};
template<class T> using scalar_type_t = typename scalar_type<T>::type;

// ============================================================================
//  scalar_value(v) — extract innermost double from any (nested) AD type
// ============================================================================

template<class T>
inline typename std::enable_if<std::is_arithmetic<T>::value, double>::type
scalar_value(const T& v) { return static_cast<double>(v); }

template<class T, unsigned int N>
inline double scalar_value(const fadbad::F<T, N>& v) {
  return scalar_value(const_cast<fadbad::F<T, N>&>(v).x());
}

template<class T, unsigned N>
inline double scalar_value(const cppode::dual<T, N>& v) {
  return scalar_value(v.x());
}

// ============================================================================
//  Bulk extraction / injection helpers (generic over any AD type with the
//  FADBAD-compatible accessor surface). All previously FADBAD-only.
// ============================================================================

// Extract value (.x()) from each element: vector<AD> -> vector<inner>
template<class AD,
         std::enable_if_t<is_ad<AD>::value, int> = 0>
inline std::vector<inner_type_t<AD>>
extract_values(const std::vector<AD>& v)
{
  using Inner = inner_type_t<AD>;
  const std::size_t n = v.size();
  std::vector<Inner> out(n);
  for (std::size_t i = 0; i < n; ++i)
    out[i] = const_cast<AD&>(v[i]).x();
  return out;
}

// Extract value (.x()) from each element: dense_matrix<AD> -> dense_matrix<inner>
template<class AD,
         std::enable_if_t<is_ad<AD>::value, int> = 0>
inline dense_matrix<inner_type_t<AD>>
extract_values(const dense_matrix<AD>& M)
{
  using Inner = inner_type_t<AD>;
  dense_matrix<Inner> out(M.rows(), M.cols());
  const std::size_t sz = M.data.size();
  for (std::size_t k = 0; k < sz; ++k)
    out.data[k] = const_cast<AD&>(M.data[k]).x();
  return out;
}

// Extract derivative .d(j) from each element of a vector
template<class AD,
         std::enable_if_t<is_ad<AD>::value, int> = 0>
inline std::vector<inner_type_t<AD>>
extract_derivs(const std::vector<AD>& v, unsigned j)
{
  using Inner = inner_type_t<AD>;
  const std::size_t n = v.size();
  std::vector<Inner> out(n);
  for (std::size_t i = 0; i < n; ++i)
    out[i] = const_cast<AD&>(v[i]).d(j);
  return out;
}

// Extract derivative .d(j) from each element of a dense_matrix
template<class AD,
         std::enable_if_t<is_ad<AD>::value, int> = 0>
inline dense_matrix<inner_type_t<AD>>
extract_derivs(const dense_matrix<AD>& M, unsigned j)
{
  using Inner = inner_type_t<AD>;
  dense_matrix<Inner> out(M.rows(), M.cols());
  const std::size_t sz = M.data.size();
  for (std::size_t k = 0; k < sz; ++k)
    out.data[k] = const_cast<AD&>(M.data[k]).d(j);
  return out;
}

// Max number of active derivative directions in a vector
template<class AD,
         std::enable_if_t<is_ad<AD>::value, int> = 0>
inline unsigned max_deriv_size(const std::vector<AD>& v)
{
  unsigned mx = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    unsigned sz = const_cast<AD&>(v[i]).size();
    if (sz > mx) mx = sz;
  }
  return mx;
}

// Max number of active derivative directions in a dense_matrix
template<class AD,
         std::enable_if_t<is_ad<AD>::value, int> = 0>
inline unsigned max_deriv_size(const dense_matrix<AD>& M)
{
  unsigned mx = 0;
  for (std::size_t k = 0; k < M.data.size(); ++k) {
    unsigned sz = const_cast<AD&>(M.data[k]).size();
    if (sz > mx) mx = sz;
  }
  return mx;
}

// ----------------------------------------------------------------------------
//  AoS -> SoA bulk extraction
//
//  bulk_extract_derivs:
//    Single pass over vector<AD>, writing into a column-major
//    n x n_derivs scratch buffer of inner-type entries. Replaces n_derivs
//    separate calls to extract_derivs(v, j).
//
//  bulk_inject_results:
//    Single pass over vector<AD>, reading values from b_val and derivative
//    columns from dx_all (column-major n x n_derivs). Activates dependence
//    via diff(0[,n_derivs]) on AD instances that are not yet "depending",
//    then overwrites all tangent slots via operator[].
// ----------------------------------------------------------------------------

template<class AD,
         std::enable_if_t<is_ad<AD>::value, int> = 0>
inline void bulk_extract_derivs(
    const std::vector<AD>& v,
    std::vector<inner_type_t<AD>>& out,
    int n, unsigned n_derivs)
{
  using Inner = inner_type_t<AD>;
  out.resize(static_cast<std::size_t>(n) * n_derivs);
  for (int i = 0; i < n; ++i) {
    auto& vi = const_cast<AD&>(v[i]);
    unsigned sz = vi.size();
    for (unsigned j = 0; j < n_derivs; ++j) {
      out[j * n + i] = (j < sz) ? vi.d(j) : Inner(0);
    }
  }
}

namespace detail {
// Compile-time width: for AD types that expose a static size (FADBAD's
// template parameter N, dual's static_size). Defaults to 0 = dynamic.
template<class AD> struct ad_static_size : std::integral_constant<unsigned, 0> {};
template<class T, unsigned int N>
struct ad_static_size<fadbad::F<T, N>> : std::integral_constant<unsigned, N> {};
template<class T, unsigned N>
struct ad_static_size<cppode::dual<T, N>> : std::integral_constant<unsigned, N> {};
} // namespace detail

template<class AD,
         std::enable_if_t<is_ad<AD>::value, int> = 0>
inline void bulk_inject_results(
    std::vector<AD>& b,
    const std::vector<inner_type_t<AD>>& b_val,
    const std::vector<inner_type_t<AD>>& dx_all,
    int n, unsigned n_derivs)
{
  constexpr unsigned StaticN = detail::ad_static_size<AD>::value;
  for (int i = 0; i < n; ++i) {
    b[i].x() = b_val[i];
    if (n_derivs > 0) {
      // Activate dependence (sets internal flag / allocates tangent vec).
      // Static N: diff(0) seeds m_diff[0]=1 etc; we overwrite all slots below.
      // Dynamic N==0: diff(0, n_derivs) sizes the tangent buffer.
      if constexpr (StaticN > 0) {
        if (!b[i].depend()) b[i].diff(0);
      } else {
        if (!b[i].depend()) b[i].diff(0, n_derivs);
      }
      for (unsigned j = 0; j < n_derivs; ++j)
        b[i][j] = dx_all[j * n + i];
    }
  }
}

} // namespace ad_traits
} // namespace cppode

#endif // CPPODE_AD_TRAITS_HPP
