/*
 ODE integration infrastructure for CppODE — Eigen-free.

 Provides:
 - state_wrapper<T> for std::vector and dense_matrix
 - Resizeable trait and resize helpers
 - Stepper tags, step result enum
 - Comparison utilities

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_ODEINT_COMPAT_HPP
#define CPPODE_ODEINT_COMPAT_HPP

#include <cstddef>
#include <cmath>
#include <type_traits>
#include <functional>
#include <utility>
#include <vector>

#include <cppode/cppode_types.hpp>
#include <cppode/cppode_step_checker.hpp>

namespace cppode {

// ============================================================================
//  Stepper category tags
// ============================================================================

struct stepper_tag {};
struct controlled_stepper_tag {};
struct dense_output_stepper_tag {};

// ============================================================================
//  Step result enum
// ============================================================================

enum controlled_step_result {
  success = 0,
  fail    = 1
};

// ============================================================================
//  state_wrapper<T> — thin wrapper providing resize semantics
// ============================================================================

template<class T>
struct state_wrapper {
  T m_v;
};

// ============================================================================
//  is_resizeable<T> trait
// ============================================================================

template<class T, class = void>
struct is_resizeable : std::false_type {};

// std::vector<T> is always resizeable
template<class T>
struct is_resizeable<std::vector<T>> : std::true_type {};

// dense_matrix<T> is always resizeable
template<class T>
struct is_resizeable<dense_matrix<T>> : std::true_type {};

// csc_matrix<T> is always resizeable
template<class T>
struct is_resizeable<csc_matrix<T>> : std::true_type {};

// ============================================================================
//  adjust_size_by_resizeability
//
//  Resizes wrapper.m_v to match reference x.
//  - vector wrapper + vector reference: match sizes
//  - dense_matrix wrapper + vector reference: resize as square (n × n)
//  - csc_matrix wrapper + vector reference: set n (pattern rebuilt lazily)
// ============================================================================

namespace detail {

// --- std::vector resize ---
template<class T, class U>
inline bool do_adjust_size(state_wrapper<std::vector<T>>& wrapper,
                          const std::vector<U>& x,
                          std::true_type)
{
 if (wrapper.m_v.size() != x.size()) {
   wrapper.m_v.assign(x.size(), T(0));
   return true;
 }
 return false;
}

// --- dense_matrix resize (square, from vector reference) ---
template<class T, class U>
inline bool do_adjust_size(state_wrapper<dense_matrix<T>>& wrapper,
                          const std::vector<U>& x,
                          std::true_type)
{
 const int n = static_cast<int>(x.size());
 if (wrapper.m_v.rows() != n || wrapper.m_v.cols() != n) {
   wrapper.m_v.resize(n, n);
   return true;
 }
 return false;
}

// --- csc_matrix resize ---
template<class T, class U>
inline bool do_adjust_size(state_wrapper<csc_matrix<T>>& wrapper,
                          const std::vector<U>& x,
                          std::true_type)
{
 const int n = static_cast<int>(x.size());
 if (wrapper.m_v.n != n) {
   wrapper.m_v.n = n;
   wrapper.m_v.nnz = 0;
   wrapper.m_v.pattern_built = false;
   wrapper.m_v.Ap.clear();
   wrapper.m_v.Ai.clear();
   wrapper.m_v.Ax.clear();
   return true;
 }
 return false;
}

// --- not resizeable (no-op) ---
template<class Wrapper, class Reference>
inline bool do_adjust_size(Wrapper&, const Reference&, std::false_type)
{
 return false;
}

} // namespace detail

template<class Wrapper, class Reference>
inline bool adjust_size_by_resizeability(Wrapper& wrapper, const Reference& x)
{
 using inner_type = typename std::decay<decltype(wrapper.m_v)>::type;
 return detail::do_adjust_size(wrapper, x, typename is_resizeable<inner_type>::type{});
}

// ============================================================================
//  initially_resizer — resize policy that adjusts once
// ============================================================================

struct initially_resizer {
 bool m_is_initialized = false;

 template<class State, class ResizeFunc>
 bool adjust_size(const State& x, ResizeFunc func)
 {
   if (!m_is_initialized) {
     m_is_initialized = true;
     return func(x);
   }
   return false;
 }
};

// ============================================================================
//  failed_step_checker
// ============================================================================

class failed_step_checker {
public:
 explicit failed_step_checker(int max_steps = 500)
   : m_max_steps(max_steps), m_count(0) {}

 void operator()() {
   if (++m_count > m_max_steps)
     throw no_progress_error("Too many failed steps in dense output stepper.");
 }

 void reset() { m_count = 0; }

private:
 int m_max_steps;
 int m_count;
};

// ============================================================================
//  Comparison utilities
// ============================================================================

namespace detail {

template<class T>
inline bool less_with_sign(T a, T b, T dt)
{ return (dt > 0) ? (a < b) : (a > b); }

template<class T>
inline bool less_eq_with_sign(T a, T b, T dt)
{ return (dt > 0) ? (a <= b) : (a >= b); }

} // namespace detail

// ============================================================================
//  unwrap_reference
// ============================================================================

template<class T>
struct unwrap_reference { typedef T type; };

template<class T>
struct unwrap_reference<std::reference_wrapper<T>> { typedef T type; };

} // namespace cppode

#endif // CPPODE_ODEINT_COMPAT_HPP
