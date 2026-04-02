/*
 Stepper traits for CppODE — compile-time dispatch between
 single-step (Rosenbrock) and multi-step (BDF) methods.

 Copyright (C) 2026 Simon Beyer

 This header defines stepper_traits<Stepper>, a type-trait struct that
 the EventEngine, integrate_times, and integrate_times_dense use to
 adapt their behaviour to different stepper families:

 - is_multistep:              true for BDF (History-based methods)
 - needs_restart_after_event: true if the stepper must discard its
 history and restart at order 1 after
 a state discontinuity (events)
 - min_order / max_order:     order range of the method

 The default (unspecialised) traits match Rosenbrock4 and other
 single-step methods.  BDF specialisations are provided via forward
 declarations; the actual bdf_stepper class is defined elsewhere.

 Distributed under the Boost Software License, Version 1.0.
 */

#ifndef CPPODE_STEPPER_TRAITS_HPP_INCLUDED
#define CPPODE_STEPPER_TRAITS_HPP_INCLUDED

#include <type_traits>

namespace cppode {

// ============================================================================
//  Primary template — defaults for single-step methods (Rosenbrock, etc.)
// ============================================================================

template<class Stepper, class = void>
struct stepper_traits {
  /// Single-step method: no history to manage
  static constexpr bool is_multistep = false;

  /// No restart needed after events — single-step methods are self-starting
  static constexpr bool needs_restart_after_event = false;

  /// Fixed order (not applicable for single-step, but provided for uniformity)
  static constexpr int min_order = 0;
  static constexpr int max_order = 0;
};
// ============================================================================
//  Trait extraction helpers (for use in if-constexpr / enable_if)
// ============================================================================

/// True if the stepper is a multi-step method (BDF, Adams, etc.)
template<class Stepper>
inline constexpr bool is_multistep_v = stepper_traits<Stepper>::is_multistep;

/// True if the stepper needs a full restart (history discard + order 1)
/// after a state discontinuity (event)
template<class Stepper>
inline constexpr bool needs_restart_after_event_v =
stepper_traits<Stepper>::needs_restart_after_event;
// ============================================================================
//  Forward declaration of BDF stepper (defined in cppode_bdf.hpp)
// ============================================================================

// The BDF specialisation is written here as a partial specialisation
// that matches any type with a nested typedef `is_bdf_stepper_tag`.
// The bdf_stepper class defines this tag, so the match is automatic
// without needing to know the full template parameter list.

/// SFINAE detector for BDF stepper tag
template<class T, class = void>
struct has_bdf_stepper_tag : std::false_type {};

template<class T>
struct has_bdf_stepper_tag<T, std::void_t<typename T::is_bdf_stepper_tag>>
: std::true_type {};

/// Specialisation for any type carrying the BDF tag
template<class Stepper>
struct stepper_traits<Stepper,
                     std::enable_if_t<has_bdf_stepper_tag<Stepper>::value>>
                     {
                       static constexpr bool is_multistep = true;
                       static constexpr bool needs_restart_after_event = true;
                       static constexpr int  min_order = 1;
                       static constexpr int  max_order = 5;
                     };
// ============================================================================
//  Trait propagation through wrapper layers
//
//  Controllers and dense-output wrappers typically store the inner stepper
//  as a member.  We propagate traits upward so that EventEngine can query
//  traits on the outermost wrapper without knowing the nesting depth.
//
//  Convention: any wrapper that exposes a `stepper_type` typedef referring
//  to its inner stepper inherits that stepper's traits, UNLESS it
//  explicitly specialises stepper_traits itself.
//
//  We detect the `stepper_type` typedef via SFINAE and forward traits.
// ============================================================================

/// SFINAE detector for nested stepper_type
template<class T, class = void>
struct has_inner_stepper_type : std::false_type {};

template<class T>
struct has_inner_stepper_type<T, std::void_t<typename T::stepper_type>>
: std::true_type {};

/// Detector for BDF tag on inner stepper (through wrapper chain)
template<class T, class = void>
struct inner_has_bdf_tag : std::false_type {};

template<class T>
struct inner_has_bdf_tag<T,
                         std::enable_if_t<
                           has_inner_stepper_type<T>::value &&
                           !has_bdf_stepper_tag<T>::value &&
                           !std::is_same<T, typename T::stepper_type>::value>>  // guard against self-referential stepper_type
                           : has_bdf_stepper_tag<typename T::stepper_type> {};

// Recursive version: chase through multiple wrapper layers
// (e.g. dense_output → controller → stepper)
//
// Guard: stop recursion when T::stepper_type is the same type as T
// (some Boost steppers like rosenbrock4 define a self-referential
// stepper_type typedef, which would cause infinite template recursion).
template<class T, class = void>
struct deep_has_bdf_tag : has_bdf_stepper_tag<T> {};

template<class T>
struct deep_has_bdf_tag<T,
                        std::enable_if_t<
                          has_inner_stepper_type<T>::value &&
                          !has_bdf_stepper_tag<T>::value &&
                          !std::is_same<T, typename T::stepper_type>::value>>
                          : deep_has_bdf_tag<typename T::stepper_type> {};

/// Propagated traits for wrapper types (controller, dense output)
/// that wrap a BDF stepper somewhere in their chain
template<class Wrapper>
struct stepper_traits<Wrapper,
                      std::enable_if_t<
                        !has_bdf_stepper_tag<Wrapper>::value &&
                        deep_has_bdf_tag<Wrapper>::value>>
                        {
                          static constexpr bool is_multistep = true;
                          static constexpr bool needs_restart_after_event = true;
                          static constexpr int  min_order = 1;
                          static constexpr int  max_order = 5;
                        };

} // namespace cppode

#endif // CPPODE_STEPPER_TRAITS_HPP_INCLUDED
