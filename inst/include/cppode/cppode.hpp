/*
 Main header for CppODE – ODE integration and sensitivity calculation
 using FADBAD++ automatic differentiation.

 The stepper architecture (Rosenbrock4, BDF) is derived from Boost.Odeint
 by Karsten Ahnert, Mario Mulansky, and Christoph Koke (2011–2015),
 distributed under the Boost Software License, Version 1.0.
 Substantially rewritten: LAPACK/KLU linear algebra, AD-aware LU
 decomposition, event handling, BDF support, PI step-size control.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_HPP
#define CPPODE_HPP

// ============================================================================
//  Standard library
// ============================================================================
#include <algorithm>
#include <vector>
#include <cmath>
#include <functional>

// ============================================================================
//  FADBAD++
// ============================================================================
#include <fadbad++/fadiff.h>

// ============================================================================
//  FADBAD++ extensions
// ============================================================================
#include <cppode/cppode_fadiff_extensions.hpp>

// ============================================================================
//  CppODE types and infrastructure
// ============================================================================
#include <cppode/cppode_types.hpp>
#include <cppode/cppode_odeint_compat.hpp>

// ============================================================================
//  CppODE AD-aware LU solvers (IFT for dense and sparse)
// ============================================================================
#include <cppode/cppode_ad_lu.hpp>
#include <cppode/cppode_sparse_ad_lu.hpp>

// ============================================================================
//  CppODE unified LU iteration matrix solver
// ============================================================================
#include <cppode/cppode_lu.hpp>

// ============================================================================
//  CppODE Newton solver
// ============================================================================
#include <cppode/cppode_newton.hpp>

// ============================================================================
//  CppODE utilities
// ============================================================================
#include <cppode/cppode_utils.hpp>
#include <cppode/cppode_pchip_forcing.hpp>

// ============================================================================
//  CppODE stepper traits (multi-step vs single-step dispatch)
// ============================================================================
#include <cppode/cppode_stepper_traits.hpp>

// ============================================================================
//  CppODE Rosenbrock4 stepper (unified: double + AD)
// ============================================================================
#include <cppode/cppode_rosenbrock4.hpp>
#include <cppode/cppode_rosenbrock4_controller.hpp>
#include <cppode/cppode_rosenbrock4_dense_output.hpp>
#include <cppode/cppode_integrate_times.hpp>
#include <cppode/cppode_step_checker.hpp>

// ============================================================================
//  CppODE BDF stepper (unified: double + AD, variable order 1–5)
// ============================================================================
#include <cppode/cppode_bdf.hpp>
#include <cppode/cppode_bdf_controller.hpp>
#include <cppode/cppode_bdf_dense_output.hpp>

#endif // CPPODE_HPP
