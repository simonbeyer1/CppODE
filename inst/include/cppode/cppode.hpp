/*
 Main header for CppODE – ODE integration and sensitivity calculation
 using FADBAD++ automatic differentiation.

 The stepper architecture (Rosenbrock4, NDF/BDF) is derived from Boost.Odeint
 by Karsten Ahnert, Mario Mulansky, and Christoph Koke (2011–2015),
 distributed under the Boost Software License, Version 1.0.
 Substantially rewritten: LAPACK/KLU linear algebra, AD-aware LU
 decomposition, event handling, NDF/BDF support, PI step-size control.

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
//  FADBAD++ (legacy AD backend)
// ============================================================================
#include <fadbad++/fadiff.h>

// ============================================================================
//  FADBAD++ extensions
// ============================================================================
#include <cppode/cppode_fadiff_extensions.hpp>

// ============================================================================
//  cppode::dual / cppode::dual2nd (custom forward AD backend)
//  Selected at codegen time via the R-side `ad_backend` argument.
// ============================================================================
#include <cppode/cppode_dual_math.hpp>
#include <cppode/cppode_dual_expr.hpp>

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
//  CppODE single-step methods (Rosenbrock4, Tsit5)
//
//  Unified onestep_controller and onestep_dense_output work with any
//  single-step stepper.  The old rosenbrock4_controller / _dense_output
//  headers are thin wrappers that include these.
// ============================================================================
#include <cppode/cppode_rosenbrock4.hpp>
#include <cppode/cppode_tsit5.hpp>
#include <cppode/cppode_onestep_controller.hpp>
#include <cppode/cppode_onestep_dense_output.hpp>
#include <cppode/cppode_integrate_times.hpp>
#include <cppode/cppode_step_checker.hpp>

// ============================================================================
//  CppODE multistep family (BDF / Adams / MSODA)
//
//  Single unified multistepper class (cppode::multistepper) with a
//  method selector (multistep_method enum).  Default coefficients are
//  the Klopfenstein-Shampine NDF family (Shampine & Reichelt 1997);
//  classical BDF, pure Adams-Moulton, and the two LSODA-style
//  switching variants are all instantiations of the same class.
// ============================================================================
#include <cppode/cppode_multistepper.hpp>
#include <cppode/cppode_multistepper_controller.hpp>
#include <cppode/cppode_multistepper_dense_output.hpp>

#endif // CPPODE_HPP
