/*
 Main header for CppODE – ODE integration and sensitivity calculation
 using FADBAD++ automatic differentiation and Boost.Odeint.

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
//  Boost.Odeint and uBLAS
// ============================================================================
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

// ============================================================================
//  CppODE utilities
// ============================================================================
#include <cppode/cppode_odeint_utils.hpp>
#include <cppode/cppode_pchip_forcing.hpp>

// ============================================================================
//  CppODE Rosenbrock controllers and dense output
// ============================================================================
#include <cppode/cppode_boost_rosenbrock4_controller_pi.hpp>
#include <cppode/cppode_boost_rosenbrock4_controller_pi_ad.hpp>
#include <cppode/cppode_boost_rosenbrock4_dense_output_pi.hpp>
#include <cppode/cppode_boost_rosenbrock4_dense_output_pi_ad.hpp>
#include <cppode/cppode_boost_integrate_times_with_events.hpp>
#include <cppode/cppode_boost_step_checker.hpp>

#endif // CPPODE_HPP
