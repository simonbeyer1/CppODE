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
//
//  Block BH's stock rosenbrock4.hpp and generation_rosenbrock4.hpp before
//  the odeint aggregator pulls them in.  Both are permanently superseded by
//  cppode_boost_rosenbrock4.hpp which redefines rosenbrock4 with an extra
//  JacobianPattern parameter, containing the sparsitity patttern, and provides
//  its own get_controller / get_dense_output specialisations.
// ============================================================================
#define BOOST_NUMERIC_ODEINT_STEPPER_ROSENBROCK4_HPP_INCLUDED
#define BOOST_NUMERIC_ODEINT_STEPPER_GENERATION_GENERATION_ROSENBROCK4_HPP_INCLUDED
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

// ============================================================================
//  CppODE sparse LU for compressed_matrix
// ============================================================================
#include <cppode/cppode_sparse_lu.hpp>

// ============================================================================
//  CppODE utilities
// ============================================================================
#include <cppode/cppode_odeint_utils.hpp>
#include <cppode/cppode_pchip_forcing.hpp>

// ============================================================================
//  CppODE Rosenbrock stepper (drop-in replacement with sparse LU extension)
// ============================================================================
#include <cppode/cppode_boost_rosenbrock4.hpp>
#include <cppode/cppode_boost_rosenbrock4_controller_pi.hpp>
#include <cppode/cppode_boost_rosenbrock4_controller_pi_ad.hpp>
#include <cppode/cppode_boost_rosenbrock4_dense_output_pi.hpp>
#include <cppode/cppode_boost_rosenbrock4_dense_output_pi_ad.hpp>
#include <cppode/cppode_boost_integrate_times_with_events.hpp>
#include <cppode/cppode_boost_step_checker.hpp>

#endif // CPPODE_HPP
