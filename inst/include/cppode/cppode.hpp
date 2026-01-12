#ifndef CPPODE_HPP
#define CPPODE_HPP

/**
 * @file cppode.hpp
 * @brief Main header for CppODE - ODE integration and sensitivity calculation with FADBAD++ automatic differentiation
 *
 * This header includes all necessary components:
 *   1. Standard library headers
 *   2. FADBAD++ (unmodified fadiff.h)
 *   3. FADBAD++ extensions (abs, nested type comparisons, std imports)
 *   4. Boost.Odeint
 *   5. CppODE utility functions (scalar extraction, norms, initial dt)
 *   6. CppODE Rosenbrock controller and dense output headers
 *
 * @author Simon Beyer <simon.beyer@fdm.uni-freiburg.de>
 */

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
#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>

// ============================================================================
//  CppODE utilities
// ============================================================================
#include <cppode/cppode_odeint_utils.hpp>

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
