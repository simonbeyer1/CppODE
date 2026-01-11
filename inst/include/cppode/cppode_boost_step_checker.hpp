/*
 Combined step checker for ODE integration monitoring.
 Extended version of Boost.Odeint's max_step_checker.

 Original work:
 Copyright 2015 Mario Mulansky
 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)

 Modified work:
 Copyright 2026 Simon Beyer

 Modifications:
 - Combined no-progress and total step checking
 - Simplified interface for event-driven integration
 - Added total step limit alongside no-progress limit
 */

#ifndef CPPODE_BOOST_STEP_CHECKER_HPP_INCLUDED
#define CPPODE_BOOST_STEP_CHECKER_HPP_INCLUDED

#include <stdexcept>
#include <cstdio>
#include <boost/throw_exception.hpp>
#include <boost/numeric/odeint/util/odeint_error.hpp>

namespace boost {
namespace numeric {
namespace odeint {

/**
 * @class StepChecker
 * @brief Combined step checker for Boost.Odeint integration
 *
 * Monitors both:
 * - **No-progress steps**: Consecutive step attempts without time advance
 * - **Total steps**: Absolute maximum number of integration steps
 *
 * This class extends the original Boost.Odeint max_step_checker concept
 * by combining two independent limits into a single unified interface.
 *
 * @par Usage
 * @code
 * StepChecker checker(500, 100000);  // max 500 no-progress, 100k total
 *
 * // In integration loop:
 * if (step_successful) {
 *     checker.reset();  // Reset no-progress counter
 * }
 * checker();  // Increment and check limits
 * @endcode
 *
 * @throw boost::numeric::odeint::no_progress_error if limits exceeded
 *
 * @see boost::numeric::odeint::max_step_checker (original implementation)
 */
class StepChecker {
public:
  /**
   * @brief Construct the step checker with configurable limits
   *
   * @param max_no_progress_steps Maximum consecutive steps without
   *                              time advance (default: 500)
   * @param max_total_steps       Absolute maximum number of steps
   *                              during integration (default: 100000)
   *
   * @par Rationale for defaults
   * - 500 no-progress steps: Allows aggressive step-size reduction
   *   in stiff regions while preventing infinite loops
   * - 100000 total steps: Permits long-duration integrations while
   *   catching runaway simulations
   */
  StepChecker(int max_no_progress_steps = 500, int max_total_steps = 100000)
    : m_max_no_progress_steps(max_no_progress_steps),
      m_max_total_steps(max_total_steps),
      m_no_progress_steps(0),
      m_total_steps(0) {}

  /**
   * @brief Reset the no-progress counter
   *
   * Call this after a successful step that advances time.
   * The total step counter is **not** reset.
   *
   * @note Typically called by integration observers after confirming
   *       time has advanced (e.g., t_new > t_old)
   */
  void reset() {
    m_no_progress_steps = 0;
  }

  /**
   * @brief Increment counters and check limits
   *
   * Increments both no-progress and total step counters.
   * Throws exception if either limit is exceeded.
   *
   * @throw boost::numeric::odeint::no_progress_error if
   *        no-progress limit exceeded
   * @throw boost::numeric::odeint::no_progress_error if
   *        total step limit exceeded
   *
   * @note Uses no_progress_error for both cases to maintain
   *       compatibility with existing Boost.Odeint error handling
   */
  void operator()() {
    if (m_no_progress_steps++ >= m_max_no_progress_steps) {
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "Maximum number of steps without progress exceeded (%d).",
                    m_max_no_progress_steps);
      BOOST_THROW_EXCEPTION(no_progress_error(msg));
    }
    if (m_total_steps++ >= m_max_total_steps) {
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "Maximum number of steps exceeded (%d).",
                    m_max_total_steps);
      BOOST_THROW_EXCEPTION(no_progress_error(msg));
    }
  }

  /**
   * @brief Get current no-progress step count
   * @return Number of consecutive steps without time advance
   */
  int no_progress_steps() const { return m_no_progress_steps; }

  /**
   * @brief Get total step count
   * @return Total number of steps since construction
   */
  int total_steps() const { return m_total_steps; }

private:
  const int m_max_no_progress_steps;  ///< No-progress step limit
  const int m_max_total_steps;        ///< Total step limit
  int m_no_progress_steps;            ///< Current no-progress count
  int m_total_steps;                  ///< Current total count
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // CPPODE_BOOST_STEP_CHECKER_HPP_INCLUDED
