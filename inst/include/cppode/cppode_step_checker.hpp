/*
 Combined step checker for ODE integration monitoring.

 The step checker concept is inspired by Boost.Odeint's max_step_checker
 (Mario Mulansky, 2015, Boost Software License 1.0).
 Independently reimplemented with extended diagnostics.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_STEP_CHECKER_HPP
#define CPPODE_STEP_CHECKER_HPP

#include <stdexcept>
#include <cstdio>

#include "cppode_return_codes.hpp"

namespace cppode {

/**
 * @class no_progress_error
 * @brief Exception thrown when the integrator cannot make progress
 *
 * Thrown when too many consecutive steps fail to advance time,
 * or when the total step count exceeds the configured limit.
 */
class no_progress_error : public std::runtime_error {
public:
  explicit no_progress_error(const char* msg) : std::runtime_error(msg) {}
  explicit no_progress_error(const std::string& msg) : std::runtime_error(msg) {}
};
/**
 * @class StepChecker
 * @brief Combined step checker for ODE integration with diagnostics
 *
 * Monitors both:
 * - **No-progress steps**: Consecutive step attempts without time advance
 * - **Total steps**: Absolute maximum number of integration steps
 *
 * Also collects solver diagnostics (return code, accepted/rejected steps,
 * function evaluations, Jacobian evaluations, last step size, last/next
 * method order, and the time reached by the solver).
 *
 * Return codes: numerically identical to the SUNDIALS CVODE flag scheme
 * (see cppode_return_codes.hpp). Typical values set by this class:
 *    0 = RC_SUCCESS          (default, set by caller on success)
 *   -1 = RC_TOO_MUCH_WORK    (maximum total steps exceeded)
 *   -4 = RC_CONV_FAILURE     (maximum no-progress steps exceeded,
 *                             i.e. too many consecutive rejected steps)
 * The caller may also set any other CVODE flag (e.g. RC_LSETUP_FAIL,
 * RC_UNRECOGNIZED_ERR) via set_return_code() when a downstream error
 * cannot be expressed as a step-limit violation.
 *
 * @par Usage
 * @code
 * StepChecker checker(500, 100000);
 *
 * // In integration loop:
 * if (step_successful) {
 *     checker.reset();  // Reset no-progress counter
 * }
 * checker();  // Increment and check limits
 * @endcode
 *
 * @throw cppode::no_progress_error if limits exceeded
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
   */
  StepChecker(int max_no_progress_steps = 500, int max_total_steps = 100000)
    : m_max_no_progress_steps(max_no_progress_steps),
      m_max_total_steps(max_total_steps),
      m_no_progress_steps(0),
      m_total_steps(0),
      m_return_code(0),
      m_n_accepted(0),
      m_n_rejected(0),
      m_n_fevals(0),
      m_n_jevals(0),
      m_n_setups(0),
      m_last_dt(0.0),
      m_last_order(0),
      m_next_order(0),
      m_t_reached(0.0) {}

  /**
   * @brief Reset the no-progress counter
   *
   * Call this after a successful step that advances time.
   * The total step counter is **not** reset.
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
   * @throw cppode::no_progress_error if
   *        no-progress or total step limit exceeded
   */
  void operator()() {
    if (m_no_progress_steps++ >= m_max_no_progress_steps) {
      m_return_code = RC_CONV_FAILURE;
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "Maximum number of steps without progress exceeded (%d).",
                    m_max_no_progress_steps);
      throw no_progress_error(msg);
    }
    if (m_total_steps++ >= m_max_total_steps) {
      m_return_code = RC_TOO_MUCH_WORK;
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "Maximum number of steps exceeded (%d).",
                    m_max_total_steps);
      throw no_progress_error(msg);
    }
  }

  // --- Accessors for step counters ---
  int no_progress_steps() const { return m_no_progress_steps; }
  int total_steps() const { return m_total_steps; }

  // --- Diagnostics ---

  // Return code using the CVODE flag scheme (see cppode_return_codes.hpp):
  //    0 = RC_SUCCESS
  //   -1 = RC_TOO_MUCH_WORK        (max steps)
  //   -4 = RC_CONV_FAILURE         (no-progress / repeated conv failures)
  //   -6 = RC_LSETUP_FAIL          (linear-solver setup failed)
  //   -8 = RC_RHSFUNC_FAIL         (RHS function returned unrecoverable error)
  //  -99 = RC_UNRECOGNIZED_ERR     (unclassified C++ exception)
  int return_code() const { return m_return_code; }
  void set_return_code(int rc) { m_return_code = rc; }

  // Number of accepted steps
  int n_accepted() const { return m_n_accepted; }
  void add_accepted() { ++m_n_accepted; }
  void add_accepted(int n) { m_n_accepted += n; }

  // Number of rejected steps
  int n_rejected() const { return m_n_rejected; }
  void add_rejected() { ++m_n_rejected; }
  void add_rejected(int n) { m_n_rejected += n; }

  // Number of RHS function evaluations
  int n_fevals() const { return m_n_fevals; }
  void add_fevals(int n = 1) { m_n_fevals += n; }

  // Number of Jacobian evaluations (and LU decompositions)
  int n_jevals() const { return m_n_jevals; }
  void add_jevals(int n = 1) { m_n_jevals += n; }

  // Number of linear solver setups (LU factorizations / refactorizations)
  int n_setups() const { return m_n_setups; }
  void add_setups(int n = 1) { m_n_setups += n; }

  // Last successful step size
  double last_dt() const { return m_last_dt; }
  void set_last_dt(double dt) { m_last_dt = dt; }

  // Last method order used (successfully)
  int last_order() const { return m_last_order; }
  void set_last_order(int o) { m_last_order = o; }

  // Method order to be attempted on the next step
  int next_order() const { return m_next_order; }
  void set_next_order(int o) { m_next_order = o; }

  // Time reached by the solver
  double t_reached() const { return m_t_reached; }
  void set_t_reached(double t) { m_t_reached = t; }

private:
  const int m_max_no_progress_steps;
  const int m_max_total_steps;
  int m_no_progress_steps;
  int m_total_steps;

  // --- Solver diagnostics ---
  int    m_return_code;
  int    m_n_accepted;
  int    m_n_rejected;
  int    m_n_fevals;
  int    m_n_jevals;
  int    m_n_setups;
  double m_last_dt;
  int    m_last_order;
  int    m_next_order;
  double m_t_reached;
};

} // namespace cppode

#endif // CPPODE_STEP_CHECKER_HPP
