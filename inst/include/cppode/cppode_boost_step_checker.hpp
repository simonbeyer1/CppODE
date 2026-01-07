#ifndef STEP_CHECKER_HPP
#define STEP_CHECKER_HPP

#include <stdexcept>
#include <cstdio>
#include <boost/throw_exception.hpp>
#include <boost/numeric/odeint/util/odeint_error.hpp>

namespace boost {
namespace numeric {
namespace odeint {

/**
 * \class StepChecker
 *
 * A combined step checker for Boost.Odeint.
 * - Monitors "no-progress" steps (like max_step_checker)
 * - Monitors an absolute maximum number of steps
 *
 * If either limit is exceeded, an exception is thrown.
 */
class StepChecker {
public:
  /**
   * Construct the step checker.
   *
   * @param max_no_progress_steps  Maximum number of consecutive steps
   *                               without advancing time (default: 500).
   * @param max_total_steps        Absolute maximum number of steps
   *                               during integration (default: 100000).
   */
  StepChecker(int max_no_progress_steps = 500, int max_total_steps = 100000)
    : m_max_no_progress_steps(max_no_progress_steps),
      m_max_total_steps(max_total_steps),
      m_no_progress_steps(0),
      m_total_steps(0) {}

  /**
   * Reset the "no-progress" counter.
   * This is called, for example, by the checked_observer
   * after a successful step that advanced time.
   */
  void reset() {
    m_no_progress_steps = 0;
  }

  /**
   * Called on every internal step attempt.
   * Increments both the no-progress counter and the total step counter.
   * Throws an exception if limits are exceeded.
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

private:
  const int m_max_no_progress_steps;
  const int m_max_total_steps;
  int m_no_progress_steps;
  int m_total_steps;
};

} // namespace odeint
} // namespace numeric
} // namespace boost

#endif // STEP_CHECKER_HPP
