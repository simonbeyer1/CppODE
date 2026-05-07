/*
 CppODE Profiler: compile-time switchable performance instrumentation
 =====================================================================

 Zero-cost when disabled: the compiler eliminates all profiler code
 entirely when CPPODE_PROFILE is not defined.

 Two levels:
   CPPODE_PROFILE         : cumulative per-category timers (summary)
   CPPODE_PROFILE_VERBOSE : additionally prints per-step detail to stderr

 Usage in stepper/controller code:

   #include <cppode/cppode_profiler.hpp>

   // Member:
   cppode::profiler m_prof;

   // Instrument a block:
   {
     auto _t = m_prof.timer(cppode::prof_cat::f_eval);
     deriv_func(y, ftemp, t);
   }

   // At end (e.g. in transfer_stepper_diagnostics):
   m_prof.report();   // prints to stderr
   m_prof.fill_diagnostics(diag_struct);  // or populate R output

 When CPPODE_PROFILE is not defined, profiler is an empty struct,
 timer() returns a no-op, and everything compiles away.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_PROFILER_HPP
#define CPPODE_PROFILER_HPP

#include <cstdio>

#ifdef CPPODE_PROFILE
#include <chrono>
#include <array>
#include <cstring>
#endif

namespace cppode {

// ============================================================================
//  Profiling categories
// ============================================================================

enum class prof_cat : unsigned {
  f_eval = 0,       // RHS function evaluations
  jac_eval,         // Jacobian evaluations (codegen functor call)
  lu_factor,        // LU factorization (LAPACK dgetrf / KLU factor+refactor)
  lu_solve,         // LU solve (forward/back substitution, incl. IFT)
  nordsieck,        // Nordsieck predict/restore/rescale/complete/increase/decrease
  dense_interp,     // Dense output interpolation (calc_state)
  dense_snapshot,   // prepare_dense_output (Nordsieck copy / Hermite coeffs)
  observer,         // Observer callback (push_back into result vectors)
  error_norm,       // WRMS norm computations (Newton convergence + error test)
  newton_overhead,  // Newton loop overhead (residual formation, updates, excl. f_eval/lu_solve/error_norm)
  w_build,          // W = (1/γh)I - J  matrix construction (copy + diagonal add)
  step_overhead,    // Everything else in do_step / try_step not covered above
  COUNT             // sentinel: number of categories
};

static constexpr unsigned PROF_N = static_cast<unsigned>(prof_cat::COUNT);

// Category names for reporting
inline const char* prof_cat_name(prof_cat c) {
  switch (c) {
    case prof_cat::f_eval:          return "f_eval";
    case prof_cat::jac_eval:        return "jac_eval";
    case prof_cat::lu_factor:       return "lu_factor";
    case prof_cat::lu_solve:        return "lu_solve";
    case prof_cat::nordsieck:       return "nordsieck";
    case prof_cat::dense_interp:    return "dense_interp";
    case prof_cat::dense_snapshot:  return "dense_snapshot";
    case prof_cat::observer:        return "observer";
    case prof_cat::error_norm:      return "error_norm";
    case prof_cat::newton_overhead: return "newton_overhead";
    case prof_cat::w_build:         return "w_build";
    case prof_cat::step_overhead:   return "step_overhead";
    default:                        return "unknown";
  }
}


// ============================================================================
//  Profiler implementation (active when CPPODE_PROFILE is defined)
// ============================================================================

#ifdef CPPODE_PROFILE

class profiler {
public:

  using clock = std::chrono::high_resolution_clock;
  using duration = std::chrono::duration<double, std::micro>;  // microseconds

  // RAII timer: stops and accumulates on destruction
  class scoped_timer {
  public:
    scoped_timer(profiler& p, prof_cat c)
      : m_prof(p), m_cat(c), m_start(clock::now()) {}
    ~scoped_timer() {
      auto elapsed = std::chrono::duration_cast<duration>(clock::now() - m_start);
      unsigned idx = static_cast<unsigned>(m_cat);
      m_prof.m_cumulative[idx] += elapsed.count();
      m_prof.m_counts[idx]++;
#ifdef CPPODE_PROFILE_VERBOSE
      std::fprintf(stderr, "[PROF] %-16s  %8.1f us\n",
                   prof_cat_name(m_cat), elapsed.count());
#endif
    }
    // non-copyable, non-movable
    scoped_timer(const scoped_timer&) = delete;
    scoped_timer& operator=(const scoped_timer&) = delete;
  private:
    profiler&  m_prof;
    prof_cat   m_cat;
    clock::time_point m_start;
  };

  profiler() {
    m_cumulative.fill(0.0);
    m_counts.fill(0);
  }

  // Start a scoped timer for a category
  [[nodiscard]] scoped_timer timer(prof_cat c) { return scoped_timer(*this, c); }

  // Manual start/stop (for cases where RAII is awkward)
  void start(prof_cat c) { m_manual_start[static_cast<unsigned>(c)] = clock::now(); }
  void stop(prof_cat c) {
    unsigned idx = static_cast<unsigned>(c);
    auto elapsed = std::chrono::duration_cast<duration>(
        clock::now() - m_manual_start[idx]);
    m_cumulative[idx] += elapsed.count();
    m_counts[idx]++;
  }

  // Merge another profiler's data into this one (e.g. from inner stepper)
  void merge(const profiler& other) {
    for (unsigned i = 0; i < PROF_N; ++i) {
      m_cumulative[i] += other.m_cumulative[i];
      m_counts[i] += other.m_counts[i];
    }
  }

  // Print summary to stderr
  void report(const char* label = "CppODE") const {
    double total = 0.0;
    for (unsigned i = 0; i < PROF_N; ++i) total += m_cumulative[i];
    if (total <= 0.0) return;

    std::fprintf(stderr,
      "\n==================  %s profiler  ==================\n", label);
    std::fprintf(stderr,
      "  %-18s %10s %8s %8s\n", "Category", "Time (ms)", "Calls", "Pct");
    std::fprintf(stderr,
      "  %-18s %10s %8s %8s\n", "------------------", "----------", "--------", "--------");

    for (unsigned i = 0; i < PROF_N; ++i) {
      if (m_counts[i] == 0) continue;
      double ms = m_cumulative[i] / 1000.0;
      double pct = 100.0 * m_cumulative[i] / total;
      std::fprintf(stderr, "  %-18s %10.3f %8d %7.1f%%\n",
                   prof_cat_name(static_cast<prof_cat>(i)),
                   ms, m_counts[i], pct);
    }
    std::fprintf(stderr,
      "  %-18s %10s %8s %8s\n", "------------------", "----------", "--------", "--------");
    std::fprintf(stderr,
      "  %-18s %10.3f\n", "TOTAL", total / 1000.0);
    std::fprintf(stderr,
      "=======================================================\n\n");
  }

  // Reset all counters
  void reset() {
    m_cumulative.fill(0.0);
    m_counts.fill(0);
  }

  // Accessors for R-side diagnostics
  double cumulative_us(prof_cat c) const { return m_cumulative[static_cast<unsigned>(c)]; }
  int    count(prof_cat c) const { return m_counts[static_cast<unsigned>(c)]; }
  double total_us() const {
    double s = 0; for (unsigned i = 0; i < PROF_N; ++i) s += m_cumulative[i]; return s;
  }

private:
  friend class scoped_timer;
  std::array<double, PROF_N>  m_cumulative;  // microseconds
  std::array<int, PROF_N>     m_counts;
  std::array<clock::time_point, PROF_N> m_manual_start;
};


#else // CPPODE_PROFILE not defined: zero-cost stubs

class profiler {
public:
  struct scoped_timer {
    // Empty: compiler eliminates entirely
  };

  scoped_timer timer(prof_cat) const { return {}; }

  void start(prof_cat) const {}
  void stop(prof_cat) const {}
  void merge(const profiler&) const {}
  void report(const char* = nullptr) const {}
  void reset() const {}

  double cumulative_us(prof_cat) const { return 0.0; }
  int    count(prof_cat) const { return 0; }
  double total_us() const { return 0.0; }
};

#endif // CPPODE_PROFILE

} // namespace cppode

#endif // CPPODE_PROFILER_HPP
