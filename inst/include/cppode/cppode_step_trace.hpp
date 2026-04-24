/*
 Step-trace support for CppODE.

 When compiled with -DCPPODE_STEP_TRACE, the multistepper and the one-step
 controller append one row per step attempt into an in-memory buffer.  The
 generated `solve_<model>()` entry point marshals this buffer into an R list
 (element `$trace` of the result), and `solveODE()` either attaches it as a
 data.frame or writes it to a user-supplied CSV path.  No compile-time
 filename is required, and no stdio hardcoding leaks into the .so.

 Schema (kept in sync with the CVODE backend's trace in codegenCVODE.py):

   nst          int     accepted steps so far (pre-this-attempt)
   t            double  time at the end of the attempted step
   h            double  signed step size
   q            int     current method order
   dsm          double  acnrm * tq[2] (error-test value; < 1 => accept)
   acnrm        double  WRMS over all error-control components (state +
                        AD-derivative slots when sens is active)
   acnrm_state  double  WRMS restricted to state slots — directly comparable
                        to CVODE's state-only acnrm
   tq2          double  current-order error constant
   gamma        double  h * rl1 (Newton iteration matrix scaling)
   gamrat       double  gamma / gammap
   newton_conv  int     1 if Newton converged this attempt
   mode         string  "BDF", "NDF", "ADAMS", "ONESTEP"
   nfe,njev     int     cumulative counters (function / Jacobian evals)
   nsetups      int     cumulative LU setups
   setup_reason string  trigger for the most recent setup
   pece_iters   int     Adams PECE iteration count (0 for BDF)
   pece_diverged int    1 if PECE hit divergence watchdog

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_STEP_TRACE_HPP
#define CPPODE_STEP_TRACE_HPP

#include <string>
#include <vector>

namespace cppode {
namespace ndf_detail {

struct TraceBuffer {
  std::vector<int>         nst;
  std::vector<double>      t;
  std::vector<double>      h;
  std::vector<int>         q;
  std::vector<double>      dsm;
  std::vector<double>      acnrm;
  std::vector<double>      acnrm_state;
  std::vector<double>      tq2;
  std::vector<double>      gamma;
  std::vector<double>      gamrat;
  std::vector<int>         newton_conv;
  std::vector<std::string> mode;
  std::vector<int>         nfe;
  std::vector<int>         njev;
  std::vector<int>         nsetups;
  std::vector<std::string> setup_reason;
  std::vector<int>         pece_iters;
  std::vector<int>         pece_diverged;

  std::size_t size() const { return nst.size(); }

  void clear() {
    nst.clear();          t.clear();            h.clear();
    q.clear();            dsm.clear();          acnrm.clear();
    acnrm_state.clear();  tq2.clear();          gamma.clear();
    gamrat.clear();       newton_conv.clear();  mode.clear();
    nfe.clear();          njev.clear();         nsetups.clear();
    setup_reason.clear(); pece_iters.clear();   pece_diverged.clear();
  }
};

// Lazy per-process (per-SO) handle to the trace buffer.  `inline` function
// + function-local static guarantees a single buffer per loaded shared
// object, so two different compiled models in the same R session do not
// share or stomp on each other's trace data.
inline TraceBuffer& get_trace_buffer() {
  static TraceBuffer buf;
  return buf;
}

} // namespace ndf_detail
} // namespace cppode

#endif // CPPODE_STEP_TRACE_HPP
