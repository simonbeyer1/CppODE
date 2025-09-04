#' Generate a generic Boost.Odeint observer (C++ code)
#'
#' This function generates a C++ `observer` struct for Boost.Odeint that
#' (1) records states at each observer callback and
#' (2) applies **fixed-time events** with a configurable time tolerance.
#'
#' Notes:
#' - Root (zero-crossing) events are **not** handled here, because an observer
#'   only sees the current (x, t). Robust root handling requires bracketing and
#'   refinement in the integration loop (sign change between successive states),
#'   which should be emitted separately.
#' - Event methods supported: "replace", "add".
#' - The observer is generic in the number of states; it pushes back all `x[i]`
#'   to a flat output vector `y` at each callback.
#'
#' @param states Character vector of state names (order defines x indices).
#' @param events Data frame of events with columns:
#'   \describe{
#'     \item{var}{State name the event targets (one of `states`).}
#'     \item{time}{Numeric time for fixed-time event (use `NA` for root events).}
#'     \item{value}{Numeric value (used with method).}
#'     \item{method}{Character: "replace" or "add". `NA` is treated as "replace".}
#'     \item{root}{Optional character expression for root event (ignored here).}
#'   }
#'   Only rows with non-`NA` `time` are embedded in the observer. Root events
#'   are intentionally skipped (they require a custom integration loop).
#' @param time_tol Numeric tolerance for matching fixed event times. If `NULL`,
#'   exact comparison (`==`) is used. Recommended: a small value like `1e-10`.
#'
#' @return A single character string with the full C++ code for:
#'   - an `apply_event` helper,
#'   - an `observer` struct that records outputs and applies fixed-time events.
#'   The code uses `CppAD::AD<double>` as `AD` and `boost::numeric::ublas::vector<AD>` as state.
#' @examples
#' ev <- data.frame(
#'   var    = c("A","B","A"),
#'   time   = c(1, 2, 5),
#'   value  = c(0.5, 1.0, 2.0),
#'   method = c("replace", "add", NA),
#'   root   = c(NA, NA, NA),
#'   stringsAsFactors = FALSE
#' )
#' code <- GetBoostObserver(states = c("A","B"), events = ev, time_tol = 1e-10)
#' cat(code)
#' @export
GetBoostObserver <- function(states, events, time_tol = NULL) {
  stopifnot(is.character(states), length(states) >= 1L)

  # Validate events
  req_cols <- c("var", "time", "value", "method", "root")
  if (!all(req_cols %in% names(events))) {
    stop("`events` must have columns: var, time, value, method, root")
  }
  if (any(!events$var %in% states)) {
    stop("All events$var must be one of the given `states`.")
  }
  # Default method: replace
  events$method[is.na(events$method)] <- "replace"
  # Accept only replace/add here
  if (any(!tolower(events$method) %in% c("replace","add"))) {
    stop("Only event methods 'replace' and 'add' are supported in this observer.")
  }

  # Use only fixed-time events in the observer
  fixed <- events[!is.na(events$time), , drop = FALSE]

  # Map state name -> index
  idx_of <- function(name, vec) match(name, vec) - 1L

  # Build the fixed-event checks inside operator()(x, t)
  if (nrow(fixed) == 0L) {
    ev_checks <- "    // no fixed-time events"
  } else {
    # Time tolerance condition string
    if (is.null(time_tol)) {
      time_cond <- function(tt) sprintf("curr_t == %s", tt)
    } else {
      tol <- sprintf("%.17g", time_tol)
      time_cond <- function(tt) sprintf("std::fabs(curr_t - %s) <= %s", tt, tol)
    }

    # One counter per event to enforce 'fire once'
    checks <- character(nrow(fixed))
    for (i in seq_len(nrow(fixed))) {
      vi   <- idx_of(fixed$var[i], states)
      tt   <- sprintf("%.17g", fixed$time[i])
      val  <- sprintf("%.17g", fixed$value[i])
      meth <- fixed$method[i]
      checks[i] <- sprintf(
        "    if (%s && evt[%d] < 1) { apply_event(x, %d, \"%s\", %s); evt[%d]++; }",
        time_cond(tt), i - 1L, vi, meth, val, i - 1L
      )
    }
    ev_checks <- paste(checks, collapse = "\n")
  }

  # Number of fixed events (for evt counters)
  n_fixed <- nrow(fixed)

  # Emit C++ helper and observer
  cpp <- sprintf('
/* -------------------------------------------
 * Helper to apply an event to a state vector
 * ------------------------------------------- */
inline void apply_event(vector<AD>& x, int var_index, const std::string& method, double value) {
  if (method == "replace") {
    x[var_index] = AD(value);
  } else if (method == "add") {
    x[var_index] = x[var_index] + AD(value);
  } else {
    // Unknown method: default to replace
    x[var_index] = AD(value);
  }
}

/* -----------------------------------------------------------
 * Generic observer for Boost.Odeint with CppAD state type.
 *
 * Responsibilities:
 *  - Push current time and state to output containers.
 *  - Fire fixed-time events when the current time matches
 *    the event time (within a user-specified tolerance).
 *
 * Notes:
 *  - Root (zero-crossing) events are NOT handled here.
 *    They must be detected and refined in the integration
 *    loop (bracketing + bisection) before applying events.
 * ----------------------------------------------------------- */
struct observer {
  std::vector<AD>& times;  // recorded times (AD for compatibility with the tape)
  std::vector<AD>& y;      // flattened state outputs: x[0], ..., x[n-1] per row
  std::vector<int>& evt;   // one-shot counters for fixed-time events
  explicit observer(std::vector<AD>& t, std::vector<AD>& y_, std::vector<int>& ec)
    : times(t), y(y_), evt(ec) {}

  void operator()(vector<AD>& x, const AD& t) {
    const double curr_t = CppAD::Value(t);

    // Fire fixed-time events (if any)
%s

    // Record time and states
    times.push_back(t);
    for (size_t i = 0; i < x.size(); ++i) {
      y.push_back(x[i]);
    }
  }
};

// Number of fixed-time events in this observer: %d
', ev_checks, n_fixed)

  cpp
}
