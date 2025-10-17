#' Generate C++ code for ODE models with events and optional sensitivities
#'
#' This function generates and compiles a C++ solver for systems of ODEs using
#' Boost.Odeint's stiff Rosenbrock 4(3) method with dense output and error control.
#' The solver can handle fixed-time and root-triggered events, and (optionally)
#' compute state and parameter sensitivities via forward-mode automatic
#' differentiation (AD) using FADBAD++.
#'
#' @section Events:
#' Events are specified in a \code{data.frame} with columns:
#' \describe{
#'   \item{var}{Name of the affected state variable.}
#'   \item{value}{Numeric value to apply at the event.}
#'   \item{method}{How the value is applied: "replace", "add", or "multiply".}
#'   \item{time}{(optional) Time point at which the event occurs.}
#'   \item{root}{(optional) Root expression in terms of states and \code{time}.}
#' }
#' Each event must define either \code{time} or \code{root}. Events with roots
#' are triggered whenever the expression crosses zero.
#'
#' @section Sensitivities:
#' If \code{deriv = TRUE}, the solver augments the system with automatic
#' differentiation and returns forward sensitivities with respect to initial
#' conditions and parameters. If \code{deriv2 = TRUE} (requires \code{deriv = TRUE}),
#' second-order sensitivities are also computed using nested AD types.
#' Fixed states or parameters (specified in \code{fixed}) are excluded from
#' the sensitivity system.
#'
#' If \code{deriv = FALSE}, the solver uses plain doubles (faster) and does not
#' compute sensitivities.
#'
#' @section Output:
#' The generated solver function (accessible via \code{.Call}) returns a named list
#' with the following structure:
#' \describe{
#'   \item{deriv = FALSE}{
#'     Returns \code{list(time, state)} where:
#'     \itemize{
#'       \item \code{time}: Numeric vector of length n_out
#'       \item \code{state}: Numeric matrix with dimensions (n_out, n_states)
#'     }
#'   }
#'   \item{deriv = TRUE, deriv2 = FALSE}{
#'     Returns \code{list(time, state, sens1)} where:
#'     \itemize{
#'       \item \code{time}: Numeric vector of length n_out
#'       \item \code{state}: Numeric matrix with dimensions (n_out, n_states)
#'       \item \code{sens1}: Numeric array with dimensions (n_out, n_states, n_sens)
#'         containing first-order sensitivities
#'     }
#'   }
#'   \item{deriv = TRUE, deriv2 = TRUE}{
#'     Returns \code{list(time, state, sens1, sens2)} where:
#'     \itemize{
#'       \item \code{time}: Numeric vector of length n_out
#'       \item \code{state}: Numeric matrix with dimensions (n_out, n_states)
#'       \item \code{sens1}: Numeric array with dimensions (n_out, n_states, n_sens)
#'         containing first-order sensitivities
#'       \item \code{sens2}: Numeric array with dimensions (n_out, n_states, n_sens, n_sens)
#'         containing second-order sensitivities (Hessian matrix)
#'     }
#'   }
#' }
#' Here \code{n_out} is the number of output time points, \code{n_states} is the number
#' of state variables, and \code{n_sens} is the number of sensitivity parameters
#' (non-fixed states and parameters).
#'
#' @param odes Named character vector of ODE right-hand sides.
#'   Names must correspond to state variables.
#' @param events Optional \code{data.frame} describing events (see Details).
#'   Default: \code{NULL} (no events).
#' @param fixed Character vector of fixed states or parameters (excluded from
#'   sensitivity system). Only relevant if \code{deriv = TRUE}.
#' @param includeTimeZero Logical. If \code{TRUE}, ensure that time \code{0} is
#'   included among integration times. Default: \code{TRUE}.
#' @param compile Logical. If \code{TRUE}, compiles and loads the generated C++ code.
#' @param modelname Optional base name for the generated C++ file. If \code{NULL},
#'   a random identifier is used.
#' @param deriv Logical. If \code{TRUE}, compute first-order sensitivities using AD.
#'   If \code{FALSE}, use plain doubles.
#' @param deriv2 Logical. If \code{TRUE}, compute second-order sensitivities using
#'   nested AD. Requires \code{deriv = TRUE}. Default: \code{FALSE}.
#' @param verbose Logical. If \code{TRUE}, print progress messages.
#'
#' @return The model name (character). The object has attributes:
#'   \itemize{
#'     \item \code{equations}: ODE definitions
#'     \item \code{variables}: State variable names
#'     \item \code{parameters}: Parameter names
#'     \item \code{events}: Events \code{data.frame} (if any)
#'     \item \code{solver}: Solver description
#'     \item \code{fixed}: Fixed states/parameters
#'     \item \code{jacobian}: Symbolic Jacobian expressions
#'     \item \code{deriv}: Logical indicating first-order derivatives
#'     \item \code{deriv2}: Logical indicating second-order derivatives
#'     \item \code{dim_names}: List with dimension names (time, state, sens)
#'   }
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @example inst/examples/example.R
#' @importFrom reticulate import source_python
#' @export
CppFun <- function(odes, events = NULL, fixed = NULL, includeTimeZero = TRUE,
                   compile = TRUE, modelname = NULL,
                   deriv = TRUE, deriv2 = FALSE, verbose = FALSE) {

  # --- Validate arguments ---
  if (deriv2 && !deriv) {
    stop("deriv2 = TRUE requires deriv = TRUE")
  }

  # --- Clean up ODE definitions ---
  odes <- unclass(odes)
  odes <- gsub("\n", "", odes)
  odes <- sanitizeExprs(odes)

  # --- Extract state and parameter names ---
  states  <- names(odes)
  symbols <- getSymbols(c(odes, if (!is.null(events)) {
    c(events$value,
      if ("time" %in% names(events)) events$time,
      if ("root" %in% names(events)) events$root)
  }))
  params  <- setdiff(symbols, c(states, "time"))

  # --- Handle fixed states/parameters ---
  if (is.null(fixed)) fixed <- character(0)
  fixed_states <- if (deriv) intersect(fixed, states) else character(0)
  fixed_params <- if (deriv) intersect(fixed, params) else character(0)
  sens_states  <- if (deriv) setdiff(states, fixed_states) else character(0)
  sens_params  <- if (deriv) setdiff(params, fixed_params) else character(0)

  # Index maps
  state_idx0 <- setNames(seq_along(states) - 1L, states)
  param_idx0 <- setNames(seq_along(params) - 1L, params)
  fixed_state_idx  <- state_idx0[fixed_states]
  fixed_param_idx  <- param_idx0[fixed_params]

  # --- Calculate dimensions ---
  n_states <- length(states)
  n_params <- length(params)
  n_sens_initials <- length(sens_states)
  n_sens_params <- length(sens_params)
  n_total_sens <- n_sens_initials + n_sens_params

  # --- Generate unique model name ---
  if (is.null(modelname)) {
    modelname <- paste(c("f", sample(c(letters, 0:9), 8, TRUE)), collapse = "")
  }

  # --- CALL PYTHON CODE GENERATOR (single call!) ---
  ensurePythonEnv("CppODE", verbose)

  # Source Python module
  py_file <- system.file("python", "codegen.py", package = "CppODE")
  reticulate::source_python(py_file)

  if (verbose) message("Generating ODE and Jacobian code...")

  # Single Python call generates everything
  if (deriv2) {
    numType <- "AD2"  # F<F<double>>
  } else if (deriv) {
    numType <- "AD"   # F<double>
  } else {
    numType <- "double"
  }

  codegen_result <- reticulate::py$generate_ode_cpp(
    odes_dict = as.list(setNames(odes, states)),
    params_list = params,
    num_type = numType,
    fixed_states = fixed_states,
    fixed_params = fixed_params
  )

  ode_code <- codegen_result$ode_code
  jac_code <- codegen_result$jac_code
  jac_matrix_str <- codegen_result$jac_matrix
  time_derivs_str <- codegen_result$time_derivs

  if (verbose) message("  ✓ ODE and Jacobian generated")

  # --- Generate event code if needed ---
  event_code <- ""
  if (!is.null(events)) {
    if (verbose) message("Generating event code...")

    event_lines <- reticulate::py$generate_event_code(
      events_df = events,
      states_list = states,
      params_list = params,
      n_states = n_states,
      num_type = numType
    )

    event_code <- paste(event_lines, collapse = "\n")
  }

  # --- C++ includes ---
  includings <- c(
    "#define R_NO_REMAP",
    "#include <R.h>",
    "#include <Rinternals.h>",
    "#include <algorithm>",
    "#include <vector>",
    "#include <cmath>",
    "#include <boost_rosenbrock34_fad.hpp>"
  )

  # --- Using declarations ---
  if (deriv2) {
    usings <- c(
      "using namespace boost::numeric::odeint;",
      "namespace ublas = boost::numeric::ublas;",
      "using AD = fadbad::F<double>;",
      "using AD2 = fadbad::F<fadbad::F<double>>;"
    )
  } else if (deriv) {
    usings <- c(
      "using namespace boost::numeric::odeint;",
      "namespace ublas = boost::numeric::ublas;",
      "using AD = fadbad::F<double>;"
    )
  } else {
    usings <- c(
      "using namespace boost::numeric::odeint;",
      "namespace ublas = boost::numeric::ublas;"
    )
  }

  # --- Observer ---
  observer_lines <- c(
    "// Observer: stores trajectory values in vectors",
    "struct observer {",
    sprintf("  std::vector<%s>& times;", numType),
    sprintf("  std::vector<%s>& y;", numType),
    "",
    sprintf("  explicit observer(std::vector<%s>& t, std::vector<%s>& y_)", numType, numType),
    "    : times(t), y(y_) {}",
    "",
    sprintf("  void operator()(const ublas::vector<%s>& x, const %s& t) {", numType, numType),
    "    times.push_back(t);",
    "    for (size_t i = 0; i < x.size(); ++i) y.push_back(x[i]);",
    "  }",
    "};"
  )
  observer_code <- paste(observer_lines, collapse = "\n")

  # --- Solver function (externC) - ALL IN ONE PLACE ---
  externC <- c(
    sprintf('extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP, SEXP root_tolSEXP, SEXP maxrootSEXP) {', modelname),
    "try {",
    "",
    "  StepChecker checker(INTEGER(maxprogressSEXP)[0], INTEGER(maxstepsSEXP)[0]);",
    "",
    sprintf("  ublas::vector<%s> x(%d);", numType, n_states),
    sprintf("  ublas::vector<%s> full_params(%d);", numType, n_states + n_params),
    ""
  )

  # initialization of states and parameters
  externC <- c(externC,
               "  // initialize states",
               sprintf("  for (int i = 0; i < %d; ++i) {", n_states))

  if (deriv2) {
    externC <- c(externC, "    x[i].x() = REAL(paramsSEXP)[i];")
    if (length(fixed_state_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) {",
                           paste(sprintf("i == %d", fixed_state_idx), collapse = " || ")),
                   sprintf("      x[i].diff(i, %d);", n_states + n_params),
                   sprintf("      for (int j = 0; j < %d; ++j) {", n_states + n_params),
                   sprintf("        if (!(%s)) x[i].d(j).diff(j, %d);",
                           paste(c(sprintf("j == %d", fixed_state_idx),
                                   sprintf("j == %d", n_states + fixed_param_idx)), collapse = " || "),
                           n_states + n_params),
                   "      }",
                   "    }")
    } else {
      externC <- c(externC,
                   sprintf("    x[i].diff(i, %d);", n_states + n_params),
                   sprintf("    for (int j = 0; j < %d; ++j) {", n_states + n_params),
                   sprintf("      x[i].d(j).diff(j, %d);", n_states + n_params),
                   "    }")
    }
  } else if (deriv) {
    externC <- c(externC, "    x[i] = REAL(paramsSEXP)[i];")
    if (length(fixed_state_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) x[i].diff(i, %d);",
                           paste(sprintf("i == %d", fixed_state_idx), collapse = " || "),
                           n_states + n_params))
    } else {
      externC <- c(externC, sprintf("    x[i].diff(i, %d);", n_states + n_params))
    }
  } else {
    externC <- c(externC, "    x[i] = REAL(paramsSEXP)[i];")
  }

  externC <- c(externC,
               "    full_params[i] = x[i];",
               "  }",
               "",
               "  // initialize parameters",
               sprintf("  for (int i = 0; i < %d; ++i) {", n_params))

  if (deriv2) {
    externC <- c(externC,
                 sprintf("    full_params[%d + i].x() = REAL(paramsSEXP)[%d + i];", n_states, n_states))
    if (length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) {",
                           paste(sprintf("i == %d", fixed_param_idx), collapse = " || ")),
                   sprintf("      full_params[%d + i].diff(%d + i, %d);", n_states, n_states, n_states + n_params),
                   sprintf("      for (int j = 0; j < %d; ++j) {", n_states + n_params),
                   sprintf("        if (!(%s)) full_params[%d + i].d(j).diff(j, %d);",
                           paste(c(sprintf("j == %d", fixed_state_idx),
                                   sprintf("j == %d", n_states + fixed_param_idx)), collapse = " || "),
                           n_states, n_states + n_params),
                   "      }",
                   "    }")
    } else {
      externC <- c(externC,
                   sprintf("    full_params[%d + i].diff(%d + i, %d);", n_states, n_states, n_states + n_params),
                   sprintf("    for (int j = 0; j < %d; ++j) {", n_states + n_params),
                   sprintf("      full_params[%d + i].d(j).diff(j, %d);", n_states, n_states + n_params),
                   "    }")
    }
  } else if (deriv) {
    externC <- c(externC,
                 sprintf("    full_params[%d + i] = REAL(paramsSEXP)[%d + i];", n_states, n_states))
    if (length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) full_params[%d + i].diff(%d + i, %d);",
                           paste(sprintf("i == %d", fixed_param_idx), collapse = " || "),
                           n_states, n_states, n_states + n_params))
    } else {
      externC <- c(externC, sprintf("    full_params[%d + i].diff(%d + i, %d);", n_states, n_states, n_states + n_params))
    }
  } else {
    externC <- c(externC,
                 sprintf("    full_params[%d + i] = REAL(paramsSEXP)[%d + i];", n_states, n_states))
  }

  externC <- c(externC,
               "  }",
               "",
               "  // --- Copy integration times ---",
               "  std::vector<double> times_dbl(REAL(timesSEXP), REAL(timesSEXP) + Rf_length(timesSEXP));",
               ""
  )

  if (includeTimeZero) {
    externC <- c(externC,
                 "  // ensure time zero is included",
                 "  if (std::find(times_dbl.begin(), times_dbl.end(), 0.0) == times_dbl.end()) {",
                 "    times_dbl.push_back(0.0);",
                 "  }",
                 ""
    )
  }

  externC <- c(externC,
               "  // sort times ascending and remove duplicates",
               "  std::sort(times_dbl.begin(), times_dbl.end());",
               "  times_dbl.erase(std::unique(times_dbl.begin(), times_dbl.end()), times_dbl.end());",
               "",
               "  // convert to AD vector",
               sprintf("  std::vector<%s> times;", numType),
               "  times.reserve(times_dbl.size());",
               "  for (double tval : times_dbl) {",
               "    times.emplace_back(tval);",
               "  }",
               "",
               "  // storage for results",
               sprintf("  std::vector<%s> result_times;", numType),
               sprintf("  std::vector<%s> y;", numType),
               "",
               "  // --- Event containers ---",
               sprintf("  std::vector<FixedEvent<%s>> fixed_events;", numType),
               sprintf("  std::vector<RootEvent<ublas::vector<%s>, %s>> root_events;", numType, numType)
  )

  # Insert event code from Python
  if (event_code != "") {
    externC <- c(externC, event_code)
  }

  # --- Integration setup ---
  if (deriv2) {
    stepper_line <- paste(
      "  auto controlledStepper = rosenbrock4_controller_ad<rosenbrock4<AD2>>(abstol, reltol);",
      "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);",
      sep = "\n"
    )
  } else if (deriv) {
    stepper_line <- paste(
      "  auto controlledStepper = rosenbrock4_controller_ad<rosenbrock4<AD>>(abstol, reltol);",
      "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);",
      sep = "\n"
    )
  } else {
    stepper_line <- paste(
      "  auto controlledStepper = rosenbrock4_controller<rosenbrock4<double>>(abstol, reltol);",
      "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);",
      sep = "\n"
    )
  }

  externC <- c(externC, "",
               "  // --- Solver setup ---",
               "  double abstol = REAL(abstolSEXP)[0];",
               "  double reltol = REAL(reltolSEXP)[0];",
               "  double root_tol = REAL(root_tolSEXP)[0];",
               "  int maxroot = INTEGER(maxrootSEXP)[0];",
               "  ode_system sys(full_params);",
               "  jacobian jac(full_params);",
               "  observer obs(result_times, y);",
               stepper_line, "",
               "  // --- Determine dt ---",
               "  auto t_test = times.front();",
               "  auto x_test = x;",
               sprintf("  %s dt0 = odeint_utils::estimate_initial_dt(sys, jac, x_test, t_test, times.back(), abstol, reltol);", numType),
               "  auto dt = dt0;",
               "  int attempts = 0;",
               "  while (controlledStepper.try_step(std::make_pair(sys, jac), x_test, t_test, dt) == fail) {",
               "    if (++attempts >= 10000) throw std::runtime_error(\"Unable to find valid initial stepsize after 10000 attempts\");",
               "  }", "",
               "  // --- Integration ---",
               "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);",
               "",
               "  const int n_out = static_cast<int>(result_times.size());",
               "  if (n_out <= 0) Rf_error(\"Integration produced no output\");",
               ""
  )

  # --- Copy back results - CONSISTENT LIST OUTPUT ---
  if (!deriv) {
    # deriv = FALSE: list(time, state)
    externC <- c(externC,
                 "  // --- Return list(time, state) ---",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 2));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"state\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP state_mat = PROTECT(Rf_allocMatrix(REALSXP, n_out, %d));", n_states),
                 "  double* time_out = REAL(time_vec);",
                 "  double* state_out = REAL(state_mat);",
                 "  auto IDX = [n_out](int r, int c){ return r + c * n_out; };",
                 "",
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i];",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_states),
                 sprintf("      state_out[IDX(i, s)] = y[i * %d + s];", n_states),
                 "    }",
                 "  }",
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, state_mat);",
                 "  UNPROTECT(4);",
                 "  return ans;")

  } else if (!deriv2) {
    # deriv = TRUE, deriv2 = FALSE: list(time, state, sens1)
    externC <- c(externC,
                 "  // --- Return list(time, state, sens1) ---",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 3));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"state\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"sens1\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP state_mat = PROTECT(Rf_allocMatrix(REALSXP, n_out, %d));", n_states),
                 "  SEXP sens1_dim = PROTECT(Rf_allocVector(INTSXP, 3));",
                 "  INTEGER(sens1_dim)[0] = n_out;",
                 sprintf("  INTEGER(sens1_dim)[1] = %d;", n_states),
                 sprintf("  INTEGER(sens1_dim)[2] = %d;", n_total_sens),
                 "  SEXP sens1_arr = PROTECT(Rf_allocArray(REALSXP, sens1_dim));",
                 "",
                 "  double* time_out = REAL(time_vec);",
                 "  double* state_out = REAL(state_mat);",
                 "  double* sens1_out = REAL(sens1_arr);",
                 "",
                 "  auto IDX_state = [n_out](int r, int c){ return r + c * n_out; };",
                 sprintf("  auto IDX_sens1 = [n_out](int t, int s, int v){ return t + n_out * (s + %d * v); };", n_states),
                 ""
    )

    # Extract with fixed parameter handling
    if (length(fixed_state_idx) > 0 || length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   "  for (int i = 0; i < n_out; ++i) {",
                   "    time_out[i] = result_times[i].x();",
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_states),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_states),
                   "      state_out[IDX_state(i, s)] = xi.x();",
                   "      int v_sens = 0;",
                   sprintf("      for (int v = 0; v < %d; ++v) {", n_states + n_params))

      fixed_checks <- character(0)
      if (length(fixed_state_idx) > 0) {
        fixed_checks <- c(fixed_checks,
                          sprintf("        bool is_fixed_init = (v < %d) && (%s);",
                                  n_states,
                                  paste(sprintf("v == %d", fixed_state_idx), collapse = " || ")))
      } else {
        fixed_checks <- c(fixed_checks, "        bool is_fixed_init = false;")
      }

      if (length(fixed_param_idx) > 0) {
        fixed_checks <- c(fixed_checks,
                          sprintf("        bool is_fixed_param = (v >= %d) && (%s);",
                                  n_states,
                                  paste(sprintf("(v - %d) == %d", n_states, fixed_param_idx), collapse = " || ")))
      } else {
        fixed_checks <- c(fixed_checks, "        bool is_fixed_param = false;")
      }

      externC <- c(externC, fixed_checks,
                   "        if (!(is_fixed_init || is_fixed_param)) {",
                   "          sens1_out[IDX_sens1(i, s, v_sens)] = xi.d(v);",
                   "          v_sens++;",
                   "        }",
                   "      }",
                   "    }",
                   "  }")
    } else {
      externC <- c(externC,
                   "  for (int i = 0; i < n_out; ++i) {",
                   "    time_out[i] = result_times[i].x();",
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_states),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_states),
                   "      state_out[IDX_state(i, s)] = xi.x();",
                   sprintf("      for (int v = 0; v < %d; ++v) {", n_total_sens),
                   "        sens1_out[IDX_sens1(i, s, v)] = xi.d(v);",
                   "      }",
                   "    }",
                   "  }")
    }

    externC <- c(externC,
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, state_mat);",
                 "  SET_VECTOR_ELT(ans, 2, sens1_arr);",
                 "  UNPROTECT(6);",
                 "  return ans;")

  } else {
    # deriv2 = TRUE: list(time, state, sens1, sens2)
    externC <- c(externC,
                 "  // --- Copy results to R list for deriv2 ---",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 4));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"state\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"sens1\"));",
                 "  SET_STRING_ELT(names, 3, Rf_mkChar(\"sens2\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP state_mat = PROTECT(Rf_allocMatrix(REALSXP, n_out, %d));", n_states),
                 "  SEXP sens1_dim = PROTECT(Rf_allocVector(INTSXP, 3));",
                 "  INTEGER(sens1_dim)[0] = n_out;",
                 sprintf("  INTEGER(sens1_dim)[1] = %d;", n_states),
                 sprintf("  INTEGER(sens1_dim)[2] = %d;", n_total_sens),
                 "  SEXP sens1_arr = PROTECT(Rf_allocArray(REALSXP, sens1_dim));",
                 "  SEXP sens2_dim = PROTECT(Rf_allocVector(INTSXP, 4));",
                 "  INTEGER(sens2_dim)[0] = n_out;",
                 sprintf("  INTEGER(sens2_dim)[1] = %d;", n_states),
                 sprintf("  INTEGER(sens2_dim)[2] = %d;", n_total_sens),
                 sprintf("  INTEGER(sens2_dim)[3] = %d;", n_total_sens),
                 "  SEXP sens2_arr = PROTECT(Rf_allocArray(REALSXP, sens2_dim));",
                 "",
                 "  double* time_out = REAL(time_vec);",
                 "  double* state_out = REAL(state_mat);",
                 "  double* sens1_out = REAL(sens1_arr);",
                 "  double* sens2_out = REAL(sens2_arr);",
                 "",
                 "  auto IDX_state = [n_out](int r, int c){ return r + c * n_out; };",
                 sprintf("  auto IDX_sens1 = [n_out](int t, int s, int v){ return t + n_out * (s + %d * v); };", n_states),
                 sprintf("  auto IDX_sens2 = [n_out](int t, int s, int v1, int v2){ return t + n_out * (s + %d * (v1 + %d * v2)); };", n_states, n_total_sens),
                 ""
    )

    # Extract with fixed parameter handling
    if (length(fixed_state_idx) > 0 || length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   "  for (int i = 0; i < n_out; ++i) {",
                   "    time_out[i] = result_times[i].x().x();",
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_states),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_states),
                   "      state_out[IDX_state(i, s)] = xi.x().x();",
                   "      int v1_sens = 0;",
                   sprintf("      for (int v1 = 0; v1 < %d; ++v1) {", n_states + n_params))

      fixed_checks_outer <- character(0)
      if (length(fixed_state_idx) > 0) {
        fixed_checks_outer <- c(fixed_checks_outer,
                                sprintf("        bool is_fixed_init1 = (v1 < %d) && (%s);",
                                        n_states,
                                        paste(sprintf("v1 == %d", fixed_state_idx), collapse = " || ")))
      } else {
        fixed_checks_outer <- c(fixed_checks_outer, "        bool is_fixed_init1 = false;")
      }

      if (length(fixed_param_idx) > 0) {
        fixed_checks_outer <- c(fixed_checks_outer,
                                sprintf("        bool is_fixed_param1 = (v1 >= %d) && (%s);",
                                        n_states,
                                        paste(sprintf("(v1 - %d) == %d", n_states, fixed_param_idx), collapse = " || ")))
      } else {
        fixed_checks_outer <- c(fixed_checks_outer, "        bool is_fixed_param1 = false;")
      }

      externC <- c(externC, fixed_checks_outer,
                   "        if (!(is_fixed_init1 || is_fixed_param1)) {",
                   "          sens1_out[IDX_sens1(i, s, v1_sens)] = xi.d(v1).x();",
                   "          int v2_sens = 0;",
                   sprintf("          for (int v2 = 0; v2 < %d; ++v2) {", n_states + n_params))

      fixed_checks_inner <- character(0)
      if (length(fixed_state_idx) > 0) {
        fixed_checks_inner <- c(fixed_checks_inner,
                                sprintf("            bool is_fixed_init2 = (v2 < %d) && (%s);",
                                        n_states,
                                        paste(sprintf("v2 == %d", fixed_state_idx), collapse = " || ")))
      } else {
        fixed_checks_inner <- c(fixed_checks_inner, "            bool is_fixed_init2 = false;")
      }

      if (length(fixed_param_idx) > 0) {
        fixed_checks_inner <- c(fixed_checks_inner,
                                sprintf("            bool is_fixed_param2 = (v2 >= %d) && (%s);",
                                        n_states,
                                        paste(sprintf("(v2 - %d) == %d", n_states, fixed_param_idx), collapse = " || ")))
      } else {
        fixed_checks_inner <- c(fixed_checks_inner, "            bool is_fixed_param2 = false;")
      }

      externC <- c(externC, fixed_checks_inner,
                   "            if (!(is_fixed_init2 || is_fixed_param2)) {",
                   "              sens2_out[IDX_sens2(i, s, v1_sens, v2_sens)] = xi.d(v1).d(v2);",
                   "              v2_sens++;",
                   "            }",
                   "          }",
                   "          v1_sens++;",
                   "        }",
                   "      }",
                   "    }",
                   "  }")
    } else {
      externC <- c(externC,
                   "  for (int i = 0; i < n_out; ++i) {",
                   "    time_out[i] = result_times[i].x().x();",
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_states),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_states),
                   "      state_out[IDX_state(i, s)] = xi.x().x();",
                   sprintf("      for (int v1 = 0; v1 < %d; ++v1) {", n_total_sens),
                   "        sens1_out[IDX_sens1(i, s, v1)] = xi.d(v1).x();",
                   sprintf("        for (int v2 = 0; v2 < %d; ++v2) {", n_total_sens),
                   "          sens2_out[IDX_sens2(i, s, v1, v2)] = xi.d(v1).d(v2);",
                   "        }",
                   "      }",
                   "    }",
                   "  }")
    }

    externC <- c(externC,
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, state_mat);",
                 "  SET_VECTOR_ELT(ans, 2, sens1_arr);",
                 "  SET_VECTOR_ELT(ans, 3, sens2_arr);",
                 "  UNPROTECT(8);",
                 "  return ans;")
  }

  # --- End try/catch ---
  externC <- c(externC,
               "  } catch (const std::exception& e) {",
               "    Rf_error(\"ODE solver failed: %s\", e.what());",
               "  } catch (...) {",
               "    Rf_error(\"ODE solver failed: unknown C++ exception\");",
               "  }",
               "}")

  externC <- paste(externC, collapse = "\n")

  # --- Write C++ file ---
  filename <- paste0(modelname, ".cpp")
  sink(filename)
  cat("/** Code auto-generated by CppODE ",
      as.character(utils::packageVersion("CppODE")), " **/\n\n", sep = "")
  cat(paste(includings, collapse = "\n")); cat("\n\n")
  cat(paste(usings, collapse = "\n")); cat("\n\n")
  cat(ode_code, "\n\n")
  cat(jac_code, "\n\n")
  cat(observer_code, "\n\n")
  cat(externC)
  sink()

  if (verbose) message("Wrote: ", normalizePath(filename))
  if (compile) compileAndLoad(modelname, verbose)

  # --- Attach attributes ---
  jac_matrix_R <- matrix(unlist(jac_matrix_str), nrow = n_states, ncol = n_states, byrow = TRUE)
  dimnames(jac_matrix_R) <- list(states, states)

  attr(modelname, "equations")     <- odes
  attr(modelname, "variables")     <- states
  attr(modelname, "parameters")    <- params
  attr(modelname, "events")        <- events
  attr(modelname, "solver")        <- "boost::odeint::rosenbrock4"
  attr(modelname, "fixed")         <- c(fixed_states, fixed_params)
  attr(modelname, "jacobian")      <- list(f.x = jac_matrix_R, f.time = time_derivs_str)
  attr(modelname, "deriv")         <- deriv
  attr(modelname, "deriv2")        <- deriv2

  # Dimension names - ALWAYS when deriv = TRUE
  if (deriv2) {
    attr(modelname, "dim_names") <- list(
      time = "time",
      state = states,
      sens = c(sens_states, sens_params),
      sens = c(sens_states, sens_params, sens_params)
    )
  } else if (deriv) {
    attr(modelname, "dim_names") <- list(
      time = "time",
      state = states,
      sens = c(sens_states, sens_params)
    )
  } else {
    attr(modelname, "dim_names") <- list(
      time = "time",
      state = states
    )
  }

  return(modelname)
}



#' Compile and load a C++ source file in R (platform-aware)
#'
#' This function compiles a C++ source file into a shared library (`.so`, `.dll`, or `.dylib`)
#' using `R CMD SHLIB`, applies platform-appropriate compiler flags, and dynamically loads
#' the resulting shared object into the current R session.
#'
#' @param filename Character string. The base name of the C++ file (without `.cpp` extension).
#' @param verbose Logical. If `TRUE`, compiler output is printed. Default is `FALSE`.
#'
#' @return Invisibly returns the name of the loaded shared object file.
#' @export
compileAndLoad <- function(filename, verbose = FALSE) {
  filename_cpp <- paste0(filename, ".cpp")
  is_windows <- .Platform$OS.type == "windows"

  cxxflags <- if (is_windows) "-std=c++20 -O2 -DNDEBUG" else "-std=c++20 -O2 -DNDEBUG -fPIC"

  include_flags <- paste0("-I", shQuote(system.file("include", package = "CppODE")))

  Sys.setenv(
    PKG_CPPFLAGS = paste(include_flags, collapse = " "),
    PKG_CXXFLAGS = cxxflags
  )

  shlibOut <- system2(
    file.path(R.home("bin"), "R"),
    args = c("CMD", "SHLIB", "--preclean", shQuote(filename_cpp)),
    stdout = TRUE, stderr = TRUE
  )

  if (verbose) {
    cat(paste(shlibOut, collapse = "\n"), "\n")
  } else if (length(shlibOut)) {
    cat(paste(shlibOut[1], "\n"))
  }

  soFile <- paste0(filename, .Platform$dynlib.ext)
  if (file.exists(soFile)) {
    try(dyn.unload(soFile), silent = TRUE)
    dyn.load(soFile)
    invisible(soFile)
  } else {
    stop("Compiled shared library not found: ", soFile)
  }
}
