#' Generate C++ code for ODE models with events and optional sensitivities
#'
#' This function generates and compiles a C++ solver for systems of ODEs using
#' Boost.Odeint's stiff Rosenbrock 4(3) method with dense output and error control.
#' The solver can handle fixed-time and root-triggered events, and (optionally)
#' compute parameter sensitivities via forward-mode automatic
#' differentiation (AD) using FADBAD++.
#'
#' @section Events:
#' Events are specified in a \code{data.frame} with columns:
#' \describe{
#'   \item{var}{Name of the affected variable.}
#'   \item{value}{Numeric value to apply at the event.}
#'   \item{method}{How the value is applied: "replace", "add", or "multiply".}
#'   \item{time}{(optional) Time point at which the event occurs.}
#'   \item{root}{(optional) Root expression in terms of variables and \code{time}.}
#' }
#' Each event must define either \code{time} or \code{root}. Events with roots
#' are triggered whenever the expression crosses zero.
#'
#' @section Sensitivities:
#' If \code{deriv = TRUE}, the solver augments the system with automatic
#' differentiation and returns forward sensitivities with respect to initial
#' conditions and parameters. If \code{deriv2 = TRUE} (requires \code{deriv = TRUE}),
#' second-order sensitivities are also computed using nested AD types.
#' Fixed initial conditions or parameters (specified in \code{fixed}) are excluded from
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
#'     Returns \code{list(time, variable)} where:
#'     \itemize{
#'       \item \code{time}: Numeric vector of length n_out
#'       \item \code{variable}: Numeric matrix with dimensions (n_out, n_variables)
#'     }
#'   }
#'   \item{deriv = TRUE, deriv2 = FALSE}{
#'     Returns \code{list(time, variable, sens1)} where:
#'     \itemize{
#'       \item \code{time}: Numeric vector of length n_out
#'       \item \code{variable}: Numeric matrix with dimensions (n_out, n_variables)
#'       \item \code{sens1}: Numeric array with dimensions (n_out, n_variables, n_sens)
#'         containing first-order sensitivities
#'     }
#'   }
#'   \item{deriv = TRUE, deriv2 = TRUE}{
#'     Returns \code{list(time, variable, sens1, sens2)} where:
#'     \itemize{
#'       \item \code{time}: Numeric vector of length n_out
#'       \item \code{variable}: Numeric matrix with dimensions (n_out, n_variables)
#'       \item \code{sens1}: Numeric array with dimensions (n_out, n_variables, n_sens)
#'         containing first-order sensitivities
#'       \item \code{sens2}: Numeric array with dimensions (n_out, n_variables, n_sens, n_sens)
#'         containing second-order sensitivities (Hessian matrix)
#'     }
#'   }
#' }
#' Here \code{n_out} is the number of output time points, \code{n_variables} is the number
#' of variables, and \code{n_sens} is the number of sensitivity parameters
#' (non-fixed initial conditions and parameters).
#'
#' @param rhs Named character vector of ODE right-hand sides.
#'   Names must correspond to variables.
#' @param events Optional \code{data.frame} describing events (see Events section below).
#'   Default: \code{NULL} (no events).
#' @param fixed Character vector of fixed initial conditions or parameters (excluded from
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
#' @param useDenseOutput Logical. If \code{TRUE}, use dense output (interpolation with hermite polynomials).
#' @param verbose Logical. If \code{TRUE}, print progress messages.
#'
#' @return The model name (character). The object has attributes:
#'   \itemize{
#'     \item \code{equations}: ODE definitions
#'     \item \code{variables}: Variable names
#'     \item \code{parameters}: Parameter names
#'     \item \code{events}: Events \code{data.frame} (if any)
#'     \item \code{solver}: Solver description
#'     \item \code{fixed}: Fixed initial conditions/parameters
#'     \item \code{jacobian}: Symbolic Jacobian expressions
#'     \item \code{deriv}: Logical indicating first-order derivatives
#'     \item \code{deriv2}: Logical indicating second-order derivatives
#'     \item \code{dim_names}: List with dimension names (time, variable, sens)
#'   }
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @example inst/examples/example.R
#' @importFrom reticulate import source_python
#' @export
CppODE <- function(rhs, events = NULL, fixed = NULL, includeTimeZero = TRUE,
                   compile = TRUE, modelname = NULL,
                   deriv = TRUE, deriv2 = FALSE,
                   useDenseOutput = TRUE,
                   verbose = FALSE) {

  # --- Validate arguments ---
  if (deriv2 && !deriv) {
    warning("deriv2 = TRUE requires deriv = TRUE. Setting deriv = TRUE automatically.")
    deriv <- TRUE
  }

  # --- Clean up ODE definitions ---
  rhs <- unclass(rhs)
  rhs <- gsub("\n", "", rhs)
  rhs <- sanitizeExprs(rhs)

  # --- Extract variable and parameter names ---
  variables <- names(rhs)
  symbols <- getSymbols(c(rhs, if (!is.null(events)) {
    c(events$value,
      if ("time" %in% names(events)) events$time,
      if ("root" %in% names(events)) events$root)
  }))
  params  <- setdiff(symbols, c(variables, "time"))

  # --- Handle fixed initial conditions and parameters ---
  if (is.null(fixed)) fixed <- character(0)
  fixed_initials <- if (deriv) intersect(fixed, variables) else character(0)
  fixed_params <- if (deriv) intersect(fixed, params) else character(0)
  sens_initials  <- if (deriv) setdiff(variables, fixed_initials) else character(0)
  sens_params  <- if (deriv) setdiff(params, fixed_params) else character(0)

  # Index maps
  variable_idx0 <- setNames(seq_along(variables) - 1L, variables)
  param_idx0 <- setNames(seq_along(params) - 1L, params)
  fixed_initial_idx  <- variable_idx0[fixed_initials]
  fixed_param_idx  <- param_idx0[fixed_params]

  # --- Calculate dimensions ---
  n_variables <- length(variables)
  n_params <- length(params)
  n_sens_initials <- length(sens_initials)
  n_sens_params <- length(sens_params)
  n_total_sens <- n_sens_initials + n_sens_params

  # --- Generate unique model name ---
  if (is.null(modelname)) {
    modelname <- paste(c("x", sample(c(letters, 0:9), 8, TRUE)), collapse = "")
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
    rhs_dict = as.list(setNames(rhs, variables)),
    params_list = params,
    num_type = numType,
    fixed_states = fixed_initials,
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
      states_list = variables,
      params_list = params,
      n_states = n_variables,
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

  # --- Solver function (externC) ---
  externC <- c(
    sprintf('extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP, SEXP root_tolSEXP, SEXP maxrootSEXP, SEXP precisionSEXP) {', modelname),
    "try {",
    "",
    "  StepChecker checker(INTEGER(maxprogressSEXP)[0], INTEGER(maxstepsSEXP)[0]);",
    "",
    sprintf("  ublas::vector<%s> x(%d);", numType, n_variables),
    sprintf("  ublas::vector<%s> full_params(%d);", numType, n_variables + n_params),
    ""
  )

  # initialization of variables and parameters
  externC <- c(externC,
               "  // initialize variables",
               sprintf("  for (int i = 0; i < %d; ++i) {", n_variables))

  if (deriv2) {
    # --- Second-order AD (AD2 = fadbad::F<fadbad::F<double>>) ---
    externC <- c(externC,
                 "    x[i].x().x() = REAL(paramsSEXP)[i];",
                 "    // Outer layer seeding (second derivative)",
                 sprintf("    x[i].diff(i, %d);", n_variables + n_params))

    if (length(fixed_initial_idx) > 0) {
      externC <- c(externC,
                   sprintf("    // Skip fixed initial conditions when seeding inner layer"),
                   sprintf("    if (!(%s)) x[i].x().diff(i, %d);",
                           paste(sprintf("i == %d", fixed_initial_idx), collapse = " || "),
                           n_variables + n_params))
    } else {
      externC <- c(externC,
                   "    // Inner layer seeding (first derivative)",
                   sprintf("    x[i].x().diff(i, %d);", n_variables + n_params))
    }

  } else if (deriv) {
    # --- First-order AD only (AD = fadbad::F<double>) ---
    externC <- c(externC,
                 "    x[i] = REAL(paramsSEXP)[i];")

    if (length(fixed_initial_idx) > 0) {
      externC <- c(externC,
                   sprintf("    // Skip fixed initial conditions when seeding"),
                   sprintf("    if (!(%s)) x[i].diff(i, %d);",
                           paste(sprintf("i == %d", fixed_initial_idx), collapse = " || "),
                           n_variables + n_params))
    } else {
      externC <- c(externC,
                   sprintf("    x[i].diff(i, %d);", n_variables + n_params))
    }

  } else {
    # --- Plain double case (no AD) ---
    externC <- c(externC,
                 "    x[i] = REAL(paramsSEXP)[i];")
  }

  # assign to parameter vector
  externC <- c(externC,
               "    full_params[i] = x[i];",
               "  }",
               "",
               "  // initialize parameters",
               sprintf("  for (int i = 0; i < %d; ++i) {", n_params))

  if (deriv2) {
    externC <- c(externC,
                 sprintf("    int param_index = %d + i;", n_variables),
                 "    full_params[param_index].x().x() = REAL(paramsSEXP)[param_index];",
                 "    // Outer layer seeding (second derivative)",
                 sprintf("    full_params[param_index].diff(param_index, %d);", n_variables + n_params))

    if (length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   "    // Skip fixed parameters when seeding inner layer",
                   sprintf("    if (!(%s)) full_params[param_index].x().diff(param_index, %d);",
                           paste(sprintf("i == %d", fixed_param_idx), collapse = " || "),
                           n_variables + n_params))
    } else {
      externC <- c(externC,
                   "    // Inner layer seeding (first derivative)",
                   sprintf("    full_params[param_index].x().diff(param_index, %d);",
                           n_variables + n_params))
    }

  } else if (deriv) {
    externC <- c(externC,
                 sprintf("    int param_index = %d + i;", n_variables),
                 "    full_params[param_index] = REAL(paramsSEXP)[param_index];")

    if (length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   "    // Skip fixed parameters when seeding",
                   sprintf("    if (!(%s)) full_params[param_index].diff(param_index, %d);",
                           paste(sprintf("i == %d", fixed_param_idx), collapse = " || "),
                           n_variables + n_params))
    } else {
      externC <- c(externC,
                   sprintf("    full_params[param_index].diff(param_index, %d);",
                           n_variables + n_params))
    }

  } else {
    externC <- c(externC,
                 sprintf("    full_params[%d + i] = REAL(paramsSEXP)[%d + i];", n_variables, n_variables))
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
  if (useDenseOutput) {
    # Dense output version (WITH Interpolation)
    if (deriv2) {
      stepper_line <- paste(
        "  auto controlledStepper = rosenbrock4_controller_ad<rosenbrock4<AD2>>(abstol, reltol);",
        "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);",
        sep = "\n"
      )
      integrate_line <- "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    } else if (deriv) {
      stepper_line <- paste(
        "  auto controlledStepper = rosenbrock4_controller_ad<rosenbrock4<AD>>(abstol, reltol);",
        "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);",
        sep = "\n"
      )
      integrate_line <- "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    } else {
      stepper_line <- paste(
        "  auto controlledStepper = rosenbrock4_controller<rosenbrock4<double>>(abstol, reltol);",
        "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);",
        sep = "\n"
      )
      integrate_line <- "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    }
  } else {
    # Controlled stepper version (WITHOUT Interpolation)
    if (deriv2) {
      stepper_line <- "  auto controlledStepper = rosenbrock4_controller_ad<rosenbrock4<AD2>>(abstol, reltol);"
      integrate_line <- "  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    } else if (deriv) {
      stepper_line <- "  auto controlledStepper = rosenbrock4_controller_ad<rosenbrock4<AD>>(abstol, reltol);"
      integrate_line <- "  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    } else {
      stepper_line <- "  auto controlledStepper = rosenbrock4_controller<rosenbrock4<double>>(abstol, reltol);"
      integrate_line <- "  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    }
  }

  externC <- c(externC, "",
               "  // --- Solver setup ---",
               "  double abstol = REAL(abstolSEXP)[0];",
               "  double reltol = REAL(reltolSEXP)[0];",
               "  double root_tol = REAL(root_tolSEXP)[0];",
               "  int maxroot = INTEGER(maxrootSEXP)[0];",
               "  double precision = REAL(precisionSEXP)[0];",
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
               integrate_line,
               "",
               "  const int n_out = static_cast<int>(result_times.size());",
               "  if (n_out <= 0) Rf_error(\"Integration produced no output\");",
               ""
  )

  # --- Copy back results - CONSISTENT LIST OUTPUT ---
  if (!deriv) {
    # deriv = FALSE: list(time, variable)
    externC <- c(externC,
                 "  // --- Return list(time, variable) ---",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 2));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"variable\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP variable_mat = PROTECT(Rf_allocMatrix(REALSXP, n_out, %d));", n_variables),
                 "  double* time_out = REAL(time_vec);",
                 "  double* variable_out = REAL(variable_mat);",
                 "  auto IDX = [n_out](int r, int c){ return r + c * n_out; };",
                 "",
                 "  // Rounding helper",
                 "  auto round_to_prec = [precision](double x) {",
                 "    if (precision > 0.0) return std::round(x / precision) * precision;",
                 "    return x;",
                 "  };",
                 "",
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i];",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      variable_out[IDX(i, s)] = round_to_prec(y[i * %d + s]);", n_variables),
                 "    }",
                 "  }",
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, variable_mat);",
                 "  UNPROTECT(4);",
                 "  return ans;")

  } else if (!deriv2) {
    # deriv = TRUE, deriv2 = FALSE: list(time, variable, sens1)
    externC <- c(externC,
                 "  // --- Return list(time, variable, sens1) ---",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 3));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"variable\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"sens1\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP variable_mat = PROTECT(Rf_allocMatrix(REALSXP, n_out, %d));", n_variables),
                 "  SEXP sens1_dim = PROTECT(Rf_allocVector(INTSXP, 3));",
                 "  INTEGER(sens1_dim)[0] = n_out;",
                 sprintf("  INTEGER(sens1_dim)[1] = %d;", n_variables),
                 sprintf("  INTEGER(sens1_dim)[2] = %d;", n_total_sens),
                 "  SEXP sens1_arr = PROTECT(Rf_allocArray(REALSXP, sens1_dim));",
                 "",
                 "  double* time_out = REAL(time_vec);",
                 "  double* variable_out = REAL(variable_mat);",
                 "  double* sens1_out = REAL(sens1_arr);",
                 "",
                 "  auto IDX_variable = [n_out](int r, int c){ return r + c * n_out; };",
                 sprintf("  auto IDX_sens1 = [n_out](int t, int s, int v){ return t + n_out * (s + %d * v); };", n_variables),
                 "",
                 "  // Rounding helper",
                 "  auto round_to_prec = [precision](double x) {",
                 "    if (precision > 0.0) return std::round(x / precision) * precision;",
                 "    return x;",
                 "  };",
                 ""
    )

    # Extract with fixed parameter handling
    if (length(fixed_initial_idx) > 0 || length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   "  for (int i = 0; i < n_out; ++i) {",
                   "    time_out[i] = result_times[i].x();",
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                   "      variable_out[IDX_variable(i, s)] = round_to_prec(xi.x());",
                   "      int v_sens = 0;",
                   sprintf("      for (int v = 0; v < %d; ++v) {", n_variables + n_params))

      fixed_checks <- character(0)
      if (length(fixed_initial_idx) > 0) {
        fixed_checks <- c(fixed_checks,
                          sprintf("        bool is_fixed_init = (v < %d) && (%s);",
                                  n_variables,
                                  paste(sprintf("v == %d", fixed_initial_idx), collapse = " || ")))
      } else {
        fixed_checks <- c(fixed_checks, "        bool is_fixed_init = false;")
      }

      if (length(fixed_param_idx) > 0) {
        fixed_checks <- c(fixed_checks,
                          sprintf("        bool is_fixed_param = (v >= %d) && (%s);",
                                  n_variables,
                                  paste(sprintf("(v - %d) == %d", n_variables, fixed_param_idx), collapse = " || ")))
      } else {
        fixed_checks <- c(fixed_checks, "        bool is_fixed_param = false;")
      }

      externC <- c(externC, fixed_checks,
                   "        if (!(is_fixed_init || is_fixed_param)) {",
                   "          sens1_out[IDX_sens1(i, s, v_sens)] = round_to_prec(xi.d(v));",
                   "          v_sens++;",
                   "        }",
                   "      }",
                   "    }",
                   "  }")
    } else {
      externC <- c(externC,
                   "  for (int i = 0; i < n_out; ++i) {",
                   "    time_out[i] = result_times[i].x();",
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                   "      variable_out[IDX_variable(i, s)] = round_to_prec(xi.x());",
                   sprintf("      for (int v = 0; v < %d; ++v) {", n_total_sens),
                   "        sens1_out[IDX_sens1(i, s, v)] = round_to_prec(xi.d(v));",
                   "      }",
                   "    }",
                   "  }")
    }

    externC <- c(externC,
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, variable_mat);",
                 "  SET_VECTOR_ELT(ans, 2, sens1_arr);",
                 "  UNPROTECT(6);",
                 "  return ans;")

  } else {
    # deriv2 = TRUE: list(time, variable, sens1, sens2)
    externC <- c(externC,
                 "  // --- Copy results to R list for deriv2 ---",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 4));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"variable\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"sens1\"));",
                 "  SET_STRING_ELT(names, 3, Rf_mkChar(\"sens2\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP variable_mat = PROTECT(Rf_allocMatrix(REALSXP, n_out, %d));", n_variables),
                 "  SEXP sens1_dim = PROTECT(Rf_allocVector(INTSXP, 3));",
                 "  INTEGER(sens1_dim)[0] = n_out;",
                 sprintf("  INTEGER(sens1_dim)[1] = %d;", n_variables),
                 sprintf("  INTEGER(sens1_dim)[2] = %d;", n_total_sens),
                 "  SEXP sens1_arr = PROTECT(Rf_allocArray(REALSXP, sens1_dim));",
                 "  SEXP sens2_dim = PROTECT(Rf_allocVector(INTSXP, 4));",
                 "  INTEGER(sens2_dim)[0] = n_out;",
                 sprintf("  INTEGER(sens2_dim)[1] = %d;", n_variables),
                 sprintf("  INTEGER(sens2_dim)[2] = %d;", n_total_sens),
                 sprintf("  INTEGER(sens2_dim)[3] = %d;", n_total_sens),
                 "  SEXP sens2_arr = PROTECT(Rf_allocArray(REALSXP, sens2_dim));",
                 "",
                 "  double* time_out = REAL(time_vec);",
                 "  double* variable_out = REAL(variable_mat);",
                 "  double* sens1_out = REAL(sens1_arr);",
                 "  double* sens2_out = REAL(sens2_arr);",
                 "",
                 "  auto IDX_variable = [n_out](int r, int c){ return r + c * n_out; };",
                 sprintf("  auto IDX_sens1 = [n_out](int t, int s, int v){ return t + n_out * (s + %d * v); };", n_variables),
                 sprintf("  auto IDX_sens2 = [n_out](int t, int s, int v1, int v2){ return t + n_out * (s + %d * (v1 + %d * v2)); };", n_variables, n_total_sens),
                 "",
                 "  // Rounding helper",
                 "  auto round_to_prec = [precision](double x) {",
                 "    if (precision > 0.0) return std::round(x / precision) * precision;",
                 "    return x;",
                 "  };",
                 ""
    )

    # Extract with fixed parameter handling
    if (length(fixed_initial_idx) > 0 || length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   "  for (int i = 0; i < n_out; ++i) {",
                   "    time_out[i] = result_times[i].x().x();",
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                   "      variable_out[IDX_variable(i, s)] = round_to_prec(xi.x().x());",
                   "      int v1_sens = 0;",
                   sprintf("      for (int v1 = 0; v1 < %d; ++v1) {", n_variables + n_params))

      fixed_checks_outer <- character(0)
      if (length(fixed_initial_idx) > 0) {
        fixed_checks_outer <- c(fixed_checks_outer,
                                sprintf("        bool is_fixed_init1 = (v1 < %d) && (%s);",
                                        n_variables,
                                        paste(sprintf("v1 == %d", fixed_initial_idx), collapse = " || ")))
      } else {
        fixed_checks_outer <- c(fixed_checks_outer, "        bool is_fixed_init1 = false;")
      }

      if (length(fixed_param_idx) > 0) {
        fixed_checks_outer <- c(fixed_checks_outer,
                                sprintf("        bool is_fixed_param1 = (v1 >= %d) && (%s);",
                                        n_variables,
                                        paste(sprintf("(v1 - %d) == %d", n_variables, fixed_param_idx), collapse = " || ")))
      } else {
        fixed_checks_outer <- c(fixed_checks_outer, "        bool is_fixed_param1 = false;")
      }

      externC <- c(externC, fixed_checks_outer,
                   "        if (!(is_fixed_init1 || is_fixed_param1)) {",
                   "          sens1_out[IDX_sens1(i, s, v1_sens)] = round_to_prec(xi.d(v1).x());",
                   "          int v2_sens = 0;",
                   sprintf("          for (int v2 = 0; v2 < %d; ++v2) {", n_variables + n_params))

      fixed_checks_inner <- character(0)
      if (length(fixed_initial_idx) > 0) {
        fixed_checks_inner <- c(fixed_checks_inner,
                                sprintf("            bool is_fixed_init2 = (v2 < %d) && (%s);",
                                        n_variables,
                                        paste(sprintf("v2 == %d", fixed_initial_idx), collapse = " || ")))
      } else {
        fixed_checks_inner <- c(fixed_checks_inner, "            bool is_fixed_init2 = false;")
      }

      if (length(fixed_param_idx) > 0) {
        fixed_checks_inner <- c(fixed_checks_inner,
                                sprintf("            bool is_fixed_param2 = (v2 >= %d) && (%s);",
                                        n_variables,
                                        paste(sprintf("(v2 - %d) == %d", n_variables, fixed_param_idx), collapse = " || ")))
      } else {
        fixed_checks_inner <- c(fixed_checks_inner, "            bool is_fixed_param2 = false;")
      }

      externC <- c(externC, fixed_checks_inner,
                   "            if (!(is_fixed_init2 || is_fixed_param2)) {",
                   "              sens2_out[IDX_sens2(i, s, v1_sens, v2_sens)] = round_to_prec(xi.d(v1).d(v2));",
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
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                   "      variable_out[IDX_variable(i, s)] = round_to_prec(xi.x().x());",
                   sprintf("      for (int v1 = 0; v1 < %d; ++v1) {", n_total_sens),
                   "        sens1_out[IDX_sens1(i, s, v1)] = round_to_prec(xi.d(v1).x());",
                   sprintf("        for (int v2 = 0; v2 < %d; ++v2) {", n_total_sens),
                   "          sens2_out[IDX_sens2(i, s, v1, v2)] = round_to_prec(xi.d(v1).d(v2));",
                   "        }",
                   "      }",
                   "    }",
                   "  }")
    }

    externC <- c(externC,
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, variable_mat);",
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

  # Warn if file already exists
  if (file.exists(filename)) {
    if (verbose) {
      message("⚠ Overwriting existing file: ", normalizePath(filename))
    }
  }

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

  # --- Attach attributes ---
  jac_matrix_R <- matrix(unlist(jac_matrix_str), nrow = n_variables, ncol = n_variables, byrow = TRUE)
  dimnames(jac_matrix_R) <- list(variables, variables)

  attr(modelname, "equations")     <- rhs
  attr(modelname, "modelname")     <- modelname
  attr(modelname, "variables")     <- variables
  attr(modelname, "parameters")    <- params
  attr(modelname, "events")        <- events
  attr(modelname, "solver")        <- "boost::odeint::rosenbrock4"
  attr(modelname, "fixed")         <- c(fixed_initials, fixed_params)
  attr(modelname, "jacobian")      <- list(f.x = jac_matrix_R, f.time = time_derivs_str)
  attr(modelname, "deriv")         <- deriv
  attr(modelname, "deriv2")        <- deriv2

  # Dimension names - ALWAYS when deriv = TRUE
  if (deriv) {
    attr(modelname, "dim_names") <- list(
      time = "time",
      variable = variables,
      sens = c(sens_initials, sens_params)
    )
  } else {
    attr(modelname, "dim_names") <- list(
      time = "time",
      variable = variables
    )
  }

  if (compile) compile(modelname, verbose = verbose)
  return(modelname)
}


#' Create and (optionally) compile C++ code for algebraic expressions
#'
#' `funCpp()` transforms a named character vector of algebraic expressions
#' into an R callable function that evaluates them either in pure R or through
#' an automatically generated and compiled C++ backend.
#'
#' If symbolic derivatives are requested (`deriv = TRUE` or `deriv2 = TRUE`),
#' the function uses [`derivSymb()`] to compute analytic Jacobians and
#' Hessians via **SymPy** (Python).  The symbolic results are attached both to
#' the returned function and, when evaluated, as numeric derivative arrays.
#'
#' Complex symbolic forms that cannot be parsed by R (for example,
#' `Piecewise()` or `DiracDelta()`) are detected automatically.  In such
#' cases, the pure-R fallback is disabled and a clear warning is issued,
#' advising the user to use `compile = TRUE` for C++ evaluation.
#'
#' @param x Named character vector of algebraic expressions.
#' @param variables Character vector of variable names.
#' @param parameters Character vector of parameter names.
#' @param compile Logical; if `TRUE`, generate, compile, and load the C++ code.
#' @param modelname Optional base name for generated files.
#' @param verbose Logical; print diagnostic output.
#' @param warnings Logical; warn about missing variables or parameters
#'   (filled with zeros).
#' @param convenient Logical; if `TRUE` (default), return a wrapper function
#'   that accepts variables and parameters as `...`.
#' @param deriv Logical; if `TRUE`, compute the symbolic Jacobian.
#' @param deriv2 Logical; if `TRUE`, compute the symbolic Hessian
#'   (implies `deriv = TRUE`).
#'
#' @return A callable function that evaluates the system of equations.
#'
#' The output of the function call is a numeric matrix
#' `[n_obs × n_outputs]`.  If derivatives are available,
#' attributes `"jacobian"` (`[n_out × n_sym × n_obs]`) and
#' `"hessian"` (`[n_out × n_sym × n_sym × n_obs]`) are attached.
#'
#' The returned function itself also carries the symbolic representations:
#' \itemize{
#'   \item `attr(f, "jacobian.symb")` – character matrix of analytic first derivatives
#'   \item `attr(f, "hessian.symb")` – character 3D array of analytic second derivatives
#' }
#'
#' If symbolic parsing fails for R fallback (e.g. due to `Piecewise` or
#' `DiracDelta`), a warning is issued at creation time and only compiled
#' evaluation will be available.
#'
#' @examples
#' \dontrun{
#' f <- funCpp(c(y = "a*x^2 + b*y^2"), deriv2 = TRUE)
#' f(x = 1:3, y = 2, a = 0.5, b = 1)
#' attr(f, "jacobian.symb")
#' attr(f, "hessian.symb")
#' }
#'
#' @seealso [derivSymb()] for symbolic differentiation,
#'   [ensurePythonEnv()] for Python setup.
#' @export
funCpp <- function(x,
                   variables  = getSymbols(x, exclude = parameters),
                   parameters = NULL,
                   compile    = FALSE,
                   modelname  = NULL,
                   verbose    = FALSE,
                   warnings   = TRUE,
                   convenient = TRUE,
                   deriv      = TRUE,
                   deriv2     = FALSE) {

  # ---------------------------------------------------------------------------
  # Validation
  # ---------------------------------------------------------------------------
  if (deriv2 && !deriv) {
    warning("deriv2 = TRUE requires deriv = TRUE. Setting deriv = TRUE automatically.")
    deriv <- TRUE
  }

  outnames <- names(x)
  if (is.null(outnames))
    outnames <- paste0("f", seq_along(x))

  innames  <- variables
  all_syms <- c(variables, parameters)

  if (is.null(modelname))
    modelname <- paste(c("f", sample(c(letters, 0:9), 8, TRUE)), collapse = "")

  # ---------------------------------------------------------------------------
  # Argument checking helper
  # ---------------------------------------------------------------------------
  checkArguments <- function(M, p) {
    n_obs <- 1
    if (is.null(innames) || length(innames) == 0) {
      M <- matrix(0, nrow = 1, ncol = 0)
      n_obs <- 1
    } else {
      if (is.null(M)) stop("Variables defined but 'vars' is NULL.")
      if (is.vector(M) && !is.list(M)) {
        M <- matrix(M, ncol = length(innames))
        colnames(M) <- innames
      }
      if (is.null(colnames(M))) {
        if (ncol(M) != length(innames))
          stop("vars has no column names and wrong length.")
        colnames(M) <- innames
      }
      if (!all(innames %in% colnames(M))) {
        if (warnings) warning("Missing variable columns → filled with 0.")
        missing <- setdiff(innames, colnames(M))
        add <- matrix(0, nrow = nrow(M), ncol = length(missing),
                      dimnames = list(NULL, missing))
        M <- cbind(M, add)
      }
      M <- M[, innames, drop = FALSE]
      n_obs <- nrow(M)
    }
    M <- t(M)
    if (is.null(parameters) || length(parameters) == 0) {
      p <- numeric(0)
    } else {
      if (is.null(names(p)))
        stop("params must be a named numeric vector.")
      if (!all(parameters %in% names(p))) {
        if (warnings) warning("Missing parameters → filled with 0.")
        missing <- setdiff(parameters, names(p))
        add <- structure(rep(0, length(missing)), names = missing)
        p <- c(p, add)
      }
      p <- p[parameters]
    }
    list(M = M, p = p, n_obs = n_obs)
  }

  # ---------------------------------------------------------------------------
  # Symbolic derivatives
  # ---------------------------------------------------------------------------
  sym_jac  <- NULL
  sym_hess <- NULL
  if (deriv || deriv2) {
    if (verbose) message("Computing symbolic derivatives via derivSymb() ...")
    ds <- derivSymb(x, deriv2 = deriv2, real = TRUE, verbose = verbose)
    sym_jac  <- ds$jacobian
    sym_hess <- ds$hessian
  }

  # ---------------------------------------------------------------------------
  # Replace Heaviside for R compatibility
  # ---------------------------------------------------------------------------
  replaceHeaviside <- function(expr_str) {
    if (is.null(expr_str) || expr_str == "0") return(expr_str)
    gsub("Heaviside\\(([^)]+)\\)", "ifelse(\\1 >= 0, 1, 0)", expr_str)
  }

  # ---------------------------------------------------------------------------
  # Parse expressions with tryCatch; detect non-R-parsable constructs
  # ---------------------------------------------------------------------------
  fallback_available <- TRUE
  safe_parse <- function(expr_str) {
    if (is.null(expr_str) || expr_str == "0") return(list(expression(0)))
    expr_str <- replaceHeaviside(expr_str)
    out <- tryCatch(list(parse(text = expr_str)),
                    error = function(e) {
                      fallback_available <<- FALSE
                      NULL
                    })
    if (is.null(out)) list(NULL) else out
  }

  parsed_exprs <- lapply(x, function(e) safe_parse(e)[[1]])

  parsed_jac <- NULL
  if (!is.null(sym_jac)) {
    parsed_jac <- matrix(vector("list", nrow(sym_jac) * ncol(sym_jac)),
                         nrow = nrow(sym_jac), ncol = ncol(sym_jac),
                         dimnames = dimnames(sym_jac))
    for (i in seq_len(nrow(sym_jac)))
      for (j in seq_len(ncol(sym_jac)))
        parsed_jac[[i, j]] <- safe_parse(sym_jac[i, j])
  }

  # ---------------------------------------------------------------------------
  # Parse Hessian safely (support list-of-matrices OR array)
  # ---------------------------------------------------------------------------
  parsed_hess <- NULL
  if (!is.null(sym_hess)) {
    if (is.list(sym_hess) && !is.array(sym_hess)) {
      # List of matrices (new format)
      parsed_hess <- lapply(names(sym_hess), function(fname) {
        H <- sym_hess[[fname]]
        parsed <- matrix(vector("list", nrow(H) * ncol(H)),
                         nrow = nrow(H), ncol = ncol(H),
                         dimnames = dimnames(H))
        for (i in seq_len(nrow(H))) {
          for (j in seq_len(ncol(H))) {
            parsed[[i, j]] <- safe_parse(H[i, j])
          }
        }
        parsed
      })
      names(parsed_hess) <- names(sym_hess)
    } else {
      # 3D array (old format)
      parsed_hess <- array(vector("list", prod(dim(sym_hess))),
                           dim = dim(sym_hess), dimnames = dimnames(sym_hess))
      for (i in seq_len(dim(sym_hess)[1]))
        for (j in seq_len(dim(sym_hess)[2]))
          for (k in seq_len(dim(sym_hess)[3]))
            parsed_hess[[i, j, k]] <- safe_parse(sym_hess[i, j, k])
    }
  }

  if (!fallback_available)
    warning("R fallback not available (expressions contain non-R constructs such as Piecewise or DiracDelta). ",
            "Please use compile = TRUE.")

  # ---------------------------------------------------------------------------
  # Generate C++ code via Python
  # ---------------------------------------------------------------------------
  cppfile <- NULL
  if (!is.null(modelname)) {
    ensurePythonEnv(envname = "CppODE", verbose = verbose)
    pytools <- reticulate::import_from_path(
      "generatefunCppCode",
      path = system.file("python", package = "CppODE")
    )
    exprs_list <- as.list(x)
    names(exprs_list) <- outnames
    py_jac <- if (!is.null(sym_jac))
      lapply(seq_len(nrow(sym_jac)), function(i) as.list(as.character(sym_jac[i,])))
    else NULL
    if (!is.null(py_jac)) names(py_jac) <- rownames(sym_jac)
    py_hess <- if (!is.null(sym_hess))
      lapply(seq_len(length(sym_hess)), function(i)
        lapply(seq_len(nrow(sym_hess[[i]])), function(j)
          as.list(as.character(sym_hess[[i]][j, ]))))
    else NULL
    if (!is.null(py_hess)) names(py_hess) <- names(sym_hess)
    if (verbose) message("Generating C++ source code ...")
    pyres <- pytools$generate_fun_cpp(
      exprs      = exprs_list,
      variables  = if (length(variables) > 0) variables else list(),
      parameters = if (length(parameters) > 0) parameters else list(),
      jacobian   = py_jac,
      hessian    = py_hess,
      modelname  = modelname
    )
    cppfile <- pyres$filename
    if (verbose) message("Wrote: ", normalizePath(cppfile))
  }

  # ---------------------------------------------------------------------------
  # Main evaluation function
  # ---------------------------------------------------------------------------
  myRfun <- function(vars, params = numeric(0), attach.input = FALSE) {
    checked <- checkArguments(vars, params)
    M <- checked$M; p <- checked$p; n_obs <- checked$n_obs

    funsym_eval <- paste0(modelname, "_eval")
    funsym_jac  <- paste0(modelname, "_jacobian")
    funsym_hess <- paste0(modelname, "_hessian")
    sofile <- paste0(modelname, .Platform$dynlib.ext)
    compiled_available <- file.exists(sofile) && is.loaded(funsym_eval)

    if (!compiled_available && !fallback_available)
      stop("R fallback not available for this model. Please use compile = TRUE.")

    # =========================================================================
    # Evaluation
    # =========================================================================
    if (compiled_available) {
      if (verbose) message("Using compiled function: ", funsym_eval)
      xvec <- as.double(as.vector(M))
      yvec <- double(length(outnames) * n_obs)
      n <- as.integer(n_obs)
      k <- as.integer(length(innames))
      l <- as.integer(length(outnames))
      out <- .C(funsym_eval, x = xvec, y = yvec, p = as.double(p), n = n, k = k, l = l)
      res <- matrix(out$y, nrow = n_obs, ncol = length(outnames), byrow = FALSE)
      colnames(res) <- outnames
    } else {
      if (verbose) message("Using R fallback (compiled version not available)")
      res <- matrix(NA_real_, nrow = n_obs, ncol = length(outnames))
      colnames(res) <- outnames
      for (i in seq_len(n_obs)) {
        env <- as.list(c(as.numeric(M[, i]), p))
        names(env) <- c(innames, parameters)
        vals <- vapply(parsed_exprs, function(expr) eval(expr, envir = env), numeric(1))
        res[i, ] <- vals
      }
    }

    # =========================================================================
    # Jacobian
    # =========================================================================
    if (deriv && !is.null(sym_jac)) {
      n_out <- length(outnames); n_sym <- length(all_syms)
      if (compiled_available && is.loaded(funsym_jac)) {
        if (verbose) message("Using compiled Jacobian: ", funsym_jac)
        xvec <- as.double(as.vector(M))
        jac_vec <- double(n_out * n_sym * n_obs)
        n <- as.integer(n_obs)
        k <- as.integer(length(innames))
        l <- as.integer(n_out)
        out_jac <- .C(funsym_jac, x = xvec, jac = jac_vec, p = as.double(p),
                      n = n, k = k, l = l)
        jac_arr <- array(out_jac$jac, dim = c(n_out, n_sym, n_obs),
                         dimnames = list(outnames, all_syms, NULL))
      } else {
        if (verbose) message("Using R fallback for Jacobian")
        jac_arr <- array(0, dim = c(n_out, n_sym, n_obs),
                         dimnames = list(outnames, all_syms, NULL))
        for (obs_i in seq_len(n_obs)) {
          env <- as.list(c(as.numeric(M[, obs_i]), p))
          names(env) <- c(innames, parameters)
          for (out_i in seq_len(n_out))
            for (sym_i in seq_len(n_sym)) {
              expr <- parsed_jac[[out_i, sym_i]][[1]]
              jac_arr[out_i, sym_i, obs_i] <- if (is.null(expr)) 0 else eval(expr, envir = env)
            }
        }
      }
      attr(res, "jacobian") <- jac_arr
    }

    # =========================================================================
    # Hessian
    # =========================================================================
    if (deriv2 && !is.null(sym_hess)) {
      n_out <- length(outnames); n_sym <- length(all_syms)
      if (compiled_available && is.loaded(funsym_hess)) {
        if (verbose) message("Using compiled Hessian: ", funsym_hess)
        xvec <- as.double(as.vector(M))
        hess_vec <- double(n_out * n_sym * n_sym * n_obs)
        n <- as.integer(n_obs)
        k <- as.integer(length(innames))
        l <- as.integer(n_out)
        out_hess <- .C(funsym_hess, x = xvec, hess = hess_vec, p = as.double(p),
                       n = n, k = k, l = l)
        hes_arr <- array(out_hess$hess, dim = c(n_out, n_sym, n_sym, n_obs),
                         dimnames = list(outnames, all_syms, all_syms, NULL))
      } else {
        if (verbose) message("Using R fallback for Hessian")
        hes_arr <- array(0, dim = c(n_out, n_sym, n_sym, n_obs),
                         dimnames = list(outnames, all_syms, all_syms, NULL))
        for (obs_i in seq_len(n_obs)) {
          env <- as.list(c(as.numeric(M[, obs_i]), p))
          names(env) <- c(innames, parameters)
          for (out_i in seq_len(n_out)) {
            if (is.list(parsed_hess) && !is.array(parsed_hess)) {
              Hmat <- parsed_hess[[outnames[out_i]]]
              for (sym_i in seq_len(n_sym))
                for (sym_j in seq_len(n_sym)) {
                  expr <- Hmat[[sym_i, sym_j]][[1]]
                  hes_arr[out_i, sym_i, sym_j, obs_i] <-
                    if (is.null(expr)) 0 else eval(expr, envir = env)
                }
            } else {
              for (sym_i in seq_len(n_sym))
                for (sym_j in seq_len(n_sym)) {
                  expr <- parsed_hess[[out_i, sym_i, sym_j]][[1]]
                  hes_arr[out_i, sym_i, sym_j, obs_i] <-
                    if (is.null(expr)) 0 else eval(expr, envir = env)
                }
            }
          }
        }
      }
      attr(res, "hessian") <- hes_arr
    }

    if (attach.input && ncol(M) > 0)
      res <- cbind(t(M), res)
    res
  }

  # ---------------------------------------------------------------------------
  # Convenience wrapper and attributes
  # ---------------------------------------------------------------------------
  outfn <- myRfun
  if (convenient) {
    outfn <- function(..., attach.input = FALSE) {
      arglist <- list(...)
      M <- if (!is.null(innames) && length(innames) > 0)
        do.call(cbind, arglist[innames]) else NULL
      p <- if (!is.null(parameters) && length(parameters) > 0)
        do.call(c, arglist[parameters]) else numeric(0)
      myRfun(M, p, attach.input)
    }
  }

  attr(outfn, "equations")     <- x
  attr(outfn, "variables")     <- variables
  attr(outfn, "parameters")    <- parameters
  attr(outfn, "modelname")     <- modelname
  if (deriv && !is.null(sym_jac))
    attr(outfn, "jacobian.symb") <- sym_jac
  if (deriv2 && !is.null(sym_hess))
    attr(outfn, "hessian.symb")  <- sym_hess

  if (compile) {
    if (verbose) message("Compiling generated C++ code ...")
    compile(outfn, verbose = verbose)
  }

  outfn
}


#' @title Internal C++ model compiler
#' @description
#' Compiles one or more generated C++ source files (from \code{funCpp0()} or \code{CppFun()})
#' into shared libraries (*.so / *.dll) using \code{R CMD SHLIB}.
#' This is an internal helper and not intended for direct user calls.
#'
#' @param ... One or more model functions that have a \code{"modelname"} attribute
#'   corresponding to a C++ source file (e.g. \code{"funCpp0_ab12cd"}).
#' @param output Optional base name for the combined shared object.
#'   If provided, all source files are compiled and linked into a single library.
#' @param args Optional compiler/linker arguments (e.g. \code{"-lm"}).
#' @param verbose Logical; if \code{TRUE}, show compiler output.
#' @param cores Number of parallel compilation jobs (ignored on Windows).
#'
#' @details
#' The function automatically sets platform-appropriate compiler flags
#' (\code{-std=c++20 -O2 -DNDEBUG -fPIC}) and includes the CppODE headers.
#' Each object must have a valid \code{modelname} attribute referring
#' to an existing \code{.cpp} or \code{.c} file.
#'
#' @importFrom parallel mclapply
#' @keywords internal
compile <- function(..., output = NULL, args = NULL, cores = 1, verbose = FALSE) {
  objects <- list(...)
  obj.names <- as.character(substitute(list(...)))[-1]

  # --- collect all source files ---
  files <- character()
  for (i in seq_along(objects)) {
    obj <- objects[[i]]
    mdl <- attr(obj, "modelname")
    if (!is.null(mdl)) {
      candidates <- c(paste0(mdl, ".cpp"), paste0(mdl, ".c"))
      candidates <- candidates[file.exists(candidates)]
      if (length(candidates))
        files <- union(files, candidates)
      else if (verbose)
        message("⚠ No source file found for model: ", mdl)
    }
  }

  if (length(files) == 0)
    stop("No valid .cpp or .c source files found for provided objects.")

  roots <- vapply(files, function(f) sub("\\.[^.]+$", "", f), character(1))
  .so <- .Platform$dynlib.ext

  # --- Clean up old compiled files ---
  for (root in roots) {
    so_file <- paste0(root, .so)
    o_file <- paste0(root, ".o")

    # Unload shared library if loaded
    try(dyn.unload(so_file), silent = TRUE)

    # Delete old .so and .o files
    if (file.exists(so_file)) {
      if (verbose) message("Removing old: ", so_file)
      unlink(so_file)
    }
    if (file.exists(o_file)) {
      if (verbose) message("Removing old: ", o_file)
      unlink(o_file)
    }
  }

  # --- Compiler flags ---
  if (Sys.info()[["sysname"]] == "Windows") cores <- 1
  include_flags <- paste0("-I", shQuote(system.file("include", package = "CppODE")))
  cxxflags <- if (Sys.info()[["sysname"]] == "Windows") {
    "-std=c++20 -O2 -DNDEBUG"
  } else {
    "-std=c++20 -O2 -DNDEBUG -fPIC -fno-var-tracking-assignments"
  }

  Sys.setenv(
    PKG_CPPFLAGS = include_flags,
    PKG_CXXFLAGS = cxxflags
  )

  # --- Compilation ---
  if (is.null(output)) {
    if (verbose) message("Compiling ", length(files), " model(s)...")
    parallel::mclapply(seq_along(files), function(i) {
      root <- roots[i]
      cmd <- paste0(R.home("bin"), "/R CMD SHLIB ", shQuote(files[i]), " ", args)
      system(cmd, intern = !verbose)
      dyn.load(paste0(root, .so))
      if (verbose) message("✔ Loaded ", root, .so)
      invisible(root)
    }, mc.cores = cores, mc.silent = !verbose)
  } else {
    # --- Combine all into one shared object ---
    output <- sub("\\.so$", "", output)

    # Clean up output file too
    output_so <- paste0(output, .so)
    output_o <- paste0(output, ".o")
    try(dyn.unload(output_so), silent = TRUE)
    if (file.exists(output_so)) {
      if (verbose) message("Removing old: ", output_so)
      unlink(output_so)
    }
    if (file.exists(output_o)) {
      if (verbose) message("Removing old: ", output_o)
      unlink(output_o)
    }

    cmd <- paste0(
      R.home("bin"), "/R CMD SHLIB ",
      paste(shQuote(files), collapse = " "),
      " -o ", shQuote(output_so), " ", args
    )
    if (verbose)
      message("Linking into shared library: ", output, .so)
    system(cmd, intern = !verbose)
    dyn.load(output_so)
    if (verbose)
      message("✔ Loaded ", output, .so)
  }

  invisible(TRUE)
}

