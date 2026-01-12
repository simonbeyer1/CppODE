#' Generate C++ code for ODE models with events and parameter sensitivities
#'
#' @description
#' This function generates and compiles a C++ solver for systems of ordinary differential
#' equations (ODEs) of the form
#'
#' \deqn{\dot{x}(t) = f\!\big(x(t), p_{\text{dyn}}\big), \quad x(t_0) = p_{\text{init}}}
#'
#' using [**Boost.Odeint's**](https://www.boost.org/doc/libs/1_89_0/libs/numeric/odeint/doc/html/index.html)
#' stiff Rosenbrock4 method with dense output and error control (using third order in combination).
#' The solver supports **time-based** and **root-triggered events** and can, optionally,
#' compute **first- and second-order sensitivities** by evaluating the *same system* with
#' **dual number** types provided by
#' [**FADBAD++**](https://uning.dk/fadbad.html).
#'
#' ## Sensitivity Computation
#'
#' If `deriv = TRUE`, all state variables and parameters are represented as
#' [**dual numbers**](https://en.wikipedia.org/wiki/Dual_number) of type `F<double>`.
#' The ODE right-hand side \eqn{f} is evaluated on these dual numbers; due to the chain rule
#' encoded in dual number arithmetic, the **derivative components propagate automatically**
#' through every operation in \eqn{f}. Consequently, the numerical integration solves
#' *exactly the same* initial value problem, only over the dual number algebra
#' \eqn{\mathbb{D}}, yielding both the state trajectories and their first derivatives
#' in a single pass.
#'
#' If `deriv2 = TRUE` (which implies `deriv = TRUE`), nested dual numbers
#' `F<F<double>>` are used. This allows the evaluation of \eqn{f} over the
#' second-order dual algebra \eqn{\mathbb{D} \otimes \mathbb{D}}, providing **second-order
#' sensitivities** directly through nested automatic differentiation.
#'
#' Fixed initial conditions or parameters (listed in `fixed`) are created as plain scalars
#' and therefore do **not** contribute sensitivity components.
#'
#' If both `deriv = FALSE` and `deriv2 = FALSE`, plain doubles are used and no sensitivities
#' are produced.
#'
#' ## Event handling
#'
#' Events are specified in a `data.frame` with the following columns:
#'
#' | Column | Description |
#' |:--|:--|
#' | `var` | Name of the affected variable |
#' | `value` | Numeric value to apply at the event |
#' | `method` | How the value is applied: `"replace"`, `"add"`, or `"multiply"` |
#' | `time` | *(optional)* Time point at which the event occurs |
#' | `root` | *(optional)* Root expression in terms of variables and `time` |
#'
#' Each event must define either `time` or `root`.
#' Root-triggered events fire when the `root` expression crosses zero.
#'
#' ## Output
#'
#' The generated solver function (accessible via `.Call`) returns a named list:
#'
#' - `deriv = FALSE`, `deriv2 = FALSE`
#'   Returns `list(time, variable)`
#'   - `time`: numeric vector of length \eqn{n_t}
#'   - `variable`: numeric matrix \eqn{X_{ij}} of shape \eqn{(n_t,n_x)}, containing \eqn{x_j(t_i)}
#'
#' - `deriv = TRUE`, `deriv2 = FALSE`
#'   Returns `list(time, variable, sens1)`
#'   - `sens1`: numeric array \eqn{\partial X_{ijk}} of shape \eqn{(n_t,n_x,n_s)}, containing
#'     \eqn{\partial x_j(t_i)/\partial p_k}
#'
#' - `deriv = TRUE`, `deriv2 = TRUE`
#'   Returns `list(time, variable, sens1, sens2)`
#'   - `sens2`: numeric array \eqn{\partial^2 X_{ijkl}} of shape \eqn{(n_t,n_x,n_s,n_s)},
#'     containing \eqn{\partial^2 x_j(t_i)/\partial p_k\,\partial p_l}
#'
#' Here \eqn{n_t} is the number of output time points, \eqn{n_x} the number of state
#' variables, and \eqn{n_s} the number of sensitivity parameters (non-fixed initials and parameters).
#'
#' @param rhs Named character vector of ODE right-hand sides; names must correspond to variables.
#' @param events Optional `data.frame` describing events (see **Events**). Default: `NULL`.
#' @param fixed Character vector of fixed initial conditions or parameters (excluded from sensitivities).
#' @param includeTimeZero Logical. If `TRUE`, ensure that time `0` is included among integration times.
#' @param compile Logical. If `TRUE`, compiles and loads the generated C++ code.
#' @param modelname Optional base name for the generated C++ source file
#'   \emph{and} for all generated C/C++ symbols (e.g. \code{solve_<modelname>})
#'   as well as the resulting shared library.
#'   If \code{NULL}, a random identifier is used.
#' @param outdir Directory where generated C++ source files are written. Defaults to `tempdir()`.
#' @param deriv Logical. If `TRUE`, enable first-order sensitivities via dual numbers.
#' @param deriv2 Logical. If `TRUE`, enable second-order sensitivities via nested dual numbers; requires `deriv = TRUE`.
#' @param fullErr Logical. If `TRUE`, compute error estimates using full state vector including derivatives. If `FALSE`, use only the value components for error control.
#' @param useDenseOutput Logical. If `TRUE`, use dense output (Hermite interpolation).
#' @param verbose Logical. If `TRUE`, print progress messages.
#'
#' @return
#' The compiled model name (character).
#' The returned object carries a set of attributes that describe the compiled solver
#' and its symbolic structure:
#'
#' | Attribute | Type | Description |
#' |:--|:--|:--|
#' | `equations` | `character` | ODE right-hand side definitions |
#' | `variables` | `character` | Names of the dynamic state variables |
#' | `parameters` | `character` | Names of model parameters |
#' | `events` | `data.frame` | Table of event specifications (if any) |
#' | `solver` | `list` | Description of the numerical solver configuration |
#' | `fixed` | `character` | Names of fixed initial conditions or parameters |
#' | `jacobian` | `eqnvec` | Symbolic expressions for the system Jacobian |
#' | `deriv` | `logical` | Indicates whether first-order sensitivities (dual numbers) were used |
#' | `deriv2` | `logical` | Indicates whether nested dual numbers were used for second-order sensitivities |
#' | `dim_names` | `list` | Dimension names for arrays: `time`, `variable`, and `sens` |
#'
#' @author Simon Beyer, <simon.beyer@@fdm.uni-freiburg.de>
#' @example inst/examples/example_ODE.R
#' @importFrom reticulate source_python
#' @importFrom stats setNames
#' @export
CppODE <- function(rhs, events = NULL, fixed = NULL, includeTimeZero = TRUE,
                   compile = TRUE, modelname = NULL, outdir = tempdir(),
                   deriv = TRUE, deriv2 = FALSE, fullErr = TRUE,
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
  ensurePythonEnv("CppODE", verbose = verbose)

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

  if (verbose) message("  \u2713 ODE and Jacobian generated")

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
    "#include <cppode/cppode.hpp>"
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
    sprintf('extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP, SEXP hiniSEXP, SEXP root_tolSEXP, SEXP maxrootSEXP) {', modelname),
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
        sprintf("  auto controlledStepper = rosenbrock4_controller_pi_ad<rosenbrock4<AD2>, %s>(abstol, reltol);", ifelse(fullErr, "true", "false")),
        "  auto denseStepper = rosenbrock4_dense_output_pi_ad<decltype(controlledStepper)>(controlledStepper);",
        sep = "\n"
      )
      integrate_line <- "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    } else if (deriv) {
      stepper_line <- paste(
        sprintf("  auto controlledStepper = rosenbrock4_controller_pi_ad<rosenbrock4<AD>, %s>(abstol, reltol);", ifelse(fullErr, "true", "false")),
        "  auto denseStepper = rosenbrock4_dense_output_pi_ad<decltype(controlledStepper)>(controlledStepper);",
        sep = "\n"
      )
      integrate_line <- "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    } else {
      stepper_line <- paste(
        "  auto controlledStepper = rosenbrock4_controller_pi<rosenbrock4<double>>(abstol, reltol);",
        "  auto denseStepper = rosenbrock4_dense_output_pi<decltype(controlledStepper)>(controlledStepper);",
        sep = "\n"
      )
      integrate_line <- "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    }
  } else {
    # Controlled stepper version (WITHOUT Interpolation)
    if (deriv2) {
      stepper_line <- sprintf("  auto controlledStepper = rosenbrock4_controller_pi_ad<rosenbrock4<AD2>, %s>(abstol, reltol);", ifelse(fullErr, "true", "false"))
      integrate_line <- "  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    } else if (deriv) {
      stepper_line <- sprintf("  auto controlledStepper = rosenbrock4_controller_pi_ad<rosenbrock4<AD>, %s>(abstol, reltol);", ifelse(fullErr, "true", "false"))
      integrate_line <- "  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    } else {
      stepper_line <- "  auto controlledStepper = rosenbrock4_controller_pi<rosenbrock4<double>>(abstol, reltol);"
      integrate_line <- "  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    }
  }

  externC <- c(externC, "",
               "  // --- Solver setup ---",
               "  double abstol = REAL(abstolSEXP)[0];",
               "  double reltol = REAL(reltolSEXP)[0];",
               "  double root_tol = REAL(root_tolSEXP)[0];",
               "  double hini = REAL(hiniSEXP)[0];",
               "  int maxroot = INTEGER(maxrootSEXP)[0];",
               "  ode_system sys(full_params);",
               "  jacobian jac(full_params);",
               "  observer obs(result_times, y);",
               stepper_line, "",
               "  // --- Determine dt ---",
               sprintf("  %s dt;", numType),
               "  if (hini == 0.0) {",
               sprintf("    dt = odeint_utils::estimate_initial_dt_local(sys, jac, x, times.front(), abstol, reltol);"),
               "  } else {",
               "    dt = hini;",
               "  }",
               "",
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
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i];",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      variable_out[IDX(i, s)] = y[i * %d + s];", n_variables),
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
                 ""
    )

    # Extract with fixed parameter handling
    if (length(fixed_initial_idx) > 0 || length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   "  for (int i = 0; i < n_out; ++i) {",
                   "    time_out[i] = result_times[i].x();",
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                   "      variable_out[IDX_variable(i, s)] = xi.x();",
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
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                   "      variable_out[IDX_variable(i, s)] = xi.x();",
                   sprintf("      for (int v = 0; v < %d; ++v) {", n_total_sens),
                   "        sens1_out[IDX_sens1(i, s, v)] = xi.d(v);",
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
                 ""
    )

    # Extract with fixed parameter handling
    if (length(fixed_initial_idx) > 0 || length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   "  for (int i = 0; i < n_out; ++i) {",
                   "    time_out[i] = result_times[i].x().x();",
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                   "      variable_out[IDX_variable(i, s)] = xi.x().x();",
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
                   "          sens1_out[IDX_sens1(i, s, v1_sens)] = xi.d(v1).x();",
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
                   sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                   sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                   "      variable_out[IDX_variable(i, s)] = xi.x().x();",
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
  if (!dir.exists(outdir)) {
    stop("outdir does not exist: ", outdir)
  }

  filename <- file.path(outdir, paste0(modelname, ".cpp"))

  # Warn if file already exists
  if (file.exists(filename)) {
    message("Overwriting existing file: ", normalizePath(filename, winslash = "/", mustWork = FALSE))
  }

  cpp_text <- c(
    paste0("/** Code auto-generated by CppODE ", as.character(utils::packageVersion("CppODE")), " **/"),
    "",
    includings,
    "",
    usings,
    "",
    ode_code,
    "",
    jac_code,
    "",
    observer_code,
    "",
    externC
  )

  writeLines(cpp_text, filename, useBytes = TRUE)

  if (verbose) message("Wrote: ", normalizePath(filename, winslash = "/", mustWork = FALSE))

  # --- Attach attributes ---
  jac_matrix_R <- matrix(unlist(jac_matrix_str), nrow = n_variables, ncol = n_variables, byrow = TRUE)
  dimnames(jac_matrix_R) <- list(variables, variables)

  attr(modelname, "equations")     <- rhs
  attr(modelname, "srcfile")       <- normalizePath(filename, winslash = "/", mustWork = FALSE)
  attr(modelname, "variables")     <- variables
  attr(modelname, "parameters")    <- params
  attr(modelname, "events")        <- events
  attr(modelname, "fixed")         <- c(fixed_initials, fixed_params)
  attr(modelname, "jacobian")      <- list(f.x = jac_matrix_R, f.time = time_derivs_str)
  attr(modelname, "deriv")         <- deriv
  attr(modelname, "deriv2")        <- deriv2

  # Dimension names
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


#' Generate a R/C++ evaluator for algebraic models with optional Jacobian/Hessian
#'
#' @description
#' `funCpp()` takes a named vector/list of algebraic expressions and returns an evaluation
#' function that can compute model outputs and, optionally, symbolic Jacobians and Hessians.
#' It supports two backends:
#' - A compiled C++ backend generated via a small Python helper (recommended for performance).
#' - A pure-R fallback that evaluates parsed R expressions.
#'
#' The function automatically:
#' - Validates and aligns variable/parameter inputs.
#' - Computes symbolic derivatives (via `derivSymb()`), if requested.
#' - Normalizes derivative arrays so that their row/column orders match the internal
#'   differentiation order used throughout the evaluator.
#'
#' @param eqns Named character vector or list of expressions (e.g. `c(A="k_p*(k2+k_d)/(k1*k_d)", B="k_p/k_d")`).
#'           Names become output column names. If `names(eqns)` is `NULL`, default names `f1, f2, ...` are used.
#' @param variables Character vector of variable names that appear in `eqns` and are supplied per observation.
#'                  If `NULL` or length 0, the model is treated as purely parametric (no per-row variables).
#' @param parameters Character vector of parameter names used by the model.
#' @param fixed Optional character vector of symbols that should be treated as parameters at evaluation time,
#'              even if they originally appear in `variables`. Useful for "fixing" variables temporarily.
#' @param compile Logical. If `TRUE`, compiles and loads the generated C++ code.
#' @param modelname Optional base name for the generated C++ source file
#'   \emph{and} for all generated C/C++ symbols (e.g. \code{modelname_eval},
#'   \code{modelname_jacobian}) and the resulting shared library.
#'   If \code{NULL}, a random identifier is used.
#' @param outdir Directory where generated C++ source files are written. Defaults to `tempdir()`.
#' @param verbose Logical; if `TRUE`, prints progress messages.
#' @param warnings Logical; if `TRUE`, prints soft warnings about missing inputs filled with zeros.
#' @param convenient Logical; if `TRUE`, returns a wrapper so you can call the function using
#'                   `f(var1 = ..., var2 = ..., k1 = ..., k2 = ...)` style arguments without
#'                   manually assembling matrices/vectors.
#' @param deriv Logical; if `TRUE`, compute and return the Jacobian. Required if `deriv2 = TRUE`.
#' @param deriv2 Logical; if `TRUE`, compute and return the Hessian in addition to the Jacobian.
#'
#' @return A function with signature
#'   \preformatted{
#'   f(vars, params = numeric(0), attach.input = FALSE, deriv = TRUE, deriv2 = FALSE, verbose = FALSE)
#'   }
#' where:
#' - `vars`: numeric matrix/data frame whose columns match `variables` (or `NULL` if none).
#' - `params`: named numeric vector containing all `parameters` (missing entries are filled with 0 if allowed).
#' - `attach.input`: if `TRUE`, input variables are prepended to the returned `out` matrix.
#' - `deriv`, `deriv2`: request derivatives at call time (must be compatible with how `f` was created).
#'
#' The returned list has elements:
#' - `out`: numeric matrix of outputs with columns named as in `eqns`.
#' - `jacobian`: 3D array `[n_obs, n_out, n_diff_syms]` if `deriv=TRUE`.
#' - `hessian`: 4D array `[n_obs, n_out, n_diff_syms, n_diff_syms]` if `deriv2=TRUE`.
#'
#' Attributes on the returned function include:
#' - `equations`, `variables`, `parameters`, `fixed`, `modelname`
#' - `jacobian.symb` (if `deriv=TRUE`)
#' - `hessian.symb`  (if `deriv2=TRUE`)
#'
#' @section Derivative ordering:
#' Derivative dimensions always follow `diff_syms = c(variables, setdiff(parameters, fixed))`.
#' Internally, the symbolic Jacobian/Hessian matrices returned by `derivSymb()` are reindexed
#' to this order to ensure compiled and R-fallback paths produce consistent layouts.
#'
#' @section Array layout:
#' Arrays are stored with observations in the first dimension for better memory locality:
#' - Jacobian: `[n_obs, n_out, n_diff_syms]`
#' - Hessian: `[n_obs, n_out, n_diff_syms, n_diff_syms]`
#' This allows efficient subsetting like `jacobian[1:10, , ]` for the first 10 observations.
#'
#' @example inst/examples/example_fun.R
#' @importFrom reticulate import_from_path
#' @export
funCpp <- function(eqns, variables  = getSymbols(eqns, exclude = parameters),
                   parameters = NULL, fixed = NULL,
                   modelname = NULL, outdir = tempdir(),
                   compile    = FALSE, verbose = FALSE, warnings = TRUE,
                   convenient = TRUE, deriv = TRUE, deriv2 = FALSE) {

  if (deriv2 && !deriv) {
    warning("deriv2 = TRUE requires deriv = TRUE. Setting deriv = TRUE automatically.")
    deriv <- TRUE
  }

  outnames <- names(eqns)
  if (is.null(outnames))
    outnames <- paste0("f", seq_along(eqns))

  if (!is.null(fixed)) {
    variables  <- setdiff(variables, fixed)
    parameters <- union(parameters, fixed)
  }

  innames     <- variables
  diff_params <- setdiff(parameters, fixed)
  diff_syms   <- c(variables, diff_params)

  if (!dir.exists(outdir))
    stop("outdir does not exist: ", outdir)

  if (is.null(modelname))
    modelname <- paste(c("f", sample(c(letters, 0:9), 8, TRUE)), collapse = "")

  checkArguments <- function(M, p) {
    # Determine n_obs from input matrix even if no variables are needed
    n_obs <- if (!is.null(M) && (is.matrix(M) || is.data.frame(M))) nrow(M)
    else if (!is.null(M) && is.vector(M) && !is.list(M)) length(M) / max(length(innames), 1)
    else 1
    n_obs <- as.integer(max(n_obs, 1))

    if (is.null(innames) || length(innames) == 0) {
      M <- matrix(0, nrow = n_obs, ncol = 0)
    } else {
      if (is.null(M)) stop("Variables defined but 'vars' is NULL.")
      if (is.vector(M) && !is.list(M)) {
        M <- matrix(M, ncol = length(innames))
        colnames(M) <- innames
      }
      if (is.null(colnames(M))) {
        if (ncol(M) != length(innames))
          stop("vars has no column names and incorrect length.")
        colnames(M) <- innames
      }
      if (!all(innames %in% colnames(M))) {
        missing <- setdiff(innames, colnames(M))
        stop("Missing variable columns: ", paste(missing, collapse = ", "))
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
        missing <- setdiff(parameters, names(p))
        stop("Missing parameters: ", paste(missing, collapse = ", "))
      }
      p <- p[parameters]
    }
    list(M = M, p = p, n_obs = n_obs)
  }

  sym_jac  <- NULL
  sym_hess <- NULL
  if (deriv || deriv2) {
    ds <- derivSymb(eqns, deriv2 = deriv2, real = TRUE, fixed = fixed, verbose = verbose)
    sym_jac  <- ds$jacobian
    sym_hess <- ds$hessian
  }

  if (!is.null(sym_jac)) {
    if (is.null(rownames(sym_jac))) rownames(sym_jac) <- outnames
    if (!identical(colnames(sym_jac), diff_syms)) {
      sym_jac <- sym_jac[, diff_syms, drop = FALSE]
    }
  }

  if (!is.null(sym_hess)) {
    for (nm in names(sym_hess)) {
      H <- sym_hess[[nm]]
      if (!identical(rownames(H), diff_syms) || !identical(colnames(H), diff_syms)) {
        sym_hess[[nm]] <- H[diff_syms, diff_syms, drop = FALSE]
      }
    }
  }

  replaceHeaviside <- function(expr_str) {
    if (is.null(expr_str) || expr_str == "0") return(expr_str)
    gsub("Heaviside\\(([^)]+)\\)", "ifelse(\\1 >= 0, 1, 0)", expr_str)
  }

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

  parsed_exprs <- lapply(eqns, function(e) safe_parse(e)[[1]])

  parsed_jac <- NULL
  if (!is.null(sym_jac)) {
    parsed_jac <- matrix(vector("list", nrow(sym_jac) * ncol(sym_jac)),
                         nrow = nrow(sym_jac), ncol = ncol(sym_jac),
                         dimnames = dimnames(sym_jac))
    for (i in seq_len(nrow(sym_jac)))
      for (j in seq_len(ncol(sym_jac)))
        parsed_jac[[i, j]] <- safe_parse(sym_jac[i, j])
  }

  parsed_hess <- NULL
  if (!is.null(sym_hess)) {
    parsed_hess <- lapply(names(sym_hess), function(fname) {
      H <- sym_hess[[fname]]
      parsed <- matrix(vector("list", nrow(H) * ncol(H)),
                       nrow = nrow(H), ncol = ncol(H),
                       dimnames = dimnames(H))
      for (i in seq_len(nrow(H)))
        for (j in seq_len(ncol(H)))
          parsed[[i, j]] <- safe_parse(H[i, j])
      parsed
    })
    names(parsed_hess) <- names(sym_hess)
  }

  if (!fallback_available)
    warning("R fallback not available. Please compile the function.")

  cppfile <- NULL
  if (!is.null(modelname)) {
    ensurePythonEnv(envname = "CppODE", verbose = verbose)
    pytools <- reticulate::import_from_path(
      "generatefunCppCode",
      path = system.file("python", package = "CppODE")
    )

    exprs_list <- as.list(eqns)
    names(exprs_list) <- outnames

    py_jac <- if (!is.null(sym_jac))
      lapply(seq_len(nrow(sym_jac)), function(i) as.list(as.character(sym_jac[i, ])))
    else NULL
    if (!is.null(py_jac)) names(py_jac) <- rownames(sym_jac)

    py_hess <- if (!is.null(sym_hess))
      lapply(seq_len(length(sym_hess)), function(i)
        lapply(seq_len(nrow(sym_hess[[i]])), function(j)
          as.list(as.character(sym_hess[[i]][j, ]))))
    else NULL
    if (!is.null(py_hess)) names(py_hess) <- names(sym_hess)

    gen_variables  <- variables
    gen_parameters <- parameters

    pytools$generate_fun_cpp(
      exprs      = exprs_list,
      variables  = if (length(gen_variables)  > 0) gen_variables  else list(),
      parameters = if (length(gen_parameters) > 0) gen_parameters else list(),
      jacobian   = py_jac,
      hessian    = py_hess,
      modelname  = modelname,
      outdir     = normalizePath(outdir, winslash = "/", mustWork = FALSE)
    )
  }

  myRfun <- function(vars, params = numeric(0),
                     attach.input = FALSE,
                     deriv = TRUE, deriv2 = FALSE,
                     fixed = NULL,
                     verbose = FALSE) {

    if (deriv2 && !deriv) {
      deriv <- TRUE
    }

    checked <- checkArguments(vars, params)
    M <- checked$M; p <- checked$p; n_obs <- checked$n_obs

    fixed_runtime <- if (is.null(fixed)) character(0) else intersect(fixed, parameters)
    diff_syms_dyn <- setdiff(diff_syms, fixed_runtime)

    funsym_eval <- paste0(modelname, "_eval")
    funsym_jac  <- paste0(modelname, "_jacobian")
    funsym_hess <- paste0(modelname, "_hessian")
    sofile <- paste0(modelname, .Platform$dynlib.ext)

    compiled_available <- file.exists(sofile) && is.loaded(funsym_eval)

    result <- list()

    if (compiled_available) {
      xvec <- as.double(as.vector(M))
      yvec <- double(length(outnames) * n_obs)
      n <- as.integer(n_obs)
      k <- as.integer(length(innames))
      l <- as.integer(length(outnames))
      out <- .C(funsym_eval, x = xvec, y = yvec, p = as.double(p),
                n = n, k = k, l = l)
      res <- matrix(out$y, nrow = n_obs, ncol = length(outnames), byrow = TRUE)
      colnames(res) <- outnames
    } else {
      res <- matrix(NA_real_, nrow = n_obs, ncol = length(outnames))
      colnames(res) <- outnames
      for (i in seq_len(n_obs)) {
        env <- as.list(c(as.numeric(M[, i]), p))
        names(env) <- c(innames, parameters)
        vals <- vapply(parsed_exprs, function(expr) eval(expr, envir = env), numeric(1))
        res[i, ] <- vals
      }
    }

    if (attach.input && ncol(M) > 0)
      res <- cbind(res, t(M))

    result[["out"]] <- res

    # ===================================================================
    # JACOBIAN: Array layout [n_obs, n_out, n_symbols]
    # ===================================================================
    if (deriv && !is.null(sym_jac)) {
      n_out <- length(outnames); n_sym <- length(diff_syms)

      if (compiled_available && is.loaded(funsym_jac)) {
        # Compiled C++ path
        xvec <- as.double(as.vector(M))
        jac_vec <- double(n_obs * n_out * n_sym)
        n <- as.integer(n_obs); k <- as.integer(length(innames)); l <- as.integer(n_out)
        out_jac <- .C(funsym_jac, x = xvec, jac = jac_vec, p = as.double(p),
                      n = n, k = k, l = l)
        # New layout: [n_obs, n_out, n_symbols]
        jac_arr <- array(out_jac$jac, dim = c(n_obs, n_out, n_sym),
                         dimnames = list(NULL, outnames, diff_syms))
      } else {
        # R fallback path
        jac_arr <- array(0, dim = c(n_obs, n_out, n_sym),
                         dimnames = list(NULL, outnames, diff_syms))
        for (obs_i in seq_len(n_obs)) {
          env <- as.list(c(as.numeric(M[, obs_i]), p))
          names(env) <- c(innames, parameters)
          for (out_i in seq_len(n_out)) {
            for (sym_i in seq_len(n_sym)) {
              sym <- diff_syms[sym_i]
              if (sym %in% fixed_runtime) {
                jac_arr[obs_i, out_i, sym_i] <- 0
                next
              }
              expr <- parsed_jac[[outnames[out_i], sym]][[1]]
              jac_arr[obs_i, out_i, sym_i] <-
                if (is.null(expr)) 0 else eval(expr, envir = env)
            }
          }
        }
      }

      # Subset to dynamic symbols (excluding runtime-fixed)
      jac_arr <- jac_arr[, , diff_syms_dyn, drop = FALSE]
      result[["jacobian"]] <- jac_arr
    }

    # ===================================================================
    # HESSIAN: Array layout [n_obs, n_out, n_symbols, n_symbols]
    # ===================================================================
    if (deriv2 && !is.null(sym_hess)) {
      n_out <- length(outnames); n_sym <- length(diff_syms)

      if (compiled_available && is.loaded(funsym_hess)) {
        # Compiled C++ path
        xvec <- as.double(as.vector(M))
        hess_vec <- double(n_obs * n_out * n_sym * n_sym)
        n <- as.integer(n_obs); k <- as.integer(length(innames)); l <- as.integer(n_out)
        out_hess <- .C(funsym_hess, x = xvec, hess = hess_vec, p = as.double(p),
                       n = n, k = k, l = l)
        # New layout: [n_obs, n_out, n_symbols, n_symbols]
        hes_arr <- array(out_hess$hess, dim = c(n_obs, n_out, n_sym, n_sym),
                         dimnames = list(NULL, outnames, diff_syms, diff_syms))
      } else {
        # R fallback path
        hes_arr <- array(0, dim = c(n_obs, n_out, n_sym, n_sym),
                         dimnames = list(NULL, outnames, diff_syms, diff_syms))
        for (obs_i in seq_len(n_obs)) {
          env <- as.list(c(as.numeric(M[, obs_i]), p))
          names(env) <- c(innames, parameters)
          for (out_i in seq_len(n_out)) {
            Hmat <- parsed_hess[[outnames[out_i]]]
            for (sym_i in seq_len(n_sym)) {
              for (sym_j in seq_len(n_sym)) {
                si <- diff_syms[sym_i]; sj <- diff_syms[sym_j]
                if (si %in% fixed_runtime || sj %in% fixed_runtime) {
                  hes_arr[obs_i, out_i, sym_i, sym_j] <- 0
                  next
                }
                expr <- Hmat[[si, sj]][[1]]
                hes_arr[obs_i, out_i, sym_i, sym_j] <-
                  if (is.null(expr)) 0 else eval(expr, envir = env)
              }
            }
          }
        }
      }

      # Subset to dynamic symbols (excluding runtime-fixed)
      hes_arr <- hes_arr[, , diff_syms_dyn, diff_syms_dyn, drop = FALSE]
      result[["hessian"]] <- hes_arr
    }

    return(result)
  }

  outfn <- myRfun

  if (convenient) {
    outfn <- function(..., attach.input = FALSE,
                      deriv = TRUE, deriv2 = FALSE,
                      fixed = NULL) {
      arglist <- list(...)
      M <- if (!is.null(innames) && length(innames) > 0)
        do.call(cbind, arglist[innames]) else NULL
      p <- if (!is.null(parameters) && length(parameters) > 0)
        do.call(c, arglist[parameters]) else numeric(0)
      myRfun(M, p, attach.input = attach.input,
             deriv = deriv, deriv2 = deriv2,
             fixed = fixed)
    }
  }

  attr(outfn, "equations")     <- eqns
  attr(outfn, "variables")     <- variables
  attr(outfn, "parameters")    <- parameters
  attr(outfn, "fixed")         <- fixed
  attr(outfn, "modelname")     <- modelname
  attr(outfn, "srcfile")       <- normalizePath(file.path(outdir, paste0(modelname, ".cpp")), winslash = "/", mustWork = FALSE)
  if (deriv && !is.null(sym_jac))
    attr(outfn, "jacobian.symb") <- sym_jac
  if (deriv2 && !is.null(sym_hess))
    attr(outfn, "hessian.symb")  <- sym_hess

  if (compile) {
    compile(outfn, verbose = verbose)
  }

  outfn
}


#' Compile generated C++ model code
#'
#' @description
#' Compiles one or more C++ source files generated by \code{CppODE()} or
#' \code{funCpp()} into shared libraries (\code{.so}, \code{.dll}) using
#' \command{R CMD SHLIB}. Source files are located via the \code{"srcfile"}
#' attribute attached to the supplied objects.
#'
#' Compilation is performed in the directory of each source file and does
#' not write to the user's home directory or working directory unless
#' explicitly requested by the user via the \code{outdir} argument when
#' generating the model code.
#'
#' The function automatically applies platform-appropriate compiler flags
#' (e.g. C++20 standard, optimization, position-independent code where
#' required) and includes headers shipped with the package itself as well as
#' Boost headers provided by the \pkg{BH} package.
#'
#' On Windows, compilation is always performed sequentially since
#' fork-based parallelism is unavailable.
#'
#' @param ... One or more objects returned by \code{CppODE()} or
#'   \code{funCpp()} that carry a \code{"srcfile"} attribute pointing to an
#'   existing C or C++ source file.
#' @param output Optional base name for a combined shared library. If
#'   supplied, all provided source files are compiled and linked into a
#'   single shared object.
#' @param args Optional additional compiler or linker arguments passed to
#'   \command{R CMD SHLIB}.
#' @param cores Number of parallel compilation jobs on Unix-like systems.
#'   Ignored on Windows.
#' @param verbose Logical; if \code{TRUE}, show compiler commands and output.
#'
#' @return
#' Invisibly returns \code{TRUE} on successful compilation.
#'
#' @keywords internal
#' @import BH
compile <- function(..., output = NULL, args = NULL, cores = 1, verbose = FALSE) {
  objects <- list(...)

  # --- collect all source files via srcfile attribute ---
  files <- character()
  for (obj in objects) {
    src <- attr(obj, "srcfile")
    if (!is.null(src) && file.exists(src)) {
      files <- union(files, normalizePath(src, winslash = "/", mustWork = FALSE))
    } else if (verbose) {
      message("No valid srcfile attribute found.")
    }
  }

  if (length(files) == 0)
    stop("No valid C/C++ source files found for compilation.")

  # Roots and directories
  roots <- sub("\\.[^.]+$", "", basename(files))
  dirs  <- dirname(files)
  .so   <- .Platform$dynlib.ext

  # --- Clean up old compiled files (same directories as sources) ---
  for (i in seq_along(files)) {
    so_file <- file.path(dirs[i], paste0(roots[i], .so))
    o_file  <- file.path(dirs[i], paste0(roots[i], ".o"))

    # ALWAYS try to unload first (before checking file existence)
    tryCatch(
      dyn.unload(so_file),
      error = function(e) {
        if (verbose) message("Note: Could not unload ", so_file, ": ", e$message)
      }
    )

    # Then delete old files
    if (file.exists(o_file)) {
      unlink(o_file, force = TRUE)
    }
    if (file.exists(so_file)) {
      unlink(so_file, force = TRUE)
      if (file.exists(so_file))
        stop("Could not delete old shared library: ", so_file)
    }
  }

  # --- Compiler flags ---
  if (Sys.info()[["sysname"]] == "Windows") cores <- 1

  include_flags <- paste(
    paste0("-I", system.file("include", package = "CppODE")),
    paste0("-I", system.file("include", package = "BH"))
  )

  sys <- Sys.info()[["sysname"]]
  cxxflags <- if (sys == "Windows") {
    "-std=c++20 -O3 -DNDEBUG -w"
  } else if (sys == "Linux") {
    "-std=c++20 -O3 -DNDEBUG -fPIC -fno-var-tracking-assignments -w"
  } else if (sys == "Darwin") {
    "-std=c++20 -O3 -DNDEBUG -fPIC -w"
  }

  # --- compile one file ---
  compile_one <- function(src, root, dir) {
    old_cppflags <- Sys.getenv("PKG_CPPFLAGS", unset = NA)
    old_cxxflags <- Sys.getenv("PKG_CXXFLAGS", unset = NA)

    Sys.setenv(
      PKG_CPPFLAGS = include_flags,
      PKG_CXXFLAGS = cxxflags
    )

    on.exit({
      if (is.na(old_cppflags)) Sys.unsetenv("PKG_CPPFLAGS")
      else Sys.setenv(PKG_CPPFLAGS = old_cppflags)
      if (is.na(old_cxxflags)) Sys.unsetenv("PKG_CXXFLAGS")
      else Sys.setenv(PKG_CXXFLAGS = old_cxxflags)
    })

    oldwd <- getwd()
    on.exit(setwd(oldwd), add = TRUE)
    setwd(dir)

    cmd <- paste0(
      shQuote(file.path(R.home("bin"), "R")),
      " CMD SHLIB ",
      shQuote(basename(src)),
      if (!is.null(args)) paste(" ", args) else ""
    )

    if (verbose) message(cmd)
    system(cmd, intern = !verbose)

    so_file <- file.path(dir, paste0(root, .so))
    if (!file.exists(so_file))
      stop("Compilation failed for ", src)

    dyn.load(so_file)
    if (verbose) message("\u2713 Loaded ", so_file)
    invisible(root)
  }

  # --- Compilation ---
  if (is.null(output)) {
    if (verbose) message("Compiling ", length(files), " model(s)...")
    parallel::mclapply(
      seq_along(files),
      function(i) compile_one(files[i], roots[i], dirs[i]),
      mc.cores = cores,
      mc.silent = !verbose
    )
  } else {
    # --- Combine all into one shared library ---
    output <- sub("\\.so$", "", output)
    outdir <- dirname(files[1])
    output_so <- file.path(outdir, paste0(output, .so))

    try(dyn.unload(output_so), silent = TRUE)
    if (file.exists(output_so)) unlink(output_so)

    old_cppflags <- Sys.getenv("PKG_CPPFLAGS", unset = NA)
    old_cxxflags <- Sys.getenv("PKG_CXXFLAGS", unset = NA)

    Sys.setenv(
      PKG_CPPFLAGS = include_flags,
      PKG_CXXFLAGS = cxxflags
    )

    on.exit({
      if (is.na(old_cppflags)) Sys.unsetenv("PKG_CPPFLAGS")
      else Sys.setenv(PKG_CPPFLAGS = old_cppflags)
      if (is.na(old_cxxflags)) Sys.unsetenv("PKG_CXXFLAGS")
      else Sys.setenv(PKG_CXXFLAGS = old_cxxflags)
    })

    oldwd <- getwd()
    on.exit(setwd(oldwd), add = TRUE)
    setwd(outdir)

    cmd <- paste0(
      shQuote(file.path(R.home("bin"), "R")),
      " CMD SHLIB ",
      paste(shQuote(basename(files)), collapse = " "),
      " -o ",
      shQuote(basename(output_so)),
      if (!is.null(args)) paste(" ", args) else ""
    )

    if (verbose) message(cmd)
    system(cmd, intern = !verbose)

    if (!file.exists(output_so))
      stop("Compilation failed for combined output")

    dyn.unload(output_so)
    dyn.load(output_so)
    if (verbose) message("\u2713 Loaded ", output_so)
  }

  invisible(TRUE)
}
