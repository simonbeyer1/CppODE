#' Generate C++ code for ODE models with events and parameter sensitivities
#'
#' @description
#' This function generates and compiles a C++ solver for systems of ordinary differential
#' equations (ODEs) of the form
#'
#' \deqn{\dot{x}(t) = f\big(x(t), p_{\text{dyn}}\big), \quad x(t_0) = p_{\text{init}}}
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
#' ## Root function (rootfunc)
#'
#' The `rootfunc` argument enables integration termination based on root-finding:
#'
#' - **`"equilibrate"`**: Stops integration when the system reaches steady state.
#'   The steady-state condition checks that all derivatives (including sensitivities
#'   when `deriv = TRUE` or `deriv2 = TRUE`) fall below the `roottol` tolerance.
#'   This is useful for equilibrating systems before further analysis.
#'
#' - **Character vector of expressions**: Similar to deSolve's `rootfunc`, you can
#'   specify one or more expressions (e.g., `"x - 0.5"` or `c("x - 0.5", "y - 1")`).
#'   Integration stops when any expression crosses zero. Variables, parameters, and
#'   `time` can be used in expressions.
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
#' @param rootfunc Optional root function specification for integration termination.
#'   Either `"equilibrate"` for steady-state detection, or a character vector of
#'   expressions that trigger termination when crossing zero. Default: `NULL`.
#' @param forcings Character vector of forcing function names used in `rhs`.
#' @param fixed Character vector of fixed initial conditions or parameters (excluded from sensitivities).
#' @param compile Logical. If `TRUE`, compiles and loads the generated C++ code.
#' @param modelname Optional base name for the generated C++ source file
#'   \emph{and} for all generated C/C++ symbols (e.g. \code{solve_<modelname>})
#'   as well as the resulting shared library.
#'   If \code{NULL}, a random identifier is used.
#' @param outdir Directory where generated C++ source files are written. Defaults to `tempdir()`.
#' @param deriv Logical. If `TRUE`, enable first-order sensitivities via dual numbers.
#' @param deriv2 Logical. If `TRUE`, enable second-order sensitivities via nested dual numbers; requires `deriv = TRUE`.
#' @param fullErr Logical. If `TRUE`, compute error estimates using full state vector including derivatives. If `FALSE`, use only the value components for error control.
#' @param includeTimeZero Logical. If `TRUE`, ensure that time `0` is included among integration times.
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
#' | `rootfunc` | `character` | Root function specification (if any) |
#' | `solver` | `list` | Description of the numerical solver configuration |
#' | `fixed` | `character` | Names of fixed initial conditions or parameters |
#' | `jacobian` | `eqnvec` | Symbolic expressions for the system Jacobian |
#' | `deriv` | `logical` | Indicates whether first-order sensitivities (dual numbers) were used |
#' | `deriv2` | `logical` | Indicates whether nested dual numbers were used for second-order sensitivities |
#' | `dim_names` | `list` | Dimension names for arrays: `time`, `variable`, and `sens` |
#'
#' @example inst/examples/example_ODE.R
#' @importFrom stats setNames
#' @seealso [solveODE()] for a solver interface of compiled models
#' @export
CppODE <- function(rhs, events = NULL, rootfunc = NULL, fixed = NULL, forcings = NULL,
                   compile = TRUE, modelname = NULL, outdir = tempdir(),
                   deriv = TRUE, deriv2 = FALSE, fullErr = TRUE,
                   includeTimeZero = TRUE, useDenseOutput = TRUE,
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

  # Collect all expressions for symbol extraction
  all_expressions <- rhs
  if (!is.null(events)) {
    bad <- which(!xor(!is.na(events$time), !is.na(events$root)))
    if (length(bad) > 0) {
      stop(
        sprintf(
          "Each event must define exactly one of 'time' or 'root'. Invalid event(s): %s",
          paste(bad, collapse = ", ")
        )
      )
    }
    all_expressions <- c(all_expressions,
                         events$value,
                         if ("time" %in% names(events)) events$time,
                         if ("root" %in% names(events)) events$root)
  }
  # Include rootfunc expressions (but not "equilibrate" which is a keyword)
  if (!is.null(rootfunc) && !identical(tolower(rootfunc), "equilibrate")) {
    all_expressions <- c(all_expressions, rootfunc)
  }

  symbols <- getSymbols(all_expressions)

  # --- Validate forcings ---
  if (is.null(forcings)) forcings <- character(0)

  # Forcings must be symbols in rhs but NOT state names
  if (length(forcings) > 0) {
    unknown_forcings <- setdiff(forcings, symbols)
    if (length(unknown_forcings) > 0) {
      stop("Unknown forcing symbols: ", paste(unknown_forcings, collapse = ", "))
    }
    forcing_states <- intersect(forcings, variables)
    if (length(forcing_states) > 0) {
      stop("Forcing names cannot be state variables: ", paste(forcing_states, collapse = ", "))
    }
  }
  n_forcings <- length(forcings)

  # Parameters are symbols that are not variables, forcings, or time

  params  <- setdiff(symbols, c(variables, forcings, "time"))

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

  # Lazy import
  codegen <- get_codegenCppODE_py()

  if (verbose) message("Generating ODE and Jacobian code...")

  # Single Python call generates everything
  if (deriv2) {
    numType <- "AD2"  # F<F<double>>
  } else if (deriv) {
    numType <- "AD"   # F<double>
  } else {
    numType <- "double"
  }

  codegen_result <- codegen$generate_ode_cpp(
    rhs_dict = as.list(setNames(rhs, variables)),
    params_list = params,
    num_type = numType,
    fixed_states = fixed_initials,
    fixed_params = fixed_params,
    forcings_list = forcings
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

    event_lines <- codegen$generate_event_code(
      events_df = events,
      states_list = variables,
      params_list = params,
      n_states = n_variables,
      num_type = numType,
      forcings_list = forcings
    )

    event_code <- paste(event_lines, collapse = "\n")
  }

  # --- Generate rootfunc code if needed ---
  rootfunc_code <- ""
  if (!is.null(rootfunc)) {
    if (verbose) message("Generating rootfunc code...")

    rootfunc_lines <- codegen$generate_rootfunc_code(
      rootfunc = rootfunc,
      states_list = variables,
      params_list = params,
      n_states = n_variables,
      num_type = numType,
      forcings_list = forcings
    )

    rootfunc_code <- paste(rootfunc_lines, collapse = "\n")
    if (verbose) message("  \u2713 rootfunc generated")
  }

  # --- Generate forcing initialization code ---
  forcing_init_code <- paste(codegen$generate_forcing_init_code(n_forcings, numType), collapse = "\n")

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
    sprintf(
      'extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP sens1iniSEXP, SEXP sens2iniSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP, SEXP hiniSEXP, SEXP root_tolSEXP, SEXP maxrootSEXP, SEXP forcingTimesSEXP, SEXP forcingValuesSEXP) {',
      modelname
    ),
    "try {",
    "",
    "  StepChecker checker(INTEGER(maxprogressSEXP)[0], INTEGER(maxstepsSEXP)[0]);",
    "",
    sprintf("  ublas::vector<%s> x(%d);", numType, n_variables),
    sprintf("  ublas::vector<%s> full_params(%d);", numType, n_variables + n_params),
    ""
  )

  # --- Custom sensitivity initial values ---
  if (deriv) {
    externC <- c(
      externC,
      "  // Custom sensitivity initial values",
      "  bool has_sens1ini = !Rf_isNull(sens1iniSEXP);",
      "  double* sens1ini = has_sens1ini ? REAL(sens1iniSEXP) : nullptr;"
    )

    externC <- c(
      externC,
      "  if (has_sens1ini) {",
      sprintf(
        "    if (Rf_length(sens1iniSEXP) != %d * %d)",
        n_variables, n_total_sens
      ),
      "      Rf_error(\"sens1ini has wrong length\");",
      "  }"
    )

    if (deriv2) {
      externC <- c(
        externC,
        "  bool has_sens2ini = !Rf_isNull(sens2iniSEXP);",
        "  double* sens2ini = has_sens2ini ? REAL(sens2iniSEXP) : nullptr;"
      )

      externC <- c(
        externC,
        "  if (has_sens2ini) {",
        sprintf(
          "    if (Rf_length(sens2iniSEXP) != %d * %d * %d)",
          n_variables, n_total_sens, n_total_sens
        ),
        "      Rf_error(\"sens2ini has wrong length\");",
        "  }"
      )
    } else {
      externC <- c(
        externC,
        "  if (!Rf_isNull(sens2iniSEXP))",
        "    Rf_error(\"sens2ini supplied but deriv2 = FALSE\");"
      )
    }

    externC <- c(externC, "")
  }

  # --- Sensitivity dimensions and index helpers ---
  if (deriv) {
    externC <- c(
      externC,
      sprintf("  const int n_states = %d;", n_variables),
      sprintf("  const int n_sens   = %d;", n_total_sens),
      "  auto IDX1 = [n_states](int s, int v) {",
      "    return s + n_states * v;",
      "  };"
    )

    if (deriv2) {
      externC <- c(
        externC,
        "  auto IDX2 = [n_states, n_sens](int s, int v1, int v2) {",
        "    return s + n_states * (v1 + n_sens * v2);",
        "  };"
      )
    }

    externC <- c(externC, "")
  }

  # --- initialize states ---
  externC <- c(
    externC,
    "  // initialize variables",
    sprintf("  for (int i = 0; i < %d; ++i) {", n_variables),
    "    bool is_fixed = false;",
    "    (void)is_fixed;  // suppress unused warning"
  )

  if (deriv && length(fixed_initial_idx) > 0) {
    externC <- c(
      externC,
      sprintf(
        "    is_fixed = (%s);",
        paste(sprintf("i == %d", fixed_initial_idx), collapse = " || ")
      )
    )
  }

  if (deriv2) {
    externC <- c(
      externC,
      "    x[i].x().x() = REAL(paramsSEXP)[i];",
      "    if (!is_fixed) {",
      "      // First-order sensitivities (inner layer)",
      "      if (has_sens1ini) {",
      "        x[i].x().diff(0, n_sens);  // Allocate first-order array",
      "        for (int v1 = 0; v1 < n_sens; ++v1) {",
      "          x[i].x().d(v1) = sens1ini[IDX1(i, v1)];",
      "        }",
      "      } else {",
      "        x[i].x().diff(i, n_sens);  // Identity: d(i) = 1",
      "      }",
      "      // Second-order sensitivities (outer layer)",
      "      if (has_sens2ini) {",
      "        // Custom second-order initialization",
      "        for (int v1 = 0; v1 < n_sens; ++v1) {",
      "          x[i].diff(v1, n_sens).diff(0, n_sens);  // Allocate second-order array",
      "          for (int v2 = 0; v2 < n_sens; ++v2) {",
      "            x[i].diff(v1, n_sens).d(v2) = sens2ini[IDX2(i, v1, v2)];",
      "          }",
      "        }",
      "      } else {",
      "        // Default: allocate outer layer with zeros",
      "        // This seeds x[i].d(v1) = 0 for all v1, which is required for",
      "        // proper second-order AD propagation",
      "        x[i].diff(i, n_sens);  // Allocate outer layer (inner derivative of outer = 0 by default)",
      "      }",
      "    }"
    )
  } else if (deriv) {
    externC <- c(
      externC,
      "    x[i] = REAL(paramsSEXP)[i];",
      "    if (!is_fixed) {",
      "      if (has_sens1ini) {",
      "        x[i].diff(0, n_sens);  // Allocate",
      "        for (int v = 0; v < n_sens; ++v) {",
      "          x[i].d(v) = sens1ini[IDX1(i, v)];",
      "        }",
      "      } else {",
      "        x[i].diff(i, n_sens);  // Identity: d(i) = 1",
      "      }",
      "    }"
    )
  } else {
    externC <- c(externC, "    x[i] = REAL(paramsSEXP)[i];")
  }

  externC <- c(
    externC,
    "    full_params[i] = x[i];",
    "  }",
    "",
    "  // initialize parameters",
    sprintf("  for (int i = 0; i < %d; ++i) {", n_params),
    sprintf("    int param_index = %d + i;", n_variables),
    "    bool is_fixed = false;",
    "    (void)is_fixed;  // suppress unused warning"
  )

  if (deriv && length(fixed_param_idx) > 0) {
    externC <- c(
      externC,
      sprintf(
        "    is_fixed = (%s);",
        paste(sprintf("i == %d", fixed_param_idx), collapse = " || ")
      )
    )
  }

  if (deriv2) {
    externC <- c(
      externC,
      "    int sens_idx = n_states + i;",
      "    full_params[param_index].x().x() = REAL(paramsSEXP)[param_index];",
      "    if (!is_fixed) {",
      "      // First-order (inner layer): Parameters use identity matrix dp_i/dp_j = delta_{ij}",
      "      full_params[param_index].x().diff(sens_idx, n_sens);  // Sets d(sens_idx) = 1",
      "      // Second-order (outer layer): d^2 p_i/dp_j dp_k = 0 (parameters are constant)",
      "      // But we still need to allocate the outer layer for AD propagation",
      "      full_params[param_index].diff(sens_idx, n_sens);  // Allocate outer layer (inner values default to 0)",
      "    }"
    )
  } else if (deriv) {
    externC <- c(
      externC,
      "    int sens_idx = n_states + i;",
      "    full_params[param_index] = REAL(paramsSEXP)[param_index];",
      "    if (!is_fixed) {",
      "      for (int v = 0; v < n_sens; ++v) {",
      "        // Parameters use identity matrix: dp_i/dp_j = delta_{ij}",
      "        // (sens1ini only provides state sensitivities, not parameter sensitivities)",
      "        double seed = (v == sens_idx ? 1.0 : 0.0);",
      "        if (seed != 0.0) {",
      "          full_params[param_index].diff(v, n_sens);",
      "          full_params[param_index].d(v) *= seed;",
      "        }",
      "      }",
      "    }"
    )
  } else {
    externC <- c(externC, "    full_params[param_index] = REAL(paramsSEXP)[param_index];")
  }

  externC <- c(externC, "  }", "")


  # --- Forcing Initialization ---
  externC <- c(externC, forcing_init_code, "")

  externC <- c(externC,
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
               sprintf("  std::vector<FixedEvent<ublas::vector<%s>, %s>> fixed_events;", numType, numType),
               sprintf("  std::vector<RootEvent<ublas::vector<%s>, %s>> root_events;", numType, numType)
  )

  # Insert event code from Python
  if (event_code != "") {
    externC <- c(externC, event_code)
  }

  # Note: rootfunc_code is inserted later, after sys is defined

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
               "  ode_system sys(full_params, F);",
               "  jacobian jac(full_params, F);",
               "  observer obs(result_times, y);")

  # Insert rootfunc code from Python (after sys is defined, needed for equilibrate)
  if (rootfunc_code != "") {
    externC <- c(externC, rootfunc_code)
  }

  externC <- c(externC,
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
    "namespace {",
    ode_code,
    "",
    jac_code,
    "",
    observer_code,
    "",
    "}",
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
  attr(modelname, "forcings")      <- forcings
  attr(modelname, "events")        <- events
  attr(modelname, "rootfunc")      <- rootfunc
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


#' Solver Interface for Ordinary Differential Equation Models
#'
#' @description
#' Numerically integrates a compiled ODE model created by [CppODE()] over a
#' specified time span.
#'
#' @param model A compiled ODE model object returned by [CppODE()].
#' @param times Numeric vector of time points at which to return the solution.
#'   Must be non-empty and contain finite values.
#' @param parms Named numeric vector containing initial conditions and parameters.
#'   The order must match `c(attr(model, "variables"), attr(model, "parameters"))`.
#'   Names are checked against the model specification.
#' @param sens1ini Optional named numeric vector of initial values for first-order
#'   sensitivities. Names must match `attr(model, "dim_names")$sens`.
#'   If `NULL`, default identity seeding is used.
#' @param sens2ini Optional numeric vector of initial values for second-order
#'   sensitivities. Only allowed if `attr(model, "deriv2") == TRUE`.
#'   If `NULL`, zero seeding is used.
#' @param forcings Optional named list of forcing function data. Each element
#'   should be a `data.frame` (or coercible object) with columns `time` and `value`,
#'   or a two-column matrix. Names must match `attr(model, "forcings")`.
#'   Default: `NULL`.
#' @param abstol Absolute error tolerance for the integrator. Default: `1e-6`.
#' @param reltol Relative error tolerance for the integrator. Default: `1e-6`.
#' @param maxprogress Maximum number of integration steps without progress
#'   before an error is raised. Default: `100`.
#' @param maxsteps Maximum total number of integration steps. Default: `1e6`.
#' @param hini Initial step size. If `0` (default), an appropriate step size
#'   is estimated automatically.
#' @param roottol Tolerance for root-finding in root-triggered events.
#'   Default: `1e-6`.
#' @param maxroot Maximum triggers per root event. Default: `1`.
#'
#' @return A named list with components `time`, `variable`, and optionally
#'   `sens1` and `sens2` depending on model configuration.
#'
#' @seealso [CppODE()] for model specification and compilation.
#'
#' @export
solveODE <- function(model, times, parms,
                     sens1ini = NULL, sens2ini = NULL,
                     forcings = NULL,
                     abstol = 1e-6, reltol = 1e-6,
                     maxprogress = 100L, maxsteps = 1e6L,
                     hini = 0, roottol = 1e-6, maxroot = 100L) {

  ## --- Model validation ---
  if (!is.character(model) || length(model) != 1L) {
    stop("'model' must be a character string (modelname from CppODE() output)")
  }

  model_attr_names <- names(attributes(model))
  required_attrs <- c("variables", "parameters", "forcings",
                      "deriv", "deriv2", "dim_names")
  missing_attrs <- required_attrs[!required_attrs %in% model_attr_names]

  if (length(missing_attrs)) {
    stop("'model' is missing required attributes: ",
         paste(missing_attrs, collapse = ", "),
         ". Was it created by CppODE()?")
  }

  variables     <- attr(model, "variables")
  parameters    <- attr(model, "parameters")
  forcing_names <- attr(model, "forcings")
  deriv         <- attr(model, "deriv")
  deriv2        <- attr(model, "deriv2")
  dim_names     <- attr(model, "dim_names")

  ## --- Sensitivity initial values ---
  if (!deriv) {
    if (!is.null(sens1ini) || !is.null(sens2ini)) {
      stop("sens1ini/sens2ini supplied but model has deriv = FALSE")
    }
    sens1ini <- NULL
    sens2ini <- NULL
  } else {
    sens_names <- dim_names$sens
    if (is.null(sens_names))
      stop("Model has deriv = TRUE but no sensitivity dim_names$sens")
    if (!is.null(sens1ini)) {
      if (!is.numeric(sens1ini))
        stop("'sens1ini' must be numeric")
      n_states <- length(variables)
      n_sens   <- length(sens_names)
      expected_len <- n_states * n_sens
      # Accept matrix [n_states, n_sens] or vector of length n_states * n_sens
      if (is.matrix(sens1ini)) {
        # Validate matrix dimensions
        if (nrow(sens1ini) != n_states || ncol(sens1ini) != n_sens)
          stop(sprintf("'sens1ini' matrix must have dimensions [%d, %d] (states x sens)",
                       n_states, n_sens))
        # Check dimnames if present
        if (!is.null(rownames(sens1ini)) && !setequal(rownames(sens1ini), variables))
          stop("'sens1ini' row names must match model variables")
        if (!is.null(colnames(sens1ini)) && !setequal(colnames(sens1ini), sens_names))
          stop("'sens1ini' column names must match model sensitivity names")
        # Reorder if named
        if (!is.null(rownames(sens1ini)) && !is.null(colnames(sens1ini)))
          sens1ini <- sens1ini[variables, sens_names, drop = FALSE]
        # Flatten column-major (R default) to match C++ IDX1(state, sens) = state + n_states * sens
        sens1ini <- as.double(sens1ini)
      } else if (is.array(sens1ini) && length(dim(sens1ini)) == 2) {
        # Same as matrix case
        if (dim(sens1ini)[1] != n_states || dim(sens1ini)[2] != n_sens)
          stop(sprintf("'sens1ini' array must have dimensions [%d, %d]", n_states, n_sens))
        sens1ini <- as.double(sens1ini)
      } else {
        # Vector case
        if (length(sens1ini) != expected_len)
          stop(sprintf("'sens1ini' vector must have length %d (n_states * n_sens)", expected_len))
        sens1ini <- as.double(sens1ini)
      }
    }
    if (!deriv2 && !is.null(sens2ini)) {
      stop("'sens2ini' supplied but model has deriv2 = FALSE")
    }
    if (deriv2 && !is.null(sens2ini)) {
      if (!is.numeric(sens2ini))
        stop("'sens2ini' must be numeric")
      n_states <- length(variables)
      n_sens   <- length(sens_names)
      expected_len <- n_states * n_sens * n_sens
      # Accept array [n_states, n_sens, n_sens] or vector
      if (is.array(sens2ini) && length(dim(sens2ini)) == 3) {
        # Validate array dimensions
        if (dim(sens2ini)[1] != n_states || dim(sens2ini)[2] != n_sens || dim(sens2ini)[3] != n_sens)
          stop(sprintf("'sens2ini' array must have dimensions [%d, %d, %d] (states x sens x sens)",
                       n_states, n_sens, n_sens))
        # Check dimnames if present
        dn <- dimnames(sens2ini)
        if (!is.null(dn[[1]]) && !setequal(dn[[1]], variables))
          stop("'sens2ini' first dimension names must match model variables")
        if (!is.null(dn[[2]]) && !setequal(dn[[2]], sens_names))
          stop("'sens2ini' second dimension names must match model sensitivity names")
        if (!is.null(dn[[3]]) && !setequal(dn[[3]], sens_names))
          stop("'sens2ini' third dimension names must match model sensitivity names")
        # Reorder if named
        if (!is.null(dn[[1]]) && !is.null(dn[[2]]) && !is.null(dn[[3]]))
          sens2ini <- sens2ini[variables, sens_names, sens_names, drop = FALSE]
        # Flatten to match C++ IDX2(state, v1, v2) = state + n_states * (v1 + n_sens * v2)
        sens2ini <- as.double(sens2ini)
      } else {
        # Vector case
        if (length(sens2ini) != expected_len)
          stop(sprintf("'sens2ini' vector must have length %d (n_states * n_sens * n_sens)", expected_len))
        sens2ini <- as.double(sens2ini)
      }
    }
  }

  ## --- Times ---
  if (!is.numeric(times) || !length(times))
    stop("'times' must be a non-empty numeric vector")
  if (anyNA(times) || any(!is.finite(times)))
    stop("'times' must contain only finite values")
  times <- as.double(times)

  ## --- Parameters ---
  if (!is.numeric(parms) || is.null(names(parms)))
    stop("'parms' must be a named numeric vector")

  required_names <- c(variables, parameters)
  missing_names <- required_names[!required_names %in% names(parms)]
  if (length(missing_names))
    stop("'parms' is missing required values: ",
         paste(missing_names, collapse = ", "))

  parms_ordered <- as.double(parms[required_names])
  if (anyNA(parms_ordered) || any(!is.finite(parms_ordered)))
    stop("'parms' must contain only finite values")

  ## --- Forcings ---
  n_forcings <- length(forcing_names)

  if (n_forcings && is.null(forcings)) {
    stop("Model requires forcings: ",
         paste(forcing_names, collapse = ", "),
         "\nProvide via 'forcings' argument.")
  }

  if (!n_forcings) {
    forcing_times_list  <- list()
    forcing_values_list <- list()
  } else {

    if (!is.list(forcings) || is.null(names(forcings)))
      stop("'forcings' must be a named list")

    missing_forcings <- forcing_names[!forcing_names %in% names(forcings)]
    if (length(missing_forcings))
      stop("Missing forcing data for: ",
           paste(missing_forcings, collapse = ", "))

    forcing_times_list  <- vector("list", n_forcings)
    forcing_values_list <- vector("list", n_forcings)

    for (i in seq_len(n_forcings)) {
      nm <- forcing_names[i]
      f  <- forcings[[nm]]

      if (is.matrix(f)) {
        if (ncol(f) != 2L)
          stop("Forcing '", nm, "' must have 2 columns (time, value)")
        f <- data.frame(time = f[, 1L], value = f[, 2L])
      } else if (!is.data.frame(f)) {
        f <- as.data.frame(f)
      }

      if (!all(c("time", "value") %in% names(f)))
        stop("Forcing '", nm, "' must have columns 'time' and 'value'")

      ft <- as.double(f$time)
      fv <- as.double(f$value)

      if (length(ft) < 2L)
        stop("Forcing '", nm, "' needs at least 2 time points")
      if (length(ft) != length(fv))
        stop("Forcing '", nm, "': 'time' and 'value' length mismatch")
      if (anyNA(ft) || any(!is.finite(ft)))
        stop("Forcing '", nm, "': non-finite 'time'")
      if (anyNA(fv) || any(!is.finite(fv)))
        stop("Forcing '", nm, "': non-finite 'value'")
      if (anyDuplicated(ft))
        stop("Forcing '", nm, "': duplicate time values")

      forcing_times_list[[i]]  <- ft
      forcing_values_list[[i]] <- fv
    }
  }

  ## --- Solver options ---
  if (!is.numeric(abstol) || abstol <= 0) stop("'abstol' must be positive")
  if (!is.numeric(reltol) || reltol <= 0) stop("'reltol' must be positive")
  if (!is.numeric(hini)   || hini   <  0) stop("'hini' must be non-negative")
  if (!is.numeric(roottol)|| roottol<=  0) stop("'roottol' must be positive")

  maxprogress <- as.integer(maxprogress)
  maxsteps    <- as.integer(maxsteps)
  maxroot     <- as.integer(maxroot)

  if (maxprogress <= 0L) stop("'maxprogress' must be positive")
  if (maxsteps    <= 0L) stop("'maxsteps' must be positive")
  if (maxroot     <= 0L) stop("'maxroot' must be positive")

  ## --- Call C++ solver ---
  solver_name <- paste0("solve_", model)

  result <- tryCatch({
    SYM <- getNativeSymbolInfo(solver_name)
    .Call(
      SYM,
      times,
      parms_ordered,
      sens1ini,
      sens2ini,
      as.double(abstol),
      as.double(reltol),
      maxprogress,
      maxsteps,
      as.double(hini),
      as.double(roottol),
      maxroot,
      forcing_times_list,
      forcing_values_list
    )
  }, error = function(e) {
    msg <- e$message
    if (grepl("not available|not found|symbol", msg, ignore.case = TRUE)) {
      stop("Compiled solver '", solver_name, "' not found.\n",
           "Possible causes:\n",
           "  - Model was created with compile = FALSE\n",
           "  - Shared library was not loaded (try: dyn.load(...))\n",
           "  - R session was restarted after compilation",
           call. = FALSE)
    }
    # Re-throw other errors with context
    stop("Error in ODE solver: ", msg, call. = FALSE)
  })

  ## --- Output decoration ---
  if (!is.null(result$variable)) {
    colnames(result$variable) <- variables
  }

  if (!is.null(result$sens1)) {
    dimnames(result$sens1) <- list(
      time = NULL,
      variable = variables,
      sens = dim_names$sens
    )
  }

  if (!is.null(result$sens2)) {
    dimnames(result$sens2) <- list(
      time = NULL,
      variable = variables,
      sens1 = dim_names$sens,
      sens2 = dim_names$sens
    )
  }

  result
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
#' @example inst/examples/example_fun.R
#' @export
funCpp <- function(eqns, variables  = getSymbols(eqns, omit = parameters),
                   parameters = NULL, fixed = NULL,
                   modelname = NULL, outdir = tempdir(),
                   compile = FALSE, verbose = FALSE, warnings = TRUE,
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

  # Lazy import
  codegen <- get_codegenfunCpp_py()

  exprs_list <- as.list(eqns)
  names(exprs_list) <- outnames

  py_jac <- if (!is.null(sym_jac)) {
    lapply(seq_len(nrow(sym_jac)), function(i)
      as.list(as.character(sym_jac[i, ])))
  } else {
    NULL
  }
  if (!is.null(py_jac)) names(py_jac) <- rownames(sym_jac)

  py_hess <- if (!is.null(sym_hess)) {
    lapply(seq_len(length(sym_hess)), function(i)
      lapply(seq_len(nrow(sym_hess[[i]])), function(j)
        as.list(as.character(sym_hess[[i]][j, ]))))
  } else {
    NULL
  }
  if (!is.null(py_hess)) names(py_hess) <- names(sym_hess)

  # Warn if file already exists
  if (file.exists(file.path(outdir, paste0(modelname, ".cpp")))) {
    message("Overwriting existing file: ", normalizePath(file.path(outdir, paste0(modelname, ".cpp")), winslash = "/", mustWork = FALSE))
  }

  codegen$generate_fun_cpp(
    exprs      = exprs_list,
    variables  = if (length(variables)  > 0) variables  else list(),
    parameters = if (length(parameters) > 0) parameters else list(),
    jacobian   = py_jac,
    hessian    = py_hess,
    modelname  = modelname,
    outdir     = normalizePath(outdir, winslash = "/", mustWork = FALSE)
  )

  outRfn <- function(vars, params = numeric(0),
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

    result <- list()

    if (is.loaded(funsym_eval)) {
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

      if (is.loaded(funsym_jac)) {
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

      if (is.loaded(funsym_hess)) {
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

  outfn <- outRfn

  if (convenient) {
    outfn <- function(..., attach.input = FALSE,
                      deriv = TRUE, deriv2 = FALSE,
                      fixed = NULL) {
      arglist <- list(...)
      M <- if (!is.null(innames) && length(innames) > 0)
        do.call(cbind, arglist[innames]) else NULL
      p <- if (!is.null(parameters) && length(parameters) > 0)
        do.call(c, arglist[parameters]) else numeric(0)
      outRfn(M, p, attach.input = attach.input,
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
