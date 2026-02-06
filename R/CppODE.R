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
      'extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP sens1iniSEXP, SEXP sens2iniSEXP, SEXP fixedSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP, SEXP hiniSEXP, SEXP root_tolSEXP, SEXP maxrootSEXP, SEXP forcingTimesSEXP, SEXP forcingValuesSEXP) {',
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

  # --- Runtime fixed parameters ---
  if (deriv) {
    externC <- c(
      externC,
      "  // Runtime fixed parameters - O(1) lookup via boolean vector",
      "  std::vector<bool> is_runtime_fixed(n_sens, false);",
      "  if (!Rf_isNull(fixedSEXP)) {",
      "    int* fixed_ptr = INTEGER(fixedSEXP);",
      "    int n_fixed = Rf_length(fixedSEXP);",
      "    for (int i = 0; i < n_fixed; ++i) {",
      "      if (fixed_ptr[i] >= 0 && fixed_ptr[i] < n_sens) {",
      "        is_runtime_fixed[fixed_ptr[i]] = true;",
      "      }",
      "    }",
      "  }",
      "  int n_runtime_fixed = std::count(is_runtime_fixed.begin(), is_runtime_fixed.end(), true);",
      ""
    )
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
      "    if (!is_fixed && !is_runtime_fixed[i]) {",
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
      "    if (!is_fixed && !is_runtime_fixed[i]) {",
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
      "    if (!is_fixed && !is_runtime_fixed[sens_idx]) {",
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
      "    if (!is_fixed && !is_runtime_fixed[sens_idx]) {",
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
    # Note: Output dimension is n_total_sens (compile-time) minus n_runtime_fixed
    externC <- c(externC,
                 "  // --- Return list(time, variable, sens1) ---",
                 "  // Effective sens dimension excludes both compile-time and runtime fixed",
                 "  int n_sens_out = n_sens - n_runtime_fixed;",
                 "",
                 "  // Helper to check if index v is fixed (compile-time or runtime)",
                 "  auto is_any_fixed = [&is_runtime_fixed](int v) {",
                 sprintf("    // Compile-time fixed checks"),
                 if (length(fixed_initial_idx) > 0 || length(fixed_param_idx) > 0) {
                   c(
                     if (length(fixed_initial_idx) > 0) {
                       sprintf("    if (%s) return true;",
                               paste(sprintf("v == %d", fixed_initial_idx), collapse = " || "))
                     },
                     if (length(fixed_param_idx) > 0) {
                       sprintf("    if (%s) return true;",
                               paste(sprintf("v == %d", n_variables + fixed_param_idx), collapse = " || "))
                     }
                   )
                 } else {
                   "    // No compile-time fixed parameters"
                 },
                 "    // Runtime fixed check",
                 "    return is_runtime_fixed[v];",
                 "  };",
                 "",
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
                 "  INTEGER(sens1_dim)[2] = n_sens_out;",
                 "  SEXP sens1_arr = PROTECT(Rf_allocArray(REALSXP, sens1_dim));",
                 "",
                 "  double* time_out = REAL(time_vec);",
                 "  double* variable_out = REAL(variable_mat);",
                 "  double* sens1_out = REAL(sens1_arr);",
                 "",
                 "  auto IDX_variable = [n_out](int r, int c){ return r + c * n_out; };",
                 sprintf("  auto IDX_sens1 = [n_out, n_sens_out](int t, int s, int v){ return t + n_out * (s + %d * v); };", n_variables),
                 "",
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i].x();",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                 "      variable_out[IDX_variable(i, s)] = xi.x();",
                 "      int v_out = 0;",
                 sprintf("      for (int v = 0; v < %d; ++v) {", n_variables + n_params),
                 "        if (!is_any_fixed(v)) {",
                 "          sens1_out[IDX_sens1(i, s, v_out)] = xi.d(v);",
                 "          v_out++;",
                 "        }",
                 "      }",
                 "    }",
                 "  }",
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, variable_mat);",
                 "  SET_VECTOR_ELT(ans, 2, sens1_arr);",
                 "  UNPROTECT(6);",
                 "  return ans;")

  } else {
    # deriv2 = TRUE: list(time, variable, sens1, sens2)
    # Note: Output dimension is n_total_sens (compile-time) minus n_runtime_fixed
    externC <- c(externC,
                 "  // --- Return list(time, variable, sens1, sens2) ---",
                 "  // Effective sens dimension excludes both compile-time and runtime fixed",
                 "  int n_sens_out = n_sens - n_runtime_fixed;",
                 "",
                 "  // Helper to check if index v is fixed (compile-time or runtime)",
                 "  auto is_any_fixed = [&is_runtime_fixed](int v) {",
                 sprintf("    // Compile-time fixed checks"),
                 if (length(fixed_initial_idx) > 0 || length(fixed_param_idx) > 0) {
                   c(
                     if (length(fixed_initial_idx) > 0) {
                       sprintf("    if (%s) return true;",
                               paste(sprintf("v == %d", fixed_initial_idx), collapse = " || "))
                     },
                     if (length(fixed_param_idx) > 0) {
                       sprintf("    if (%s) return true;",
                               paste(sprintf("v == %d", n_variables + fixed_param_idx), collapse = " || "))
                     }
                   )
                 } else {
                   "    // No compile-time fixed parameters"
                 },
                 "    // Runtime fixed check",
                 "    return is_runtime_fixed[v];",
                 "  };",
                 "",
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
                 "  INTEGER(sens1_dim)[2] = n_sens_out;",
                 "  SEXP sens1_arr = PROTECT(Rf_allocArray(REALSXP, sens1_dim));",
                 "  SEXP sens2_dim = PROTECT(Rf_allocVector(INTSXP, 4));",
                 "  INTEGER(sens2_dim)[0] = n_out;",
                 sprintf("  INTEGER(sens2_dim)[1] = %d;", n_variables),
                 "  INTEGER(sens2_dim)[2] = n_sens_out;",
                 "  INTEGER(sens2_dim)[3] = n_sens_out;",
                 "  SEXP sens2_arr = PROTECT(Rf_allocArray(REALSXP, sens2_dim));",
                 "",
                 "  double* time_out = REAL(time_vec);",
                 "  double* variable_out = REAL(variable_mat);",
                 "  double* sens1_out = REAL(sens1_arr);",
                 "  double* sens2_out = REAL(sens2_arr);",
                 "",
                 "  auto IDX_variable = [n_out](int r, int c){ return r + c * n_out; };",
                 sprintf("  auto IDX_sens1 = [n_out, n_sens_out](int t, int s, int v){ return t + n_out * (s + %d * v); };", n_variables),
                 sprintf("  auto IDX_sens2 = [n_out, n_sens_out](int t, int s, int v1, int v2){ return t + n_out * (s + %d * (v1 + n_sens_out * v2)); };", n_variables),
                 "",
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i].x().x();",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                 "      variable_out[IDX_variable(i, s)] = xi.x().x();",
                 "      int v1_out = 0;",
                 sprintf("      for (int v1 = 0; v1 < %d; ++v1) {", n_variables + n_params),
                 "        if (!is_any_fixed(v1)) {",
                 "          sens1_out[IDX_sens1(i, s, v1_out)] = xi.d(v1).x();",
                 "          int v2_out = 0;",
                 sprintf("          for (int v2 = 0; v2 < %d; ++v2) {", n_variables + n_params),
                 "            if (!is_any_fixed(v2)) {",
                 "              sens2_out[IDX_sens2(i, s, v1_out, v2_out)] = xi.d(v1).d(v2);",
                 "              v2_out++;",
                 "            }",
                 "          }",
                 "          v1_out++;",
                 "        }",
                 "      }",
                 "    }",
                 "  }",
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
#' @param fixed Optional character vector of sensitivity parameter names to treat as
#'   fixed at runtime. These parameters will not have their sensitivities computed,
#'   resulting in faster integration. Names must be a subset of
#'   `attr(model, "dim_names")$sens`. Unlike compile-time `fixed` in [CppODE()],
#'   runtime fixed parameters can be changed between calls without recompilation.
#'   The output arrays will have reduced dimensions, excluding the fixed parameters.
#'   Default: `NULL` (all parameters variable).
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
                     fixed = NULL,
                     forcings = NULL,
                     abstol = 1e-6, reltol = 1e-6,
                     maxprogress = 100L, maxsteps = 1e6L,
                     hini = 0, roottol = 1e-6, maxroot = 1L) {

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

  ## --- Runtime fixed parameters ---
  # Convert fixed names to 0-based indices into the sensitivity dimension
  # sens_names contains both state initials and parameters (in that order)

  if (!is.null(fixed)) {
    if (!deriv) {
      warning("'fixed' argument ignored when model has deriv = FALSE")
      fixed_indices <- integer(0)
    } else {
      if (!is.character(fixed))
        stop("'fixed' must be a character vector of sensitivity names")
      sens_names <- dim_names$sens
      unknown <- setdiff(fixed, sens_names)
      if (length(unknown))
        stop("Unknown names in 'fixed': ", paste(unknown, collapse = ", "),
             "\nValid names: ", paste(sens_names, collapse = ", "))
      # Convert to 0-based indices
      fixed_indices <- match(fixed, sens_names) - 1L
    }
  } else {
    fixed_indices <- integer(0)
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
      fixed_indices,
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

  # Compute effective sens names (excluding runtime fixed)
  sens_names_out <- if (deriv) {
    all_sens <- dim_names$sens
    if (length(fixed_indices) > 0) {
      # fixed_indices are 0-based, convert to 1-based for R
      all_sens[-(fixed_indices + 1L)]
    } else {
      all_sens
    }
  } else {
    NULL
  }

  if (!is.null(result$sens1)) {
    dimnames(result$sens1) <- list(
      time = NULL,
      variable = variables,
      sens = sens_names_out
    )
  }

  if (!is.null(result$sens2)) {
    dimnames(result$sens2) <- list(
      time = NULL,
      variable = variables,
      sens1 = sens_names_out,
      sens2 = sens_names_out
    )
  }

  result
}


#' Generate Algebraic Model Functions with Optional C++ Backend and Derivatives
#'
#' @description
#' Generates functions for evaluating a system of algebraic expressions and,
#' optionally, their Jacobian and Hessian matrices. Model evaluation can be
#' performed either via generated and compiled C++ code for improved performance
#' or via a pure R fallback.
#'
#' @param eqns Named character vector or list of algebraic expressions.
#'   Names define the output variables. If unnamed, default names
#'   \code{f1}, \code{f2}, \ldots are assigned.
#' @param variables Character vector of variable names supplied per observation.
#'   Defaults to all symbols found in \code{eqns} not listed in \code{parameters}.
#' @param parameters Character vector of parameter names (constant across observations).
#' @param fixed Optional character vector of symbols to treat as fixed
#'   (excluded from derivative computation).
#' @param modelname Optional base name for generated C++ symbols and files.
#' @param outdir Directory for generated C++ source files. Defaults to \code{tempdir()}.
#' @param compile Logical; if \code{TRUE}, compile and load generated C++ code.
#' @param verbose Logical; if \code{TRUE}, print progress messages.
#' @param warnings Logical; reserved for future use.
#' @param convenient Logical; if \code{TRUE}, return wrappers accepting named arguments.
#' @param deriv Logical; if \code{TRUE}, enable Jacobian computation.
#' @param deriv2 Logical; if \code{TRUE}, enable Hessian computation (implies \code{deriv}).
#'
#' @return
#' A list with components \code{func}, \code{jac}, and \code{hess},
#' carrying the following attributes:
#' \code{equations} (original expressions),
#' \code{variables} (variables),
#' \code{parameters} (parameters),
#' \code{fixed} (fixed symbols),
#' \code{modelname} (C++ identifier),
#' \code{srcfile} (source file),
#' \code{jacobian.symb}, \code{hessian.symb} (symbolic derivatives).
#'
#' @details
#' The function generates C++ code for efficient evaluation of algebraic
#' expressions. When \code{compile = TRUE}, the code is compiled using
#' \code{\link{compile}} and the shared library is loaded.
#'
#' The \code{attach.input} argument allows pass-through of additional inputs
#' that are not part of the model equations:
#' \itemize{
#'   \item Unknown variables (vectors with length matching \code{n_obs}) are
#'     appended to outputs. Their Jacobian columns are zero (no model
#'     dependence). Their Hessian slices are zero.
#'   \item Unknown parameters (scalar values) are appended to outputs
#'     (broadcast across observations). Their Jacobian shows identity
#'     (derivative of parameter w.r.t. itself is 1). Their Hessian is zero.
#' }
#'
#' @seealso \code{\link{compile}} for compilation, \code{\link{derivSymb}}
#'   for symbolic differentiation.
#'
#' @export
funCpp <- function(eqns, variables = getSymbols(eqns, omit = parameters), parameters = NULL,
                   fixed = NULL, modelname = NULL, outdir = tempdir(), compile = FALSE,
                   verbose = FALSE, warnings = TRUE, convenient = TRUE, deriv = TRUE, deriv2 = FALSE) {

  if (deriv2 && !deriv) { warning("deriv2 requires deriv. Setting deriv = TRUE."); deriv <- TRUE }

  outnames <- names(eqns) %||% paste0("f", seq_along(eqns))
  if (!is.null(fixed)) { variables <- setdiff(variables, fixed); parameters <- union(parameters, fixed) }
  innames <- variables; diff_params <- setdiff(parameters, fixed); diff_syms <- c(variables, diff_params)
  if (!dir.exists(outdir)) stop("outdir does not exist: ", outdir)
  modelname <- modelname %||% paste0("f", paste(sample(c(letters, 0:9), 8, TRUE), collapse = ""))

  # --- Input validation ---
  checkInputs <- function(vars, params, attach = FALSE) {
    n_obs <- if (is.matrix(vars) || is.data.frame(vars)) nrow(vars)
    else if (is.vector(vars) && !is.list(vars)) length(vars) / max(length(innames), 1L) else 1L
    n_obs <- max(as.integer(n_obs), 1L); extra_vars <- extra_params <- NULL

    if (!length(innames)) {
      M <- matrix(0, n_obs, 0)
      if (attach && !is.null(vars) && (is.matrix(vars) || is.data.frame(vars)) && ncol(vars) > 0) {
        extra_vars <- as.matrix(vars); n_obs <- nrow(extra_vars)
      }
    } else {
      if (is.null(vars)) stop("Variables defined but 'vars' is NULL.")
      if (is.vector(vars) && !is.list(vars)) vars <- matrix(vars, ncol = length(innames), dimnames = list(NULL, innames))
      colnames(vars) <- colnames(vars) %||% innames
      miss <- setdiff(innames, colnames(vars)); if (length(miss)) stop("Missing variables: ", paste(miss, collapse = ", "))
      M <- vars[, innames, drop = FALSE]; n_obs <- nrow(M)
      if (attach) { ex <- setdiff(colnames(vars), innames); if (length(ex)) extra_vars <- vars[, ex, drop = FALSE] }
    }

    if (!length(parameters)) {
      p <- numeric(0); if (attach && length(params)) extra_params <- params
    } else {
      if (is.null(names(params))) stop("params must be named.")
      miss <- setdiff(parameters, names(params)); if (length(miss)) stop("Missing parameters: ", paste(miss, collapse = ", "))
      p <- params[parameters]
      if (attach) { ex <- setdiff(names(params), parameters); if (length(ex)) extra_params <- params[ex] }
    }
    list(M = t(M), p = p, n_obs = n_obs, extra_vars = extra_vars, extra_params = extra_params)
  }

  # --- Symbolic derivatives ---
  sym_jac <- sym_hess <- NULL
  if (deriv || deriv2) {
    ds <- derivSymb(eqns, deriv2 = deriv2, real = TRUE, fixed = fixed, verbose = verbose)
    sym_jac <- ds$jacobian; sym_hess <- ds$hessian
  }
  if (!is.null(sym_jac)) { rownames(sym_jac) <- rownames(sym_jac) %||% outnames; sym_jac <- sym_jac[, intersect(diff_syms, colnames(sym_jac)), drop = FALSE] }
  if (!is.null(sym_hess)) for (nm in names(sym_hess)) { av <- intersect(diff_syms, rownames(sym_hess[[nm]])); sym_hess[[nm]] <- sym_hess[[nm]][av, av, drop = FALSE] }

  # --- Expression parsing ---
  fallback_ok <- TRUE
  safeParse <- function(s) {
    if (is.null(s) || s == "0") return(expression(0))
    s <- gsub("Heaviside\\(([^)]+)\\)", "ifelse(\\1 >= 0, 1, 0)", s)
    s <- gsub("exp10\\(([^)]+)\\)", "exp((\\1) * log(10))", s)
    tryCatch(parse(text = s), error = function(e) { fallback_ok <<- FALSE; NULL })
  }
  parsed_exprs <- lapply(eqns, safeParse)
  parsed_jac <- if (!is.null(sym_jac)) { m <- matrix(vector("list", length(sym_jac)), nrow(sym_jac), dimnames = dimnames(sym_jac)); for (i in seq_along(sym_jac)) m[[i]] <- safeParse(sym_jac[i]); m }
  parsed_hess <- if (!is.null(sym_hess)) lapply(sym_hess, function(H) { m <- matrix(vector("list", length(H)), nrow(H), dimnames = dimnames(H)); for (i in seq_along(H)) m[[i]] <- safeParse(H[i]); m })
  if (!fallback_ok) warning("R fallback unavailable. Please compile.")

  # --- C++ codegen ---
  codegen <- get_codegenfunCpp_py()
  toList <- function(mat) if (is.null(mat)) NULL else setNames(lapply(seq_len(nrow(mat)), function(i) as.list(as.character(mat[i,]))), rownames(mat))
  toHess <- function(hl) if (is.null(hl)) NULL else setNames(lapply(hl, function(H) lapply(seq_len(nrow(H)), function(i) as.list(as.character(H[i,])))), names(hl))
  cpp_file <- file.path(outdir, paste0(modelname, ".cpp"))
  if (file.exists(cpp_file)) message("Overwriting: ", normalizePath(cpp_file, "/", FALSE))
  codegen$generate_fun_cpp(exprs = setNames(as.list(eqns), outnames), variables = as.list(variables),
                           parameters = as.list(parameters), jacobian = toList(sym_jac), hessian = toHess(sym_hess),
                           modelname = modelname, outdir = normalizePath(outdir, "/", FALSE), version = as.character(utils::packageVersion("CppODE")))

  # --- Attach helpers ---
  abind3 <- function(a, b) { d <- dim(a); db <- dim(b); r <- array(0, c(d[1], d[2]+db[2], d[3]), list(NULL, c(dimnames(a)[[2]], dimnames(b)[[2]]), dimnames(a)[[3]])); r[,1:d[2],] <- a; r[,d[2]+1:db[2],] <- b; r }
  abind4 <- function(a, b) { d <- dim(a); db <- dim(b); r <- array(0, c(d[1], d[2]+db[2], d[3], d[4]), list(NULL, c(dimnames(a)[[2]], dimnames(b)[[2]]), dimnames(a)[[3]], dimnames(a)[[4]])); r[,1:d[2],,] <- a; r[,d[2]+1:db[2],,] <- b; r }

  attachExtras <- function(res, n_obs, ev, ep, type) {
    if (is.null(ev) && is.null(ep)) return(res)
    if (type == "fun") {
      if (!is.null(ev)) res <- cbind(res, ev)
      if (!is.null(ep)) res <- cbind(res, matrix(ep, n_obs, length(ep), TRUE, list(NULL, names(ep))))
    } else if (type == "jac") {
      cs <- dimnames(res)[[3]]; ncs <- length(cs)
      if (!is.null(ev)) res <- abind3(res, array(0, c(n_obs, ncol(ev), ncs), list(NULL, colnames(ev), cs)))
      if (!is.null(ep)) { np <- length(ep); pn <- names(ep); d <- dim(res); new <- array(0, c(d[1], d[2]+np, d[3]+np), list(NULL, c(dimnames(res)[[2]], pn), c(dimnames(res)[[3]], pn))); new[,1:d[2],1:d[3]] <- res; for (k in seq_len(np)) new[,d[2]+k,d[3]+k] <- 1; res <- new }
    } else {
      cs <- dimnames(res)[[3]]; ncs <- length(cs)
      if (!is.null(ev)) res <- abind4(res, array(0, c(n_obs, ncol(ev), ncs, ncs), list(NULL, colnames(ev), cs, cs)))
      if (!is.null(ep)) { np <- length(ep); pn <- names(ep); d <- dim(res); new <- array(0, c(d[1], d[2]+np, d[3]+np, d[4]+np), list(NULL, c(dimnames(res)[[2]], pn), c(dimnames(res)[[3]], pn), c(dimnames(res)[[4]], pn))); new[,1:d[2],1:d[3],1:d[4]] <- res; res <- new }
    }
    res
  }

  # --- Core implementations ---
  fun_impl <- function(vars, params = numeric(0), attach.input = FALSE, fixed = NULL) {
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    funsym <- paste0(modelname, "_eval")
    if (is.loaded(funsym)) {
      out <- .C(funsym, x = as.double(M), y = double(length(outnames) * n_obs), p = as.double(p), n = as.integer(n_obs), k = as.integer(length(innames)), l = as.integer(length(outnames)))
      res <- matrix(out$y, n_obs, length(outnames), TRUE, list(NULL, outnames))
    } else {
      res <- matrix(NA_real_, n_obs, length(outnames), dimnames = list(NULL, outnames))
      for (i in seq_len(n_obs)) { env <- setNames(as.list(c(M[,i], p)), c(innames, parameters)); res[i,] <- vapply(parsed_exprs, function(e) eval(e, env), numeric(1)) }
    }
    attachExtras(res, n_obs, chk$extra_vars, chk$extra_params, "fun")
  }

  jac_impl <- if (deriv && !is.null(sym_jac)) function(vars, params = numeric(0), attach.input = FALSE, fixed = NULL) {
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    fixed_rt <- if (is.null(fixed)) character(0) else intersect(fixed, parameters); dsyms <- setdiff(diff_syms, fixed_rt)
    funsym <- paste0(modelname, "_jacobian"); n_out <- length(outnames); n_sym <- length(diff_syms)
    if (is.loaded(funsym)) {
      out <- .C(funsym, x = as.double(M), jac = double(n_obs * n_out * n_sym), p = as.double(p), n = as.integer(n_obs), k = as.integer(length(innames)), l = as.integer(n_out))
      arr <- array(out$jac, c(n_obs, n_out, n_sym), list(NULL, outnames, diff_syms))
    } else {
      arr <- array(0, c(n_obs, n_out, n_sym), list(NULL, outnames, diff_syms))
      for (i in seq_len(n_obs)) { env <- setNames(as.list(c(M[,i], p)), c(innames, parameters)); for (o in seq_len(n_out)) for (s in seq_len(n_sym)) if (!(diff_syms[s] %in% fixed_rt)) { e <- parsed_jac[[outnames[o], diff_syms[s]]]; if (!is.null(e)) arr[i,o,s] <- eval(e, env) } }
    }
    attachExtras(arr[,,dsyms,drop=FALSE], n_obs, chk$extra_vars, chk$extra_params, "jac")
  }

  hess_impl <- if (deriv2 && !is.null(sym_hess)) function(vars, params = numeric(0), attach.input = FALSE, fixed = NULL) {
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    fixed_rt <- if (is.null(fixed)) character(0) else intersect(fixed, parameters); dsyms <- setdiff(diff_syms, fixed_rt)
    funsym <- paste0(modelname, "_hessian"); n_out <- length(outnames); n_sym <- length(diff_syms)
    if (is.loaded(funsym)) {
      out <- .C(funsym, x = as.double(M), hess = double(n_obs * n_out * n_sym^2), p = as.double(p), n = as.integer(n_obs), k = as.integer(length(innames)), l = as.integer(n_out))
      arr <- array(out$hess, c(n_obs, n_out, n_sym, n_sym), list(NULL, outnames, diff_syms, diff_syms))
    } else {
      arr <- array(0, c(n_obs, n_out, n_sym, n_sym), list(NULL, outnames, diff_syms, diff_syms))
      for (i in seq_len(n_obs)) { env <- setNames(as.list(c(M[,i], p)), c(innames, parameters)); for (o in seq_len(n_out)) { Hmat <- parsed_hess[[outnames[o]]]; for (s1 in seq_len(n_sym)) for (s2 in seq_len(n_sym)) if (!(diff_syms[s1] %in% fixed_rt) && !(diff_syms[s2] %in% fixed_rt)) { e <- Hmat[[diff_syms[s1], diff_syms[s2]]]; if (!is.null(e)) arr[i,o,s1,s2] <- eval(e, env) } } }
    }
    attachExtras(arr[,,dsyms,dsyms,drop=FALSE], n_obs, chk$extra_vars, chk$extra_params, "hess")
  }

  # --- Convenient wrapper ---
  makeWrapper <- function(impl) {
    if (is.null(impl)) return(NULL)
    function(..., attach.input = FALSE, fixed = NULL) {
      args <- list(...); M <- if (length(innames)) do.call(cbind, args[innames]); p <- if (length(parameters)) do.call(c, args[parameters]) else numeric(0)
      if (attach.input) { extra <- setdiff(names(args), c(innames, parameters)); n_obs <- if (!is.null(M)) nrow(M) else 1L
      for (nm in extra) { v <- args[[nm]]; if (length(v) == n_obs) { M <- if (is.null(M)) matrix(v, ncol=1, dimnames=list(NULL,nm)) else cbind(M, setNames(data.frame(v), nm)) } else if (length(v) == 1) p <- c(p, setNames(v, nm)) else warning("Extra '", nm, "' ignored") } }
      impl(M, p, attach.input, fixed)
    }
  }

  # --- Output ---
  outfn <- list(func = if (convenient) makeWrapper(fun_impl) else fun_impl, jac = if (convenient) makeWrapper(jac_impl) else jac_impl, hess = if (convenient) makeWrapper(hess_impl) else hess_impl)
  attr(outfn, "equations") <- eqns; attr(outfn, "variables") <- variables; attr(outfn, "parameters") <- parameters
  attr(outfn, "fixed") <- fixed; attr(outfn, "modelname") <- modelname; attr(outfn, "srcfile") <- normalizePath(cpp_file, "/", FALSE)
  if (deriv && !is.null(sym_jac)) attr(outfn, "jacobian.symb") <- sym_jac
  if (deriv2 && !is.null(sym_hess)) attr(outfn, "hessian.symb") <- sym_hess
  if (compile) compile(outfn, verbose = verbose)
  outfn
}
