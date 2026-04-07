#' Generate C++ code for ODE models with events and parameter sensitivities
#'
#' @description
#' This function generates and compiles a C++ solver for systems of ordinary differential
#' equations (ODEs) of the form
#'
#' \deqn{\dot{x}(t) = f\big(x(t), p_{\text{dyn}}\big), \quad x(t_0) = p_{\text{init}}}
#'
#' Two integration methods are available: **BDF** (Backward Differentiation Formulas,
#' variable-order up to 5, default) and **Rosenbrock4** (a fourth-order L-stable method
#' derived from Boost.Odeint).
#' Both steppers support dense output, error control, sparse and dense LU factorization,
#' **time-based** and **root-triggered events**, and, optionally,
#' **first- and second-order sensitivities** by evaluating the *same system* with
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
#'   - `variable`: numeric matrix \eqn{X_{ij}} of shape \eqn{(n_x,n_t)}, containing \eqn{x_i(t_j)}
#'
#' - `deriv = TRUE`, `deriv2 = FALSE`
#'   Returns `list(time, variable, sens1)`
#'   - `sens1`: numeric array \eqn{\partial X_{ijk}} of shape \eqn{(n_x,n_s,n_t)}, containing
#'     \eqn{\partial x_i(t_k)/\partial p_j}
#'
#' - `deriv = TRUE`, `deriv2 = TRUE`
#'   Returns `list(time, variable, sens1, sens2)`
#'   - `sens2`: numeric array \eqn{\partial^2 X_{ijkl}} of shape \eqn{(n_x,n_s,n_s,n_t)},
#'     containing \eqn{\partial^2 x_i(t_k)/\partial p_j\,\partial p_k}
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
#'   *and* for all generated C/C++ symbols (e.g. `solve_<modelname>`)
#'   as well as the resulting shared library.
#'   If `NULL`, a random identifier is used.
#' @param outdir Directory where generated C++ source files are written. Defaults to `tempdir()`.
#' @param deriv Logical. If `TRUE`, enable first-order sensitivities via dual numbers.
#' @param deriv2 Logical. If `TRUE`, enable second-order sensitivities via nested dual numbers;
#'   requires `deriv = TRUE`.
#' @param fullErr Logical. If `TRUE`, compute error estimates using full state vector
#'   including derivatives. If `FALSE`, use only the value components for error control.
#' @param includeTimeZero Logical. If `TRUE`, ensure that time `0` is included among integration times.
#' @param useDenseOutput Logical. If `TRUE`, use dense output (Hermite interpolation).
#' @param sparse Controls sparse LU factorization.
#'   `NULL` (default) auto-selects based on Jacobian sparsity.
#'   `TRUE` forces sparse LU; `FALSE` forces dense LU.
#' @param method Character string selecting the integration method: `"bdf"` (default),
#'   `"rb4"`, or `"rosenbrock4"` (alias for `"rb4"`).
#' @param profile Logical. If `TRUE`, compile with profiling counters (`-DCPPODE_PROFILE`).
#' @param stepTrace Logical. If `TRUE`, emit per-step diagnostics (order, step size, gamrat,
#'   setup triggers) to stderr. Verbose; intended for debugging only.
#' @param verbose Logical. If `TRUE`, print progress messages.
#'
#' @return
#' The compiled model name (character string).
#' The returned object carries a set of attributes that describe the compiled solver
#' and its symbolic structure:
#'
#' | Attribute | Type | Description |
#' |:--|:--|:--|
#' | `equations` | `character` | ODE right-hand side definitions |
#' | `srcfile` | `character` | Path to the generated C++ source file |
#' | `variables` | `character` | Names of the dynamic state variables |
#' | `parameters` | `character` | Names of model parameters |
#' | `forcings` | `character` | Names of forcing functions |
#' | `events` | `data.frame` | Table of event specifications (if any) |
#' | `rootfunc` | `character` | Root function specification (if any) |
#' | `fixed` | `character` | Names of fixed initial conditions or parameters |
#' | `jacobian` | `list` | Jacobian: `f.x` (character matrix) and `f.time` (time derivatives) |
#' | `deriv` | `logical` | Whether first-order sensitivities (dual numbers) are enabled |
#' | `deriv2` | `logical` | Whether second-order sensitivities (nested dual numbers) are enabled |
#' | `sparse` | `logical` | Whether sparse LU factorization is used |
#' | `dim_names` | `list` | Dimension names for arrays: `time`, `variable`, and `sens` |
#' | `compileArgs` | `character` | Compiler flags (profile, step trace, KLU settings) |
#'
#' @example inst/examples/example_ODE.R
#' @importFrom stats setNames
#' @seealso [solveODE()] for a solver interface of compiled models
#' @export
CppODE <- function(rhs, events = NULL, rootfunc = NULL, fixed = NULL, forcings = NULL,
                   compile = TRUE, modelname = NULL, outdir = tempdir(),
                   deriv = TRUE, deriv2 = FALSE, fullErr = TRUE,
                   includeTimeZero = TRUE, useDenseOutput = TRUE,
                   sparse = NULL, method = c("bdf", "rb4"),
                   profile = FALSE, stepTrace = FALSE, verbose = FALSE) {

  # --- Validate arguments ---
  if (deriv2 && !deriv) {
    warning("deriv2 = TRUE requires deriv = TRUE. Setting deriv = TRUE automatically.")
    deriv <- TRUE
  }
  method <- match.arg(method)
  if (method == "rb4") method <- "rosenbrock4"

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

  params <- setdiff(symbols, c(variables, forcings, "time"))

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
    forcings_list = forcings,
    sparse = sparse
  )

  ode_code <- codegen_result$ode_code
  jac_code <- codegen_result$jac_code
  time_derivs_str <- codegen_result$time_derivs

  if (verbose) message("  \u2713 ODE and Jacobian generated")

  # --- Sparse LU decision ---
  # The codegen already decided and generated the matching Jacobian functor.
  # use_sparse MUST match what codegen produced, otherwise the Jacobian
  # functor signature won't match the stepper's matrix type.
  use_sparse <- isTRUE(codegen_result$use_sparse)

  if (use_sparse) {
    stats <- codegen_result$sparsity_stats
    message(sprintf("Sparse Jacobian detected (%dx%d, %d nnz, %.3f%% sparse)",
                    stats$n, stats$n, stats$jac_nnz, stats$jac_zeros_pct))
  }

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
      forcings_list = forcings,
      rhs_dict = as.list(setNames(rhs, variables))
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
      "using namespace cppode;",
      sprintf("using AD = fadbad::F<double, %d>;", n_total_sens),
      sprintf("using AD2 = fadbad::F<fadbad::F<double, %d>, %d>;", n_total_sens, n_total_sens)
    )
  } else if (deriv) {
    usings <- c(
      "using namespace cppode;",
      sprintf("using AD = fadbad::F<double, %d>;", n_total_sens)
    )
  } else {
    usings <- c(
      "using namespace cppode;"
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
    sprintf("  void operator()(const cppode::vector_t<%s>& x, const %s& t) {", numType, numType),
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
    sprintf("  cppode::vector_t<%s> x(%d);", numType, n_variables),
    sprintf("  cppode::vector_t<%s> full_params(%d);", numType, n_variables + n_params),
    ""
  )

  # --- Custom sensitivity initial values ---
  if (deriv) {
    externC <- c(
      externC,
      "  // Custom sensitivity initial values (length validated after n_sens is computed)",
      "  bool has_sens1ini = !Rf_isNull(sens1iniSEXP);",
      "  double* sens1ini = has_sens1ini ? REAL(sens1iniSEXP) : nullptr;"
    )

    if (deriv2) {
      externC <- c(
        externC,
        "  bool has_sens2ini = !Rf_isNull(sens2iniSEXP);",
        "  double* sens2ini = has_sens2ini ? REAL(sens2iniSEXP) : nullptr;"
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

  # --- Runtime fixed parameters (must be defined before n_sens and active_idx) ---
  if (deriv) {
    externC <- c(
      externC,
      "  // Runtime fixed parameters - O(1) lookup via boolean vector",
      sprintf("  std::vector<bool> is_runtime_fixed(%d, false);  // size = n_sens_total (compile-time)", n_total_sens),
      "  if (!Rf_isNull(fixedSEXP)) {",
      "    int* fixed_ptr = INTEGER(fixedSEXP);",
      "    int n_fixed = Rf_length(fixedSEXP);",
      "    for (int i = 0; i < n_fixed; ++i) {",
      sprintf("      if (fixed_ptr[i] >= 0 && fixed_ptr[i] < %d) {", n_total_sens),
      "        is_runtime_fixed[fixed_ptr[i]] = true;",
      "      }",
      "    }",
      "  }",
      "  int n_runtime_fixed = std::count(is_runtime_fixed.begin(), is_runtime_fixed.end(), true);",
      ""
    )
  }

  # --- Sensitivity dimensions and index helpers (depends on is_runtime_fixed) ---
  if (deriv) {
    externC <- c(
      externC,
      sprintf("  const int n_states     = %d;", n_variables),
      sprintf("  const int n_sens_total = %d;  // compile-time total (excl. compile-time fixed)", n_total_sens),
      "  // n_sens: active sens dimension after removing runtime-fixed parameters",
      "  const int n_sens = n_sens_total - n_runtime_fixed;",
      "",
      "  // active_idx[i]: compile-time sens index i -> active index in [0, n_sens), or -1 if runtime-fixed",
      "  std::vector<int> active_idx(n_sens_total, -1);",
      "  {",
      "    int k = 0;",
      "    for (int i = 0; i < n_sens_total; ++i) {",
      "      if (!is_runtime_fixed[i]) active_idx[i] = k++;",
      "    }",
      "  }",
      "",
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

    # Build global_to_sens compile-time literal:
    # global index v -> sens index in [0, n_sens_total), or -1 if compile-time-fixed
    n_global <- n_variables + n_params
    fixed_global_idx <- c(fixed_initial_idx, n_variables + fixed_param_idx)
    sens_counter <- 0L
    global_to_sens_vals <- integer(n_global)
    for (v in seq_len(n_global) - 1L) {
      if (v %in% fixed_global_idx) {
        global_to_sens_vals[v + 1L] <- -1L
      } else {
        global_to_sens_vals[v + 1L] <- sens_counter
        sens_counter <- sens_counter + 1L
      }
    }
    global_to_sens_literal <- paste(global_to_sens_vals, collapse = ", ")

    externC <- c(
      externC,
      "",
      "  // global_to_sens[v]: global index v -> compile-time sens index, or -1 if compile-time-fixed",
      sprintf("  static const int global_to_sens_arr[%d] = {%s};", n_global, global_to_sens_literal),
      sprintf("  auto global_to_sens = [](int v) -> int { return global_to_sens_arr[v]; };"),
      "",
      "  // Validate sens1ini / sens2ini length against active dimension n_sens",
      sprintf("  if (has_sens1ini && Rf_length(sens1iniSEXP) != %d * n_sens)", n_variables),
      "    Rf_error(\"sens1ini has wrong length: expected n_states * n_active_sens = %d * %d = %d\",",
      sprintf("             %d, n_sens, %d * n_sens);", n_variables, n_variables)
    )

    if (deriv2) {
      externC <- c(
        externC,
        sprintf("  if (has_sens2ini && Rf_length(sens2iniSEXP) != %d * n_sens * n_sens)", n_variables),
        "    Rf_error(\"sens2ini has wrong length: expected n_states * n_active_sens^2 = %d * %d^2\",",
        sprintf("             %d, n_sens);", n_variables)
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
      "      int si = global_to_sens(i);  // compile-time sens index (-1 if compile-time-fixed)",
      "      int ai = (si >= 0) ? active_idx[si] : -1;  // active index (-1 if any-fixed)",
      "      if (ai >= 0) {",
      "        // First-order sensitivities (inner layer)",
      "        if (has_sens1ini) {",
      "          x[i].x().diff(0);  // Allocate n_sens (active) first-order components (static)",
      "          for (int v1 = 0; v1 < n_sens_total; ++v1) {",
      "            int av1 = active_idx[v1];",
      "            if (av1 >= 0) x[i].x().d(av1) = sens1ini[IDX1(i, av1)];",
      "          }",
      "        } else {",
      "          x[i].x().diff(ai);  // Identity: d(ai) = 1 (static)",
      "        }",
      "        // Second-order sensitivities (outer layer)",
      "        if (has_sens2ini) {",
      "          // Custom second-order initialization",
      "          for (int v1 = 0; v1 < n_sens_total; ++v1) {",
      "            int av1 = active_idx[v1];",
      "            if (av1 < 0) continue;",
      "            x[i].diff(av1).diff(0);  // Allocate (static)",
      "            for (int v2 = 0; v2 < n_sens_total; ++v2) {",
      "              int av2 = active_idx[v2];",
      "              if (av2 >= 0) x[i].diff(av1).d(av2) = sens2ini[IDX2(i, av1, av2)];",
      "            }",
      "          }",
      "        } else {",
      "          // Default: allocate outer layer with zeros",
      "          x[i].diff(ai);  // Allocate n_sens outer components (inner defaults to 0, static)",
      "        }",
      "      }",
      "    }"
    )
  } else if (deriv) {
    externC <- c(
      externC,
      "    x[i] = REAL(paramsSEXP)[i];",
      "    if (!is_fixed) {",
      "      int si = global_to_sens(i);  // compile-time sens index (-1 if compile-time-fixed)",
      "      int ai = (si >= 0) ? active_idx[si] : -1;  // active index (-1 if any-fixed)",
      "      if (ai >= 0) {",
      "        if (has_sens1ini) {",
      "          x[i].diff(0);  // Allocate n_sens (= n_active) dual components (static)",
      "          for (int v = 0; v < n_sens_total; ++v) {",
      "            int av = active_idx[v];",
      "            if (av >= 0) x[i].d(av) = sens1ini[IDX1(i, av)];",
      "          }",
      "        } else {",
      "          x[i].diff(ai);  // Identity: d(ai) = 1, only n_sens components (static)",
      "        }",
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
      "    int global_idx = n_states + i;",
      "    full_params[param_index].x().x() = REAL(paramsSEXP)[param_index];",
      "    if (!is_fixed) {",
      "      int si = global_to_sens(global_idx);  // compile-time sens index",
      "      int ai = (si >= 0) ? active_idx[si] : -1;  // active index",
      "      if (ai >= 0) {",
      "        // First-order (inner layer): identity seeding with active index",
      "        full_params[param_index].x().diff(ai);",
      "        // Second-order (outer layer): allocate n_sens components (inner defaults to 0)",
      "        full_params[param_index].diff(ai);",
      "      }",
      "    }"
    )
  } else if (deriv) {
    externC <- c(
      externC,
      "    int global_idx = n_states + i;",
      "    full_params[param_index] = REAL(paramsSEXP)[param_index];",
      "    if (!is_fixed) {",
      "      int si = global_to_sens(global_idx);  // compile-time sens index",
      "      int ai = (si >= 0) ? active_idx[si] : -1;  // active index",
      "      if (ai >= 0) {",
      "        // Parameters use identity seeding: dp_i/dp_j = delta_{ij}",
      "        full_params[param_index].diff(ai);",
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
               "  result_times.reserve(times_dbl.size());",
               sprintf("  y.reserve(times_dbl.size() * %d);", n_variables),
               "",
               "  // --- Event containers ---",
               sprintf("  std::vector<FixedEvent<cppode::vector_t<%s>, %s>> fixed_events;", numType, numType),
               sprintf("  std::vector<RootEvent<cppode::vector_t<%s>, %s>> root_events;", numType, numType)
  )

  # Insert event code from Python
  if (event_code != "") {
    externC <- c(externC, event_code)
  }

  # Note: rootfunc_code is inserted later, after sys is defined

  # --- Integration setup ---
  # Stepper types: dense or sparse LU, Rosenbrock4 or BDF
  if (use_sparse) {
    rb4_double <- "rosenbrock4<double, sparse_lu_tag>"
    rb4_AD     <- "rosenbrock4<AD, sparse_lu_tag>"
    rb4_AD2    <- "rosenbrock4<AD2, sparse_lu_tag>"
    bdf_double <- "cppode::bdf_stepper<double, sparse_lu_tag>"
    bdf_AD     <- "cppode::bdf_stepper<AD, sparse_lu_tag>"
    bdf_AD2    <- "cppode::bdf_stepper<AD2, sparse_lu_tag>"
  } else {
    rb4_double <- "rosenbrock4<double>"
    rb4_AD     <- "rosenbrock4<AD>"
    rb4_AD2    <- "rosenbrock4<AD2>"
    bdf_double <- "cppode::bdf_stepper<double>"
    bdf_AD     <- "cppode::bdf_stepper<AD>"
    bdf_AD2    <- "cppode::bdf_stepper<AD2>"
  }

  if (method == "bdf") {
    # ---- BDF stepper ----
    if (useDenseOutput) {
      if (deriv2) {
        stepper_line <- paste(
          sprintf("  auto controlledStepper = cppode::bdf_controller_ad<%s, %s>(abstol, reltol);",
                  bdf_AD2, ifelse(fullErr, "true", "false")),
          "  auto denseStepper = cppode::bdf_dense_output<decltype(controlledStepper)>(std::move(controlledStepper));",
          sep = "\n"
        )
      } else if (deriv) {
        stepper_line <- paste(
          sprintf("  auto controlledStepper = cppode::bdf_controller_ad<%s, %s>(abstol, reltol);",
                  bdf_AD, ifelse(fullErr, "true", "false")),
          "  auto denseStepper = cppode::bdf_dense_output<decltype(controlledStepper)>(std::move(controlledStepper));",
          sep = "\n"
        )
      } else {
        stepper_line <- paste(
          sprintf("  auto controlledStepper = cppode::bdf_controller<%s>(abstol, reltol);",
                  bdf_double),
          "  auto denseStepper = cppode::bdf_dense_output<decltype(controlledStepper)>(std::move(controlledStepper));",
          sep = "\n"
        )
      }
      integrate_line <- "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot, dt_est);"
    } else {
      if (deriv2) {
        stepper_line <- sprintf("  auto controlledStepper = cppode::bdf_controller_ad<%s, %s>(abstol, reltol);",
                                bdf_AD2, ifelse(fullErr, "true", "false"))
      } else if (deriv) {
        stepper_line <- sprintf("  auto controlledStepper = cppode::bdf_controller_ad<%s, %s>(abstol, reltol);",
                                bdf_AD, ifelse(fullErr, "true", "false"))
      } else {
        stepper_line <- sprintf("  auto controlledStepper = cppode::bdf_controller<%s>(abstol, reltol);",
                                bdf_double)
      }
      integrate_line <- "  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot, dt_est);"
    }
  } else {
    # ---- Rosenbrock4 stepper (existing) ----
    if (useDenseOutput) {
      if (deriv2) {
        stepper_line <- paste(
          sprintf("  auto controlledStepper = cppode::rosenbrock4_controller<%s>(abstol, reltol);", rb4_AD2),
          "  auto denseStepper = cppode::rosenbrock4_dense_output<decltype(controlledStepper)>(std::move(controlledStepper));",
          sep = "\n"
        )
      } else if (deriv) {
        stepper_line <- paste(
          sprintf("  auto controlledStepper = cppode::rosenbrock4_controller<%s>(abstol, reltol);", rb4_AD),
          "  auto denseStepper = cppode::rosenbrock4_dense_output<decltype(controlledStepper)>(std::move(controlledStepper));",
          sep = "\n"
        )
      } else {
        stepper_line <- paste(
          sprintf("  auto controlledStepper = cppode::rosenbrock4_controller<%s>(abstol, reltol);", rb4_double),
          "  auto denseStepper = cppode::rosenbrock4_dense_output<decltype(controlledStepper)>(std::move(controlledStepper));",
          sep = "\n"
        )
      }
      integrate_line <- "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    } else {
      if (deriv2) {
        stepper_line <- sprintf("  auto controlledStepper = cppode::rosenbrock4_controller<%s>(abstol, reltol);", rb4_AD2)
      } else if (deriv) {
        stepper_line <- sprintf("  auto controlledStepper = cppode::rosenbrock4_controller<%s>(abstol, reltol);", rb4_AD)
      } else {
        stepper_line <- sprintf("  auto controlledStepper = cppode::rosenbrock4_controller<%s>(abstol, reltol);", rb4_double)
      }
      integrate_line <- "  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);"
    }
  }

  # Both dense and sparse paths use the same Jacobian functor name.
  # Codegen produces exactly one 'struct jacobian' matching the path.
  externC <- c(externC, "",
               sprintf("  // --- Solver setup (%s LU) ---",
                       if (use_sparse) "sparse" else "dense"),
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

  # --- Initial step size estimation ---
  if (method == "bdf") {
    dt_func_dense  <- "bdf_utils::estimate_initial_dt_bdf"
    dt_func_sparse <- "bdf_utils::estimate_initial_dt_bdf_sparse"
  } else {
    dt_func_dense  <- "odeint_utils::estimate_initial_dt_local"
    dt_func_sparse <- "odeint_utils::estimate_initial_dt_local_sparse"
  }
  if (use_sparse) {
    estimate_dt_block <- c(
      sprintf("  // --- Determine dt (%s, sparse J x dxdt, O(nnz)) ---", method),
      sprintf("  %s dt;", numType),
      "  if (hini == 0.0) {",
      sprintf("    dt = %s(sys, jac, x, times.front(), abstol, reltol);", dt_func_sparse),
      "  } else {",
      "    dt = hini;",
      "  }"
    )
  } else {
    estimate_dt_block <- c(
      sprintf("  // --- Determine dt (%s) ---", method),
      sprintf("  %s dt;", numType),
      "  if (hini == 0.0) {",
      sprintf("    dt = %s(sys, jac, x, times.front(), abstol, reltol);", dt_func_dense),
      "  } else {",
      "    dt = hini;",
      "  }"
    )
  }

  # --- BDF: dt re-estimator lambda for event restarts ---
  dt_est_block <- character(0)
  if (method == "bdf") {
    dt_func_event <- if (use_sparse) dt_func_sparse else dt_func_dense
    dt_est_block <- c(
      "  // --- BDF event restart: re-estimate dt from post-event state ---",
      sprintf("  auto dt_est = [&](auto& x_ev, auto t_ev) { return %s(sys, jac, x_ev, t_ev, abstol, reltol); };",
              dt_func_event)
    )
  }

  externC <- c(externC,
               stepper_line, "",
               estimate_dt_block,
               dt_est_block,
               "",
               "  // --- Integration (catch step-limit errors for partial results) ---",
               "  std::string solver_message;",
               "  try {",
               paste0("    ", integrate_line),
               "  } catch (const cppode::no_progress_error& e) {",
               "    solver_message = e.what();",
               "  }",
               "",
               "  // --- Populate diagnostics from available state ---",
               "  if (!result_times.empty()) {",
               sprintf("    checker.set_t_reached(static_cast<double>(%s));",
                       if (deriv2) "result_times.back().x().x()"
                       else if (deriv) "result_times.back().x()"
                       else "result_times.back()"),
               "  }",
               "  // last_dt is set by the integration loop (process_dense/process_controlled)",
               "  // and reflects the true controller step size, not the output grid spacing.",
               "",
               "  const int n_out = static_cast<int>(result_times.size());",
               "  if (n_out <= 0) Rf_error(\"Integration produced no output\");",
               "",
               "  // --- Build diagnostics list ---",
               "  // diagnostics: list(return_code, message, accepted, rejected,",
               "  //                   fevals, jevals, setups, last_dt, last_order, t_reached)",
               "  SEXP diag = PROTECT(Rf_allocVector(VECSXP, 10));",
               "  SEXP diag_names = PROTECT(Rf_allocVector(STRSXP, 10));",
               "  SET_STRING_ELT(diag_names, 0, Rf_mkChar(\"return_code\"));",
               "  SET_STRING_ELT(diag_names, 1, Rf_mkChar(\"message\"));",
               "  SET_STRING_ELT(diag_names, 2, Rf_mkChar(\"accepted\"));",
               "  SET_STRING_ELT(diag_names, 3, Rf_mkChar(\"rejected\"));",
               "  SET_STRING_ELT(diag_names, 4, Rf_mkChar(\"fevals\"));",
               "  SET_STRING_ELT(diag_names, 5, Rf_mkChar(\"jevals\"));",
               "  SET_STRING_ELT(diag_names, 6, Rf_mkChar(\"setups\"));",
               "  SET_STRING_ELT(diag_names, 7, Rf_mkChar(\"last_dt\"));",
               "  SET_STRING_ELT(diag_names, 8, Rf_mkChar(\"last_order\"));",
               "  SET_STRING_ELT(diag_names, 9, Rf_mkChar(\"t_reached\"));",
               "  Rf_setAttrib(diag, R_NamesSymbol, diag_names);",
               "  SET_VECTOR_ELT(diag, 0, Rf_ScalarInteger(checker.return_code()));",
               "  {",
               "    SEXP msg_sexp = PROTECT(Rf_allocVector(STRSXP, 1));",
               "    SET_STRING_ELT(msg_sexp, 0, Rf_mkChar(solver_message.empty() ? \"Integration was successful.\" : solver_message.c_str()));",
               "    SET_VECTOR_ELT(diag, 1, msg_sexp);",
               "    UNPROTECT(1);",
               "  }",
               "  SET_VECTOR_ELT(diag, 2, Rf_ScalarInteger(checker.n_accepted()));",
               "  SET_VECTOR_ELT(diag, 3, Rf_ScalarInteger(checker.n_rejected()));",
               "  SET_VECTOR_ELT(diag, 4, Rf_ScalarInteger(checker.n_fevals()));",
               "  SET_VECTOR_ELT(diag, 5, Rf_ScalarInteger(checker.n_jevals()));",
               "  SET_VECTOR_ELT(diag, 6, Rf_ScalarInteger(checker.n_setups()));",
               "  SET_VECTOR_ELT(diag, 7, Rf_ScalarReal(checker.last_dt()));",
               "  SET_VECTOR_ELT(diag, 8, Rf_ScalarInteger(checker.last_order()));",
               "  SET_VECTOR_ELT(diag, 9, Rf_ScalarReal(checker.t_reached()));",
               ""
  )
  # --- Copy back results - CONSISTENT LIST OUTPUT ---
  if (!deriv) {
    # deriv = FALSE: list(time, variable)
    externC <- c(externC,
                 "  // --- Return list(time, variable, diagnostics) ---",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 3));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"variable\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"diagnostics\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP variable_mat = PROTECT(Rf_allocMatrix(REALSXP, %d, n_out));", n_variables),
                 "  double* time_out = REAL(time_vec);",
                 "  double* variable_out = REAL(variable_mat);",
                 "",
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i];",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      variable_out[s + %d * i] = y[i * %d + s];", n_variables, n_variables),
                 "    }",
                 "  }",
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, variable_mat);",
                 "  SET_VECTOR_ELT(ans, 2, diag);",
                 "  UNPROTECT(6);",
                 "  return ans;")

  } else if (!deriv2) {
    # deriv = TRUE, deriv2 = FALSE: list(time, variable, sens1)
    # Note: Output dimension is n_total_sens (compile-time) minus n_runtime_fixed
    externC <- c(externC,
                 "  // --- Return list(time, variable, sens1) ---",
                 "  // Effective sens dimension excludes both compile-time and runtime fixed",
                 "  int n_sens_out = n_sens;",
                 "",
                 "  // Helper: is global index v fixed (compile-time OR runtime)?",
                 "  auto is_any_fixed = [&active_idx, &global_to_sens](int v) -> bool {",
                 "    int si = global_to_sens(v);",
                 "    if (si < 0) return true;   // compile-time fixed",
                 "    return active_idx[si] < 0;  // runtime fixed",
                 "  };",
                 "",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 4));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"variable\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"sens1\"));",
                 "  SET_STRING_ELT(names, 3, Rf_mkChar(\"diagnostics\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP variable_mat = PROTECT(Rf_allocMatrix(REALSXP, %d, n_out));", n_variables),
                 "  SEXP sens1_dim = PROTECT(Rf_allocVector(INTSXP, 3));",
                 sprintf("  INTEGER(sens1_dim)[0] = %d;", n_variables),
                 "  INTEGER(sens1_dim)[1] = n_sens_out;",
                 "  INTEGER(sens1_dim)[2] = n_out;",
                 "  SEXP sens1_arr = PROTECT(Rf_allocArray(REALSXP, sens1_dim));",
                 "",
                 "  double* time_out = REAL(time_vec);",
                 "  double* variable_out = REAL(variable_mat);",
                 "  double* sens1_out = REAL(sens1_arr);",
                 "",
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i].x();",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                 sprintf("      variable_out[s + %d * i] = xi.x();", n_variables),
                 "      int v_out = 0;",
                 sprintf("      for (int v = 0; v < %d; ++v) {", n_variables + n_params),
                 "        if (!is_any_fixed(v)) {",
                 "          int av = active_idx[global_to_sens(v)];",
                 sprintf("          sens1_out[s + %d * (v_out + n_sens_out * i)] = xi.d(av);", n_variables),
                 "          v_out++;",
                 "        }",
                 "      }",
                 "    }",
                 "  }",
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, variable_mat);",
                 "  SET_VECTOR_ELT(ans, 2, sens1_arr);",
                 "  SET_VECTOR_ELT(ans, 3, diag);",
                 "  UNPROTECT(8);",
                 "  return ans;")

  } else {
    # deriv2 = TRUE: list(time, variable, sens1, sens2)
    # Note: Output dimension is n_total_sens (compile-time) minus n_runtime_fixed
    externC <- c(externC,
                 "  // --- Return list(time, variable, sens1, sens2) ---",
                 "  // Effective sens dimension excludes both compile-time and runtime fixed",
                 "  int n_sens_out = n_sens;",
                 "",
                 "  // Helper: is global index v fixed (compile-time OR runtime)?",
                 "  auto is_any_fixed = [&active_idx, &global_to_sens](int v) -> bool {",
                 "    int si = global_to_sens(v);",
                 "    if (si < 0) return true;   // compile-time fixed",
                 "    return active_idx[si] < 0;  // runtime fixed",
                 "  };",
                 "",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 5));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"variable\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"sens1\"));",
                 "  SET_STRING_ELT(names, 3, Rf_mkChar(\"sens2\"));",
                 "  SET_STRING_ELT(names, 4, Rf_mkChar(\"diagnostics\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP variable_mat = PROTECT(Rf_allocMatrix(REALSXP, %d, n_out));", n_variables),
                 "  SEXP sens1_dim = PROTECT(Rf_allocVector(INTSXP, 3));",
                 sprintf("  INTEGER(sens1_dim)[0] = %d;", n_variables),
                 "  INTEGER(sens1_dim)[1] = n_sens_out;",
                 "  INTEGER(sens1_dim)[2] = n_out;",
                 "  SEXP sens1_arr = PROTECT(Rf_allocArray(REALSXP, sens1_dim));",
                 "  SEXP sens2_dim = PROTECT(Rf_allocVector(INTSXP, 4));",
                 sprintf("  INTEGER(sens2_dim)[0] = %d;", n_variables),
                 "  INTEGER(sens2_dim)[1] = n_sens_out;",
                 "  INTEGER(sens2_dim)[2] = n_sens_out;",
                 "  INTEGER(sens2_dim)[3] = n_out;",
                 "  SEXP sens2_arr = PROTECT(Rf_allocArray(REALSXP, sens2_dim));",
                 "",
                 "  double* time_out = REAL(time_vec);",
                 "  double* variable_out = REAL(variable_mat);",
                 "  double* sens1_out = REAL(sens1_arr);",
                 "  double* sens2_out = REAL(sens2_arr);",
                 "",
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i].x().x();",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                 sprintf("      variable_out[s + %d * i] = xi.x().x();", n_variables),
                 "      int v1_out = 0;",
                 sprintf("      for (int v1 = 0; v1 < %d; ++v1) {", n_variables + n_params),
                 "        if (!is_any_fixed(v1)) {",
                 "          int av1 = active_idx[global_to_sens(v1)];",
                 sprintf("          sens1_out[s + %d * (v1_out + n_sens_out * i)] = xi.d(av1).x();", n_variables),
                 "          int v2_out = 0;",
                 sprintf("          for (int v2 = 0; v2 < %d; ++v2) {", n_variables + n_params),
                 "            if (!is_any_fixed(v2)) {",
                 "              int av2 = active_idx[global_to_sens(v2)];",
                 sprintf("              sens2_out[s + %d * (v1_out + n_sens_out * (v2_out + n_sens_out * i))] = xi.d(av1).d(av2);", n_variables),
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
                 "  SET_VECTOR_ELT(ans, 4, diag);",
                 "  UNPROTECT(10);",
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
    "", includings, "", usings, "", "namespace {",
    ode_code, "", jac_code,
    "", observer_code,
    "", "}", "", externC
  )

  writeLines(cpp_text, filename, useBytes = TRUE)

  if (verbose) message("Wrote: ", normalizePath(filename, winslash = "/", mustWork = FALSE))

  # --- Attach attributes ---
  # Reconstruct character Jacobian from sparse triplet format
  jac_matrix_R <- matrix("0", nrow = n_variables, ncol = n_variables,
                         dimnames = list(variables, variables))
  jac_rows <- codegen_result$jac_nnz_rows
  jac_cols <- codegen_result$jac_nnz_cols
  jac_exprs <- codegen_result$jac_nnz_exprs
  if (length(jac_rows) > 0L) {
    jac_matrix_R[cbind(as.integer(jac_rows) + 1L,
                       as.integer(jac_cols) + 1L)] <- as.character(jac_exprs)
  }

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
  attr(modelname, "sparse")        <- use_sparse
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

  # --- Build compile arguments ---
  compile_args <- character(0)
  if (isTRUE(profile))   compile_args <- c(compile_args, "-DCPPODE_PROFILE")
  if (isTRUE(stepTrace)) compile_args <- c(compile_args, "-DCPPODE_STEP_TRACE")

  # KLU auto-tuning: pass codegen-determined settings as compile-time defines
  klu_settings <- codegen_result$klu_settings
  if (!is.null(klu_settings)) {
    klu_btf <- if (isTRUE(klu_settings$use_btf)) 1L else 0L
    klu_ord <- as.integer(klu_settings$ordering)
    compile_args <- c(compile_args,
                      sprintf("-DKLUBTF=%d", klu_btf),
                      sprintf("-DKLUAMD=%d", klu_ord))
    if (verbose) {
      message(sprintf("  KLU settings (codegen): BTF=%s, ordering=%s (nblocks=%d, cv=%.2f)",
                      if (klu_btf) "on" else "off",
                      klu_settings$ordering_name,
                      as.integer(klu_settings$nblocks),
                      klu_settings$cv_row_degree))
    }
  }

  attr(modelname, "compileArgs") <- paste(compile_args, collapse = " ")

  if (compile) {
    compile(modelname, verbose = verbose)
  }
  return(modelname)
}


#' Solver Interface for Ordinary Differential Equation Models
#'
#' @description
#' Numerically integrates a compiled ODE model created by [CppODE()] over a
#' specified time span.
#'
#' ## Sensitivity initial values and `fixed`
#'
#' `sens1ini` and `sens2ini` always refer to the **active** sensitivity
#' parameters, i.e., `attr(model, "dim_names")$sens` minus whatever is listed
#' in `fixed`. This means:
#'
#' - If `fixed = NULL` (default), the active set equals all compile-time
#'   sensitivity names and `sens1ini` must have dimensions
#'   `[n_states, n_active]`.
#' - If `fixed` names some parameters, the active set shrinks accordingly and
#'   `sens1ini`/`sens2ini` need only cover the remaining columns.
#' - Columns in `sens1ini` are matched by name when column names are present,
#'   or by position otherwise (position follows the order of active sens names).
#'
#' This design is intentional: you should not have to supply initial values for
#' parameters whose sensitivities you do not compute.
#'
#' @param model A compiled ODE model object returned by [CppODE()].
#' @param times Numeric vector of time points at which to return the solution.
#'   Must be non-empty and contain only finite values.
#' @param parms Named numeric vector of initial conditions and parameters.
#'   Names must include all of `c(attr(model, "variables"), attr(model, "parameters"))`.
#' @param sens1ini Optional numeric matrix `[n_states, n_active]` (or equivalent
#'   flat vector) of first-order sensitivity initial values, where `n_active` is
#'   the number of non-fixed sensitivity parameters. Row names must match model
#'   variables; column names, if present, must match the active sensitivity names
#'   (i.e., `setdiff(attr(model, "dim_names")$sens, fixed)`).
#'   Default `NULL` uses identity seeding (each state is seeded with respect to
#'   its own initial condition, each parameter with a unit seed).
#' @param sens2ini Optional numeric array `[n_states, n_active, n_active]` (or
#'   flat vector) of second-order sensitivity initial values. Only allowed when
#'   `attr(model, "deriv2") == TRUE`. Default `NULL` uses zero seeding.
#' @param fixed Optional character vector of sensitivity parameter names to treat
#'   as fixed at runtime. These parameters will not have their dual-number
#'   components allocated, so the integrator runs with a strictly smaller AD
#'   state. Names must be a subset of `attr(model, "dim_names")$sens`. Unlike
#'   compile-time `fixed` in [CppODE()], runtime fixed parameters can be changed
#'   between calls without recompilation. Default `NULL` (all parameters active).
#' @param forcings Optional named list of forcing function data. Each element
#'   must be a `data.frame` (or coercible object) with columns `time` and `value`,
#'   or a two-column matrix. Names must match `attr(model, "forcings")`.
#'   Default `NULL`.
#' @param abstol Absolute error tolerance for the integrator. Default `1e-6`.
#' @param reltol Relative error tolerance for the integrator. Default `1e-6`.
#' @param maxprogress Maximum number of integration steps without progress
#'   before an error is raised. Default `10`.
#' @param maxsteps Maximum total number of integration steps. Default `1e6`.
#' @param hini Initial step size; `0` (default) triggers automatic estimation.
#' @param roottol Tolerance for root-finding in root-triggered events. Default `1e-6`.
#' @param maxroot Maximum triggers per root event. Default `1`.
#'
#' @return A named list with components `time`, `variable`, `diagnostics`,
#'   and, when `attr(model, "deriv") == TRUE`, `sens1`, and additionally
#'   `sens2` when `attr(model, "deriv2") == TRUE`. Dimension names of
#'   `sens1`/`sens2` reflect only the active (non-fixed) sensitivity
#'   parameters. The `diagnostics` element is a list with solver statistics
#'   (see [diagnostics()]).
#'
#' @seealso [CppODE()] for model specification and compilation.
#'
#' @export
solveODE <- function(model, times, parms,
                     sens1ini = NULL, sens2ini = NULL,
                     fixed = NULL, forcings = NULL,
                     abstol = 1e-6, reltol = 1e-6,
                     maxprogress = 10L, maxsteps = 1e6L,
                     hini = 0, roottol = 1e-6, maxroot = 1L) {

  ## --- Unpack model attributes ---
  stopifnot(is.character(model), length(model) == 1L)
  required_attrs <- c("variables", "parameters", "forcings", "deriv", "deriv2", "dim_names")
  missing_attrs <- setdiff(required_attrs, names(attributes(model)))
  if (length(missing_attrs))
    stop("'model' is missing attributes: ", paste(missing_attrs, collapse = ", "))

  variables     <- attr(model, "variables")
  parameters    <- attr(model, "parameters")
  forcing_names <- attr(model, "forcings")
  deriv         <- attr(model, "deriv")
  deriv2        <- attr(model, "deriv2")
  all_sens      <- if (deriv) attr(model, "dim_names")$sens else character(0)

  ## --- Runtime fixed: resolve early so sens1ini/sens2ini see the active set ---
  fixed_indices <- integer(0)
  if (!is.null(fixed)) {
    if (!deriv) { warning("'fixed' ignored when deriv = FALSE") }
    else {
      if (!is.character(fixed)) stop("'fixed' must be a character vector")
      bad <- setdiff(fixed, all_sens)
      if (length(bad)) stop("Unknown 'fixed' names: ", paste(bad, collapse = ", "),
                            "\nValid: ", paste(all_sens, collapse = ", "))
      fixed_indices <- match(fixed, all_sens) - 1L  # 0-based for C++
    }
  }
  active_sens <- if (deriv && length(fixed_indices))
    all_sens[-( fixed_indices + 1L)] else all_sens

  ## --- Validate and coerce sens1ini (w.r.t. active sens) ---
  if (!is.null(sens1ini) && !deriv)
    stop("'sens1ini' supplied but model has deriv = FALSE")
  if (!is.null(sens2ini) && !deriv2)
    stop("'sens2ini' supplied but model has deriv2 = FALSE")

  coerce_sens_matrix <- function(x, n_s, n_a, active_nms, arg) {
    if (!is.numeric(x)) stop("'", arg, "' must be numeric")
    if (is.matrix(x) || (is.array(x) && length(dim(x)) == 2)) {
      if (nrow(x) != n_s || ncol(x) != n_a)
        stop(sprintf("'%s' must be [%d, %d] (states x active_sens)", arg, n_s, n_a))
      if (!is.null(rownames(x)) && !setequal(rownames(x), variables))
        stop("'", arg, "' row names must match variables")
      if (!is.null(colnames(x))) {
        if (!setequal(colnames(x), active_nms))
          stop("'", arg, "' column names must match active sens: ", paste(active_nms, collapse = ", "))
        x <- x[variables, active_nms, drop = FALSE]
      }
      as.double(x)
    } else {
      if (length(x) != n_s * n_a)
        stop(sprintf("'%s' must have length %d (n_states * n_active_sens)", arg, n_s * n_a))
      as.double(x)
    }
  }

  n_states <- length(variables)
  n_active <- length(active_sens)

  sens1ini <- if (!is.null(sens1ini))
    coerce_sens_matrix(sens1ini, n_states, n_active, active_sens, "sens1ini")

  if (!is.null(sens2ini)) {
    if (!is.numeric(sens2ini)) stop("'sens2ini' must be numeric")
    if (is.array(sens2ini) && length(dim(sens2ini)) == 3) {
      if (any(dim(sens2ini) != c(n_states, n_active, n_active)))
        stop(sprintf("'sens2ini' must be [%d, %d, %d]", n_states, n_active, n_active))
      dn <- dimnames(sens2ini)
      if (!is.null(dn[[1]]) && !setequal(dn[[1]], variables))
        stop("'sens2ini' dim 1 must match variables")
      if (!is.null(dn[[2]]) && !setequal(dn[[2]], active_sens))
        stop("'sens2ini' dim 2 must match active sens")
      if (!is.null(dn[[3]]) && !setequal(dn[[3]], active_sens))
        stop("'sens2ini' dim 3 must match active sens")
      if (!is.null(dn[[1]]) && !is.null(dn[[2]]) && !is.null(dn[[3]]))
        sens2ini <- sens2ini[variables, active_sens, active_sens, drop = FALSE]
      sens2ini <- as.double(sens2ini)
    } else {
      if (length(sens2ini) != n_states * n_active^2)
        stop(sprintf("'sens2ini' must have length %d", n_states * n_active^2))
      sens2ini <- as.double(sens2ini)
    }
  }

  ## --- times ---
  if (!is.numeric(times) || !length(times) || anyNA(times) || any(!is.finite(times)))
    stop("'times' must be a non-empty finite numeric vector")
  times <- as.double(times)

  ## --- parms ---
  if (!is.numeric(parms) || is.null(names(parms)))
    stop("'parms' must be a named numeric vector")
  required_nms <- c(variables, parameters)
  miss <- setdiff(required_nms, names(parms))
  if (length(miss)) stop("'parms' missing: ", paste(miss, collapse = ", "))
  parms_ordered <- as.double(parms[required_nms])
  if (anyNA(parms_ordered) || any(!is.finite(parms_ordered)))
    stop("'parms' must be finite")

  ## --- forcings ---
  n_forcings <- length(forcing_names)
  if (n_forcings && is.null(forcings))
    stop("Model requires forcings: ", paste(forcing_names, collapse = ", "))

  parse_forcing <- function(nm) {
    f <- forcings[[nm]]
    if (is.matrix(f)) f <- data.frame(time = f[,1L], value = f[,2L])
    else if (!is.data.frame(f)) f <- as.data.frame(f)
    if (!all(c("time","value") %in% names(f)))
      stop("Forcing '", nm, "' needs columns 'time' and 'value'")
    ft <- as.double(f$time); fv <- as.double(f$value)
    if (length(ft) < 2L) stop("Forcing '", nm, "' needs >= 2 time points")
    if (length(ft) != length(fv)) stop("Forcing '", nm, "': length mismatch")
    if (anyNA(ft) || any(!is.finite(ft))) stop("Forcing '", nm, "': non-finite time")
    if (anyNA(fv) || any(!is.finite(fv))) stop("Forcing '", nm, "': non-finite value")
    if (anyDuplicated(ft)) stop("Forcing '", nm, "': duplicate times")
    list(times = ft, values = fv)
  }

  if (!n_forcings) {
    forcing_times_list <- forcing_values_list <- list()
  } else {
    if (!is.list(forcings) || is.null(names(forcings)))
      stop("'forcings' must be a named list")
    miss_f <- setdiff(forcing_names, names(forcings))
    if (length(miss_f)) stop("Missing forcings: ", paste(miss_f, collapse = ", "))
    parsed <- lapply(forcing_names, parse_forcing)
    forcing_times_list  <- lapply(parsed, `[[`, "times")
    forcing_values_list <- lapply(parsed, `[[`, "values")
  }

  ## --- solver options ---
  if (!is.numeric(abstol)  || abstol  <= 0) stop("'abstol' must be positive")
  if (!is.numeric(reltol)  || reltol  <= 0) stop("'reltol' must be positive")
  if (!is.numeric(hini)    || hini    <  0) stop("'hini' must be non-negative")
  if (!is.numeric(roottol) || roottol <= 0) stop("'roottol' must be positive")
  maxprogress <- as.integer(maxprogress); maxsteps <- as.integer(maxsteps); maxroot <- as.integer(maxroot)
  if (maxprogress <= 0L) stop("'maxprogress' must be positive")
  if (maxsteps    <= 0L) stop("'maxsteps' must be positive")
  if (maxroot     <= 0L) stop("'maxroot' must be positive")

  ## --- Call C++ solver ---
  sym_name <- paste0("solve_", as.character(model))
  SYM <- tryCatch(getNativeSymbolInfo(sym_name),
                  error = function(e) stop("Model not loaded. Run compile() first.", call. = FALSE))
  result <- tryCatch(
    .Call(SYM, times, parms_ordered, sens1ini, sens2ini, fixed_indices,
          as.double(abstol), as.double(reltol), maxprogress, maxsteps,
          as.double(hini), as.double(roottol), maxroot,
          forcing_times_list, forcing_values_list),
    error = function(e) stop("ODE solver error: ", e$message, call. = FALSE))

  ## --- Attach dimension names ---
  if (!is.null(result$variable))
    rownames(result$variable) <- variables
  if (!is.null(result$sens1))
    dimnames(result$sens1) <- list(variable = variables, sens = active_sens, time = NULL)
  if (!is.null(result$sens2))
    dimnames(result$sens2) <- list(variable = variables, sens1 = active_sens,
                                   sens2 = active_sens, time = NULL)

  ## --- Handle diagnostics ---
  diag <- result$diagnostics
  if (!is.null(diag) && diag$return_code != 0L) {
    warning(
      "Solver did not complete: ", diag$message,
      "\n  Returning partial results up to t = ", format(diag$t_reached, digits = 6),
      " (", length(result$time), " of ", length(times), " time points).",
      call. = FALSE, immediate. = TRUE
    )
  }

  result
}


#' Print Solver Diagnostics
#'
#' @description
#' Prints a summary of solver diagnostics.
#'
#' @param result A list returned by [solveODE()], containing a `diagnostics` element.
#'
#' @return Invisibly returns the diagnostics list.
#'
#' @examples
#' \dontrun{
#' res <- solveODE(model, times, params)
#' diagnostics(res)
#' }
#'
#' @export
diagnostics <- function(result) {
  UseMethod("diagnostics")
}

#' @export
diagnostics.default <- function(result) {
  diag <- result$diagnostics
  if (is.null(diag)) {
    message("No solver diagnostics available.")
    return(invisible(NULL))
  }

  rc <- diag$return_code
  rc_text <- switch(as.character(rc),
                    "0"  = "Integration was successful.",
                    "1"  = "Maximum number of total steps exceeded.",
                    "2"  = "Maximum number of steps without progress exceeded.",
                    "-1" = "Other solver error.",
                    paste0("Unknown return code: ", rc)
  )

  cat("--------------------  CppODE solver diagnostics  --------------------\n")
  cat(sprintf("  Return code                  : %d\n", rc))
  cat(sprintf("  Message                      : %s\n", rc_text))
  cat("---------------------------------------------------------------------\n")
  cat(sprintf("  Accepted steps               : %d\n", diag$accepted))
  cat(sprintf("  Rejected steps               : %d\n", diag$rejected))
  cat(sprintf("  Function evaluations         : %d\n", diag$fevals))
  cat(sprintf("  Jacobian evaluations         : %d\n", diag$jevals))
  cat(sprintf("  LU factorizations            : %d\n", diag$setups))
  cat(sprintf("  Last step size (successful)  : %g\n", diag$last_dt))
  cat(sprintf("  Last method order            : %d\n", diag$last_order))
  cat(sprintf("  Time reached                 : %g\n", diag$t_reached))
  cat("---------------------------------------------------------------------\n")

  invisible(diag)
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
                   verbose = FALSE, convenient = TRUE, deriv = TRUE, deriv2 = FALSE) {

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
  abind3 <- function(a, b) { d <- dim(a); db <- dim(b); r <- array(0, c(d[1]+db[1], d[2], d[3]), list(c(dimnames(a)[[1]], dimnames(b)[[1]]), dimnames(a)[[2]], dimnames(a)[[3]])); r[1:d[1],,] <- a; r[d[1]+1:db[1],,] <- b; r }
  abind4 <- function(a, b) { d <- dim(a); db <- dim(b); r <- array(0, c(d[1]+db[1], d[2], d[3], d[4]), list(c(dimnames(a)[[1]], dimnames(b)[[1]]), dimnames(a)[[2]], dimnames(a)[[3]], dimnames(a)[[4]])); r[1:d[1],,,] <- a; r[d[1]+1:db[1],,,] <- b; r }

  attachExtras <- function(res, n_obs, ev, ep, type) {
    if (is.null(ev) && is.null(ep)) return(res)
    if (type == "fun") {
      if (!is.null(ev)) res <- rbind(res, t(ev))
      if (!is.null(ep)) res <- rbind(res, matrix(rep(ep, each = n_obs), length(ep), n_obs, dimnames = list(names(ep), NULL)))
    } else if (type == "jac") {
      cs <- dimnames(res)[[2]]; ncs <- length(cs)
      if (!is.null(ev)) res <- abind3(res, array(0, c(ncol(ev), ncs, n_obs), list(colnames(ev), cs, NULL)))
      if (!is.null(ep)) { np <- length(ep); pn <- names(ep); d <- dim(res); new <- array(0, c(d[1]+np, d[2]+np, d[3]), list(c(dimnames(res)[[1]], pn), c(dimnames(res)[[2]], pn), NULL)); new[1:d[1],1:d[2],] <- res; for (k in seq_len(np)) new[d[1]+k,d[2]+k,] <- 1; res <- new }
    } else {
      cs <- dimnames(res)[[2]]; ncs <- length(cs)
      if (!is.null(ev)) res <- abind4(res, array(0, c(ncol(ev), ncs, ncs, n_obs), list(colnames(ev), cs, cs, NULL)))
      if (!is.null(ep)) { np <- length(ep); pn <- names(ep); d <- dim(res); new <- array(0, c(d[1]+np, d[2]+np, d[3]+np, d[4]), list(c(dimnames(res)[[1]], pn), c(dimnames(res)[[2]], pn), c(dimnames(res)[[3]], pn), NULL)); new[1:d[1],1:d[2],1:d[3],] <- res; res <- new }
    }
    res
  }

  # --- Core implementations ---
  fun_impl <- function(vars, params = numeric(0), attach.input = FALSE, fixed = NULL) {
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    funsym <- paste0(modelname, "_eval")
    if (is.loaded(funsym)) {
      out <- .C(funsym, x = as.double(M), y = double(length(outnames) * n_obs), p = as.double(p), n = as.integer(n_obs), k = as.integer(length(innames)), l = as.integer(length(outnames)))
      res <- matrix(out$y, length(outnames), n_obs, dimnames = list(outnames, NULL))
    } else {
      res <- matrix(NA_real_, length(outnames), n_obs, dimnames = list(outnames, NULL))
      for (i in seq_len(n_obs)) { env <- setNames(as.list(c(M[,i], p)), c(innames, parameters)); res[,i] <- vapply(parsed_exprs, function(e) eval(e, env), numeric(1)) }
    }
    attachExtras(res, n_obs, chk$extra_vars, chk$extra_params, "fun")
  }

  jac_impl <- if (deriv && !is.null(sym_jac)) function(vars, params = numeric(0), attach.input = FALSE, fixed = NULL) {
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    fixed_rt <- if (is.null(fixed)) character(0) else intersect(fixed, parameters); dsyms <- setdiff(diff_syms, fixed_rt)
    funsym <- paste0(modelname, "_jacobian"); n_out <- length(outnames); n_sym <- length(diff_syms)
    if (is.loaded(funsym)) {
      out <- .C(funsym, x = as.double(M), jac = double(n_obs * n_out * n_sym), p = as.double(p), n = as.integer(n_obs), k = as.integer(length(innames)), l = as.integer(n_out))
      arr <- array(out$jac, c(n_out, n_sym, n_obs), list(outnames, diff_syms, NULL))
    } else {
      arr <- array(0, c(n_out, n_sym, n_obs), list(outnames, diff_syms, NULL))
      for (i in seq_len(n_obs)) { env <- setNames(as.list(c(M[,i], p)), c(innames, parameters)); for (o in seq_len(n_out)) for (s in seq_len(n_sym)) if (!(diff_syms[s] %in% fixed_rt)) { e <- parsed_jac[[outnames[o], diff_syms[s]]]; if (!is.null(e)) arr[o,s,i] <- eval(e, env) } }
    }
    attachExtras(arr[,dsyms,,drop=FALSE], n_obs, chk$extra_vars, chk$extra_params, "jac")
  }

  hess_impl <- if (deriv2 && !is.null(sym_hess)) function(vars, params = numeric(0), attach.input = FALSE, fixed = NULL) {
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    fixed_rt <- if (is.null(fixed)) character(0) else intersect(fixed, parameters); dsyms <- setdiff(diff_syms, fixed_rt)
    funsym <- paste0(modelname, "_hessian"); n_out <- length(outnames); n_sym <- length(diff_syms)
    if (is.loaded(funsym)) {
      out <- .C(funsym, x = as.double(M), hess = double(n_obs * n_out * n_sym^2), p = as.double(p), n = as.integer(n_obs), k = as.integer(length(innames)), l = as.integer(n_out))
      arr <- array(out$hess, c(n_out, n_sym, n_sym, n_obs), list(outnames, diff_syms, diff_syms, NULL))
    } else {
      arr <- array(0, c(n_out, n_sym, n_sym, n_obs), list(outnames, diff_syms, diff_syms, NULL))
      for (i in seq_len(n_obs)) { env <- setNames(as.list(c(M[,i], p)), c(innames, parameters)); for (o in seq_len(n_out)) { Hmat <- parsed_hess[[outnames[o]]]; for (s1 in seq_len(n_sym)) for (s2 in seq_len(n_sym)) if (!(diff_syms[s1] %in% fixed_rt) && !(diff_syms[s2] %in% fixed_rt)) { e <- Hmat[[diff_syms[s1], diff_syms[s2]]]; if (!is.null(e)) arr[o,s1,s2,i] <- eval(e, env) } } }
    }
    attachExtras(arr[,dsyms,dsyms,,drop=FALSE], n_obs, chk$extra_vars, chk$extra_params, "hess")
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
