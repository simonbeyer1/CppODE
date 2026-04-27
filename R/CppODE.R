#' Generate and Compile a C++ ODE Solver
#'
#' @description
#' Generates C++ source code for a system of ordinary differential equations
#' (ODEs) of the form
#'
#' \deqn{\dot{x}(t) = f\big(x(t), p_{\text{dyn}}\big), \quad x(t_0) = p_{\text{init}},}
#'
#' where the parameter vector \eqn{p = (p_{\text{init}}, p_{\text{dyn}})}
#' concatenates the initial conditions \eqn{p_{\text{init}}} and the
#' dynamic parameters \eqn{p_{\text{dyn}}} that appear in the right-hand
#' side. [solveODE()] takes this combined `p` as the `parms` argument.
#'
#' The generated C++ is compiled into a shared library and loaded for use
#' with [solveODE()]. Optionally, first- and second-order sensitivities of
#' the state trajectory with respect to every component of \eqn{p} are
#' computed via forward-mode automatic differentiation (AD) on dual numbers
#' in a single integration pass.
#'
#' @details
#' ## Methods
#'
#' Four integration methods are available:
#'
#' \tabular{llrl}{
#'   \strong{Method} \tab \strong{Family} \tab \strong{Order} \tab \strong{Type} \cr
#'   \code{"bdf"} \emph{(default)} \tab BDF / NDF                          \tab 5  \tab Stiff     \cr
#'   \code{"adams"}                \tab Adams-Moulton (PECE)               \tab 12 \tab Non-stiff \cr
#'   \code{"rb4"}                  \tab Rosenbrock4 (L-stable, single-step)\tab 4  \tab Stiff     \cr
#'   \code{"tsit5"}                \tab Tsitouras 5(4) (explicit RK, FSAL) \tab 5  \tab Non-stiff \cr
#' }
#'
#' For BDF, `useNDF = TRUE` (default) selects Shampine's NDF kappa
#' modification (Shampine and Reichelt 1997), which is typically about
#' 10\% faster than classical BDF. Adams-Moulton (PECE) requires no
#' Newton iteration and is significantly cheaper per step on non-stiff
#' problems but stalls on stiff ones. Rosenbrock4 is a fourth-order
#' L-stable single-step method (Boost.Odeint origin) that uses one
#' Jacobian evaluation and one LU factorization per step. Tsit5
#' (Tsitouras 2011) is a fifth-order explicit Runge-Kutta method with
#' an embedded fourth-order error estimator and FSAL property; it is
#' the method of choice for non-stiff problems with expensive RHS
#' evaluations.
#'
#' All steppers support dense (Hermite) output, error control, sparse
#' and dense LU factorization, time- and root-triggered events, and
#' optional sensitivities.
#'
#' ## Sensitivity computation
#'
#' If `deriv = TRUE`, state variables and parameters are represented as
#' dual numbers and the right-hand side is evaluated on those dual
#' numbers. Derivative components propagate automatically through every
#' operation, so the integrator solves the same initial-value problem
#' over the dual-number algebra and returns state trajectories and first
#' sensitivities together. With `deriv2 = TRUE`, nested dual numbers
#' additionally yield second-order sensitivities. Variables and
#' parameters listed in `fixed` are stored as plain scalars and do not
#' contribute sensitivity components.
#'
#' Sensitivities are reported with respect to whichever basis the
#' AD seeds are initialised in. By default this is the canonical
#' \eqn{p}-basis (initial conditions stacked on dynamic parameters),
#' but [solveODE()] also accepts a Jacobian \eqn{\Phi'(\theta)} of a
#' user-defined reparametrization \eqn{p = \Phi(\theta)} via
#' `sens1ini` (and \eqn{\Phi''(\theta)} via `sens2ini`), in which case the
#' returned sensitivities are
#' \eqn{\partial x / \partial \theta} and
#' \eqn{\partial^{2} x / \partial \theta^2}.
#' See the "Sensitivity with respect to a
#' reparametrization" section of [solveODE()] for the math and the
#' relevant shapes. The `nStack` argument controls the compile-time
#' upper bound on the number of \eqn{\theta}-directions \eqn{M};
#' passing `nStack = Inf` (the default) selects the heap-allocated
#' version, which imposes no compile-time bound at the cost of dynamic
#' memory allocation.
#'
#' ## Events
#'
#' Events are described by a `data.frame` with columns:
#'
#' | Column   | Description |
#' |:--|:--|
#' | `var`    | Name of the affected variable |
#' | `value`  | Expression applied at the event |
#' | `method` | One of `"replace"`, `"add"`, or `"multiply"` |
#' | `time`   | *(optional)* Event time |
#' | `root`   | *(optional)* Root expression in terms of variables and `time` |
#'
#' Each event must define exactly one of `time` or `root`. Root-triggered
#' events fire when the root expression crosses zero.
#'
#' ## Root function
#'
#' The `rootfunc` argument enables integration termination via root
#' finding:
#'
#' - `"equilibrate"` stops integration when the system reaches steady
#'   state. The steady-state condition checks that all derivatives
#'   (including sensitivities when `deriv = TRUE`) fall below `roottol`.
#' - A character vector of expressions stops integration as soon as any
#'   expression crosses zero. Variables, parameters and `time` may be used.
#'
#' ## Output of the compiled solver
#'
#' The generated solver returns a named list. Output arrays are
#' time-first, with the leading dimension indexing time:
#'
#' - With `deriv = FALSE`: `time` (length \eqn{n_t}) and `variable`
#'   (matrix \eqn{(n_t, n_x)} of \eqn{x_j(t_i)}).
#' - With `deriv = TRUE`: additionally `sens1`, an array of shape
#'   \eqn{(n_t, n_x, n_s)} containing
#'   \eqn{\partial x_j(t_i)/\partial p_k}.
#' - With `deriv2 = TRUE`: additionally `sens2`, an array of shape
#'   \eqn{(n_t, n_x, n_s, n_s)} containing
#'   \eqn{\partial^2 x_j(t_i)/\partial p_k \, \partial p_n}.
#'
#' Here \eqn{n_t} is the number of output time points, \eqn{n_x} the
#' number of state variables, and \eqn{n_s} the number of sensitivity
#' parameters (non-fixed initials and parameters).
#'
#' @param rhs Named character vector of ODE right-hand sides. Names
#'   correspond to state variables.
#' @param events Optional `data.frame` of event specifications. See
#'   Details. Default `NULL`.
#' @param rootfunc Optional root function specification. Either
#'   `"equilibrate"` for steady-state detection, or a character vector of
#'   expressions that trigger termination when crossing zero. Default
#'   `NULL`.
#' @param fixed Optional character vector of state or parameter names
#'   treated as fixed (excluded from sensitivities at compile time).
#' @param forcings Optional character vector of forcing-function names
#'   that appear in `rhs`.
#' @param compile Logical. If `TRUE` (default), compile and load the
#'   generated C++ code.
#' @param modelname Optional base name for the generated source file and
#'   the corresponding C/C++ symbols (e.g. `solve_<modelname>`). If
#'   `NULL`, a random identifier is used.
#' @param outdir Directory in which the generated C++ source file is
#'   written. Defaults to `tempdir()`.
#' @param deriv Logical. If `TRUE` (default), compute first-order
#'   sensitivities via dual numbers.
#' @param deriv2 Logical. If `TRUE`, also compute second-order
#'   sensitivities via nested dual numbers (implies `deriv = TRUE`).
#'   Default `FALSE`.
#' @param nStack Compile-time AD slab width.
#'   \describe{
#'     \item{`Inf` *(default)*}{Heap-allocated AD; the per-call sensitivity
#'       dimension is determined at run time from `ncol(sens1ini)` in
#'       [solveODE()]. Useful when the parameter or theta dimension is only
#'       known at run time. Incompatible with `deriv2 = TRUE`; if both are
#'       supplied, `nStack` is silently switched to the stack default.}
#'     \item{Positive integer `K`}{AD width is fixed to `K` at compile time
#'       (stack-allocated). Suitable when integrating with respect to a
#'       reparametrization \eqn{p = \Phi(\theta)} of known compile-time
#'       upper bound `K`. In [solveODE()], `sens1ini` may have up to `K`
#'       columns.}
#'     \item{`NULL`}{AD width equals the number of non-fixed (state +
#'       parameter) slots; stack-allocated.}
#'   }
#'   The choice between legacy `sens1ini` shape and the full
#'   reparametrization shape is per call at [solveODE()] time, independent
#'   of `nStack`. Requires `deriv = TRUE`.
#' @param derivMode Character; forward-AD backend used to compute
#'   sensitivities. Either `"dual"` *(default)*, the package's in-tree
#'   dual-number backend (with a thread-local arena allocator for the
#'   heap-AD path), or `"fadbad"`, the legacy FADBAD++ backend.
#' @param includeTimeZero Logical. If `TRUE` (default), ensure that
#'   time `0` is included among the integration times.
#' @param useDenseOutput Logical. If `TRUE` (default), use Hermite
#'   dense output for user time points.
#' @param sparse Logical or `NULL`. Controls sparse LU factorization.
#'   `NULL` (default) auto-selects based on Jacobian sparsity; `TRUE`
#'   forces sparse LU; `FALSE` forces dense LU. Sparse LU requires the
#'   KLU linear solver to have been found at install time.
#' @param method Character; integration method. One of `"bdf"`
#'   (default), `"adams"`, `"rb4"` (or `"rosenbrock4"`), or `"tsit5"`.
#'   Use `"bdf"` for stiff problems and `"adams"` or `"tsit5"` for
#'   non-stiff problems; `"rb4"` is an alternative to BDF when the
#'   Jacobian is cheap relative to the right-hand side.
#' @param useNDF Logical. If `TRUE` (default), the BDF corrector uses
#'   Klopfenstein-Shampine NDF kappa coefficients; if `FALSE`, classical
#'   BDF is used. Applies to method `"bdf"`; ignored otherwise.
#' @param profile Logical. If `TRUE`, compile with profiling counters.
#'   Default `FALSE`.
#' @param stepTrace Logical. If `TRUE`, compile the solver to record
#'   per-step diagnostics (order, step size, error norms, cumulative
#'   counters). The trace is returned as the `$trace` element of the
#'   result from [solveODE()] and may be written to CSV via the
#'   `traceFile` argument of that function. Intended for debugging.
#' @param verbose Logical. If `TRUE`, print progress messages.
#'
#' @return
#' The compiled model name (character string), with attributes describing
#' the compiled solver and its symbolic structure:
#'
#' | Attribute     | Type         | Description |
#' |:--|:--|:--|
#' | `equations`   | `character`  | ODE right-hand side definitions |
#' | `srcfile`     | `character`  | Path to the generated C++ source file |
#' | `variables`   | `character`  | Names of the state variables |
#' | `parameters`  | `character`  | Names of model parameters |
#' | `forcings`    | `character`  | Names of forcing functions |
#' | `events`      | `data.frame` | Event specifications, if any |
#' | `rootfunc`    | `character`  | Root function specification, if any |
#' | `fixed`       | `character`  | Names of fixed initial conditions or parameters |
#' | `jacobian`    | `list`       | `f.x` (character matrix) and `f.time` (time derivatives) |
#' | `deriv`       | `logical`    | Whether first-order sensitivities are enabled |
#' | `deriv2`      | `logical`    | Whether second-order sensitivities are enabled |
#' | `sparse`      | `logical`    | Whether sparse LU factorization is used |
#' | `dimNames`    | `list`       | Dimension names for `time`, `variable`, and `sens` |
#' | `compileArgs` | `character`  | Compiler flags |
#'
#' @references
#' Shampine, L. F. and Reichelt, M. W. (1997). The MATLAB ODE Suite.
#' \emph{SIAM Journal on Scientific Computing} \strong{18}(1), 1-22.
#'
#' Soederlind, G. (2003). Digital filters in adaptive time-stepping.
#' \emph{ACM Transactions on Mathematical Software} \strong{29}(1), 1-26.
#'
#' Gustafsson, K., Lundh, M., and Soederlind, G. (1988). A PI stepsize
#' control for the numerical solution of ordinary differential equations.
#' \emph{BIT} \strong{28}(2), 270-287.
#'
#' Tsitouras, Ch. (2011). Runge-Kutta pairs of order 5(4) satisfying only
#' the first column simplifying assumption. \emph{Computers and
#' Mathematics with Applications} \strong{62}(2), 770-775.
#'
#' @example inst/examples/example_ODE.R
#' @importFrom stats setNames
#' @seealso [solveODE()] for the solver interface;
#'   [CVODE()] for the SUNDIALS-backed alternative.
#' @export
CppODE <- function(rhs, events = NULL, rootfunc = NULL, fixed = NULL, forcings = NULL,
                   compile = TRUE, modelname = NULL, outdir = tempdir(),
                   deriv = TRUE, deriv2 = FALSE,
                   nStack = Inf,
                   derivMode = c("dual", "fadbad"),
                   includeTimeZero = TRUE, useDenseOutput = TRUE,
                   sparse = NULL,
                   method = c("bdf", "adams", "rb4", "tsit5"),
                   useNDF = TRUE,
                   profile = FALSE, stepTrace = FALSE, verbose = FALSE) {

  # --- Validate arguments ---
  if (deriv2 && !deriv) {
    warning("deriv2 = TRUE requires deriv = TRUE. Setting deriv = TRUE automatically.")
    deriv <- TRUE
  }
  method <- match.arg(method)
  derivMode <- match.arg(derivMode)
  if (method == "rb4") method <- "rosenbrock4"
  # Both multistep methods ("bdf", "adams") are instantiations of the
  # cppode::multistepper class template, selected at compile time via
  # the multistep_method enum.  The single-step methods (rb4, tsit5)
  # use onestep_controller / onestep_dense_output.
  # The internal helper is_multistep() centralises the dispatch.
  is_multistep <- function(m) m %in% c("bdf", "adams")
  is_explicit  <- function(m) m %in% c("tsit5")

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

  # --- Resolve nStack (compile-time AD slab width) ---
  # Inf (default): heap-allocated AD (dual<double, 0> / F<double, 0>); width
  #                determined at runtime from ncol(sens1ini). Incompatible with
  #                deriv2 -- promoted to NULL silently in that case so the
  #                default does not break second-order use.
  # NULL:          stack-allocated with width = n_total_sens.
  # K (positive integer): stack with width K. The runtime per-call active
  #                sens dimension M = ncol(sens1ini) must satisfy M <= K.
  if (deriv2 && is.numeric(nStack) && length(nStack) == 1L && is.infinite(nStack)) {
    nStack <- NULL  # heap AD is incompatible with deriv2; fall back to stack default
  }
  if (!deriv && is.numeric(nStack) && length(nStack) == 1L && is.infinite(nStack)) {
    nStack <- NULL  # heap AD is meaningful only with deriv = TRUE
  }
  if (is.null(nStack)) {
    nStack_width <- as.integer(n_total_sens)
    is_heap <- FALSE
  } else if (is.numeric(nStack) && length(nStack) == 1L && is.infinite(nStack) && nStack > 0) {
    if (!deriv) stop("'nStack = Inf' requires deriv = TRUE")
    if (deriv2) stop("'nStack = Inf' is not supported with deriv2 = TRUE")
    nStack_width <- 0L  # routes codegen to <double, 0> (heap spec)
    is_heap <- TRUE
  } else if (is.numeric(nStack) && length(nStack) == 1L && is.finite(nStack) && nStack >= 0 &&
             nStack == as.integer(nStack)) {
    if (!deriv) stop("'nStack' is only meaningful when deriv = TRUE")
    nStack_width <- as.integer(nStack)
    is_heap <- FALSE
  } else {
    stop("'nStack' must be NULL, a non-negative integer, or Inf")
  }
  # Codegen helper: under heap AD, every diff() seeding call must pass the
  # runtime size as a second arg so the tangent slab is allocated. Stack AD
  # uses the static-N spec where diff(idx) takes no size arg.
  dyn_arg <- if (is_heap) ", n_sens" else ""

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
    sparse = sparse,
    skip_jacobian = is_explicit(method),
    derivMode = derivMode
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
      rhs_dict = as.list(setNames(rhs, variables)),
      derivMode = derivMode
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
      forcings_list = forcings,
      derivMode = derivMode
    )

    rootfunc_code <- paste(rootfunc_lines, collapse = "\n")
    if (verbose) message("  \u2713 rootfunc generated")
  }

  # --- Generate forcing initialization code ---
  forcing_init_code <- paste(codegen$generate_forcing_init_code(n_forcings, numType, derivMode = derivMode), collapse = "\n")

  # --- C++ includes ---
  includings <- c(
    "#define R_NO_REMAP",
    "#include <R.h>",
    "#include <Rinternals.h>",
    "#include <algorithm>",
    "#include <vector>",
    "#include <cmath>",
    "#include <climits>",
    "#include <cppode/cppode.hpp>"
  )

  # --- Using declarations ---
  # AD width N is bound to nStack_width: equals n_total_sens in the default
  # case, an explicit positive integer when nStack = K, or 0 (heap spec) when
  # nStack = Inf. The runtime per-call active sens dimension M may be smaller
  # than N (with a stack of N empty slots ignored).
  # derivMode selection: "fadbad" -> fadbad::F<double, N> (legacy);
  #                      "dual"   -> cppode::dual<double, N> (custom forward AD,
  #                                  arena-allocated for N=0).
  if (deriv2) {
    if (derivMode == "dual") {
      # Nested dual: cppode::dual<cppode::dual<double, N>, N> mirrors fadbad's
      # nested AD layout. The recursive AD-LU solver and traits already handle
      # arbitrary nesting, so no new specialisations are required.
      usings <- c(
        "using namespace cppode;",
        sprintf("using AD = cppode::dual<double, %d>;", nStack_width),
        sprintf("using AD2 = cppode::dual<cppode::dual<double, %d>, %d>;", nStack_width, nStack_width)
      )
    } else {
      usings <- c(
        "using namespace cppode;",
        sprintf("using AD = fadbad::F<double, %d>;", nStack_width),
        sprintf("using AD2 = fadbad::F<fadbad::F<double, %d>, %d>;", nStack_width, nStack_width)
      )
    }
  } else if (deriv) {
    if (derivMode == "dual") {
      usings <- c(
        "using namespace cppode;",
        sprintf("using AD = cppode::dual<double, %d>;", nStack_width)
      )
    } else {
      usings <- c(
        "using namespace cppode;",
        sprintf("using AD = fadbad::F<double, %d>;", nStack_width)
      )
    }
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
      'extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP sens1iniSEXP, SEXP sens2iniSEXP, SEXP fixedSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP, SEXP hiniSEXP, SEXP root_tolSEXP, SEXP maxrootSEXP, SEXP forcingTimesSEXP, SEXP forcingValuesSEXP, SEXP pidModeSEXP) {',
      modelname
    ),
    "try {",
    "",
    "  // RAII scope for the cppode::dual arena: snapshots the bump pointer on",
    "  // entry and restores it on exit so the arena is recycled between",
    "  // solveODE() calls. No-op for the FADBAD backend (just reads/writes a",
    "  // few thread_local fields). Within one solveODE call the arena grows",
    "  // naturally; live state in x / full_params / stepper buffers points",
    "  // into it and stays valid until this scope ends.",
    "  cppode::dual_arena::scope _cppode_arena_scope;",
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
      sprintf("  const int n_params_all = %d;", n_params),
      sprintf("  const int n_phi_rows   = %d;  // n_states + n_params (full Phi' row count)", n_variables + n_params),
      sprintf("  const int n_sens_total = %d;  // compile-time total (excl. compile-time fixed)", n_total_sens),
      sprintf("  const int n_stack_max  = %s;  // compile-time AD slab width (INT_MAX under heap AD)",
              if (is_heap) "INT_MAX" else as.character(nStack_width)),
      "  // Per-call active sens dimension: from sens1ini's column count when supplied",
      "  // (which is the auto-extended full Phi' shape on the R side -- legacy",
      "  // [n_states, n_active] input is padded with an identity block on the param",
      "  // rows before reaching here), or n_sens_total - n_runtime_fixed when not",
      "  // supplied (identity-fallback seeding).",
      "  const int n_sens = has_sens1ini",
      "      ? static_cast<int>(Rf_ncols(sens1iniSEXP))",
      "      : (n_sens_total - n_runtime_fixed);",
      "  if (n_sens > n_stack_max)",
      "    Rf_error(\"sens1ini has %d columns but the model's compile-time nStack is %d\",",
      "             n_sens, n_stack_max);",
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
      "  // IDX1 / IDX2 index into sens1ini / sens2ini, which are passed as the full",
      "  // Phi'(theta) / Phi''(theta) with first dim = n_phi_rows (state rows + param rows).",
      "  // R-side coerce auto-extends legacy [n_states, n_active] input with an identity",
      "  // block on param rows, so C++ always sees the full shape here.",
      "  auto IDX1 = [n_phi_rows](int g, int v) {",
      "    return g + n_phi_rows * v;",
      "  };"
    )

    if (deriv2) {
      externC <- c(
        externC,
        "  auto IDX2 = [n_phi_rows, n_sens](int g, int v1, int v2) {",
        "    return g + n_phi_rows * (v1 + n_sens * v2);",
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
      "  // Validate sens1ini / sens2ini length against full Phi'/Phi'' shape",
      "  // [n_phi_rows, n_sens] and [n_phi_rows, n_sens, n_sens] respectively.",
      "  if (has_sens1ini && Rf_length(sens1iniSEXP) != n_phi_rows * n_sens)",
      "    Rf_error(\"sens1ini has wrong length: expected n_phi_rows * n_sens = %d * %d = %d, got %d\",",
      "             n_phi_rows, n_sens, n_phi_rows * n_sens, Rf_length(sens1iniSEXP));"
    )

    if (deriv2) {
      externC <- c(
        externC,
        "  if (has_sens2ini && Rf_length(sens2iniSEXP) != n_phi_rows * n_sens * n_sens)",
        "    Rf_error(\"sens2ini has wrong length: expected n_phi_rows * n_sens^2 = %d * %d^2 = %d, got %d\",",
        "             n_phi_rows, n_sens, n_phi_rows * n_sens * n_sens, Rf_length(sens2iniSEXP));",
        "  if (has_sens2ini && !has_sens1ini)",
        "    Rf_error(\"sens2ini requires sens1ini (Phi'' without Phi' is inconsistent)\");"
      )
    }

    externC <- c(externC, "")
  }

  # --- Slab-bind state and parameter tangents -------------------------------
  # For the heap-AD path (cppode::dual<S, 0>) we route every state and
  # parameter dual's tan_ pointer into one contiguous [n_rows × n_sens] slab
  # block per vector. Subsequent .diff(idx, n_sens) calls and codegen-emitted
  # dxdt[i] = expr materialisations then write into the slab — SoA-friendly,
  # zero per-RHS arena traffic, and the W-matrix factorise reads neighbouring
  # tangents from one cache line. For non-dynamic-dual num types (static N
  # dual<S, N!=0>, fadbad::F<>, double) the slab is the empty stub so the
  # prime() calls compile to no-ops.
  if (deriv) {
    externC <- c(
      externC,
      sprintf("  cppode::detail::tangent_slab<%s> _x_slab;", numType),
      sprintf("  cppode::detail::tangent_slab<%s> _full_params_slab;", numType),
      "  if (n_sens > 0) {",
      sprintf("    _x_slab.prime(x, static_cast<unsigned>(%d), static_cast<unsigned>(n_sens));",
              n_variables),
      sprintf("    _full_params_slab.prime(full_params, static_cast<unsigned>(%d), static_cast<unsigned>(n_sens));",
              n_variables + n_params),
      "  }",
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
      "    if (!is_fixed) {",
      "      int si = global_to_sens(i);  // compile-time sens index (-1 if compile-time-fixed)",
      "      int ai = (si >= 0) ? active_idx[si] : -1;  // active index (-1 if any-fixed)",
      "      if (ai >= 0) {",
      "        // First-order sensitivities (inner layer): seed from Phi' or identity",
      "        if (has_sens1ini) {",
      "          if (n_sens > 0) {  // M=0 under reparam: leave F default (no propagation)",
      "            x[i].x().diff(0);  // allocate n_sens components",
      "            for (int av = 0; av < n_sens; ++av)",
      "              x[i].x().d(av) = sens1ini[IDX1(i, av)];",
      "          }",
      "        } else {",
      "          x[i].x().diff(ai);  // identity: d(ai) = 1",
      "        }",
      "        // Second-order sensitivities (outer layer): seed from Phi'' or zeros.",
      "        // FADBAD note: .diff(idx) is DESTRUCTIVE (zeros m_diff, sets [idx]=1,",
      "        // m_depend=true). We therefore call diff(0) ONCE per layer to arm",
      "        // m_depend, then use .d() (non-destructive accessor) to write values.",
      "        if (has_sens2ini) {",
      "          if (n_sens > 0) {",
      "            x[i].diff(0);  // arm outer m_depend",
      "            for (int av1 = 0; av1 < n_sens; ++av1) {",
      "              x[i].d(av1).diff(0);  // arm inner m_depend of m_diff[av1]",
      "              x[i].d(av1).x() = sens1ini[IDX1(i, av1)];  // first-order value",
      "              for (int av2 = 0; av2 < n_sens; ++av2)",
      "                x[i].d(av1).d(av2) = sens2ini[IDX2(i, av1, av2)];",
      "            }",
      "          }",
      "        } else {",
      "          x[i].diff(ai);  // identity/allocate outer (inner m_depend stays false)",
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
      "          if (n_sens > 0) {  // M=0 under reparam: leave F default",
      "            // Seed from Phi'(theta): row i of sens1ini",
      sprintf("            x[i].diff(0%s);  // allocate n_sens components", dyn_arg),
      "            for (int av = 0; av < n_sens; ++av)",
      "              x[i].d(av) = sens1ini[IDX1(i, av)];",
      "          }",
      "        } else {",
      sprintf("          x[i].diff(ai%s);  // identity: d(ai) = 1", dyn_arg),
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
      "        // First-order (inner layer): seed from Phi' or identity fallback",
      "        if (has_sens1ini) {",
      "          if (n_sens > 0) {  // M=0 under reparam: leave F default",
      "            full_params[param_index].x().diff(0);  // allocate n_sens components",
      "            for (int av = 0; av < n_sens; ++av)",
      "              full_params[param_index].x().d(av) = sens1ini[IDX1(global_idx, av)];",
      "          }",
      "        } else {",
      "          full_params[param_index].x().diff(ai);  // identity: dp_i/dp_j = delta_ij",
      "        }",
      "        // Second-order (outer layer): see note in state-seed block.",
      "        if (has_sens2ini) {",
      "          if (n_sens > 0) {",
      "            full_params[param_index].diff(0);  // arm outer m_depend",
      "            for (int av1 = 0; av1 < n_sens; ++av1) {",
      "              full_params[param_index].d(av1).diff(0);  // arm inner m_depend",
      "              full_params[param_index].d(av1).x() = sens1ini[IDX1(global_idx, av1)];",
      "              for (int av2 = 0; av2 < n_sens; ++av2)",
      "                full_params[param_index].d(av1).d(av2) = sens2ini[IDX2(global_idx, av1, av2)];",
      "            }",
      "          }",
      "        } else {",
      "          full_params[param_index].diff(ai);  // identity/allocate outer",
      "        }",
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
      "        if (has_sens1ini) {",
      "          if (n_sens > 0) {  // M=0 under reparam: leave F default",
      "            // Seed from Phi'(theta): row n_states + i of sens1ini",
      sprintf("            full_params[param_index].diff(0%s);  // allocate n_sens components", dyn_arg),
      "            for (int av = 0; av < n_sens; ++av)",
      "              full_params[param_index].d(av) = sens1ini[IDX1(global_idx, av)];",
      "          }",
      "        } else {",
      "          // Identity fallback: dp_i/dp_j = delta_{ij}",
      sprintf("          full_params[param_index].diff(ai%s);", dyn_arg),
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
  # Stepper types: dense or sparse LU, Rosenbrock4 or one of the multistep
  # methods (bdf / adams).  Both multistep methods are instantiations
  # of cppode::multistepper<Method, V, J, R>:
  #
  #   method == "bdf"    -> multistepper<multistep_method::bdf,    V, J, R>
  #   method == "adams"  -> multistepper<multistep_method::adams,  V, J, R>
  #
  # NDF vs BDF is controlled at runtime via set_use_ndf_kappa().
  # Rosenbrock4 (method == "rosenbrock4") follows a separate code path.
  resizer_tag   <- "cppode::initially_resizer"

  # Generate the C++ stepper type for value type V and LU pattern J.
  # Only meaningful for is_multistep(method) -- callers on the rb4 path
  # never invoke this helper.
  make_stepper_type <- function(V, J) {
    method_enum <- switch(method,
      "bdf"   = "cppode::multistep_method::bdf",
      "adams" = "cppode::multistep_method::adams",
      stop(sprintf("internal: unhandled multistep method '%s'", method))
    )
    sprintf("cppode::multistepper<%s, %s, %s, %s>",
            method_enum, V, J, resizer_tag)
  }

  # The multistepper_* stepper type strings are only meaningful for the
  # multistep methods.  For Rosenbrock4 the make_stepper_type() helper
  # would error out on the unknown method name, so we instantiate them
  # lazily -- only for is_multistep(method).
  ms_double <- ms_AD <- ms_AD2 <- NULL
  if (use_sparse) {
    rb4_double <- "rosenbrock4<double, sparse_lu_tag>"
    rb4_AD     <- "rosenbrock4<AD, sparse_lu_tag>"
    rb4_AD2    <- "rosenbrock4<AD2, sparse_lu_tag>"
    if (is_multistep(method)) {
      ms_double <- make_stepper_type("double", "sparse_lu_tag")
      ms_AD     <- make_stepper_type("AD",     "sparse_lu_tag")
      ms_AD2    <- make_stepper_type("AD2",    "sparse_lu_tag")
    }
  } else {
    rb4_double <- "rosenbrock4<double>"
    rb4_AD     <- "rosenbrock4<AD>"
    rb4_AD2    <- "rosenbrock4<AD2>"
    if (is_multistep(method)) {
      ms_double <- make_stepper_type("double", "cppode::dense_lu_tag")
      ms_AD     <- make_stepper_type("AD",     "cppode::dense_lu_tag")
      ms_AD2    <- make_stepper_type("AD2",    "cppode::dense_lu_tag")
    }
  }
  # Tsit5 stepper types (no Jacobian pattern -- explicit method)
  tsit5_double <- "cppode::tsit5<double>"
  tsit5_AD     <- "cppode::tsit5<AD>"
  tsit5_AD2    <- "cppode::tsit5<AD2>"

  if (is_multistep(method)) {
    # ---- Multistep stepper (bdf / adams) ----
    # All multistep methods reuse cppode::multistepper_controller -- the
    # controller is templated on the stepper type and works uniformly
    # with any multistepper instantiation.  The pid_mode integer is
    # parsed from R as 0 (none), 1 (intermediate), or 2 (full); the C++
    # enum cppode::multistepper_controller::pid_mode has the same values,
    # so a static_cast is sufficient.
    ms_type <- if (deriv2) ms_AD2 else if (deriv) ms_AD else ms_double
    pid_setup_lines <- c(
      "  int pid_mode_int = INTEGER(pidModeSEXP)[0];",
      sprintf("  auto pid_mode = static_cast<cppode::multistepper_controller<%s>::pid_mode>(pid_mode_int);",
              ms_type),
      "  controlledStepper.set_pid_mode(pid_mode);",
      sprintf("  controlledStepper.stepper().set_use_ndf_kappa(%s);",
              if (useNDF) "true" else "false"),
      # Slab priming for any AD path (heap dual<T,0> or static-N dual<T,N>).
      # The stepper's prepare_sensitivities is `if constexpr` -gated on
      # is_dynamic_dual<value_type>, so this is a no-op for non-AD and
      # nested-AD types. Must happen BEFORE the std::move into denseStepper
      # below — otherwise the call lands on a moved-from object and the slab
      # inside denseStepper stays unprimed for the whole solve.
      if (deriv) {
        "  controlledStepper.prepare_sensitivities(static_cast<unsigned>(n_sens));"
      } else {
        character()
      }
    )
    # Termination argument for equilibrate (empty string when not used)
    is_equilibrate <- identical(tolower(rootfunc), "equilibrate")
    termination_arg <- if (is_equilibrate) ", ss_termination" else ""

    if (useDenseOutput) {
      stepper_line <- paste(
        c(sprintf("  auto controlledStepper = cppode::multistepper_controller<%s>(abstol, reltol);",
                  ms_type),
          pid_setup_lines,
          "  auto denseStepper = cppode::multistepper_dense_output<decltype(controlledStepper)>(std::move(controlledStepper));"),
        collapse = "\n"
      )
      integrate_line <- sprintf("  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot, dt_est%s);", termination_arg)
    } else {
      stepper_line <- paste(
        c(sprintf("  auto controlledStepper = cppode::multistepper_controller<%s>(abstol, reltol);",
                  ms_type),
          pid_setup_lines),
        collapse = "\n"
      )
      integrate_line <- sprintf("  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot, dt_est%s);", termination_arg)
    }
  } else {
    # ---- Single-step stepper (rb4 or tsit5) ----
    # Both use the generic onestep_controller with Gustafsson PI control.
    # The pidModeSEXP value is ignored; cast to void to suppress warning.
    pid_silencer <- "  (void) pidModeSEXP;"

    # Select the stepper type string based on method and AD level.
    if (method == "tsit5") {
      os_type <- if (deriv2) tsit5_AD2 else if (deriv) tsit5_AD else tsit5_double
    } else {
      # rosenbrock4
      os_type <- if (deriv2) rb4_AD2 else if (deriv) rb4_AD else rb4_double
    }

    # Termination argument for equilibrate (empty string when not used)
    is_equilibrate <- identical(tolower(rootfunc), "equilibrate")
    termination_arg <- if (is_equilibrate) ", cppode::detail::no_dt_estimator{}, ss_termination" else ""

    # Slab priming for the single-step path (any AD level — heap dual<T,0>
    # or static-N dual<T,N>). Mirrors the multistep branch: the call lands
    # on controlledStepper BEFORE the std::move into denseStepper so the
    # slabs reachable via denseStepper are primed for the whole solve.
    # The stepper's prepare_sensitivities is `if constexpr`-gated on
    # is_dynamic_dual<value_type>, so this is a no-op for non-AD and
    # nested-AD types.
    onestep_prep_line <- if (deriv) {
      "  controlledStepper.prepare_sensitivities(static_cast<unsigned>(n_sens));"
    } else {
      character()
    }

    if (useDenseOutput) {
      stepper_line <- paste(
        c(sprintf("  auto controlledStepper = cppode::onestep_controller<%s>(abstol, reltol);", os_type),
          pid_silencer,
          onestep_prep_line,
          "  auto denseStepper = cppode::onestep_dense_output<decltype(controlledStepper)>(std::move(controlledStepper));"),
        collapse = "\n"
      )
      integrate_line <- sprintf("  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot%s);", termination_arg)
    } else {
      stepper_line <- paste(
        c(sprintf("  auto controlledStepper = cppode::onestep_controller<%s>(abstol, reltol);", os_type),
          pid_silencer,
          onestep_prep_line),
        collapse = "\n"
      )
      integrate_line <- sprintf("  integrate_times(controlledStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot%s);", termination_arg)
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
  #
  # Multistep methods (bdf/adams) use cppode_hin, a faithful port of
  # CVODES's cvHin algorithm.  Single-step methods (rb4, tsit5) use the
  # unified estimate_initial_dt with a method-specific ydd closure: analytic
  # (J*f + dfdt) for rb4, FD for tsit5.  order=1 is the right regime for
  # all of these -- BDF/NDF/Adams start at q=1 by design, and for
  # single-step methods the HNW formula with higher p would overestimate h
  # on stiff problems.
  method_order <- 1L

  if (is_multistep(method)) {
    # cppode_hin -- needs only `sys`; no compute_ydd closure, no order param.
    estimate_dt_block <- c(
      sprintf("  // --- Determine initial dt (%s, CVODES cvHin port) ---", method),
      sprintf("  %s dt;", numType),
      "  if (hini == 0.0) {",
      sprintf("    dt = odeint_utils::cppode_hin<%s>(", numType),
      "      sys, x, times.front(),",
      "      odeint_utils::scalar_value(times.back()),",
      "      abstol, reltol);",
      "  } else {",
      "    dt = hini;",
      "  }"
    )
  } else {
    if (is_explicit(method)) {
      compute_ydd_lines <- c(
        sprintf("  // --- compute_ydd (FD fallback for explicit method) ---"),
        "  auto compute_ydd = odeint_utils::make_fd_ydd(sys);"
      )
    } else if (use_sparse) {
      compute_ydd_lines <- c(
        "  // --- compute_ydd (analytic: dfdt + sparse J*f) ---",
        "  auto compute_ydd = [&](const auto& x_, auto t_, const auto& f_,",
        "                          double /*h_trial*/, auto& ydd_) {",
        sprintf("    cppode::csc_matrix<%s> J_init;", numType),
        sprintf("    std::vector<%s> dfdt_(x_.size());", numType),
        "    jac(x_, J_init, t_, dfdt_);",
        "    ydd_ = dfdt_;",
        "    cppode::csc_matvec_add(J_init, f_, ydd_);",
        "  };"
      )
    } else {
      compute_ydd_lines <- c(
        "  // --- compute_ydd (analytic: dfdt + dense J*f) ---",
        "  auto compute_ydd = [&](const auto& x_, auto t_, const auto& f_,",
        "                          double /*h_trial*/, auto& ydd_) {",
        sprintf("    cppode::dense_matrix<%s> J_init(x_.size(), x_.size());", numType),
        sprintf("    std::vector<%s> dfdt_(x_.size());", numType),
        "    jac(x_, J_init, t_, dfdt_);",
        "    ydd_ = dfdt_;",
        "    for (std::size_t i = 0; i < x_.size(); ++i)",
        "      for (std::size_t j = 0; j < x_.size(); ++j)",
        "        ydd_[i] += J_init(i,j) * f_[j];",
        "  };"
      )
    }

    estimate_dt_block <- c(
      compute_ydd_lines,
      sprintf("  // --- Determine initial dt (%s, order=%d) ---", method, method_order),
      sprintf("  %s dt;", numType),
      "  if (hini == 0.0) {",
      sprintf("    dt = odeint_utils::estimate_initial_dt<%s>(", numType),
      "      sys, compute_ydd, x, times.front(),",
      "      odeint_utils::scalar_value(times.back()),",
      sprintf("      abstol, reltol, /*order=*/%d);", method_order),
      "  } else {",
      "    dt = hini;",
      "  }"
    )
  }

  # --- Multistep methods: dt re-estimator lambda for event restarts ---
  #
  # The lambda is passed to integrate_times{,_dense} as the DtEstimator and
  # invoked by EventEngine::init_stepper_after_event whenever a multistep
  # stepper discards its Nordsieck history at an event.  We re-estimate h0
  # from scratch via cppode_hin, passing times.back() as the upper-bound
  # hint so the geometric-mean seed isn't collapsed to zero on the
  # remaining integration window.
  dt_est_block <- character(0)
  if (is_multistep(method)) {
    dt_est_block <- c(
      "  // --- Multistep event restart: re-estimate dt from post-event state ---",
      "  auto dt_est = [&](auto& x_ev, auto t_ev) {",
      sprintf("    return odeint_utils::cppode_hin<%s>(", numType),
      "      sys, x_ev, t_ev,",
      "      odeint_utils::scalar_value(times.back()),",
      "      abstol, reltol);",
      "  };"
    )
  }

  # The heap-dual slab priming used to live here, but had to move into
  # pid_setup_lines so it runs before the std::move(controlledStepper)
  # into denseStepper. See is_multistep branch above.

  externC <- c(externC,
               stepper_line, "",
               estimate_dt_block,
               dt_est_block,
               "",
               "  // --- Integration (catch recoverable errors for partial results) ---",
               "  std::string solver_message;",
               "  try {",
               paste0("    ", integrate_line),
               "  } catch (const cppode::no_progress_error& e) {",
               "    // StepChecker sets RC_TOO_MUCH_WORK / RC_CONV_FAILURE before",
               "    // throwing, but the dense-output wrappers' failed_step_checker",
               "    // throws the same exception type without touching StepChecker.",
               "    // Guarantee a non-zero return code in that case.",
               "    if (checker.return_code() == cppode::RC_SUCCESS)",
               "      checker.set_return_code(cppode::RC_CONV_FAILURE);",
               "    solver_message = e.what();",
               "  } catch (const std::runtime_error& e) {",
               "    // KLU factor/analyze failures and similar linear-solver issues.",
               "    checker.set_return_code(cppode::RC_LSETUP_FAIL);",
               "    solver_message = e.what();",
               "  } catch (const std::exception& e) {",
               "    // Anything else that derives from std::exception (bad_alloc,",
               "    // overflow_error from a wild RHS, ...) -- classify as unclassified",
               "    // but still return partial results rather than aborting R.",
               "    checker.set_return_code(cppode::RC_UNRECOGNIZED_ERR);",
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
               "",
               "  // --- Build trace list from the step-trace buffer ---",
               "  // Returns an empty named list when the model was built without",
               "  // -DCPPODE_STEP_TRACE (buffer never populated) or when this solve",
               "  // produced zero rows.  The buffer is cleared after marshalling so",
               "  // repeated solveODE() calls start with a fresh trace.",
               "  SEXP trace_list;",
               "  {",
               "    auto& tb = cppode::ndf_detail::get_trace_buffer();",
               "    const R_xlen_t n_trace = static_cast<R_xlen_t>(tb.size());",
               "    constexpr int n_cols = 18;",
               "    static const char* col_names[n_cols] = {",
               "      \"nst\",\"t\",\"h\",\"q\",\"dsm\",\"acnrm\",\"acnrm_state\",\"tq2\",",
               "      \"gamma\",\"gamrat\",\"newton_conv\",\"mode\",\"nfe\",\"njev\",",
               "      \"nsetups\",\"setup_reason\",\"pece_iters\",\"pece_diverged\"",
               "    };",
               "    trace_list = PROTECT(Rf_allocVector(VECSXP, n_cols));",
               "    SEXP tn = PROTECT(Rf_allocVector(STRSXP, n_cols));",
               "    for (int i = 0; i < n_cols; ++i)",
               "      SET_STRING_ELT(tn, i, Rf_mkChar(col_names[i]));",
               "    Rf_setAttrib(trace_list, R_NamesSymbol, tn);",
               "    UNPROTECT(1); // tn",
               "",
               "    auto put_int = [&](int slot, const std::vector<int>& v) {",
               "      SEXP s = PROTECT(Rf_allocVector(INTSXP, n_trace));",
               "      int* p = INTEGER(s);",
               "      for (R_xlen_t i = 0; i < n_trace; ++i) p[i] = v[i];",
               "      SET_VECTOR_ELT(trace_list, slot, s);",
               "      UNPROTECT(1);",
               "    };",
               "    auto put_dbl = [&](int slot, const std::vector<double>& v) {",
               "      SEXP s = PROTECT(Rf_allocVector(REALSXP, n_trace));",
               "      double* p = REAL(s);",
               "      for (R_xlen_t i = 0; i < n_trace; ++i) p[i] = v[i];",
               "      SET_VECTOR_ELT(trace_list, slot, s);",
               "      UNPROTECT(1);",
               "    };",
               "    auto put_str = [&](int slot, const std::vector<std::string>& v) {",
               "      SEXP s = PROTECT(Rf_allocVector(STRSXP, n_trace));",
               "      for (R_xlen_t i = 0; i < n_trace; ++i)",
               "        SET_STRING_ELT(s, i, Rf_mkChar(v[i].c_str()));",
               "      SET_VECTOR_ELT(trace_list, slot, s);",
               "      UNPROTECT(1);",
               "    };",
               "    put_int(0,  tb.nst);         put_dbl(1,  tb.t);",
               "    put_dbl(2,  tb.h);           put_int(3,  tb.q);",
               "    put_dbl(4,  tb.dsm);         put_dbl(5,  tb.acnrm);",
               "    put_dbl(6,  tb.acnrm_state); put_dbl(7,  tb.tq2);",
               "    put_dbl(8,  tb.gamma);       put_dbl(9,  tb.gamrat);",
               "    put_int(10, tb.newton_conv); put_str(11, tb.mode);",
               "    put_int(12, tb.nfe);         put_int(13, tb.njev);",
               "    put_int(14, tb.nsetups);     put_str(15, tb.setup_reason);",
               "    put_int(16, tb.pece_iters);  put_int(17, tb.pece_diverged);",
               "    tb.clear();",
               "  }",
               ""
  )
  # --- Copy back results - CONSISTENT LIST OUTPUT ---
  if (!deriv) {
    # deriv = FALSE: list(time, variable, diagnostics, trace)
    externC <- c(externC,
                 "  // --- Return list(time, variable, diagnostics, trace) ---",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 4));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"variable\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"diagnostics\"));",
                 "  SET_STRING_ELT(names, 3, Rf_mkChar(\"trace\"));",
                 "  Rf_setAttrib(ans, R_NamesSymbol, names);",
                 "",
                 "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out));",
                 sprintf("  SEXP variable_mat = PROTECT(Rf_allocMatrix(REALSXP, n_out, %d));", n_variables),
                 "  double* time_out = REAL(time_vec);",
                 "  double* variable_out = REAL(variable_mat);",
                 "",
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i];",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      variable_out[i + n_out * s] = y[i * %d + s];", n_variables),
                 "    }",
                 "  }",
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, variable_mat);",
                 "  SET_VECTOR_ELT(ans, 2, diag);",
                 "  SET_VECTOR_ELT(ans, 3, trace_list);",
                 "  UNPROTECT(7);  // trace_list, diag, diag_names, time_vec, variable_mat, ans, names",
                 "  return ans;")

  } else if (!deriv2) {
    # deriv = TRUE, deriv2 = FALSE: list(time, variable, sens1).
    # sens1 has n_sens columns; the av-th column of Phi' (auto-extended on
    # the R side for legacy input, direct for full Phi') drives AD slot av,
    # so output reads xi.d(av) directly.
    externC <- c(externC,
                 "  // --- Return list(time, variable, sens1) ---",
                 "  int n_sens_out = n_sens;",
                 "",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 5));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"variable\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"sens1\"));",
                 "  SET_STRING_ELT(names, 3, Rf_mkChar(\"diagnostics\"));",
                 "  SET_STRING_ELT(names, 4, Rf_mkChar(\"trace\"));",
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
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i].x();",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                 "      variable_out[i + n_out * s] = xi.x();",
                 "      for (int av = 0; av < n_sens_out; ++av)",
                 sprintf("        sens1_out[i + n_out * (s + %d * av)] = xi.d(av);", n_variables),
                 "    }",
                 "  }",
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, variable_mat);",
                 "  SET_VECTOR_ELT(ans, 2, sens1_arr);",
                 "  SET_VECTOR_ELT(ans, 3, diag);",
                 "  SET_VECTOR_ELT(ans, 4, trace_list);",
                 "  UNPROTECT(9);  // trace_list, diag, diag_names, time_vec, variable_mat, sens1_dim, sens1_arr, ans, names",
                 "  return ans;")

  } else {
    # deriv2 = TRUE: list(time, variable, sens1, sens2). Same direct-AD-slot
    # output pattern as the deriv-only branch, extended to the nested layer.
    externC <- c(externC,
                 "  // --- Return list(time, variable, sens1, sens2) ---",
                 "  int n_sens_out = n_sens;",
                 "",
                 "  SEXP ans = PROTECT(Rf_allocVector(VECSXP, 6));",
                 "  SEXP names = PROTECT(Rf_allocVector(STRSXP, 6));",
                 "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
                 "  SET_STRING_ELT(names, 1, Rf_mkChar(\"variable\"));",
                 "  SET_STRING_ELT(names, 2, Rf_mkChar(\"sens1\"));",
                 "  SET_STRING_ELT(names, 3, Rf_mkChar(\"sens2\"));",
                 "  SET_STRING_ELT(names, 4, Rf_mkChar(\"diagnostics\"));",
                 "  SET_STRING_ELT(names, 5, Rf_mkChar(\"trace\"));",
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
                 "  for (int i = 0; i < n_out; ++i) {",
                 "    time_out[i] = result_times[i].x().x();",
                 sprintf("    for (int s = 0; s < %d; ++s) {", n_variables),
                 sprintf("      %s& xi = y[i * %d + s];", numType, n_variables),
                 "      variable_out[i + n_out * s] = xi.x().x();",
                 "      for (int av1 = 0; av1 < n_sens_out; ++av1) {",
                 sprintf("        sens1_out[i + n_out * (s + %d * av1)] = xi.d(av1).x();", n_variables),
                 "        for (int av2 = 0; av2 < n_sens_out; ++av2)",
                 sprintf("          sens2_out[i + n_out * (s + %d * (av1 + n_sens_out * av2))] = xi.d(av1).d(av2);", n_variables),
                 "      }",
                 "    }",
                 "  }",
                 "",
                 "  SET_VECTOR_ELT(ans, 0, time_vec);",
                 "  SET_VECTOR_ELT(ans, 1, variable_mat);",
                 "  SET_VECTOR_ELT(ans, 2, sens1_arr);",
                 "  SET_VECTOR_ELT(ans, 3, sens2_arr);",
                 "  SET_VECTOR_ELT(ans, 4, diag);",
                 "  SET_VECTOR_ELT(ans, 5, trace_list);",
                 "  UNPROTECT(11);  // trace_list + previous 10",
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
  attr(modelname, "nStack")        <- if (is_heap) Inf else as.numeric(nStack_width)
  attr(modelname, "derivMode")     <- derivMode
  attr(modelname, "sparse")        <- use_sparse
  attr(modelname, "method")        <- method
  attr(modelname, "useNDF")        <- useNDF
  # Dimension names -- under reparametrization the sens columns are theta slots
  # The sens dim defaults to model-parameter names (legacy / identity seeding
  # basis). solveODE() overrides this per call when sens1ini is supplied with
  # full Phi'(theta) shape (uses colnames(sens1ini) or theta1..M).
  if (deriv) {
    attr(modelname, "dimNames") <- list(
      time = "time",
      variable = variables,
      sens = c(sens_initials, sens_params)
    )
  } else {
    attr(modelname, "dimNames") <- list(
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
