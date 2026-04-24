#' Compile an ODE model against the SUNDIALS CVODE(S) library
#'
#' @description
#' Generates and compiles a C++ solver for the given ODE system against
#' **SUNDIALS CVODE** (or **CVODES** when `deriv = TRUE`), mirroring the
#' interface of [CppODE()] so that compiled models can be run with the
#' same [solveODE()] interface.
#'
#' This is intended primarily for direct, apples-to-apples comparison of
#' CppODE's built-in steppers against the CVODE(S) reference implementation.
#' The generated model links against the system-installed
#' `libsundials_cvodes` (and, when a sparse Jacobian is selected,
#' `libsundials_sunlinsolklu` + `libklu`).
#'
#' Like [CppODE()], the symbolic Jacobian `df/dx` is derived via SymPy at
#' code-generation time and emitted as analytic C++.  When `deriv = TRUE`,
#' forward-mode sensitivities use CVODES' `CVodeSensInit1` with an
#' **analytic** sensitivity RHS:
#' \deqn{\dot S_k = J(x,p)\,S_k + \partial f / \partial p_k}
#' (with the `df/dp` term present only for sens slots that correspond to
#' parameters — initial-condition sens slots contribute only through
#' `J * S_k`).  This matches the work CppODE does via forward-mode AD.
#'
#' ## Scope
#'
#' The CVODE backend currently supports:
#' * `method` may be `"bdf"` or `"adams"` (CVODE's two multistep families).
#' * First-order forward sensitivities via `deriv = TRUE`.
#' * Dense or KLU-sparse linear solver (auto-selected or forced via `sparse`).
#' * `fixed` at both compile and run time, with the same semantics as
#'   [CppODE()].
#' * `forcings` — PCHIP-interpolated time-dependent inputs.  Forcings do
#'   not contribute to sensitivities (they are data, not parameters).
#' * `rootfunc` — integration termination via `CVodeRootInit`.
#' * `events` — time- and root-triggered.  Time events apply via
#'   `CVodeReInit`; root events via `CVodeRootInit` + `CVodeGetRootInfo`
#'   dispatch on `CV_ROOT_RETURN`.  Sensitivities are corrected through
#'   the full first-order saltation: time events use the explicit
#'   `∂t_e/∂p_k` you supply in the event expression, root events use the
#'   IFT-derived `dt_root/dp_k = −(∂r/∂p_k + Σ ∂r/∂x_i · S_i[k]) / ġ`.
#'
#' The following features of [CppODE()] are **not** available via the CVODE
#' backend: second-order sensitivities, `profile`, and the
#' `"msoda"`/`"rb4"`/`"tsit5"` methods.
#'
#' ## Step trace (`stepTrace = TRUE`)
#'
#' Compiles the solver with `-DCVODE_STEP_TRACE`, driving integration in
#' `CV_ONE_STEP` mode (with `CVodeSetStopTime` + `CVodeGetDky` for
#' user-time dense output).  Each accepted internal step appends a row
#' into the shared trace buffer declared in `cppode_step_trace.hpp`;
#' the row schema mirrors CppODE's multistepper trace.  The buffer is
#' marshalled into the `$trace` element of the result list returned to R
#' and [solveODE()] exposes it as a `data.frame` (optionally writing a
#' CSV when `traceFile` is supplied).  **Not supported together with
#' `events` or `rootfunc`** — in those cases the trace macro is silently
#' a no-op and the solver falls back to the standard `CV_NORMAL` loop.
#'
#' @inheritParams CppODE
#' @param method One of `"bdf"` (default) or `"adams"`.
#' @param stepTrace Logical. If `TRUE`, emit per-step diagnostics to
#'   `cvode_trace.csv` in the current working directory. See details.
#'
#' @return
#' A compiled model name (character string) with the same set of attributes
#' as a model returned by [CppODE()], so it can be passed directly to
#' [solveODE()].  The attribute `backend` is set to `"cvode"`.
#'
#' @seealso [CppODE()], [solveODE()].
#' @export
CVODE <- function(rhs, events = NULL, rootfunc = NULL, fixed = NULL, forcings = NULL,
                  compile = TRUE, modelname = NULL, outdir = tempdir(),
                  deriv = FALSE,
                  ntheta = NULL,
                  sparse = NULL,
                  method = c("bdf", "adams"),
                  stepTrace = FALSE,
                  verbose = FALSE) {

  method <- match.arg(method)

  # --- Availability check (populated by configure at install time) ---
  if (!isTRUE(cvodeConfig$available)) {
    stop(
      "The CVODE backend was disabled at install time because SUNDIALS ",
      "(>= 6.0) was not found on the build host.\n",
      "  Install the SUNDIALS development headers and re-install CppODE:\n",
      "    Debian/Ubuntu : sudo apt install libsundials-dev\n",
      "    Fedora        : sudo dnf install sundials-devel\n",
      "    macOS (brew)  : brew install sundials\n",
      "  Then: R CMD INSTALL <path/to/CppODE>",
      call. = FALSE)
  }

  # --- Normalize rhs (same as CppODE) ---
  rhs <- unclass(rhs)
  rhs <- gsub("\n", "", rhs)
  rhs <- sanitizeExprs(rhs)
  variables <- names(rhs)
  if (is.null(variables) || any(!nzchar(variables)))
    stop("'rhs' must be a named character vector")

  # --- Identify parameters via getSymbols (same helper as CppODE) ---
  # Collect symbols from rhs and any event/rootfunc expressions too,
  # so params captures everything the generated code will reference.
  all_expressions <- rhs
  if (!is.null(events)) {
    bad <- which(!xor(!is.na(events$time), !is.na(events$root)))
    if (length(bad) > 0) {
      stop(sprintf(
        "Each event must define exactly one of 'time' or 'root'. Invalid event(s): %s",
        paste(bad, collapse = ", ")))
    }
    all_expressions <- c(all_expressions,
                         events$value,
                         if ("time" %in% names(events)) events$time,
                         if ("root" %in% names(events)) events$root)
  }
  if (!is.null(rootfunc) && !identical(tolower(rootfunc), "equilibrate")) {
    all_expressions <- c(all_expressions, rootfunc)
  }
  symbols <- getSymbols(all_expressions)

  if (is.null(forcings)) forcings <- character(0)
  if (length(forcings) > 0) {
    unknown_forcings <- setdiff(forcings, symbols)
    if (length(unknown_forcings) > 0)
      stop("Unknown forcing symbols: ", paste(unknown_forcings, collapse = ", "))
    forcing_states <- intersect(forcings, variables)
    if (length(forcing_states) > 0)
      stop("Forcing names cannot be state variables: ", paste(forcing_states, collapse = ", "))
  }

  params <- setdiff(symbols, c(variables, forcings, "time"))

  # --- Handle fixed ---
  if (is.null(fixed)) fixed <- character(0)
  fixed_initials <- if (deriv) intersect(fixed, variables) else character(0)
  fixed_params   <- if (deriv) intersect(fixed, params)    else character(0)
  sens_initials  <- if (deriv) setdiff(variables, fixed_initials) else character(0)
  sens_params    <- if (deriv) setdiff(params,    fixed_params)   else character(0)
  sens_names     <- c(sens_initials, sens_params)
  n_total_sens   <- length(sens_names)

  # --- Resolve ntheta (compile-time theta dimension) ---
  has_reparam <- !is.null(ntheta)
  if (has_reparam) {
    if (!is.numeric(ntheta) || length(ntheta) != 1L || ntheta < 0)
      stop("'ntheta' must be a single non-negative integer")
    if (!deriv) stop("'ntheta' is only meaningful when deriv = TRUE")
    ntheta_resolved <- as.integer(ntheta)
  } else {
    ntheta_resolved <- as.integer(n_total_sens)
  }

  # --- Unique model name ---
  if (is.null(modelname)) {
    modelname <- paste(c("c", sample(c(letters, 0:9), 8, TRUE)), collapse = "")
  }

  if (!dir.exists(outdir)) stop("outdir does not exist: ", outdir)

  # --- Early KLU check: explicit sparse = TRUE with no KLU is fatal ---
  if (isTRUE(sparse) && !isTRUE(cvodeConfig$klu_available)) {
    stop(
      "sparse = TRUE requested but the KLU linear solver was not available\n",
      "at install time.  Install SuiteSparse development headers and\n",
      "re-install CppODE:\n",
      "    Debian/Ubuntu : sudo apt install libsuitesparse-dev\n",
      "    Fedora        : sudo dnf install suitesparse-devel\n",
      "    macOS (brew)  : brew install suite-sparse",
      call. = FALSE)
  }
  # Auto-selected sparse without KLU → force dense.
  sparse_for_codegen <- sparse
  if (is.null(sparse) && !isTRUE(cvodeConfig$klu_available)) {
    sparse_for_codegen <- FALSE
  }

  # --- Codegen ---
  codegen <- get_codegenCVODE_py()
  if (verbose) message("Generating CVODE C++ source...")

  res <- codegen$generate_cvode_cpp(
    rhs_dict = as.list(setNames(rhs, variables)),
    params_list = params,
    modelname = modelname,
    outdir = normalizePath(outdir, winslash = "/", mustWork = FALSE),
    deriv = deriv,
    fixed_states = fixed_initials,
    fixed_params = fixed_params,
    sparse = sparse_for_codegen,
    method = method,
    forcings_list = forcings,
    events = events,
    rootfunc = rootfunc,
    ntheta = ntheta_resolved,
    has_reparam = has_reparam,
    version = as.character(utils::packageVersion("CppODE"))
  )

  use_sparse <- isTRUE(res$use_sparse)
  if (use_sparse && verbose) {
    message(sprintf("  Sparse Jacobian detected (%d states, %d nnz)",
                    length(variables), length(res$jac_nnz_rows)))
  }

  # --- Build jacobian matrix (char) for attr, like CppODE ---
  jac_matrix_R <- matrix("0", nrow = length(variables), ncol = length(variables),
                         dimnames = list(variables, variables))
  if (length(res$jac_nnz_rows)) {
    jac_matrix_R[cbind(as.integer(res$jac_nnz_rows) + 1L,
                       as.integer(res$jac_nnz_cols) + 1L)] <- as.character(res$jac_nnz_exprs)
  }

  # --- Attributes (mirror CppODE so solveODE works unchanged) ---
  attr(modelname, "equations")   <- rhs
  attr(modelname, "srcfile")     <- normalizePath(res$srcfile, winslash = "/", mustWork = FALSE)
  attr(modelname, "variables")   <- variables
  attr(modelname, "parameters")  <- params
  attr(modelname, "forcings")    <- forcings
  attr(modelname, "events")      <- events
  attr(modelname, "rootfunc")    <- rootfunc
  attr(modelname, "fixed")       <- c(fixed_initials, fixed_params)
  attr(modelname, "jacobian")    <- list(f.x = jac_matrix_R, f.time = unlist(res$time_derivs))
  attr(modelname, "deriv")       <- isTRUE(deriv)
  attr(modelname, "deriv2")      <- FALSE
  attr(modelname, "ntheta")      <- ntheta_resolved
  attr(modelname, "has_reparam") <- has_reparam
  attr(modelname, "sparse")      <- use_sparse
  attr(modelname, "method")      <- method
  attr(modelname, "useNDF")      <- NA  # not meaningful for CVODE
  attr(modelname, "backend")     <- "cvode"

  attr(modelname, "dim_names") <- if (deriv) {
    sens_col_names <- if (has_reparam) sprintf("theta%d", seq_len(ntheta_resolved))
                      else              sens_names
    list(time = "time", variable = variables, sens = sens_col_names)
  } else {
    list(time = "time", variable = variables)
  }

  # --- Compile args: codegen preprocessor defs (+ -DCVODE_KLU in sparse mode)
  # plus the SUNDIALS include path discovered at install time. Linker flags
  # come from `cvodeConfig` (populated by ./configure), not from codegen.
  compile_args <- c(unlist(res$compile_defs), cvodeConfig$cflags)
  link_libs    <- cvodeConfig$libs
  if (use_sparse) {
    compile_args <- c(compile_args, cvodeConfig$klu_cflags)
    link_libs    <- paste(link_libs, cvodeConfig$klu_libs)
  }
  if (isTRUE(stepTrace)) {
    compile_args <- c(compile_args, "-DCVODE_STEP_TRACE")
  }
  attr(modelname, "compileArgs") <- paste(compile_args[nzchar(compile_args)], collapse = " ")
  attr(modelname, "linkArgs")    <- link_libs

  if (compile) {
    compile(modelname, verbose = verbose)
  }
  modelname
}
