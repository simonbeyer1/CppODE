#' Generate and Compile an ODE Solver Linked Against SUNDIALS CVODE(S)
#'
#' @description
#' Generates and compiles a C++ ODE solver that links against the
#' system-installed SUNDIALS CVODE library (or CVODES when
#' `deriv = TRUE`) for a system of ordinary differential equations of
#' the form
#'
#' \deqn{\dot{x}(t) = f\big(x(t), p_{\text{dyn}}\big), \quad x(t_0) = p_{\text{init}},}
#'
#' where the parameter vector \eqn{p = (p_{\text{init}}, p_{\text{dyn}})}
#' concatenates the initial conditions \eqn{p_{\text{init}}} and the
#' dynamic parameters \eqn{p_{\text{dyn}}} that appear in the right-hand
#' side. [solveODE()] takes this combined `p` as the `parms` argument.
#'
#' The compiled model exposes the same interface as a model produced by
#' [CppODE()] and can be passed directly to [solveODE()].
#'
#' @details
#' The symbolic Jacobian \eqn{df/dx} is derived via SymPy at code
#' generation time and emitted as analytic C++. With `deriv = TRUE`,
#' first-order forward sensitivities use the analytic sensitivity
#' right-hand side \eqn{\dot S_k = J(x, p)\, S_k + \partial f / \partial p_k},
#' so the work matches the AD-based path in [CppODE()] up to the
#' integrator's internals.
#'
#' Like [CppODE()], the CVODE backend also supports sensitivities with
#' respect to a user-defined reparametrization \eqn{p = \Phi(\theta)}
#' by seeding the columns of \eqn{S_k(0)} with the columns of
#' \eqn{\Phi'(\theta)} (passed via `sens1ini` to [solveODE()]). The
#' returned trajectories then carry \eqn{\partial x / \partial \theta}
#' directly. See [solveODE()] for the chain-rule derivation and the
#' accepted shapes of `sens1ini`. Second-order sensitivities are not
#' available through this backend.
#'
#' SUNDIALS (>= 6.0) must be available on the build host at install time;
#' otherwise [CVODE()] raises an informative error at the first call.
#' KLU (used for sparse Jacobians) is detected the same way and is
#' required when `sparse = TRUE`.
#'
#' ## Scope
#'
#' The CVODE backend supports:
#' * `method` is `"bdf"` or `"adams"` (CVODE's two multistep families);
#' * first-order forward sensitivities via `deriv = TRUE`;
#' * dense or KLU-sparse linear solver (auto-selected or forced with
#'   `sparse`);
#' * `fixed` at both compile and run time, with the same semantics as
#'   [CppODE()];
#' * `forcings`, PCHIP-interpolated time-dependent inputs (forcings do
#'   not contribute to sensitivities);
#' * `rootfunc` for integration termination;
#' * `events`, both time- and root-triggered, with first-order
#'   saltation correction of sensitivities.
#'
#' Not available via the CVODE backend: second-order sensitivities, the
#' `"rb4"` and `"tsit5"` methods, and the `profile` flag.
#'
#' @inheritParams CppODE
#' @param method One of `"bdf"` (default) or `"adams"`.
#' @param stepTrace Logical. If `TRUE`, record per-step diagnostics (the
#'   trace `data.frame` is attached as `$trace` to the result of
#'   [solveODE()] and may be written to CSV via the `traceFile` argument
#'   of that function). Not supported together with `events` or
#'   `rootfunc`. Default `FALSE`.
#'
#' @return
#' The compiled model name (character string), with the same attribute
#' set as a model returned by [CppODE()], so it can be passed directly to
#' [solveODE()]. The attribute `backend` is `"cvode"`.
#'
#' @references
#' Hindmarsh, A. C., Brown, P. N., Grant, K. E., Lee, S. L., Serban, R.,
#' Shumaker, D. E., and Woodward, C. S. (2005). SUNDIALS: Suite of
#' Nonlinear and Differential/Algebraic Equation Solvers.
#' \emph{ACM Transactions on Mathematical Software} \strong{31}(3), 363-396.
#'
#' @seealso [CppODE()], [solveODE()].
#' @export
CVODE <- function(rhs, events = NULL, rootfunc = NULL, fixed = NULL, forcings = NULL,
                  compile = TRUE, modelname = NULL, outdir = tempdir(),
                  deriv = FALSE,
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
      "    Windows       : from any shell (PowerShell / cmd / Git Bash)\n",
      "                    call Rtools' pacman by full path -- substitute\n",
      "                    your installed version for <ver> (e.g. 44 or 45):\n",
      "                      C:/rtools<ver>/usr/bin/pacman.exe -Sy --noconfirm mingw-w64-ucrt-x86_64-sundials\n",
      "                    The .pc files land in C:/rtools<ver>/ucrt64/\n",
      "                    where the package's configure.win picks them up\n",
      "                    automatically on re-install.\n",
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
      "    macOS (brew)  : brew install suite-sparse\n",
      "    Windows       : from any shell (PowerShell / cmd / Git Bash)\n",
      "                    call Rtools' pacman by full path -- substitute\n",
      "                    your installed version for <ver> (e.g. 44 or 45):\n",
      "                      C:/rtools<ver>/usr/bin/pacman.exe -Sy --noconfirm mingw-w64-ucrt-x86_64-suitesparse\n",
      "                    then re-run R CMD INSTALL <path/to/CppODE>",
      call. = FALSE)
  }
  # Auto-selected sparse without KLU -> force dense.
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
  # CVODE always uses runtime-sized sensitivity slots (CVodeSensInit1 allocates
  # Ns_active vectors at solve time), so it's effectively heap AD from the
  # compile-time-width perspective.
  attr(modelname, "nStack")      <- Inf
  attr(modelname, "sparse")      <- use_sparse
  attr(modelname, "method")      <- method
  attr(modelname, "useNDF")      <- NA  # not meaningful for CVODE
  attr(modelname, "backend")     <- "cvode"

  # The sens dim defaults to model-parameter names (legacy / identity seeding
  # basis). solveODE() overrides this per call when sens1ini is supplied with
  # full Phi'(theta) shape (uses colnames(sens1ini) or theta1..M).
  attr(modelname, "dimNames") <- if (deriv) {
    list(time = "time", variable = variables, sens = sens_names)
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
