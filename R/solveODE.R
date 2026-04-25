#' Solver Interface for Ordinary Differential Equation Models
#'
#' @description
#' Numerically integrates a compiled ODE model created by [CppODE()] over a
#' specified time span.
#'
#' ## Sensitivity initial values, `fixed`, and reparametrization
#'
#' `sens1ini` and `sens2ini` describe how the *full* initial-condition and
#' parameter vector depends on the sensitivity-column basis -- i.e., they are
#' the Jacobian \eqn{\Phi'(\theta)} and Hessian \eqn{\Phi''(\theta)} of the
#' reparametrization \eqn{p = \Phi(\theta)}. Two shapes are accepted:
#'
#' - **Legacy (non-reparam) shape** `[n_states, n_active]`: assumes identity
#'   seeding on the parameter block. The active set equals the model's
#'   sensitivity names minus `fixed`. This is the shape emitted by
#'   `solveODE()` itself (`res$sens1[t, , ]`) and is directly usable as
#'   `sens1ini` for warm-starting subsequent solves.
#' - **Full shape** `[n_states + n_params, n_active]` (or, under explicit
#'   compile-time `ntheta`, `[n_states + n_params, n_theta]`): represents
#'   \eqn{\Phi'(\theta)} directly. State rows (first `n_states`) seed state
#'   ICs; parameter rows (next `n_params`) seed the dynamic parameters.
#'   Required when the model was compiled with `ntheta` (reparametrization
#'   mode); then `fixed` is not accepted and columns correspond to theta
#'   slots, not model parameters.
#'
#' Column names, when present, must match the relevant column basis
#' (`active_sens` legacy, or user-chosen theta names under reparam).
#'
#' @param model A compiled ODE model object returned by [CppODE()].
#' @param times Numeric vector of time points at which to return the solution.
#'   Must be non-empty and contain only finite values.
#' @param parms Named numeric vector of initial conditions and parameters.
#'   Names must include all of `c(attr(model, "variables"), attr(model, "parameters"))`.
#' @param sens1ini Optional numeric matrix of first-order sensitivity initial
#'   values, interpreted as the Jacobian \eqn{\Phi'(\theta)}. Accepts both the
#'   legacy shape `[n_states, n_active]` (auto-extended internally with an
#'   identity block on the parameter rows) and the full shape
#'   `[n_states + n_params, n_active]`. Under explicit `ntheta` (compile-time
#'   reparametrization) only the full shape `[n_states + n_params, ntheta]`
#'   is accepted and `sens1ini` is required. Row names, when present, must
#'   match the relevant row basis; column names must match the sensitivity
#'   column basis. Default `NULL` uses identity seeding (legacy mode only).
#' @param sens2ini Optional numeric array of second-order sensitivity initial
#'   values, interpreted as the Hessian tensor \eqn{\Phi''(\theta)}. Shapes
#'   analogous to `sens1ini`: `[n_states, n_active, n_active]` (legacy) or
#'   `[n_states + n_params, n_theta, n_theta]` (full). Only allowed when
#'   `attr(model, "deriv2") == TRUE`. Default `NULL` uses zero seeding
#'   (correct when \eqn{\Phi} is linear-affine).
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
#' @param maxprogress Maximum number of consecutive integration steps without
#'   time advance (i.e. consecutive rejected steps) before the solver aborts
#'   with `CV_CONV_FAILURE` (`return_code = -4`). Default `50`. Very stiff
#'   problems with sharp transients or discontinuities may legitimately reject
#'   several steps in a row when the controller first adapts, so setting this
#'   too low (e.g. `10`) can produce false positives; Boost.odeint's reference
#'   value is `500`. Lower this for "fail fast" behaviour in optimisation
#'   pipelines where a stuck solve should be rejected cheaply.
#' @param maxsteps Maximum total number of integration steps. Default `1e6`.
#' @param hini Initial step size; `0` (default) triggers automatic estimation.
#' @param roottol Tolerance for root-finding in root-triggered events. Default `1e-6`.
#' @param maxroot Maximum triggers per root event. Default `1`.
#' @param usePID Character string selecting the step-size control
#'   strategy for NDF/BDF. One of:
#'   \describe{
#'     \item{`"none"`}{(default) Classical CVODE-style I-controller
#'       with the BIAS2 safety factor. This is what `solveODE` did
#'       before the H211b experiments and is the recommended
#'       baseline for stiff problems on KLU-equipped Jacobians.}
#'     \item{`"intermediate"`}{Same I-controller and same order
#'       selection as `"none"`, plus a geometric (log-space) low-pass
#'       filter applied to the final step-size ratio:
#'       \deqn{\log \eta_{\mathrm{filt}} = (1-\alpha)\log \eta + \alpha \log \eta_{n-1}}
#'       with default \eqn{\alpha = 0.4}. The filter smooths the
#'       step-size sequence without interfering with order changes
#'       and resets its history on order changes, failures, and
#'       events.}
#'     \item{`"full"`}{Soederlind's H211b second-order digital filter
#'       replaces the I-controller at the current order. The
#'       order-decrease and order-increase candidates are lifted onto
#'       the same safety-only scale as the H211b proposal so that
#'       order selection is not biased against the filter path. Uses
#'       a default safety factor of 0.8.}
#'   }
#'   For Rosenbrock4 the value is silently forced to `"full"` because
#'   Rosenbrock4 always uses its built-in Gustafsson PI controller and
#'   has no toggle for it. Note that "PID" is a slight misnomer for
#'   all of these -- H211b is a second-order digital filter and the
#'   Gustafsson form is a PI controller, neither is a classical
#'   PID -- but the name follows common usage in the adaptive-stepping
#'   literature. References: Soederlind (2003) "Digital Filters in
#'   Adaptive Time-Stepping", ACM TOMS 29(1); Soederlind & Wang (2006)
#'   "Adaptive Time-Stepping and Computational Stability"; Gustafsson,
#'   Lundh & Soederlind (1988) "A PI stepsize control for the numerical
#'   solution of ordinary differential equations", BIT 28.
#' @param onFailure How to react when the solver returns `return_code != 0`
#'   (step limit hit, no progress, etc.). One of `"warn"` (default, historical
#'   behaviour: emit a warning and return partial results up to `t_reached`),
#'   `"stop"` (raise an error with the solver message; no partial results
#'   are returned), or `"silent"` (return the partial result without any
#'   signal). Use `"stop"` in optimisation pipelines where a silently
#'   truncated trajectory would be misinterpreted as a valid prediction.
#' @param traceFile Optional character. If the compiled model was built
#'   with `stepTrace = TRUE` and a non-empty path is supplied here, the
#'   per-step trace data.frame is also written to this CSV file in the
#'   current working directory.  The trace is additionally attached to
#'   the returned list as `$trace` regardless of whether a file path is
#'   provided.  Ignored for models compiled without trace support (the
#'   `$trace` element is `NULL` in that case).
#'
#' @return A named list with components `time`, `variable`, `diagnostics`,
#'   and, when `attr(model, "deriv") == TRUE`, `sens1`, and additionally
#'   `sens2` when `attr(model, "deriv2") == TRUE`. All output arrays are
#'   **time-first**: `variable` is `[n_t, n_x]`, `sens1` is `[n_t, n_x, n_s]`,
#'   and `sens2` is `[n_t, n_x, n_s, n_s]`. Dimension names of `sens1`/`sens2`
#'   reflect only the active (non-fixed) sensitivity parameters. The
#'   `diagnostics` element is a list with solver statistics (see
#'   [diagnostics()]). When the model was compiled with `stepTrace = TRUE`,
#'   an additional `$trace` data.frame with per-step diagnostics is attached.
#'
#' @seealso [CppODE()] for model specification and compilation.
#'
#' @export
solveODE <- function(model, times, parms,
                     sens1ini = NULL, sens2ini = NULL,
                     fixed = NULL, forcings = NULL,
                     abstol = 1e-6, reltol = 1e-6,
                     maxprogress = 50L, maxsteps = 1e6L,
                     hini = 0, roottol = 1e-6, maxroot = 1L,
                     usePID = c("none", "intermediate", "full"),
                     onFailure = c("warn", "stop", "silent"),
                     traceFile = NULL) {

  onFailure <- match.arg(onFailure)

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
  has_reparam   <- isTRUE(attr(model, "has_reparam"))
  dynamic_ad    <- isTRUE(attr(model, "dynamic_ad"))
  ntheta        <- attr(model, "ntheta")

  ## --- Runtime fixed: resolve early so sens1ini/sens2ini see the active set ---
  fixed_indices <- integer(0)
  if (!is.null(fixed)) {
    if (!deriv) { warning("'fixed' ignored when deriv = FALSE") }
    else if (has_reparam) {
      stop("'fixed' is not supported together with ntheta reparametrization; ",
           "express fixedness through zero rows in sens1ini instead")
    } else {
      if (!is.character(fixed)) stop("'fixed' must be a character vector")
      bad <- setdiff(fixed, all_sens)
      if (length(bad)) stop("Unknown 'fixed' names: ", paste(bad, collapse = ", "),
                            "\nValid: ", paste(all_sens, collapse = ", "))
      fixed_indices <- match(fixed, all_sens) - 1L  # 0-based for C++
    }
  }
  active_sens <- if (deriv && length(fixed_indices))
    all_sens[-( fixed_indices + 1L)] else all_sens

  ## --- Validate and coerce sens1ini / sens2ini ---
  ## Under the unified semantics sens1ini is Phi'(theta) with full shape
  ## [n_states + n_params, n_theta_active]. Legacy shape [n_states, n_active]
  ## is accepted in non-reparam mode and auto-extended with an identity block
  ## on the param rows (zeros for fixed params).
  if (!is.null(sens1ini) && !deriv)
    stop("'sens1ini' supplied but model has deriv = FALSE")
  if (!is.null(sens2ini) && !deriv2)
    stop("'sens2ini' supplied but model has deriv2 = FALSE")

  n_states  <- length(variables)
  n_params  <- length(parameters)
  n_phi_rows <- n_states + n_params
  n_active  <- length(active_sens)
  ## Under reparam, `ntheta` (model attr) is the compile-time upper bound on
  ## the AD width. `n_theta_active` is the per-call count read from
  ## ncol(sens1ini) and must satisfy 0 <= n_theta_active <= n_theta_max.
  n_theta_max <- if (dynamic_ad) .Machine$integer.max else
                 if (has_reparam) as.integer(ntheta) else n_active
  n_theta_active <- if (has_reparam) {
    if (is.null(sens1ini))
      stop("'sens1ini' is required when the model was compiled with an explicit ntheta")
    if (is.matrix(sens1ini) || (is.array(sens1ini) && length(dim(sens1ini)) == 2L)) {
      as.integer(ncol(sens1ini))
    } else {
      stop("'sens1ini' must be a matrix under reparametrization ",
           "(need a 'dim' attribute to infer the active theta count)")
    }
  } else n_active

  ## Identity-on-active-params padding for legacy shape under non-reparam.
  build_param_identity <- function(col_names) {
    pad <- matrix(0, nrow = n_params, ncol = length(col_names),
                  dimnames = list(parameters, col_names))
    for (j in seq_len(n_params)) {
      col_idx <- match(parameters[j], col_names)
      if (!is.na(col_idx)) pad[j, col_idx] <- 1.0
    }
    pad
  }

  ## Reorder a matrix's rows to `row_order` if dimnames present.
  reorder_rows <- function(x, row_order, arg) {
    if (is.null(rownames(x))) return(x)
    miss <- setdiff(row_order, rownames(x))
    if (length(miss))
      stop(sprintf("'%s' row names missing: %s", arg, paste(miss, collapse = ", ")))
    x[row_order, , drop = FALSE]
  }

  ## Full-shape coercion: returns flat double of length n_phi_rows * n_cols.
  coerce_full_2d <- function(x, n_cols, col_names, arg) {
    if (!is.numeric(x)) stop("'", arg, "' must be numeric")
    if (is.matrix(x) || (is.array(x) && length(dim(x)) == 2)) {
      if (nrow(x) == n_states && ncol(x) == n_cols && !has_reparam) {
        ## Legacy shape: reorder, then append identity block on param rows.
        if (!is.null(rownames(x))) {
          if (!setequal(rownames(x), variables))
            stop("'", arg, "' row names must match variables")
          x <- x[variables, , drop = FALSE]
        }
        if (!is.null(colnames(x))) {
          if (!setequal(colnames(x), col_names))
            stop("'", arg, "' column names must match active sens: ",
                 paste(col_names, collapse = ", "))
          x <- x[, col_names, drop = FALSE]
        }
        pad <- build_param_identity(col_names)
        full <- rbind(unname(x), unname(pad))
        return(as.double(full))
      }
      if (nrow(x) != n_phi_rows || ncol(x) != n_cols)
        stop(sprintf("'%s' must be [%d, %d] (Phi' full shape)%s",
                     arg, n_phi_rows, n_cols,
                     if (!has_reparam) sprintf(" or [%d, %d] (legacy)", n_states, n_cols) else ""))
      if (!is.null(rownames(x))) {
        expected_rows <- c(variables, parameters)
        if (!setequal(rownames(x), expected_rows))
          stop("'", arg, "' row names must be c(variables, parameters)")
        x <- x[expected_rows, , drop = FALSE]
      }
      if (!is.null(colnames(x))) {
        if (!setequal(colnames(x), col_names))
          stop("'", arg, "' column names must match: ", paste(col_names, collapse = ", "))
        x <- x[, col_names, drop = FALSE]
      }
      as.double(x)
    } else {
      if (length(x) == n_states * n_cols && !has_reparam) {
        xmat <- matrix(as.double(x), nrow = n_states, ncol = n_cols)
        pad  <- build_param_identity(col_names)
        return(as.double(rbind(xmat, unname(pad))))
      }
      if (length(x) != n_phi_rows * n_cols)
        stop(sprintf("'%s' must have length %d (n_phi_rows * n_cols)",
                     arg, n_phi_rows * n_cols))
      as.double(x)
    }
  }

  ## Column names: theta names under reparam come from user via colnames(sens1ini).
  ## If user didn't provide colnames, fall back to the first n_theta_active of
  ## the auto-generated names in attr(model, "dim_names")$sens (length n_theta_max).
  sens_col_names <- if (has_reparam) {
    if (!is.null(sens1ini) && !is.null(colnames(sens1ini))) colnames(sens1ini)
    else if (dynamic_ad) sprintf("theta%d", seq_len(n_theta_active))
    else all_sens[seq_len(n_theta_active)]
  } else {
    active_sens
  }
  if (has_reparam && length(sens_col_names) != n_theta_active)
    stop(sprintf("colnames(sens1ini) length (%d) does not match ncol(sens1ini) (%d)",
                 length(sens_col_names), n_theta_active))
  if (has_reparam && n_theta_active > n_theta_max)
    stop(sprintf("sens1ini column count (%d) exceeds model ntheta upper bound (%d)",
                 n_theta_active, n_theta_max))

  sens1ini <- if (!is.null(sens1ini)) {
    flat <- coerce_full_2d(sens1ini, n_theta_active, sens_col_names, "sens1ini")
    ## Preserve 2-D shape for C++ Rf_ncols() -- distinguishes [phi_rows, M] from a flat vector.
    dim(flat) <- c(n_phi_rows, n_theta_active)
    flat
  }

  if (!is.null(sens2ini)) {
    if (!is.numeric(sens2ini)) stop("'sens2ini' must be numeric")
    if (is.array(sens2ini) && length(dim(sens2ini)) == 3) {
      ## Accept legacy [n_states, n_active, n_active] in non-reparam mode
      ## by padding with zeros on the param block (correct when Phi'' = 0 for
      ## linear-affine Phi, which is the legacy identity case).
      if (all(dim(sens2ini) == c(n_states, n_theta_active, n_theta_active)) && !has_reparam) {
        dn <- dimnames(sens2ini)
        if (!is.null(dn[[1]])) sens2ini <- sens2ini[variables, , , drop = FALSE]
        if (!is.null(dn[[2]])) sens2ini <- sens2ini[, sens_col_names, , drop = FALSE]
        if (!is.null(dn[[3]])) sens2ini <- sens2ini[, , sens_col_names, drop = FALSE]
        pad_arr <- array(0, dim = c(n_params, n_theta_active, n_theta_active))
        full <- array(0, dim = c(n_phi_rows, n_theta_active, n_theta_active))
        full[seq_len(n_states), , ] <- sens2ini
        ## pad rows are zero by init
        sens2ini <- as.double(full)
      } else if (all(dim(sens2ini) == c(n_phi_rows, n_theta_active, n_theta_active))) {
        dn <- dimnames(sens2ini)
        if (!is.null(dn[[1]])) {
          expected_rows <- c(variables, parameters)
          if (!setequal(dn[[1]], expected_rows))
            stop("'sens2ini' dim 1 must be c(variables, parameters)")
          sens2ini <- sens2ini[expected_rows, , , drop = FALSE]
        }
        if (!is.null(dn[[2]])) {
          if (!setequal(dn[[2]], sens_col_names))
            stop("'sens2ini' dim 2 must match sens columns")
          sens2ini <- sens2ini[, sens_col_names, , drop = FALSE]
        }
        if (!is.null(dn[[3]])) {
          if (!setequal(dn[[3]], sens_col_names))
            stop("'sens2ini' dim 3 must match sens columns")
          sens2ini <- sens2ini[, , sens_col_names, drop = FALSE]
        }
        sens2ini <- as.double(sens2ini)
      } else {
        stop(sprintf("'sens2ini' must be [%d, %d, %d] (Phi'' full shape)",
                     n_phi_rows, n_theta_active, n_theta_active))
      }
    } else {
      len <- length(sens2ini)
      if (len == n_phi_rows * n_theta_active^2) {
        sens2ini <- as.double(sens2ini)
      } else if (len == n_states * n_theta_active^2 && !has_reparam) {
        pad <- array(0, dim = c(n_phi_rows, n_theta_active, n_theta_active))
        pad[seq_len(n_states), , ] <- array(as.double(sens2ini),
                                            dim = c(n_states, n_theta_active, n_theta_active))
        sens2ini <- as.double(pad)
      } else {
        stop(sprintf("'sens2ini' must have length %d", n_phi_rows * n_theta_active^2))
      }
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
  ## usePID: character with three modes (NDF/BDF only)
  ##   "none"         -- classical CVODE I-controller (default)
  ##   "intermediate" -- classical I-controller + log-space LP filter on m_eta
  ##   "full"         -- Soederlind H211b replaces the I-controller
  ## For Rosenbrock4 the value is forced to "full" because RB4 always
  ## uses its built-in Gustafsson PI controller.
  usePID <- match.arg(usePID)
  model_method <- attr(model, "method")
  if (!is.null(model_method) && model_method %in% c("rosenbrock4", "tsit5")) {
    usePID <- "full"
  }
  pid_mode_int <- match(usePID, c("none", "intermediate", "full")) - 1L
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
          forcing_times_list, forcing_values_list, as.integer(pid_mode_int)),
    error = function(e) stop("ODE solver error: ", e$message, call. = FALSE))

  ## --- Attach dimension names (time-first layout) ---
  out_sens <- if (has_reparam) sens_col_names else active_sens
  if (!is.null(result$variable))
    colnames(result$variable) <- variables
  if (!is.null(result$sens1))
    dimnames(result$sens1) <- list(time = NULL, variable = variables, sens = out_sens)
  if (!is.null(result$sens2))
    dimnames(result$sens2) <- list(time = NULL, variable = variables,
                                   sens1 = out_sens, sens2 = out_sens)

  ## --- Handle diagnostics ---
  diag <- result$diagnostics
  if (!is.null(diag)) {
    diag$method  <- attr(model, "method")
    diag$useNDF  <- attr(model, "useNDF")
    diag$backend <- attr(model, "backend")
    result$diagnostics <- diag
  }
  if (!is.null(diag) && diag$return_code != 0L) {
    msg <- paste0(
      "Solver did not complete: ", diag$message,
      "\n  Reached t = ", format(diag$t_reached, digits = 6),
      " (", length(result$time), " of ", length(times), " time points)."
    )
    switch(onFailure,
           stop   = stop(msg, call. = FALSE),
           warn   = warning(paste0(msg, "\n  Returning partial results."),
                            call. = FALSE, immediate. = TRUE),
           silent = invisible(NULL))
  }

  ## --- Handle step trace (element `$trace` is a named list of vectors) ---
  if (!is.null(result$trace) && length(result$trace$nst) > 0L) {
    result$trace <- as.data.frame(result$trace, stringsAsFactors = FALSE)
    if (!is.null(traceFile)) {
      stopifnot(is.character(traceFile), length(traceFile) == 1L, nzchar(traceFile))
      utils::write.csv(result$trace, traceFile, row.names = FALSE)
    }
  } else {
    result$trace <- NULL
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
  # Return codes follow the SUNDIALS CVODE flag scheme
  # (see inst/include/cppode/cppode_return_codes.hpp).
  rc_text <- switch(as.character(rc),
                    "0"   = "Integration was successful.",
                    "1"   = "Reached TSTOP (requested stop time).",
                    "2"   = "Root function returned a root.",
                    "-1"  = "Too much work: maximum number of integration steps exceeded (CV_TOO_MUCH_WORK).",
                    "-2"  = "Too much accuracy requested for the requested step (CV_TOO_MUCH_ACC).",
                    "-3"  = "Error test failures too numerous (CV_ERR_FAILURE).",
                    "-4"  = "Convergence test failures too numerous / no progress (CV_CONV_FAILURE).",
                    "-5"  = "Linear solver initialisation failed (CV_LINIT_FAIL).",
                    "-6"  = "Linear solver setup failed unrecoverably (CV_LSETUP_FAIL).",
                    "-7"  = "Linear solver solve failed unrecoverably (CV_LSOLVE_FAIL).",
                    "-8"  = "RHS function failed unrecoverably (CV_RHSFUNC_FAIL).",
                    "-9"  = "RHS function failed at the first call (CV_FIRST_RHSFUNC_ERR).",
                    "-10" = "RHS function had repeated recoverable errors (CV_REPTD_RHSFUNC_ERR).",
                    "-11" = "RHS function had a recoverable error that could not be handled (CV_UNREC_RHSFUNC_ERR).",
                    "-12" = "Root function failed unrecoverably (CV_RTFUNC_FAIL).",
                    "-13" = "Nonlinear solver initialisation failed (CV_NLS_INIT_FAIL).",
                    "-14" = "Nonlinear solver setup failed (CV_NLS_SETUP_FAIL).",
                    "-15" = "Inequality constraint check failed (CV_CONSTR_FAIL).",
                    "-16" = "Nonlinear solver failed (CV_NLS_FAIL).",
                    "-20" = "Memory allocation request failed (CV_MEM_FAIL).",
                    "-21" = "Integrator memory is NULL (CV_MEM_NULL).",
                    "-22" = "Illegal input provided (CV_ILL_INPUT).",
                    "-23" = "Integrator memory was not allocated (CV_NO_MALLOC).",
                    "-24" = "Bad k value in CVodeGetDky (CV_BAD_K).",
                    "-25" = "Bad t value in CVodeGetDky (CV_BAD_T).",
                    "-26" = "Bad dky argument in CVodeGetDky (CV_BAD_DKY).",
                    "-27" = "Output time too close to initial time (CV_TOO_CLOSE).",
                    "-99" = "Unrecognised error (CV_UNRECOGNIZED_ERR).",
                    paste0("Unknown return code: ", rc)
  )

  label <- if (identical(diag$backend, "cvode")) {
    paste0("CVODE: ", toupper(diag$method %||% "bdf"))
  } else {
    switch(diag$method %||% "bdf",
           bdf         = if (isFALSE(diag$useNDF)) "BDF" else "NDF",
           adams       = "ADAMS",
           msoda       = "MSODA",
           rosenbrock4 = "ROSENBROCK4",
           tsit5       = "TSIT5",
           toupper(diag$method))
  }
  label <- paste(label, "solver statistics")
  total <- 69L
  inner <- paste0("  ", label, "  ")
  dash_n <- max(total - nchar(inner), 2L)
  left  <- dash_n %/% 2L
  right <- dash_n - left
  cat(strrep("-", left), inner, strrep("-", right), "\n", sep = "")
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
