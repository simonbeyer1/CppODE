#' Run a Compiled ODE Model
#'
#' @description
#' Numerically integrates a compiled ODE model created by [CppODE()] (or
#' [CVODE()]) over a specified time span. Returns the state trajectory and,
#' when the model was compiled with sensitivities, first- and (optionally)
#' second-order parameter sensitivities.
#'
#' @details
#' ## Sensitivity with respect to a reparametrization
#'
#' By default, the compiled solver computes sensitivities of the state
#' trajectory \eqn{x(t)} with respect to the full input vector
#' \eqn{p = (p_{\text{init}}, p_{\text{dyn}}) \in \mathbb{R}^{n_x + n_p}},
#' i.e. all initial conditions stacked on all dynamic parameters. In
#' fitting and identifiability workflows the quantity of interest is
#' often instead the gradient with respect to a smaller set of free
#' variables \eqn{\theta \in \mathbb{R}^{M}}, where the model inputs are
#' obtained from a smooth reparametrization
#'
#' \deqn{p = \Phi(\theta) \in \mathbb{R}^{n_x + n_p}.}
#'
#' Typical examples are log-parametrization
#' (\eqn{p_i = \exp \theta_i}) to enforce positivity, parameters that
#' are shared across compartments or experimental conditions, and
#' components of \eqn{p} that are held constant and therefore drop out
#' of \eqn{\theta}.
#'
#' By the chain rule, the trajectory's first- and second-order
#' sensitivities with respect to \eqn{\theta} are
#'
#' \deqn{\frac{\partial x(t)}{\partial \theta} \;=\;
#'       \frac{\partial x(t)}{\partial p}\,\Phi'(\theta),}
#'
#' \deqn{\frac{\partial^{2} x(t)}{\partial \theta\,\partial \theta^{\top}}
#'       \;=\;
#'       \Phi'(\theta)^{\top}\,
#'       \frac{\partial^{2} x(t)}{\partial p\,\partial p^{\top}}\,
#'       \Phi'(\theta) \;+\;
#'       \frac{\partial x(t)}{\partial p}\,\Phi''(\theta),}
#'
#' where \eqn{\Phi'(\theta)} is an \eqn{(n_x + n_p) \times M} Jacobian
#' and \eqn{\Phi''(\theta)} is the corresponding
#' \eqn{(n_x + n_p) \times M \times M} Hessian tensor (contracted on its
#' first index in the formula above).
#'
#' Rather than first computing \eqn{\partial x(t)/\partial p} along all
#' \eqn{n_x + n_p} canonical directions and contracting with
#' \eqn{\Phi'(\theta)} afterwards, the solver applies the chain rule
#' *inside* the AD pass: each of the \eqn{M} forward-AD directions is
#' seeded with the corresponding column of \eqn{\Phi'(\theta)}, so the
#' integrator directly returns
#' \eqn{\partial x(t)/\partial \theta} (and analogously
#' \eqn{\partial^{2} x(t)/\partial \theta\,\partial \theta^{\top}} when
#' \eqn{\Phi''(\theta)} is supplied for the second-order seeds). The
#' parameter sweep then has cost \eqn{O(M)} rather than
#' \eqn{O(n_x + n_p)}, which is the main practical benefit when
#' \eqn{M \ll n_x + n_p}.
#'
#' The default identity reparametrization \eqn{\Phi(\theta) = \theta}
#' corresponds to \eqn{\Phi'(\theta) = I_{(n_x + n_p) \times (n_x + n_p)}}
#' and \eqn{\Phi''(\theta) = 0}; this is the legacy seeding emitted when
#' `sens1ini` / `sens2ini` are omitted, restricted to the active
#' (non-fixed) sensitivity columns.
#'
#' ## Sensitivity initial values
#'
#' `sens1ini` and `sens2ini` are exactly \eqn{\Phi'(\theta)} and
#' \eqn{\Phi''(\theta)} from the chain-rule derivation above, evaluated
#' at the current \eqn{\theta}. Three shapes are accepted, selected per
#' call from the row count and row names of the supplied matrix:
#'
#' - **Legacy shape** `[n_states, n_active]`: identity seeding on the
#'   parameter block is implied. The active set equals the model's
#'   sensitivity names minus `fixed`. Detected when `nrow == n_states`
#'   and row names are absent or are a permutation of `variables`. This
#'   is the shape emitted by `solveODE()` itself (`res$sens1[t, , ]`) and
#'   can therefore be used directly for warm-starting subsequent solves.
#' - **Full shape** `[n_states + n_params, M]`: \eqn{\Phi'(\theta)}
#'   directly. State rows seed state ICs; parameter rows seed the dynamic
#'   parameters. The column count `M` may vary across calls; under
#'   stack-mode AD it must satisfy \eqn{M \le \mathtt{nStack}}.
#' - **Partial shape** `[k, M]` with `k < n_states + n_params`: row
#'   names are required and must be a subset of
#'   `c(variables, parameters)`. The supplied rows are placed at the
#'   matching positions of \eqn{\Phi'(\theta)}; missing rows are zero-
#'   padded, i.e. those slots are treated as fixed. This is the
#'   row-name-driven equivalent of run-time `fixed`.
#'
#' Run-time `fixed` is incompatible with the full and partial shapes
#' (those already encode fixedness via row presence / row values).
#'
#' Column names, when present, must match the relevant column basis
#' (the active sensitivity names for the legacy shape, user-chosen theta
#' names for the full / partial shapes).
#'
#' @param model A compiled ODE model returned by [CppODE()] or [CVODE()].
#' @param times Numeric vector of time points at which to return the
#'   solution. Must be non-empty and contain only finite values.
#' @param parms Named numeric vector of initial conditions and parameters.
#'   Names must include all of
#'   `c(attr(model, "variables"), attr(model, "parameters"))`.
#' @param sens1ini Optional numeric matrix of first-order sensitivity
#'   initial values, interpreted as the Jacobian \eqn{\Phi'(\theta)}.
#'   Accepts three shapes (see Details): legacy `[n_states, n_active]`
#'   (auto-extended with identity on parameter rows), full
#'   `[n_states + n_params, M]`, or partial `[k, M]` with row names
#'   identifying a subset of `c(variables, parameters)` (missing rows
#'   are zero-padded, i.e. implicitly fixed). Column names label the
#'   resulting sensitivity output columns. Default `NULL` uses identity
#'   seeding on the active sensitivity basis.
#' @param sens2ini Optional numeric array of second-order sensitivity
#'   initial values, interpreted as the Hessian tensor
#'   \eqn{\Phi''(\theta)}. Shapes are analogous to those of `sens1ini`:
#'   `[n_states, n_active, n_active]` (legacy),
#'   `[n_states + n_params, M, M]` (full), or `[k, M, M]` with dim-1
#'   names identifying a subset of `c(variables, parameters)` (partial,
#'   zero-padded). Allowed only when `attr(model, "deriv2")` is `TRUE`.
#'   Default `NULL` uses zero seeding (correct when \eqn{\Phi} is
#'   linear-affine).
#' @param fixed Optional character vector of sensitivity-parameter names
#'   to treat as fixed at run time. The integrator then runs with a
#'   smaller AD state. Names must be a subset of
#'   `attr(model, "dimNames")$sens`. Unlike compile-time `fixed` in
#'   [CppODE()], the run-time `fixed` set can be changed between calls
#'   without recompilation. Incompatible with full / partial `sens1ini`
#'   (those encode fixedness through row values or row presence).
#'   Default `NULL` (all parameters active).
#' @param forcings Optional named list of forcing-function data. Each
#'   element must be a `data.frame` (or coercible object) with columns
#'   `time` and `value`, or a two-column matrix. Names must match
#'   `attr(model, "forcings")`. Default `NULL`.
#' @param abstol Absolute error tolerance. Default `1e-6`.
#' @param reltol Relative error tolerance. Default `1e-6`.
#' @param maxprogress Maximum number of consecutive integration steps
#'   without time advance (consecutive rejected steps) before the solver
#'   aborts with `CV_CONV_FAILURE` (`return_code = -4`). Default `50`.
#'   Lower values can be useful for fail-fast behaviour in optimisation
#'   pipelines; very stiff problems with sharp transients may legitimately
#'   reject several steps in a row when the controller first adapts.
#' @param maxsteps Maximum total number of integration steps. Default
#'   `1e6`.
#' @param hini Initial step size; `0` (default) triggers automatic
#'   estimation.
#' @param roottol Tolerance for root finding in root-triggered events.
#'   Default `1e-6`.
#' @param maxroot Maximum number of triggers per root event. Default `1`.
#' @param usePID Step-size control strategy for NDF/BDF. One of:
#'   \describe{
#'     \item{`"none"`}{(default) Classical CVODE-style I-controller with
#'       the BIAS2 safety factor. Recommended baseline for stiff problems.}
#'     \item{`"intermediate"`}{The same I-controller and order selection
#'       as `"none"`, plus a geometric (log-space) low-pass filter on the
#'       final step-size ratio. The filter resets on order changes,
#'       failures, and events.}
#'     \item{`"full"`}{Soederlind's H211b second-order digital filter
#'       replaces the I-controller at the current order. Uses a default
#'       safety factor of 0.8.}
#'   }
#'   For methods `"rb4"` and `"tsit5"` the value is silently set to
#'   `"full"` because those methods always use their built-in PI
#'   controller. The label "PID" follows common usage in the
#'   adaptive-stepping literature.
#' @param onFailure How to react when the solver returns a non-zero
#'   return code. One of `"warn"` (default; emit a warning and return
#'   partial results up to `t_reached`), `"stop"` (raise an error with
#'   the solver message and no partial results), or `"silent"` (return
#'   the partial result without any signal).
#' @param traceFile Optional character giving a CSV file path. If the
#'   model was compiled with `stepTrace = TRUE` and a non-empty path is
#'   supplied, the per-step trace `data.frame` is written to that path.
#'   The trace is also attached to the returned list as `$trace`. Ignored
#'   for models compiled without trace support (`$trace` is `NULL` in
#'   that case).
#'
#' @return
#' A named list with components `time`, `variable`, `diagnostics`, and,
#' when `attr(model, "deriv")` is `TRUE`, `sens1`, plus `sens2` when
#' `attr(model, "deriv2")` is `TRUE`. Output arrays are time-first:
#' `variable` is `[n_t, n_x]`, `sens1` is `[n_t, n_x, n_s]`, and
#' `sens2` is `[n_t, n_x, n_s, n_s]`. The dimension names of `sens1`
#' and `sens2` reflect the active (non-fixed) sensitivity parameters.
#' The `diagnostics` element is a list of solver statistics (see
#' [diagnostics()]). When the model was compiled with `stepTrace = TRUE`,
#' an additional `$trace` `data.frame` with per-step diagnostics is
#' attached.
#'
#' @seealso [CppODE()] and [CVODE()] for model compilation;
#'   [diagnostics()] for printing solver statistics.
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
  required_attrs <- c("variables", "parameters", "forcings", "deriv", "deriv2", "dimNames")
  missing_attrs <- setdiff(required_attrs, names(attributes(model)))
  if (length(missing_attrs))
    stop("'model' is missing attributes: ", paste(missing_attrs, collapse = ", "))

  variables     <- attr(model, "variables")
  parameters    <- attr(model, "parameters")
  forcing_names <- attr(model, "forcings")
  deriv         <- attr(model, "deriv")
  deriv2        <- attr(model, "deriv2")
  all_sens      <- if (deriv) attr(model, "dimNames")$sens else character(0)
  nStack_attr   <- attr(model, "nStack")
  backend       <- attr(model, "backend")  # "cvode" for CVODE, NULL/other for native
  is_cvode      <- identical(backend, "cvode")

  n_states  <- length(variables)
  n_params  <- length(parameters)
  n_phi_rows <- n_states + n_params

  ## --- Detect sens1ini shape per call ---
  ## Three accepted shapes:
  ## - Legacy [n_states, n_active]: implicit identity on parameter rows.
  ##   Detected by nrow == n_states and rownames absent or all in `variables`.
  ## - Full Phi' [n_phi_rows, M]: parameter rows supplied explicitly.
  ## - Partial Phi' [k, M], k < n_phi_rows: rownames identify which rows are
  ##   supplied; missing rows are padded with zeros (= implicit fixed).
  ## NULL means identity seeding on the whole non-fixed sens basis.
  ## Vector form is treated as legacy/full as today (M = n_active, no partial).
  if (!is.null(sens1ini) && !deriv)
    stop("'sens1ini' supplied but model has deriv = FALSE")
  if (!is.null(sens2ini) && !deriv2)
    stop("'sens2ini' supplied but model has deriv2 = FALSE")

  is_2d_sens1 <- !is.null(sens1ini) &&
    (is.matrix(sens1ini) ||
       (is.array(sens1ini) && length(dim(sens1ini)) == 2L))
  sens1ini_is_legacy <-
    is_2d_sens1 && n_phi_rows != n_states && nrow(sens1ini) == n_states && {
      rn <- rownames(sens1ini)
      is.null(rn) || all(rn %in% variables)
    }
  ## "is_full" here means "user supplied a Phi' (full or partial)" — anything
  ## that is not legacy. Used to gate runtime `fixed` and to drive
  ## n_theta_active / sens2ini coupling.
  sens1ini_is_full <- is_2d_sens1 && !sens1ini_is_legacy

  ## --- Runtime fixed: incompatible with full-shape sens1ini ---
  fixed_indices <- integer(0)
  if (!is.null(fixed)) {
    if (!deriv) { warning("'fixed' ignored when deriv = FALSE") }
    else if (sens1ini_is_full) {
      stop("'fixed' is not supported together with full-shape sens1ini; ",
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
  n_active  <- length(active_sens)

  ## --- Per-call active sens dimension and stack-width check ---
  ## sens1ini full shape: M = ncol(sens1ini) (theta count, may differ from n_active).
  ## otherwise: M = n_active (legacy / identity seeding produces the active basis).
  n_theta_active <- if (sens1ini_is_full) as.integer(ncol(sens1ini)) else n_active
  if (deriv) {
    nStack_max <- if (is.null(nStack_attr) || is.infinite(nStack_attr))
                    .Machine$integer.max
                  else as.integer(nStack_attr)
    if (n_theta_active > nStack_max)
      stop(sprintf("sens1ini column count (%d) exceeds the model's compile-time nStack (%d)",
                   n_theta_active, nStack_max))
  }

  ## --- Identity-on-active-params padding for legacy shape ---
  build_param_identity <- function(col_names) {
    pad <- matrix(0, nrow = n_params, ncol = length(col_names),
                  dimnames = list(parameters, col_names))
    for (j in seq_len(n_params)) {
      col_idx <- match(parameters[j], col_names)
      if (!is.na(col_idx)) pad[j, col_idx] <- 1.0
    }
    pad
  }

  ## --- Output sens column names (per call) ---
  ## - full + colnames present  -> user-provided theta names
  ## - full + no colnames       -> theta1..thetaM
  ## - legacy / NULL            -> active_sens (model-parameter basis)
  sens_col_names <- if (sens1ini_is_full) {
    cn <- colnames(sens1ini)
    if (!is.null(cn)) cn else sprintf("theta%d", seq_len(n_theta_active))
  } else {
    active_sens
  }

  ## --- Coerce sens1ini to flat [n_phi_rows, n_theta_active] ---
  ## Performance: skip column/row reorder when the order already matches
  ## (the common case in optimisation loops where the same shape is reused).
  ## Legacy padding uses one preallocated zero matrix + indexed assignment
  ## rather than rbind (which would allocate twice).
  coerce_sens1ini <- function(x, n_cols, col_names, arg) {
    if (!is.numeric(x)) stop("'", arg, "' must be numeric")

    is_2d <- is.matrix(x) || (is.array(x) && length(dim(x)) == 2L)
    if (!is_2d) {
      ## Vector form: legacy if length == n_states*n_cols, else full.
      ## (Partial-row form requires names and is therefore matrix-only.)
      if (length(x) == n_states * n_cols && n_phi_rows != n_states) {
        out <- matrix(0, n_phi_rows, n_cols)
        out[seq_len(n_states), ] <- x
        out[(n_states + 1L):n_phi_rows, ] <- build_param_identity(col_names)
        return(as.double(out))
      }
      if (length(x) != n_phi_rows * n_cols)
        stop(sprintf("'%s' must have length %d (n_phi_rows * n_cols) or %d (legacy)",
                     arg, n_phi_rows * n_cols, n_states * n_cols))
      return(as.double(x))
    }

    nr <- nrow(x); nc <- ncol(x)
    rn <- rownames(x); cn <- colnames(x)
    expected_rows <- c(variables, parameters)

    ## Column reorder/check (once, regardless of row interpretation).
    if (!is.null(cn)) {
      if (!setequal(cn, col_names))
        stop("'", arg, "' column names must match: ",
             paste(col_names, collapse = ", "))
      if (!identical(cn, col_names))
        x <- x[, col_names, drop = FALSE]
    } else if (nc != n_cols) {
      stop(sprintf("'%s' must have %d columns", arg, n_cols))
    }

    ## Full Phi' shape: [n_phi_rows, n_cols].
    if (nr == n_phi_rows) {
      if (!is.null(rn)) {
        if (!setequal(rn, expected_rows))
          stop("'", arg, "' row names must be c(variables, parameters)")
        if (!identical(rn, expected_rows))
          x <- x[expected_rows, , drop = FALSE]
      }
      return(as.double(x))
    }

    ## Legacy shape: [n_states, n_cols], no rownames or all-variable rownames.
    if (nr == n_states && n_phi_rows != n_states &&
        (is.null(rn) || all(rn %in% variables))) {
      if (!is.null(rn)) {
        if (!setequal(rn, variables))
          stop("'", arg, "' row names must match variables (legacy shape)")
        if (!identical(rn, variables))
          x <- x[variables, , drop = FALSE]
      }
      out <- matrix(0, n_phi_rows, n_cols)
      out[seq_len(n_states), ] <- x
      out[(n_states + 1L):n_phi_rows, ] <- build_param_identity(col_names)
      return(as.double(out))
    }

    ## Partial-row shape: rownames required, subset of c(variables, parameters).
    ## Missing rows are padded with zeros (= implicit fixed for those slots).
    if (is.null(rn))
      stop(sprintf(
        "'%s' has shape [%d, %d]; expected [%d, %d] (full Phi'), [%d, %d] (legacy), or a partial-row matrix with rownames identifying a subset of c(variables, parameters)",
        arg, nr, nc, n_phi_rows, n_cols, n_states, n_cols))
    bad <- setdiff(rn, expected_rows)
    if (length(bad))
      stop("'", arg, "' has unknown row names: ", paste(bad, collapse = ", "))
    if (anyDuplicated(rn))
      stop("'", arg, "' has duplicate row names")
    out <- matrix(0, n_phi_rows, n_cols)
    out[match(rn, expected_rows), ] <- x
    as.double(out)
  }

  ## --- Build sens1ini for the C++ side ---
  ## CVODE always needs a full Phi' (codegen has no identity-fallback). Native
  ## CppODE accepts NULL: the generated C++ then identity-seeds via diff(ai).
  ## Runtime `fixed` becomes zero rows when we synthesize the default Phi'.
  if (deriv && is.null(sens1ini) && is_cvode) {
    default_pp <- matrix(0, nrow = n_phi_rows, ncol = n_active,
                         dimnames = list(c(variables, parameters), active_sens))
    for (j in seq_along(active_sens)) {
      r <- match(active_sens[j], c(variables, parameters))
      if (!is.na(r)) default_pp[r, j] <- 1.0
    }
    sens1ini <- as.double(default_pp)
    dim(sens1ini) <- c(n_phi_rows, n_active)
  } else if (!is.null(sens1ini)) {
    flat <- coerce_sens1ini(sens1ini, n_theta_active, sens_col_names, "sens1ini")
    ## Preserve 2-D shape for C++ Rf_ncols() -- distinguishes [phi_rows, M] from a flat vector.
    dim(flat) <- c(n_phi_rows, n_theta_active)
    sens1ini <- flat
  }

  if (!is.null(sens2ini)) {
    if (!is.numeric(sens2ini)) stop("'sens2ini' must be numeric")
    ## Legacy [n_states, M, M] is accepted only when sens1ini is legacy / NULL
    ## (then Phi'' = 0 on the param block, correct for affine Phi). Full
    ## [n_phi_rows, M, M] is the unified shape; partial [k, M, M] is allowed
    ## with dim-1 names identifying a subset of c(variables, parameters)
    ## (missing rows = zero, i.e. implicit fixed). Partial coexists with
    ## partial sens1ini and with full sens1ini -- the row interpretation is
    ## independent because the C++ side receives the padded full tensor.
    legacy_d2_ok <- !sens1ini_is_full && n_phi_rows != n_states
    expected_rows <- c(variables, parameters)

    if (is.array(sens2ini) && length(dim(sens2ini)) == 3) {
      d <- dim(sens2ini); nr <- d[1L]; nc1 <- d[2L]; nc2 <- d[3L]
      dn <- dimnames(sens2ini)
      rn  <- if (length(dn) >= 1L) dn[[1L]] else NULL
      cn1 <- if (length(dn) >= 2L) dn[[2L]] else NULL
      cn2 <- if (length(dn) >= 3L) dn[[3L]] else NULL

      if (nc1 != n_theta_active || nc2 != n_theta_active)
        stop(sprintf("'sens2ini' must have dim 2 and 3 equal to %d", n_theta_active))

      ## Column reorder/check on dims 2 and 3 (once each).
      if (!is.null(cn1)) {
        if (!setequal(cn1, sens_col_names))
          stop("'sens2ini' dim 2 must match sens columns")
        if (!identical(cn1, sens_col_names))
          sens2ini <- sens2ini[, sens_col_names, , drop = FALSE]
      }
      if (!is.null(cn2)) {
        if (!setequal(cn2, sens_col_names))
          stop("'sens2ini' dim 3 must match sens columns")
        if (!identical(cn2, sens_col_names))
          sens2ini <- sens2ini[, , sens_col_names, drop = FALSE]
      }

      ## Full shape: [n_phi_rows, M, M].
      if (nr == n_phi_rows) {
        if (!is.null(rn)) {
          if (!setequal(rn, expected_rows))
            stop("'sens2ini' dim 1 must be c(variables, parameters)")
          if (!identical(rn, expected_rows))
            sens2ini <- sens2ini[expected_rows, , , drop = FALSE]
        }
        sens2ini <- as.double(sens2ini)

      ## Legacy shape: [n_states, M, M], state-only or no rownames.
      } else if (nr == n_states && legacy_d2_ok &&
                 (is.null(rn) || all(rn %in% variables))) {
        if (!is.null(rn)) {
          if (!setequal(rn, variables))
            stop("'sens2ini' dim 1 must match variables (legacy shape)")
          if (!identical(rn, variables))
            sens2ini <- sens2ini[variables, , , drop = FALSE]
        }
        full <- array(0, dim = c(n_phi_rows, n_theta_active, n_theta_active))
        full[seq_len(n_states), , ] <- sens2ini
        sens2ini <- as.double(full)

      ## Partial-row shape: dim-1 names required, subset of expected_rows.
      } else {
        if (is.null(rn))
          stop(sprintf(
            "'sens2ini' has shape [%d, %d, %d]; expected [%d, %d, %d] (full Phi''), [%d, %d, %d] (legacy), or a partial-row array with dim-1 names identifying a subset of c(variables, parameters)",
            nr, nc1, nc2, n_phi_rows, n_theta_active, n_theta_active,
            n_states, n_theta_active, n_theta_active))
        bad <- setdiff(rn, expected_rows)
        if (length(bad))
          stop("'sens2ini' has unknown dim-1 names: ", paste(bad, collapse = ", "))
        if (anyDuplicated(rn))
          stop("'sens2ini' has duplicate dim-1 names")
        full <- array(0, dim = c(n_phi_rows, n_theta_active, n_theta_active))
        full[match(rn, expected_rows), , ] <- sens2ini
        sens2ini <- as.double(full)
      }

    } else {
      ## Vector form (no partial — needs names). Same as before.
      len <- length(sens2ini)
      if (len == n_phi_rows * n_theta_active^2) {
        sens2ini <- as.double(sens2ini)
      } else if (len == n_states * n_theta_active^2 && legacy_d2_ok) {
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
  out_sens <- sens_col_names
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
#' Prints a summary of the solver diagnostics returned by [solveODE()].
#'
#' @param result A list returned by [solveODE()], containing a
#'   `diagnostics` element.
#'
#' @return Invisibly returns the `diagnostics` list.
#'
#' @examples
#' \dontrun{
#' res <- solveODE(model, times, parms)
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
