#' Compile Algebraic Functions with Optional Derivatives
#'
#' Generates and compiles C++ code that evaluates a system of algebraic
#' expressions \eqn{y = g(x, p)} on one or more rows of input, with
#' optional first- and second-order derivatives. There is no time
#' integration; the principal use cases are observation maps for
#' likelihood-based inference and reparametrisation Jacobians for
#' [solveODE()]. Both `derivMode = "dual"` and `derivMode = "symbolic"`
#' support `deriv` and `deriv2`, and both expose the same `func` / `jac`
#' / `hess` / `evaluate` API. The chain rule is available through the
#' optional seed arguments `dX`, `dP`, `dX2`, and `dP2`. See
#' `vignette("Methods", package = "CppODE")` for the two computational
#' paths and the pass-through convention for unmodelled inputs.
#'
#' @param eqns Named character vector or list of algebraic expressions.
#'   Names define the output variables; defaults to `f1`, `f2`, ... when
#'   unnamed.
#' @param variables Character vector of variable names supplied per
#'   observation. Defaults to all symbols in `eqns` not in `parameters`.
#' @param parameters Character vector of parameter names (constant
#'   across observations).
#' @param fixed Optional character vector of symbols excluded from
#'   derivative computation.
#' @param modelname Optional base name for generated C++ symbols and
#'   files.
#' @param outdir Directory for generated C++ source files. Default
#'   `tempdir()`.
#' @param compile Logical. Compile and load the generated C++ code.
#'   Default `FALSE`.
#' @param verbose Logical. Print progress messages.
#' @param convenient Logical. Return wrappers that accept named
#'   arguments rather than the low-level `(vars, params)` signature.
#' @param deriv Logical. Generate first-order derivative entry points.
#' @param deriv2 Logical. Generate Hessian entry points; implies
#'   `deriv = TRUE`.
#' @param derivMode One of `"dual"` (default) or `"symbolic"`. Selects
#'   the computational path: forward-mode AD on `cppode::dual` (single-
#'   or nested-dual) versus analytic SymPy-derived Jacobian/Hessian
#'   contracted via BLAS. Both modes deliver `jac`, `hess`, and
#'   `evaluate` with the same signatures.
#'
#' @return A list with components `func`, `jac`, `hess`, and
#'   `evaluate` (`NULL` when not generated). Carries attributes
#'   `equations`, `variables`, `parameters`, `fixed`, `modelname`,
#'   `srcfile`, `derivMode`, and (for `derivMode = "symbolic"`)
#'   `jacobian.symb`, `hessian.symb`.
#'
#' @seealso [compile()] for compilation; [derivSymb()] for symbolic
#'   differentiation; [CppODE()] and [CVODE()] for ODE integration;
#'   `vignette("Methods", package = "CppODE")`.
#' @export
funCpp <- function(eqns, variables = getSymbols(eqns, omit = parameters), parameters = NULL,
                   fixed = NULL, modelname = NULL, outdir = tempdir(), compile = FALSE,
                   verbose = FALSE, convenient = TRUE, deriv = TRUE, deriv2 = FALSE,
                   derivMode = c("dual", "symbolic")) {

  derivMode <- match.arg(derivMode)
  if (deriv2 && !deriv) { warning("deriv2 requires deriv. Setting deriv = TRUE."); deriv <- TRUE }
  emit_deriv <- deriv || deriv2
  use_ad     <- emit_deriv && derivMode == "dual"

  outnames <- names(eqns) %||% paste0("f", seq_along(eqns))
  if (!is.null(fixed)) { variables <- setdiff(variables, fixed); parameters <- union(parameters, fixed) }
  innames <- variables; diff_params <- setdiff(parameters, fixed); diff_syms <- c(variables, diff_params)
  if (!dir.exists(outdir)) stop("outdir does not exist: ", outdir)
  modelname <- modelname %||% paste0("f", paste(sample(c(letters, 0:9), 8, TRUE), collapse = ""))
  modelname <- unique_modelname(modelname)

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

  # --- Symbolic derivatives (only in symbolic mode) ---
  sym_jac <- sym_hess <- NULL
  if (emit_deriv && derivMode == "symbolic") {
    ds <- derivSymb(eqns, deriv2 = deriv2, real = TRUE, fixed = fixed, verbose = verbose)
    sym_jac <- ds$jacobian; sym_hess <- ds$hessian
  }
  if (!is.null(sym_jac)) { rownames(sym_jac) <- rownames(sym_jac) %||% outnames; sym_jac <- sym_jac[, intersect(diff_syms, colnames(sym_jac)), drop = FALSE] }
  if (!is.null(sym_hess)) for (nm in names(sym_hess)) { av <- intersect(diff_syms, rownames(sym_hess[[nm]])); sym_hess[[nm]] <- sym_hess[[nm]][av, av, drop = FALSE] }

  # --- Expression parsing (R fallback for symbolic mode without compile) ---
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
                           ad = use_ad, deriv2 = deriv2,
                           modelname = modelname, outdir = normalizePath(outdir, "/", FALSE), version = as.character(utils::packageVersion("CppODE")))

  # --- Attach helpers (pass-through of unmodelled inputs) ---
  abind3 <- function(a, b) { d <- dim(a); db <- dim(b); r <- array(0, c(d[1], d[2]+db[2], d[3]), list(dimnames(a)[[1]], c(dimnames(a)[[2]], dimnames(b)[[2]]), dimnames(a)[[3]])); r[,1:d[2],] <- a; r[,d[2]+1:db[2],] <- b; r }
  abind4 <- function(a, b) { d <- dim(a); db <- dim(b); r <- array(0, c(d[1], d[2]+db[2], d[3], d[4]), list(dimnames(a)[[1]], c(dimnames(a)[[2]], dimnames(b)[[2]]), dimnames(a)[[3]], dimnames(a)[[4]])); r[,1:d[2],,] <- a; r[,d[2]+1:db[2],,] <- b; r }

  attachExtras <- function(res, n_obs, ev, ep, type) {
    if (is.null(ev) && is.null(ep)) return(res)
    if (type == "fun") {
      if (!is.null(ev)) res <- cbind(res, ev)
      if (!is.null(ep)) res <- cbind(res, matrix(rep(ep, each = n_obs), n_obs, length(ep), dimnames = list(NULL, names(ep))))
    } else if (type == "jac") {
      cs <- dimnames(res)[[3]]; ncs <- length(cs)
      if (!is.null(ev)) res <- abind3(res, array(0, c(n_obs, ncol(ev), ncs), list(NULL, colnames(ev), cs)))
      if (!is.null(ep)) { np <- length(ep); pn <- names(ep); d <- dim(res); new <- array(0, c(d[1], d[2]+np, d[3]+np), list(dimnames(res)[[1]], c(dimnames(res)[[2]], pn), c(dimnames(res)[[3]], pn))); new[,1:d[2],1:d[3]] <- res; for (k in seq_len(np)) new[,d[2]+k,d[3]+k] <- 1; res <- new }
    } else {
      cs <- dimnames(res)[[3]]; ncs <- length(cs)
      if (!is.null(ev)) res <- abind4(res, array(0, c(n_obs, ncol(ev), ncs, ncs), list(NULL, colnames(ev), cs, cs)))
      if (!is.null(ep)) { np <- length(ep); pn <- names(ep); d <- dim(res); new <- array(0, c(d[1], d[2]+np, d[3]+np, d[4]+np), list(dimnames(res)[[1]], c(dimnames(res)[[2]], pn), c(dimnames(res)[[3]], pn), c(dimnames(res)[[4]], pn))); new[,1:d[2],1:d[3],1:d[4]] <- res; res <- new }
    }
    res
  }

  # --- Chain-rule helpers ---

  # Pull theta names from any seed; raise if seeds disagree.
  resolveTheta <- function(dX, dP, dX2 = NULL, dP2 = NULL) {
    cands <- list()
    if (!is.null(dX))  cands[[length(cands) + 1L]] <- dimnames(dX)[[3]]
    if (!is.null(dP))  cands[[length(cands) + 1L]] <- colnames(dP)
    if (!is.null(dX2)) cands[[length(cands) + 1L]] <- dimnames(dX2)[[3]]
    if (!is.null(dP2)) cands[[length(cands) + 1L]] <- dimnames(dP2)[[2]]
    cands <- cands[lengths(cands) > 0]
    if (!length(cands)) return(character(0))
    theta <- cands[[1]]
    for (k in seq_along(cands)[-1])
      if (!setequal(theta, cands[[k]])) stop("Seed theta names disagree across dX/dP/dX2/dP2")
    theta
  }

  # Align seeds onto the function's internal (vars, params) order, in flat
  # arrays ready for the dual-mode .C() entries. dX2/dP2 are returned as
  # length-zero numeric vectors when not provided; the C side guards them
  # via the `has_dX2` / `has_dP2` flags.
  alignSeedsDual <- function(dX, dP, dX2, dP2, n_obs, theta, fixed_rt) {
    n_vars <- length(innames); n_params <- length(parameters); n_theta <- length(theta)
    dX_arr <- array(0, c(n_obs, n_vars, n_theta))
    dP_mat <- matrix(0, n_params, n_theta)
    if (n_theta > 0) {
      if (!is.null(dX) && n_vars > 0) {
        idx <- match(innames, dimnames(dX)[[2]]); pres <- !is.na(idx)
        if (any(pres)) dX_arr[, pres, ] <- dX[, idx[pres], theta, drop = FALSE]
      }
      if (!is.null(dP) && n_params > 0) {
        idx <- match(parameters, rownames(dP)); pres <- !is.na(idx)
        if (any(pres)) dP_mat[pres, ] <- dP[idx[pres], theta, drop = FALSE]
      }
      if (length(fixed_rt) && n_params > 0) {
        fp <- match(fixed_rt, parameters); fp <- fp[!is.na(fp)]
        if (length(fp)) dP_mat[fp, ] <- 0
      }
    }
    has_dX2 <- !is.null(dX2) && n_theta > 0 && n_vars > 0
    has_dP2 <- !is.null(dP2) && n_theta > 0 && n_params > 0
    dX2_arr <- if (has_dX2) {
      r <- array(0, c(n_obs, n_vars, n_theta, n_theta))
      idx <- match(innames, dimnames(dX2)[[2]]); pres <- !is.na(idx)
      if (any(pres)) r[, pres, , ] <- dX2[, idx[pres], theta, theta, drop = FALSE]
      as.double(r)
    } else double(0)
    dP2_arr <- if (has_dP2) {
      r <- array(0, c(n_params, n_theta, n_theta))
      idx <- match(parameters, dimnames(dP2)[[1]]); pres <- !is.na(idx)
      if (any(pres)) r[pres, , ] <- dP2[idx[pres], theta, theta, drop = FALSE]
      if (length(fixed_rt)) {
        fp <- match(fixed_rt, parameters); fp <- fp[!is.na(fp)]
        if (length(fp)) r[fp, , ] <- 0
      }
      as.double(r)
    } else double(0)
    list(dX = as.double(dX_arr), dP = as.double(dP_mat),
         dX2 = dX2_arr, dP2 = dP2_arr,
         has_dX2 = as.integer(has_dX2), has_dP2 = as.integer(has_dP2))
  }

  # Bundle dX/dP into S [n_obs, n_diff, n_theta] for the symbolic chain rule.
  buildSeedMatrix <- function(dX, dP, n_obs, theta, fixed_rt) {
    n_diff <- length(diff_syms); n_theta <- length(theta)
    S <- array(0, c(n_obs, n_diff, n_theta), dimnames = list(NULL, diff_syms, theta))
    if (n_theta > 0 && !is.null(dX)) {
      var_part <- intersect(diff_syms, dimnames(dX)[[2]])
      if (length(var_part)) S[, var_part, ] <- dX[, var_part, theta, drop = FALSE]
    }
    if (n_theta > 0 && !is.null(dP)) {
      par_part <- intersect(diff_syms, rownames(dP))
      if (length(par_part)) {
        seed_p <- dP[par_part, theta, drop = FALSE]
        for (k in seq_len(n_theta))
          S[, par_part, k] <- rep(seed_p[, k], each = n_obs)
      }
    }
    if (length(fixed_rt)) {
      pf <- intersect(fixed_rt, diff_syms)
      if (length(pf)) S[, pf, , drop = FALSE] <- 0
    }
    S
  }

  # Bundle dX2/dP2 into S2 [n_obs, n_diff, n_theta, n_theta]; NULL if both seeds absent.
  buildSeedTensor2 <- function(dX2, dP2, n_obs, theta, fixed_rt) {
    if (is.null(dX2) && is.null(dP2)) return(NULL)
    n_diff <- length(diff_syms); n_theta <- length(theta)
    S2 <- array(0, c(n_obs, n_diff, n_theta, n_theta),
                dimnames = list(NULL, diff_syms, theta, theta))
    if (n_theta > 0 && !is.null(dX2)) {
      var_part <- intersect(diff_syms, dimnames(dX2)[[2]])
      if (length(var_part)) S2[, var_part, , ] <- dX2[, var_part, theta, theta, drop = FALSE]
    }
    if (n_theta > 0 && !is.null(dP2)) {
      par_part <- intersect(diff_syms, dimnames(dP2)[[1]])
      if (length(par_part)) {
        sp <- dP2[par_part, theta, theta, drop = FALSE]
        for (k1 in seq_len(n_theta)) for (k2 in seq_len(n_theta))
          S2[, par_part, k1, k2] <- rep(sp[, k1, k2], each = n_obs)
      }
    }
    if (length(fixed_rt)) {
      pf <- intersect(fixed_rt, diff_syms)
      if (length(pf)) S2[, pf, , ] <- 0
    }
    S2
  }

  # Identity seeds for raw dual-mode J/H: dX = I on vars, dP = I on params.
  # The combined θ-basis is c(innames, parameters) so the dy/d2y output
  # carries the canonical-symbol axes that the symbolic mode also produces.
  # (Values for parameters listed in `fixed` carry zero seed at AD time but
  # their column is kept; the caller drops it.)
  identitySeedsRaw <- function(n_obs, fixed_rt) {
    n_vars <- length(innames); n_params <- length(parameters)
    theta_full <- c(innames, parameters)
    n_theta <- length(theta_full)
    dX <- array(0, c(n_obs, n_vars, n_theta), dimnames = list(NULL, innames, theta_full))
    if (n_vars > 0) for (i in seq_along(innames)) dX[, innames[i], innames[i]] <- 1
    dP <- matrix(0, n_params, n_theta, dimnames = list(parameters, theta_full))
    for (i in seq_along(parameters))
      if (!(parameters[i] %in% fixed_rt))
        dP[parameters[i], parameters[i]] <- 1
    list(dX = dX, dP = dP, theta = theta_full)
  }

  # --- Core implementations (outputs time-first) ---

  fun_impl <- function(vars, params = numeric(0), attach.input = FALSE, fixed = NULL) {
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    funsym <- paste0(modelname, "_eval")
    if (is.loaded(funsym)) {
      out <- .C(funsym, x = as.double(M), y = double(length(outnames) * n_obs), p = as.double(p), n = as.integer(n_obs), k = as.integer(length(innames)), l = as.integer(length(outnames)))
      res <- matrix(out$y, n_obs, length(outnames), dimnames = list(NULL, outnames))
    } else {
      res <- matrix(NA_real_, n_obs, length(outnames), dimnames = list(NULL, outnames))
      for (i in seq_len(n_obs)) { env <- setNames(as.list(c(M[,i], p)), c(innames, parameters)); res[i,] <- vapply(parsed_exprs, function(e) eval(e, env), numeric(1)) }
    }
    attachExtras(res, n_obs, chk$extra_vars, chk$extra_params, "fun")
  }

  # --- Dual-path .C() helpers ---

  # Single-dual AD pass; returns list(y, dy) with dy of shape [n_obs, n_out, n_theta]
  # (or [..., 0] if n_theta == 0). Used by the dual path for jac and for
  # evaluate() at deriv2 = FALSE.
  call_eval_ad <- function(M, p, dX_seed, dP_seed, n_obs, theta) {
    funsym <- paste0(modelname, "_eval_ad"); n_out <- length(outnames); n_theta <- length(theta)
    if (!is.loaded(funsym)) stop("AD entry '", funsym, "' is not loaded.")
    out <- .C(funsym,
              x        = as.double(M),
              p        = as.double(p),
              dX       = dX_seed,
              dP       = dP_seed,
              y        = double(n_out * n_obs),
              dy       = double(n_out * max(n_theta, 1L) * n_obs),
              n_obs    = as.integer(n_obs),
              n_vars   = as.integer(length(innames)),
              n_params = as.integer(length(parameters)),
              n_out    = as.integer(n_out),
              n_theta  = as.integer(n_theta))
    y <- matrix(out$y, n_obs, n_out, dimnames = list(NULL, outnames))
    dy <- if (n_theta > 0)
      array(out$dy[seq_len(n_out * n_theta * n_obs)], c(n_obs, n_out, n_theta),
            list(NULL, outnames, theta))
    else array(0, c(n_obs, n_out, 0L), list(NULL, outnames, NULL))
    list(y = y, dy = dy)
  }

  # Nested-dual AD pass; returns list(y, dy, d2y).
  call_eval_ad2 <- function(M, p, aligned, n_obs, theta) {
    funsym <- paste0(modelname, "_eval_ad2"); n_out <- length(outnames); n_theta <- length(theta)
    if (!is.loaded(funsym)) stop("AD entry '", funsym, "' is not loaded.")
    out <- .C(funsym,
              x        = as.double(M),
              p        = as.double(p),
              dX       = aligned$dX,
              dP       = aligned$dP,
              dX2_in   = aligned$dX2,
              dP2_in   = aligned$dP2,
              has_dX2  = aligned$has_dX2,
              has_dP2  = aligned$has_dP2,
              y        = double(n_out * n_obs),
              dy       = double(n_out * max(n_theta, 1L) * n_obs),
              d2y      = double(n_out * max(n_theta, 1L)^2 * n_obs),
              n_obs    = as.integer(n_obs),
              n_vars   = as.integer(length(innames)),
              n_params = as.integer(length(parameters)),
              n_out    = as.integer(n_out),
              n_theta  = as.integer(n_theta))
    y <- matrix(out$y, n_obs, n_out, dimnames = list(NULL, outnames))
    if (n_theta > 0) {
      dy  <- array(out$dy[seq_len(n_out * n_theta * n_obs)], c(n_obs, n_out, n_theta),
                   list(NULL, outnames, theta))
      d2y <- array(out$d2y[seq_len(n_out * n_theta^2 * n_obs)], c(n_obs, n_out, n_theta, n_theta),
                   list(NULL, outnames, theta, theta))
    } else {
      dy  <- array(0, c(n_obs, n_out, 0L), list(NULL, outnames, NULL))
      d2y <- array(0, c(n_obs, n_out, 0L, 0L), list(NULL, outnames, NULL, NULL))
    }
    list(y = y, dy = dy, d2y = d2y)
  }

  # --- Symbolic-path raw evaluators ---

  raw_jac_sym <- function(M, p, n_obs, fixed_rt) {
    funsym <- paste0(modelname, "_jacobian"); n_out <- length(outnames); n_diff <- length(diff_syms)
    if (is.loaded(funsym)) {
      out <- .C(funsym, x = as.double(M), jac = double(n_obs * n_out * n_diff),
                p = as.double(p), n = as.integer(n_obs),
                k = as.integer(length(innames)), l = as.integer(n_out))
      arr <- array(out$jac, c(n_obs, n_out, n_diff), list(NULL, outnames, diff_syms))
    } else {
      arr <- array(0, c(n_obs, n_out, n_diff), list(NULL, outnames, diff_syms))
      for (i in seq_len(n_obs)) {
        env <- setNames(as.list(c(M[,i], p)), c(innames, parameters))
        for (o in seq_len(n_out)) for (s in seq_len(n_diff))
          if (!(diff_syms[s] %in% fixed_rt)) {
            e <- parsed_jac[[outnames[o], diff_syms[s]]]
            if (!is.null(e)) arr[i, o, s] <- eval(e, env)
          }
      }
    }
    arr
  }

  raw_hess_sym <- function(M, p, n_obs, fixed_rt) {
    funsym <- paste0(modelname, "_hessian"); n_out <- length(outnames); n_diff <- length(diff_syms)
    if (is.loaded(funsym)) {
      out <- .C(funsym, x = as.double(M), hess = double(n_obs * n_out * n_diff^2),
                p = as.double(p), n = as.integer(n_obs),
                k = as.integer(length(innames)), l = as.integer(n_out))
      arr <- array(out$hess, c(n_obs, n_out, n_diff, n_diff), list(NULL, outnames, diff_syms, diff_syms))
    } else {
      arr <- array(0, c(n_obs, n_out, n_diff, n_diff), list(NULL, outnames, diff_syms, diff_syms))
      for (i in seq_len(n_obs)) {
        env <- setNames(as.list(c(M[,i], p)), c(innames, parameters))
        for (o in seq_len(n_out)) {
          Hmat <- parsed_hess[[outnames[o]]]
          for (s1 in seq_len(n_diff)) for (s2 in seq_len(n_diff))
            if (!(diff_syms[s1] %in% fixed_rt) && !(diff_syms[s2] %in% fixed_rt)) {
              e <- Hmat[[diff_syms[s1], diff_syms[s2]]]
              if (!is.null(e)) arr[i, o, s1, s2] <- eval(e, env)
            }
        }
      }
    }
    arr
  }

  # BLAS-3 chain rule for symbolic mode. Falls back to per-obs %*% if the C
  # entry isn't loaded (e.g. compile = FALSE).
  chain_jac_sym <- function(J_raw, S, n_obs) {
    n_out <- length(outnames); n_diff <- length(diff_syms); n_theta <- dim(S)[3]
    chsym <- paste0(modelname, "_chain_jac")
    theta <- dimnames(S)[[3]]
    if (is.loaded(chsym)) {
      out <- .C(chsym,
                J        = as.double(J_raw),
                S        = as.double(S),
                J_theta  = double(n_obs * n_out * n_theta),
                n_obs    = as.integer(n_obs),
                n_out    = as.integer(n_out),
                n_diff   = as.integer(n_diff),
                n_theta  = as.integer(n_theta))
      array(out$J_theta, c(n_obs, n_out, n_theta), list(NULL, outnames, theta))
    } else {
      arr <- array(0, c(n_obs, n_out, n_theta), list(NULL, outnames, theta))
      for (obs in seq_len(n_obs))
        arr[obs,,] <- matrix(J_raw[obs,,], n_out, n_diff) %*% matrix(S[obs,,], n_diff, n_theta)
      arr
    }
  }

  chain_hess_sym <- function(H_raw, J_raw, S, S2, n_obs) {
    n_out <- length(outnames); n_diff <- length(diff_syms); n_theta <- dim(S)[3]
    chsym <- paste0(modelname, "_chain_hess"); theta <- dimnames(S)[[3]]
    has_S2 <- !is.null(S2)
    S2_flat <- if (has_S2) as.double(S2) else double(0)
    if (is.loaded(chsym)) {
      out <- .C(chsym,
                H        = as.double(H_raw),
                J        = as.double(J_raw),
                S        = as.double(S),
                S2_in    = S2_flat,
                H_theta  = double(n_obs * n_out * n_theta * n_theta),
                has_S2   = as.integer(has_S2),
                n_obs    = as.integer(n_obs),
                n_out    = as.integer(n_out),
                n_diff   = as.integer(n_diff),
                n_theta  = as.integer(n_theta))
      array(out$H_theta, c(n_obs, n_out, n_theta, n_theta),
            list(NULL, outnames, theta, theta))
    } else {
      arr <- array(0, c(n_obs, n_out, n_theta, n_theta),
                   list(NULL, outnames, theta, theta))
      for (obs in seq_len(n_obs)) {
        Sobs <- matrix(S[obs,,], n_diff, n_theta)
        for (o in seq_len(n_out)) {
          Hslice <- matrix(H_raw[obs, o, , ], n_diff, n_diff)
          arr[obs, o, , ] <- t(Sobs) %*% Hslice %*% Sobs
          if (has_S2) {
            Jslice <- as.numeric(J_raw[obs, o, ])
            for (i in seq_len(n_diff))
              arr[obs, o, , ] <- arr[obs, o, , ] + Jslice[i] * matrix(S2[obs, i, , ], n_theta, n_theta)
          }
        }
      }
      arr
    }
  }

  # --- Public derivative implementations ---

  jac_impl <- if (deriv) function(vars, params = numeric(0), dX = NULL, dP = NULL,
                                  attach.input = FALSE, fixed = NULL) {
    if (is.null(dX)) dX <- attr(vars, "deriv")
    if (is.null(dP)) dP <- attr(params, "deriv")
    has_seeds <- !is.null(dX) || !is.null(dP)
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    fixed_rt <- if (is.null(fixed)) character(0) else intersect(fixed, parameters)

    if (use_ad) {
      # Dual path: AD-seeded, identity seed for the raw case.
      if (!has_seeds) {
        seeds <- identitySeedsRaw(n_obs, fixed_rt)
        dX <- seeds$dX; dP <- seeds$dP
      }
      theta <- resolveTheta(dX, dP)
      aligned <- alignSeedsDual(dX, dP, NULL, NULL, n_obs, theta, fixed_rt)
      res <- call_eval_ad(M, p, aligned$dX, aligned$dP, n_obs, theta)
      arr <- res$dy
      if (!has_seeds) {
        # Drop runtime-fixed columns from the canonical-basis output.
        dsyms <- setdiff(theta, fixed_rt)
        arr <- arr[, , dsyms, drop = FALSE]
      }
    } else {
      # Symbolic path.
      raw <- raw_jac_sym(M, p, n_obs, fixed_rt)
      if (!has_seeds) {
        dsyms <- setdiff(diff_syms, fixed_rt)
        arr <- raw[, , dsyms, drop = FALSE]
      } else {
        theta <- resolveTheta(dX, dP)
        S <- buildSeedMatrix(dX, dP, n_obs, theta, fixed_rt)
        arr <- chain_jac_sym(raw, S, n_obs)
      }
    }
    attachExtras(arr, n_obs, chk$extra_vars, chk$extra_params, "jac")
  }

  hess_impl <- if (deriv2) function(vars, params = numeric(0),
                                    dX = NULL, dP = NULL, dX2 = NULL, dP2 = NULL,
                                    attach.input = FALSE, fixed = NULL) {
    if (is.null(dX))  dX  <- attr(vars,   "deriv")
    if (is.null(dP))  dP  <- attr(params, "deriv")
    if (is.null(dX2)) dX2 <- attr(vars,   "deriv2")
    if (is.null(dP2)) dP2 <- attr(params, "deriv2")
    has_seeds <- !is.null(dX) || !is.null(dP) || !is.null(dX2) || !is.null(dP2)
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    fixed_rt <- if (is.null(fixed)) character(0) else intersect(fixed, parameters)

    if (use_ad) {
      if (!has_seeds) {
        seeds <- identitySeedsRaw(n_obs, fixed_rt)
        dX <- seeds$dX; dP <- seeds$dP
      }
      theta <- resolveTheta(dX, dP, dX2, dP2)
      aligned <- alignSeedsDual(dX, dP, dX2, dP2, n_obs, theta, fixed_rt)
      res <- call_eval_ad2(M, p, aligned, n_obs, theta)
      arr <- res$d2y
      if (!has_seeds) {
        dsyms <- setdiff(theta, fixed_rt)
        arr <- arr[, , dsyms, dsyms, drop = FALSE]
      }
    } else {
      raw <- raw_hess_sym(M, p, n_obs, fixed_rt)
      if (!has_seeds) {
        dsyms <- setdiff(diff_syms, fixed_rt)
        arr <- raw[, , dsyms, dsyms, drop = FALSE]
      } else {
        theta <- resolveTheta(dX, dP, dX2, dP2)
        S  <- buildSeedMatrix(dX, dP, n_obs, theta, fixed_rt)
        S2 <- buildSeedTensor2(dX2, dP2, n_obs, theta, fixed_rt)
        # Need the raw Jacobian for the J*S2 contribution.
        J_raw <- if (!is.null(S2)) raw_jac_sym(M, p, n_obs, fixed_rt)
                 else array(0, c(n_obs, length(outnames), length(diff_syms)),
                            list(NULL, outnames, diff_syms))
        arr <- chain_hess_sym(raw, J_raw, S, S2, n_obs)
      }
    }
    attachExtras(arr, n_obs, chk$extra_vars, chk$extra_params, "hess")
  }

  evaluate_impl <- if (emit_deriv) function(vars, params = numeric(0),
                                            dX = NULL, dP = NULL,
                                            dX2 = NULL, dP2 = NULL,
                                            deriv2 = FALSE,
                                            attach.input = FALSE, fixed = NULL) {
    if (is.null(dX))  dX  <- attr(vars,   "deriv")
    if (is.null(dP))  dP  <- attr(params, "deriv")
    if (is.null(dX2)) dX2 <- attr(vars,   "deriv2")
    if (is.null(dP2)) dP2 <- attr(params, "deriv2")
    has_seeds <- !is.null(dX) || !is.null(dP) || !is.null(dX2) || !is.null(dP2)
    chk <- checkInputs(vars, params, attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    fixed_rt <- if (is.null(fixed)) character(0) else intersect(fixed, parameters)
    n_out <- length(outnames)

    if (use_ad) {
      if (!has_seeds) {
        seeds <- identitySeedsRaw(n_obs, fixed_rt)
        dX <- seeds$dX; dP <- seeds$dP
      }
      theta <- resolveTheta(dX, dP, dX2, dP2)
      aligned <- alignSeedsDual(dX, dP, dX2, dP2, n_obs, theta, fixed_rt)
      if (deriv2) {
        res <- call_eval_ad2(M, p, aligned, n_obs, theta)
        y <- res$y; dy <- res$dy; d2y <- res$d2y
        if (!has_seeds) {
          dsyms <- setdiff(theta, fixed_rt)
          dy  <- dy [, , dsyms, drop = FALSE]
          d2y <- d2y[, , dsyms, dsyms, drop = FALSE]
        }
      } else {
        res <- call_eval_ad(M, p, aligned$dX, aligned$dP, n_obs, theta)
        y <- res$y; dy <- res$dy; d2y <- NULL
        if (!has_seeds) {
          dsyms <- setdiff(theta, fixed_rt)
          dy <- dy[, , dsyms, drop = FALSE]
        }
      }
    } else {
      # Symbolic path: separate eval / jac / hess, optional chain rule.
      y <- {
        funsym <- paste0(modelname, "_eval")
        if (is.loaded(funsym)) {
          out <- .C(funsym, x = as.double(M), y = double(n_out * n_obs), p = as.double(p),
                    n = as.integer(n_obs), k = as.integer(length(innames)),
                    l = as.integer(n_out))
          matrix(out$y, n_obs, n_out, dimnames = list(NULL, outnames))
        } else {
          res <- matrix(NA_real_, n_obs, n_out, dimnames = list(NULL, outnames))
          for (i in seq_len(n_obs)) { env <- setNames(as.list(c(M[,i], p)), c(innames, parameters)); res[i,] <- vapply(parsed_exprs, function(e) eval(e, env), numeric(1)) }
          res
        }
      }
      raw_J <- raw_jac_sym(M, p, n_obs, fixed_rt)
      if (has_seeds) {
        theta <- resolveTheta(dX, dP, dX2, dP2)
        S <- buildSeedMatrix(dX, dP, n_obs, theta, fixed_rt)
        dy <- chain_jac_sym(raw_J, S, n_obs)
      } else {
        dsyms <- setdiff(diff_syms, fixed_rt)
        dy <- raw_J[, , dsyms, drop = FALSE]
      }
      if (deriv2) {
        raw_H <- raw_hess_sym(M, p, n_obs, fixed_rt)
        if (has_seeds) {
          S2 <- buildSeedTensor2(dX2, dP2, n_obs, theta, fixed_rt)
          d2y <- chain_hess_sym(raw_H, raw_J, S, S2, n_obs)
        } else {
          d2y <- raw_H[, , dsyms, dsyms, drop = FALSE]
        }
      } else d2y <- NULL
    }

    y   <- attachExtras(y,   n_obs, chk$extra_vars, chk$extra_params, "fun")
    dy  <- attachExtras(dy,  n_obs, chk$extra_vars, chk$extra_params, "jac")
    out <- list(y = y, dy = dy)
    if (deriv2) out$d2y <- attachExtras(d2y, n_obs, chk$extra_vars, chk$extra_params, "hess")
    out
  }

  # --- Convenient wrappers ---

  makeFunWrapper <- function(impl) {
    if (is.null(impl)) return(NULL)
    function(..., attach.input = FALSE, fixed = NULL) {
      args <- list(...); M <- if (length(innames)) do.call(cbind, args[innames]); p <- if (length(parameters)) do.call(c, args[parameters]) else numeric(0)
      if (attach.input) { extra <- setdiff(names(args), c(innames, parameters)); n_obs <- if (!is.null(M)) nrow(M) else 1L
      for (nm in extra) { v <- args[[nm]]; if (length(v) == n_obs) { M <- if (is.null(M)) matrix(v, ncol=1, dimnames=list(NULL,nm)) else cbind(M, setNames(data.frame(v), nm)) } else if (length(v) == 1) p <- c(p, setNames(v, nm)) else warning("Extra '", nm, "' ignored") } }
      impl(M, p, attach.input, fixed)
    }
  }

  makeDerivWrapper <- function(impl, has_d2 = FALSE) {
    if (is.null(impl)) return(NULL)
    if (has_d2) {
      function(..., dX = NULL, dP = NULL, dX2 = NULL, dP2 = NULL,
               attach.input = FALSE, fixed = NULL) {
        args <- list(...); M <- if (length(innames)) do.call(cbind, args[innames]); p <- if (length(parameters)) do.call(c, args[parameters]) else numeric(0)
        if (attach.input) { extra <- setdiff(names(args), c(innames, parameters)); n_obs <- if (!is.null(M)) nrow(M) else 1L
        for (nm in extra) { v <- args[[nm]]; if (length(v) == n_obs) { M <- if (is.null(M)) matrix(v, ncol=1, dimnames=list(NULL,nm)) else cbind(M, setNames(data.frame(v), nm)) } else if (length(v) == 1) p <- c(p, setNames(v, nm)) else warning("Extra '", nm, "' ignored") } }
        impl(M, p, dX, dP, dX2, dP2, attach.input, fixed)
      }
    } else {
      function(..., dX = NULL, dP = NULL,
               attach.input = FALSE, fixed = NULL) {
        args <- list(...); M <- if (length(innames)) do.call(cbind, args[innames]); p <- if (length(parameters)) do.call(c, args[parameters]) else numeric(0)
        if (attach.input) { extra <- setdiff(names(args), c(innames, parameters)); n_obs <- if (!is.null(M)) nrow(M) else 1L
        for (nm in extra) { v <- args[[nm]]; if (length(v) == n_obs) { M <- if (is.null(M)) matrix(v, ncol=1, dimnames=list(NULL,nm)) else cbind(M, setNames(data.frame(v), nm)) } else if (length(v) == 1) p <- c(p, setNames(v, nm)) else warning("Extra '", nm, "' ignored") } }
        impl(M, p, dX, dP, attach.input, fixed)
      }
    }
  }

  makeEvalWrapper <- function(impl) {
    if (is.null(impl)) return(NULL)
    function(..., dX = NULL, dP = NULL, dX2 = NULL, dP2 = NULL, deriv2 = FALSE,
             attach.input = FALSE, fixed = NULL) {
      args <- list(...); M <- if (length(innames)) do.call(cbind, args[innames]); p <- if (length(parameters)) do.call(c, args[parameters]) else numeric(0)
      if (attach.input) { extra <- setdiff(names(args), c(innames, parameters)); n_obs <- if (!is.null(M)) nrow(M) else 1L
      for (nm in extra) { v <- args[[nm]]; if (length(v) == n_obs) { M <- if (is.null(M)) matrix(v, ncol=1, dimnames=list(NULL,nm)) else cbind(M, setNames(data.frame(v), nm)) } else if (length(v) == 1) p <- c(p, setNames(v, nm)) else warning("Extra '", nm, "' ignored") } }
      impl(M, p, dX, dP, dX2, dP2, deriv2, attach.input, fixed)
    }
  }

  # --- Output ---
  outfn <- list(
    func     = if (convenient) makeFunWrapper(fun_impl) else fun_impl,
    jac      = if (convenient) makeDerivWrapper(jac_impl, FALSE) else jac_impl,
    hess     = if (convenient) makeDerivWrapper(hess_impl, TRUE) else hess_impl,
    evaluate = if (convenient) makeEvalWrapper(evaluate_impl) else evaluate_impl
  )
  attr(outfn, "equations") <- eqns; attr(outfn, "variables") <- variables; attr(outfn, "parameters") <- parameters
  attr(outfn, "fixed") <- fixed; attr(outfn, "modelname") <- modelname; attr(outfn, "srcfile") <- normalizePath(cpp_file, "/", FALSE)
  attr(outfn, "derivMode") <- derivMode
  for (nm in c("func", "jac", "hess", "evaluate")) {
    if (!is.null(outfn[[nm]])) {
      attr(outfn[[nm]], "modelname") <- modelname
      attr(outfn[[nm]], "srcfile")   <- attr(outfn, "srcfile")
    }
  }
  if (!is.null(sym_jac))  attr(outfn, "jacobian.symb") <- sym_jac
  if (!is.null(sym_hess)) attr(outfn, "hessian.symb")  <- sym_hess
  if (compile) compile(outfn, verbose = verbose)
  outfn
}
