#' Compile Algebraic Functions with Optional Symbolic and AD Derivatives
#'
#' @description
#' Generates functions for evaluating a system of algebraic expressions
#' and, optionally, their Jacobian and Hessian. Model evaluation runs
#' through generated and compiled C++ code for performance; a pure R
#' fallback is also available when compilation is skipped.
#'
#' @details
#' When \code{compile = TRUE}, the generated source is compiled by
#' \code{\link{compile}()} and loaded. The returned helper functions
#' accept an \code{attach.input} argument that passes through inputs
#' that are not part of the model equations:
#' \itemize{
#'   \item Unknown variables (vectors of length matching the number of
#'     observations) are appended to the outputs; their Jacobian
#'     columns are zero and their Hessian slices are zero.
#'   \item Unknown parameters (scalars) are appended (broadcast over
#'     observations); their Jacobian is the identity (the parameter
#'     differentiated against itself), and the Hessian is zero.
#' }
#' For the AD path (\code{jac_chain}) pass-through derivatives are
#' pulled from the upstream seeds: extra-variable rows are taken from
#' \code{dX[name, theta, ]} and extra-parameter rows from
#' \code{dP[name, theta]}; symbols absent from both seeds contribute
#' zero rows.
#'
#' @param eqns Named character vector or list of algebraic expressions.
#'   Names define the output variables. If unnamed, default names
#'   \code{f1}, \code{f2}, \ldots are used.
#' @param variables Character vector of variable names supplied per
#'   observation. Defaults to all symbols found in \code{eqns} that are
#'   not listed in \code{parameters}.
#' @param parameters Character vector of parameter names (constant
#'   across observations).
#' @param fixed Optional character vector of symbols to treat as fixed
#'   (excluded from derivative computation).
#' @param modelname Optional base name for generated C++ symbols and
#'   files.
#' @param outdir Directory for generated C++ source files. Defaults to
#'   \code{tempdir()}.
#' @param compile Logical; if \code{TRUE}, compile and load the
#'   generated C++ code. Default \code{FALSE}.
#' @param verbose Logical; if \code{TRUE}, print progress messages.
#'   Default \code{FALSE}.
#' @param convenient Logical; if \code{TRUE} (default), return wrappers
#'   that accept named arguments rather than the low-level
#'   \code{(vars, params)} signature.
#' @param deriv Logical; if \code{TRUE} (default), enable derivative
#'   computation. When both \code{deriv} and \code{deriv2} are
#'   \code{FALSE}, no derivative entry points are generated and
#'   \code{derivMode} has no effect.
#' @param deriv2 Logical; if \code{TRUE}, enable Hessian computation
#'   (implies \code{deriv}). Hessians are only produced by the symbolic
#'   path; if \code{derivMode != "symbolic"} and \code{deriv2 = TRUE},
#'   \code{derivMode} is forced to \code{"symbolic"} with a warning.
#'   Default \code{FALSE}.
#' @param derivMode Character; one of \code{"dual"} (default) or
#'   \code{"symbolic"}. Selects how derivatives are generated when
#'   \code{deriv = TRUE} or \code{deriv2 = TRUE}:
#'   \describe{
#'     \item{\code{"dual"}}{Forward-mode AD via the in-tree
#'       \code{cppode::dual} backend (arena-allocated heap AD). Emits the
#'       chain-ruled entry \code{jac_chain} that takes upstream seeds
#'       \code{dX} (state sensitivities per observation) and \code{dP}
#'       (parameter-transform Jacobian) and returns the value together
#'       with \code{dY/dtheta} in one pass.}
#'     \item{\code{"symbolic"}}{Classical symbolic Jacobian (and
#'       Hessian if \code{deriv2}) via SymPy; exposed as \code{jac} and
#'       \code{hess}.}
#'   }
#'
#' @return
#' A list with components \code{func}, \code{jac}, \code{hess}, and
#' \code{jac_chain} (\code{NULL} when not generated), with attributes
#' \code{equations} (original expressions), \code{variables},
#' \code{parameters}, \code{fixed}, \code{modelname} (C++ identifier),
#' \code{srcfile} (path to the generated source file), and, when
#' applicable, \code{jacobian.symb} and \code{hessian.symb} (symbolic
#' derivatives).
#'
#' @seealso \code{\link{compile}} for compilation;
#'   \code{\link{derivSymb}} for symbolic differentiation.
#'
#' @export
funCpp <- function(eqns, variables = getSymbols(eqns, omit = parameters), parameters = NULL,
                   fixed = NULL, modelname = NULL, outdir = tempdir(), compile = FALSE,
                   verbose = FALSE, convenient = TRUE, deriv = TRUE, deriv2 = FALSE,
                   derivMode = c("dual", "symbolic")) {

  derivMode <- match.arg(derivMode)
  if (deriv2 && !deriv) { warning("deriv2 requires deriv. Setting deriv = TRUE."); deriv <- TRUE }
  if (deriv2 && derivMode != "symbolic") {
    warning(sprintf("deriv2 = TRUE requires derivMode = 'symbolic' (Hessian only available via the symbolic path). Forcing derivMode from '%s' to 'symbolic'.", derivMode))
    derivMode <- "symbolic"
  }
  emit_deriv    <- deriv || deriv2
  want_symbolic <- emit_deriv && derivMode == "symbolic"
  want_ad       <- emit_deriv && derivMode == "dual"

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
  if (want_symbolic) {
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
                           ad = want_ad,
                           modelname = modelname, outdir = normalizePath(outdir, "/", FALSE), version = as.character(utils::packageVersion("CppODE")))

  # --- Attach helpers ---
  # Layouts (time-first): fun res [n_obs, n_out]; jac res [n_obs, n_out, n_sym];
  # hess res [n_obs, n_out, n_sym, n_sym]. abind3/4 bind along the "output" axis (dim 2).
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

  jac_chain_impl <- if (want_ad) function(vars, params = numeric(0), dX = NULL, dP = NULL,
                                          attach.input = FALSE, fixed = NULL) {
    # Auto-extract seeds from attr(., "deriv") when not given explicitly.
    # Matches the dMod convention: solveODE() stashes state sensitivities in
    # attr(out, "deriv"); parameter transformations stash dP in attr(pars, "deriv").
    if (is.null(dX)) dX <- attr(vars, "deriv")
    if (is.null(dP)) dP <- attr(params, "deriv")
    chk <- checkInputs(vars, params, attach = attach.input); M <- chk$M; p <- chk$p; n_obs <- chk$n_obs
    n_vars <- length(innames); n_params <- length(parameters); n_out <- length(outnames)

    # Derive theta from whichever seed carries names; dX rules if both present.
    # dX is time-first [n_obs, n_vars, n_theta]; theta lives in dim 3.
    theta <- NULL
    if (!is.null(dX)) theta <- dimnames(dX)[[3]]
    if (!is.null(dP)) {
      tp <- colnames(dP)
      if (is.null(theta)) theta <- tp
      else if (!setequal(theta, tp)) stop("dX and dP have inconsistent theta names")
    }
    n_theta <- length(theta %||% character(0))

    # Align seeds to internal variable/parameter order, fill missing with 0.
    # dX is time-first: [n_obs, n_vars, n_theta] (e.g. sens1 from solveODE).
    # dP stays obs-less: [n_params, n_theta].
    dX_arr <- array(0, c(n_obs, n_vars, n_theta))
    dP_mat <- matrix(0, n_params, n_theta)
    if (n_theta > 0 && !is.null(dX)) {
      idx <- match(innames, dimnames(dX)[[2]]); present <- !is.na(idx)
      if (any(present)) dX_arr[, present, ] <- dX[, idx[present], theta, drop = FALSE]
    }
    if (n_theta > 0 && !is.null(dP)) {
      idx <- match(parameters, rownames(dP)); present <- !is.na(idx)
      if (any(present)) dP_mat[present,] <- dP[idx[present], theta, drop = FALSE]
    }
    # Fixed params don't propagate upstream derivatives.
    fixed_rt <- if (is.null(fixed)) character(0) else intersect(fixed, parameters)
    if (length(fixed_rt) && n_theta > 0) dP_mat[match(fixed_rt, parameters), ] <- 0

    funsym <- paste0(modelname, "_eval_ad")
    if (!is.loaded(funsym)) stop("AD entry '", funsym, "' is not loaded. Did you compile with derivMode = 'dual'?")
    out <- .C(funsym,
              x        = as.double(M),
              p        = as.double(p),
              dX       = as.double(dX_arr),
              dP       = as.double(dP_mat),
              y        = double(n_out * n_obs),
              dy       = double(n_out * max(n_theta, 1L) * n_obs),
              n_obs    = as.integer(n_obs),
              n_vars   = as.integer(n_vars),
              n_params = as.integer(n_params),
              n_out    = as.integer(n_out),
              n_theta  = as.integer(n_theta))
    y_mat <- matrix(out$y, n_obs, n_out, dimnames = list(NULL, outnames))
    dy_arr <- if (n_theta > 0)
      array(out$dy[seq_len(n_out * n_theta * n_obs)], c(n_obs, n_out, n_theta),
            list(NULL, outnames, theta))
    else NULL

    # attach.input: append pass-through values + chain-ruled derivative rows.
    # Values reuse the "fun" branch of attachExtras. For derivatives, each
    # extra symbol's row is pulled from dX (vars) or dP (params); missing
    # upstream rows fall back to zero.
    if (attach.input && (!is.null(chk$extra_vars) || !is.null(chk$extra_params))) {
      ev <- chk$extra_vars; ep <- chk$extra_params
      y_mat <- attachExtras(y_mat, n_obs, ev, ep, "fun")
      if (n_theta > 0) {
        extra_names <- c(if (!is.null(ev)) colnames(ev), if (!is.null(ep)) names(ep))
        extra_dy <- array(0, c(n_obs, length(extra_names), n_theta),
                          dimnames = list(NULL, extra_names, theta))
        if (!is.null(ev) && !is.null(dX)) {
          evn <- colnames(ev); idx <- match(evn, dimnames(dX)[[2]])
          present <- !is.na(idx)
          if (any(present))
            extra_dy[, evn[present], ] <- dX[, idx[present], theta, drop = FALSE]
        }
        if (!is.null(ep) && !is.null(dP)) {
          epn <- names(ep); idx <- match(epn, rownames(dP))
          present <- !is.na(idx)
          if (any(present)) {
            dP_sub <- dP[idx[present], theta, drop = FALSE]
            target <- match(epn[present], extra_names)
            for (k in seq_along(target)) extra_dy[, target[k], ] <- rep(dP_sub[k, ], each = n_obs)
          }
        }
        dy_arr <- abind3(dy_arr, extra_dy)
      }
    }

    list(y = y_mat, dy = dy_arr)
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
  outfn <- list(
    func      = if (convenient) makeWrapper(fun_impl) else fun_impl,
    jac       = if (convenient) makeWrapper(jac_impl) else jac_impl,
    hess      = if (convenient) makeWrapper(hess_impl) else hess_impl,
    jac_chain = jac_chain_impl
  )
  attr(outfn, "equations") <- eqns; attr(outfn, "variables") <- variables; attr(outfn, "parameters") <- parameters
  attr(outfn, "fixed") <- fixed; attr(outfn, "modelname") <- modelname; attr(outfn, "srcfile") <- normalizePath(cpp_file, "/", FALSE)
  attr(outfn, "derivMode") <- derivMode
  # Mirror modelname/srcfile onto each individual function so downstream
  # callers (e.g. dMod's collectCompileInfo) can introspect them without
  # needing a handle to the parent list.
  for (nm in c("func", "jac", "hess", "jac_chain")) {
    if (!is.null(outfn[[nm]])) {
      attr(outfn[[nm]], "modelname") <- modelname
      attr(outfn[[nm]], "srcfile")   <- attr(outfn, "srcfile")
    }
  }
  if (deriv && !is.null(sym_jac)) attr(outfn, "jacobian.symb") <- sym_jac
  if (deriv2 && !is.null(sym_hess)) attr(outfn, "hessian.symb") <- sym_hess
  if (compile) compile(outfn, verbose = verbose)
  outfn
}
