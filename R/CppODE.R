#' Generate C++ code for ODE models with sensitivities
#'
#' This function builds a complete C++ implementation of a system of ordinary
#' differential equations (ODEs), including:
#'
#' * The ODE right-hand side and its Jacobian (for stiff solvers)
#' * Optional fixed-time events (state replacement, addition, multiplication)
#' * First- and second-order sensitivities with respect to model parameters
#'   and/or initial conditions
#' * Selection of "fixed" variables that are excluded from sensitivity analysis
#'
#' The generated C++ file can be compiled and called from R to efficiently
#' solve the ODE system and return trajectories together with sensitivities.
#'
#' @param odes Named character vector of ODEs (one entry per state).
#' @param events Optional `data.frame` specifying events with columns:
#'   `var` (state), `time` (numeric or symbol), `value`, and `method`
#'   (`"replace"`, `"add"`, or `"multiply"`).
#' @param fixed Optional character vector naming states and/or parameters
#'   to be treated as fixed (i.e., not included as independent variables
#'   in sensitivity analysis). Defaults to none.
#' @param compile Logical, if `TRUE` compile the generated C++ file
#'   and load the source file in R (platform-aware)
#' @param modelname Optional string used for the output filename.
#'   Defaults to a random identifier.
#' @param deriv Logical, compute first-order sensitivities (Jacobian).
#' @param secderiv Logical, compute second-order sensitivities (Hessian).
#' @param verbose Logical, print progress messages.
#'
#' @return The name of the generated C++ file (character scalar). Attributes
#'   include the ODE equations, variables, parameters, fixed variables, events,
#'   symbolic Jacobian, and solver type.
#' @import reticulate
#' @export
CppFun <- function(odes, events = NULL, fixed = NULL, compile = TRUE, modelname = NULL,
                   deriv = TRUE, secderiv = FALSE, verbose = FALSE) {

  # --- Clean ODEs ---
  odes <- unclass(odes)
  odes <- gsub("\n", "", odes)
  odes <- sanitizeExprs(odes)
  odes_attr <- attributes(odes)

  # --- States & params ---
  states  <- names(odes)
  symbols <- getSymbols(c(odes, if (!is.null(events)) c(events$value, events$time)))
  params  <- setdiff(symbols, c(states, "time"))

  # --- fixed vs sensitive ---
  if (is.null(fixed)) fixed <- character(0)
  if (!is.character(fixed)) stop("`fixed` must be character vector")

  fixed_states <- intersect(fixed, states)
  fixed_params <- intersect(fixed, params)
  sens_states  <- setdiff(states, fixed_states)
  sens_params  <- setdiff(params, fixed_params)

  # positions (0-based)
  state_idx0 <- setNames(seq_along(states) - 1L, states)
  param_idx0 <- setNames(seq_along(params) - 1L, params)

  sens_state_idx0 <- unname(state_idx0[sens_states])
  sens_param_idx0 <- unname(param_idx0[sens_params])

  # maps back to domain positions
  state_sens_pos <- rep(-1L, length(states))
  if (length(sens_state_idx0))
    state_sens_pos[sens_state_idx0 + 1L] <- seq_along(sens_state_idx0) - 1L

  param_sens_pos <- rep(-1L, length(params))
  if (length(sens_param_idx0))
    param_sens_pos[sens_param_idx0 + 1L] <- seq_along(sens_param_idx0) - 1L


  if (is.null(modelname)) {
    modelname <- paste(c("f", sample(c(letters, 0:9), 8, TRUE)), collapse = "")
  }

  # --- Sympy parser ---
  sympy  <- reticulate::import("sympy")
  parser <- reticulate::import("sympy.parsing.sympy_parser")
  syms_states <- lapply(states, function(s) sympy$Symbol(s, real = TRUE))
  names(syms_states) <- states
  syms_params <- lapply(params, function(p) sympy$Symbol(p, real = TRUE))
  names(syms_params) <- params
  local_dict <- c(syms_states, syms_params)
  local_dict[["time"]] <- sympy$Symbol("time", real = TRUE)

  transformations <- reticulate::tuple(
    c(parser$standard_transformations,
      list(parser$convert_xor, parser$implicit_multiplication_application))
  )

  # --- Includes ---
  includings <- c(
    "#define R_NO_REMAP",
    "#include <R.h>",
    "#include <Rinternals.h>",
    "#include <algorithm>",
    "#include <vector>",
    "#include <cmath>",
    "#include <boost/numeric/odeint.hpp>",
    "#include <boost/numeric/ublas/vector.hpp>",
    "#include <boost/numeric/ublas/matrix.hpp>",
    "#include <StepChecker.hpp>"
  )
  usings <- c(
    "using namespace boost::numeric::odeint;",
    "using boost::numeric::ublas::vector;",
    "using boost::numeric::ublas::matrix;"
  )

  # --- Use AD if sensitivities are requested ---
  if (deriv || secderiv) {
    numType   <- "AD"
    AD        <- TRUE
    includings <- c(includings, "#include <cppad/cppad.hpp>")
    usings     <- c(usings, "using AD = CppAD::AD<double>;")
  } else {
    numType <- "double"
    AD      <- FALSE
  }

  # --- ODE system struct ---
  ode_lines <- c(
    "// ODE system",
    "struct ode_system {",
    sprintf("  vector<%s> params;", numType),
    sprintf("  explicit ode_system(const vector<%s>& p_) : params(p_) {}", numType),
    sprintf("  void operator()(const vector<%s>& x, vector<%s>& dxdt, const %s& t) {",
            numType, numType, numType)
  )
  for (i in seq_along(states)) {
    expr <- parser$parse_expr(
      odes[[i]],
      local_dict      = local_dict,
      transformations = transformations,
      evaluate        = TRUE
    )
    rhs <- Sympy2CppCode(expr, states, params,
                         length(states), expr_name = states[i], AD)
    rhs <- if (numType == "AD")
      gsub("std::", "CppAD::", rhs, fixed = TRUE) else rhs
    ode_lines <- c(ode_lines, sprintf("    dxdt[%d] = %s;", i - 1L, rhs))
  }
  ode_lines <- c(ode_lines, "  }", "};")
  ode_lines <- paste(ode_lines, collapse = "\n")

  # --- Jacobian struct ---
  jac       <- ComputeJacobianSymb(odes, states = states, params = params, AD)
  jac_lines <- attr(jac, "CppCode")

  # --- Observer (events + output) ---
  observer_lines <- c(
    "inline void apply_event(",
    sprintf("  vector<%s>& x, int var_index, int method, const %s& value) {",
            numType, numType),
    "  if (method == 1) {",
    "    x[var_index] = value; // replace",
    "  } else if (method == 2) {",
    "    x[var_index] = x[var_index] + value; // add",
    "  } else if (method == 3) {",
    "    x[var_index] = x[var_index] * value; // multiply",
    "  }",
    "}",
    "",
    "struct observer {",
    sprintf("  std::vector<%s>& times;", numType),
    sprintf("  std::vector<%s>& y;", numType),
    sprintf("  const vector<%s>& params;", numType),
    "",
    sprintf("  explicit observer(std::vector<%s>& t, std::vector<%s>& y_, const vector<%s>& p_)",
            numType, numType, numType),
    "    : times(t), y(y_), params(p_) {}",
    "",
    # const signature
    sprintf("  void operator()(const vector<%s>& x, const %s& t) {", numType, numType),
    # const_cast hack
    sprintf("    vector<%s>& x_nc = const_cast<vector<%s>&>(x);", numType, numType)
  )

  if (!is.null(events)) {
    method_map <- c("replace" = 1L, "add" = 2L, "multiply" = 3L)
    events$method <- vapply(events$method, function(m) {
      if (is.numeric(m)) {
        if (!(m %in% 1:3)) stop("Method integers must be 1, 2 or 3.")
        as.integer(m)
      } else if (is.character(m)) {
        ml <- tolower(m)
        if (!ml %in% names(method_map)) stop("Unknown method: ", m)
        method_map[ml]
      } else {
        stop("Invalid method type: must be integer or string.")
      }
    }, integer(1))

    cpp_expr_events <- function(x) {
      num <- suppressWarnings(as.numeric(x))
      if (!is.na(num)) return(sprintf("%.17g", num))
      si <- match(x, states)
      if (!is.na(si)) return(sprintf("x_nc[%d]", si - 1L))   # <--- wichtig: x_nc
      pi <- match(x, params)
      if (!is.na(pi)) return(sprintf("params[%d]", length(states) + pi - 1L))
      stop("Unknown symbol in event: ", x)
    }
    idx_of <- function(name) match(name, states) - 1L

    ev_checks <- vapply(seq_len(nrow(events)), function(i) {
      vi    <- idx_of(events$var[i])
      texpr <- cpp_expr_events(events$time[i])
      vexpr <- cpp_expr_events(events$value[i])
      meth  <- events$method[i]
      if (numType == "AD") {
        sprintf("    if (CppAD::Value(t) == %s) apply_event(x_nc, %d, %d, %s);",
                texpr, vi, meth, vexpr)
      } else {
        sprintf("    if (t == %s) apply_event(x_nc, %d, %d, %s);",
                texpr, vi, meth, vexpr)
      }
    }, character(1))
    observer_lines <- c(observer_lines, ev_checks)
  }

  observer_lines <- c(
    observer_lines,
    "    times.push_back(t);",
    "    for (size_t i = 0; i < x_nc.size(); ++i) y.push_back(x_nc[i]);",  # <--- x_nc
    "  }",
    "};"
  )
  observer_lines <- paste(observer_lines, collapse = "\n")

  # --- Solve function ---
  xN <- length(states)
  pN <- length(params)
  sens_xN <- length(sens_states)
  sens_pN <- length(sens_params)


  # Column names
  indep_names <- c(sens_states, sens_params)
  nvars       <- length(indep_names)
  base_colnames <- c("time", states)

  deriv_colnames <- character(0)
  if (deriv) {
    deriv_colnames <- unlist(
      lapply(states, function(si) paste(si, indep_names, sep = ".")),
      use.names = FALSE
    )
  }
  sec_colnames <- character(0)
  if (secderiv) {
    pair_idx <- do.call(rbind, lapply(seq_len(nvars), function(j) {
      cbind(j, seq.int(j, nvars))
    }))
    pair_names <- paste(indep_names[pair_idx[, 1]], indep_names[pair_idx[, 2]], sep = ".")
    sec_colnames <- unlist(
      lapply(states, function(si) paste(si, pair_names, sep = ".")),
      use.names = FALSE
    )
  }


  all_colnames <- c(base_colnames, deriv_colnames, sec_colnames)
  n_base       <- length(base_colnames)
  n_deriv      <- length(deriv_colnames)
  n_sec        <- length(sec_colnames)
  ncol_total   <- n_base + n_deriv + n_sec

  base_cols_count <- n_base
  base2_offset    <- n_base + n_deriv
  triN            <- nvars * (nvars + 1L) / 2L

  # --- Extern "C" solve_modelname() ---
  externC <- c(
    sprintf('extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP) {', modelname),
    "try {",
    sprintf('  const int x_N = %d;', xN),
    sprintf('  const int p_N = %d;', pN),
    '  if (!Rf_isReal(timesSEXP) || Rf_length(timesSEXP) < 2) Rf_error("times must be numeric, length >= 2");',
    sprintf('  if (!Rf_isReal(paramsSEXP) || Rf_length(paramsSEXP) != x_N + p_N) Rf_error("params must be numeric length %d");', xN + pN),
    '  if (!Rf_isReal(abstolSEXP) || !Rf_isReal(reltolSEXP)) Rf_error("abstol/reltol must be numeric scalars");',
    '',
    '  const int T_N = Rf_length(timesSEXP);',
    '  const double* times  = REAL(timesSEXP);',
    '  const double* params = REAL(paramsSEXP);',
    '  const double abstol  = REAL(abstolSEXP)[0];',
    '  const double reltol  = REAL(reltolSEXP)[0];',
    '',
    '  if (!Rf_isInteger(maxprogressSEXP) || Rf_length(maxprogressSEXP) != 1) Rf_error("maxprogress must be a single integer");',
    '  if (!Rf_isInteger(maxstepsSEXP) || Rf_length(maxstepsSEXP) != 1) Rf_error("maxsteps must be a single integer");',
    '',
    '  int tmp_progress = INTEGER(maxprogressSEXP)[0];',
    '  int tmp_steps    = INTEGER(maxstepsSEXP)[0];',
    '  if (tmp_progress <= 0) Rf_error("maxprogress must be > 0");',
    '  if (tmp_steps    <= 0) Rf_error("maxsteps must be > 0");',
    '',
    '  StepChecker checker(tmp_progress, tmp_steps);',
    ''
  )

  # Independent + AD setup
  if (AD) {
    externC <- c(
      externC,
      sprintf("  const int sens_xN = %d;", sens_xN),
      sprintf("  const int sens_pN = %d;", sens_pN),
      "  const int dom_N = sens_xN + sens_pN;",
      if (length(sens_state_idx0)) {
        sprintf("  const int sens_state_idx[%d] = {%s};",
                length(sens_state_idx0), paste(sens_state_idx0, collapse = ","))
      } else "  const int* sens_state_idx = nullptr;",
      if (length(sens_param_idx0)) {
        sprintf("  const int sens_param_idx[%d] = {%s};",
                length(sens_param_idx0), paste(sens_param_idx0, collapse = ","))
      } else "  const int* sens_param_idx = nullptr;",
      sprintf("  const int state_sens_pos[%d] = {%s};",
              length(state_sens_pos),
              paste(state_sens_pos, collapse = ","))
      ,
      sprintf("  const int param_sens_pos[%d] = {%s};",
              length(param_sens_pos),
              paste(param_sens_pos, collapse = ","))
      ,
      sprintf("  std::vector<%s> indep(dom_N);", numType),
      "// fill sensitive states",
      "  for (int j = 0; j < sens_xN; ++j) indep[j] = params[sens_state_idx[j]];",
      "// fill sensitive params",
      "  for (int j = 0; j < sens_pN; ++j) indep[sens_xN + j] = params[x_N + sens_param_idx[j]];",
      "  CppAD::Independent(indep);",
      ""
    )
  }

  # Initial states + params vector
  externC <- c(externC,
               sprintf('  vector<%s> x(x_N);', numType),
               if (AD) c(
                 "// build x from indep or fixed params",
                 "  for (int i = 0; i < x_N; ++i) {",
                 "    if (state_sens_pos[i] >= 0) x[i] = indep[state_sens_pos[i]];",
                 "    else x[i] = params[i];",
                 "  }"
               ) else
                 "  for (int i = 0; i < x_N; ++i) x[i] = params[i];",
               '',
               sprintf('  vector<%s> full_params(x_N + p_N);', numType),
               if (AD) c(
                 "// full_params in original order: [x0..., params...]",
                 "  for (int i = 0; i < x_N; ++i) full_params[i] = x[i];",
                 "  for (int j = 0; j < p_N; ++j) {",
                 "    if (param_sens_pos[j] >= 0) full_params[x_N + j] = indep[sens_xN + param_sens_pos[j]];",
                 "    else full_params[x_N + j] = params[x_N + j];",
                 "  }"
               ) else
                 "  for (int i = 0; i < x_N + p_N; ++i) full_params[i] = params[i];",
               '',
               sprintf('  std::vector<%s> t_ad;', numType),
               '  for (int i = 0; i < T_N; ++i) t_ad.push_back(times[i]);'
  )

  # Event times
  if (!is.null(events)) {
    for (i in seq_len(nrow(events))) {
      time_str <- events$time[i]
      tnum <- suppressWarnings(as.numeric(time_str))
      if (!is.na(tnum)) {
        externC <- c(externC,
                     sprintf("  t_ad.push_back(%.17g);", tnum))
      } else {
        pi <- match(time_str, c(states, params))
        if (!is.na(pi)) {
          idx <- pi - 1L
          externC <- c(externC,
                       sprintf("  t_ad.push_back(full_params[%d]);", idx))
        } else {
          warning("Unknown event time symbol: ", time_str)
        }
      }
    }
  }

  # Sorting times
  if (AD) {
    externC <- c(externC,
                 '  std::sort(t_ad.begin(), t_ad.end(), [](const AD& a, const AD& b) { return CppAD::Value(a) < CppAD::Value(b); });',
                 ''
    )
  } else {
    externC <- c(externC,
                 '  std::sort(t_ad.begin(), t_ad.end());',
                 ''
    )
  }

  # ODE integration
  externC <- c(externC,
               sprintf('  ode_system sys(full_params);'),
               sprintf('  jacobian jac(full_params);'),
               sprintf('  rosenbrock4_controller<rosenbrock4<%s>> stepper(abstol, reltol);', numType),
               sprintf('  %s dt = (t_ad.back() - t_ad.front()) / %s(100.0);', numType, numType),
               '',
               sprintf('  std::vector<%s> result_times;', numType),
               sprintf('  std::vector<%s> y;', numType),
               sprintf('  observer obs(result_times, y, full_params);'),
               '  integrate_times(stepper, std::make_pair(sys, jac), x, t_ad.begin(), t_ad.end(), dt, obs, checker);',
               '',
               '  const int n_out = static_cast<int>(result_times.size());',
               '  if (n_out <= 0) Rf_error("Integration produced no output");',
               ''
  )

  # Build ADFun
  if (AD && (deriv || secderiv)) {
    externC <- c(externC,
                 '  CppAD::ADFun<double> f(indep, y);',
                 '  f.optimize();',
                 '',
                 '  CppAD::vector<double> xval(indep.size());',
                 '  for (size_t i = 0; i < xval.size(); ++i) xval[i] = CppAD::Value(indep[i]);',
                 sprintf('  const int nvars = %d;', nvars),
                 ''
    )
  }

  # Allocate output
  externC <- c(externC,
               sprintf('  const int nrow = n_out, ncol = %d;', ncol_total),
               '  SEXP ans = PROTECT(Rf_allocMatrix(REALSXP, nrow, ncol));',
               '  double* out = REAL(ans);',
               '  auto IDX = [nrow](int r, int c){ return r + c * nrow; };',
               ''
  )

  # Write times + states
  for_state_cols <- c(
    '  for (int i = 0; i < n_out; ++i) {',
    if (AD)
      '    out[IDX(i, 0)] = CppAD::Value(result_times[i]);'
    else
      '    out[IDX(i, 0)] = result_times[i];',
    sprintf('    for (int s = 0; s < %d; ++s) {', xN),
    if (AD)
      paste0('      out[IDX(i, 1 + s)] = CppAD::Value(y[i*', xN, ' + s]);')
    else
      paste0('      out[IDX(i, 1 + s)] = y[i*', xN, ' + s];'),
    '    }',
    '  }',
    ''
  )
  externC <- c(externC, for_state_cols)

  # Jacobian sensitivities
  if (AD && deriv) {
    externC <- c(externC,
                 '  // First-order sensitivities wrt all independent variables',
                 sprintf('  const int base_col = %d;', base_cols_count),
                 '  {',
                 '    CppAD::vector<double> J = f.Jacobian(xval);',
                 '    for (int i = 0; i < n_out; ++i) {',
                 sprintf('      for (int s = 0; s < %d; ++s) {', xN),
                 '        const int r = i * x_N + s;',
                 '        for (int v = 0; v < dom_N; ++v) {',
                 '          const int jcol = v;',
                 '          const int Jidx = r * dom_N + jcol;',
                 '          out[IDX(i, base_col + s * dom_N + v)] = J[Jidx];',
                 '        }',
                 '      }',
                 '    }',
                 '  }',
                 ''
    )
  }

  # Hessian sensitivities
  if (AD && secderiv) {
    externC <- c(externC,
                 '  // Second-order sensitivities wrt all independent variables (upper triangle)',
                 sprintf('  const int base2 = %d;', base2_offset),
                 sprintf('  const int M = %d;', triN),
                 '  {',
                 '    CppAD::vector<size_t> row, col;',
                 '    for (int r = 0; r < dom_N; ++r) {',
                 '      for (int c = r; c < dom_N; ++c) {',
                 '        row.push_back(r);',
                 '        col.push_back(c);',
                 '      }',
                 '    }',
                 '    CppAD::vector<double> w(f.Range()), h(row.size());',
                 '    CppAD::vectorBool pattern(dom_N * dom_N);',
                 '    for (int r = 0; r < dom_N; ++r) {',
                 '      for (int c = 0; c < dom_N; ++c) {',
                 '        if (c >= r) pattern[r * dom_N + c] = true;',
                 '        else        pattern[r * dom_N + c] = false;',
                 '      }',
                 '    }',
                 '    CppAD::sparse_hessian_work work;',
                 '    for (int i = 0; i < n_out; ++i) {',
                 sprintf('      for (int s = 0; s < %d; ++s) {', xN),
                 '        for (size_t k = 0; k < w.size(); ++k) w[k] = 0.0;',
                 '        const int r = i * x_N + s;',
                 '        w[r] = 1.0;',
                 '        f.SparseHessian(xval, w, pattern, row, col, h, work);',
                 '        int idx = 0;',
                 '        for (int vj = 0; vj < dom_N; ++vj) {',
                 '          for (int vk = vj; vk < dom_N; ++vk) {',
                 '            out[IDX(i, base2 + s * M + idx)] = h[idx];',
                 '            ++idx;',
                 '          }',
                 '        }',
                 '      }',
                 '    }',
                 '  }',
                 ''
    )
  }


  # Column names
  externC <- c(externC,
               sprintf('  SEXP coln = PROTECT(Rf_allocVector(STRSXP, %d));', ncol_total)
  )
  for (j in seq_len(ncol_total)) {
    nm <- all_colnames[j]
    externC <- c(externC,
                 sprintf('  SET_STRING_ELT(coln, %d, Rf_mkChar("%s"));', j - 1L, nm))
  }
  externC <- c(externC,
               '  SEXP dimn = PROTECT(Rf_allocVector(VECSXP, 2));',
               '  SET_VECTOR_ELT(dimn, 0, R_NilValue);',
               '  SET_VECTOR_ELT(dimn, 1, coln);',
               '  Rf_setAttrib(ans, R_DimNamesSymbol, dimn);',
               '',
               '  UNPROTECT(3);',
               '  return ans;',
               "  } catch (const std::exception& e) {",
               "    Rf_error(\"ODE solver failed: %s\", e.what());",
               "  } catch (...) {",
               "    Rf_error(\"ODE solver failed: unknown C++ exception\");",
               "  }",
               "}"
  )

  externC <- paste(externC, collapse = "\n")

  # --- Write cpp file ---
  filename <- paste0(modelname, ".cpp")
  sink(filename)
  cat("/** Code auto-generated by CppODE ",
      as.character(utils::packageVersion("CppODE")), " **/\n\n", sep = "")
  cat(paste(includings, collapse = "\n")); cat("\n\n")
  cat(paste(usings, collapse = "\n")); cat("\n\n")
  cat(ode_lines, "\n\n")
  cat(jac_lines, "\n\n")
  cat(observer_lines, "\n\n")
  cat(externC)
  sink()

  if (verbose) message("Wrote: ", normalizePath(filename))
  if (compile) compileAndLoad(modelname, verbose)

  attr(modelname, "equations") <- odes
  attr(modelname, "variables") <- states
  attr(modelname, "sensvariables") <- deriv_colnames
  attr(modelname, "secsensvariables") <- sec_colnames
  attr(modelname, "parameters") <- params
  attr(modelname, "events") <- events
  attr(modelname, "fixed") <- c(fixed_states, fixed_params)
  attr(modelname, "jacobian") <- list(jac$f.x, jac$f.time)
  attr(modelname, "solver") <- "boost::odeint::rosenbrock4"

  return(modelname)
}

#' Compile and load a C++ source file in R (platform-aware)
#'
#' This function compiles a C++ source file into a shared library (`.so`, `.dll`, or `.dylib`)
#' using `R CMD SHLIB`, applies platform-appropriate compiler flags, and dynamically loads
#' the resulting shared object into the current R session.
#'
#' @param filename Character string. The base name of the C++ file (without `.cpp` extension).
#' @param verbose Logical. If `TRUE`, compiler output is printed. Default is `FALSE`.
#'
#' @return Invisibly returns the name of the loaded shared object file.
#' @export
compileAndLoad <- function(filename, verbose = FALSE) {
  filename_cpp <- paste0(filename, ".cpp")
  is_windows <- .Platform$OS.type == "windows"

  cxxflags <- if (is_windows) "-std=c++17 -O2 -DNDEBUG" else "-std=c++17 -O2 -DNDEBUG -fPIC"

  include_flags <- c(
    if (!is_windows) c("-I/usr/include", "-I/usr/local/include"),
    paste0("-I", shQuote(system.file("include", package = "CppODE")))
  )

  Sys.setenv(
    PKG_CPPFLAGS = paste(include_flags, collapse = " "),
    PKG_CXXFLAGS = cxxflags
  )

  shlibOut <- system2(
    file.path(R.home("bin"), "R"),
    args = c("CMD", "SHLIB", "--preclean", shQuote(filename_cpp)),
    stdout = TRUE, stderr = TRUE
  )

  if (verbose) {
    cat(paste(shlibOut, collapse = "\n"), "\n")
  } else if (length(shlibOut)) {
    cat(paste(shlibOut[1], "\n"))
  }

  soFile <- paste0(filename, .Platform$dynlib.ext)
  if (file.exists(soFile)) {
    try(dyn.unload(soFile), silent = !verbose)
    dyn.load(soFile)
    invisible(soFile)
  } else {
    stop("Compiled shared library not found: ", soFile)
  }
}


