#' Generate full C++ code for an ODE model with sensitivities
#'
#' This function orchestrates ODE definition, Jacobian generation,
#' event handling, and sensitivity calculation into a complete C++ file.
#'
#' @param odes Named character vector of ODEs.
#' @param events Optional data.frame of fixed-time events (see GetBoostObserver).
#' @param fixed Not yet implemented.
#' @param compile Not yet implemented.
#' @param modelname Optional string, defaults to random.
#' @param deriv Logical, include first derivatives (Jacobian)?
#' @param secderiv Logical, include second derivatives (Hessian)?
#' @param verbose Logical, print progress?
#'
#' @return Character name of file
#' @import reticulate
#' @export
CppFun <- function(odes, events = NULL,
                   compile = TRUE, modelname = NULL,
                   deriv = TRUE, secderiv = FALSE, verbose = FALSE) {

  odes <- unclass(odes)
  odes <- gsub("\n", "", odes)
  odes_attr <- attributes(odes)

  # States & params
  states <- names(odes)
  symbols <- getSymbols(c(odes, if (!is.null(events)) c(events$value, events$time)))
  params <- setdiff(symbols, c(states, "time"))

  if (is.null(modelname)) {
    modelname <- paste(c("f", sample(c(letters, 0:9), 8, TRUE)), collapse = "")
  }

  # --- Generate ODE system code ---
  sympy   <- reticulate::import("sympy")
  parser  <- reticulate::import("sympy.parsing.sympy_parser")

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

  includings <- c("#define R_NO_REMAP",
                  "#include <R.h>",
                  "#include <Rinternals.h>",
                  "#include <algorithm>",
                  "#include <vector>",
                  "#include <cmath>",
                  "#include <boost/numeric/odeint.hpp>",
                  "#include <boost/numeric/ublas/vector.hpp>",
                  "#include <boost/numeric/ublas/matrix.hpp>")

  usings <- c("using namespace boost::numeric::odeint;",
              "using boost::numeric::ublas::vector;",
              "using boost::numeric::ublas::matrix;")

  # USE CppAD:AD<double> if sensitivities are requested
  if (deriv || secderiv) {
    numType <- "AD"
    AD <- TRUE
    includings <- c(includings, "#include <cppad/cppad.hpp>")
    usings <- c(usings, "using AD = CppAD::AD<double>;")
  } else {
    AD <- FALSE
    numType <- "double"
  }

  # --- ODE struct (rhs) ---
  ode_lines <- c(
    "// ODE system",
    "struct ode_system {",
    sprintf("  vector<%s> params;", numType),
    sprintf("  explicit ode_system(const vector<%s>& p_) : params(p_) {}", numType),
    sprintf("  void operator()(const vector<%s>& x, vector<%s>& dxdt, const %s& t) {", numType, numType, numType)
  )
  for (i in seq_along(states)) {
    expr <- parser$parse_expr(odes[[i]],
                              local_dict      = local_dict,
                              transformations = transformations,
                              evaluate        = TRUE)
    rhs <- Sympy2CppCode(expr, states, params, length(states), expr_name = states[i], AD)
    rhs <- if (numType == "AD") gsub("std::", "CppAD::", rhs, fixed = TRUE) else rhs
    ode_lines <- c(ode_lines, sprintf("    dxdt[%d] = %s;", i - 1L, rhs))
  }
  ode_lines <- c(ode_lines, "  }", "};")
  ode_lines <- paste(ode_lines, collapse = "\n")

  # --- Jacobian code (system Jacobian wrt states for Rosenbrock) ---
  jac <- ComputeJacobianSymb(odes, states = states, params = params, AD)
  jac_lines <- attr(jac, "CppCode")
  # (Annahme: jacobian-struct enthält bereits passenden Konstruktor)

  # --- Observer code (Events + Speicherung) ---
  observer_lines <- c(
    "inline void apply_event(",
    sprintf("  vector<%s>& x, int var_index, int method, const %s& value) {", numType, numType),
    "  if (method == 1) {",
    "    x[var_index] = value;                 // replace",
    "  } else if (method == 2) {",
    "    x[var_index] = x[var_index] + value;  // add",
    "  } else if (method == 3) {",
    "    x[var_index] = x[var_index] * value;  // multiply",
    "  }",
    "}",
    "",
    "struct observer {",
    sprintf("  std::vector<%s>& times;", numType),
    sprintf("  std::vector<%s>& y;", numType),
    sprintf("  const vector<%s>& params;", numType),
    "",
    sprintf("  explicit observer(std::vector<%s>& t, std::vector<%s>& y_, const vector<%s>& p_)", numType, numType, numType),
    "    : times(t), y(y_), params(p_) {}",
    "",
    sprintf("  void operator()(vector<%s>& x, const %s& t) {", numType, numType)
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
      if (!is.na(si)) return(sprintf("x[%d]", si - 1L))
      pi <- match(x, params)
      if (!is.na(pi)) return(sprintf("params[%d]", length(states) + pi - 1L))
      stop("Unknown symbol in event: ", x)
    }

    idx_of <- function(name) match(name, states) - 1L

    ev_checks <- vapply(seq_len(nrow(events)), function(i) {
      vi <- idx_of(events$var[i])
      texpr <- cpp_expr_events(events$time[i])
      vexpr <- cpp_expr_events(events$value[i])
      meth <- events$method[i]
      if (numType == "AD") {
        sprintf("    if (CppAD::Value(t) == %s) apply_event(x, %d, %d, %s);",
                texpr, vi, meth, vexpr)
      } else {
        sprintf("    if (t == %s) apply_event(x, %d, %d, %s);",
                texpr, vi, meth, vexpr)
      }
    }, character(1))

    observer_lines <- c(observer_lines, ev_checks)
  }

  observer_lines <- c(
    observer_lines,
    "    times.push_back(t);",
    "    for (size_t i = 0; i < x.size(); ++i) y.push_back(x[i]);",
    "  }",
    "};"
  )
  observer_lines <- paste(observer_lines, collapse = "\n")

  # --- Extern C solve() ---
  xN <- length(states)
  pN <- length(params)

  # === Column names & sizes (auto from states + params) ===
  indep_names <- c(states, params)
  nvars <- length(indep_names)

  base_colnames <- c("time", states)
  deriv_colnames <- character(0)
  if (deriv) {
    # Für jeden Zustand si: si.<alle indep in Reihenfolge>
    deriv_colnames <- unlist(lapply(states, function(si) paste(si, indep_names, sep = ".")), use.names = FALSE)
  }
  sec_colnames <- character(0)
  if (secderiv) {
    # Oberes Dreieck über alle indep (j <= k)
    idx <- which(outer(seq_len(nvars), seq_len(nvars), function(i, j) i <= j), arr.ind = TRUE)
    pair_names <- paste(indep_names[idx[, 1]], indep_names[idx[, 2]], sep = ".")
    # Für jeden Zustand si: si.<alle (j<=k)-Paare in Reihenfolge>
    sec_colnames <- unlist(lapply(states, function(si) paste(si, pair_names, sep = ".")), use.names = FALSE)
  }
  all_colnames <- c(base_colnames, deriv_colnames, sec_colnames)

  n_base  <- length(base_colnames)                           # 1 + xN
  n_deriv <- if (deriv)     length(deriv_colnames) else 0L   # xN * nvars
  n_sec   <- if (secderiv)  length(sec_colnames)  else 0L    # xN * nvars*(nvars+1)/2
  ncol_total <- n_base + n_deriv + n_sec

  base_cols_count <- n_base
  base2_offset    <- n_base + n_deriv
  triN <- nvars * (nvars + 1L) / 2L

  # C++ helper for upper-triangle indices over ALL indep (only if secderiv)
  make_upper_indices_code <- if (secderiv) {
    paste0(
      "inline void make_upper_triangle_indices_all(size_t nvars, ",
      "CppAD::vector<size_t>& row, CppAD::vector<size_t>& col) {\n",
      "  row.clear(); col.clear();\n",
      "  for (size_t j = 0; j < nvars; ++j)\n",
      "    for (size_t k = j; k < nvars; ++k) { row.push_back(j); col.push_back(k); }\n",
      "}\n"
    )
  } else ""

  externC <- c(
    'extern "C" SEXP solve(SEXP timesSEXP, SEXP paramsSEXP, SEXP abstolSEXP, SEXP reltolSEXP) {',
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
    ''
  )

  if (AD) {
    externC <- c(externC,
                 sprintf('  std::vector<%s> indep(x_N + p_N);', numType),
                 '  for (int i = 0; i < x_N + p_N; ++i) indep[i] = params[i];',
                 '  CppAD::Independent(indep);',
                 ''
    )
  }

  externC <- c(externC,
               sprintf('  vector<%s> x(x_N);', numType),
               if (AD) '  for (int i = 0; i < x_N; ++i) x[i] = indep[i];'
               else    '  for (int i = 0; i < x_N; ++i) x[i] = params[i];',
               '',
               # full_params: indep (AD) bzw. copy aus params (double)
               sprintf('  vector<%s> full_params(x_N + p_N);', numType),
               if (AD) '  for (int i = 0; i < x_N + p_N; ++i) full_params[i] = indep[i];'
               else    '  for (int i = 0; i < x_N + p_N; ++i) full_params[i] = params[i];',
               '',
               sprintf('  std::vector<%s> t_ad;', numType),
               '  for (int i = 0; i < T_N; ++i) t_ad.push_back(times[i]);'
  )

  # Add event times (constants or symbols in full_params)
  if (!is.null(events)) {
    for (i in seq_len(nrow(events))) {
      time_str <- events$time[i]
      tnum <- suppressWarnings(as.numeric(time_str))
      if (!is.na(tnum)) {
        externC <- c(externC, sprintf("  t_ad.push_back(%.17g);", tnum))
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

  # sorting
  if (AD) {
    externC <- c(externC,
                 '  std::sort(t_ad.begin(), t_ad.end(), [](const AD& a, const AD& b) { return CppAD::Value(a) < CppAD::Value(b); });',
                 ''
    )
  } else {
    externC <- c(externC, '  std::sort(t_ad.begin(), t_ad.end());', '')
  }

  externC <- c(externC,
               sprintf('  ode_system sys(full_params);'),
               sprintf('  jacobian jac(full_params);'),
               sprintf('  rosenbrock4_controller< rosenbrock4<%s> > stepper(abstol, reltol);', numType),
               sprintf('  %s dt = (t_ad.back() - t_ad.front()) / %s(100.0);', numType, numType),
               '',
               sprintf('  std::vector<%s> result_times;', numType),
               sprintf('  std::vector<%s> y;', numType),
               sprintf('  observer obs(result_times, y, full_params);'),
               '  integrate_times(stepper, std::make_pair(sys, jac), x, t_ad.begin(), t_ad.end(), dt, obs);',
               '',
               '  const int n_out = static_cast<int>(result_times.size());',
               '  if (n_out <= 0) Rf_error("Integration produced no output");',
               ''
  )

  # ADFun + sensitivities (only if requested)
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

  # Allocate R matrix
  externC <- c(externC,
               sprintf('  const int nrow = n_out, ncol = %d;', ncol_total),
               '  SEXP ans = PROTECT(Rf_allocMatrix(REALSXP, nrow, ncol));',
               '  double* out = REAL(ans);',
               '  auto IDX = [nrow](int r, int c){ return r + c * nrow; };',
               ''
  )

  # Write time and states
  for_state_cols <- c(
    '  for (int i = 0; i < n_out; ++i) {',
    if (AD) '    out[IDX(i, 0)] = CppAD::Value(result_times[i]);'
    else    '    out[IDX(i, 0)] = result_times[i];',
    sprintf('    for (int s = 0; s < %d; ++s) {', xN),
    if (AD) paste0('      out[IDX(i, 1 + s)] = CppAD::Value(y[i*', xN, ' + s]);')
    else    paste0('      out[IDX(i, 1 + s)] = y[i*', xN, ' + s];'),
    '    }',
    '  }',
    ''
  )
  externC <- c(externC, for_state_cols)

  # 1st order sensitivities wrt ALL indep (inits + params)
  if (AD && deriv) {
    externC <- c(externC,
                 '  // First-order sensitivities wrt all independent variables',
                 sprintf('  const int base_col = %d;', base_cols_count),
                 '  {',
                 '    CppAD::vector<double> J = f.Jacobian(xval);',
                 '    for (int i = 0; i < n_out; ++i) {',
                 sprintf('      for (int s = 0; s < %d; ++s) {', xN),
                 '        const int r = i * x_N + s;',
                 sprintf('        for (int v = 0; v < %d; ++v) {', nvars),
                 '          const int jcol = v;',
                 '          const int Jidx = r * (x_N + p_N) + jcol;',
                 '          out[IDX(i, base_col + s * (x_N + p_N) + v)] = J[Jidx];',
                 '        }',
                 '      }',
                 '    }',
                 '  }',
                 ''
    )
  }

  # 2nd order sensitivities wrt ALL indep (upper triangle only)
  if (AD && secderiv) {
    externC <- c(
      externC,
      '  // Second-order sensitivities wrt all independent variables (upper triangle)',
      sprintf('  const int base2 = %d;', base2_offset),
      sprintf('  const int M = %d;', triN),
      '  {',
      '    CppAD::vector<size_t> row, col;',
      '    make_upper_triangle_indices_all(M == 0 ? 0 : (x_N + p_N), row, col);',
      '    CppAD::vector<double> w(f.Range()), h(row.size());',
      '    CppAD::vector<bool> pattern((x_N + p_N) * (x_N + p_N));',
      '    for (size_t iP = 0; iP < pattern.size(); ++iP) pattern[iP] = true;',
      '    CppAD::sparse_hessian_work work;',
      '    for (int i = 0; i < n_out; ++i) {',
      sprintf('      for (int s = 0; s < %d; ++s) {', xN),
      '        for (size_t k = 0; k < w.size(); ++k) w[k] = 0.0;',
      '        const int r = i * x_N + s;',
      '        w[r] = 1.0;',
      '        f.SparseHessian(xval, w, pattern, row, col, h, work);',
      '        int idx = 0;',
      '        for (int vj = 0; vj < (x_N + p_N); ++vj) {',
      '          for (int vk = vj; vk < (x_N + p_N); ++vk) {',
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
    externC <- c(externC, sprintf('  SET_STRING_ELT(coln, %d, Rf_mkChar("%s"));', j - 1L, nm))
  }
  externC <- c(externC,
               '  SEXP dimn = PROTECT(Rf_allocVector(VECSXP, 2));',
               '  SET_VECTOR_ELT(dimn, 0, R_NilValue);',
               '  SET_VECTOR_ELT(dimn, 1, coln);',
               '  Rf_setAttrib(ans, R_DimNamesSymbol, dimn);',
               '',
               '  UNPROTECT(3);',
               '  return ans;',
               '}'
  )

  externC_lines <- paste(externC, collapse = "\n")

  # --- Cpp file merging
  filename <- paste0(modelname, ".cpp")
  sink(filename)
  cat("/** Code auto-generated by CppODE ", as.character(utils::packageVersion("CppODE")), " **/\n\n", sep = "")
  cat(paste(includings, collapse = "\n"))
  cat("\n\n")
  cat(paste(usings, collapse = "\n"))
  cat("\n\n")
  cat(ode_lines)
  cat("\n\n")
  cat(jac_lines)
  cat("\n\n")
  cat(observer_lines)
  cat("\n\n")
  if (secderiv) cat(make_upper_indices_code, "\n")
  cat(externC_lines)
  sink()

  if (verbose) {
    message("Wrote: ", normalizePath(filename))
  }

  return(modelname)
}



