#' Generate C++ code for ODE models with sensitivities and events
#'
#' This function generates a C++ ODE solver using Boost.Odeint’s stiff
#' Rosenbrock 4 method with embedded 3(4) error control and dense output.
#' The generated C++ code supports:
#' \itemize{
#'   \item Fixed-time and root-triggered events
#'   \item Optional sensitivity analysis via automatic differentiation (AD) in forward mode
#' }
#'
#' ## Events
#' Events can be specified in a data.frame with columns:
#' \describe{
#'   \item{var}{Name of the affected state variable}
#'   \item{value}{Numeric value to apply at the event}
#'   \item{method}{"replace", "add", or "multiply"}
#'   \item{time}{(optional) numeric time point of the event}
#'   \item{root}{(optional) root expression in terms of \code{x} and \code{time}}
#' }
#' Each event must specify either a `time` or a `root`.
#'
#' ## Sensitivities
#' If \code{deriv = TRUE}, the system is augmented with automatic differentiation
#' using FADBAD++. State and parameter sensitivities are computed and returned.
#' If \code{deriv = FALSE}, the solver runs with plain doubles and no sensitivities.
#'
#' @param odes Named character vector of ODE right-hand sides.
#'   Names correspond to state variables.
#' @param events Data frame describing events (see Details).
#'   Default: \code{NULL} (no events).
#' @param fixed Character vector of fixed states or parameters.
#'   Only used if \code{deriv = TRUE}.
#' @param includeTimeZero Logical. If \code{TRUE}, ensure that \code{0} is
#'   included in the integration times even if not supplied by the user.
#'   Default: \code{TRUE}.
#' @param compile Logical. If \code{TRUE}, compiles and loads the generated C++ code.
#' @param modelname Optional character string for the output filename base.
#'   If \code{NULL}, a random identifier is generated.
#' @param deriv Logical. If \code{TRUE}, compute sensitivities using AD in forward mode.
#'   If \code{FALSE}, use plain doubles (faster, but no sensitivities).
#' @param verbose Logical. If \code{TRUE}, print progress messages.
#'
#' @return A string with the model name (same as \code{modelname}).
#'   Attributes:
#'   \itemize{
#'     \item \code{equations}: ODE definitions
#'     \item \code{variables}: State variable names
#'     \item \code{parameters}: Parameter names
#'     \item \code{events}: Events data.frame
#'     \item \code{solver}: Solver description
#'     \item \code{fixed}: Fixed states/parameters
#'     \item \code{jacobian}: Symbolic Jacobian expressions
#'     \item \code{sensvariables}: Names of sensitivity variables (if any)
#'   }
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @importFrom reticulate import
#' @example
#' \dontrun{
#' eqns <- c(
#' A = "-k1*A^2 * time",
#' B = "k1*A^2 * time - k2*B"
#' )
#'
#' events <- data.frame(
#'   var   = "A",
#'   time  = "t_e",
#'   value = 1,
#'   method= "add",
#'   root = NA
#' )
#'
#' f <- CppODE::CppFun(eqns, events = events, modelname = "Amodel_s")
#'
#' solve <- function(times, params, abstol = 1e-8, reltol = 1e-6, maxattemps = 5000, maxsteps = 1e6) {
#'   paramnames <- c(attr(f,"variables"), attr(f,"parameters"))
#'   # check for missing parameters
#'   missing <- setdiff(paramnames, names(params))
#'   if (length(missing) > 0) stop(sprintf("Missing parameters: %s", paste(missing, collapse = ", ")))
#'   params <- params[paramnames]
#'   .Call(paste0("solve_",as.character(f)),
#'         as.numeric(times),
#'         as.numeric(params),
#'         as.numeric(abstol),
#'         as.numeric(reltol),
#'         as.integer(maxattemps),
#'         as.integer(maxsteps))
#' }
#'
#' params <- c(A = 1, B=0, k1 = 0.1, k2= 0.2, t_e = 3)
#' times  <- seq(0, 10, length.out = 300)
#'
#' res <- solve(times, params, abstol = 1e-6, reltol = 1e-6)
#' head(res)
#' }
#' @export
CppFun <- function(odes, events = NULL, fixed = NULL, includeTimeZero = TRUE,
                   compile = TRUE, modelname = NULL,
                   deriv = TRUE, verbose = FALSE) {

  # --- Clean up ODE definitions ---
  odes <- unclass(odes)
  odes <- gsub("\n", "", odes)
  odes <- sanitizeExprs(odes)

  # --- Extract state and parameter names ---
  states  <- names(odes)
  symbols <- getSymbols(c(odes, if (!is.null(events)) {
    c(events$value,
      if ("time" %in% names(events)) events$time,
      if ("root" %in% names(events)) events$root)
  }))
  params  <- setdiff(symbols, c(states, "time"))

  # --- Handle fixed states/parameters ---
  if (is.null(fixed)) fixed <- character(0)
  fixed_states <- if (deriv) intersect(fixed, states) else character(0)
  fixed_params <- if (deriv) intersect(fixed, params) else character(0)
  sens_states  <- if (deriv) setdiff(states, fixed_states) else character(0)
  sens_params  <- if (deriv) setdiff(params, fixed_params) else character(0)

  # Index maps
  state_idx0 <- setNames(seq_along(states) - 1L, states)
  param_idx0 <- setNames(seq_along(params) - 1L, params)
  fixed_state_idx  <- state_idx0[fixed_states]
  fixed_param_idx  <- param_idx0[fixed_params]

  # --- Generate unique model name if not provided ---
  if (is.null(modelname)) {
    modelname <- paste(c("f", sample(c(letters, 0:9), 8, TRUE)), collapse = "")
  }

  # --- Includes ---
  includings <- c(
    "#define R_NO_REMAP",
    "#include <R.h>",
    "#include <Rinternals.h>",
    "#include <algorithm>",
    "#include <vector>",
    "#include <cmath>",
    "#include <boost_rosenbrock34_fad.hpp>"
  )

  # --- Select numerical type ---
  if (deriv) {
    numType <- "AD"
    usings <- c(
      "using namespace boost::numeric::odeint;",
      "namespace ublas = boost::numeric::ublas;",
      "using AD = fadbad::F<double>;"
    )
  } else {
    numType <- "double"
    usings <- c(
      "using namespace boost::numeric::odeint;",
      "namespace ublas = boost::numeric::ublas;"
    )
  }

  # --- ODE system ---
  sympy  <- reticulate::import("sympy")
  parser <- reticulate::import("sympy.parsing.sympy_parser")

  syms_states <- lapply(states, function(s) sympy$Symbol(s, real = TRUE))
  names(syms_states) <- states
  syms_params <- lapply(params, function(p) sympy$Symbol(p, real = TRUE))
  names(syms_params) <- params
  t <- sympy$Symbol("time", real = TRUE)

  local_dict <- c(syms_states, syms_params)
  local_dict[["time"]] <- t

  transformations <- reticulate::tuple(
    c(parser$standard_transformations,
      list(parser$convert_xor, parser$implicit_multiplication_application))
  )

  ode_lines <- c(
    "// ODE system",
    "struct ode_system {",
    sprintf("  ublas::vector<%s> params;", numType),
    sprintf("  explicit ode_system(const ublas::vector<%s>& p_) : params(p_) {}", numType),
    sprintf("  void operator()(const ublas::vector<%s>& x, ublas::vector<%s>& dxdt, const %s& t) {",
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
                         length(states),
                         expr_name = states[i],
                         AD = (numType == "AD"))
    ode_lines <- c(ode_lines, sprintf("    dxdt[%d] = %s;", i - 1L, rhs))
  }
  ode_lines <- c(ode_lines, "  }", "};")
  ode_lines <- paste(ode_lines, collapse = "\n")

  # --- Jacobian ---
  jac <- ComputeJacobianSymb(odes, states = states, params = params, AD = deriv)
  jac_lines <- attr(jac, "CppCode")

  # --- Observer ---
  observer_lines <- c(
    "// Observer: stores trajectory values",
    "struct observer {",
    sprintf("  std::vector<%s>& times;", numType),
    sprintf("  std::vector<%s>& y;", numType),
    "",
    sprintf("  explicit observer(std::vector<%s>& t, std::vector<%s>& y_)",
            numType, numType),
    "    : times(t), y(y_) {}",
    "",
    sprintf("  void operator()(const ublas::vector<%s>& x, const %s& t) {", numType, numType),
    "    times.push_back(t);",
    "    for (size_t i = 0; i < x.size(); ++i) y.push_back(x[i]);",
    "  }",
    "};"
  )
  observer_lines <- paste(observer_lines, collapse = "\n")

  # --- Solver function ---
  externC <- c(
    sprintf('extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP) {', modelname),
    "try {",
    sprintf('  const int x_N = %d;', length(states)),
    sprintf('  const int p_N = %d;', length(params)),
    "  const int dom_N = x_N + p_N;",
    "",
    "  StepChecker checker(INTEGER(maxprogressSEXP)[0], INTEGER(maxstepsSEXP)[0]);",
    "",
    sprintf("  ublas::vector<%s> x(x_N);", numType),
    sprintf("  ublas::vector<%s> full_params(dom_N);", numType),
    "")

  if (deriv) {
    # declare fixed indices
    if (length(fixed_state_idx) > 0) {
      externC <- c(externC,
                   sprintf("  std::vector<int> fixed_state_idx = {%s};",
                           paste(fixed_state_idx, collapse = ",")))
    } else {
      externC <- c(externC, "  std::vector<int> fixed_state_idx;")
    }
    if (length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   sprintf("  std::vector<int> fixed_param_idx = {%s};",
                           paste(fixed_param_idx, collapse = ",")))
    } else {
      externC <- c(externC, "  std::vector<int> fixed_param_idx;")
    }
  }

  # initialization of states and parameters
  externC <- c(externC,
               "  // initialize states",
               "  for (int i = 0; i < x_N; ++i) {",
               "    x[i] = REAL(paramsSEXP)[i];",
               if (deriv) "    x[i].diff(i, dom_N);" else "",
               "    full_params[i] = x[i];",
               "  }",
               "",
               "  // initialize parameters",
               "  for (int i = 0; i < p_N; ++i) {",
               "    full_params[x_N + i] = REAL(paramsSEXP)[x_N + i];",
               if (deriv) paste0(
                 "    if (std::find(fixed_param_idx.begin(), fixed_param_idx.end(), i) == fixed_param_idx.end()) ",
                 "full_params[x_N + i].diff(x_N + i, dom_N);"
               ) else "",
               "  }",
               "",
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
               "  // --- Event containers ---",
               sprintf("  std::vector<FixedEvent<%s>> fixed_events;", numType),
               sprintf("  std::vector<RootEvent<ublas::vector<%s>, %s>> root_events;", numType, numType)
  )

  # --- Event parsing helper ---
  parse_or_literal <- function(expr, expr_name) {
    if (is.na(expr)) return(NULL)
    if (is.numeric(expr)) {
      return(as.character(expr))
    } else {
      e <- parser$parse_expr(
        expr,
        local_dict      = local_dict,
        transformations = transformations,
        evaluate        = TRUE
      )
      code <- Sympy2CppCode(e, states, params, length(states),
                            expr_name = expr_name,
                            AD = (numType == "AD"))
      code <- gsub("\\bparams\\[", "full_params[", code)
      return(code)
    }
  }

  # --- Events ---
  if (!is.null(events)) {
    for (i in seq_len(nrow(events))) {
      var_idx <- state_idx0[events$var[i]]
      val_code  <- parse_or_literal(events$value[i],  paste0("event_val_", i))
      time_code <- parse_or_literal(events$time[i],   paste0("event_time_", i))
      root_code <- parse_or_literal(events$root[i],   paste0("event_root_", i))

      method <- switch(tolower(events$method[i]),
                       "replace"  = "EventMethod::Replace",
                       "add"      = "EventMethod::Add",
                       "multiply" = "EventMethod::Multiply",
                       stop("Unknown method"))

      if (!is.null(time_code)) {
        externC <- c(externC,
                     sprintf("  fixed_events.emplace_back(FixedEvent<%s>{%s, %d, %s, %s});",
                             numType, time_code, var_idx, val_code, method))
      } else if (!is.null(root_code)) {
        externC <- c(externC,
                     sprintf("  root_events.push_back(RootEvent<ublas::vector<%s>, %s>{",
                             numType, numType),
                     sprintf("    [](const ublas::vector<%s>& x, const %s& t){ return %s; },",
                             numType, numType, root_code),
                     sprintf("    %d, %s, %s});", var_idx, val_code, method))
      } else {
        stop("Event row ", i, " has neither time nor root defined")
      }
    }
  }

  # --- Integration ---
  if (deriv) {
    stepper_line <- paste(
      "  auto controlledStepper = rosenbrock4_controller_ad<rosenbrock4<AD>>(abstol, reltol);",
      "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);",
      sep = "\n"
    )
  } else {
    stepper_line <- paste(
      "  auto controlledStepper = rosenbrock4_controller<rosenbrock4<double>>(abstol, reltol);",
      "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);",
      sep = "\n"
    )
  }

  externC <- c(externC,'\n',
               '  // --- Solver setup ---',
               '  double abstol = REAL(abstolSEXP)[0];',
               '  double reltol = REAL(reltolSEXP)[0];',
               '  ode_system sys(full_params);',
               '  jacobian jac(full_params);',
               stepper_line, '\n',
               '  // --- Determine dt ---',
               sprintf('  %s dt0 = odeint_utils::estimate_initial_dt(sys, jac, x, times.front(), times.back(), abstol, reltol);', numType),
               '  auto t_test = times.front();',
               '  auto x_test = x;',
               '  auto dt = dt0;',
               '  int attempts = 0;',
               '  while (controlledStepper.try_step(std::make_pair(sys, jac), x_test, t_test, dt) == fail) {',
               '    if (++attempts >= 10000) throw std::runtime_error("Unable to find valid initial stepsize after 10000 attempts");',
               '  }\n',
               '  // --- Integration ---',
               sprintf('  std::vector<%s> result_times;', numType),
               sprintf('  std::vector<%s> y;', numType),
               '  observer obs(result_times, y);',
               '  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker);',
               '  const int n_out = static_cast<int>(result_times.size());',
               '  if (n_out <= 0) Rf_error("Integration produced no output");\n'
  )

  if (deriv) {
    # precompute counts
    n_sens_states <- length(sens_states)
    n_sens_params <- length(sens_params)
    externC <- c(externC,
                 '  // --- Store Results ---',
                 sprintf("  const int ncol = 1 + x_N + %d * (%d + %d);",
                         length(states), n_sens_states, n_sens_params),
                 '  SEXP ans = PROTECT(Rf_allocMatrix(REALSXP, n_out, ncol));',
                 '  double* out = REAL(ans);',
                 '  auto IDX = [n_out](int r, int c){ return r + c * n_out; };',
                 '  for (int i = 0; i < n_out; ++i) {',
                 '    out[IDX(i, 0)] = result_times[i].x();',
                 '    for (int s = 0; s < x_N; ++s) {',
                 '      AD& xi = y[i * x_N + s];',
                 '      out[IDX(i, 1 + s)] = xi.x();',
                 '    }',
                 '    int col_base = 1 + x_N;',
                 '    for (int s = 0; s < x_N; ++s) {',
                 '      AD& xi = y[i * x_N + s];',
                 '      for (int v = 0; v < dom_N; ++v) {',
                 '        bool is_fixed_init  = (v < x_N) && (std::find(fixed_state_idx.begin(), fixed_state_idx.end(), v) != fixed_state_idx.end());',
                 '        bool is_fixed_param = (v >= x_N) && (std::find(fixed_param_idx.begin(), fixed_param_idx.end(), v - x_N) != fixed_param_idx.end());',
                 '        if (!(is_fixed_init || is_fixed_param)) {',
                 '          out[IDX(i, col_base++)] = xi.d(v);',
                 '        }',
                 '      }',
                 '    }',
                 '  }',
                 '  SEXP coln = PROTECT(Rf_allocVector(STRSXP, ncol));',
                 '  int col = 0;',
                 '  SET_STRING_ELT(coln, col++, Rf_mkChar("time"));')
    for (s in states) externC <- c(externC, sprintf('  SET_STRING_ELT(coln, col++, Rf_mkChar("%s"));', s))
    for (s in states) {
      for (v in c(sens_states, sens_params)) {
        externC <- c(externC,
                     sprintf('  SET_STRING_ELT(coln, col++, Rf_mkChar("%s.%s"));', s, v))
      }
    }
    externC <- c(externC,
                 '  SEXP dimn = PROTECT(Rf_allocVector(VECSXP, 2));',
                 '  SET_VECTOR_ELT(dimn, 0, R_NilValue);',
                 '  SET_VECTOR_ELT(dimn, 1, coln);',
                 '  Rf_setAttrib(ans, R_DimNamesSymbol, dimn);',
                 '  UNPROTECT(3);',
                 '  return ans;')
  } else {
    # --- Output without sensitivities ---
    externC <- c(externC,
                 '  const int ncol = 1 + x_N;',
                 '  SEXP ans = PROTECT(Rf_allocMatrix(REALSXP, n_out, ncol));',
                 '  double* out = REAL(ans);',
                 '  auto IDX = [n_out](int r, int c){ return r + c * n_out; };',
                 '  for (int i = 0; i < n_out; ++i) {',
                 '    out[IDX(i, 0)] = result_times[i];',
                 '    for (int s = 0; s < x_N; ++s) out[IDX(i, 1 + s)] = y[i * x_N + s];',
                 '  }',
                 '  SEXP coln = PROTECT(Rf_allocVector(STRSXP, ncol));',
                 '  int col = 0;',
                 '  SET_STRING_ELT(coln, col++, Rf_mkChar("time"));')
    for (s in states) externC <- c(externC, sprintf('  SET_STRING_ELT(coln, col++, Rf_mkChar("%s"));', s))
    externC <- c(externC,
                 '  SEXP dimn = PROTECT(Rf_allocVector(VECSXP, 2));',
                 '  SET_VECTOR_ELT(dimn, 0, R_NilValue);',
                 '  SET_VECTOR_ELT(dimn, 1, coln);',
                 '  Rf_setAttrib(ans, R_DimNamesSymbol, dimn);',
                 '  UNPROTECT(3);',
                 '  return ans;')
  }

  # --- End try/catch ---
  externC <- c(externC,
               '  } catch (const std::exception& e) {',
               '    Rf_error("ODE solver failed: %s", e.what());',
               '  } catch (...) {',
               '    Rf_error("ODE solver failed: unknown C++ exception. Good Luck!");',
               '  }',
               '}')

  externC <- paste(externC, collapse = "\n")

  # --- Write C++ file ---
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

  # --- Attach attributes ---
  attr(modelname, "equations")     <- odes
  attr(modelname, "variables")     <- states
  attr(modelname, "parameters")    <- params
  attr(modelname, "events")        <- events
  attr(modelname, "solver")        <- "boost::odeint::rosenbrock4"
  attr(modelname, "fixed")         <- c(fixed_states, fixed_params)
  attr(modelname, "jacobian")      <- list(f.x = jac$f.x, f.time = jac$f.time)
  if (deriv) {
    deriv_colnames <- character(0)
    for (s in states) for (v in c(sens_states, sens_params))
      deriv_colnames <- c(deriv_colnames, sprintf("%s.%s", s, v))
    attr(modelname, "sensvariables") <- deriv_colnames
  } else {
    attr(modelname, "sensvariables") <- character(0)
  }

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

  cxxflags <- if (is_windows) "-std=c++23 -O2 -DNDEBUG" else "-std=c++20 -O2 -DNDEBUG -fPIC"

  include_flags <- paste0("-I", shQuote(system.file("include", package = "CppODE")))

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
    try(dyn.unload(soFile), silent = TRUE)
    dyn.load(soFile)
    invisible(soFile)
  } else {
    stop("Compiled shared library not found: ", soFile)
  }
}
