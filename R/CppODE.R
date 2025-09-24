#' Generate C++ code for ODE models with events and optional sensitivities
#'
#' This function generates and compiles a C++ solver for systems of ODEs using
#' Boost.Odeint’s stiff Rosenbrock 4(3) method with dense output and error control.
#' The solver can handle fixed-time and root-triggered events, and (optionally)
#' compute state and parameter sensitivities via forward-mode automatic
#' differentiation (AD) using FADBAD++.
#'
#' ## Events
#' Events are specified in a \code{data.frame} with columns:
#' \describe{
#'   \item{var}{Name of the affected state variable.}
#'   \item{value}{Numeric value to apply at the event.}
#'   \item{method}{How the value is applied: "replace", "add", or "multiply".}
#'   \item{time}{(optional) Time point at which the event occurs.}
#'   \item{root}{(optional) Root expression in terms of states and \code{time}.}
#' }
#' Each event must define either \code{time} or \code{root}. Events with roots
#' are triggered whenever the expression crosses zero.
#'
#' ## Sensitivities
#' If \code{deriv = TRUE}, the solver augments the system with automatic
#' differentiation and returns forward sensitivities with respect to initial
#' conditions and parameters. Fixed states or parameters (specified in
#' \code{fixed}) are excluded from the sensitivity system.
#'
#' If \code{deriv = FALSE}, the solver uses plain doubles (faster) and does not
#' compute sensitivities.
#'
#' ## Output
#' The generated solver function (accessible via \code{.Call}) returns an R
#' numeric matrix with columns:
#' \itemize{
#'   \item Time
#'   \item State variables
#'   \item (if \code{deriv = TRUE}) Sensitivities in the form \code{state.parameter}
#' }
#'
#' @param odes Named character vector of ODE right-hand sides.
#'   Names must correspond to state variables.
#' @param events Optional \code{data.frame} describing events (see Details).
#'   Default: \code{NULL} (no events).
#' @param fixed Character vector of fixed states or parameters (excluded from
#'   sensitivity system). Only relevant if \code{deriv = TRUE}.
#' @param includeTimeZero Logical. If \code{TRUE}, ensure that time \code{0} is
#'   included among integration times. Default: \code{TRUE}.
#' @param compile Logical. If \code{TRUE}, compiles and loads the generated C++ code.
#' @param modelname Optional base name for the generated C++ file. If \code{NULL},
#'   a random identifier is used.
#' @param deriv Logical. If \code{TRUE}, compute sensitivities using AD.
#'   If \code{FALSE}, use plain doubles.
#' @param verbose Logical. If \code{TRUE}, print progress messages.
#'
#' @return The model name (character). The object has attributes:
#'   \itemize{
#'     \item \code{equations}: ODE definitions
#'     \item \code{variables}: State variable names
#'     \item \code{parameters}: Parameter names
#'     \item \code{events}: Events \code{data.frame} (if any)
#'     \item \code{solver}: Solver description
#'     \item \code{fixed}: Fixed states/parameters
#'     \item \code{jacobian}: Symbolic Jacobian expressions
#'     \item \code{sensvariables}: Sensitivity variable names (if any)
#'   }
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @example inst/examples/example.R
#' @importFrom reticulate import
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

  # --- Calculate dimensions R-side ---
  n_states <- length(states)
  n_params <- length(params)
  n_sens_initials <- length(sens_states)
  n_sens_params <- length(sens_params)
  n_total_sens <- n_sens_initials + n_sens_params

  if (deriv) {
    ncol_total <- 1 + n_states + n_states * n_total_sens
  } else {
    ncol_total <- 1 + n_states
  }

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
  ensurePythonEnv()
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
  jac <- suppressWarnings(ComputeJacobianSymb(odes, states = states, params = params, AD = deriv))
  jac_lines <- attr(jac, "CppCode")

  # --- Observer with vector storage ---
  observer_lines <- c(
    "// Observer: stores trajectory values in vectors",
    "struct observer {",
    sprintf("  std::vector<%s>& times;", numType),
    sprintf("  std::vector<%s>& y;", numType),
    "",
    sprintf("  explicit observer(std::vector<%s>& t, std::vector<%s>& y_)", numType, numType),
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
    sprintf('extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP, SEXP root_tolSEXP, SEXP maxrootSEXP) {', modelname),
    "try {",
    "",
    "  StepChecker checker(INTEGER(maxprogressSEXP)[0], INTEGER(maxstepsSEXP)[0]);",
    "",
    sprintf("  ublas::vector<%s> x(%d);", numType, n_states),
    sprintf("  ublas::vector<%s> full_params(%d);", numType, n_states + n_params),
    ""
  )

  # initialization of states and parameters
  externC <- c(externC,
               "  // initialize states",
               sprintf("  for (int i = 0; i < %d; ++i) {", n_states),
               "    x[i] = REAL(paramsSEXP)[i];")

  if (deriv) {
    if (length(fixed_state_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) x[i].diff(i, %d);",
                           paste(sprintf("i == %d", fixed_state_idx), collapse = " || "),
                           n_states + n_params))
    } else {
      externC <- c(externC, sprintf("    x[i].diff(i, %d);", n_states + n_params))
    }
  }

  externC <- c(externC,
               "    full_params[i] = x[i];",
               "  }",
               "",
               "  // initialize parameters",
               sprintf("  for (int i = 0; i < %d; ++i) {", n_params),
               sprintf("    full_params[%d + i] = REAL(paramsSEXP)[%d + i];", n_states, n_states))

  if (deriv) {
    if (length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) full_params[%d + i].diff(%d + i, %d);",
                           paste(sprintf("i == %d", fixed_param_idx), collapse = " || "),
                           n_states, n_states, n_states + n_params))
    } else {
      externC <- c(externC, sprintf("    full_params[%d + i].diff(%d + i, %d);", n_states, n_states, n_states + n_params))
    }
  }

  externC <- c(externC,
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
               "  // storage for results",
               sprintf("  std::vector<%s> result_times;", numType),
               sprintf("  std::vector<%s> y;", numType),
               sprintf("  const int ncol = %d;", ncol_total),
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
    observer_init <- "  observer obs(result_times, y);"
  }

  externC <- c(externC,'\n',
               '  // --- Solver setup ---',
               '  double abstol = REAL(abstolSEXP)[0];',
               '  double reltol = REAL(reltolSEXP)[0];',
               '  double root_tol = REAL(root_tolSEXP)[0];',
               '  int maxroot = INTEGER(maxrootSEXP)[0];',
               '  ode_system sys(full_params);',
               '  jacobian jac(full_params);',
               '  observer obs(result_times, y);',
               stepper_line, '\n',
               '  // --- Determine dt ---',
               sprintf('  auto t_test = times.front();'),
               '  auto x_test = x;',
               sprintf('  %s dt0 = odeint_utils::estimate_initial_dt(sys, jac, x_test, t_test, times.back(), abstol, reltol);', numType),
               '  auto dt = dt0;',
               '  int attempts = 0;',
               '  while (controlledStepper.try_step(std::make_pair(sys, jac), x_test, t_test, dt) == fail) {',
               '    if (++attempts >= 10000) throw std::runtime_error("Unable to find valid initial stepsize after 10000 attempts");',
               '  }\n',
               '  // --- Integration ---',
               '  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, root_tol, maxroot);',
               '',
               '  // --- Copy results to R matrix ---',
               '  const int n_out = static_cast<int>(result_times.size());',
               '  if (n_out <= 0) Rf_error("Integration produced no output");',
               '  SEXP ans = PROTECT(Rf_allocMatrix(REALSXP, n_out, ncol));',
               '  double* out = REAL(ans);',
               '  auto IDX = [n_out](int r, int c){ return r + c * n_out; };'
  )

  # --- Copy back results (sensitivities optional) ---
  if (deriv) {
    externC <- c(externC,
                 '  // Copy with sensitivities',
                 '  for (int i = 0; i < n_out; ++i) {',
                 '    out[IDX(i, 0)] = result_times[i].x();',
                 sprintf('    for (int s = 0; s < %d; ++s) {', n_states),
                 sprintf('      %s& xi = y[i * %d + s];', numType, n_states),
                 '      out[IDX(i, 1 + s)] = xi.x();',
                 '    }',
                 sprintf('    int col_base = 1 + %d;', n_states),
                 sprintf('    for (int s = 0; s < %d; ++s) {', n_states),
                 sprintf('      %s& xi = y[i * %d + s];', numType, n_states),
                 sprintf('      for (int v = 0; v < %d; ++v) {', n_states + n_params))

    # Add fixed checks
    if (length(fixed_state_idx) > 0 || length(fixed_param_idx) > 0) {
      fixed_state_check <- if (length(fixed_state_idx) > 0) {
        sprintf("        bool is_fixed_init  = (v < %d) && (%s);",
                n_states,
                paste(sprintf("v == %d", fixed_state_idx), collapse = " || "))
      } else {
        sprintf("        bool is_fixed_init = false;", n_states)
      }

      fixed_param_check <- if (length(fixed_param_idx) > 0) {
        sprintf("        bool is_fixed_param = (v >= %d) && (%s);",
                n_states,
                paste(sprintf("(v - %d) == %d", n_states, fixed_param_idx), collapse = " || "))
      } else {
        "        bool is_fixed_param = false;"
      }

      externC <- c(externC,
                   fixed_state_check,
                   fixed_param_check,
                   '        if (!(is_fixed_init || is_fixed_param)) {',
                   '          out[IDX(i, col_base++)] = xi.d(v);',
                   '        }')
    } else {
      externC <- c(externC,
                   '        out[IDX(i, col_base++)] = xi.d(v);')
    }

    externC <- c(externC,
                 '      }',
                 '    }',
                 '  }')
  } else {
    externC <- c(externC,
                 '  // Copy without sensitivities',
                 '  for (int i = 0; i < n_out; ++i) {',
                 '    out[IDX(i, 0)] = result_times[i];',
                 sprintf('    for (int s = 0; s < %d; ++s) out[IDX(i, 1 + s)] = y[i * %d + s];', n_states, n_states),
                 '  }')
  }

  externC <- c(externC,
               '',
               '  UNPROTECT(1);',
               '  return ans;'
  )

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
