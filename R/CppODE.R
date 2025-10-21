#' Generate C++ code for ODE models with events and optional sensitivities
#'
#' This function generates and compiles a C++ solver for systems of ODEs using
#' Boost.Odeint's stiff Rosenbrock 4(3) method with dense output and error control.
#' The solver can handle fixed-time and root-triggered events, and (optionally)
#' compute first/second-order sensitivities via forward-mode automatic
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
#' The function supports three sensitivity modes:
#' \describe{
#'   \item{\code{deriv = FALSE, deriv2 = FALSE}}{No sensitivities. Uses plain doubles.
#'     Returns: \code{list(time = numeric[n_out], states = matrix[n_out, n_states])}}
#'   \item{\code{deriv = TRUE, deriv2 = FALSE}}{First-order sensitivities using \code{F<double>}.
#'     Returns: \code{list(time, states, sens1 = array[n_out, n_states, n_sens])}}
#'   \item{\code{deriv = TRUE, deriv2 = TRUE}}{Second-order sensitivities using \code{F<F<double>>}.
#'     Returns: \code{list(time, states, sens1, sens2 = array[n_out, n_states, n_sens, n_sens])}}
#' }
#'
#' Fixed states or parameters (specified in \code{fixed}) are excluded from the
#' sensitivity system. Setting \code{deriv2 = TRUE} requires \code{deriv = TRUE}.
#'
#' @param odes Named character vector of ODE right-hand sides.
#' @param events Optional \code{data.frame} describing events. Default: \code{NULL}.
#' @param fixed Character vector of fixed states or parameters. Default: \code{NULL}.
#' @param includeTimeZero Logical. Include time 0 in integration times. Default: \code{TRUE}.
#' @param compile Logical. Compile and load generated C++ code. Default: \code{TRUE}.
#' @param modelname Optional base name for C++ file. Default: random identifier.
#' @param deriv Logical. Compute first-order sensitivities. Default: \code{TRUE}.
#' @param deriv2 Logical. Compute second-order sensitivities. Default: \code{FALSE}.
#' @param verbose Logical. Print progress messages. Default: \code{FALSE}.
#'
#' @return The model name (character) with attributes: equations, variables, parameters,
#'   events, solver, fixed, jacobian, n_sens.
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @example inst/examples/example.R
#' @importFrom reticulate import source_python
#' @export
CppFun <- function(odes, events = NULL, fixed = NULL, includeTimeZero = TRUE,
                   compile = TRUE, modelname = NULL,
                   deriv = TRUE, deriv2 = FALSE, verbose = FALSE) {

  if (deriv2 && !deriv) stop("deriv2 = TRUE requires deriv = TRUE")

  odes <- unclass(odes)
  odes <- gsub("\n", "", odes)
  odes <- sanitizeExprs(odes)

  states  <- names(odes)
  symbols <- getSymbols(c(odes, if (!is.null(events)) {
    c(events$value,
      if ("time" %in% names(events)) events$time,
      if ("root" %in% names(events)) events$root)
  }))
  params  <- setdiff(symbols, c(states, "time"))

  if (is.null(fixed)) fixed <- character(0)
  fixed_states <- if (deriv) intersect(fixed, states) else character(0)
  fixed_params <- if (deriv) intersect(fixed, params) else character(0)
  sens_states  <- if (deriv) setdiff(states, fixed_states) else character(0)
  sens_params  <- if (deriv) setdiff(params, fixed_params) else character(0)

  state_idx0 <- setNames(seq_along(states) - 1L, states)
  param_idx0 <- setNames(seq_along(params) - 1L, params)
  fixed_state_idx <- state_idx0[fixed_states]
  fixed_param_idx <- param_idx0[fixed_params]

  n_states <- length(states)
  n_params <- length(params)
  n_total_sens <- length(sens_states) + length(sens_params)

  if (is.null(modelname)) {
    modelname <- paste(c("f", sample(c(letters, 0:9), 8, TRUE)), collapse = "")
  }

  ensurePythonEnv("CppODE", verbose)
  py_file <- system.file("python", "codegen.py", package = "CppODE")
  reticulate::source_python(py_file)

  if (verbose) message("Generating ODE and Jacobian code...")

  numType <- if (deriv2) "AD2" else if (deriv) "AD" else "double"

  codegen_result <- reticulate::py$generate_ode_cpp(
    odes_dict = as.list(setNames(odes, states)),
    params_list = params,
    num_type = numType,
    fixed_states = fixed_states,
    fixed_params = fixed_params
  )

  ode_code <- codegen_result$ode_code
  jac_code <- codegen_result$jac_code
  jac_matrix_str <- codegen_result$jac_matrix
  time_derivs_str <- codegen_result$time_derivs

  if (verbose) message("  ✓ ODE and Jacobian generated")

  event_code <- ""
  if (!is.null(events)) {
    if (verbose) message("Generating event code...")
    event_lines <- reticulate::py$generate_event_code(
      events_df = events,
      states_list = states,
      params_list = params,
      n_states = n_states,
      num_type = numType
    )
    event_code <- paste(event_lines, collapse = "\n")
  }

  # === C++ Code Generation ===

  includings <- c(
    "#define R_NO_REMAP",
    "#include <R.h>",
    "#include <Rinternals.h>",
    "#include <algorithm>",
    "#include <vector>",
    "#include <cmath>",
    "#include <boost_rosenbrock34_fad.hpp>"
  )

  if (deriv2) {
    usings <- c(
      "using namespace boost::numeric::odeint;",
      "namespace ublas = boost::numeric::ublas;",
      "using AD = fadbad::F<double>;",
      "using AD2 = fadbad::F<fadbad::F<double>>;"
    )
  } else if (deriv) {
    usings <- c(
      "using namespace boost::numeric::odeint;",
      "namespace ublas = boost::numeric::ublas;",
      "using AD = fadbad::F<double>;"
    )
  } else {
    usings <- c(
      "using namespace boost::numeric::odeint;",
      "namespace ublas = boost::numeric::ublas;"
    )
  }

  observer_lines <- c(
    "// Observer",
    "struct observer {",
    sprintf("  std::vector<%s>& times;", numType),
    sprintf("  std::vector<%s>& y;", numType),
    sprintf("  explicit observer(std::vector<%s>& t, std::vector<%s>& y_) : times(t), y(y_) {}", numType, numType),
    sprintf("  void operator()(const ublas::vector<%s>& x, const %s& t) {", numType, numType),
    "    times.push_back(t);",
    "    for (size_t i = 0; i < x.size(); ++i) y.push_back(x[i]);",
    "  }",
    "};"
  )
  observer_code <- paste(observer_lines, collapse = "\n")

  # === Generate Solver Function ===

  externC <- c(
    sprintf('extern "C" SEXP solve_%s(SEXP timesSEXP, SEXP paramsSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP, SEXP maxstepsSEXP, SEXP root_tolSEXP, SEXP maxrootSEXP) {', modelname),
    "try {",
    "  StepChecker checker(INTEGER(maxprogressSEXP)[0], INTEGER(maxstepsSEXP)[0]);",
    sprintf("  ublas::vector<%s> x(%d), full_params(%d);", numType, n_states, n_states + n_params),
    ""
  )

  # Initialize states and parameters
  externC <- c(externC, "  // Initialize states")
  if (deriv2) {
    externC <- c(externC,
                 sprintf("  for (int i = 0; i < %d; ++i) {", n_states),
                 "    double val = REAL(paramsSEXP)[i];",
                 "    x[i] = AD2(AD(val));")
    if (length(fixed_state_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) {",
                           paste(sprintf("i == %d", fixed_state_idx), collapse = " || ")),
                   # outer and inner seeding on x() (not on .d())
                   sprintf("      x[i].diff(i, %d);", n_states + n_params),
                   sprintf("      x[i].x().diff(i, %d);", n_states + n_params),
                   "    }")
    } else {
      externC <- c(externC,
                   sprintf("    x[i].diff(i, %d);", n_states + n_params),
                   sprintf("    x[i].x().diff(i, %d);", n_states + n_params))
    }
    externC <- c(externC, "    full_params[i] = x[i];", "  }")

  } else if (deriv) {
    externC <- c(externC,
                 sprintf("  for (int i = 0; i < %d; ++i) {", n_states),
                 "    x[i] = REAL(paramsSEXP)[i];")
    if (length(fixed_state_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) x[i].diff(i, %d);",
                           paste(sprintf("i == %d", fixed_state_idx), collapse = " || "),
                           n_states + n_params))
    } else {
      externC <- c(externC, sprintf("    x[i].diff(i, %d);", n_states + n_params))
    }
    externC <- c(externC, "    full_params[i] = x[i];", "  }")

  } else {
    externC <- c(externC,
                 sprintf("  for (int i = 0; i < %d; ++i) {", n_states),
                 "    x[i] = REAL(paramsSEXP)[i];",
                 "    full_params[i] = x[i];",
                 "  }")
  }

  # Initialize parameters
  externC <- c(externC, "", "  // Initialize parameters")
  if (deriv2) {
    externC <- c(externC,
                 sprintf("  for (int i = 0; i < %d; ++i) {", n_params),
                 sprintf("    double val = REAL(paramsSEXP)[%d + i];", n_states),
                 sprintf("    full_params[%d + i] = AD2(AD(val));", n_states))
    if (length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) {",
                           paste(sprintf("i == %d", fixed_param_idx), collapse = " || ")),
                   sprintf("      full_params[%d + i].diff(%d + i, %d);",
                           n_states, n_states, n_states + n_params),
                   sprintf("      full_params[%d + i].x().diff(%d + i, %d);",
                           n_states, n_states + i, n_states + n_params),
                   "    }")
    } else {
      externC <- c(externC,
                   sprintf("    full_params[%d + i].diff(%d + i, %d);",
                           n_states, n_states, n_states + n_params),
                   sprintf("    full_params[%d + i].x().diff(%d + i, %d);",
                           n_states, n_states, n_states + n_params))
    }
    externC <- c(externC, "  }")

  } else if (deriv) {
    externC <- c(externC,
                 sprintf("  for (int i = 0; i < %d; ++i) {", n_params),
                 sprintf("    full_params[%d + i] = REAL(paramsSEXP)[%d + i];",
                         n_states, n_states))
    if (length(fixed_param_idx) > 0) {
      externC <- c(externC,
                   sprintf("    if (!(%s)) full_params[%d + i].diff(%d + i, %d);",
                           paste(sprintf("i == %d", fixed_param_idx), collapse = " || "),
                           n_states, n_states, n_states + n_params))
    } else {
      externC <- c(externC,
                   sprintf("    full_params[%d + i].diff(%d + i, %d);",
                           n_states, n_states, n_states + n_params))
    }
    externC <- c(externC, "  }")

  } else {
    externC <- c(externC,
                 sprintf("  for (int i = 0; i < %d; ++i) {", n_params),
                 sprintf("    full_params[%d + i] = REAL(paramsSEXP)[%d + i];",
                         n_states, n_states),
                 "  }")
  }

  # Time setup
  externC <- c(externC, "", "  // Setup times",
               "  std::vector<double> times_dbl(REAL(timesSEXP), REAL(timesSEXP) + Rf_length(timesSEXP));")
  if (includeTimeZero) {
    externC <- c(externC, "  if (std::find(times_dbl.begin(), times_dbl.end(), 0.0) == times_dbl.end()) times_dbl.push_back(0.0);")
  }
  externC <- c(externC,
               "  std::sort(times_dbl.begin(), times_dbl.end());",
               "  times_dbl.erase(std::unique(times_dbl.begin(), times_dbl.end()), times_dbl.end());",
               sprintf("  std::vector<%s> times;", numType),
               "  times.reserve(times_dbl.size());",
               "  for (double tval : times_dbl) times.emplace_back(tval);",
               sprintf("  std::vector<%s> result_times, y;", numType),
               sprintf("  std::vector<FixedEvent<%s>> fixed_events;", numType),
               sprintf("  std::vector<RootEvent<ublas::vector<%s>, %s>> root_events;", numType, numType))

  if (event_code != "") externC <- c(externC, event_code)

  # Stepper
  if (deriv2) {
    externC <- c(externC, "",
                 "  auto controlledStepper = rosenbrock4_controller_ad<rosenbrock4<AD2>>(REAL(abstolSEXP)[0], REAL(reltolSEXP)[0]);",
                 "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);")
  } else if (deriv) {
    externC <- c(externC, "",
                 "  auto controlledStepper = rosenbrock4_controller_ad<rosenbrock4<AD>>(REAL(abstolSEXP)[0], REAL(reltolSEXP)[0]);",
                 "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);")
  } else {
    externC <- c(externC, "",
                 "  auto controlledStepper = rosenbrock4_controller<rosenbrock4<double>>(REAL(abstolSEXP)[0], REAL(reltolSEXP)[0]);",
                 "  auto denseStepper = rosenbrock4_dense_output<decltype(controlledStepper)>(controlledStepper);")
  }

  externC <- c(externC,
               "  ode_system sys(full_params);",
               "  jacobian jac(full_params);",
               "  observer obs(result_times, y);",
               # "  auto t_test = times.front();",
               # "  auto x_test = x;",
               sprintf("  %s dt = odeint_utils::estimate_initial_dt(sys, jac, x, times.front(), REAL(abstolSEXP)[0], REAL(reltolSEXP)[0]);", numType),
               # "  int attempts = 0;",
               # "  while (controlledStepper.try_step(std::make_pair(sys, jac), x_test, t_test, dt) == fail) {",
               # "    if (++attempts >= 10000) throw std::runtime_error(\"Unable to find initial stepsize\");",
               # "  }",
               "  integrate_times_dense(denseStepper, std::make_pair(sys, jac), x, times.begin(), times.end(), dt, obs, fixed_events, root_events, checker, REAL(root_tolSEXP)[0], INTEGER(maxrootSEXP)[0]);",
               "  const int n_out = static_cast<int>(result_times.size());",
               "  if (n_out <= 0) Rf_error(\"No output\");",
               "")

  # === Create output list + extract results ===

  n_components <- if (deriv2) 4 else if (deriv) 3 else 2
  externC <- c(
    externC,
    sprintf("  SEXP result = PROTECT(Rf_allocVector(VECSXP, %d));", n_components),
    sprintf("  SEXP names  = PROTECT(Rf_allocVector(STRSXP, %d));", n_components),
    "  int protect_count = 2;",
    "",
    "  // Time vector",
    "  SEXP time_vec = PROTECT(Rf_allocVector(REALSXP, n_out)); protect_count++;",
    "  double* time_ptr = REAL(time_vec);",
    "  for (int i = 0; i < n_out; ++i) {",
    "    time_ptr[i] = ::boost::numeric::odeint::detail::scalar_value(result_times[i]);",
    "  }",
    "  SET_VECTOR_ELT(result, 0, time_vec);",
    "  SET_STRING_ELT(names, 0, Rf_mkChar(\"time\"));",
    "",
    sprintf("  SEXP states_mat = PROTECT(Rf_allocMatrix(REALSXP, n_out, %d)); protect_count++;", n_states),
    "  double* states_ptr = REAL(states_mat);",
    "  for (int i = 0; i < n_out; ++i) {",
    sprintf("    for (int s = 0; s < %d; ++s) {", n_states),
    sprintf("      states_ptr[i + s * n_out] = ::boost::numeric::odeint::detail::scalar_value(y[i * %d + s]);", n_states),
    "    }",
    "  }",
    "  SET_VECTOR_ELT(result, 1, states_mat);",
    "  SET_STRING_ELT(names, 1, Rf_mkChar(\"states\"));"
  )

  # ===  sens1 ===
  if (deriv || deriv2) {
    externC <- c(
      externC, "",
      "  // First-order sensitivities: array [n_out, n_states, n_sens]",
      "  SEXP dims1 = PROTECT(Rf_allocVector(INTSXP, 3)); protect_count++;",
      "  INTEGER(dims1)[0] = n_out;",
      sprintf("  INTEGER(dims1)[1] = %d;", n_states),
      sprintf("  INTEGER(dims1)[2] = %d;", n_total_sens),
      "  SEXP sens1_arr = PROTECT(Rf_allocArray(REALSXP, dims1)); protect_count++;",
      "  double* sens1_ptr = REAL(sens1_arr);",
      "",
      "  // Build list of non-fixed AD indices (states+params)",
      "  std::vector<int> sens_indices;",
      sprintf("  for (int v = 0; v < %d; ++v) {", n_states + n_params)
    )

    # Filter fixed indices
    if (length(fixed_state_idx) > 0 || length(fixed_param_idx) > 0) {
      checks <- character(0)
      if (length(fixed_state_idx) > 0) {
        checks <- c(checks,
                    sprintf("    bool is_fixed = (v < %d && (%s));",
                            n_states, paste(sprintf("v == %d", fixed_state_idx), collapse = " || ")))
      }
      if (length(fixed_param_idx) > 0) {
        if (length(checks) > 0) {
          checks <- c(checks,
                      sprintf("    is_fixed = is_fixed || (v >= %d && (%s));",
                              n_states, paste(sprintf("(v-%d) == %d", n_states, fixed_param_idx), collapse = " || ")))
        } else {
          checks <- c(checks,
                      sprintf("    bool is_fixed = (v >= %d && (%s));",
                              n_states, paste(sprintf("(v-%d) == %d", n_states, fixed_param_idx), collapse = " || ")))
        }
      }
      externC <- c(externC, checks, "    if (!is_fixed) sens_indices.push_back(v);")
    } else {
      externC <- c(externC, "    sens_indices.push_back(v);")
    }

    externC <- c(
      externC,
      "  }",
      "",
      "  // Fill sens1 (R column-major): idx = i + s*n_out + v_idx*n_out*Nstates",
      sprintf("  const int Nstates = %d;", n_states),
      "  const int Nsens = (int)sens_indices.size();",
      "  for (int i = 0; i < n_out; ++i) {",
      "    for (int s = 0; s < Nstates; ++s) {",
      sprintf("      %s& xi = y[i * %d + s];", numType, n_states),
      "      for (int v_idx = 0; v_idx < Nsens; ++v_idx) {",
      "        int v = sens_indices[v_idx];",
      if (deriv2) {
        # AD2: 1. Ableitung ist xi.d(v).x()
        "        const double val = xi.d(v).x();"
      } else {
        # AD1: 1. Ableitung ist xi.d(v)
        "        const double val = xi.d(v);"
      },
      "        const int idx = i + s * n_out + v_idx * n_out * Nstates;",
      "        sens1_ptr[idx] = val;",
      "      }",
      "    }",
      "  }",
      "  SET_VECTOR_ELT(result, 2, sens1_arr);",
      "  SET_STRING_ELT(names, 2, Rf_mkChar(\"sens1\"));"
    )
  }

  # === sens2 (nur AD2) ===
  if (deriv2) {
    externC <- c(
      externC, "",
      "  // Second-order sensitivities: array [n_out, n_states, n_sens, n_sens]",
      "  SEXP dims2 = PROTECT(Rf_allocVector(INTSXP, 4)); protect_count++;",
      "  INTEGER(dims2)[0] = n_out;",
      sprintf("  INTEGER(dims2)[1] = %d;", n_states),
      "  INTEGER(dims2)[2] = Nsens;",
      "  INTEGER(dims2)[3] = Nsens;",
      "  SEXP sens2_arr = PROTECT(Rf_allocArray(REALSXP, dims2)); protect_count++;",
      "  double* sens2_ptr = REAL(sens2_arr);",
      "",
      "  // idx = i + s*n_out + v_idx*n_out*Nstates + w_idx*n_out*Nstates*Nsens",
      "  for (int i = 0; i < n_out; ++i) {",
      "    for (int s = 0; s < Nstates; ++s) {",
      sprintf("      AD2& xi = y[i * %d + s];", n_states),
      "      for (int v_idx = 0; v_idx < Nsens; ++v_idx) {",
      "        int v = sens_indices[v_idx];",
      "        for (int w_idx = 0; w_idx < Nsens; ++w_idx) {",
      "          int w = sens_indices[w_idx];",
      "          const double val = xi.d(v).d(w);",
      "          const int idx = i",
      "                       + s      * n_out",
      "                       + v_idx  * n_out * Nstates",
      "                       + w_idx  * n_out * Nstates * Nsens;",
      "          sens2_ptr[idx] = val;",
      "        }",
      "      }",
      "    }",
      "  }",
      "  SET_VECTOR_ELT(result, 3, sens2_arr);",
      "  SET_STRING_ELT(names, 3, Rf_mkChar(\"sens2\"));"
    )
  }

  externC <- c(
    externC,
    "",
    "  Rf_setAttrib(result, R_NamesSymbol, names);",
    "  UNPROTECT(protect_count);",
    "  return result;",
    "  } catch (const std::exception& e) {",
    "    Rf_error(\"ODE solver failed: %s\", e.what());",
    "  } catch (...) {",
    "    Rf_error(\"ODE solver failed: unknown exception\");",
    "  }",
    "}"
  )

  externC <- paste(externC, collapse = "\n")


  # === Write C++ file ===

  filename <- paste0(modelname, ".cpp")
  sink(filename)
  cat("/** Code auto-generated by CppODE ",
      as.character(utils::packageVersion("CppODE")), " **/\n\n", sep = "")
  cat(paste(includings, collapse = "\n")); cat("\n\n")
  cat(paste(usings, collapse = "\n")); cat("\n\n")
  cat(ode_code, "\n\n")
  cat(jac_code, "\n\n")
  cat(observer_code, "\n\n")
  cat(externC)
  sink()

  if (verbose) message("Wrote: ", normalizePath(filename))
  if (compile) compileAndLoad(modelname, verbose)

  # === Attach attributes ===

  jac_matrix_R <- matrix(unlist(jac_matrix_str), nrow = n_states, ncol = n_states, byrow = TRUE)
  dimnames(jac_matrix_R) <- list(states, states)

  attr(modelname, "equations")   <- odes
  attr(modelname, "variables")   <- states
  attr(modelname, "parameters")  <- params
  attr(modelname, "events")      <- events
  attr(modelname, "solver")      <- "boost::odeint::rosenbrock4"
  attr(modelname, "fixed")       <- c(fixed_states, fixed_params)
  attr(modelname, "jacobian")    <- list(f.x = jac_matrix_R, f.time = time_derivs_str)
  attr(modelname, "n_sens")      <- n_total_sens

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

  cxxflags <- if (is_windows) "-std=c++20 -O2 -DNDEBUG" else "-std=c++20 -O2 -DNDEBUG -fPIC"

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
