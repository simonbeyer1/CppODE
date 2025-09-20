#' Compute the Jacobian of a system of ODEs and generate C++ code
#'
#' This function takes a system of ordinary differential equations (ODEs)
#' defined as character strings and:
#' \enumerate{
#'   \item Computes the Jacobian matrix with respect to the state variables.
#'   \item Computes the explicit time derivatives of each ODE.
#'   \item Generates a C++ struct suitable for stiff solvers in Boost.Odeint.
#' }
#'
#' The mapping to C++ is:
#' \itemize{
#'   \item State variables \code{x1,...,xn} → \code{x[0],...,x[n-1]}.
#'   \item Initial values \code{x1_0,...,xn_0} → \code{params[0],...,params[n-1]}.
#'   \item Dynamical parameters \code{p1,...,pm} → \code{params[n],...,params[n+m-1]}.
#'   \item The time variable \code{time} → \code{t}.
#' }
#'
#' @param odes Named character vector of ODE right-hand sides.
#'   Names correspond to state variables.
#' @param states Character vector of state variable names.
#'   If \code{NULL}, taken from \code{names(odes)}.
#' @param params Character vector of parameter names (excluding state and "time").
#'   If \code{NULL}, inferred automatically.
#' @param AD Logical. If \code{TRUE}, numerical type is set to \code{AD},
#'   otherwise to \code{double}.
#'
#' @return A list with:
#' \describe{
#'   \item{f.x}{Jacobian matrix entries as strings (R representation).}
#'   \item{f.time}{Explicit derivatives with respect to time as strings.}
#' }
#' The returned list has an attribute \code{CppCode} containing the full C++ struct.
#'
#' @examples
#' odes <- c(x = "v", v = "mu*(1 - x^2)*v - x")
#' res <- ComputeJacobianSymb(odes)
#' res$f.x
#' res$f.time
#' cat(attr(res, "CppCode"))
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @importFrom reticulate import
#' @export
ComputeJacobianSymb <- function(odes, states = NULL, params = NULL, AD = TRUE) {

  # --- Select numeric type for generated C++ code ---
  if (AD) numType <- "AD" else numType <- "double"

  # --- Import SymPy via reticulate ---
  sympy  <- reticulate::import("sympy")
  parser <- reticulate::import("sympy.parsing.sympy_parser")

  # --- State and parameter setup ---
  if (is.null(states)) states <- names(odes)
  n <- length(states)
  syms_states <- lapply(states, function(s) sympy$Symbol(s, real = TRUE))
  names(syms_states) <- states

  if (is.null(params)) {
    params <- setdiff(getSymbols(odes), c(states, "time"))
  }
  m <- length(params)
  syms_params <- lapply(params, function(p) sympy$Symbol(p, real = TRUE))
  names(syms_params) <- params

  t <- sympy$Symbol("time", real = TRUE)

  local_dict <- c(syms_states, syms_params)
  local_dict[["time"]] <- t

  transformations <- reticulate::tuple(
    c(parser$standard_transformations,
      list(parser$convert_xor, parser$implicit_multiplication_application))
  )

  # --- Parse ODE RHS into SymPy expressions ---
  exprs <- lapply(odes, function(eq)
    parser$parse_expr(eq,
                      local_dict      = local_dict,
                      transformations = transformations,
                      evaluate        = TRUE)
  )

  # --- Collect Jacobian entries (as R strings) ---
  f.x <- matrix("", nrow = n, ncol = n, dimnames = list(states, states))
  f.time <- character(n)
  for (i in seq_along(states)) {
    for (j in seq_along(states)) {
      f.x[i, j] <- as.character(sympy$diff(exprs[[i]], syms_states[[j]]))
    }
    f.time[i] <- as.character(sympy$diff(exprs[[i]], t))
  }

  # --- Generate C++ Jacobian struct ---
  cpp_lines <- c(
    "// Jacobian for stiff solver",
    "struct jacobian {",
    sprintf("  ublas::vector<%s> params;", numType),
    sprintf("  explicit jacobian(const ublas::vector<%s>& p_) : params(p_) {}", numType),
    sprintf("  void operator()(const ublas::vector<%s>& x, ublas::matrix<%s>& J, const %s& t, ublas::vector<%s>& dfdt) {",
            numType, numType, numType, numType)
  )

  # --- Fill Jacobian matrix J(i,j) ---
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      code <- Sympy2CppCode(sympy$diff(exprs[[i]], syms_states[[j]]),
                            states, params, n, expr_name = states[i], AD)
      cpp_lines <- c(cpp_lines, sprintf("    J(%d,%d) = %s;", i-1, j-1, code))
    }
  }

  # --- Fill time derivatives dfdt[i] ---
  for (i in seq_len(n)) {
    code <- Sympy2CppCode(sympy$diff(exprs[[i]], t),
                          states, params, n, expr_name = states[i], AD)
    cpp_lines <- c(cpp_lines, sprintf("    dfdt[%d] = %s;", i-1, code))
  }

  cpp_lines <- c(cpp_lines, "  }", "};")
  cpp_code <- paste(cpp_lines, collapse = "\n")

  # --- Return R-level results ---
  out <- list(f.x = f.x, f.time = f.time)
  attr(out, "CppCode") <- cpp_code
  return(out)
}



#' Convert a SymPy expression to C++ code
#'
#' This helper converts a SymPy expression into valid C++ code.
#' Symbol names are systematically replaced:
#' \itemize{
#'   \item State variables \code{x1,...,xn} → \code{x[0],...,x[n-1]}.
#'   \item Initial values \code{x1_0,...,xn_0} → \code{params[0],...,params[n-1]}.
#'   \item Parameters \code{p1,...,pm} → \code{params[n],...,params[n+m-1]}.
#'   \item Time variable \code{time} → \code{t}.
#' }
#'
#' No additional rewrites (e.g. for non-smooth functions or ternary operators)
#' are performed – the SymPy C++ output is used directly.
#'
#' @param expr SymPy expression (reticulate/sympy object).
#' @param states Character vector of state variable names.
#' @param params Character vector of parameter names.
#' @param n Number of states.
#' @param expr_name Optional: name of the ODE (used only in error messages).
#' @param AD Logical. If \code{TRUE}, the numeric type is \code{AD}, otherwise \code{double}.
#'
#' @return A character string with valid C++ code.
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @importFrom reticulate import
#' @keywords internal
Sympy2CppCode <- function(expr, states, params, n, expr_name = NULL, AD = TRUE) {
  sympy <- reticulate::import("sympy")
  code <- sympy$cxxcode(expr, standard = "c++17", strict = FALSE)

  if (grepl("NotsupportedinC\\+\\+", gsub("\\s+", "", code))) {
    stop(paste0(
      "Cannot generate C++ code for rhs of",
      if (!is.null(expr_name)) paste0(" '", expr_name, "'"),
      ": contains terms not supported in C++ (see SymPy cxxcode output)."
    ))
  }

  for (i in seq_along(states)) {
    code <- gsub(paste0("\\b", states[i], "_0\\b"), sprintf("params[%d]", i - 1), code, perl = TRUE)
    code <- gsub(paste0("\\b", states[i], "\\b"),   sprintf("x[%d]",      i - 1), code, perl = TRUE)
  }
  for (i in seq_along(params)) {
    code <- gsub(paste0("\\b", params[i], "\\b"), sprintf("params[%d]", n + i - 1), code, perl = TRUE)
  }
  code <- gsub("\\btime\\b", "t", code, perl = TRUE)

  code <- gsub("\\s+", "", code)
  return(code)
}


#' Clean up generated C++ code
#'
#' Performs light post-processing on generated C++ code:
#' \itemize{
#'   \item Removes redundant parentheses around simple values.
#'   \item Normalizes whitespace.
#'   \item Adds minimal spacing around arithmetic operators.
#' }
#'
#' This step is optional and purely cosmetic.
#'
#' @param code Character string with raw C++ code.
#' @return Character string with cleaned-up code.
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @keywords internal
cleanup_generated_code <- function(code) {
  code <- gsub("\\(([0-9\\.]+)\\)", "\\1", code, perl = TRUE)
  code <- gsub("\\(([AD()0-9\\.]+)\\)", "\\1", code, perl = TRUE)
  code <- gsub("\\s+", "", code)
  code <- gsub("([+\\-*/])", " \\1 ", code)
  code <- gsub("\\s+", " ", code)
  code <- gsub("^ | $", "", code)
  return(code)
}

#' Extract symbols from expressions
#'
#' Parses a character vector of expressions and returns all symbol names
#' (variable identifiers).
#'
#' @param char Character vector of expressions.
#' @param exclude Optional character vector of names to exclude.
#'
#' @return Character vector with unique symbol names.
#' @examples
#' getSymbols("a*b + c")
#' getSymbols(c("x + y", "z"), exclude = "y")
#'
#' @author Daniel Kaschek
#' @keywords internal
getSymbols <- function(char, exclude = NULL) {
  if (is.null(char)) return(NULL)
  char <- char[char!="0"]
  out <- parse(text=char, keep.source = TRUE)
  out <- utils::getParseData(out)
  names <- unique(out$text[out$token == "SYMBOL"])
  if(!is.null(exclude)) names <- names[!names%in%exclude]
  return(names)
}


#' Sanitize symbol names for SymPy compatibility
#'
#' Reserved identifiers (Python keywords and some built-ins) cannot be used
#' directly as variable names. This function replaces such names with the same
#' name plus an underscore appended (e.g. \code{while} → \code{while_}).
#'
#' A warning is emitted for each renamed symbol.
#'
#' @param symbols Character vector of symbol names.
#' @return Character vector with sanitized names.
#' @examples
#' sanitizeSymbols(c("for", "x", "while"))
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @keywords internal
sanitizeSymbols <- function(symbols) {
  reserved <- c(
    "False","None","True","and","as","assert","async","await","break","class",
    "continue","def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","nonlocal","not","or","pass","raise","return",
    "try","while","with","yield",
    "list","dict","set","int","float"
  )

  out <- symbols
  for (i in seq_along(symbols)) {
    if (symbols[i] %in% reserved) {
      newname <- paste0(symbols[i], "_")
      warning(sprintf("Symbol '%s' is reserved – renamed to '%s'.",
                      symbols[i], newname))
      out[i] <- newname
    }
  }
  out
}


#' Sanitize expressions for SymPy compatibility
#'
#' Scans character expressions for reserved Python keywords and replaces them
#' with the same name plus an underscore appended.
#'
#' Function names like \code{min}, \code{max}, \code{abs}, and \code{sum} are
#' left untouched, since they are valid in SymPy.
#'
#' A warning is emitted for each replacement performed.
#'
#' @param exprs Character vector of expressions.
#' @return Character vector with sanitized expressions.
#' @examples
#' sanitizeExprs("if + 1")   # becomes "if_ + 1"
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @keywords internal
sanitizeExprs <- function(exprs) {
  sanitized <- exprs
  reserved_keywords <- c(
    "False","None","True","and","as","assert","async","await","break","class",
    "continue","def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","nonlocal","not","or","pass","raise","return",
    "try","while","with","yield"
  )

  for (sym in reserved_keywords) {
    pattern <- paste0("\\b", sym, "\\b")
    repl    <- paste0(sym, "_")
    if (any(grepl(pattern, sanitized))) {
      warning(sprintf("Reserved keyword '%s' found in expression – replaced by '%s'.",
                      sym, repl))
      sanitized <- gsub(pattern, repl, sanitized)
    }
  }
  sanitized
}
