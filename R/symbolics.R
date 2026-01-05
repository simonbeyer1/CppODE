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


#' Symbolic differentiation (Jacobian and optional Hessian) via SymPy
#'
#' Computes symbolic first- and second-order derivatives of a system of
#' algebraic expressions using Python's **SymPy** library via the
#' \pkg{reticulate} interface.
#'
#' The Jacobian (first derivatives) is returned as a character matrix,
#' and, if requested, the Hessian (second derivatives) as a list of
#' 3D arrays — one per expression.
#'
#' The Python backend automatically infers all variables occurring in
#' the expressions using SymPy's internal symbol detection.
#'
#' @param exprs Named character vector of algebraic expressions. Each name
#'   corresponds to a dependent variable \eqn{f_i}, and the element content
#'   defines the right-hand side expression in terms of other variables.
#'   Both \code{^} and \code{**} are supported for exponentiation.
#' @param real Logical; if \code{TRUE}, imaginary parts of symbolic expressions
#'   are set to zero post hoc, and real parts are replaced by their argument.
#'   This ensures real-valued simplifications even for non-analytic functions
#'   such as \code{abs()}, \code{max()}, \code{min()}, or \code{sign()}.
#'   Default is \code{FALSE}.
#' @param deriv2 Logical; if \code{TRUE}, second derivatives (Hessians)
#'   are computed and returned. Default is \code{FALSE}.
#' @param fixed Character vector of variable names that should be treated
#'   as *fixed parameters* (no derivatives are computed with respect to them).
#'   Default: `NULL`.
#' @param verbose Logical; if \code{TRUE}, print diagnostic information
#'   during backend setup and execution. Default is \code{FALSE}.
#'
#' @return
#' A list with components:
#' \describe{
#'   \item{jacobian}{Character matrix \code{[n_functions × n_variables]}
#'     containing first derivatives.}
#'   \item{hessian}{List of character arrays \code{[n_variables × n_variables]},
#'     one per function, or \code{NULL} if \code{deriv2 = FALSE}.}
#' }
#'
#' @details
#' This function calls the Python module \code{derivSymb.py} (shipped with the
#' \pkg{CppODE} package) using \pkg{reticulate}. The Python side performs
#' symbolic differentiation using SymPy. If \code{variables = NULL},
#' all free symbols in the expressions are automatically detected.
#'
#' Setting \code{real = TRUE} simplifies all results under the assumption
#' that all variables are real, without invoking SymPy's \code{refine()}
#' (which can cause recursion issues for non-analytic functions).
#'
#' @examples
#'
#' eqs <- c(
#'   f1 = "a*x^2 + b*y^2",
#'   f2 = "x*y + exp(2*c) + abs(max(x,y))"
#' )
#'
#' # Compute Jacobian only
#' result <- derivSymb(eqs, real = TRUE)
#' result$jacobian
#'
#' # Compute Jacobian and Hessian
#' result2 <- derivSymb(eqs, real = TRUE, deriv2 = TRUE)
#' result2$hessian[[1]]  # Hessian of f1
#' result2$hessian[[2]]  # Hessian of f2
#'
#' @export
derivSymb <- function(exprs, real = FALSE, deriv2 = FALSE, fixed = NULL, verbose = FALSE) {
  # --- ensure environment ---
  ensurePythonEnv(envname = "CppODE", verbose = verbose)

  # --- import Python backend ---
  sympy_tools <- reticulate::import_from_path(
    "derivSymb",
    path = system.file("python", package = "CppODE")
  )

  # --- prepare input ---
  if (is.null(names(exprs))) {
    names(exprs) <- paste0("f", seq_along(exprs))
  }
  expr_dict <- as.list(exprs)

  # --- call Python backend ---
  result <- sympy_tools$jac_hess_symb(
    exprs = expr_dict,
    variables = NULL,
    fixed = fixed,
    deriv2 = deriv2,
    real = real
  )

  # --- format Jacobian ---
  J <- do.call(rbind, result$jacobian)
  colnames(J) <- result$vars
  rownames(J) <- result$names

  # --- format Hessian (if present) ---
  H <- NULL
  if (!is.null(result$hessian)) {
    varnames <- result$vars
    funcnames <- result$names

    H <- lapply(seq_along(result$hessian), function(i) {
      H_i <- result$hessian[[i]]
      matrix(
        unlist(H_i),
        nrow = length(varnames),
        ncol = length(varnames),
        byrow = TRUE,
        dimnames = list(varnames, varnames)
      )
    })
    names(H) <- funcnames
  }

  # --- return ---
  list(jacobian = J, hessian = H)
}
