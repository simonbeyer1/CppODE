#' Extract Symbol Names from R Expressions
#'
#' Returns the unique names of all symbols occurring in a character
#' vector of R expressions.
#'
#' @param expr Character vector of R expressions.
#' @param omit Optional character vector of symbol names to remove.
#'
#' @return Character vector of unique symbol names.
#'
#' @keywords internal
getSymbols <- function(expr, omit = NULL) {
  if (is.null(expr)) return(character(0))

  expr <- expr[expr != "0"]
  if (!length(expr)) return(character(0))

  parsed <- tryCatch(parse(text = expr, keep.source = TRUE), error = function(e) NULL)
  if (is.null(parsed)) return(character(0))

  pd <- utils::getParseData(parsed)
  syms <- unique(pd[pd$token == "SYMBOL", "text"])

  if (!is.null(omit)) syms <- setdiff(syms, omit)
  syms
}
#' Sanitize Expressions for SymPy Compatibility
#'
#' Scans character expressions for reserved Python keywords and
#' replaces each occurrence with the keyword followed by an underscore.
#' Function names like \code{min}, \code{max}, \code{abs}, and
#' \code{sum} are left untouched because they are valid in SymPy. A
#' warning is emitted for each replacement.
#'
#' @param exprs Character vector of expressions.
#' @return Character vector with sanitized expressions.
#'
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
      warning(sprintf("Reserved keyword '%s' found in expression - replaced by '%s'.",
                      sym, repl))
      sanitized <- gsub(pattern, repl, sanitized)
    }
  }
  sanitized
}
#' Symbolic Differentiation via SymPy
#'
#' @description
#' Computes symbolic first- and (optionally) second-order derivatives
#' of a system of algebraic expressions using the SymPy library via the
#' \pkg{reticulate} interface. The Jacobian is returned as a character
#' matrix of shape \eqn{(n_f, n_v)}; the Hessian (when requested) as a
#' list of character matrices of shape \eqn{(n_v, n_v)}.
#'
#' @details
#' This function calls a Python module shipped with the package
#' (\code{derivSymb.py}) through \pkg{reticulate}. All free symbols in
#' the expressions are detected automatically by the Python backend.
#'
#' Setting \code{real = TRUE} simplifies all results under the
#' assumption that variables are real, without invoking SymPy's
#' \code{refine()} (which can recurse on non-analytic functions).
#'
#' @param exprs Named character vector of algebraic expressions. Each
#'   name corresponds to a dependent variable \eqn{f_i}; the element
#'   content defines the right-hand side expression. Both \code{^} and
#'   \code{**} are accepted for exponentiation.
#' @param real Logical; if \code{TRUE}, imaginary parts of symbolic
#'   expressions are set to zero and real parts are replaced by their
#'   argument. This ensures real-valued simplifications even for
#'   non-analytic functions such as \code{abs()}, \code{max()},
#'   \code{min()}, or \code{sign()}. Default \code{FALSE}.
#' @param deriv2 Logical; if \code{TRUE}, also compute second
#'   derivatives (Hessians). Default \code{FALSE}.
#' @param fixed Character vector of variable names to be treated as
#'   fixed parameters (no derivatives are taken with respect to them).
#'   Default \code{NULL}.
#' @param verbose Logical; if \code{TRUE}, print diagnostic information
#'   during backend setup and execution. Default \code{FALSE}.
#'
#' @return
#' A list with components:
#' \describe{
#'   \item{\code{jacobian}}{Character matrix of shape \eqn{(n_f, n_v)}
#'     containing the first derivatives
#'     \eqn{\partial f_i / \partial v_j}, where \eqn{n_f} is the number
#'     of functions and \eqn{n_v} the number of differentiation
#'     variables (excluding fixed parameters).}
#'   \item{\code{hessian}}{List of \eqn{n_f} character matrices, each of
#'     shape \eqn{(n_v, n_v)}, containing the second derivatives
#'     \eqn{\partial^2 f_i / \partial v_j \partial v_k}. Returns
#'     \code{NULL} when \code{deriv2 = FALSE}.}
#' }
#'
#' @examples
#' \dontrun{
#' eqs <- c(
#'   f1 = "a*x^2 + b*y^2",
#'   f2 = "x*y + exp(2*c) + abs(max(x, y))"
#' )
#'
#' # Jacobian only
#' result <- derivSymb(eqs, real = TRUE)
#' result$jacobian
#'
#' # Jacobian and Hessian
#' result2 <- derivSymb(eqs, real = TRUE, deriv2 = TRUE)
#' result2$hessian[[1]]
#' result2$hessian[[2]]
#' }
#'
#' @export
derivSymb <- function(exprs, real = FALSE, deriv2 = FALSE, fixed = NULL, verbose = FALSE) {

  # Lazy import
  py_deriv <- get_derivSymb_py()

  # --- prepare input ---
  if (is.null(names(exprs))) {
    names(exprs) <- paste0("f", seq_along(exprs))
  }
  expr_dict <- as.list(exprs)

  # --- call Python backend ---
  result <- py_deriv$jac_hess_symb(
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
