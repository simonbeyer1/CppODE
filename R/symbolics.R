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


#' Symbolic Jacobian and Hessian computation using Python's SymPy
#'
#' Computes symbolic first- and second-order derivatives of a system of
#' algebraic expressions using Python's **SymPy** library via the
#' \pkg{reticulate} interface. The Jacobian is returned as a matrix and,
#' optionally, the Hessian as a three-dimensional array.
#'
#' @param f Named character vector of algebraic expressions. Each name
#'   corresponds to the dependent variable \eqn{f_i}, and the element content
#'   defines the right-hand side expression in terms of other variables.
#'   Both \code{^} and \code{**} are supported for exponentiation.
#' @param variables Character vector of variables with respect to which the
#'   derivatives are computed. If \code{NULL} (default), all symbols found
#'   in \code{f} are used as variables.
#' @param deriv2 Logical; if \code{TRUE}, second derivatives are also
#'   computed and returned as a 3D array (the Hessian tensor). Default is
#'   \code{FALSE}.
#' @param verbose Logical; if \code{TRUE}, print diagnostic output during
#'   Python environment setup and SymPy execution. Default is \code{FALSE}.
#'
#' @return
#' If \code{deriv2 = FALSE}, returns a character matrix of first derivatives
#' (the Jacobian) with dimensions \code{[length(f) × length(variables)]}.
#' Row names correspond to the names of \code{f}, and column names to
#' \code{variables}.
#'
#' If \code{deriv2 = TRUE}, returns a list with two components:
#' \describe{
#'   \item{jacobian}{Character matrix \code{[n_functions × n_variables]}
#'     containing \eqn{\partial f_i / \partial v_j}.}
#'   \item{hessian}{Character 3D array \code{[n_functions × n_variables × n_variables]}
#'     containing \eqn{\partial^2 f_i / (\partial v_j \partial v_k)}.}
#' }
#'
#' @details
#' The function automatically ensures that a Python environment with
#' **SymPy** is available (via \code{\link{ensurePythonEnv}}). Symbolic
#' computation is delegated to a Python backend
#' (\code{inst/python/derivSymb.py}) that implements symbolic differentiation
#' using SymPy.
#'
#' Non-analytic expressions such as \code{abs()}, \code{min()}, \code{max()},
#' and \code{sign()} are supported; their derivatives are expressed using
#' symbolic constructs such as \code{Heaviside()} or \code{sign()}.
#'
#' Reserved Python keywords (e.g., \code{if}, \code{for}, \code{lambda}) in
#' expressions are automatically sanitized by appending an underscore.
#'
#' If \code{variables} is an empty vector or \code{f} contains no symbols,
#' an empty matrix or list is returned.
#'
#' @examples
#' \dontrun{
#' # Simple example with named expressions
#' f <- c(k1 = "exp(K1)", k2 = "exp(K2)")
#' derivSymb(f)
#' #>      K1          K2
#' #> k1 "exp(K1)"   "0"
#' #> k2 "0"         "exp(K2)"
#'
#' # Using both ^ and ** for exponentiation
#' f <- c(y = "a*x^2 + b*x**3")
#' derivSymb(f, variables = c("x", "a", "b"))
#' #>   x              a       b
#' #> y "2*a*x+3*b*x**2" "x**2" "x**3"
#'
#' # Computing Jacobian and Hessian
#' result <- derivSymb(f, deriv2 = TRUE)
#' result$jacobian
#' result$hessian
#'
#' # Automatic variable detection
#' f <- c(z = "alpha + beta*x")
#' derivSymb(f)  # uses variables = c("alpha", "beta", "x")
#' }
#'
#' @seealso
#' \code{\link{ensurePythonEnv}} for Python environment setup,
#' \code{\link{getSymbols}} for symbol extraction,
#' \code{\link{sanitizeExprs}} for expression sanitization.
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @importFrom reticulate import_from_path
#' @export
derivSymb <- function(f, variables = NULL, deriv2 = FALSE, verbose = FALSE) {
  # ensure Python/SymPy
  ensurePythonEnv(envname = "CppODE", verbose = verbose)

  # load backend
  sympy_tools <- reticulate::import_from_path(
    "derivSymb",
    path = system.file("python", package = "CppODE")
  )

  f <- sanitizeExprs(f)
  if (is.null(variables)) variables <- getSymbols(f)
  if (length(variables) == 0) {
    if (!deriv2) return(matrix(nrow = length(f), ncol = 0))
    return(list(
      jacobian = matrix(nrow = length(f), ncol = 0),
      hessian = array(dim = c(length(f), 0, 0))
    ))
  }

  fnames <- names(f)
  if (is.null(fnames)) fnames <- paste0("f", seq_along(f))

  # Convert named vector to list (dict in Python)
  f_list <- as.list(f)
  names(f_list) <- fnames

  result <- sympy_tools$jac_hess_symb(f_list, variables, deriv2)

  n_i <- length(f)
  n_j <- length(variables)

  # --- Jacobian as matrix [inner × outer] ---
  Jmat <- matrix(result$jacobian, nrow = n_i, ncol = n_j, byrow = TRUE,
                 dimnames = list(fnames, variables))

  if (!deriv2)
    return(Jmat)

  # --- Hessian as 3D array [inner × outer × outer] ---
  H_array <- array(dim = c(n_i, n_j, n_j),
                   dimnames = list(fnames, variables, variables))

  for (i in seq_len(n_i)) {
    H_array[i, , ] <- matrix(unlist(result$hessian[[i]]),
                             nrow = n_j, ncol = n_j, byrow = TRUE)
  }

  list(jacobian = Jmat, hessian = H_array)
}
