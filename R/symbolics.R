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
