#' Compute the Jacobian of a system of ODEs and generate C++ code
#'
#' This function takes a system of ordinary differential equations (ODEs) defined
#' as character strings and:
#' \enumerate{
#'   \item Computes the Jacobian matrix with respect to the state variables.
#'   \item Computes the explicit time derivatives of each ODE.
#'   \item Generates a C++ struct suitable for stiff solvers in Boost.Odeint,
#'         using \code{CppAD::AD} for automatic differentiation.
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
#' @param numType Character describing the numerical type
#'
#' @return A list with:
#' \describe{
#'   \item{f.x}{Jacobian matrix entries as strings (R representation).}
#'   \item{f.time}{Explicit derivatives with respect to time as strings.}
#' }
#' The returned list has an attribute \code{CppCode} containing the full C++ struct.
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @import reticulate
#' @export
#' @examples
#' odes <- c(x = "v", v = "mu*(1 - x^2)*v - x")
#' res <- ComputeJacobianSymb(odes)
#' res$f.x
#' res$f.time
#' cat(attr(res, "CppCode"))
ComputeJacobianSymb <- function(odes, states = NULL, params = NULL, AD = TRUE) {

  if (AD) numType <- "AD" else numType <- "double"
  # Import SymPy + Parser via reticulate
  sympy   <- reticulate::import("sympy")
  parser  <- reticulate::import("sympy.parsing.sympy_parser")

  # --- Define state variables ---
  if (is.null(states)) states <- names(odes)
  n <- length(states)
  syms_states <- lapply(states, function(s) sympy$Symbol(s, real = TRUE))
  names(syms_states) <- states

  # --- Identify parameters ---
  if (is.null(params)) {
    params <- setdiff(getSymbols(odes), c(states, "time"))
  }
  m <- length(params)
  syms_params <- lapply(params, function(p) sympy$Symbol(p, real = TRUE))
  names(syms_params) <- params

  # --- Define time symbol ---
  t <- sympy$Symbol("time", real = TRUE)

  # --- Build local dict for the parser ---
  local_dict <- c(syms_states, syms_params)
  local_dict[["time"]] <- t

  # --- Parser transformations (for ^ and implicit mult) ---
  transformations <- reticulate::tuple(
    c(parser$standard_transformations,
      list(parser$convert_xor, parser$implicit_multiplication_application))
  )

  # --- Parse ODEs into SymPy expressions ---
  exprs <- lapply(odes, function(eq)
    parser$parse_expr(eq,
                      local_dict       = local_dict,
                      transformations  = transformations,
                      evaluate         = TRUE)
  )

  # --- Compute R matrices for Jacobian and explicit time derivatives ---
  f.x <- matrix("", nrow = n, ncol = n, dimnames = list(states, states))
  f.time <- character(n)
  for (i in seq_along(states)) {
    for (j in seq_along(states)) {
      f.x[i, j] <- as.character(sympy$diff(exprs[[i]], syms_states[[j]]))
    }
    f.time[i] <- as.character(sympy$diff(exprs[[i]], t))
  }

  # --- Generate C++ code ---
  cpp_lines <- c(
    "// Jacobian for stiff solver",
    "struct jacobian {",
    sprintf("  vector<%s> params;", numType),
    sprintf("  explicit jacobian(const vector<%s>& p_) : params(p_) {}", numType),
    sprintf("  void operator()(const vector<%s>& x, matrix<%s>& J, const %s& t, vector<%s>& dfdt) {", numType, numType, numType, numType)
  )

  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      code <- Sympy2CppCode(sympy$diff(exprs[[i]], syms_states[[j]]), states, params, n, expr_name = states[i], AD)
      cpp_lines <- c(cpp_lines, sprintf("    J(%d,%d)=%s;", i-1, j-1, code))
    }
  }

  for (i in seq_len(n)) {
    code <- Sympy2CppCode(sympy$diff(exprs[[i]], t), states, params, n, expr_name = states[i], AD)
    cpp_lines <- c(cpp_lines, sprintf("    dfdt[%d]=%s;", i-1, code))
  }

  cpp_lines <- c(cpp_lines, "  }", "};")
  cpp_lines <- if (numType == "AD") gsub("std::", "CppAD::", cpp_lines, fixed = TRUE) else cpp_lines
  cpp_code <- paste(cpp_lines, collapse = "\n")

  out <- list(f.x = f.x, f.time = f.time)
  attr(out, "CppCode") <- cpp_code
  return(out)
}

#' Sympy2CppCode: Convert SymPy expression to C++ code (CppAD-compatible if needed)
#'
#' This internal helper converts a SymPy expression into valid C++ code.
#' Symbol names are replaced systematically, and—if `AD = TRUE`—non-smooth
#' functions (`abs`, `max`, `min`, `Heaviside`, `sign`/`sgn`) and any
#' ternary operators are rewritten into `CppAD::CondExp*` expressions so the
#' code is compatible with CppAD (automatic differentiation).
#'
#' @param expr SymPy expression (reticulate/sympy object).
#' @param states Character vector of state names (e.g. `c("s1","s2",...)`).
#' @param params Character vector of parameter names (e.g. `c("p1","p2",...)`).
#' @param n Number of states.
#' @param expr_name Optional: name of the ODE (used only in error messages).
#' @param AD Logical. If `TRUE`, output is made CppAD-compatible.
#'
#' @return A character string with valid C++ code.
#' @import reticulate
#' @keywords internal
#' @author Simon Beyer (revised)
Sympy2CppCode <- function(expr, states, params, n, expr_name = NULL, AD = TRUE) {
  sympy <- reticulate::import("sympy")

  # 1) Generate C++ from SymPy as-is
  code <- sympy$cxxcode(expr, standard = "c++17", strict = FALSE)

  # 2) Fail early if SymPy says it emitted unsupported constructs
  if (grepl("NotsupportedinC\\+\\+", gsub("\\s+", "", code))) {
    stop(paste0(
      "Cannot generate C++ code for rhs of",
      if (!is.null(expr_name)) paste0(" '", expr_name, "'"),
      ": contains terms not supported in C++ (see SymPy cxxcode output)."
    ))
  }

  # 3) Replace variable names
  for (i in seq_along(states)) {
    code <- gsub(paste0("\\b", states[i], "_0\\b"), sprintf("params[%d]", i - 1), code, perl = TRUE)
    code <- gsub(paste0("\\b", states[i], "\\b"),   sprintf("x[%d]",      i - 1), code, perl = TRUE)
  }
  for (i in seq_along(params)) {
    code <- gsub(paste0("\\b", params[i], "\\b"), sprintf("params[%d]", n + i - 1), code, perl = TRUE)
  }
  code <- gsub("\\btime\\b", "t", code, perl = TRUE)

  # 4) AD-specific transformations
  if (AD) {
    code <- replace_abs_max_min_for_AD(code)
    code <- replace_sign_heaviside(code, TRUE)

    # Convert ternary operators - simplified robust version
    code <- convert_ternaries_simple(code)

    # Final check and warning if ternaries remain
    if (grepl("\\?", code, perl = TRUE)) {
      warning("Some ternary operators (?:) could not be converted to CppAD::CondExp*. Expression: ", code)
    }
  } else {
    code <- replace_sign_heaviside(code, FALSE)
  }

  # 5) Normalize whitespace
  code <- gsub("\\s+", "", code)
  return(code)
}

#' Convert ternary operators to CppAD::CondExp* - Simplified robust version
#'
#' This function uses a straightforward iterative approach to convert ternary
#' operators. It processes them from innermost to outermost.
#'
#' @param txt Character string containing C++ code with ternary operators
#' @return Character string with ternaries converted to CppAD::CondExp*
convert_ternaries_simple <- function(txt) {
  # Map operators to CppAD suffix
  ops_map <- list(
    ">=" = "Ge", "<=" = "Le", "==" = "Eq",
    "!=" = "Ne", ">" = "Gt", "<" = "Lt"
  )

  # Remove spaces to make parsing easier
  txt <- gsub("\\s+", "", txt)

  max_iter <- 20
  iter <- 0

  while (grepl("\\?", txt) && iter < max_iter) {
    iter <- iter + 1
    old_txt <- txt

    # Find the first ternary operator
    question_pos <- regexpr("\\?", txt)
    if (question_pos == -1) break

    question_pos <- as.numeric(question_pos)

    # Find condition start (work backwards from ?)
    cond_start <- find_condition_start_simple(txt, question_pos)
    condition <- substr(txt, cond_start, question_pos - 1)

    # Find the matching colon
    colon_pos <- find_matching_colon_simple(txt, question_pos)
    if (is.null(colon_pos)) break

    # Extract then part
    then_part <- substr(txt, question_pos + 1, colon_pos - 1)

    # Find else part end
    else_end <- find_else_end_simple(txt, colon_pos)
    else_part <- substr(txt, colon_pos + 1, else_end)

    # Try to parse the condition
    parsed_cond <- parse_condition_simple(condition, ops_map)

    if (!is.null(parsed_cond)) {
      # Build replacement
      replacement <- sprintf("CppAD::CondExp%s(%s,%s,%s,%s)",
                             parsed_cond$op, parsed_cond$lhs, parsed_cond$rhs,
                             then_part, else_part)

      # Replace in string
      before <- substr(txt, 1, cond_start - 1)
      after <- substr(txt, else_end + 1, nchar(txt))
      txt <- paste0(before, replacement, after)
    } else {
      # If we can't parse condition, skip this one
      break
    }

    # Safety check - if nothing changed, break
    if (txt == old_txt) break
  }

  return(txt)
}

#' Find start of condition for a ternary operator
find_condition_start_simple <- function(txt, question_pos) {
  pos <- question_pos - 1
  paren_count <- 0

  while (pos > 0) {
    char <- substr(txt, pos, pos)

    if (char == ")") {
      paren_count <- paren_count + 1
    } else if (char == "(") {
      if (paren_count == 0) {
        return(pos + 1)
      }
      paren_count <- paren_count - 1
    } else if (paren_count == 0) {
      # Look for operators that would end an expression
      if (char %in% c("+", "-", "*", "/", ",", ";", "=", "{", "(")) {
        return(pos + 1)
      }
    }

    pos <- pos - 1
  }

  return(1)
}

#' Find matching colon for a question mark
find_matching_colon_simple <- function(txt, question_pos) {
  pos <- question_pos + 1
  paren_count <- 0
  ternary_count <- 1  # We have one unmatched ?

  while (pos <= nchar(txt)) {
    char <- substr(txt, pos, pos)

    if (char == "(") {
      paren_count <- paren_count + 1
    } else if (char == ")") {
      paren_count <- paren_count - 1
    } else if (paren_count == 0) {
      if (char == "?") {
        ternary_count <- ternary_count + 1
      } else if (char == ":") {
        ternary_count <- ternary_count - 1
        if (ternary_count == 0) {
          return(pos)
        }
      }
    }

    pos <- pos + 1
  }

  return(NULL)
}

#' Find end of else part
find_else_end_simple <- function(txt, colon_pos) {
  pos <- colon_pos + 1
  paren_count <- 0

  while (pos <= nchar(txt)) {
    char <- substr(txt, pos, pos)

    if (char == "(") {
      paren_count <- paren_count + 1
    } else if (char == ")") {
      if (paren_count == 0) {
        return(pos - 1)
      }
      paren_count <- paren_count - 1
    } else if (paren_count == 0) {
      # End of expression indicators
      if (char %in% c("+", "-", "*", "/", ",", ";", "}", ")", "?", ":")) {
        return(pos - 1)
      }
    }

    pos <- pos + 1
  }

  return(nchar(txt))
}

#' Parse condition into parts
parse_condition_simple <- function(condition, ops_map) {
  # Remove outer parentheses
  condition <- gsub("^\\((.*)\\)$", "\\1", condition)

  # Try each operator in order of specificity (longer operators first)
  operators <- c(">=", "<=", "==", "!=", ">", "<")

  for (op in operators) {
    # Escape the operator for regex
    op_escaped <- gsub("([<>=!])", "\\\\\\1", op)

    # Look for the operator
    pattern <- paste0("^(.+?)(", op_escaped, ")(.+)$")

    if (grepl(pattern, condition, perl = TRUE)) {
      # Split on this operator
      parts <- strsplit(condition, op, fixed = TRUE)[[1]]

      if (length(parts) >= 2) {
        lhs <- trimws(parts[1])
        rhs <- trimws(paste(parts[-1], collapse = op))  # In case op appears in rhs

        return(list(
          lhs = lhs,
          op = ops_map[[op]],
          rhs = rhs
        ))
      }
    }
  }

  return(NULL)
}

# Keep the other helper functions as they are
trim <- function(s) gsub("^\\s+|\\s+$", "", s)

replace_abs_max_min_for_AD <- function(txt) {
  # abs-family
  txt <- gsub("\\bstd::abs\\s*\\(([^\\)]+)\\)", "CppAD::CondExpGt(\\1, AD(0), \\1, -\\1)", txt, perl = TRUE)
  txt <- gsub("\\bfabs\\s*\\(([^\\)]+)\\)",     "CppAD::CondExpGt(\\1, AD(0), \\1, -\\1)", txt, perl = TRUE)
  txt <- gsub("\\babs\\s*\\(([^\\)]+)\\)",      "CppAD::CondExpGt(\\1, AD(0), \\1, -\\1)", txt, perl = TRUE)

  # max-family
  txt <- gsub("\\bstd::max\\s*\\(([^,]+),([^\\)]+)\\)", "CppAD::CondExpGt(\\1, \\2, \\1, \\2)", txt, perl = TRUE)
  txt <- gsub("\\bmax\\s*\\(([^,]+),([^\\)]+)\\)",      "CppAD::CondExpGt(\\1, \\2, \\1, \\2)", txt, perl = TRUE)
  txt <- gsub("\\bfmax\\s*\\(([^,]+),([^\\)]+)\\)",     "CppAD::CondExpGt(\\1, \\2, \\1, \\2)", txt, perl = TRUE)

  # min-family
  txt <- gsub("\\bstd::min\\s*\\(([^,]+),([^\\)]+)\\)", "CppAD::CondExpLt(\\1, \\2, \\1, \\2)", txt, perl = TRUE)
  txt <- gsub("\\bmin\\s*\\(([^,]+),([^\\)]+)\\)",      "CppAD::CondExpLt(\\1, \\2, \\1, \\2)", txt, perl = TRUE)
  txt <- gsub("\\bfmin\\s*\\(([^,]+),([^\\)]+)\\)",     "CppAD::CondExpLt(\\1, \\2, \\1, \\2)", txt, perl = TRUE)

  txt
}

replace_sign_heaviside <- function(txt, AD) {
  if (AD) {
    # sign / sgn / copysign(1, u) → {-1, 0, 1}
    txt <- gsub("\\bstd::copysign\\s*\\(\\s*1\\s*,\\s*([^\\)]+)\\)",
                "CppAD::CondExpGt(\\1, AD(0), AD(1), CppAD::CondExpLt(\\1, AD(0), AD(-1), AD(0)))",
                txt, perl = TRUE)
    txt <- gsub("\\bcopysign\\s*\\(\\s*1\\s*,\\s*([^\\)]+)\\)",
                "CppAD::CondExpGt(\\1, AD(0), AD(1), CppAD::CondExpLt(\\1, AD(0), AD(-1), AD(0)))",
                txt, perl = TRUE)
    txt <- gsub("(?i)\\bsign\\s*\\(([^\\)]+)\\)",
                "CppAD::CondExpGt(\\1, AD(0), AD(1), CppAD::CondExpLt(\\1, AD(0), AD(-1), AD(0)))",
                txt, perl = TRUE)
    txt <- gsub("(?i)\\bsgn\\s*\\(([^\\)]+)\\)",
                "CppAD::CondExpGt(\\1, AD(0), AD(1), CppAD::CondExpLt(\\1, AD(0), AD(-1), AD(0)))",
                txt, perl = TRUE)

    # Heaviside(u, h0) first (two-arg)
    txt <- gsub("(?i)\\bheaviside\\s*\\(([^,\\)]+)\\s*,\\s*([^\\)]+)\\)",
                "CppAD::CondExpGt(\\1, AD(0), AD(1), CppAD::CondExpLt(\\1, AD(0), AD(0), \\2))",
                txt, perl = TRUE)
    # Heaviside(u) with default h0 = 0.5
    txt <- gsub("(?i)\\bheaviside\\s*\\(([^\\)]+)\\)",
                "CppAD::CondExpGt(\\1, AD(0), AD(1), CppAD::CondExpLt(\\1, AD(0), AD(0), AD(0.5)))",
                txt, perl = TRUE)
  } else {
    # Non-AD: fall back to a safe ternary expansion
    txt <- gsub("\\bstd::copysign\\s*\\(\\s*1\\s*,\\s*([^\\)]+)\\)",
                "((\\1) > 0 ? 1 : ((\\1) < 0 ? -1 : 0))", txt, perl = TRUE)
    txt <- gsub("\\bcopysign\\s*\\(\\s*1\\s*,\\s*([^\\)]+)\\)",
                "((\\1) > 0 ? 1 : ((\\1) < 0 ? -1 : 0))", txt, perl = TRUE)
    txt <- gsub("(?i)\\bsign\\s*\\(([^\\)]+)\\)",
                "((\\1) > 0 ? 1 : ((\\1) < 0 ? -1 : 0))", txt, perl = TRUE)
    txt <- gsub("(?i)\\bsgn\\s*\\(([^\\)]+)\\)",
                "((\\1) > 0 ? 1 : ((\\1) < 0 ? -1 : 0))", txt, perl = TRUE)

    txt <- gsub("(?i)\\bheaviside\\s*\\(([^,\\)]+)\\s*,\\s*([^\\)]+)\\)",
                "((\\1) > 0 ? 1 : ((\\1) < 0 ? 0 : (\\2)))",
                txt, perl = TRUE)
    txt <- gsub("(?i)\\bheaviside\\s*\\(([^\\)]+)\\)",
                "((\\1) > 0 ? 1 : ((\\1) < 0 ? 0 : 0.5))",
                txt, perl = TRUE)
  }
  txt
}

#' Clean up generated C++ code
cleanup_generated_code <- function(code) {
  # Remove excessive parentheses around single values
  code <- gsub("\\(([0-9\\.]+)\\)", "\\1", code, perl = TRUE)
  code <- gsub("\\(([AD()0-9\\.]+)\\)", "\\1", code, perl = TRUE)

  # Clean up spacing around operators (but keep it compact)
  code <- gsub("\\s+", "", code)

  # Add minimal spacing for readability around major operators
  code <- gsub("([+\\-*/])", " \\1 ", code)
  code <- gsub("\\s+", " ", code)
  code <- gsub("^ | $", "", code)

  return(code)
}

#' Get symbols from a character
#'
#' @param char Character vector (e.g. equation)
#' @param exclude Character vector, the symbols to be excluded from the return value
#' @return character vector with the symbols
#' @author Daniel Kaschek
getSymbols <- function(char, exclude = NULL) {
  if (is.null(char))
    return(NULL)
  char <- char[char!="0"]
  out <- parse(text=char, keep.source = TRUE)
  out <- utils::getParseData(out)
  names <- unique(out$text[out$token == "SYMBOL"])
  if(!is.null(exclude)) names <- names[!names%in%exclude]
  return(names)

}

#' Sanitize symbol names for compatibility with the SymPy parser
#'
#' Reserved identifiers (Python keywords and selected built-ins) cannot be used
#' directly as variable names. This function replaces such names with the same
#' name plus an underscore appended (e.g. `while` -> `while_`).
#'
#' A warning is emitted for each renamed symbol.
#'
#' @param symbols Character vector of symbol names.
#' @return Character vector with sanitized names.
sanitizeSymbols <- function(symbols) {
  reserved <- c(
    # Python keywords
    "False","None","True","and","as","assert","async","await","break","class",
    "continue","def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","nonlocal","not","or","pass","raise","return",
    "try","while","with","yield",
    # Some built-ins that should not be used as variable names
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

#' Sanitize expressions for compatibility with the SymPy parser
#'
#' This function scans character expressions (ODE right-hand sides, event
#' values, etc.) for reserved Python keywords and replaces them with the same
#' name plus an underscore appended.
#'
#' Function names like `min`, `max`, `abs`, and `sum` are left untouched,
#' since they are valid in SymPy expressions.
#'
#' A warning is emitted for each replacement performed.
#'
#' @param exprs Character vector of expressions.
#' @return Character vector with sanitized expressions.
sanitizeExprs <- function(exprs) {
  sanitized <- exprs
  reserved_keywords <- c(
    "False","None","True","and","as","assert","async","await","break","class",
    "continue","def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","nonlocal","not","or","pass","raise","return",
    "try","while","with","yield"
  )

  # Replace reserved keywords only if they appear as standalone names
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


