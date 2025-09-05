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
#'
#' @return A list with:
#' \describe{
#'   \item{f.x}{Jacobian matrix entries as strings (R representation).}
#'   \item{f.time}{Explicit derivatives with respect to time as strings.}
#' }
#' The returned list has an attribute \code{CppCode} containing the full C++ struct.
#'
#' @author Simon Beyer, \email{simon.beyer@@fdm.uni-freiburg.de}
#' @export
#' @examples
#' odes <- c(x = "v", v = "mu*(1 - x^2)*v - x")
#' res <- ComputeJacobianSymb(odes)
#' res$f.x
#' res$f.time
#' cat(attr(res, "CppCode"))
ComputeJacobianSymb <- function(odes, states = NULL, params = NULL) {
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
    "  vector<AD> params;",
    "  void operator()(const vector<AD>& x, matrix<AD>& J, const AD& t, vector<AD>& dfdt) {"
  )

  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      code <- Sympy2CppCode(sympy$diff(exprs[[i]], syms_states[[j]]), states, params, n, expr_name = states[i])
      cpp_lines <- c(cpp_lines, sprintf("    J(%d,%d)=%s;", i-1, j-1, code))
    }
  }

  for (i in seq_len(n)) {
    code <- Sympy2CppCode(sympy$diff(exprs[[i]], t), states, params, n, expr_name = states[i])
    cpp_lines <- c(cpp_lines, sprintf("    dfdt[%d]=%s;", i-1, code))
  }

  cpp_lines <- c(cpp_lines, "  }", "};")
  cpp_code <- paste(cpp_lines, collapse = "\n")

  out <- list(f.x = f.x, f.time = f.time)
  attr(out, "CppCode") <- cpp_code
  return(out)
}


#' Convert a SymPy expression to C++ code suitable for Boost solvers with CppAD
#'
#' This internal function converts a SymPy expression into valid C++ code,
#' replacing symbols according to the following rules:
#' \itemize{
#'   \item States \code{s1,...,sn} → \code{x[0],...,x[n-1]}.
#'   \item Initial values \code{s1_0,...,sn_0} → \code{params[0],...,params[n-1]}.
#'   \item Parameters \code{p1,...,pm} → \code{params[n],...,params[n+m-1]}.
#'   \item The time variable \code{time} → \code{t}.
#' }
#'
#' Throws an error if the expression contains constructs not supported in C++.
#'
#' @param expr SymPy expression.
#' @param states Character vector of state variable names.
#' @param params Character vector of parameter names.
#' @param n Number of states.
#' @param expr_name Optional. Name of the ODE (used only in error messages).
#'
#' @return A character string with valid C++ code for use in AD solvers.
#'
#' @keywords internal
#' @author Simon Beyer
Sympy2CppCode <- function(expr, states, params, n, expr_name = NULL) {
  sympy <- reticulate::import("sympy")
  code <- sympy$cxxcode(expr, standard = "c++17", strict = FALSE)

  if (grepl("NotsupportedinC\\+\\+", gsub("\\s+", "", code))) {
    stop(paste0(
      "Cannot generate C++ code for rhs of",
      if (!is.null(expr_name)) paste0(" '", expr_name, "'"),
      ": contains terms not supported in C++ (see SymPy cxxcode output)."
    ))
  }

  # Replace state vars
  for (i in seq_along(states)) {
    code <- gsub(paste0("\\b", states[i], "\\b"), sprintf("x[%d]", i-1), code)
    code <- gsub(paste0("\\b", states[i], "_0\\b"), sprintf("params[%d]", i-1), code)
  }
  # Replace parameters
  for (i in seq_along(params)) {
    code <- gsub(paste0("\\b", params[i], "\\b"), sprintf("params[%d]", n + i - 1), code)
  }
  # Replace time
  code <- gsub("\\btime\\b", "t", code)

  gsub("\\s+", "", code)
}



#' Replace symbols in a character vector by other symbols
#'
#' @param what vector of type character, the symbols to be replaced, e.g. c("A", "B")
#' @param by vector of type character, the replacement, e.g. c("x[0]", "x[1]")
#' @param x vector of type character, the object where the replacement should take place
#' @return vector of type character, conserves the names of x.
#' @examples replaceSymbols(c("A", "B"), c("x[0]", "x[1]"), c("A*B", "A+B+C"))
#' @author Daniel Kaschek
#' @export
replaceSymbols <- function(what, by, x) {

  xOrig <- x
  is.not.zero <- which(x!="0")
  x <- x[is.not.zero]

  mynames <- names(x)

  x.parsed <- parse(text = x, keep.source = TRUE)
  data <- utils::getParseData(x.parsed)

  by <- rep(by, length.out=length(what))
  names(by) <- what

  data$text[data$text%in%what] <- by[data$text[data$text%in%what]]
  data <- data[data$token!="expr",]



  breaks <- c(0, which(diff(data$line1) == 1), length(data$line1))

  out <- lapply(1:(length(breaks)-1), function(i) {

    paste(data$text[(breaks[i]+1):(breaks[i+1])], collapse="")

  })

  names(out) <- mynames
  out <- unlist(out)

  xOrig[is.not.zero] <- out

  return(xOrig)
}

#' Replace integer number in a character vector by other double
#'
#' @param x vector of type character, the object where the replacement should take place
#' @return vector of type character, conserves the names of x.
#' @author Daniel Kaschek
#' @export
replaceNumbers <- function(x) {

  xOrig <- x
  is.not.zero <- which(x!="0")
  x <- x[is.not.zero]

  mynames <- names(x)

  x.parsed <- parse(text = x, keep.source = TRUE)
  data <- utils::getParseData(x.parsed)
  data$text[data$token == "NUM_CONST"] <- format(as.numeric(data$text[data$token == "NUM_CONST"]), nsmall = 1)
  breaks <- c(0, which(diff(data$line1) == 1), length(data$line1))

  out <- lapply(1:(length(breaks)-1), function(i) {

    paste(data$text[(breaks[i]+1):(breaks[i+1])], collapse="")

  })

  names(out) <- mynames
  out <- unlist(out)

  xOrig[is.not.zero] <- out

  return(xOrig)


}


#' Get symbols from a character
#'
#' @param char Character vector (e.g. equation)
#' @param exclude Character vector, the symbols to be excluded from the return value
#' @return character vector with the symbols
#' @examples getSymbols(c("A*AB+B^2"))
#' @author Daniel Kaschek
#' @export
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


#' Compute matrix product symbolically
#'
#' @param M matrix of type character
#' @param N matrix of type character
#' @return Matrix of type character, the matrix product of M and N
#' @export
prodSymb <- function(M, N) {

  red <- sapply(list(M, N), is.null)
  if(all(red)) {
    return()
  } else if(red[1]) {
    return(N)
  } else if(red[2]) {
    return(M)
  }

  dimM <- dim(M)
  dimN <- dim(N)
  if(dimM[2] != dimN[1]) {
    cat("Something is wrong with the dimensions of the matrices\n")
    return(NA)
  }
  m <- 1:dimM[1]
  n <- 1:dimN[2]
  grid <- expand.grid(m,n)

  MN <- apply(grid, 1, function(ik) {

    v <- M[ik[1],]
    w <- N[,ik[2]]
    result <- ""
    for(i in 1:length(v)) {
      if(i==1 & !(0 %in% c(w[i], v[i]))) result <- paste("(", v[i], ") * (", w[i], ")", sep="")
      if(i==1 & (0 %in% c(w[i], v[i]))) result <- "0"
      if(i >1 & !(0 %in% c(w[i], v[i]))) result <- paste(result," + (", v[i], ") * (", w[i], ")", sep="")
      if(i >1 & (0 %in% c(w[i], v[i]))) result <- result

    }
    if(substr(result, 1,3)=="0 +") result <- substr(result, 5, nchar(result))
    return(result)})

  return(matrix(MN, nrow=dimM[1], ncol=dimN[2]))


}

#' Compute matrix sumSymbolically
#'
#' @param M matrix of type character
#' @param N matrix of type character
#' @return Matrix of type character, the matrix sum of M and N
#' @author Daniel Kaschek
#' @export
sumSymb <- function(M, N) {

  red <- sapply(list(M, N), is.null)
  if(all(red)) {
    return()
  } else if(red[1]) {
    return(N)
  } else if(red[2]) {
    return(M)
  }


  if (inherits(M, "matrix")) dimM <- dim(M) else dimM <- c(length(M),1)
  if (inherits(N, "matrix")) dimN <- dim(N) else dimN <- c(length(N),1)

  M <- as.character(M)
  N <- as.character(N)
  result <- c()

  for(i in 1:length(M)) {
    if(M[i] == "0" & N[i] == "0") result[i] <- "0"
    if(M[i] == "0" & N[i] != "0") result[i] <- N[i]
    if(M[i] != "0" & N[i] == "0") result[i] <- M[i]
    if(M[i] != "0" & N[i] != "0") result[i] <- paste(M[i]," + ",N[i], sep="")
  }
  return(matrix(result, nrow=dimM[1], ncol=dimM[2]))

}

