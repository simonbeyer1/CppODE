\dontrun{
# CppODE() makes use of the Python module 'sympy'
# Define ODE system
eqns <- c(
  A = "-k1*A^2 * time",
  B = "k1*A^2 * time - k2*B"
)

# Define an event
events <- data.frame(
  var   = "A",
  time  = "t_e",
  value = 1,
  method= "add",
  root  = NA,
  stringsAsFactors = FALSE
)

# Generate and compile solver
f <- CppODE(eqns, events = events, deriv2 = TRUE)

solve <- function(times, params,
                  abstol = 1e-6, reltol = 1e-6,
                  maxattemps = 10L, maxsteps = 7e5L,
                  hini = 0.0, roottol = 1e-10, maxroot = 1L) {

  paramnames <- c(attr(f, "variables"), attr(f, "parameters"))
  missing <- setdiff(paramnames, names(params))
  if (length(missing) > 0) stop("Missing parameters: ", paste(missing, collapse = ", "))
  params <- params[paramnames]
  out <- .Call(paste0("solve_", as.character(f)),
               as.numeric(times),
               as.numeric(params),
               as.numeric(abstol),
               as.numeric(reltol),
               as.integer(maxattemps),
               as.integer(maxsteps),
               as.numeric(hini),
               as.numeric(roottol),
               as.integer(maxroot))

  # Extract dimension names
  dims <- attr(f, "dim_names")

  # Add column names to state matrix
  colnames(out$variable) <- dims$variable

  # Add dimension names to sens1 array if present
  if (!is.null(out$sens1)) {
    dimnames(out$sens1) <- list(NULL, dims$variable, dims$sens)
  }

  if (!is.null(out$sens2)) {
    dimnames(out$sens2) <- list(NULL, dims$variable, dims$sens, dims$sens)
  }

  return(out)
}
# Example run
params <- c(A = 1, B = 0, k1 = 0.1, k2 = 0.2, t_e = 3)
times  <- seq(0, 10, length.out = 300)
res <- solve(times, params)

head(res$time)             # time vector
head(res$variable)         # variables
head(res$sens1[, "B", ])   # Sensitivities of B(t) w.r.t. parameters
res$sens2[10, "A", , ]     # second order sensitivities of A(time[10])
}
