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


# Example run
params <- c(A = 1, B = 0, k1 = 0.1, k2 = 0.2, t_e = 3)
times  <- seq(0, 10, length.out = 300)
res <- solveODE(f, times, params)

head(res$time)             # time vector
head(res$variable)         # variables
head(res$sens1[, "B", ])   # Sensitivities of B(t) w.r.t. parameters
res$sens2[10, "A", , ]     # second order sensitivities of A(time[10])
}
