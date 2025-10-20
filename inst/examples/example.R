\dontrun{
  # Define ODE system
  eqns <- c(
    A = "-k1 * A^2 * time",
    B = "k1 * A^2 * time - k2 * B"
  )

  # Define an event
  events <- data.frame(
    var    = "A",
    time   = "t_e",
    value  = 1,
    method = "add",
    root   = NA
  )

  # Generate and compile solver (first-order sensitivities)
  f <- CppODE::CppFun(
    eqns,
    events = events,
    modelname = "Amodel_s",
    deriv = TRUE,
    deriv2 = FALSE
  )

  # R wrapper for the compiled C++ solver
  solve <- function(times, params,
                     abstol = 1e-6, reltol = 1e-6,
                     maxattempts = 5000L, maxsteps = 1e6L,
                     roottol = 1e-8, maxroot = 1L) {

    # Determine parameter order
    paramnames <- c(attr(f, "variables"), attr(f, "parameters"))
    missing <- setdiff(paramnames, names(params))
    if (length(missing) > 0)
      stop("Missing parameters: ", paste(missing, collapse = ", "))

    params <- params[paramnames]

    # Call compiled solver
    result <- .Call(
      paste0("solve_", f),
      as.numeric(times),
      as.numeric(params),
      as.numeric(abstol),
      as.numeric(reltol),
      as.integer(maxattempts),
      as.integer(maxsteps),
      as.numeric(roottol),
      as.integer(maxroot)
    )

    # Extract output components
    states <- attr(f, "variables")
    sens_vars <- setdiff(c(states, attr(f, "parameters")), attr(f, "fixed"))

    # Add dimnames for clarity
    colnames(result$states) <- states
    dimnames(result$sens1) <- list(NULL, states, sens_vars)

    return(result)
  }

  # Define parameters and time grid
  params <- c(A = 1, B = 0, k1 = 0.1, k2 = 0.2, t_e = 3)
  times <- seq(0, 10, length.out = 300)

  # Run solver
  res <- solve(times, params)

  # Inspect output
  str(res)

}
