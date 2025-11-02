\dontrun{
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
    root  = NA
  )

  # Generate and compile solver
  f <- CppODE(eqns, events = events, modelname = "Amodel_s")

  # Wrap in an R solver function
  solve <- function(times, params,
                    abstol = 1e-8, reltol = 1e-6,
                    maxattemps = 5000, maxsteps = 1e6,
                    roottol = 1e-8, maxroot = 1) {
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
                 as.numeric(roottol),
                 as.integer(maxroot))
    colnames(out) <- c("time", attr(f, "variables"), attr(f, "sensvariables"))

    return(out)
  }

  # Example run
  params <- c(A = 1, B = 0, k1 = 0.1, k2 = 0.2, t_e = 3)
  times  <- seq(0, 10, length.out = 300)
  res <- solve(times, params, abstol = 1e-6, reltol = 1e-6)
  head(res)
}
