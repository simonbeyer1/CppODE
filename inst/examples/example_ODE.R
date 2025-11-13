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
  f <- CppODE(eqns, events = events, modelname = "Amodel_s", deriv2 = T)

  # Wrap in an R solver function
  solve <- function(times, params,
                    abstol = 1e-6, reltol = 1e-6,
                    maxattemps = 5000L, maxsteps = 1e6L,
                    roottol = 1e-6, maxroot = 1L) {
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

    # Extract dimension names
    dims <- attr(f, "dim_names")

    # Add column names to state matrix
    colnames(out$variable) <- dims$variable

    # Add dimension names to sensitivity arrays
    dimnames(out$sens1) <- list(NULL, dims$variable, dims$sens)
    dimnames(out$sens2) <- list(NULL, dims$variable, dims$sens, dims$sens)

    return(out)
  }

  # Example run
  params <- c(A = 1, B = 0, k1 = 0.1, k2 = 0.2, t_e = 3)
  times  <- seq(0, 10, length.out = 300)
  res <- solve(times, params)
  head(res$variable)
  head(res$sens1[, "A", ])
  res$sens2[10, "A", , ]
}
