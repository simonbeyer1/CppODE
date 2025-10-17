rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)

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
f <- CppODE::CppFun(eqns, events = events, modelname = "Amodel_s",
                    deriv = T, deriv2 = F, verbose = F)

# Wrap in an R solver function
solve <- function(times, params,
                  abstol = 1e-8, reltol = 1e-6,
                  maxattemps = 5000, maxsteps = 1e6,
                  roottol = 1e-8, maxroot = 1) {

  paramnames <- c(attr(f, "variables"), attr(f, "parameters"))
  missing <- setdiff(paramnames, names(params))
  if (length(missing) > 0) stop("Missing parameters: ", paste(missing, collapse = ", "))

  params <- params[paramnames]

  # Call C++ solver - returns a list now
  result <- .Call(paste0("solve_", as.character(f)),
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
  colnames(result$state) <- dims$state

  # Add dimension names to sens1 array if present
  if (!is.null(result$sens1)) {
    dimnames(result$sens1) <- list(NULL, dims$state, dims$sens)
  }

  return(result)
}

# Example run
params <- c(A = 1, B = 0, k1 = 0.1, k2 = 0.2, t_e=3)
times  <- seq(0, 10, length.out = 3000)

res <- solve(times, params, abstol = 1e-6, reltol = 1e-6)

outtime_deriv <- system.time({
  solve(times, params, abstol = 1e-6, reltol = 1e-6)
})
outtime_deriv

# Access results
head(res$time)
head(res$state)

# If you want a matrix like before (for plotting etc):
out_matrix <- cbind(time = res$time, res$state)
head(out_matrix)

# Access sensitivities
if (!is.null(res$sens1)) {
  # First-order sensitivities for state A at time index 10
  res$sens1[10, "A", ]

  # Flatten to matrix format if needed (like old output)
  n_out <- length(res$time)
  n_states <- ncol(res$state)
  n_sens <- dim(res$sens1)[3]

  sens_matrix <- matrix(res$sens1, nrow = n_out, ncol = n_states * n_sens)

  # Generate column names (state.param format)
  dims <- attr(f, "dim_names")
  sens_colnames <- as.vector(outer(dims$state, dims$sens, paste, sep = "."))
  colnames(sens_matrix) <- sens_colnames

  # Combine everything into old-style matrix if needed
  out_full <- cbind(time = res$time, res$state, sens_matrix)
  head(out_full)
}

# For deriv2 = TRUE example:
f2 <- CppODE::CppFun(eqns, events = events, modelname = "Amodel_s2",
                     deriv = TRUE, deriv2 = TRUE, verbose = T)

solve2 <- function(times, params,
                   abstol = 1e-8, reltol = 1e-6,
                   maxattemps = 5000, maxsteps = 1e6,
                   roottol = 1e-8, maxroot = 1) {

  paramnames <- c(attr(f2, "variables"), attr(f2, "parameters"))
  missing <- setdiff(paramnames, names(params))
  if (length(missing) > 0) stop("Missing parameters: ", paste(missing, collapse = ", "))

  params <- params[paramnames]

  result <- .Call(paste0("solve_", as.character(f2)),
                  as.numeric(times),
                  as.numeric(params),
                  as.numeric(abstol),
                  as.numeric(reltol),
                  as.integer(maxattemps),
                  as.integer(maxsteps),
                  as.numeric(roottol),
                  as.integer(maxroot))

  # Add dimension names
  dims <- attr(f2, "dim_names")
  colnames(result$state) <- dims$state
  dimnames(result$sens1) <- list(NULL, dims$state, dims$sens)
  dimnames(result$sens2) <- list(NULL, dims$state, dims$sens, dims$sens)

  return(result)
}

res2 <- solve2(times, params, abstol = 1e-6, reltol = 1e-6)

outtime_deriv2 <- system.time({
  solve2(times, params, abstol = 1e-6, reltol = 1e-6)
})
outtime_deriv2
# Access second-order sensitivities
# Hessian for state A at time index 10
res2$sens2[10, "A", , ]

res2$sens1[1, "A", ]

plotderiv2 <- function(res, state) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotderiv2()")
  }
  if (!requireNamespace("tidyr", quietly = TRUE)) {
    stop("Package 'tidyr' is required for plotderiv2()")
  }
  # Check if res has sens2
  if (is.null(res$sens2)) {
    stop("Result object does not contain second-order sensitivities (sens2)")
  }
  # Check if state exists
  state_names <- dimnames(res$state)[[2]]
  if (!state %in% state_names) {
    stop("State '", state, "' not found. Available states: ",
         paste(state_names, collapse = ", "))
  }
  # Get dimension names for sensitivity parameters
  sens_names <- dimnames(res$sens2)[[3]]
  # Extract data: res$sens2[time_idx, state, sens_param1, sens_param2]
  sens2_data <- res$sens2[, state, , ]
  n_times <- dim(sens2_data)[1]
  n_sens <- dim(sens2_data)[2]
  # Create long-format data frame
  df_list <- list()
  idx <- 1
  # Nur oberes Dreieck (inkl. Diagonale)
  for (i in 1:n_sens) {
    for (j in i:n_sens) {  # ← Nur j >= i
      df_list[[idx]] <- data.frame(
        time = res$time,
        value = sens2_data[, i, j],
        deriv_label = paste0("∂²", state, "/∂", sens_names[i], "∂", sens_names[j])
      )
      idx <- idx + 1
    }
  }

  df <- do.call(rbind, df_list)
  df$deriv_label <- factor(df$deriv_label, levels = unique(df$deriv_label))

  p <- ggplot2::ggplot(df, ggplot2::aes(x = time, y = value)) +
    ggplot2::geom_line() +
    ggplot2::facet_wrap(~ deriv_label, scales = "free", ncol = 3) +
    ggplot2::labs(
      title = paste0("Second-Order Sensitivities of State '", state, "'"),
      subtitle = "Unique Hessian matrix elements over time"
    ) +
    dMod::theme_dMod()

  return(p)
}


plotderiv2(res2, "A")
