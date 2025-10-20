\dontrun{
  # =============================================================================
  # Example: ODE system
  # =============================================================================

  library(CppODE)
  library(ggplot2)
  library(gridExtra)

  # Define ODE system (harmonic oscillator)
  eqns <- c(
    A = "-k^2*B",
    B = "A"
  )

  # =============================================================================
  # Generate solvers
  # =============================================================================

  f0 <- CppODE::CppFun(eqns, deriv = FALSE, deriv2 = FALSE,
                       modelname = "HO_nosens")

  f1 <- CppODE::CppFun(eqns, deriv = TRUE, deriv2 = FALSE,
                       modelname = "HO_deriv1")

  f2 <- CppODE::CppFun(eqns, deriv = TRUE, deriv2 = TRUE,
                       modelname = "HO_deriv2")

  # =============================================================================
  # Wrapper functions (adapted for list output)
  # =============================================================================

  solve0 <- function(times, params, abstol = 1e-6, reltol = 1e-6) {
    paramnames <- c(attr(f0, "variables"), attr(f0, "parameters"))
    params <- params[paramnames]

    result <- .Call(paste0("solve_", f0), as.numeric(times), as.numeric(params),
                    as.numeric(abstol), as.numeric(reltol), 5000L, 1e6L, 1e-8, 1L)

    # result is a list with: time, states
    # Add column names to states matrix
    states <- attr(f0, "variables")
    colnames(result$states) <- states

    return(result)
  }

  solve1 <- function(times, params, abstol = 1e-6, reltol = 1e-6) {
    paramnames <- c(attr(f1, "variables"), attr(f1, "parameters"))
    params <- params[paramnames]

    result <- .Call(paste0("solve_", f1), as.numeric(times), as.numeric(params),
                    as.numeric(abstol), as.numeric(reltol), 5000L, 1e6L, 1e-8, 1L)

    # result is a list with: time, states, sens1
    # Add dimnames to arrays for easier access
    states <- attr(f1, "variables")
    sens_vars <- setdiff(c(states, attr(f1, "parameters")), attr(f1, "fixed"))

    colnames(result$states) <- states
    dimnames(result$sens1) <- list(NULL, states, sens_vars)

    return(result)
  }

  solve2 <- function(times, params, abstol = 1e-6, reltol = 1e-6) {
    paramnames <- c(attr(f2, "variables"), attr(f2, "parameters"))
    params <- params[paramnames]

    result <- .Call(paste0("solve_", f2), as.numeric(times), as.numeric(params),
                    as.numeric(abstol), as.numeric(reltol), 5000L, 1e6L, 1e-8, 1L)

    # result is a list with: time, states, sens1, sens2
    # Add dimnames to arrays
    states <- attr(f2, "variables")
    sens_vars <- setdiff(c(states, attr(f2, "parameters")), attr(f2, "fixed"))

    colnames(result$states) <- states
    dimnames(result$sens1) <- list(NULL, states, sens_vars)
    dimnames(result$sens2) <- list(NULL, states, sens_vars, sens_vars)

    return(result)
  }

  # =============================================================================
  # Run examples
  # =============================================================================

  params <- c(A = 1, B = 0, k = 2)
  times  <- seq(0, 2*pi, length.out = 300)

  res0 <- solve0(times, params)
  res1 <- solve1(times, params)
  res2 <- solve2(times, params)

  # =============================================================================
  # Access examples
  # =============================================================================

  # Access states: res0$states is a matrix[n_out, n_states]
  # Access sens1: res1$sens1 is an array[n_out, n_states, n_sens]
  # Access sens2: res2$sens2 is an array[n_out, n_states, n_sens, n_sens]

  # Example: Extract dA/dk (first-order sensitivity)
  dA_dk <- res1$sens1[, "A", "k"]

  # Example: Extract d²A/dk² (second-order sensitivity)
  d2A_dk2 <- res2$sens2[, "A", "k", "k"]

  cat("dA/dk at t=π:", dA_dk[which.min(abs(res1$time - pi))], "\n")
  cat("d²A/dk² at t=π:", d2A_dk2[which.min(abs(res2$time - pi))], "\n")

  # =============================================================================
  # Prepare data for plotting
  # =============================================================================

  df0 <- data.frame(
    time = res0$time,
    A = res0$states[, "A"],
    B = res0$states[, "B"]
  )

  df1 <- data.frame(
    time = res1$time,
    A = res1$states[, "A"],
    B = res1$states[, "B"],
    dA_dk = res1$sens1[, "A", "k"],
    dB_dk = res1$sens1[, "B", "k"]
  )

  df2 <- data.frame(
    time = res2$time,
    A = res2$states[, "A"],
    B = res2$states[, "B"],
    dA_dk = res2$sens1[, "A", "k"],
    dB_dk = res2$sens1[, "B", "k"],
    d2A_dk2 = res2$sens2[, "A", "k", "k"],
    d2B_dk2 = res2$sens2[, "B", "k", "k"]
  )

  solve_A <- function(time, A0, k) {
    # Ensure numeric
    time <- as.numeric(time)
    A0 <- as.numeric(A0)
    k <- as.numeric(k)

    # Solution and sensitivities
    A <- A0 * cos(k * time)
    dA_dk <- -A0 * time * sin(k * time)
    d2A_dk2 <- -A0 * time^2 * cos(k * time)

    # Return as data frame
    data.frame(
      time = time,
      A = A,
      dA_dk = dA_dk,
      d2A_dk2 = d2A_dk2
    )
  }

  df2_analytical <- solve_A(times, 1, 2)
  # =============================================================================
  # Plot results
  # =============================================================================

  p1 <- ggplot(df0, aes(x = time, y = A)) +
    geom_line(color = "blue", linewidth = 1) +
    labs(title = "State A (cos-like)", x = "Time", y = "A") +
    dMod::theme_dMod()

  p2 <- ggplot(df0, aes(x = time, y = B)) +
    geom_line(color = "green", linewidth = 1) +
    labs(title = "State B (sin-like)", x = "Time", y = "B") +
    dMod::theme_dMod()

  p3 <- ggplot(df1, aes(x = time, y = dA_dk)) +
    geom_line(color = "purple", linewidth = 1) +
    labs(title = "First-order: dA/dk", x = "Time", y = "dA/dk") +
    dMod::theme_dMod()

  p4 <- ggplot(df2, aes(x = time, y = d2A_dk2)) +
    geom_line(color = "orange", linewidth = 1) +
    labs(title = "Second-order: d²A/dk²", x = "Time", y = "d²A/dk²") +
    dMod::theme_dMod()

  grid.arrange(p1, p2, p3, p4, ncol = 2)

  # Comparison analytical and numerical solution
  ggplot() +
    geom_line(data = df2_analytical, aes(x = time, y = d2A_dk2, color = "Analytical"),
              linewidth = 1) +
    geom_line(data = df2, aes(x = time, y = d2A_dk2, color = "Numerical (CppODE)"),
              linewidth = 1, linetype = "dashed") +
    scale_color_manual(
      values = c("Analytical" = "red", "Numerical (CppODE)" = "orange")
    ) +
    labs(
      title = "Second-order: d²A/dk²",
      x = "Time", y = "d²A/dk²",
      color = "Legend"
    ) +
    dMod::theme_dMod()
}
