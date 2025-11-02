rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)

# Define ODE system
rhs <- c(
  A = "B",
  B = "-k^2 * A"
)

# Define an event
events <- data.frame(
  var   = "A",
  time  = "t_e",
  value = 1,
  method= "add",
  root  = NA
)

# # Generate and compile solver
# f <- CppODE::CppFun(eqns, events = events, modelname = "Amodel_s_dense",
#                     deriv = T, deriv2 = F, useDenseOutput=TRUE, verbose = T)

# Generate and compile solver
f <- CppODE::CppODE(rhs, events = events, modelname = "Amodel_s", deriv = T, deriv2 = F, useDenseOutput=T, verbose = F, compile = F)
CppODE:::compile(f)
# Wrap in an R solver function
solve <- function(times, params,
                  abstol = 1e-6, reltol = 1e-6,
                  maxattemps = 5000, maxsteps = 1e6,
                  roottol = 1e-8, maxroot = 1, precision = 1e-5) {

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
                  as.integer(maxroot),
                  as.numeric(precision))

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
params <- c(A = 1, B = 0, k = 1, t_e = 2)
times  <- seq(0, 2*pi, length.out = 300)

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

out_long <- dMod::wide2long(out_matrix)

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

out_long <- dMod::wide2long(out_full) %>%
  mutate(
    solver= "CppODE"
  )

library(dMod)

x <- odemodel(rhs, events = events) %>% Xs()

out_dMod <- x(times, params)
out_dMod_derivs <- getDerivs(out_dMod)[[1]] %>% wide2long()

out_dMod_full <- rbind(wide2long(out_dMod[[1]]),out_dMod_derivs) %>%
  mutate(
    solver= "cOde"
  )

outtime_cOde <- system.time({x(times, params)})
outtime_cOde


out_analytical = data.frame(
  time = times,
  name = "A.k",
  value = -times*sin(times),
  solver = "analytical"
)

out_all <- rbind(out_long, out_dMod_full, out_analytical)

ggplot(out_all, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line(linewidth = 1) +
  facet_wrap(~name, scales = "free") +
  dMod::theme_dMod() +
  theme(legend.position = "bottom")


# # For deriv2 = TRUE example:
# f2 <- CppODE::CppODE(rhs, events = events, modelname = "Amodel_s2",
#                      deriv = TRUE, deriv2 = TRUE, useDenseOutput=F, verbose = T)

# For deriv2 = TRUE example:
f2 <- CppODE::CppODE(rhs, events = events, modelname = "Amodel_s2_dense",
                     deriv = TRUE, deriv2 = TRUE, useDenseOutput=T, verbose = T)

solve2 <- function(times, params,
                   abstol = 1e-6, reltol = 1e-6,
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

res2 <- solve2(times, params)

outtime_deriv2 <- system.time({
  solve2(times, params)
})
outtime_deriv2
# Access second-order sensitivities
# Hessian for state A at time index 10
res2$sens2[10, "A", , ]

res2$sens1[1, "A", ]

res2$sens2[200, "A", , ]

calculate_AB <- function(time, A0, B0, k, t0 = min(time)) {
  # Sortiere Zeit
  time <- sort(unique(time))
  n <- length(time)
  dt <- time - t0

  # analytische Lösungen
  A <- A0 * cos(k * dt) + B0 * sin(k * dt)
  B <- B0 * cos(k * dt) - A0 * sin(k * dt)

  # erste Ableitungen nach k
  A_k <- dt * (-A0 * sin(k * dt) + B0 * cos(k * dt))
  B_k <- dt * (-B0 * sin(k * dt) - A0 * cos(k * dt))

  # zweite Ableitungen nach k
  A_kk <- -(dt^2) * A
  B_kk <- -(dt^2) * B

  # Ausgabe im Long-Format
  data.frame(
    time  = rep(time, 6),
    name  = c(rep("A", n), rep("B", n),
              rep("A.k", n), rep("B.k", n),
              rep("A.kk", n), rep("B.kk", n)),
    value = c(A, B, A_k, B_k, A_kk, B_kk),
    solver = "analytical",
    row.names = NULL
  )
}


df_analytical <- calculate_AB(times, 1, 0, 1)

plotderiv2 <- function(res, state, analytical_df = NULL, add_first_deriv = FALSE) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) stop("Package 'ggplot2' is required")
  if (!requireNamespace("tidyr", quietly = TRUE))   stop("Package 'tidyr' is required")

  if (is.null(res$sens2)) stop("Result object does not contain second-order sensitivities (sens2)")

  state_names <- dimnames(res$state)[[2]]
  if (!state %in% state_names) {
    stop("State '", state, "' not found. Available states: ", paste(state_names, collapse = ", "))
  }

  sens_names <- dimnames(res$sens2)[[3]]

  # --- CppODE second derivatives (unique upper triangle) ---
  sens2_data <- res$sens2[, state, , ]
  n_times <- dim(sens2_data)[1]
  n_sens  <- dim(sens2_data)[2]

  df_list <- list(); idx <- 1
  for (i in 1:n_sens) {
    for (j in i:n_sens) {
      df_list[[idx]] <- data.frame(
        time = res$time,
        value = sens2_data[, i, j],
        deriv_label = paste0("∂²", state, "/∂", sens_names[i], "∂", sens_names[j]),
        solver = "CppODE",
        stringsAsFactors = FALSE
      )
      idx <- idx + 1
    }
  }
  df <- do.call(rbind, df_list)

  # --- Add analytical: ONLY the (k,k) entry; put it into the SAME facet label ---
  if (!is.null(analytical_df)) {
    # ensure same grid (optional)
    analytical_df <- analytical_df[analytical_df$time %in% res$time, , drop = FALSE]

    # name for second derivative in analytical df
    kk_name <- paste0(state, ".kk")
    df_kk   <- analytical_df[analytical_df$name == kk_name, c("time", "value")]
    if (nrow(df_kk) > 0) {
      label_k2 <- paste0("∂²", state, "/∂k∂k")  # EXACTLY like CppODE
      df_add <- data.frame(
        time = df_kk$time,
        value = df_kk$value,
        deriv_label = label_k2,
        solver = "analytical",
        stringsAsFactors = FALSE
      )
      df <- rbind(df, df_add)
    }

    # (optional) also overlay 1st deriv in a separate facet if desired
    if (isTRUE(add_first_deriv)) {
      k_name <- paste0(state, ".k")
      df_k   <- analytical_df[analytical_df$name == k_name, c("time", "value")]
      if (nrow(df_k) > 0) {
        df_add1 <- data.frame(
          time = df_k$time,
          value = df_k$value,
          deriv_label = paste0("∂", state, "/∂k (1st deriv)"),
          solver = "analytical",
          stringsAsFactors = FALSE
        )
        df <- rbind(df, df_add1)
      }
    }
  }

  df$deriv_label <- factor(df$deriv_label, levels = unique(df$deriv_label))
  df$solver <- factor(df$solver, levels = c("CppODE", "analytical"))

  p <- ggplot2::ggplot(df, ggplot2::aes(x = time, y = value, color = solver, linetype = solver)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::scale_color_manual(values = c("CppODE" = "#00BFC4", "analytical" = "#F8766D")) +
    ggplot2::scale_linetype_manual(values = c("CppODE" = "solid", "analytical" = "dashed")) +
    ggplot2::facet_wrap(~ deriv_label, scales = "free", ncol = 3) +
    ggplot2::labs(
      title = paste0("Second-Order Sensitivities of State '", state, "'"),
      subtitle = "CppODE (solid) vs Analytical (dashed)",
      color = "Solver", linetype = "Solver"
    ) +
    dMod::theme_dMod() +
    ggplot2::theme(legend.position = "bottom")

  p
}

plotderiv2(res2, "A", analytical_df = df_analytical)
