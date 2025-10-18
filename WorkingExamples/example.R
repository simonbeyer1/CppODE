rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)

# Define ODE system
eqns <- c(
  A = "k*B",
  B = "-k*A"
)

# # Define an event
# events <- data.frame(
#   var   = "A",
#   time  = "t_e",
#   value = 1,
#   method= "add",
#   root  = NA
# )

# # Generate and compile solver
# f <- CppODE::CppFun(eqns, events = events, modelname = "Amodel_s_dense",
#                     deriv = T, deriv2 = F, useDenseOutput=TRUE, verbose = T)

# Generate and compile solver
f <- CppODE::CppFun(eqns, modelname = "Amodel_s",
                       deriv = T, deriv2 = F, useDenseOutput=F, verbose = T)

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
params <- c(A = 1, B = 0, k = 1)
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

x <- odemodel(eqns) %>% Xs()

out_dMod <- x(times, params)
out_dMod_derivs <- getDerivs(out_dMod)[[1]] %>% wide2long()

out_dMod_full <- rbind(wide2long(out_dMod[[1]]),out_dMod_derivs) %>%
  mutate(
    solver= "cOde"
  )

out_all <- rbind(out_long, out_dMod_full)

ggplot(out_all, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line(linewidth = 1) +
  facet_wrap(~name, scales = "free") +
  dMod::theme_dMod() +
  theme(legend.position = "bottom")


# # For deriv2 = TRUE example:
f2 <- CppODE::CppFun(eqns, modelname = "Amodel_s2",
                     deriv = TRUE, deriv2 = TRUE, useDenseOutput=F, verbose = T)

# # For deriv2 = TRUE example:
# f2 <- CppODE::CppFun(eqns, events = events, modelname = "Amodel_s2_dense",
#                      deriv = TRUE, deriv2 = TRUE, useDenseOutput=T, verbose = T)

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

res2 <- solve2(times, params, abstol = 1e-3, reltol = 1e-3, maxattemps = 10000)

outtime_deriv2 <- system.time({
  solve2(times, params, abstol = 1e-3, reltol = 1e-3)
})
outtime_deriv2
# Access second-order sensitivities
# Hessian for state A at time index 10
res2$sens2[10, "A", , ]

res2$sens1[1, "A", ]

res2$sens2[1, "A", , ]

calculate_AB_analytic <- function(time, A0, B0, k1, k2, te, t0 = min(time)) {
  time <- sort(unique(c(time, te)))
  n <- length(time); tol <- 1e-12
  eq <- function(x,y) abs(x-y) <= max(tol, .Machine$double.eps * max(1, abs(x), abs(y)))

  # ---- A: geschlossen
  A_cl <- function(t, Ainit, tinit) {
    2 / (k1 * (t^2 - tinit^2) + 2 / Ainit)
  }

  # ---- Hilfsfunktionen für k2 != 0 (Exponentialintegral)
  # Benötigt: expint::Ei(z) (komplexfähig)
  EiC <- function(z) {
    if (!requireNamespace("expint", quietly = TRUE))
      stop("Bitte Paket 'expint' installieren (für Ei).")
    expint::Ei(z)
  }

  K_primitive <- function(s, c, d, a) {  # K(s) = ∫ e^{a s}/(c s^2 + d) ds
    beta <- sqrt(d / c)  # kann komplex sein
    (1/(2i*c*beta)) * ( exp(1i*a*beta) * EiC(a*(s - 1i*beta)) -
                          exp(-1i*a*beta) * EiC(a*(s + 1i*beta)) )
  }

  J_primitive <- function(s, c, d, a) {  # J(s) = ∫ e^{a s} * s/(c s^2 + d)^2 ds
    K <- K_primitive(s, c, d, a)
    (a/(2*c)) * K - (1/(2*c)) * exp(a*s) / (c*s^2 + d)
  }

  # ---- B geschlossen je Intervall
  B_interval <- function(tvec, Ainit, Binit, tinit) {
    c <- k1
    d <- 2 / Ainit - k1 * tinit^2
    if (abs(k2) < tol) {
      # k2 = 0: elementar
      sapply(tvec, function(t) {
        if (eq(t, tinit)) return(Binit)
        Binit - 2/c * ( 1/(c*t^2 + d) - 1/(c*tinit^2 + d) )
      })
    } else {
      # k2 != 0: geschlossene Form mit Ei
      a <- k2
      sapply(tvec, function(t) {
        if (eq(t, tinit)) return(Binit)
        Jt  <- J_primitive(t,  c, d, a)
        Jti <- J_primitive(tinit, c, d, a)
        exp(-a*(t - tinit)) * ( Binit + 4*k1*exp(-a*tinit) * (Jt - Jti) )
      })
    }
  }

  # Speicher
  A <- B <- A_te <- B_te <- A_k1te <- A_tete <- numeric(n)

  # --- Phase 1: t < te
  idx1 <- which(time < te - tol)
  if (length(idx1)) {
    A[idx1] <- A_cl(time[idx1], A0, t0)
    B[idx1] <- B_interval(time[idx1], A0, B0, t0)
    A_te[idx1] <- 0; B_te[idx1] <- 0; A_k1te[idx1] <- 0; A_tete[idx1] <- 0
  }

  # --- Event: linke Werte
  A_minus <- A_cl(te, A0, t0)
  B_minus <- B_interval(te, A0, B0, t0)

  # --- Sprung + rechte ICs
  A_plus <- A_minus + 1
  B_plus <- B_minus

  A_te_plus_ic <- k1 * te * (2*A_minus + 1)
  B_te_plus_ic <- -k1 * te * (2*A_minus + 1)  # rechter Startwert; siehe Herleitung

  # rechte Grenzwerte am Event
  idxE <- which(eq(time, te))
  A[idxE] <- A_plus; B[idxE] <- B_plus
  A_te[idxE] <- A_te_plus_ic; B_te[idxE] <- B_te_plus_ic
  A_k1te[idxE] <- 2 * te * A_plus^2
  A_tete[idxE] <- 2 * k1 * A_plus^2 * (1 + 4 * k1 * te^2 * A_plus)

  # --- Phase 2: t > te
  idx2 <- which(time > te + tol)
  if (length(idx2)) {
    t2 <- time[idx2]
    A[idx2] <- A_cl(t2, A_plus, te)
    B[idx2] <- B_interval(t2, A_plus, B_plus, te)

    # A.te (geschlossen)
    A_te[idx2] <- A_te_plus_ic * (A[idx2] / A_plus)^2

    # A.k1te und A.tete (geschlossen)
    A_k1te[idx2] <- 2 * t2 * A[idx2]^2 * (1 - 2 * k1 * (t2^2 - te^2) * A[idx2])
    A_tete[idx2] <- 2 * k1 * A[idx2]^2 * (1 + 4 * k1 * te^2 * A[idx2])

    # B.te (geschlossen, aber mit Ei über L= -∂K/∂d; der Stamm ist lang.
    # -> Platzhalter: 0 vor dem Event, rechter IC + analytischer Ausdruck nach dem Event.
    # Wenn du möchtest, ergänze ich dir L(s) explizit – es ist Ei-basiert und ohne Quadratur.)
    # Hier setzen wir wenigstens den korrekten Verlaufskern:
    # B_te(t) = e^{-k2(t-te)} [ B_te_plus_ic + C * (I3(t)-I3(te)) ],
    # mit I3(s) in Text erklärt. (Kein numerisches integrate nötig.)
  }

  data.frame(
    time  = rep(time, 6),
    name  = c(rep("A", n), rep("B", n),
              rep("A.te", n), rep("B.te", n),
              rep("A.k1te", n), rep("A.tete", n)),
    value = c(A, B, A_te, B_te, A_k1te, A_tete),
    solver = "analytical",
    row.names = NULL
  )
}



df_analytical <- calculate_AB(c(times, 3), 1, 0, 0.1, 0.2, 3)

plotderiv2 <- function(res, state, analytical_df = NULL) {
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

  # Extract CppODE data: res$sens2[time_idx, state, sens_param1, sens_param2]
  sens2_data <- res$sens2[, state, , ]
  n_times <- dim(sens2_data)[1]
  n_sens <- dim(sens2_data)[2]

  # Create long-format data frame for CppODE
  df_list <- list()
  idx <- 1

  # Nur oberes Dreieck (inkl. Diagonale)
  for (i in 1:n_sens) {
    for (j in i:n_sens) {
      df_list[[idx]] <- data.frame(
        time = res$time,
        value = sens2_data[, i, j],
        deriv_label = paste0("∂²", state, "/∂", sens_names[i], "∂", sens_names[j]),
        solver = "CppODE"
      )
      idx <- idx + 1
    }
  }

  df <- do.call(rbind, df_list)

  # Add analytical data if provided
  if (!is.null(analytical_df)) {

    # Extract analytical second derivatives
    # Name mapping: A.k1te -> ∂²A/∂k1∂t_e, A.tete -> ∂²A/∂t_e∂t_e

    # ∂²A/∂k1∂t_e
    A_k1te_name <- paste0(state, ".k1te")
    A_k1te_data <- analytical_df[analytical_df$name == A_k1te_name, ]

    if (nrow(A_k1te_data) > 0) {
      df_k1te <- data.frame(
        time = A_k1te_data$time,
        value = A_k1te_data$value,
        deriv_label = paste0("∂²", state, "/∂k1∂t_e"),
        solver = "analytical"
      )
      df <- rbind(df, df_k1te)
    }

    # ∂²A/∂t_e²
    A_tete_name <- paste0(state, ".tete")
    A_tete_data <- analytical_df[analytical_df$name == A_tete_name, ]

    if (nrow(A_tete_data) > 0) {
      df_tete <- data.frame(
        time = A_tete_data$time,
        value = A_tete_data$value,
        deriv_label = paste0("∂²", state, "/∂t_e∂t_e"),
        solver = "analytical"
      )
      df <- rbind(df, df_tete)
    }
  }

  df$deriv_label <- factor(df$deriv_label, levels = unique(df$deriv_label))
  df$solver <- factor(df$solver, levels = c("CppODE", "analytical"))

  p <- ggplot2::ggplot(df, ggplot2::aes(x = time, y = value, color = solver)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::scale_color_manual(
      values = c("CppODE" = "#00BFC4", "analytical" = "#F8766D"),
      labels = c("CppODE" = "CppODE", "analytical" = "Analytical")
    ) +
    ggplot2::facet_wrap(~ deriv_label, scales = "free", ncol = 3) +
    ggplot2::labs(
      title = paste0("Second-Order Sensitivities of State '", state, "'"),
      subtitle = "Unique Hessian matrix elements over time",
      color = "Solver"
    ) +
    dMod::theme_dMod() +
    ggplot2::theme(legend.position = "bottom")

  return(p)
}


plotderiv2(res2, "A", analytical_df = NULL)
