rm(list = ls(all.names = TRUE))
.workingDir <- file.path(tempdir(), "CppODE_example_coupled_vanderpol")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(cOde)
library(deSolve)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

# ============================================================================
# Coupled Van der Pol Oscillators -- configurable coupling reach
#
#   dx_i/dt = v_i
#   dv_i/dt = mu_i*(1 - x_i^2)*v_i - omega_i^2*x_i
#             + kappa * sum_{j in neighbors(i)} (x_j - x_i)
#
#   - "neighbors" controlled by `reach` parameter:
#       reach = 1  => nearest-neighbor coupling only
#       reach = k  => couple to k neighbors on each side
#       reach = N  => all-to-all coupling (dense Jacobian)
#
#   Each oscillator has its own mu_i, omega_i.
#   => N oscillators, 2*N states, 2*N + 1 parameters (mu_i, omega_i, kappa)
#
# Comparison: cOde + lsoda (with sensitivitiesSymb) vs CppODE sparse (deriv)
# ============================================================================


# ===== CONFIGURATION =====
N <- 20L                    # Number of oscillators
target_sparsity <- 80       # Target Jacobian sparsity in %  (0 = dense, 95 = very sparse)
# ==========================

# ----------------------------------------------------------------------------
# Compute reach from target sparsity
# ----------------------------------------------------------------------------
compute_sparsity <- function(N, reach) {
  n_states <- 2L * N
  if (reach >= N) {
    nnz <- N + N + N * N
    return(100 * (1 - nnz / n_states^2))
  }
  nnz_dvdx <- 0L
  for (i in seq_len(N)) {
    lo <- max(1L, i - reach)
    hi <- min(N, i + reach)
    nnz_dvdx <- nnz_dvdx + (hi - lo + 1L)
  }
  nnz <- N + N + nnz_dvdx
  100 * (1 - nnz / n_states^2)
}

reach <- pmax(1L, round((4 * N * (1 - target_sparsity / 100) - 3) / 2))
reach <- pmin(reach, N)

n_states <- 2L * N
actual_sparsity <- compute_sparsity(N, reach)

cat(sprintf("System: %d coupled Van der Pol oscillators\n", N))
cat(sprintf("  States:          %d\n", n_states))
cat(sprintf("  Coupling reach:  %d (target sparsity: %d%%, actual: %.1f%%)\n",
            reach, target_sparsity, actual_sparsity))
cat(sprintf("  Parameters:      %d (mu_i, omega_i, kappa)\n", 2L * N + 1L))
cat(sprintf("  Sens dims:       %d states x %d params = %d\n",
            n_states, n_states + 2L * N + 1L,
            n_states * (n_states + 2L * N + 1L)))


# ----------------------------------------------------------------------------
# Build RHS with configurable reach
# ----------------------------------------------------------------------------
build_coupled_vdp <- function(N, reach) {
  x_vars <- paste0("x", seq_len(N))
  v_vars <- paste0("v", seq_len(N))
  n_states <- 2L * N

  rhs <- character(n_states)
  names(rhs) <- c(x_vars, v_vars)

  for (i in seq_len(N)) {
    rhs[x_vars[i]] <- v_vars[i]

    mu_i    <- paste0("mu", i)
    omega_i <- paste0("omega", i)

    self_terms <- sprintf("%s * (1 - %s^2) * %s - %s^2 * %s",
                          mu_i, x_vars[i], v_vars[i], omega_i, x_vars[i])

    lo <- max(1L,  i - reach)
    hi <- min(N, i + reach)
    nbrs <- setdiff(seq(lo, hi), i)

    coupling_terms <- character(0)
    for (j in nbrs) {
      coupling_terms <- c(coupling_terms,
                          sprintf("kappa * (%s - %s)", x_vars[j], x_vars[i]))
    }

    if (length(coupling_terms) > 0) {
      rhs[v_vars[i]] <- paste(c(self_terms, coupling_terms), collapse = " + ")
    } else {
      rhs[v_vars[i]] <- self_terms
    }
  }

  rhs
}

rhs <- build_coupled_vdp(N, reach)


# ============================================================================
# Compile models
# ============================================================================
cat("\n--- Compiling models ---\n")

# cOde: generate sensitivity equations symbolically
cat("  Generating sensitivity equations (cOde::sensitivitiesSymb)...\n")
t_senssymb <- system.time({
  rhs.sens <- cOde::sensitivitiesSymb(rhs)
})
cat(sprintf("  sensitivitiesSymb: %.1f s\n", t_senssymb["elapsed"]))

n_sens_eqns <- length(rhs.sens)
cat(sprintf("  Sensitivity ODE system: %d equations total\n",
            length(rhs) + n_sens_eqns))

cat("  Compiling cOde model (rhs only)...\n")
t_compile_code_nosens <- system.time({
  func <- cOde::funC(rhs,
                     modelname = sprintf("vdp%d_r%d_nosens_code", N, reach),
                     compile = TRUE)
})
cat(sprintf("  cOde compile (no sens): %.1f s\n", t_compile_code_nosens["elapsed"]))

cat("  Compiling cOde model (rhs + sensitivities)...\n")
t_compile_code_sens <- system.time({
  func_sens <- cOde::funC(c(rhs, rhs.sens),
                          modelname = sprintf("vdp%d_r%d_sens_code", N, reach),
                          compile = TRUE)
})
cat(sprintf("  cOde compile (with sens): %.1f s\n", t_compile_code_sens["elapsed"]))

# CppODE: NO sensitivities -- baselines
cat("  Compiling CppODE sparse model (deriv = FALSE)...\n")
t_compile_nosens_sparse <- system.time({
  funcpp_nosens_sparse <- CppODE(rhs, outdir = getwd(), method = "bdf",
                                 modelname = sprintf("vdp%d_r%d_nosens_sparse", N, reach),
                                 compile = TRUE, deriv = FALSE)
})
cat(sprintf("  CppODE sparse (no sens): %.1f s\n", t_compile_nosens_sparse["elapsed"]))


cat("  Compiling CppODE dense model (deriv = FALSE)...\n")
t_compile_nosens_dense <- system.time({
  funcpp_nosens_dense <- CppODE(rhs, outdir = getwd(), method = "bdf",
                                modelname = sprintf("vdp%d_r%d_nosens_dense", N, reach),
                                compile = TRUE, deriv = FALSE)
})
cat(sprintf("  CppODE dense (no sens): %.1f s\n", t_compile_nosens_dense["elapsed"]))

# CppODE: WITH sensitivities (AD)
cat("  Compiling CppODE sparse BDF (deriv = TRUE)...\n")
t_compile_cppode_sparse <- system.time({
  funcpp_sparse <- CppODE(rhs, outdir = getwd(), method = "bdf",
                          modelname = sprintf("vdp%d_r%d_sparse_sens", N, reach),
                          compile = TRUE, deriv = TRUE)
})
cat(sprintf("  CppODE sparse BDF (AD): %.1f s\n", t_compile_cppode_sparse["elapsed"]))

cat("  Compiling CppODE sparse RB4 (deriv = TRUE)...\n")
t_compile_cppode_sparse_rb <- system.time({
  funcpp_sparse_rb <- CppODE(rhs, outdir = getwd(), method = "rb4",
                             modelname = sprintf("vdp%d_r%d_sparse_sens_rb", N, reach),
                             compile = TRUE, deriv = TRUE)
})
cat(sprintf("  CppODE sparse RB4 (AD): %.1f s\n", t_compile_cppode_sparse_rb["elapsed"]))

cat("  Compiling CppODE dense BDF (deriv = TRUE)...\n")
t_compile_cppode_dense <- system.time({
  funcpp_dense <- CppODE(rhs, outdir = getwd(), method = "bdf",
                         modelname = sprintf("vdp%d_r%d_dense_sens", N, reach),
                         compile = TRUE, deriv = TRUE, sparse = FALSE)
})
cat(sprintf("  CppODE dense BDF (AD): %.1f s\n", t_compile_cppode_dense["elapsed"]))

cat("  Compiling CppODE dense RB4 (deriv = TRUE)...\n")
t_compile_cppode_dense_rb <- system.time({
  funcpp_dense_rb <- CppODE(rhs, outdir = getwd(), method = "rb4",
                            modelname = sprintf("vdp%d_r%d_dense_sens_rb", N, reach),
                            compile = TRUE, deriv = TRUE, sparse = FALSE)
})
cat(sprintf("  CppODE dense RB4 (AD): %.1f s\n", t_compile_cppode_dense_rb["elapsed"]))

# ============================================================================
# Initial conditions & parameters
# ============================================================================

param_vals <- c()
for (i in seq_len(N)) {
  param_vals <- c(param_vals,
                  setNames(7.0 + 0.1 * sin(i),       paste0("mu", i)),
                  setNames(1.0 + 0.05 * cos(i),      paste0("omega", i)))
}
param_vals <- c(param_vals, kappa = 0.8)

yini_x <- setNames(sin(seq_len(N) * pi / (N + 1L)), paste0("x", seq_len(N)))
yini_v <- setNames(rep(0, N), paste0("v", seq_len(N)))

params <- c(yini_x, yini_v, param_vals)
state_names <- c(paste0("x", seq_len(N)), paste0("v", seq_len(N)))

yini_code_sens <- c(params[state_names], attr(rhs.sens, "yini"))
yini_code      <- params[state_names]
parsC <- params[setdiff(names(params), state_names)]

times <- seq(0, 10, length.out = 300)


# ============================================================================
# Benchmark 0: No sensitivities -- pure ODE solve (baseline)
# ============================================================================

cat("\n\n========== BASELINE (no sensitivities) ==========\n\n")

cat("--- cOde + lsodes (sparse, NO sensitivities) ---\n")
cat(sprintf("  ODE dimension: %d\n", length(yini_code)))
t_lsodes_nosens <- system.time({
  res_cOde_nosens <- cOde::odeC(yini_code, times, func, parsC, method = "lsodes")
})
cat(sprintf("  Time: %.3f s\n", t_lsodes_nosens["elapsed"]))

cat("\n--- CppODE BDF sparse (no sens) ---\n")
t_nosens_sparse <- system.time({
  res_nosens_sparse <- solveODE(funcpp_nosens_sparse, times, params)
})
cat(sprintf("  Time: %.3f s\n", t_nosens_sparse["elapsed"]))
CppODE::diagnostics(res_nosens_sparse)

cat("\n--- CppODE BDF dense (no sens) ---\n")
t_nosens_dense <- system.time({
  res_nosens_dense <- solveODE(funcpp_nosens_dense, times, params)
})
cat(sprintf("  Time: %.3f s\n", t_nosens_dense["elapsed"]))
CppODE::diagnostics(res_nosens_dense)


# ============================================================================
# Benchmark 1: With sensitivities
# ============================================================================

cat("\n\n========== BENCHMARK (with sensitivities) ==========\n\n")

cat("--- cOde + lsodes (variational sensitivities) ---\n")
cat(sprintf("  Total ODE dimension: %d (states + sensitivity states)\n",
            length(yini_code_sens)))
t_lsodes_sens <- system.time({
  res_cOde_sens <- cOde::odeC(yini_code_sens, times, func_sens, parsC,
                              method = "lsodes")
})
cat(sprintf("  Time: %.3f s\n", t_lsodes_sens["elapsed"]))

cat("\n--- CppODE BDF sparse (AD sensitivities) ---\n")
cat(sprintf("  ODE dimension: %d (dual numbers propagate sensitivities)\n", n_states))
t_cppode_sparse <- system.time({
  res_cppode_sparse <- solveODE(funcpp_sparse, times, params)
})
cat(sprintf("  Time: %.3f s\n", t_cppode_sparse["elapsed"]))
CppODE::diagnostics(res_cppode_sparse)

cat("\n--- CppODE RB4 sparse (AD sensitivities) ---\n")
t_cppode_sparse_rb <- system.time({
  res_cppode_sparse_rb <- solveODE(funcpp_sparse_rb, times, params)
})
cat(sprintf("  Time: %.3f s\n", t_cppode_sparse_rb["elapsed"]))
CppODE::diagnostics(res_cppode_sparse_rb)

cat("\n--- CppODE BDF dense (AD sensitivities) ---\n")
t_cppode_dense <- system.time({
  res_cppode_dense <- solveODE(funcpp_dense, times, params)
})
cat(sprintf("  Time: %.3f s\n", t_cppode_dense["elapsed"]))
CppODE::diagnostics(res_cppode_dense)

cat("\n--- CppODE RB4 dense (AD sensitivities) ---\n")
t_cppode_dense_rb <- system.time({
  res_cppode_dense_rb <- solveODE(funcpp_dense_rb, times, params)
})
cat(sprintf("  Time: %.3f s\n", t_cppode_dense_rb["elapsed"]))
CppODE::diagnostics(res_cppode_dense_rb)

# deSolve direct call (variational sensitivities)
cat("\n--- deSolve::ode lsodes (variational sensitivities) ---\n")
argsdeSolve <- list(
  y = yini_code_sens[attr(func_sens, "variables")],
  times = times,
  func = paste0(attr(func_sens, "modelname"), "_derivs"),
  parms = c(parsC, rep(0, length(yini_code_sens))),
  dllname = attr(func_sens, "modelname"),
  initfunc = paste0(attr(func_sens, "modelname"), "_initmod"),
  method = "vode"
)
t_deSolve <- system.time({
  res_deSolve_sparse <- do.call(deSolve::ode, argsdeSolve)
})
cat(sprintf("  Time: %.3f s\n", t_deSolve["elapsed"]))
deSolve::diagnostics(res_deSolve_sparse)


# ============================================================================
# Comparison plots: x1 trajectory + x1.mu1 sensitivity
# ============================================================================

cat("\n\n========== PLOTTING ==========\n\n")

df_plots <- list()

# CppODE no-sens baselines
for (info in list(
  list(res = res_nosens_sparse, lab = "CppODE sparse (no sens)"),
  list(res = res_nosens_dense,  lab = "CppODE dense (no sens)")
)) {
  res <- info$res
  if (!is.null(res) && length(res$time) > 1) {
    df_plots <- c(df_plots, list(data.frame(
      time = res$time, x1 = res$variable[, 1], solver = info$lab
    )))
  }
}

# CppODE with AD sensitivities
for (info in list(
  list(res = res_cppode_sparse,    lab = "CppODE BDF sparse (AD)"),
  list(res = res_cppode_sparse_rb, lab = "CppODE RB4 sparse (AD)"),
  list(res = res_cppode_dense,     lab = "CppODE BDF dense (AD)"),
  list(res = res_cppode_dense_rb,  lab = "CppODE RB4 dense (AD)")
)) {
  res <- info$res
  if (!is.null(res) && length(res$time) > 1) {
    df_plots <- c(df_plots, list(data.frame(
      time = res$time, x1 = res$variable[, 1], solver = info$lab
    )))
  }
}

# cOde / deSolve
for (info in list(
  list(res = res_cOde_sens,      lab = "cOde+lsodes (var)"),
  list(res = res_deSolve_sparse, lab = "deSolve lsodes (var)")
)) {
  res <- info$res
  if (!is.null(res) && nrow(res) > 1) {
    df <- as.data.frame(res)
    if ("x1" %in% colnames(df)) {
      df_plots <- c(df_plots, list(data.frame(
        time = df$time, x1 = df$x1, solver = info$lab
      )))
    }
  }
}

# --- Plot 1: x1 trajectory ---
if (length(df_plots) > 0) {
  all_x1 <- bind_rows(df_plots)
  p1 <- ggplot(all_x1, aes(x = time, y = x1, color = solver, linetype = solver)) +
    geom_line(alpha = 0.8, linewidth = 0.6) +
    labs(title = "x1(t) -- state trajectory", y = "x1", x = "time") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom", legend.title = element_blank())
} else {
  p1 <- ggplot() + labs(title = "No x1 data available")
}

# --- Extract x1.mu1 sensitivity ---
df_sens <- list()

# CppODE AD results
for (info in list(
  list(res = res_cppode_sparse,    lab = "CppODE BDF sparse (AD)"),
  list(res = res_cppode_sparse_rb, lab = "CppODE RB4 sparse (AD)"),
  list(res = res_cppode_dense,     lab = "CppODE BDF dense (AD)"),
  list(res = res_cppode_dense_rb,  lab = "CppODE RB4 dense (AD)")
)) {
  res <- info$res
  if (!is.null(res) && length(res$time) > 1 && !is.null(res$sens1)) {
    dn <- attr(res, "dim_names")
    if (!is.null(dn) && "sens" %in% names(dn)) {
      sens_names <- dn$sens
      mu1_idx <- which(sens_names == "mu1")
      if (length(mu1_idx) == 1) {
        s <- res$sens1[, 1, mu1_idx]
        df_sens <- c(df_sens, list(data.frame(
          time = res$time, sens = s, solver = info$lab
        )))
        cat(sprintf("  %s: found dx1/dmu1 (sens index %d)\n", info$lab, mu1_idx))
      }
    }
  }
}

# cOde / deSolve: sensitivity column "x1.mu1"
for (info in list(
  list(res = res_cOde_sens,      lab = "cOde+lsodes (var)"),
  list(res = res_deSolve_sparse, lab = "deSolve lsodes (var)")
)) {
  res <- info$res
  if (!is.null(res) && nrow(res) > 1) {
    df <- as.data.frame(res)
    if ("x1.mu1" %in% colnames(df)) {
      df_sens <- c(df_sens, list(data.frame(
        time = df$time, sens = df$x1.mu1, solver = info$lab
      )))
      cat(sprintf("  %s: found x1.mu1 column\n", info$lab))
    }
  }
}

# --- Plot 2: dx1/dmu1 sensitivity ---
if (length(df_sens) > 0) {
  all_sens <- bind_rows(df_sens)
  p2 <- ggplot(all_sens, aes(x = time, y = sens, color = solver, linetype = solver)) +
    geom_line(alpha = 0.8, linewidth = 0.6) +
    labs(title = expression(dx[1]/dmu[1] ~ "-- sensitivity"),
         y = expression(dx[1]/dmu[1]), x = "time") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom", legend.title = element_blank())
} else {
  p2 <- ggplot() + labs(title = "No sensitivity data available") +
    theme_minimal()
  cat("  No sensitivity data found for plotting.\n")
}

# --- Combined plot ---
combined <- p1 / p2 +
  plot_annotation(
    title = sprintf("N=%d VdP, reach=%d (sparsity %.0f%%): x1 + dx1/dmu1",
                    N, reach, actual_sparsity)
  )
print(combined)

plot_file <- sprintf("vdp_N%d_r%d_comparison.png", N, reach)
ggsave(plot_file, combined, width = 12, height = 8, dpi = 150)
cat(sprintf("  Plot saved: %s\n", plot_file))


# ============================================================================
# Summary table
# ============================================================================

cat("\n\n========== SUMMARY ==========\n\n")
cat(sprintf("N=%d, reach=%d, sparsity=%.1f%%, n_states=%d\n\n",
            N, reach, actual_sparsity, n_states))

fmt <- "%-40s %10s %8s %8s %8s\n"
cat(sprintf(fmt, "Solver", "Time (s)", "Steps", "Reject", "f-eval"))
cat(paste(rep("-", 72), collapse = ""), "\n")

print_cppode_row <- function(label, elapsed, res) {
  da <- attr(res, "diagnostics")
  if (!is.null(da)) {
    cat(sprintf("%-40s %10.3f %8d %8d %8d\n",
                label, elapsed,
                da$total_steps, da$rejected_steps,
                da$function_evaluations))
  } else {
    cat(sprintf("%-40s %10.3f %8s\n", label, elapsed, "(no diag)"))
  }
}

print_desolve_row <- function(label, elapsed, res) {
  d <- deSolve::diagnostics(res)
  cat(sprintf("%-40s %10.3f %8d %8s %8d\n",
              label, elapsed, d$istate[2], "-", d$istate[3]))
}

cat("\n--- No sensitivities ---\n")
print_desolve_row("cOde + lsodes (no sens)",
                  t_lsodes_nosens["elapsed"], res_cOde_nosens)
print_cppode_row("CppODE BDF sparse (no sens)",
                 t_nosens_sparse["elapsed"], res_nosens_sparse)
print_cppode_row("CppODE BDF dense (no sens)",
                 t_nosens_dense["elapsed"], res_nosens_dense)

cat("\n--- With sensitivities ---\n")
print_desolve_row("cOde + lsodes (var sens)",
                  t_lsodes_sens["elapsed"], res_cOde_sens)
print_desolve_row("deSolve lsodes (var sens)",
                  t_deSolve["elapsed"], res_deSolve_sparse)
print_cppode_row("CppODE BDF sparse (AD)",
                 t_cppode_sparse["elapsed"], res_cppode_sparse)
print_cppode_row("CppODE RB4 sparse (AD)",
                 t_cppode_sparse_rb["elapsed"], res_cppode_sparse_rb)
print_cppode_row("CppODE BDF dense (AD)",
                 t_cppode_dense["elapsed"], res_cppode_dense)
print_cppode_row("CppODE RB4 dense (AD)",
                 t_cppode_dense_rb["elapsed"], res_cppode_dense_rb)
