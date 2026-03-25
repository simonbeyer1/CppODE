rm(list = ls(all.names = TRUE))

# Create and set working directory
.workingDir <- file.path(
  dirname(rstudioapi::getSourceEditorContext()$path),
  "wd"
)
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(cOde)
library(deSolve)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

# ============================================================================
# Coupled Van der Pol Oscillators — moderately stiff, scalable
#
#   dx_i/dt = v_i
#   dv_i/dt = mu*(1 - x_i^2)*v_i - omega^2*x_i
#             + kappa * (x_{i-1} - 2*x_i + x_{i+1})     [nearest neighbor]
#             + gamma * (x_{i-2} - 2*x_i + x_{i+2})     [2nd neighbor]
#             + delta * (x_{i-10} - 2*x_i + x_{i+10})   [long-range, stride 10]
#
# N oscillators => 2*N states.
# Non-periodic boundary (coupling terms omitted at edges).
#
# - Stiffness controlled via mu (mu=1 mild, mu=10 moderate, mu=1000 very stiff)
# - Jacobian has extended band structure (bandwidth ~20 from stride-10 coupling)
# - Genuinely nonlinear (x^2 * v term)
# - Oscillatory dynamics — solver must track phase accurately
# - Scalable: just change N
# ============================================================================

N <- 200L
n_states <- 2L * N

cat(sprintf("System: %d coupled Van der Pol oscillators, %d states\n", N, n_states))

# Build RHS
rhs <- character(n_states)
names(rhs) <- character(n_states)

for (i in seq_len(N)) {
  xi <- paste0("x", i)
  vi <- paste0("v", i)

  # dx_i/dt = v_i
  names(rhs)[2L * i - 1L] <- xi
  rhs[2L * i - 1L] <- vi

  # Nearest-neighbor coupling (Laplacian-like)
  nn <- c()
  if (i > 1L) nn <- c(nn, paste0("x", i - 1L))
  nn <- c(nn, paste0("-2*", xi))
  if (i < N) nn <- c(nn, paste0("x", i + 1L))
  coupling_nn <- paste(nn, collapse = " + ")

  # Second-neighbor coupling
  sn <- c()
  if (i > 2L) sn <- c(sn, paste0("x", i - 2L))
  sn <- c(sn, paste0("-2*", xi))
  if (i < N - 1L) sn <- c(sn, paste0("x", i + 2L))
  coupling_sn <- paste(sn, collapse = " + ")

  # Long-range coupling (stride 10)
  tn <- c()
  if (i > 10L) tn <- c(tn, paste0("x", i - 10L))
  tn <- c(tn, paste0("-2*", xi))
  if (i < N - 9L) tn <- c(tn, paste0("x", i + 10L))
  coupling_tn <- paste(tn, collapse = " + ")

  # dv_i/dt
  names(rhs)[2L * i] <- vi
  rhs[2L * i] <- sprintf(
    "mu*(1 - %s^2)*%s - omega^2*%s + kappa*(%s) + gamma*(%s) + delta*(%s)",
    xi, vi, xi,
    coupling_nn,
    coupling_sn,
    coupling_tn
  )
}


# ============================================================================
# Compile models
# ============================================================================

# cOde model (for lsode/lsodes/vode/bdf via deSolve)
func <- cOde::funC(rhs, modelname = sprintf("vdp%d_code", N), jacobian = "full", compile = TRUE)
func_lsodes <- cOde::funC(rhs, modelname = sprintf("vdp%d_code_lsodes", N), jacobian = "inz.lsodes", compile = TRUE)

# CppODE models — dense and sparse LU
funcpp_dense  <- CppODE(rhs, outdir = getwd(),
                        modelname = sprintf("vdp%d_dense", N),
                        compile = TRUE, deriv = FALSE, sparse = FALSE,
                        verbose = FALSE)

funcpp_sparse <- CppODE(rhs, outdir = getwd(),
                        modelname = sprintf("vdp%d_sparse", N),
                        compile = TRUE, deriv = FALSE)


# ============================================================================
# Initial conditions & parameters
# ============================================================================
params <- c(
  setNames(sin(seq_len(N) * pi / (N + 1L)), paste0("x", seq_len(N))),
  setNames(rep(0, N), paste0("v", seq_len(N))),
  omega = 1.0,
  mu    = 4.0,
  kappa = 1.3,
  gamma = 0.25,
  delta = 0.1
)

state_names <- c(paste0("x", seq_len(N)), paste0("v", seq_len(N)))
yini  <- params[state_names]
parsC <- params[setdiff(names(params), state_names)]

times <- seq(0, 50, length.out = 3000)


# ============================================================================
# Benchmark
# ============================================================================

cat("--- cOde + lsode (dense, BDF variable order) ---\n")
t_lsode <- system.time({
  res_cOde_lsode <- cOde::odeC(yini, times, func, parsC, method = "lsode")
})
cat(sprintf("  Time: %.3f s\n", t_lsode["elapsed"]))

cat("\n--- cOde + lsodes (sparse, BDF variable order) ---\n")
t_lsodes <- system.time({
  res_cOde_lsodes <- cOde::odeC(yini, times, func, parsC, method = "lsodes")
})
cat(sprintf("  Time: %.3f s\n", t_lsodes["elapsed"]))

cat("\n--- cOde + vode (dense, BDF/Adams variable order) ---\n")
t_vode <- system.time({
  res_cOde_vode <- cOde::odeC(yini, times, func, parsC, method = "vode")
})
cat(sprintf("  Time: %.3f s\n", t_vode["elapsed"]))

cat("\n--- cOde + bdf (dense, BDF) ---\n")
t_bdf <- system.time({
  res_cOde_bdf <- cOde::odeC(yini, times, func, parsC, method = "bdf")
})
cat(sprintf("  Time: %.3f s\n", t_bdf["elapsed"]))

cat("\n--- CppODE Rosenbrock4 (dense LU) ---\n")
t_rb_dense <- system.time({
  res_cpp_dense <- solveODE(funcpp_dense, times, params)
})
cat(sprintf("  Time: %.3f s\n", t_rb_dense["elapsed"]))

cat("\n--- CppODE Rosenbrock4 (sparse LU) ---\n")
t_rb_sparse <- system.time({
  res_cpp_sparse <- solveODE(funcpp_sparse, times, params)
})
cat(sprintf("  Time: %.3f s\n", t_rb_sparse["elapsed"]))


# ============================================================================
# Summary table
# ============================================================================

cat("\n\n============ SUMMARY ============\n")
results <- data.frame(
  Method = c("cOde + lsode (dense BDF)",
             "cOde + lsodes (sparse BDF)",
             "cOde + vode (BDF/Adams)",
             "cOde + bdf (dense BDF)",
             "CppODE Rosenbrock4 (dense)",
             "CppODE Rosenbrock4 (sparse)"),
  Time_s = c(t_lsode["elapsed"],
             t_lsodes["elapsed"],
             t_vode["elapsed"],
             t_bdf["elapsed"],
             t_rb_dense["elapsed"],
             t_rb_sparse["elapsed"])
)
results$Ratio_vs_best <- round(results$Time_s / min(results$Time_s), 1)
print(results, row.names = FALSE)


# ============================================================================
# Verify agreement (CppODE sparse vs cOde vode at final time)
# ============================================================================

cat("\n--- Solution agreement check ---\n")
check_names <- paste0("x", c(1, 10, 50, 100, 256, 500))
check_names <- check_names[check_names %in% state_names]

cpp_final  <- res_cpp_sparse$variable[check_names, length(times)]
code_final <- res_cOde_vode[nrow(res_cOde_vode), check_names]

comp <- data.frame(
  State         = check_names,
  CppODE_sparse = as.numeric(cpp_final),
  cOde_vode     = as.numeric(code_final)
)
comp$AbsDiff <- abs(comp$CppODE_sparse - comp$cOde_vode)
print(comp, row.names = FALSE)
cat(sprintf("\nMax absolute difference: %.2e\n", max(comp$AbsDiff)))


# ============================================================================
# Visualization
# ============================================================================

idx <- c(1, 5, 10, 50, 100, 250, 500)
idx <- idx[idx <= N]

df_list <- lapply(idx, function(i) {
  xi <- paste0("x", i)
  vi <- paste0("v", i)
  data.frame(
    time = res_cpp_sparse$time,
    x    = res_cpp_sparse$variable[xi, ],
    v    = res_cpp_sparse$variable[vi, ],
    osc  = factor(paste0("osc ", i), levels = paste0("osc ", idx))
  )
})
df <- do.call(rbind, df_list)

# --- Displacement x(t) ---
p_x <- ggplot(df, aes(time, x, colour = osc)) +
  geom_line(linewidth = 0.4) +
  facet_grid(osc ~ ., scales = "free_y") +
  labs(x = "Time", y = "x", title = "Displacement") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  theme(legend.position = "none",
        strip.text.y = element_text(size = 7))

# --- Velocity v(t) ---
p_v <- ggplot(df, aes(time, v, colour = osc)) +
  geom_line(linewidth = 0.4) +
  facet_grid(osc ~ ., scales = "free_y") +
  labs(x = "Time", y = "v", title = "Velocity") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  theme(legend.position = "none",
        strip.text.y = element_blank())

# --- Phase portrait ---
p_xv <- ggplot(df, aes(x, v, colour = osc)) +
  geom_path(linewidth = 0.3, alpha = 0.7) +
  facet_grid(osc ~ ., scales = "free") +
  labs(x = "x", y = "v", title = "Phase portrait") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  theme(legend.position = "none",
        strip.text.y = element_blank())

# --- Benchmark barplot ---
results$Method <- factor(results$Method, levels = results$Method)
p_bench <- ggplot(results, aes(x = Method, y = Time_s, fill = Method)) +
  geom_col(show.legend = FALSE) +
  # geom_text(aes(label = sprintf("%.2fs", Time_s)), vjust = -0.3, size = 3.5) +
  labs(title = "Solver Benchmark", x = NULL, y = "Wall time (s)") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

# --- Assemble ---
p_dynamics <- p_x + p_v + p_xv + plot_layout(ncol = 3, widths = c(1, 1, 1))

p_combined <- p_dynamics / p_bench +
  plot_layout(heights = c(3, 1)) +
  plot_annotation(
    title = sprintf("Coupled Van der Pol Chain (N = %d, mu = %.1f)", N, params["mu"]),
    theme = theme(plot.title = element_text(size = 14, face = "bold"))
  )

print(p_combined)
