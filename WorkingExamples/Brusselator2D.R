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
# 2D Brusselator (Method of Lines) — Turing pattern formation
#
# u_t = A + u^2*v - (B+1)*u + Du * laplacian(u)
# v_t = B*u - u^2*v         + Dv * laplacian(v)
#
# Discretized on an Nx x Ny grid with periodic boundary conditions.
# 5-point stencil for the Laplacian: (u_{i-1,j} + u_{i+1,j} + u_{i,j-1}
#   + u_{i,j+1} - 4*u_{i,j}) / dx^2
# ============================================================================

Nx <- 32L
Ny <- 32L
n_cells <- Nx * Ny
n_states <- 2L * n_cells
dx <- 1.0 / Nx

cat(sprintf("Grid: %d x %d = %d cells, %d states\n", Nx, Ny, n_cells, n_states))

# Helper: map (ix, iy) to linear index (1-based), with periodic BC
idx <- function(ix, iy) {
  ((iy - 1L) %% Ny) * Nx + ((ix - 1L) %% Nx) + 1L
}

# State names: u_001 .. u_1024, v_001 .. v_1024
u_names <- sprintf("u_%03d", seq_len(n_cells))
v_names <- sprintf("v_%03d", seq_len(n_cells))

# Build RHS as named character vector
rhs <- character(n_states)
names(rhs) <- c(u_names, v_names)

for (iy in seq_len(Ny)) {
  for (ix in seq_len(Nx)) {
    k <- idx(ix, iy)
    u_k  <- u_names[k]
    v_k  <- v_names[k]

    # Neighbor indices (periodic)
    u_l <- u_names[idx(ix - 1L, iy)]
    u_r <- u_names[idx(ix + 1L, iy)]
    u_d <- u_names[idx(ix, iy - 1L)]
    u_u <- u_names[idx(ix, iy + 1L)]

    v_l <- v_names[idx(ix - 1L, iy)]
    v_r <- v_names[idx(ix + 1L, iy)]
    v_d <- v_names[idx(ix, iy - 1L)]
    v_u <- v_names[idx(ix, iy + 1L)]

    # Laplacian terms (factor 1/dx^2 absorbed into diffusion parameters)
    lap_u <- sprintf("(%s + %s + %s + %s - 4*%s)", u_l, u_r, u_d, u_u, u_k)
    lap_v <- sprintf("(%s + %s + %s + %s - 4*%s)", v_l, v_r, v_d, v_u, v_k)

    # u_t = A + u^2*v - (B+1)*u + Du * lap_u / dx^2
    rhs[u_k] <- sprintf(
      "A + %s^2 * %s - (B + 1) * %s + Du * %s * inv_dx2",
      u_k, v_k, u_k, lap_u
    )

    # v_t = B*u - u^2*v + Dv * lap_v / dx^2
    rhs[v_k] <- sprintf(
      "B * %s - %s^2 * %s + Dv * %s * inv_dx2",
      u_k, u_k, v_k, lap_v
    )
  }
}

# ============================================================================
# Compile models
# ============================================================================

model_sparse <- CppODE(
  rhs,
  outdir    = getwd(),
  modelname = "bruss2d_cppode_sparse",
  compile   = TRUE,
  deriv     = FALSE,
  sparse    = TRUE,
  verbose   = TRUE
)

model_dense <- CppODE(
  rhs,
  outdir    = getwd(),
  modelname = "bruss2d_cppode_dense",
  compile   = TRUE,
  deriv     = FALSE,
  sparse    = FALSE
)

func <- funC(rhs, modelname = "bruss2d_cOde", compile = T)
# ============================================================================
# Initial conditions & parameters
# ============================================================================

# Turing instability parameters (classic choice)
A_val  <- 1.0
B_val  <- 3.0
Du_val <- 0.01
Dv_val <- 0.1

# Homogeneous steady state: u* = A, v* = B/A
# Perturb with small random noise to seed pattern formation
set.seed(42)
u_init <- A_val + 0.1 * rnorm(n_cells)
v_init <- B_val / A_val + 0.1 * rnorm(n_cells)

params <- c(
  setNames(u_init, u_names),
  setNames(v_init, v_names),
  A      = A_val,
  B      = B_val,
  Du     = Du_val,
  Dv     = Dv_val,
  inv_dx2 = 1.0 / dx^2
)

yini <- params[c(u_names, v_names)]
parsC <- c(
  A      = A_val,
  B      = B_val,
  Du     = Du_val,
  Dv     = Dv_val,
  inv_dx2 = 1.0 / dx^2
)

# ============================================================================
# Solve & benchmark
# ============================================================================

times <- seq(0, 5, length.out = 200)
system.time({res_cOde_dense <- odeC(yini, times, func, parsC, method = "lsode")})
system.time({res_cOde_sparse <- odeC(yini, times, func, parsC, method = "lsodes")})
# system.time({res_dense <- solveODE(model_dense, times, params)})
system.time({res_sparse <- solveODE(model_sparse, times, params)})





# ============================================================================
# Verify agreement between solvers
# ============================================================================

check_names <- u_names[1:20]
cpp_final  <- res_sparse$variable[check_names, length(times)]
code_final <- res_code[nrow(res_code), check_names]

cat("\nMax absolute difference (CppODE sparse vs cOde) at t_end:\n")
cat(sprintf("  %.2e\n", max(abs(cpp_final - code_final))))

# ============================================================================
# Plotting: Turing pattern snapshots
# ============================================================================

# Extract u field at selected time points as 2D matrices
snap_times <- c(0.0, 0.5, 1.0, 2.0, 5.0)
snap_idx   <- sapply(snap_times, function(t) which.min(abs(res_sparse$time - t)))

df_snap <- do.call(rbind, lapply(seq_along(snap_times), function(s) {
  u_vec <- res_sparse$variable[u_names, snap_idx[s]]
  expand.grid(ix = seq_len(Nx), iy = seq_len(Ny)) |>
    mutate(
      u    = u_vec[idx(ix, iy)],
      time = factor(sprintf("t = %.1f", snap_times[s]),
                    levels = sprintf("t = %.1f", snap_times))
    )
}))

# Heatmap of u field over time
p_pattern <- ggplot(df_snap, aes(ix, iy, fill = u)) +
  geom_raster() +
  facet_wrap(~ time, nrow = 1) +
  scale_fill_viridis_c(option = "C", name = "u") +
  coord_equal() +
  labs(
    title = sprintf("2D Brusselator Turing Patterns (%d x %d grid)", Nx, Ny),
    x = "x", y = "y"
  ) +
  dMod::theme_dMod() +
  theme(
    strip.text   = element_text(size = 10),
    axis.text    = element_blank(),
    axis.ticks   = element_blank(),
    panel.grid   = element_blank()
  )

# Time series of u at selected grid points
probe_cells <- c(
  idx(Nx %/% 4, Ny %/% 4),
  idx(Nx %/% 2, Ny %/% 2),
  idx(3 * Nx %/% 4, Ny %/% 2),
  idx(Nx %/% 2, 3 * Ny %/% 4)
)
probe_labels <- sprintf("(%d,%d)",
                        (probe_cells - 1L) %% Nx + 1L,
                        (probe_cells - 1L) %/% Nx + 1L
)

df_ts <- do.call(rbind, lapply(seq_along(probe_cells), function(p) {
  k <- probe_cells[p]
  data.frame(
    time  = res_sparse$time,
    u     = res_sparse$variable[u_names[k], ],
    v     = res_sparse$variable[v_names[k], ],
    probe = factor(probe_labels[p], levels = probe_labels)
  )
}))

p_u <- ggplot(df_ts, aes(time, u, colour = probe)) +
  geom_line(linewidth = 0.5) +
  facet_grid(probe ~ ., scales = "free_y") +
  labs(x = "Time", y = "u", title = "u(t) at probes") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  theme(legend.position = "none",
        strip.text.y = element_text(size = 7))

p_v <- ggplot(df_ts, aes(time, v, colour = probe)) +
  geom_line(linewidth = 0.5) +
  facet_grid(probe ~ ., scales = "free_y") +
  labs(x = "Time", y = "v", title = "v(t) at probes") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  theme(legend.position = "none",
        strip.text.y = element_blank())

p_phase <- ggplot(df_ts, aes(u, v, colour = probe)) +
  geom_path(linewidth = 0.3, alpha = 0.7) +
  facet_grid(probe ~ ., scales = "free") +
  labs(x = "u", y = "v", title = "Phase portrait") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  theme(legend.position = "none",
        strip.text.y = element_blank())

# Assemble: pattern snapshots on top, time series below
p_pattern /
  (p_u + p_v + p_phase + plot_layout(ncol = 3)) +
  plot_layout(heights = c(1, 2)) +
  plot_annotation(
    title = sprintf("2D Brusselator (N = %d x %d, %d states)", Nx, Ny, n_states)
  )
