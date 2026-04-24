rm(list = ls(all.names = TRUE))
.workingDir <- file.path(tempdir(), "CppODE_example_brusselator_2d")
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

model_bdf <- CppODE(
  rhs,
  outdir    = getwd(),
  modelname = "bruss2d_bdf_sparse",
  method    = "bdf",
  compile   = FALSE,
  deriv     = FALSE,
  profile   = TRUE,
  stepTrace = FALSE,
  sparse    = TRUE
)

model_msoda <- CppODE(
  rhs,
  outdir    = getwd(),
  modelname = "bruss2d_msoda_sparse",
  method    = "msoda",
  compile   = FALSE,
  deriv     = FALSE,
  profile   = TRUE,
  stepTrace = FALSE,
  sparse    = TRUE
)

model_rb4 <- CppODE(
  rhs,
  outdir    = getwd(),
  modelname = "bruss2d_rb4_sparse",
  method    = "rb4",
  compile   = FALSE,
  deriv     = FALSE,
  profile   = TRUE,
  stepTrace = FALSE,
  sparse    = TRUE
)

model_bdf_dense <- CppODE(
  rhs,
  outdir    = getwd(),
  modelname = "bruss2d_bdf_dense",
  method    = "bdf",
  compile   = FALSE,
  deriv     = FALSE,
  profile   = TRUE,
  stepTrace = FALSE,
  sparse    = FALSE
)

model_msoda_dense <- CppODE(
  rhs,
  outdir    = getwd(),
  modelname = "bruss2d_msoda_dense",
  method    = "msoda",
  compile   = FALSE,
  deriv     = FALSE,
  profile   = TRUE,
  stepTrace = FALSE,
  sparse    = FALSE
)

model_rb4_dense <- CppODE(
  rhs,
  outdir    = getwd(),
  modelname = "bruss2d_rb4_dense",
  method    = "rb4",
  compile   = FALSE,
  deriv     = FALSE,
  profile   = TRUE,
  stepTrace = FALSE,
  sparse    = FALSE
)

CppODE:::compile(model_bdf, model_msoda, model_rb4, model_bdf_dense, model_msoda_dense, model_rb4_dense, cores = 6)

# rhs.sens <- sensitivitiesSymb(rhs)
func <- funC(c(rhs), modelname = "bruss2d_cOde", compile = T)
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

system.time({res_lsodes <- odeC(yini, times, func, parsC, method = "lsodes", atol = 1e-10, rtol = 1e-10)})
system.time({res_vode <- odeC(yini, times, func, parsC, method = "vode", atol = 1e-10, rtol = 1e-10)})
system.time({res_radau <- odeC(yini, times, func, parsC, method = "radau", atol = 1e-10, rtol = 1e-10)})
system.time({res_deSolvebdf <- odeC(yini, times, func, parsC, method = "bdf", atol = 1e-10, rtol = 1e-10)})
system.time({res_bdf <- solveODE(model_bdf, times, params, abstol = 1e-10, reltol = 1e-10)})
system.time({res_rb4 <- solveODE(model_rb4, times, params, abstol = 1e-10, reltol = 1e-10)})
system.time({res_msoda <- solveODE(model_msoda, times, params, abstol = 1e-10, reltol = 1e-10)})
system.time({res_bdf_dense <- solveODE(model_bdf_dense, times, params, abstol = 1e-10, reltol = 1e-10)})
system.time({res_rb4_dense <- solveODE(model_rb4_dense, times, params, abstol = 1e-10, reltol = 1e-10)})
system.time({res_msoda_dense <- solveODE(model_msoda_dense, times, params, abstol = 1e-10, reltol = 1e-10)})

CppODE::diagnostics(res_bdf)
CppODE::diagnostics(res_msoda)
deSolve::diagnostics(res_lsodes)
deSolve::diagnostics(res_vode)

resBDF <- res_bdf$variable %>% t()
resMSODA <- res_msoda$variable %>% t()
resRB <- res_rb4$variable %>% t()
resBDFD <- res_bdf_dense$variable %>% t()
resMSODAD <- res_msoda_dense$variable %>% t()
resRBD <- res_rb4_dense$variable %>% t()
resLSODES <- res_lsodes[,colnames(resBDF)]
resVODE <- res_vode[,colnames(resBDF)]
resRADAU <- res_radau[,colnames(resBDF)]
resDSBDF <- res_deSolvebdf[,colnames(resBDF)]

norm(resRADAU - resLSODES, type = "I")
norm(resDSBDF - resLSODES, type = "I")
norm(resBDF - resLSODES, type = "I")
norm(resMSODA - resLSODES, type = "I")
norm(resRB - resLSODES, type = "I")
norm(resBDFD - resLSODES, type = "I")
norm(resMSODAD - resLSODES, type = "I")
norm(resRBD - resLSODES, type = "I")

norm(resBDFD - resBDF, type = "I")
norm(resMSODAD - resMSODA, type = "I")
norm(resRBD - resRB, type = "I")
