## =================================================================
## Stiff solver benchmark suite
##
## Compares CppODE (bdf, ndf, rb4) against CVODE (BDF) -- both with
## compiled C++ RHS + analytic Jacobian, so the comparison is
## apples-to-apples (no R callback overhead either side).
##
## Each (problem, solver, tol) triple reports:
##   - accepted steps
##   - rejected steps
##   - rhs evaluations
##   - wallclock (median over nrep)
##   - error vs high-accuracy reference (CVODE at atol=1e-14, rtol=1e-12)
##
## At the end: geometric-mean fevals & wallclock ratio vs CVODE_bdf,
## aggregated across problems and tolerances.
## =================================================================
rm(list = ls(all.names = TRUE))

.workingDir <- file.path(tempdir(), "CppODE_bench_stiff_suite")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)

NREP <- 3L   # benchmark repetitions
TOLS <- list(
  loose = list(atol = 1e-8,  rtol = 1e-6),
  tight = list(atol = 1e-10, rtol = 1e-8)
)

# --- Harness ---------------------------------------------------------

bench <- function(fn, nrep = NREP) {
  fn()  # warmup
  t <- replicate(nrep, system.time(fn())[["elapsed"]])
  list(median = median(t), min = min(t), max = max(t))
}

fmt5 <- function(x) if (is.na(x)) "   NA" else formatC(as.integer(x), width = 5)
fmt6 <- function(x) if (is.na(x)) "    NA" else formatC(as.integer(x), width = 6)

safe <- function(expr, label) {
  tryCatch(expr, error = function(e) {
    cat(sprintf("    [%s] FAILED: %s\n", label, conditionMessage(e)))
    NULL
  })
}

# --- Results accumulator ---------------------------------------------

results <- list()
record <- function(problem, solver, tol, npts, timing, diag, err) {
  results[[length(results) + 1L]] <<- data.frame(
    problem = problem, solver = solver, tol = tol, npts = npts,
    time_ms = if (!is.null(timing)) timing$median * 1000 else NA_real_,
    accepted = if (!is.null(diag)) diag$accepted else NA_integer_,
    rejected = if (!is.null(diag)) diag$rejected else NA_integer_,
    fevals   = if (!is.null(diag)) diag$fevals   else NA_integer_,
    err      = err,
    stringsAsFactors = FALSE
  )
}

# --- Per-problem runner ----------------------------------------------

run_problem <- function(prob) {
  cat(sprintf("\n=== %s (%d states) ===\n", prob$name, prob$nstate))

  # Compile CppODE models (bdf/ndf, and optionally rb4)
  cppode_models <- list()
  cat("  compiling CppODE models...\n")
  cppode_models$ndf <- CppODE(prob$rhs, deriv = FALSE, outdir = getwd(),
                              method = "bdf", sparse = prob$sparse,
                              modelname = paste0(prob$id, "_ndf"), compile = TRUE)
  cppode_models$bdf <- CppODE(prob$rhs, deriv = FALSE, outdir = getwd(),
                              method = "bdf", sparse = prob$sparse, useNDF = FALSE,
                              modelname = paste0(prob$id, "_bdf"), compile = TRUE)
  if (isTRUE(prob$include_rb4)) {
    cppode_models$rb4 <- CppODE(prob$rhs, deriv = FALSE, outdir = getwd(),
                                method = "rb4", sparse = prob$sparse,
                                modelname = paste0(prob$id, "_rb4"), compile = TRUE)
  }

  # Compile CVODE (BDF) -- the external reference implementation.
  cat("  compiling CVODE model...\n")
  cvode_models <- list()
  cvode_models$bdf <- safe(
    CVODE(prob$rhs, deriv = FALSE, outdir = getwd(),
          method = "bdf", sparse = prob$sparse,
          modelname = paste0(prob$id, "_cvode_bdf"), compile = TRUE),
    "CVODE_bdf compile")

  # Reference solution -- CVODE at very tight tolerance (no deSolve).
  cat("  computing reference solution (CVODE BDF, atol=1e-14, rtol=1e-12)...\n")
  ref_model <- safe(
    CVODE(prob$rhs, deriv = FALSE, outdir = getwd(),
          method = "bdf", sparse = prob$sparse,
          modelname = paste0(prob$id, "_cvode_ref"), compile = TRUE),
    "reference compile")
  if (is.null(ref_model)) { cat("  !!! reference compile failed -- skipping\n"); return() }
  ref_res <- safe(
    solveODE(ref_model, prob$times, prob$parms, abstol = 1e-14, reltol = 1e-12),
    "reference solve")
  if (is.null(ref_res)) { cat("  !!! reference solve failed -- skipping\n"); return() }
  ref_mat <- t(ref_res$variable[names(prob$y0), , drop = FALSE])

  err_of <- function(mat) {
    n <- min(nrow(mat), nrow(ref_mat))
    if (n < 2) return(NA_real_)
    max(abs(mat[seq_len(n), , drop = FALSE] -
            ref_mat[seq_len(n), , drop = FALSE]), na.rm = TRUE)
  }

  for (tol_name in names(TOLS)) {
    tol <- TOLS[[tol_name]]
    cat(sprintf("  -- %s (atol=%g, rtol=%g) --\n", tol_name, tol$atol, tol$rtol))

    for (sname in names(cppode_models)) {
      local({
        m <- cppode_models[[sname]]
        label <- paste0("CppODE_", sname)
        run_once <- function() solveODE(m, prob$times, prob$parms,
                                        abstol = tol$atol, reltol = tol$rtol)
        r <- safe(run_once(), label)
        if (is.null(r)) return(invisible())
        mat <- t(r$variable[names(prob$y0), , drop = FALSE])
        err <- err_of(mat)
        timing <- bench(run_once)
        record(prob$name, label, tol_name, ncol(r$variable), timing, r$diagnostics, err)
        cat(sprintf("    %-14s  %7.1f ms  acc=%5s rej=%5s fev=%6s err=%.2e\n",
                    label, timing$median * 1000,
                    fmt5(r$diagnostics$accepted),
                    fmt5(r$diagnostics$rejected),
                    fmt6(r$diagnostics$fevals),
                    err))
      })
    }

    for (sname in names(cvode_models)) {
      local({
        m <- cvode_models[[sname]]
        if (is.null(m)) return(invisible())
        label <- paste0("CVODE_", sname)
        run_once <- function() solveODE(m, prob$times, prob$parms,
                                        abstol = tol$atol, reltol = tol$rtol)
        r <- safe(run_once(), label)
        if (is.null(r)) return(invisible())
        mat <- t(r$variable[names(prob$y0), , drop = FALSE])
        err <- err_of(mat)
        timing <- bench(run_once)
        record(prob$name, label, tol_name, ncol(r$variable), timing, r$diagnostics, err)
        cat(sprintf("    %-14s  %7.1f ms  acc=%5s rej=%5s fev=%6s err=%.2e\n",
                    label, timing$median * 1000,
                    fmt5(r$diagnostics$accepted),
                    fmt5(r$diagnostics$rejected),
                    fmt6(r$diagnostics$fevals),
                    err))
      })
    }

  }
}

# =====================================================================
# Problem definitions
# =====================================================================

problems <- list()

# --- Robertson (3 states) --------------------------------------------
problems$robertson <- list(
  id = "rob", name = "Robertson", nstate = 3,
  rhs = c(
    y1 = "-k1*y1 + k2*y2*y3",
    y2 = " k1*y1 - k2*y2*y3 - k3*y2^2",
    y3 = " k3*y2^2"
  ),
  parms = c(y1 = 1, y2 = 0, y3 = 0, k1 = 0.04, k2 = 1e4, k3 = 3e7),
  y0 = c(y1 = 1, y2 = 0, y3 = 0),
  times = c(0, 10^seq(-5, 5, length.out = 200)),
  desolve_rhs = function(t, y, p) with(as.list(c(y, p)), list(c(
    -k1*y1 + k2*y2*y3,
     k1*y1 - k2*y2*y3 - k3*y2^2,
     k3*y2^2))),
  desolve_methods = c("lsoda", "vode", "bdf", "radau"),
  sparse = FALSE, include_rb4 = TRUE
)

# --- Van der Pol mu=1000 (2 states) ----------------------------------
problems$vdp <- list(
  id = "vdp", name = "VanderPol_mu1000", nstate = 2,
  rhs = c(x = "y", y = "mu * (1 - x^2) * y - x"),
  parms = c(x = 2, y = 0, mu = 1000),
  y0 = c(x = 2, y = 0),
  times = seq(0, 3000, length.out = 1000),
  desolve_rhs = function(t, y, p) with(as.list(c(y, p)), list(c(
    y, mu * (1 - x^2) * y - x))),
  desolve_methods = c("lsoda", "vode", "bdf"),
  sparse = FALSE, include_rb4 = TRUE
)

# --- HIRES (8 states) - High Irradiance RESponse ---------------------
# Hairer & Wanner, "Solving ODEs II", IVPtestset
problems$hires <- list(
  id = "hires", name = "HIRES", nstate = 8,
  rhs = c(
    y1 = "-1.71*y1 + 0.43*y2 + 8.32*y3 + 0.0007",
    y2 = " 1.71*y1 - 8.75*y2",
    y3 = "-10.03*y3 + 0.43*y4 + 0.035*y5",
    y4 = " 8.32*y2 + 1.71*y3 - 1.12*y4",
    y5 = "-1.745*y5 + 0.43*y6 + 0.43*y7",
    y6 = "-280.0*y6*y8 + 0.69*y4 + 1.71*y5 - 0.43*y6 + 0.69*y7",
    y7 = " 280.0*y6*y8 - 1.81*y7",
    y8 = "-280.0*y6*y8 + 1.81*y7"
  ),
  parms = c(y1=1, y2=0, y3=0, y4=0, y5=0, y6=0, y7=0, y8=0.0057),
  y0 = c(y1=1, y2=0, y3=0, y4=0, y5=0, y6=0, y7=0, y8=0.0057),
  times = seq(0, 321.8122, length.out = 200),
  desolve_rhs = function(t, y, p) list(c(
    -1.71*y[1] + 0.43*y[2] + 8.32*y[3] + 0.0007,
     1.71*y[1] - 8.75*y[2],
    -10.03*y[3] + 0.43*y[4] + 0.035*y[5],
     8.32*y[2] + 1.71*y[3] - 1.12*y[4],
    -1.745*y[5] + 0.43*y[6] + 0.43*y[7],
    -280.0*y[6]*y[8] + 0.69*y[4] + 1.71*y[5] - 0.43*y[6] + 0.69*y[7],
     280.0*y[6]*y[8] - 1.81*y[7],
    -280.0*y[6]*y[8] + 1.81*y[7])),
  desolve_methods = c("lsoda", "vode", "bdf", "radau"),
  sparse = FALSE, include_rb4 = TRUE
)

# --- OREGO (3 states) - Oregonator -----------------------------------
problems$orego <- list(
  id = "orego", name = "OREGO", nstate = 3,
  rhs = c(
    y1 = "77.27 * (y2 + y1*(1 - 8.375e-6*y1 - y2))",
    y2 = "(y3 - (1 + y1)*y2) / 77.27",
    y3 = "0.161 * (y1 - y3)"
  ),
  parms = c(y1 = 1, y2 = 2, y3 = 3),
  y0 = c(y1 = 1, y2 = 2, y3 = 3),
  times = seq(0, 360, length.out = 400),
  desolve_rhs = function(t, y, p) list(c(
    77.27 * (y[2] + y[1]*(1 - 8.375e-6*y[1] - y[2])),
    (y[3] - (1 + y[1])*y[2]) / 77.27,
    0.161 * (y[1] - y[3]))),
  desolve_methods = c("lsoda", "vode", "bdf", "radau"),
  sparse = FALSE, include_rb4 = TRUE
)

# --- E5 (4 states) - chemical kinetics, extreme scale separation ------
# Hairer/Wanner IVPtestset; A=7.89e-10, B=1.1e7, C=1.13e3, M=1e6
problems$e5 <- list(
  id = "e5", name = "E5", nstate = 4,
  rhs = c(
    y1 = "-A*y1 - B*y1*y3",
    y2 = " A*y1 - M*C*y2*y3",
    y3 = " A*y1 - B*y1*y3 - M*C*y2*y3 + C*y4",
    y4 = " B*y1*y3 - C*y4"
  ),
  parms = c(y1 = 1.76e-3, y2 = 0, y3 = 0, y4 = 0,
            A = 7.89e-10, B = 1.1e7, C = 1.13e3, M = 1e6),
  y0 = c(y1 = 1.76e-3, y2 = 0, y3 = 0, y4 = 0),
  times = c(0, 10^seq(-5, 11, length.out = 200)),
  desolve_rhs = function(t, y, p) with(as.list(p), list(c(
    -A*y[1] - B*y[1]*y[3],
     A*y[1] - M*C*y[2]*y[3],
     A*y[1] - B*y[1]*y[3] - M*C*y[2]*y[3] + C*y[4],
     B*y[1]*y[3] - C*y[4]))),
  desolve_methods = c("lsoda", "vode", "bdf", "radau"),
  sparse = FALSE, include_rb4 = TRUE
)

# --- Pollution (20 states) -- DETEST stiff chemistry --------------------
# Hairer & Wanner IVPtestset, 25 reactions of atmospheric chemistry
problems$pollution <- local({
  # Rate constants (selected from Verwer 1994 formulation)
  k  <- c(.35, .266e2, .123e5, .86e-3, .82e-3, .15e5, .13e-3, .24e5, .165e5, .9e4,
          .22e-1, .12e5, .188e1, .163e5, .48e7, .35e-3, .175e-1, .1e9, .444e12, .124e4,
          .21e1, .578e1, .474e-1, .178e4, .312e1)
  # Initial conditions from literature (species 1..20)
  y0 <- c(y1=0, y2=.2, y3=0, y4=.04, y5=0, y6=0, y7=.1, y8=.3, y9=.01, y10=0,
          y11=0, y12=0, y13=0, y14=0, y15=0, y16=0, y17=.007, y18=0, y19=0, y20=0)
  # 25 reaction terms r1..r25
  r <- c(
    "k1*y1",           "k2*y2*y4",         "k3*y5*y2",      "k4*y7",         "k5*y7",
    "k6*y7*y6",        "k7*y9",            "k8*y9*y6",      "k9*y11*y2",     "k10*y11*y1",
    "k11*y13",         "k12*y10*y2",       "k13*y14",       "k14*y1*y6",     "k15*y3",
    "k16*y4",          "k17*y4",           "k18*y16",       "k19*y16",       "k20*y17*y6",
    "k21*y19",         "k22*y19",          "k23*y1*y4",     "k24*y19*y1",    "k25*y20"
  )
  # For CppODE we need symbolic rhs. Use the named vector approach.
  rhs <- c(
    y1  = "-k1*y1 - k10*y11*y1 - k14*y1*y6 - k23*y1*y4 - k24*y19*y1 + k2*y2*y4 + k3*y5*y2 + k9*y11*y2 + k11*y13 + k12*y10*y2 + k22*y19 + k25*y20",
    y2  = "-k2*y2*y4 - k3*y5*y2 - k9*y11*y2 - k12*y10*y2 + k1*y1 + k21*y19",
    y3  = "-k15*y3 + k1*y1 + k17*y4 + k19*y16 + k22*y19",
    y4  = "-k2*y2*y4 - k16*y4 - k17*y4 - k23*y1*y4 + k15*y3",
    y5  = "-k3*y5*y2 + 2*k4*y7 + k6*y7*y6 + k7*y9 + k13*y14 + k20*y17*y6",
    y6  = "-k6*y7*y6 - k8*y9*y6 - k14*y1*y6 - k20*y17*y6 + k3*y5*y2 + 2*k18*y16",
    y7  = "-k4*y7 - k5*y7 - k6*y7*y6 + k13*y14",
    y8  = "k4*y7 + k5*y7 + k6*y7*y6 + k7*y9",
    y9  = "-k7*y9 - k8*y9*y6",
    y10 = "-k12*y10*y2 + k7*y9 + k9*y11*y2",
    y11 = "-k9*y11*y2 - k10*y11*y1 + k8*y9*y6 + k11*y13",
    y12 = "k9*y11*y2",
    y13 = "-k11*y13 + k10*y11*y1",
    y14 = "-k13*y14 + k12*y10*y2",
    y15 = "k14*y1*y6",
    y16 = "-k18*y16 - k19*y16 + k16*y4",
    y17 = "-k20*y17*y6",
    y18 = "k20*y17*y6",
    y19 = "-k21*y19 - k22*y19 - k24*y19*y1 + k23*y1*y4 + k25*y20",
    y20 = "-k25*y20 + k24*y19*y1"
  )
  parms <- c(y0, setNames(k, paste0("k", seq_along(k))))
  ds_rhs <- function(t, y, p) {
    kk <- p[paste0("k", 1:25)]
    rr <- with(as.list(c(y, kk)), c(
      k1*y1, k2*y2*y4, k3*y5*y2, k4*y7, k5*y7,
      k6*y7*y6, k7*y9, k8*y9*y6, k9*y11*y2, k10*y11*y1,
      k11*y13, k12*y10*y2, k13*y14, k14*y1*y6, k15*y3,
      k16*y4, k17*y4, k18*y16, k19*y16, k20*y17*y6,
      k21*y19, k22*y19, k23*y1*y4, k24*y19*y1, k25*y20))
    list(c(
      -rr[1]-rr[10]-rr[14]-rr[23]-rr[24]+rr[2]+rr[3]+rr[9]+rr[11]+rr[12]+rr[22]+rr[25],
      -rr[2]-rr[3]-rr[9]-rr[12]+rr[1]+rr[21],
      -rr[15]+rr[1]+rr[17]+rr[19]+rr[22],
      -rr[2]-rr[16]-rr[17]-rr[23]+rr[15],
      -rr[3]+2*rr[4]+rr[6]+rr[7]+rr[13]+rr[20],
      -rr[6]-rr[8]-rr[14]-rr[20]+rr[3]+2*rr[18],
      -rr[4]-rr[5]-rr[6]+rr[13],
      rr[4]+rr[5]+rr[6]+rr[7],
      -rr[7]-rr[8],
      -rr[12]+rr[7]+rr[9],
      -rr[9]-rr[10]+rr[8]+rr[11],
      rr[9],
      -rr[11]+rr[10],
      -rr[13]+rr[12],
      rr[14],
      -rr[18]-rr[19]+rr[16],
      -rr[20],
      rr[20],
      -rr[21]-rr[22]-rr[24]+rr[23]+rr[25],
      -rr[25]+rr[24]
    ))
  }
  list(
    id = "poll", name = "Pollution", nstate = 20,
    rhs = rhs, parms = parms, y0 = y0,
    times = seq(0, 60, length.out = 200),
    desolve_rhs = ds_rhs,
    desolve_methods = c("lsoda", "vode", "bdf", "radau"),
    sparse = FALSE, include_rb4 = TRUE
  )
})

# =====================================================================
# Brusselator 2D MOL -- parameterized by grid size
# =====================================================================
build_bruss2d <- function(N, id, name, include_rb4 = TRUE) {
  Nx <- Ny <- as.integer(N)
  n_cells <- Nx * Ny
  dx <- 1.0 / Nx

  idx <- function(ix, iy) ((iy - 1L) %% Ny) * Nx + ((ix - 1L) %% Nx) + 1L
  u_names <- sprintf("u_%04d", seq_len(n_cells))
  v_names <- sprintf("v_%04d", seq_len(n_cells))

  rhs <- character(2L * n_cells)
  names(rhs) <- c(u_names, v_names)
  for (iy in seq_len(Ny)) for (ix in seq_len(Nx)) {
    k <- idx(ix, iy)
    u_k <- u_names[k]; v_k <- v_names[k]
    u_l <- u_names[idx(ix - 1L, iy)]; u_r <- u_names[idx(ix + 1L, iy)]
    u_d <- u_names[idx(ix, iy - 1L)]; u_u <- u_names[idx(ix, iy + 1L)]
    v_l <- v_names[idx(ix - 1L, iy)]; v_r <- v_names[idx(ix + 1L, iy)]
    v_d <- v_names[idx(ix, iy - 1L)]; v_u <- v_names[idx(ix, iy + 1L)]
    lap_u <- sprintf("(%s + %s + %s + %s - 4*%s)", u_l, u_r, u_d, u_u, u_k)
    lap_v <- sprintf("(%s + %s + %s + %s - 4*%s)", v_l, v_r, v_d, v_u, v_k)
    rhs[u_k] <- sprintf("A + %s^2 * %s - (B + 1) * %s + Du * %s * inv_dx2",
                        u_k, v_k, u_k, lap_u)
    rhs[v_k] <- sprintf("B * %s - %s^2 * %s + Dv * %s * inv_dx2",
                        u_k, u_k, v_k, lap_v)
  }

  set.seed(42)
  u0 <- 1.0 + 0.1 * rnorm(n_cells)
  v0 <- 3.0 + 0.1 * rnorm(n_cells)
  parms <- c(setNames(u0, u_names), setNames(v0, v_names),
             A = 1.0, B = 3.0, Du = 0.01, Dv = 0.1, inv_dx2 = 1.0 / dx^2)
  y0 <- c(setNames(u0, u_names), setNames(v0, v_names))

  # deSolve function that operates on indices (fast)
  A <- 1.0; B <- 3.0; Du <- 0.01; Dv <- 0.1; inv_dx2 <- 1.0 / dx^2
  nc <- n_cells
  # Precompute neighbor indices for vectorized Laplacian
  i1 <- seq_len(nc); row <- ((i1 - 1L) %/% Nx); col <- ((i1 - 1L) %% Nx)
  left  <- row * Nx + ((col - 1L) %% Nx) + 1L
  right <- row * Nx + ((col + 1L) %% Nx) + 1L
  down  <- ((row - 1L) %% Ny) * Nx + col + 1L
  up    <- ((row + 1L) %% Ny) * Nx + col + 1L
  ds_rhs <- function(t, y, p) {
    u <- y[1:nc]; v <- y[(nc+1):(2*nc)]
    lap_u <- u[left] + u[right] + u[down] + u[up] - 4*u
    lap_v <- v[left] + v[right] + v[down] + v[up] - 4*v
    du <- A + u^2 * v - (B + 1) * u + Du * lap_u * inv_dx2
    dv <- B * u - u^2 * v + Dv * lap_v * inv_dx2
    list(c(du, dv))
  }

  list(
    id = id, name = name, nstate = 2 * n_cells,
    rhs = rhs, parms = parms, y0 = y0,
    times = seq(0, 5, length.out = 50),
    desolve_rhs = ds_rhs,
    desolve_methods = c("lsoda"),   # vode/radau/bdf too slow at this size
    sparse = TRUE, include_rb4 = include_rb4
  )
}

# --- Medium Brusselator: 20x20 = 800 states --------------------------
problems$bruss_med <- build_bruss2d(20L, "bruss20", "Brusselator2D_20x20")

# --- Large Brusselator: 63x63 = 7938 states (~=8000) ------------------
# problems$bruss_large <- build_bruss2d(63L, "bruss63", "Brusselator2D_63x63",
#                                       include_rb4 = FALSE)

# =====================================================================
# HVAC-like thermal RC network (~8000 states)
#
# Grid of N_zones zones arranged linearly ("hotel corridor").
# Each zone has:
#   - air temperature T_air (1 state)
#   - wall temperatures to neighbors: N_layers per wall (multi-layer wall)
# With N_zones = 200, N_layers = 20: 200 + 200*19 = 4000 states
# Scale up: N_zones = 400, N_layers = 20 -> ~8000 states
#
# Dynamics:
#   C_air * dT_air_i/dt  = U_in * (T_wall_i_first_layer - T_air_i)
#                        + h_i * (T_set_i - T_air_i)   (HVAC proportional)
#                        + q_int_i  (internal gain)
#   C_w * dT_wall_i,j/dt = k*(T_wall_i,j-1 - 2*T_wall_i,j + T_wall_i,j+1)
#   boundary: T_wall_i,1 ~ T_air_i, T_wall_i,L ~ T_air_{i+1}
# =====================================================================
build_hvac <- function(N_zones = 400L, N_layers = 20L) {
  N_zones <- as.integer(N_zones); N_layers <- as.integer(N_layers)
  stopifnot(N_zones >= 2L, N_layers >= 2L)

  # State names: T_1..T_N, W_i_j for zone i wall-layer j (layers between i and i+1)
  T_names <- sprintf("T%04d", seq_len(N_zones))
  # Walls between zone i and i+1, for i=1..(N_zones-1), layers j=1..N_layers
  wall_pairs <- N_zones - 1L
  W_names <- character(wall_pairs * N_layers)
  k <- 1L
  for (i in seq_len(wall_pairs)) for (j in seq_len(N_layers)) {
    W_names[k] <- sprintf("W%04d_%02d", i, j); k <- k + 1L
  }

  rhs <- character(length(T_names) + length(W_names))
  names(rhs) <- c(T_names, W_names)

  # Zone air equations
  for (i in seq_len(N_zones)) {
    Ti <- T_names[i]
    # Left wall (between zone i-1 and i) -- use innermost layer adjacent to zone i
    # For wall between i-1 and i (wall index i-1), layer N_layers (adjacent to zone i)
    left_term <- if (i > 1L) {
      wLN <- sprintf("W%04d_%02d", i - 1L, N_layers)
      sprintf("U_in * (%s - %s)", wLN, Ti)
    } else ""
    # Right wall (between zone i and i+1), layer 1 (adjacent to zone i)
    right_term <- if (i < N_zones) {
      wR1 <- sprintf("W%04d_%02d", i, 1L)
      sprintf("U_in * (%s - %s)", wR1, Ti)
    } else ""
    # HVAC proportional control toward setpoint
    hvac_term <- sprintf("h_ctrl * (T_set - %s)", Ti)
    # Internal gain (piecewise could be a forcing -- here constant)
    gain_term <- "q_int"
    # Ambient loss at endpoints (to outside)
    amb_term <- if (i == 1L || i == N_zones)
      sprintf("U_out * (T_amb - %s)", Ti) else ""

    terms <- c(left_term, right_term, hvac_term, gain_term, amb_term)
    terms <- terms[nzchar(terms)]
    rhs[Ti] <- sprintf("(%s) / C_air", paste(terms, collapse = " + "))
  }

  # Wall layer equations: 1D heat diffusion along wall thickness
  for (i in seq_len(wall_pairs)) {
    for (j in seq_len(N_layers)) {
      Wij <- sprintf("W%04d_%02d", i, j)
      # Left neighbor: layer j-1, or zone i (if j==1)
      left <- if (j == 1L) T_names[i] else sprintf("W%04d_%02d", i, j - 1L)
      # Right neighbor: layer j+1, or zone i+1 (if j==N_layers)
      right <- if (j == N_layers) T_names[i + 1L] else sprintf("W%04d_%02d", i, j + 1L)
      rhs[Wij] <- sprintf("k_wall * (%s - 2*%s + %s) / C_wall",
                          left, Wij, right)
    }
  }

  # Initial conditions & parameters
  # Zones start at 20 degC except first zone hot (30) and last cold (10)
  T0 <- rep(20.0, N_zones); T0[1] <- 30.0; T0[N_zones] <- 10.0
  W0 <- rep(15.0, length(W_names))  # walls at intermediate temperature

  parms <- c(
    setNames(T0, T_names),
    setNames(W0, W_names),
    C_air  = 1200.0,   # J/K per zone air
    C_wall = 8000.0,   # J/K per wall layer (heavy)
    U_in   = 5.0,      # W/K zone-wall coupling
    U_out  = 2.0,      # W/K exterior loss at endpoints
    k_wall = 0.8,      # W/K inter-layer conduction
    h_ctrl = 10.0,     # W/K HVAC proportional gain
    T_set  = 22.0,     # setpoint
    T_amb  = 5.0,      # ambient
    q_int  = 200.0     # W internal gain per zone
  )
  y0 <- c(setNames(T0, T_names), setNames(W0, W_names))

  # deSolve function (fast vectorized)
  nT <- N_zones; nW <- wall_pairs * N_layers
  ds_rhs <- function(t, y, p) {
    with(as.list(p), {
      T <- y[1:nT]
      W <- matrix(y[(nT+1):(nT+nW)], nrow = N_layers, ncol = wall_pairs)
      # Wall-layer dynamics: 1D diffusion, boundary = adjacent zones
      dW <- matrix(0, N_layers, wall_pairs)
      for (i in seq_len(wall_pairs)) {
        left_bc  <- T[i]
        right_bc <- T[i + 1L]
        W_ext <- c(left_bc, W[, i], right_bc)
        dW[, i] <- k_wall * (W_ext[1:N_layers] - 2*W_ext[2:(N_layers+1)] +
                             W_ext[3:(N_layers+2)]) / C_wall
      }
      # Zone air dynamics
      dT <- numeric(N_zones)
      for (i in seq_len(N_zones)) {
        s <- 0.0
        if (i > 1L)        s <- s + U_in * (W[N_layers, i - 1L] - T[i])
        if (i < N_zones)   s <- s + U_in * (W[1L, i] - T[i])
        s <- s + h_ctrl * (T_set - T[i]) + q_int
        if (i == 1L || i == N_zones) s <- s + U_out * (T_amb - T[i])
        dT[i] <- s / C_air
      }
      list(c(dT, as.vector(dW)))
    })
  }

  list(
    id = sprintf("hvac_%d_%d", N_zones, N_layers),
    name = sprintf("HVAC_%dzones_%dlayers", N_zones, N_layers),
    nstate = N_zones + wall_pairs * N_layers,
    rhs = rhs, parms = parms, y0 = y0,
    times = seq(0, 3600 * 12, length.out = 40),  # 12 hours
    desolve_rhs = ds_rhs,
    desolve_methods = c("lsoda"),  # only lsoda; radau/vode with dense jac would be too slow
    sparse = TRUE, include_rb4 = FALSE
  )
}

# --- HVAC ~8000 states -----------------------------------------------
# problems$hvac_large <- build_hvac(N_zones = 400L, N_layers = 20L)

# =====================================================================
# Run selection
# =====================================================================

# Allow running a subset via env var BENCH_SUBSET (comma-separated names)
subset_env <- Sys.getenv("BENCH_SUBSET", unset = "")
if (nzchar(subset_env)) {
  sel <- strsplit(subset_env, ",")[[1]]
  problems <- problems[sel]
  cat(sprintf("Running subset: %s\n", paste(sel, collapse = ", ")))
}

for (p in problems) {
  t0 <- proc.time()[["elapsed"]]
  safe(run_problem(p), p$name)
  cat(sprintf("  [%s done in %.1fs]\n", p$name, proc.time()[["elapsed"]] - t0))
}

# =====================================================================
# Summary
# =====================================================================
df <- do.call(rbind, results)
if (!is.null(df)) {
  cat("\n\n======================================================\n")
  cat("SUMMARY -- fevals & wallclock relative to CVODE_bdf (per tol)\n")
  cat("======================================================\n")
  for (tol_name in names(TOLS)) {
    sub <- df[df$tol == tol_name & is.finite(df$fevals), ]
    if (!nrow(sub)) next
    ref <- sub[sub$solver == "CVODE_bdf", c("problem", "fevals", "time_ms")]
    if (!nrow(ref)) next
    names(ref) <- c("problem", "cvode_fev", "cvode_ms")
    m <- merge(sub, ref, by = "problem")
    m$fev_ratio  <- m$fevals  / m$cvode_fev
    m$time_ratio <- m$time_ms / m$cvode_ms

    cat(sprintf("\n--- %s ---\n", tol_name))
    # Per-solver geometric mean across problems
    agg <- aggregate(
      cbind(fev_ratio, time_ratio) ~ solver,
      data = m, FUN = function(x) exp(mean(log(x), na.rm = TRUE))
    )
    agg <- agg[order(agg$fev_ratio), ]
    for (k in seq_len(nrow(agg))) {
      cat(sprintf("  %-18s  fev_ratio=%.2f  time_ratio=%.2f\n",
                  agg$solver[k], agg$fev_ratio[k], agg$time_ratio[k]))
    }
  }

  # Write full table for post-analysis
  out_csv <- file.path(.workingDir, "bench_stiff_suite_results.csv")
  write.csv(df, out_csv, row.names = FALSE)
  cat(sprintf("\nFull results written to: %s\n", out_csv))
}

cat("\nDone.\n")
