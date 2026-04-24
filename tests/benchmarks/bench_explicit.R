## =================================================================
## Benchmark: explicit solvers -- CppODE (tsit5, adams) vs deSolve
## =================================================================
rm(list = ls(all.names = TRUE))

.workingDir <- file.path(tempdir(), "CppODE_bench_explicit")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(deSolve)

# =====================================================================
#  Benchmark harness
# =====================================================================
bench <- function(label, expr, nrep = 5L) {
  eval(expr)  # warmup
  times <- replicate(nrep, system.time(eval(expr))[["elapsed"]])
  list(median = median(times), min = min(times), max = max(times))
}

print_header <- function(title) {
  cat(sprintf("\n%s\n%s\n", title, strrep("=", nchar(title))))
}

desolve_diag <- function(res) {
  ## Extract solver diagnostics from deSolve result attributes.
  ## istate layout (lsoda/lsode/vode): [2]=steps, [3]=fevals
  ## rk methods (ode45, rk4, ...) don't populate istate reliably.
  ist <- attr(res, "istate")
  if (is.null(ist)) return(NULL)
  steps <- ist[2]; fevals <- ist[3]
  if (is.na(steps) || steps == 0) return(NULL)
  list(accepted = steps, rejected = NA, fevals = fevals)
}

print_row <- function(solver, npts, timing, diag = NULL, err = NA) {
  acc  <- if (!is.null(diag) && !is.na(diag$accepted)) sprintf("%6d", diag$accepted) else "     ?"
  rej  <- if (!is.null(diag) && !is.na(diag$rejected)) sprintf("%6d", diag$rejected) else "     ?"
  fev  <- if (!is.null(diag) && !is.na(diag$fevals))   sprintf("%6d", diag$fevals)   else "     ?"
  err_str <- if (is.na(err)) "          ?" else sprintf("%10.2e", err)
  cat(sprintf("  %-22s  %5d pts  %8.1f ms  acc=%s  rej=%s  fev=%s  err=%s\n",
              solver, npts, timing$median * 1000, acc, rej, fev, err_str))
}

nrep <- 10L

# =====================================================================
#  Problem 1: Harmonic oscillator (2 states) -- exact solution
# =====================================================================
print_header("1. Harmonic oscillator (2 states, exact reference)")

ho_eqns <- c(x = "y", y = "-x")
ho_pars <- c(x = 1, y = 0)
ho_times <- seq(0, 100, length.out = 300)

ho_ref_x <- cos(ho_times)
ho_ref_y <- -sin(ho_times)

# --- CppODE ---
ho_tsit5 <- CppODE(ho_eqns, deriv = FALSE, outdir = getwd(),
                   method = "tsit5", modelname = "ho_tsit5", compile = TRUE)
ho_adams <- CppODE(ho_eqns, deriv = FALSE, outdir = getwd(),
                   method = "adams", modelname = "ho_adams", compile = TRUE)

cat(sprintf("  %-22s  %5s  %8s  %6s  %6s  %6s  %10s\n",
            "Solver", "Pts", "ms(med)", "Accept", "Reject", "Fevals", "max|err|"))
cat("  ", strrep("-", 86), "\n")

r <- solveODE(ho_tsit5, ho_times, ho_pars)
err <- max(abs(r$variable["x",] - ho_ref_x), abs(r$variable["y",] - ho_ref_y))
t1 <- bench("tsit5", quote(solveODE(ho_tsit5, ho_times, ho_pars)), nrep)
print_row("CppODE tsit5", ncol(r$variable), t1, r$diagnostics, err)

r <- solveODE(ho_adams, ho_times, ho_pars)
err <- max(abs(r$variable["x",] - ho_ref_x), abs(r$variable["y",] - ho_ref_y))
t1 <- bench("adams", quote(solveODE(ho_adams, ho_times, ho_pars)), nrep)
print_row("CppODE adams", ncol(r$variable), t1, r$diagnostics, err)

# --- deSolve ---
ho_desolve <- function(t, y, p) list(c(y[2], -y[1]))

r_ode45 <- ode(ho_pars, ho_times, ho_desolve, NULL, method = "ode45")
err_ode45 <- max(abs(r_ode45[,"x"] - ho_ref_x), abs(r_ode45[,"y"] - ho_ref_y))
t1 <- bench("ode45", quote(ode(ho_pars, ho_times, ho_desolve, NULL, method = "ode45")), nrep)
print_row("deSolve ode45", nrow(r_ode45), t1, desolve_diag(r_ode45), err_ode45)

r_lsoda <- lsoda(ho_pars, ho_times, ho_desolve, NULL)
err_lsoda <- max(abs(r_lsoda[,"x"] - ho_ref_x), abs(r_lsoda[,"y"] - ho_ref_y))
t1 <- bench("lsoda", quote(lsoda(ho_pars, ho_times, ho_desolve, NULL)), nrep)
print_row("deSolve lsoda", nrow(r_lsoda), t1, desolve_diag(r_lsoda), err_lsoda)

r_rk4 <- ode(ho_pars, ho_times, ho_desolve, NULL, method = "rk4")
err_rk4 <- max(abs(r_rk4[,"x"] - ho_ref_x), abs(r_rk4[,"y"] - ho_ref_y))
t1 <- bench("rk4", quote(ode(ho_pars, ho_times, ho_desolve, NULL, method = "rk4")), nrep)
print_row("deSolve rk4", nrow(r_rk4), t1, desolve_diag(r_rk4), err_rk4)


# =====================================================================
#  Problem 2: Lotka-Volterra (2 states) -- deSolve lsoda as reference
# =====================================================================
print_header("2. Lotka-Volterra (2 states)")

lv_eqns <- c(x = "a*x - b*x*y", y = "c*x*y - d*y")
lv_pars <- c(x = 10, y = 5, a = 1.5, b = 1, c = 0.75, d = 1)
lv_times <- seq(0, 50, length.out = 5001)

# Reference: lsoda at very tight tolerance
lv_desolve <- function(t, y, p) {
  with(as.list(c(y, p)), list(c(a*x - b*x*y, c*x*y - d*y)))
}
lv_ref <- lsoda(lv_pars[1:2], lv_times, lv_desolve, as.list(lv_pars[3:6]),
                atol = 1e-14, rtol = 1e-14)

lv_tsit5 <- CppODE(lv_eqns, deriv = FALSE, outdir = getwd(),
                   method = "tsit5", modelname = "lv_tsit5", compile = TRUE)
lv_adams <- CppODE(lv_eqns, deriv = FALSE, outdir = getwd(),
                   method = "adams", modelname = "lv_adams", compile = TRUE)

cat(sprintf("  %-22s  %5s  %8s  %6s  %6s  %6s  %10s\n",
            "Solver", "Pts", "ms(med)", "Accept", "Reject", "Fevals", "max|err|"))
cat("  ", strrep("-", 86), "\n")

r <- solveODE(lv_tsit5, lv_times, lv_pars, abstol = 1e-8, reltol = 1e-6)
err <- max(abs(t(r$variable) - lv_ref[, c("x","y")]))
t1 <- bench("tsit5", quote(solveODE(lv_tsit5, lv_times, lv_pars, abstol = 1e-8, reltol = 1e-6)), nrep)
print_row("CppODE tsit5", ncol(r$variable), t1, r$diagnostics, err)

r <- solveODE(lv_adams, lv_times, lv_pars, abstol = 1e-8, reltol = 1e-6)
err <- max(abs(t(r$variable) - lv_ref[, c("x","y")]))
t1 <- bench("adams", quote(solveODE(lv_adams, lv_times, lv_pars, abstol = 1e-8, reltol = 1e-6)), nrep)
print_row("CppODE adams", ncol(r$variable), t1, r$diagnostics, err)

r_ode45 <- ode(lv_pars[1:2], lv_times, lv_desolve, as.list(lv_pars[3:6]),
               method = "ode45", atol = 1e-8, rtol = 1e-6)
err_ode45 <- max(abs(r_ode45[, c("x","y")] - lv_ref[, c("x","y")]))
t1 <- bench("ode45", quote(ode(lv_pars[1:2], lv_times, lv_desolve, as.list(lv_pars[3:6]),
                                method = "ode45", atol = 1e-8, rtol = 1e-6)), nrep)
print_row("deSolve ode45", nrow(r_ode45), t1, desolve_diag(r_ode45), err_ode45)

r_lsoda <- lsoda(lv_pars[1:2], lv_times, lv_desolve, as.list(lv_pars[3:6]),
                 atol = 1e-8, rtol = 1e-6)
err_lsoda <- max(abs(r_lsoda[, c("x","y")] - lv_ref[, c("x","y")]))
t1 <- bench("lsoda", quote(lsoda(lv_pars[1:2], lv_times, lv_desolve, as.list(lv_pars[3:6]),
                                  atol = 1e-8, rtol = 1e-6)), nrep)
print_row("deSolve lsoda", nrow(r_lsoda), t1, desolve_diag(r_lsoda), err_lsoda)


# =====================================================================
#  Problem 3: Pleiades 7-body (28 states) -- classic non-stiff benchmark
# =====================================================================
print_header("3. Pleiades 7-body (28 states)")

pleiades_rhs <- character(28)
names(pleiades_rhs) <- c(
  paste0("x", 1:7), paste0("y", 1:7),
  paste0("vx", 1:7), paste0("vy", 1:7)
)
for (i in 1:7) {
  pleiades_rhs[paste0("x", i)]  <- paste0("vx", i)
  pleiades_rhs[paste0("y", i)]  <- paste0("vy", i)
  ax_terms <- ay_terms <- character()
  for (j in 1:7) {
    if (j != i) {
      rij <- sprintf("((x%d - x%d)^2 + (y%d - y%d)^2)^(3/2)", i, j, i, j)
      ax_terms <- c(ax_terms, sprintf("%d * (x%d - x%d) / %s", j, j, i, rij))
      ay_terms <- c(ay_terms, sprintf("%d * (y%d - y%d) / %s", j, j, i, rij))
    }
  }
  pleiades_rhs[paste0("vx", i)] <- paste(ax_terms, collapse = " + ")
  pleiades_rhs[paste0("vy", i)] <- paste(ay_terms, collapse = " + ")
}

pl_pars <- c(
  x1 = 3, x2 = 3, x3 = -1, x4 = -3, x5 = 2, x6 = -2, x7 = 2,
  y1 = 3, y2 = -3, y3 = 2, y4 = 0, y5 = 0, y6 = -4, y7 = 4,
  vx1 = 0, vx2 = 0, vx3 = 0, vx4 = 0, vx5 = 0, vx6 = 0, vx7 = 0,
  vy1 = 0, vy2 = 0, vy3 = 0, vy4 = 0, vy5 = 0, vy6 = 0, vy7 = 0
)
pl_times <- seq(0, 3, length.out = 301)

cat("Compiling Pleiades (tsit5, skip_jacobian)...\n")
pl_tsit5 <- CppODE(pleiades_rhs, deriv = FALSE, outdir = getwd(),
                   method = "tsit5", modelname = "pl_tsit5", compile = TRUE)
cat("Compiling Pleiades (adams)...\n")
pl_adams <- CppODE(pleiades_rhs, deriv = FALSE, outdir = getwd(),
                   method = "adams", modelname = "pl_adams", compile = TRUE)

# deSolve version
pl_desolve <- function(t, y, p) {
  x <- y[1:7]; yy <- y[8:14]; vx <- y[15:21]; vy <- y[22:28]
  ax <- numeric(7); ay <- numeric(7)
  for (i in 1:7) {
    for (j in 1:7) {
      if (j != i) {
        rij3 <- ((x[i] - x[j])^2 + (yy[i] - yy[j])^2)^(3/2)
        ax[i] <- ax[i] + j * (x[j] - x[i]) / rij3
        ay[i] <- ay[i] + j * (yy[j] - yy[i]) / rij3
      }
    }
  }
  list(c(vx, vy, ax, ay))
}

cat(sprintf("  %-22s  %5s  %8s  %6s  %6s  %6s\n",
            "Solver", "Pts", "ms(med)", "Accept", "Reject", "Fevals"))
cat("  ", strrep("-", 72), "\n")

r <- solveODE(pl_tsit5, pl_times, pl_pars)
t1 <- bench("tsit5", quote(solveODE(pl_tsit5, pl_times, pl_pars)), nrep)
print_row("CppODE tsit5", ncol(r$variable), t1, r$diagnostics)

r <- solveODE(pl_adams, pl_times, pl_pars)
t1 <- bench("adams", quote(solveODE(pl_adams, pl_times, pl_pars)), nrep)
print_row("CppODE adams", ncol(r$variable), t1, r$diagnostics)

r_ode45 <- ode(pl_pars, pl_times, pl_desolve, NULL, method = "ode45")
t1 <- bench("ode45", quote(ode(pl_pars, pl_times, pl_desolve, NULL, method = "ode45")), nrep)
print_row("deSolve ode45", nrow(r_ode45), t1, desolve_diag(r_ode45))

r_lsoda <- lsoda(pl_pars, pl_times, pl_desolve, NULL)
t1 <- bench("lsoda", quote(lsoda(pl_pars, pl_times, pl_desolve, NULL)), nrep)
print_row("deSolve lsoda", nrow(r_lsoda), t1, desolve_diag(r_lsoda))


# =====================================================================
#  Problem 4: Exponential decay (1 state) -- overhead measurement
# =====================================================================
print_header("4. Exponential decay (1 state, overhead test)")

exp_eqns <- c(x = "-k*x")
exp_pars <- c(x = 1, k = 1)
exp_times <- seq(0, 10, length.out = 10001)

exp_tsit5 <- CppODE(exp_eqns, deriv = FALSE, outdir = getwd(),
                    method = "tsit5", modelname = "exp_tsit5", compile = TRUE)
exp_adams <- CppODE(exp_eqns, deriv = FALSE, outdir = getwd(),
                    method = "adams", modelname = "exp_adams", compile = TRUE)

exp_ref <- exp(-exp_times)

exp_desolve <- function(t, y, p) list(-p * y)

cat(sprintf("  %-22s  %5s  %8s  %6s  %6s  %6s  %10s\n",
            "Solver", "Pts", "ms(med)", "Accept", "Reject", "Fevals", "max|err|"))
cat("  ", strrep("-", 86), "\n")

r <- solveODE(exp_tsit5, exp_times, exp_pars)
err <- max(abs(as.numeric(r$variable) - exp_ref))
t1 <- bench("tsit5", quote(solveODE(exp_tsit5, exp_times, exp_pars)), nrep)
print_row("CppODE tsit5", ncol(r$variable), t1, r$diagnostics, err)

r <- solveODE(exp_adams, exp_times, exp_pars)
err <- max(abs(as.numeric(r$variable) - exp_ref))
t1 <- bench("adams", quote(solveODE(exp_adams, exp_times, exp_pars)), nrep)
print_row("CppODE adams", ncol(r$variable), t1, r$diagnostics, err)

r_ode45 <- ode(exp_pars[1], exp_times, exp_desolve, exp_pars[2], method = "ode45")
err_ode45 <- max(abs(r_ode45[,"x"] - exp_ref))
t1 <- bench("ode45", quote(ode(exp_pars[1], exp_times, exp_desolve, exp_pars[2], method = "ode45")), nrep)
print_row("deSolve ode45", nrow(r_ode45), t1, desolve_diag(r_ode45), err_ode45)

r_lsoda <- lsoda(exp_pars[1], exp_times, exp_desolve, exp_pars[2])
err_lsoda <- max(abs(r_lsoda[,2] - exp_ref))
t1 <- bench("lsoda", quote(lsoda(exp_pars[1], exp_times, exp_desolve, exp_pars[2])), nrep)
print_row("deSolve lsoda", nrow(r_lsoda), t1, desolve_diag(r_lsoda), err_lsoda)

cat("\nDone.\n")
