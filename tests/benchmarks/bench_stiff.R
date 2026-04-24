## =================================================================
## Benchmark: stiff solvers -- CppODE (bdf, rb4, msoda) vs deSolve
## =================================================================
rm(list = ls(all.names = TRUE))

.workingDir <- file.path(tempdir(), "CppODE_bench_stiff")
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

print_row <- function(solver, npts, timing, diag = NULL, mass_err = NULL) {
  acc  <- if (!is.null(diag) && !is.na(diag$accepted)) sprintf("%6d", diag$accepted) else "     ?"
  rej  <- if (!is.null(diag) && !is.na(diag$rejected)) sprintf("%6d", diag$rejected) else "     ?"
  fev  <- if (!is.null(diag) && !is.na(diag$fevals))   sprintf("%6d", diag$fevals)   else "     ?"
  me   <- if (!is.null(mass_err)) sprintf("  mass=%.1e", mass_err) else ""
  cat(sprintf("  %-22s  %5d pts  %8.1f ms (med)  acc=%s  rej=%s  fev=%s%s\n",
              solver, npts, timing$median * 1000, acc, rej, fev, me))
}

# Mass conservation error for Robertson: max|y1+y2+y3 - 1|
rob_mass_err_cpp <- function(r) max(abs(colSums(r$variable[c("y1","y2","y3"), ]) - 1))
rob_mass_err_ds  <- function(r) max(abs(r[, "y1"] + r[, "y2"] + r[, "y3"] - 1))

nrep <- 10L

# =====================================================================
#  Problem 1: Robertson chemical kinetics (3 states, stiff classic)
#
#  dy1/dt = -0.04*y1 + 1e4*y2*y3
#  dy2/dt =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
#  dy3/dt =  3e7*y2^2
# =====================================================================
print_header("1. Robertson chemical kinetics (3 states)")

rob_eqns <- c(
  y1 = "-k1*y1 + k2*y2*y3",
  y2 = " k1*y1 - k2*y2*y3 - k3*y2^2",
  y3 = " k3*y2^2"
)
rob_pars <- c(y1 = 1, y2 = 0, y3 = 0, k1 = 0.04, k2 = 1e4, k3 = 3e7)
rob_times <- c(0, 10^seq(-5, 5, length.out = 500))

# --- CppODE models ---
rob_ndf <- CppODE(rob_eqns, deriv = FALSE, outdir = getwd(),
                  method = "bdf", modelname = "rob_ndf", compile = TRUE)
rob_bdf <- CppODE(rob_eqns, deriv = FALSE, outdir = getwd(), useNDF = FALSE,
                  method = "bdf", modelname = "rob_bdf", compile = TRUE)
rob_rb4 <- CppODE(rob_eqns, deriv = FALSE, outdir = getwd(),
                  method = "rb4", modelname = "rob_rb4", compile = TRUE)
rob_msoda <- CppODE(rob_eqns, deriv = FALSE, outdir = getwd(),
                    method = "msoda", modelname = "rob_msoda", compile = TRUE)

# --- deSolve function ---
rob_desolve <- function(t, y, p) {
  with(as.list(c(y, p)), {
    list(c(
      -k1*y1 + k2*y2*y3,
       k1*y1 - k2*y2*y3 - k3*y2^2,
       k3*y2^2
    ))
  })
}

# --- Run ---
cat(sprintf("  %-22s  %5s  %13s  %6s  %6s  %6s\n",
            "Solver", "Pts", "Time(ms)", "Accept", "Reject", "Fevals"))
cat("  ", strrep("-", 78), "\n")

r <- solveODE(rob_bdf, rob_times, rob_pars)
t_bdf <- bench("bdf", quote(solveODE(rob_bdf, rob_times, rob_pars)), nrep)
print_row("CppODE bdf", ncol(r$variable), t_bdf, r$diagnostics, rob_mass_err_cpp(r))

r <- solveODE(rob_rb4, rob_times, rob_pars)
t_rb4 <- bench("rb4", quote(solveODE(rob_rb4, rob_times, rob_pars)), nrep)
print_row("CppODE rb4", ncol(r$variable), t_rb4, r$diagnostics, rob_mass_err_cpp(r))

r <- solveODE(rob_msoda, rob_times, rob_pars)
t_msoda <- bench("msoda", quote(solveODE(rob_msoda, rob_times, rob_pars)), nrep)
print_row("CppODE msoda", ncol(r$variable), t_msoda, r$diagnostics, rob_mass_err_cpp(r))

r_lsoda <- lsoda(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6])
t_lsoda <- bench("lsoda", quote(lsoda(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6])), nrep)
print_row("deSolve lsoda", nrow(r_lsoda), t_lsoda, desolve_diag(r_lsoda), rob_mass_err_ds(r_lsoda))

r_vode <- ode(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], method = "vode")
t_vode <- bench("vode", quote(ode(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], method = "vode")), nrep)
print_row("deSolve vode", nrow(r_vode), t_vode, desolve_diag(r_vode), rob_mass_err_ds(r_vode))

r_radau <- ode(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], method = "radau")
t_radau <- bench("radau", quote(ode(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], method = "radau")), nrep)
print_row("deSolve radau", nrow(r_radau), t_radau, desolve_diag(r_radau), rob_mass_err_ds(r_radau))

# Verify agreement
ref_mat <- r_lsoda[, c("y1", "y2", "y3")]
cpp_mat <- t(r$variable[c("y1","y2","y3"), ])
cat(sprintf("\n  Max |CppODE msoda - deSolve lsoda|: %.3e\n",
            max(abs(cpp_mat[1:nrow(ref_mat), ] - ref_mat))))


# =====================================================================
#  Problem 2: Van der Pol oscillator (mu = 1000, 2 states, very stiff)
# =====================================================================
print_header("2. Van der Pol oscillator (mu=1000, 2 states)")

vdp_eqns <- c(
  x = "y",
  y = "mu * (1 - x^2) * y - x"
)
vdp_pars <- c(x = 2, y = 0, mu = 1000)
vdp_times <- seq(0, 3000, length.out = 3000)

vdp_bdf <- CppODE(vdp_eqns, deriv = FALSE, outdir = getwd(),
                  method = "bdf", modelname = "vdp_bdf", compile = TRUE)
vdp_rb4 <- CppODE(vdp_eqns, deriv = FALSE, outdir = getwd(),
                  method = "rb4", modelname = "vdp_rb4", compile = TRUE)
vdp_msoda <- CppODE(vdp_eqns, deriv = FALSE, outdir = getwd(),
                    method = "msoda", modelname = "vdp_msoda", compile = TRUE)

vdp_desolve <- function(t, y, p) {
  with(as.list(c(y, p)), {
    list(c(y, mu * (1 - x^2) * y - x))
  })
}

cat(sprintf("  %-22s  %5s  %13s  %6s  %6s  %6s\n",
            "Solver", "Pts", "Time(ms)", "Accept", "Reject", "Fevals"))
cat("  ", strrep("-", 78), "\n")

r <- solveODE(vdp_bdf, vdp_times, vdp_pars)
t_bdf <- bench("bdf", quote(solveODE(vdp_bdf, vdp_times, vdp_pars)), nrep)
print_row("CppODE bdf", ncol(r$variable), t_bdf, r$diagnostics)

r <- solveODE(vdp_rb4, vdp_times, vdp_pars)
t_rb4 <- bench("rb4", quote(solveODE(vdp_rb4, vdp_times, vdp_pars)), nrep)
print_row("CppODE rb4", ncol(r$variable), t_rb4, r$diagnostics)

r <- solveODE(vdp_msoda, vdp_times, vdp_pars)
t_msoda <- bench("msoda", quote(solveODE(vdp_msoda, vdp_times, vdp_pars)), nrep)
print_row("CppODE msoda", ncol(r$variable), t_msoda, r$diagnostics)

r_lsoda <- lsoda(vdp_pars[1:2], vdp_times, vdp_desolve, vdp_pars[3])
t_lsoda <- bench("lsoda", quote(lsoda(vdp_pars[1:2], vdp_times, vdp_desolve, vdp_pars[3])), nrep)
print_row("deSolve lsoda", nrow(r_lsoda), t_lsoda, desolve_diag(r_lsoda))

r_vode <- ode(vdp_pars[1:2], vdp_times, vdp_desolve, vdp_pars[3], method = "vode")
t_vode <- bench("vode", quote(ode(vdp_pars[1:2], vdp_times, vdp_desolve, vdp_pars[3], method = "vode")), nrep)
print_row("deSolve vode", nrow(r_vode), t_vode, desolve_diag(r_vode))

r_radau <- ode(vdp_pars[1:2], vdp_times, vdp_desolve, vdp_pars[3], method = "radau")
t_radau <- bench("radau", quote(ode(vdp_pars[1:2], vdp_times, vdp_desolve, vdp_pars[3], method = "radau")), nrep)
print_row("deSolve radau", nrow(r_radau), t_radau, desolve_diag(r_radau))


# =====================================================================
#  Problem 3: ROBER at tight tolerances
# =====================================================================
print_header("3. Robertson at tight tolerances (atol=1e-10, rtol=1e-8)")

cat(sprintf("  %-22s  %5s  %13s  %6s  %6s  %6s\n",
            "Solver", "Pts", "Time(ms)", "Accept", "Reject", "Fevals"))
cat("  ", strrep("-", 78), "\n")

r_bdf <- solveODE(rob_bdf, rob_times, rob_pars, abstol = 1e-10, reltol = 1e-8)
t_bdf <- bench("bdf", quote(solveODE(rob_bdf, rob_times, rob_pars, abstol = 1e-10, reltol = 1e-8)), nrep)
print_row("CppODE bdf", ncol(r_bdf$variable), t_bdf, r_bdf$diagnostics, rob_mass_err_cpp(r_bdf))

r_ndf <- solveODE(rob_ndf, rob_times, rob_pars, abstol = 1e-10, reltol = 1e-8)
t_ndf <- bench("ndf", quote(solveODE(rob_ndf, rob_times, rob_pars, abstol = 1e-10, reltol = 1e-8)), nrep)
print_row("CppODE ndf", ncol(r_ndf$variable), t_ndf, r_ndf$diagnostics, rob_mass_err_cpp(r_ndf))

r <- solveODE(rob_rb4, rob_times, rob_pars, abstol = 1e-10, reltol = 1e-8)
t_rb4 <- bench("rb4", quote(solveODE(rob_rb4, rob_times, rob_pars, abstol = 1e-10, reltol = 1e-8)), nrep)
print_row("CppODE rb4", ncol(r$variable), t_rb4, r$diagnostics, rob_mass_err_cpp(r))

r_lsoda <- lsoda(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], atol = 1e-10, rtol = 1e-8)
t_lsoda <- bench("lsoda", quote(lsoda(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], atol = 1e-10, rtol = 1e-8)), nrep)
print_row("deSolve lsoda", nrow(r_lsoda), t_lsoda, desolve_diag(r_lsoda), rob_mass_err_ds(r_lsoda))

r_vode <- ode(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], method = "vode", atol = 1e-10, rtol = 1e-8)
t_vode <- bench("vode", quote(ode(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], method = "vode", atol = 1e-10, rtol = 1e-8)), nrep)
print_row("deSolve vode", nrow(r_vode), t_vode, desolve_diag(r_vode), rob_mass_err_ds(r_vode))

r_dsbdf <- ode(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], method = "bdf", atol = 1e-10, rtol = 1e-8)
t_dsbdf <- bench("bdf", quote(ode(rob_pars[1:3], rob_times, rob_desolve, rob_pars[4:6], method = "bdf", atol = 1e-10, rtol = 1e-8)), nrep)
print_row("deSolve bdf", nrow(r_dsbdf), t_dsbdf, desolve_diag(r_dsbdf), rob_mass_err_ds(r_dsbdf))

# --- CVODE standalone (C++) ---
# Optional: a prebuilt `cvode_robertson` binary next to the benchmark script
# gives a C-only reference. Skipped if absent.
cvode_bin <- file.path(dirname(normalizePath(
  if (nzchar(Sys.getenv("CPPODE_BENCH_SCRIPT")))
    Sys.getenv("CPPODE_BENCH_SCRIPT") else tempdir())),
  "cvode_robertson")

if (file.exists(cvode_bin)) {
  cvode_out <- system2(cvode_bin, args = c(as.character(nrep), "tight"),
                       stdout = TRUE, stderr = FALSE)
  # CSV line: solver,npts,median_ms,steps,fevals,mass_err
  parts <- strsplit(cvode_out[1], ",")[[1]]
  cvode_mass <- if (length(parts) >= 6) as.numeric(parts[6]) else NULL
  cvode_diag <- list(accepted = as.integer(parts[4]),
                     rejected = NA,
                     fevals   = as.integer(parts[5]))
  print_row("CVODE BDF (C++)", as.integer(parts[2]),
            list(median = as.numeric(parts[3]) / 1000,
                 min = NA, max = NA),
            cvode_diag, cvode_mass)
} else {
  cat(sprintf("  %-22s  (binary not found at %s -- skip)\n", "CVODE BDF (C++)", cvode_bin))
}

cat("\nDone.\n")
