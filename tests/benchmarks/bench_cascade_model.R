## =================================================================
## Benchmark: cascade signaling network -- stiff solvers
##            CppODE (bdf, rb4) vs CVODE (bdf)
## =================================================================
rm(list = ls(all.names = TRUE))

.args     <- commandArgs(trailingOnly = TRUE)
.calcSens <- if (length(.args) >= 1) as.logical(.args[1]) else TRUE
.abstol   <- if (length(.args) >= 2) as.numeric(.args[2]) else 1e-8
.reltol   <- if (length(.args) >= 3) as.numeric(.args[3]) else 1e-6

# -----------------------------------------------------------------
# Working directory: ephemeral build dir (perfect for benchmarking)
# -----------------------------------------------------------------
.workingDir <- file.path(tempdir(), "CppODE_cascade_benchmark")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(cOde)
library(microbenchmark)

# =====================================================================
#  Helper functions for formatted output
# =====================================================================
.sep  <- function(char = "=", n = 60) cat(strrep(char, n), "\n")
.sep2 <- function(char = "-", n = 60) cat(strrep(char, n), "\n")
.hdr  <- function(title) { .sep();  cat(sprintf("  %s\n", title)); .sep()  }
.hdr2 <- function(title) { .sep2(); cat(sprintf("  %s\n", title)); .sep2() }

.print_norms <- function(norm_cv_code, norm_ndf_code, norm_ndf_cv,
                         norm_bdf_code, norm_bdf_cv, norm_rb4_code, norm_rb4_cv) {
  cat(sprintf("  %-32s  %.3e\n", "CVODE BDF   vs. deSolve VODE", norm_cv_code))
  cat(sprintf("  %-32s  %.3e\n", "CppODE NDF  vs. deSolve VODE", norm_ndf_code))
  cat(sprintf("  %-32s  %.3e\n", "CppODE NDF  vs. CVODE BDF",    norm_ndf_cv))
  cat(sprintf("  %-32s  %.3e\n", "CppODE BDF  vs. deSolve VODE", norm_bdf_code))
  cat(sprintf("  %-32s  %.3e\n", "CppODE BDF  vs. CVODE BDF",    norm_bdf_cv))
  cat(sprintf("  %-32s  %.3e\n", "CppODE RB4  vs. deSolve VODE", norm_rb4_code))
  cat(sprintf("  %-32s  %.3e\n", "CppODE RB4  vs. CVODE BDF",    norm_rb4_cv))
  .sep2()
}

# =====================================================================
#  14-state cascade signaling network
# =====================================================================
.hdr(sprintf("Cascade signaling network (14 states)  [sens = %s | abstol = %.0e | reltol = %.0e]",
             .calcSens, .abstol, .reltol))

rhs <- c(
  R           = "-kon * L * R + koff * LR + ksyn_R",
  LR          = " kon * L * R - koff * LR - kint * LR",
  KKK         = "-k1 * LR * KKK / (Km1 + KKK) + k2 * KKKa / (Km2 + KKKa) * (1 + ki * P_inhib)",
  KKKa        = " k1 * LR * KKK / (Km1 + KKK) - k2 * KKKa / (Km2 + KKKa) * (1 + ki * P_inhib)",
  KK          = "-k3 * KKKa * KK / (Km3 + KK) + k4 * KKa / (Km4 + KKa)",
  KKa         = " k3 * KKKa * KK / (Km3 + KK) - k4 * KKa / (Km4 + KKa)",
  K           = "-k5 * KKa * K / (Km5 + K) + k6 * Ka / (Km6 + Ka)",
  Ka          = " k5 * KKa * K / (Km5 + K) - k6 * Ka / (Km6 + Ka)",
  TF          = "-k7 * Ka * TF + k8 * TFa",
  TFa         = " k7 * Ka * TF - k8 * TFa",
  mRNA_target = "ktx * TFa - kdeg_mRNA * mRNA_target",
  P_target    = "ktl * mRNA_target - kdeg_P * P_target",
  mRNA_inhib  = "ktx_i * TFa - kdeg_mRNA_i * mRNA_inhib",
  P_inhib     = "ktl_i * mRNA_inhib - kdeg_P_i * P_inhib"
)

params <- c(
  R = 100, LR = 0, KKK = 100, KKKa = 0,
  KK = 100, KKa = 0, K = 100, Ka = 0,
  TF = 100, TFa = 0, mRNA_target = 0, P_target = 0,
  mRNA_inhib = 0, P_inhib = 0,
  L = 10, kon = 0.1, koff = 0.05, kint = 0.01, ksyn_R = 0.5,
  k1 = 0.5, Km1 = 10, k2 = 0.2, Km2 = 10,
  k3 = 0.3, Km3 = 10, k4 = 0.15, Km4 = 10,
  k5 = 0.4, Km5 = 10, k6 = 0.1, Km6 = 10, ki = 0.5,
  k7 = 0.2, k8 = 0.05,
  ktx = 1.0, kdeg_mRNA = 0.1, ktl = 0.5, kdeg_P = 0.05,
  ktx_i = 0.3, kdeg_mRNA_i = 0.2, ktl_i = 0.4, kdeg_P_i = 0.1
)

times <- seq(0, 100, length.out = 500)

# --- Compile models ---
cat("\nCompiling models...\n")
m_ndf          <- CppODE(rhs, modelname = "cascade_ndf", method = "bdf", useNDF = TRUE,
                         outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_bdf          <- CppODE(rhs, modelname = "cascade_bdf", method = "bdf", useNDF = FALSE,
                         outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_rb4          <- CppODE(rhs, modelname = "cascade_rb4", method = "rb4", outdir = getwd(),
                         sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_tsit5        <- CppODE(rhs, modelname = "cascade_tsit5", method = "tsit5", outdir = getwd(),
                         sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_msoda        <- CppODE(rhs, modelname = "cascade_msoda", method = "msoda", outdir = getwd(),
                         sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_cvode        <- CVODE( rhs, modelname = "cascade_cvode", method = "bdf",
                         outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_cvode_sparse <- CVODE( rhs, modelname = "cascade_cvode_sparse", method = "bdf",
                         outdir = getwd(), sparse = TRUE,  deriv = .calcSens, compile = FALSE)

CppODE:::compile(m_ndf, m_bdf, m_rb4, m_tsit5, m_msoda, m_cvode, m_cvode_sparse, cores = 8)
cat("Done.\n\n")

if (.calcSens) rhs.sens <- sensitivitiesSymb(rhs)
yini      <- if (.calcSens) c(params[names(rhs)], attr(rhs.sens, "yini")) else params[names(rhs)]
eqns_code <- if (.calcSens) c(rhs, rhs.sens) else rhs
m_cOde    <- cOde::funC(eqns_code, modelname = "cascade_code", compile = TRUE)
parsC     <- params[setdiff(names(params), names(rhs))]

# --- Benchmark ---
.hdr("Benchmark  (10 evaluations each)")

mb <- microbenchmark(
  `CppODE BDF`     = solveODE(m_bdf,          times, params, abstol = .abstol, reltol = .reltol),
  `CppODE NDF`     = solveODE(m_ndf,          times, params, abstol = .abstol, reltol = .reltol),
  `CppODE RB4`     = solveODE(m_rb4,          times, params, abstol = .abstol, reltol = .reltol),
  `CppODE TSIT5`   = solveODE(m_tsit5,        times, params, abstol = .abstol, reltol = .reltol),
  `CppODE MSODA`   = solveODE(m_msoda,        times, params, abstol = .abstol, reltol = .reltol),
  `CVODE BDF`      = solveODE(m_cvode,        times, params, abstol = .abstol, reltol = .reltol),
  `CVODE BDF KLU`  = solveODE(m_cvode_sparse, times, params, abstol = .abstol, reltol = .reltol),
  `deSolve LSODES` = odeC(yini, times, m_cOde, parsC, method = "lsodes", atol = .abstol, rtol = .reltol),
  times = 10L
)
print(mb, unit = "ms")

# --- Full diagnostics ---
cat("\n\n")
.hdr("Solver diagnostics")


res_ndf <- solveODE(m_ndf, times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
diagnostics(res_ndf)

res_bdf <- solveODE(m_bdf, times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
diagnostics(res_bdf)

res_rb4 <- solveODE(m_rb4, times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
diagnostics(res_rb4)

res_tsit5 <- solveODE(m_tsit5, times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
diagnostics(res_tsit5)

res_msoda <- solveODE(m_msoda, times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
diagnostics(res_msoda)

res_cvode <- solveODE(m_cvode, times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
diagnostics(res_cvode)

res_cvode_sparse <- solveODE(m_cvode_sparse, times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
diagnostics(res_cvode_sparse)

res_vode <- odeC(yini, times, m_cOde, parsC, method = "vode", atol = .abstol, rtol = .reltol, hini = 1e-12)
deSolve::diagnostics(res_vode)

res_lsodes <- odeC(yini, times, m_cOde, parsC, method = "lsodes", atol = .abstol, rtol = .reltol, hini = 1e-12)
deSolve::diagnostics(res_lsodes)

# --- Solution accuracy ---
.hdr("Solution accuracy  (infinity norm vs. reference)")

toWide <- function(x) {
  if (.calcSens) {
    `colnames<-`(
      cbind(time = x$time, x$variable, matrix(x$sens1, nrow = dim(x$sens1)[1])),
      c("time", dimnames(x$sens1)$variable,
        as.vector(outer(dimnames(x$sens1)$variable, dimnames(x$sens1)$sens, paste, sep="."))))
    }
  else {
    `colnames<-`(cbind(time = x$time, x$variable),
                 c("time", colnames(x$variable)))
  }
}
res_ndf_wide   <- toWide(res_ndf)
res_bdf_wide   <- toWide(res_bdf)
res_rb4_wide   <- toWide(res_rb4)
res_cvode_wide <- toWide(res_cvode)
res_ndf_wide   <- res_ndf_wide[,   colnames(res_vode)]
res_bdf_wide   <- res_bdf_wide[,   colnames(res_vode)]
res_cvode_wide <- res_cvode_wide[, colnames(res_vode)]

.print_norms(
  norm(res_cvode_wide - res_vode,       type = "I"),
  norm(res_ndf_wide   - res_vode,       type = "I"),
  norm(res_ndf_wide   - res_cvode_wide, type = "I"),
  norm(res_bdf_wide   - res_vode,       type = "I"),
  norm(res_bdf_wide   - res_cvode_wide, type = "I"),
  norm(res_rb4_wide   - res_vode,       type = "I"),
  norm(res_rb4_wide   - res_cvode_wide, type = "I")
)

.hdr("Done")
