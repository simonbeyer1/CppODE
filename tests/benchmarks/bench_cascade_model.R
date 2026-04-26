## =================================================================
## Benchmark: cascade signaling network -- stiff solvers
##   CppODE (bdf/ndf/rb4/tsit5/msoda) heap vs static ntheta = 44
##   CVODE (dense vs KLU)  +  deSolve LSODES
## =================================================================
rm(list = ls(all.names = TRUE))

.args     <- commandArgs(trailingOnly = TRUE)
.calcSens <- if (length(.args) >= 1) as.logical(.args[1]) else TRUE
.abstol   <- if (length(.args) >= 2) as.numeric(.args[2]) else 1e-8
.reltol   <- if (length(.args) >= 3) as.numeric(.args[3]) else 1e-6

# Compile-time AD width for the "static" variants: 42 (= n_states + n_params)
# rounded up to the next AVX2 lane multiple (4 doubles).
.NTHETA <- 44L

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

# =====================================================================
#  14-state cascade signaling network
# =====================================================================
.hdr(sprintf("Cascade signaling network (14 states)  [sens = %s | abstol = %.0e | reltol = %.0e | ntheta_static = %d]",
             .calcSens, .abstol, .reltol, .NTHETA))

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

# Heap (default) variants -- AD width = n_states + n_params (= 42 here)
m_ndf       <- CppODE(rhs, modelname = "cascade_ndf",       method = "bdf",   useNDF = TRUE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_bdf       <- CppODE(rhs, modelname = "cascade_bdf",       method = "bdf",   useNDF = FALSE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_rb4       <- CppODE(rhs, modelname = "cascade_rb4",       method = "rb4",
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_tsit5     <- CppODE(rhs, modelname = "cascade_tsit5",     method = "tsit5",
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)

# Static-ntheta variants -- AD width fixed to .NTHETA at compile time (stack)
m_ndf_s     <- CppODE(rhs, modelname = "cascade_ndf_s",     method = "bdf",   useNDF = TRUE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      ntheta = .NTHETA)
m_bdf_s     <- CppODE(rhs, modelname = "cascade_bdf_s",     method = "bdf",   useNDF = FALSE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      ntheta = .NTHETA)
m_rb4_s     <- CppODE(rhs, modelname = "cascade_rb4_s",     method = "rb4",
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      ntheta = .NTHETA)
m_tsit5_s   <- CppODE(rhs, modelname = "cascade_tsit5_s",   method = "tsit5",
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      ntheta = .NTHETA)

# Heap variants -- F<double, 0> with runtime-sized m_diff (= new T[n_sens]).
m_ndf_h     <- CppODE(rhs, modelname = "cascade_ndf_h",     method = "bdf",   useNDF = TRUE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      dynamic_ad = TRUE)
m_bdf_h     <- CppODE(rhs, modelname = "cascade_bdf_h",     method = "bdf",   useNDF = FALSE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      dynamic_ad = TRUE)
m_rb4_h     <- CppODE(rhs, modelname = "cascade_rb4_h",     method = "rb4",
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      dynamic_ad = TRUE)
m_tsit5_h   <- CppODE(rhs, modelname = "cascade_tsit5_h",   method = "tsit5",
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      dynamic_ad = TRUE)

# cppode::dual backend — same three width regimes (stack42, stack44, heap)
# These compile from the same generator output but route std::* to cppode::*
# and instantiate cppode::dual<double, N> instead of fadbad::F<double, N>.
m_bdf_d     <- CppODE(rhs, modelname = "cascade_bdf_d",     method = "bdf",   useNDF = FALSE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      ad_backend = "dual")
m_bdf_sd    <- CppODE(rhs, modelname = "cascade_bdf_sd",    method = "bdf",   useNDF = FALSE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      ntheta = .NTHETA, ad_backend = "dual")
m_bdf_hd    <- CppODE(rhs, modelname = "cascade_bdf_hd",    method = "bdf",   useNDF = FALSE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                      dynamic_ad = TRUE, ad_backend = "dual")

m_cvode       <- CVODE( rhs, modelname = "cascade_cvode",       method = "bdf",
                        outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_cvode_klu   <- CVODE( rhs, modelname = "cascade_cvode_klu",   method = "bdf",
                        outdir = getwd(), sparse = TRUE,  deriv = .calcSens, compile = FALSE)
# Reparam variants (compile-time ntheta -> reparam sens_rhs1_fn with general Phi)
m_cvode_r     <- CVODE( rhs, modelname = "cascade_cvode_r",     method = "bdf",
                        outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE,
                        ntheta = .NTHETA)
m_cvode_klu_r <- CVODE( rhs, modelname = "cascade_cvode_klu_r", method = "bdf",
                        outdir = getwd(), sparse = TRUE,  deriv = .calcSens, compile = FALSE,
                        ntheta = .NTHETA)

CppODE:::compile(m_ndf,   m_bdf,   m_tsit5, m_rb4,
                 m_ndf_s, m_bdf_s, m_tsit5_s, m_rb4_s,
                 m_ndf_h, m_bdf_h, m_tsit5_h, m_rb4_h,
                 m_bdf_d, m_bdf_sd, m_bdf_hd,
                 m_cvode, m_cvode_klu, m_cvode_r, m_cvode_klu_r, cores = 10)
cat("Done.\n\n")

# --- Build sens1ini for static-ntheta variants ---
# Reparametrization equivalent to the legacy identity Phi: identity on the
# (state + param) block, plus (.NTHETA - n_phi_rows) zero columns of padding.
build_sens1ini <- function(model, ntheta) {
  vars <- attr(model, "variables")
  pars <- attr(model, "parameters")
  n_phi <- length(vars) + length(pars)
  stopifnot(ntheta >= n_phi)
  # Column names = (states, params, "pad1", "pad2", ...). solveODE picks these
  # up as the sens-dimension labels so toWide() produces "state.paramname"
  # columns matching the cOde / deSolve reference layout.
  col_nms <- c(vars, pars)
  if (ntheta > n_phi) col_nms <- c(col_nms, sprintf("pad%d", seq_len(ntheta - n_phi)))
  M <- matrix(0, nrow = n_phi, ncol = ntheta,
              dimnames = list(c(vars, pars), col_nms))
  for (i in seq_len(n_phi)) M[i, i] <- 1
  M
}
S1   <- build_sens1ini(m_ndf_s, length(params))   # static (n=44 stack, 42 used)
S1_h <- build_sens1ini(m_ndf_h, length(params))   # heap   (runtime width = 42)

# --- deSolve / cOde reference ---
if (.calcSens) rhs.sens <- sensitivitiesSymb(rhs)
yini      <- if (.calcSens) c(params[names(rhs)], attr(rhs.sens, "yini")) else params[names(rhs)]
eqns_code <- if (.calcSens) c(rhs, rhs.sens) else rhs
m_cOde    <- cOde::funC(eqns_code, modelname = "cascade_code", compile = TRUE)
parsC     <- params[setdiff(names(params), names(rhs))]

# --- Benchmark ---
.hdr("Benchmark  (10 evaluations each)")

mb <- microbenchmark(
  `CppODE BDF   (stack42)`         = solveODE(m_bdf,    times, params, abstol = .abstol, reltol = .reltol),
  `CppODE BDF   (stack44)`         = solveODE(m_bdf_s,  times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1),
  `CppODE BDF   (heap)`            = solveODE(m_bdf_h,  times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1_h),
  `CppODE BDF   (stack42 dual)`    = solveODE(m_bdf_d,  times, params, abstol = .abstol, reltol = .reltol),
  `CppODE BDF   (stack44 dual)`    = solveODE(m_bdf_sd, times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1),
  `CppODE BDF   (heap   dual)`     = solveODE(m_bdf_hd, times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1_h),
  `CppODE NDF   (stack42)` = solveODE(m_ndf,       times, params, abstol = .abstol, reltol = .reltol),
  `CppODE NDF   (stack44)` = solveODE(m_ndf_s,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1),
  `CppODE NDF   (heap)`    = solveODE(m_ndf_h,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1_h),
  `CppODE RB4   (stack42)` = solveODE(m_rb4,       times, params, abstol = .abstol, reltol = .reltol),
  `CppODE RB4   (stack44)` = solveODE(m_rb4_s,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1),
  `CppODE RB4   (heap)`    = solveODE(m_rb4_h,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1_h),
  `CppODE TSIT5 (stack42)` = solveODE(m_tsit5,     times, params, abstol = .abstol, reltol = .reltol),
  `CppODE TSIT5 (stack44)` = solveODE(m_tsit5_s,   times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1),
  `CppODE TSIT5 (heap)`    = solveODE(m_tsit5_h,   times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1_h),
  `CVODE  BDF   (dense)`        = solveODE(m_cvode,       times, params, abstol = .abstol, reltol = .reltol),
  `CVODE  BDF   (KLU)`          = solveODE(m_cvode_klu,   times, params, abstol = .abstol, reltol = .reltol),
  `CVODE  BDF   (dense reparam)` = solveODE(m_cvode_r,    times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1),
  `CVODE  BDF   (KLU   reparam)` = solveODE(m_cvode_klu_r,times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1),
  `deSolve LSODES`         = odeC(yini, times, m_cOde, parsC, method = "lsodes", atol = .abstol, rtol = .reltol),
  times = 100L
)
print(mb, unit = "ms")

# --- Full diagnostics ---
# cat("\n\n")
# .hdr("Solver diagnostics")
#
# res_ndf       <- solveODE(m_ndf,       times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
# res_ndf_s     <- solveODE(m_ndf_s,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12, sens1ini = S1)
# res_ndf_h     <- solveODE(m_ndf_h,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12, sens1ini = S1_h)
# res_bdf       <- solveODE(m_bdf,       times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
# res_bdf_s     <- solveODE(m_bdf_s,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12, sens1ini = S1)
# res_bdf_h     <- solveODE(m_bdf_h,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12, sens1ini = S1_h)
# res_rb4       <- solveODE(m_rb4,       times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
# res_rb4_s     <- solveODE(m_rb4_s,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12, sens1ini = S1)
# res_rb4_h     <- solveODE(m_rb4_h,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12, sens1ini = S1_h)
# res_tsit5     <- solveODE(m_tsit5,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
# res_tsit5_s   <- solveODE(m_tsit5_s,   times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12, sens1ini = S1)
# res_tsit5_h   <- solveODE(m_tsit5_h,   times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12, sens1ini = S1_h)
# res_cvode     <- solveODE(m_cvode,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
# res_cvode_klu <- solveODE(m_cvode_klu, times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
#
# for (nm in c("res_ndf","res_ndf_s","res_ndf_h",
#              "res_bdf","res_bdf_s","res_bdf_h",
#              "res_rb4","res_rb4_s","res_rb4_h",
#              "res_tsit5","res_tsit5_s","res_tsit5_h",
#              "res_cvode","res_cvode_klu")) {
#   cat("\n>> ", nm, "\n", sep = "")
#   diagnostics(get(nm))
# }
#
# res_vode   <- odeC(yini, times, m_cOde, parsC, method = "vode",   atol = .abstol, rtol = .reltol, hini = 1e-12)
# res_lsodes <- odeC(yini, times, m_cOde, parsC, method = "lsodes", atol = .abstol, rtol = .reltol, hini = 1e-12)
# cat("\n>> deSolve VODE\n");   deSolve::diagnostics(res_vode)
# cat("\n>> deSolve LSODES\n"); deSolve::diagnostics(res_lsodes)

# --- Solution accuracy ---
.hdr("Solution accuracy  (infinity norm vs. reference)")

# Recompute solver results here (diagnostics block above is commented out).
res_ndf       <- solveODE(m_ndf,       times, params, abstol = .abstol, reltol = .reltol)
res_ndf_s     <- solveODE(m_ndf_s,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1)
res_ndf_h     <- solveODE(m_ndf_h,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1_h)
res_bdf       <- solveODE(m_bdf,       times, params, abstol = .abstol, reltol = .reltol)
res_bdf_s     <- solveODE(m_bdf_s,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1)
res_bdf_h     <- solveODE(m_bdf_h,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1_h)
res_rb4       <- solveODE(m_rb4,       times, params, abstol = .abstol, reltol = .reltol)
res_rb4_s     <- solveODE(m_rb4_s,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1)
res_rb4_h     <- solveODE(m_rb4_h,     times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1_h)
res_tsit5     <- solveODE(m_tsit5,     times, params, abstol = .abstol, reltol = .reltol)
res_tsit5_s   <- solveODE(m_tsit5_s,   times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1)
res_tsit5_h   <- solveODE(m_tsit5_h,   times, params, abstol = .abstol, reltol = .reltol, sens1ini = S1_h)
res_cvode     <- solveODE(m_cvode,     times, params, abstol = .abstol, reltol = .reltol)
res_cvode_klu <- solveODE(m_cvode_klu, times, params, abstol = .abstol, reltol = .reltol)
res_vode      <- odeC(yini, times, m_cOde, parsC, method = "vode", atol = .abstol, rtol = .reltol)

# Convert solveODE result to a wide matrix matching cOde column order. For
# static-ntheta variants we drop the (.NTHETA - n_phi_rows) zero pad columns
# so the layout matches the heap output exactly.
n_phi_rows <- length(attr(m_ndf, "variables")) + length(attr(m_ndf, "parameters"))
toWide <- function(x) {
  if (.calcSens) {
    sens <- x$sens1
    if (dim(sens)[3] > n_phi_rows)
      sens <- sens[, , seq_len(n_phi_rows), drop = FALSE]
    sens_names <- if (!is.null(dimnames(sens)$sens))
                    dimnames(sens)$sens[seq_len(n_phi_rows)]
                  else sprintf("s%d", seq_len(n_phi_rows))
    `colnames<-`(
      cbind(time = x$time, x$variable, matrix(sens, nrow = dim(sens)[1])),
      c("time", dimnames(sens)$variable,
        as.vector(outer(dimnames(sens)$variable, sens_names, paste, sep = "."))))
  } else {
    `colnames<-`(cbind(time = x$time, x$variable),
                 c("time", colnames(x$variable)))
  }
}

wide_ref       <- res_vode
align <- function(w) w[, colnames(wide_ref)]
norms <- list(
  `CppODE BDF   (stack42) vs deSolve VODE` = norm(align(toWide(res_bdf))       - wide_ref, type = "I"),
  `CppODE BDF   (stack44) vs deSolve VODE` = norm(align(toWide(res_bdf_s))     - wide_ref, type = "I"),
  `CppODE BDF   (heap)    vs deSolve VODE` = norm(align(toWide(res_bdf_h))     - wide_ref, type = "I"),
  `CppODE NDF   (stack42) vs deSolve VODE` = norm(align(toWide(res_ndf))       - wide_ref, type = "I"),
  `CppODE NDF   (stack44) vs deSolve VODE` = norm(align(toWide(res_ndf_s))     - wide_ref, type = "I"),
  `CppODE NDF   (heap)    vs deSolve VODE` = norm(align(toWide(res_ndf_h))     - wide_ref, type = "I"),
  `CppODE RB4   (stack42) vs deSolve VODE` = norm(align(toWide(res_rb4))       - wide_ref, type = "I"),
  `CppODE RB4   (stack44) vs deSolve VODE` = norm(align(toWide(res_rb4_s))     - wide_ref, type = "I"),
  `CppODE RB4   (heap)    vs deSolve VODE` = norm(align(toWide(res_rb4_h))     - wide_ref, type = "I"),
  `CppODE TSIT5 (stack42) vs deSolve VODE` = norm(align(toWide(res_tsit5))     - wide_ref, type = "I"),
  `CppODE TSIT5 (stack44) vs deSolve VODE` = norm(align(toWide(res_tsit5_s))   - wide_ref, type = "I"),
  `CppODE TSIT5 (heap)    vs deSolve VODE` = norm(align(toWide(res_tsit5_h))   - wide_ref, type = "I"),
  `CVODE  BDF   (dense)   vs deSolve VODE` = norm(align(toWide(res_cvode))     - wide_ref, type = "I"),
  `CVODE  BDF   (KLU)     vs deSolve VODE` = norm(align(toWide(res_cvode_klu)) - wide_ref, type = "I")
)
for (nm in names(norms))
  cat(sprintf("  %-44s  %.3e\n", nm, norms[[nm]]))
.sep2()

# Parity: stack44 and heap should both match stack42 to ~ machine precision
# (modulo solver step-size sensitivity to AD-induced rounding).
.hdr2("AD-mode parity vs stack42  (infinity norm)")
parity <- list(
  `CppODE BDF   stack44`  = norm(align(toWide(res_bdf_s))   - align(toWide(res_bdf)),   type = "I"),
  `CppODE BDF   heap`     = norm(align(toWide(res_bdf_h))   - align(toWide(res_bdf)),   type = "I"),
  `CppODE NDF   stack44`  = norm(align(toWide(res_ndf_s))   - align(toWide(res_ndf)),   type = "I"),
  `CppODE NDF   heap`     = norm(align(toWide(res_ndf_h))   - align(toWide(res_ndf)),   type = "I"),
  `CppODE RB4   stack44`  = norm(align(toWide(res_rb4_s))   - align(toWide(res_rb4)),   type = "I"),
  `CppODE RB4   heap`     = norm(align(toWide(res_rb4_h))   - align(toWide(res_rb4)),   type = "I"),
  `CppODE TSIT5 stack44`  = norm(align(toWide(res_tsit5_s)) - align(toWide(res_tsit5)), type = "I"),
  `CppODE TSIT5 heap`     = norm(align(toWide(res_tsit5_h)) - align(toWide(res_tsit5)), type = "I")
)
for (nm in names(parity))
  cat(sprintf("  %-32s  %.3e\n", nm, parity[[nm]]))
.sep2()

.hdr("Done")
