## =================================================================
## Benchmark: cascade signaling network -- stiff solvers
##   CppODE (bdf/ndf/rb4/tsit5) heap vs static ntheta = 44
##   CVODE (dense vs KLU)  +  deSolve LSODES
## =================================================================
rm(list = ls(all.names = TRUE))

.args     <- commandArgs(trailingOnly = TRUE)
.calcSens <- if (length(.args) >= 1) as.logical(.args[1]) else TRUE
.abstol   <- if (length(.args) >= 2) as.numeric(.args[2]) else 1e-6
.reltol   <- if (length(.args) >= 3) as.numeric(.args[3]) else 1e-6

# -----------------------------------------------------------------
# Resolve the directory holding this script (before we setwd away),
# so we can find the companion Julia benchmark.
# -----------------------------------------------------------------
.scriptDir <- local({
  af <- commandArgs(trailingOnly = FALSE)
  hit <- grep("^--file=", af, value = TRUE)
  if (length(hit)) normalizePath(dirname(sub("^--file=", "", hit[1])), mustWork = FALSE)
  else if (!is.null(sys.frames()) && length(sys.frames()) &&
           !is.null(sf <- sys.frame(1)$ofile)) normalizePath(dirname(sf), mustWork = FALSE)
  else normalizePath("tests/benchmarks", mustWork = FALSE)
})
.juliaScript <- file.path(.scriptDir, "bench_cascade_julia.jl")

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
m_ndf       <- CppODE(rhs, modelname = "cascade_ndf", method = "bdf", useNDF = TRUE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_bdf       <- CppODE(rhs, modelname = "cascade_bdf", method = "bdf", useNDF = FALSE,
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_rb4       <- CppODE(rhs, modelname = "cascade_rb4", method = "rb4",
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_tsit5     <- CppODE(rhs, modelname = "cascade_tsit5", method = "tsit5",
                      outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)

m_cvode       <- CVODE(rhs, modelname = "cascade_cvode", method = "bdf",
                       outdir = getwd(), sparse = FALSE, deriv = .calcSens, compile = FALSE)
m_cvode_klu   <- CVODE(rhs, modelname = "cascade_cvode_klu", method = "bdf",
                       outdir = getwd(), sparse = TRUE, deriv = .calcSens, compile = FALSE)

CppODE:::compile(m_ndf, m_bdf, m_tsit5, m_rb4, m_cvode, m_cvode_klu, cores = 6)
cat("Done.\n\n")

# --- deSolve / cOde reference ---
if (.calcSens) rhs.sens <- sensitivitiesSymb(rhs)
yini      <- if (.calcSens) c(params[names(rhs)], attr(rhs.sens, "yini")) else params[names(rhs)]
eqns_code <- if (.calcSens) c(rhs, rhs.sens) else rhs
m_cOde    <- cOde::funC(eqns_code, modelname = "cascade_code", compile = TRUE)
parsC     <- params[setdiff(names(params), names(rhs))]
.sep2()
cat("\n\n")
# --- Benchmark ---
.hdr("Benchmark  (20 evaluations each)")

res_lsodes    <- odeC(yini, times, m_cOde, parsC, method = "lsodes", atol = .abstol, rtol = .reltol, hini = 1e-12)
res_bdf       <- solveODE(m_bdf,       times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
res_ndf       <- solveODE(m_ndf,       times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
res_rb4       <- solveODE(m_rb4,       times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
res_tsit5     <- solveODE(m_tsit5,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
res_cvode     <- solveODE(m_cvode,     times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)
res_cvode_klu <- solveODE(m_cvode_klu, times, params, abstol = .abstol, reltol = .reltol, hini = 1e-12)

mb <- microbenchmark(
  `CppODE BDF`       = solveODE(m_bdf, times, params, abstol = .abstol, reltol = .reltol),
  `CppODE NDF`       = solveODE(m_ndf, times, params, abstol = .abstol, reltol = .reltol),
  `CppODE RB4`       = solveODE(m_rb4, times, params, abstol = .abstol, reltol = .reltol),
  `CppODE TSIT5`     = solveODE(m_tsit5, times, params, abstol = .abstol, reltol = .reltol),
  `CVODE BDF`       = solveODE(m_cvode, times, params, abstol = .abstol, reltol = .reltol),
  `CVODE BDF (KLU)` = solveODE(m_cvode_klu, times, params, abstol = .abstol, reltol = .reltol),
  `deSolve LSODES` = odeC(yini, times, m_cOde, parsC, method = "lsodes", atol = .abstol, rtol = .reltol),
  times = 20L
)
print(mb, unit = "ms")
.sep2()
cat("\n\n")
# --- Solution accuracy ---
.hdr("Solution accuracy  (infinity norm vs. reference (LSODES))")

# Convert solveODE result to a wide matrix matching cOde column order.
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

wide_ref       <- res_lsodes
align <- function(w) w[, colnames(wide_ref)]
norms <- list(
  `CppODE BDF        vs deSolve lsodes()` = norm(align(toWide(res_bdf))       - wide_ref, type = "I"),
  `CppODE NDF        vs deSolve lsodes()` = norm(align(toWide(res_ndf))       - wide_ref, type = "I"),
  `CppODE RB4        vs deSolve lsodes()` = norm(align(toWide(res_rb4))       - wide_ref, type = "I"),
  `CppODE TSIT5      vs deSolve lsodes()` = norm(align(toWide(res_tsit5))     - wide_ref, type = "I"),
  `CVODE  BDF        vs deSolve lsodes()` = norm(align(toWide(res_cvode))     - wide_ref, type = "I"),
  `CVODE  BDF (KLU)  vs deSolve lsodes()` = norm(align(toWide(res_cvode_klu)) - wide_ref, type = "I")
)
for (nm in names(norms))
  cat(sprintf("  %-44s  %.3e\n", nm, norms[[nm]]))
.sep2()
cat("\n\n")

# =====================================================================
#  Julia OrdinaryDiffEq.jl comparison
#    Tsit5 / AutoTsit5(Rodas4) / QNDF -- bare ODE solve, no sens
# =====================================================================
.hdr("Julia OrdinaryDiffEq.jl benchmark")

.juliaBin <- Sys.which("julia")
if (!nzchar(.juliaBin)) {
  cat("julia not on PATH -- skipping Julia comparison.\n")
} else if (!file.exists(.juliaScript)) {
  cat(sprintf("Julia script not found at %s -- skipping.\n", .juliaScript))
} else {
  .juliaCSV    <- file.path(.workingDir, "julia_bench_results.csv")
  .juliaSolCSV <- file.path(.workingDir, "julia_solution.csv")

  cat(sprintf("Running %s\n  calcSens = %s  abstol = %.0e  reltol = %.0e  samples = 20\n",
              .juliaScript, .calcSens, .abstol, .reltol))
  if (.calcSens)
    cat("  (Julia: ForwardDiff over solve + dual-aware internalnorm so sens are tolerance-controlled)\n")
  .sep2()

  # R prepends /usr/lib/R/lib (libRblas etc.) to LD_LIBRARY_PATH; if Julia
  # inherits that, it can pick up R's BLAS/LAPACK over its own bundled MKL
  # and segfault inside Dual{Float64,N} matmul during ForwardDiff. Clear it
  # for the child process.
  .rc <- system2(.juliaBin,
                 args = c(sprintf("--project=%s", shQuote(.scriptDir)),
                          shQuote(.juliaScript),
                          if (isTRUE(.calcSens)) "true" else "false",
                          formatC(.abstol, format = "e", digits = 6),
                          formatC(.reltol, format = "e", digits = 6),
                          "20",
                          shQuote(.juliaCSV),
                          shQuote(.juliaSolCSV)),
                 env = "LD_LIBRARY_PATH=")

  if (.rc != 0 || !file.exists(.juliaCSV)) {
    cat(sprintf("Julia benchmark failed (rc=%d).\n", .rc))
  } else {
    jres <- read.csv(.juliaCSV, stringsAsFactors = FALSE)
    .sep2()
    cat(sprintf("  %-26s %10s %10s %10s %10s %5s\n",
                "expr (Julia)", "min", "mean", "median", "max", "neval"))
    for (i in seq_len(nrow(jres)))
      cat(sprintf("  %-26s %10.3f %10.3f %10.3f %10.3f %5d\n",
                  jres$solver[i], jres$min_ms[i], jres$mean_ms[i],
                  jres$median_ms[i], jres$max_ms[i], jres$neval[i]))
    cat("  (units: ms)\n")
    .sep2()

    if (file.exists(.juliaSolCSV)) {
      jsol <- read.csv(.juliaSolCSV, stringsAsFactors = FALSE, check.names = FALSE)
      jmat <- as.matrix(jsol)
      state_cols <- c("time", names(rhs))
      if (all(state_cols %in% colnames(jmat)) &&
          all(state_cols %in% colnames(wide_ref))) {
        .hdr2("Julia Tsit5 vs deSolve lsodes()  (inf norm)")
        d_state <- jmat[, state_cols, drop = FALSE] -
                   wide_ref[, state_cols, drop = FALSE]
        cat(sprintf("  %-44s  %.3e\n",
                    "Julia Tsit5 (states)",
                    norm(d_state, type = "I")))

        if (.calcSens) {
          # Sens columns follow the "state.sensvar" convention on both sides.
          j_sens   <- setdiff(colnames(jmat),     state_cols)
          ref_sens <- setdiff(colnames(wide_ref), state_cols)
          common   <- intersect(j_sens, ref_sens)
          if (length(common) > 0) {
            # Tight ground-truth reference: LSODES on the augmented system at
            # atol=rtol=1e-12. Lets us see the *actual* sens-error of each
            # method at the bench tolerance, instead of comparing two
            # similarly-loose estimates against each other.
            cat(sprintf("\n  Tight ref: LSODES @ atol=rtol=1e-12 (computing...)\n"))
            ref_tight <- tryCatch(
              odeC(yini, times, m_cOde, parsC, method = "lsodes",
                   atol = 1e-12, rtol = 1e-12, hini = 1e-14),
              error = function(e) { cat("    failed:", conditionMessage(e), "\n"); NULL })

            res_ndf_tight       <- solveODE(m_ndf,       times, params, abstol = 1e-12, reltol = 1e-12, hini = 1e-12)
            res_cvode_tight <- solveODE(m_cvode_klu, times, params, abstol = 1e-12, reltol = 1e-12, hini = 1e-12)

            ndf_wide   <- align(toWide(res_ndf_tight))
            cvode_wide <- align(toWide(res_cvode_tight))

            if (!is.null(ref_tight)) {
              tcols <- intersect(common, colnames(ref_tight))
              .nm <- function(M) norm(M[, tcols, drop = FALSE] -
                                      ref_tight[, tcols, drop = FALSE], type = "I")
              cat(sprintf("  %-44s  %.3e\n",
                          "Julia Tsit5 +sens (ForwardDiff + dual norm)", .nm(jmat)))
              cat(sprintf("  %-44s  %.3e\n",
                          "CppODE NDF +sens",                   .nm(ndf_wide)))
              cat(sprintf("  %-44s  %.3e\n",
                          "CVODE  BDF +sens",                   .nm(cvode_wide)))
              cat(sprintf("    (vs tight ref, %d sens cols, "  ,
                          length(tcols)))
              cat(sprintf("requested tol = %.0e)\n", .abstol))
            } else {
              # Fall back to pairwise vs LSODES@bench-tol if tight ref failed.
              cat(sprintf("  %-44s  %.3e   (%d sens cols)\n",
                          "Julia Tsit5 +sens (ForwardDiff vs LSODES@same-tol)",
                          norm(jmat[, common, drop = FALSE] -
                               wide_ref[, common, drop = FALSE], type = "I"),
                          length(common)))
            }
          } else {
            cat("  (no common sens columns Julia vs LSODES -- naming mismatch)\n")
          }
        }
        .sep2()
      }
    }
  }
}

