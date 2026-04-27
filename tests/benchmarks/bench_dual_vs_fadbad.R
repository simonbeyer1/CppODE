## =================================================================
##  Cascade signaling network: dual backend vs fadbad backend
##  Same 14-state model as bench_cascade_model.R, but trimmed to a
##  6-row comparison for the three width regimes (stack42, stack44,
##  heap) on a single solver (BDF). Compile + memory budget kept
##  small enough to run inside 8 GB / 4 cores.
## =================================================================
rm(list = ls(all.names = TRUE))

.calcSens <- TRUE
.abstol   <- 1e-8
.reltol   <- 1e-6
.NTHETA   <- 44L

.workingDir <- file.path(tempdir(), "CppODE_dual_vs_fadbad")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(microbenchmark)

cat("============================================================\n")
cat("  dual vs fadbad backend  [BDF, sens=TRUE, ntheta=", .NTHETA, "]\n", sep = "")
cat("============================================================\n")

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
  R = 100, LR = 0, KKK = 100, KKKa = 0, KK = 100, KKa = 0,
  K = 100, Ka = 0, TF = 100, TFa = 0,
  mRNA_target = 0, P_target = 0, mRNA_inhib = 0, P_inhib = 0,
  L = 10, kon = 0.1, koff = 0.05, kint = 0.01, ksyn_R = 0.5,
  k1 = 0.5, Km1 = 10, k2 = 0.2, Km2 = 10,
  k3 = 0.3, Km3 = 10, k4 = 0.15, Km4 = 10,
  k5 = 0.4, Km5 = 10, k6 = 0.1, Km6 = 10, ki = 0.5,
  k7 = 0.2, k8 = 0.05,
  ktx = 1.0, kdeg_mRNA = 0.1, ktl = 0.5, kdeg_P = 0.05,
  ktx_i = 0.3, kdeg_mRNA_i = 0.2, ktl_i = 0.4, kdeg_P_i = 0.1
)
times <- seq(0, 100, length.out = 500)

cat("\nCompiling 6 models (3 fadbad + 3 dual) ...\n")

# fadbad backend — three width regimes
m_fb_42 <- CppODE(rhs, modelname = "cb_fb_42", method = "bdf", outdir = getwd(),
                  deriv = TRUE, ad_backend = "fadbad", compile = FALSE)
m_fb_44 <- CppODE(rhs, modelname = "cb_fb_44", method = "bdf", outdir = getwd(),
                  deriv = TRUE, ad_backend = "fadbad", nStack = .NTHETA, compile = FALSE)
m_fb_h  <- CppODE(rhs, modelname = "cb_fb_h",  method = "bdf", outdir = getwd(),
                  deriv = TRUE, ad_backend = "fadbad", nStack = Inf, compile = FALSE)

# dual backend — same three width regimes
m_du_42 <- CppODE(rhs, modelname = "cb_du_42", method = "bdf", outdir = getwd(),
                  deriv = TRUE, ad_backend = "dual", compile = FALSE)
m_du_44 <- CppODE(rhs, modelname = "cb_du_44", method = "bdf", outdir = getwd(),
                  deriv = TRUE, ad_backend = "dual", nStack = .NTHETA, compile = FALSE)
m_du_h  <- CppODE(rhs, modelname = "cb_du_h",  method = "bdf", outdir = getwd(),
                  deriv = TRUE, ad_backend = "dual", nStack = Inf, compile = FALSE)

CppODE:::compile(m_fb_42, m_fb_44, m_fb_h, m_du_42, m_du_44, m_du_h, cores = 4)
cat("Done.\n\n")

# sens1ini for the static-44 and heap variants
build_sens1ini <- function(ntheta) {
  vars <- attr(m_du_42, "variables")
  pars <- attr(m_du_42, "parameters")
  n_phi <- length(vars) + length(pars)
  stopifnot(ntheta >= n_phi)
  col_nms <- c(vars, pars)
  if (ntheta > n_phi) col_nms <- c(col_nms, sprintf("pad%d", seq_len(ntheta - n_phi)))
  M <- matrix(0, nrow = n_phi, ncol = ntheta,
              dimnames = list(c(vars, pars), col_nms))
  for (i in seq_len(n_phi)) M[i, i] <- 1
  M
}
S1   <- build_sens1ini(.NTHETA)
S1_h <- build_sens1ini(length(params))

cat("============================================================\n")
cat("  Benchmark  (10 evaluations each)\n")
cat("============================================================\n")
mb <- microbenchmark(
  `BDF fadbad stack42` = solveODE(m_fb_42, times, params, abstol=.abstol, reltol=.reltol),
  `BDF dual   stack42` = solveODE(m_du_42, times, params, abstol=.abstol, reltol=.reltol),
  `BDF fadbad stack44` = solveODE(m_fb_44, times, params, abstol=.abstol, reltol=.reltol, sens1ini=S1),
  `BDF dual   stack44` = solveODE(m_du_44, times, params, abstol=.abstol, reltol=.reltol, sens1ini=S1),
  `BDF fadbad heap   ` = solveODE(m_fb_h,  times, params, abstol=.abstol, reltol=.reltol, sens1ini=S1_h),
  `BDF dual   heap   ` = solveODE(m_du_h,  times, params, abstol=.abstol, reltol=.reltol, sens1ini=S1_h),
  times = 10L
)
print(mb, unit = "ms")

cat("\n============================================================\n")
cat("  Solution parity (dual vs fadbad)\n")
cat("============================================================\n")
o_fb_42 <- solveODE(m_fb_42, times, params, abstol=.abstol, reltol=.reltol)
o_du_42 <- solveODE(m_du_42, times, params, abstol=.abstol, reltol=.reltol)
o_fb_h  <- solveODE(m_fb_h,  times, params, abstol=.abstol, reltol=.reltol, sens1ini=S1_h)
o_du_h  <- solveODE(m_du_h,  times, params, abstol=.abstol, reltol=.reltol, sens1ini=S1_h)
cat(sprintf("stack42  max |dvar|=%.2e  max |dsens|=%.2e\n",
            max(abs(o_fb_42$variable - o_du_42$variable)),
            max(abs(o_fb_42$sens1 - o_du_42$sens1))))
cat(sprintf("heap     max |dvar|=%.2e  max |dsens|=%.2e\n",
            max(abs(o_fb_h$variable - o_du_h$variable)),
            max(abs(o_fb_h$sens1 - o_du_h$sens1))))
