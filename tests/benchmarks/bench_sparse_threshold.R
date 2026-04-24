#!/usr/bin/env Rscript
# ============================================================================
# Benchmark: Dense vs Sparse LU threshold -- Microbenchmark edition
#
# Tests all combinations of:
#   method:  bdf, rb4
#   sparse:  TRUE, FALSE
#   type:    double (deriv=FALSE), Fdouble (deriv=TRUE)
#
# System: Brusselator 2D on NxN grid with variable coupling reach
#   u_ij' = A + u^2v - (B+1)u + alpha * Sigma_{neighbors} (u_nb - u_ij)
#   v_ij' = Bu - u^2v           + alpha * Sigma_{neighbors} (v_nb - v_ij)
#
#   reach=1: 4 direct neighbors (5-point stencil, very sparse)
#   reach=2: 24 neighbors (Chebyshev ball radius 2)
#   reach=N-1: fully coupled (dense Jacobian)
#
# Protocol: 1 warmup + 20 microbenchmark iterations per config
# ============================================================================

.workingDir <- file.path(tempdir(), "CppODE_bench_sparse_threshold")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

suppressPackageStartupMessages({
  library(CppODE)
  library(microbenchmark)
  library(parallel)
})

# --- Configuration ---
#
# Hand-picked (N, reach) combos covering sparsity from ~98% down to ~37%
# and system dimensions from 8 to 2048 states.
#
#   N  reach  dim  sparsity
#   2    1      8    37.5%    <- smallest system, barely sparse
#   3    1     18    64.2%
#   3    2     18    44.4%
#   4    1     32    77.3%
#   4    3     32    46.9%
#   6    1     72    88.7%
#   6    3     72    63.9%
#   6    5     72    48.6%
#   8    1    128    93.3%
#   8    3    128    75.6%
#   8    5    128    58.2%
#  10    1    200    95.6%
#  10    3    200    82.7%
#  10    5    200    67.5%
#  16    1    512    98.2%   <- full range for large systems
#  16    3    512    92.2%
#  16    5    512    83.5%
#  16    7    512    74.0%
#  16   10    512    60.8%
#  16   13    512    52.1%
#  16   15    512    49.8%
#  24    1   1152    99.2%
#  24    4   1152    94.1%
#  24    8   1152    82.9%
#  24   12   1152    70.2%
#  24   16   1152    59.2%
#  24   20   1152    52.0%
#  24   23   1152    49.9%
#  32    1   2048    99.5%
#  32    5   2048    95.0%
#  32   10   2048    84.9%
#  32   15   2048    73.0%
#  32   20   2048    62.0%
#  32   25   2048    54.0%
#  32   31   2048    50.0%

SYSTEM_CONFIGS <- data.frame(
  N     = c( 2,  3, 3,  4, 4,  6, 6, 6,  8, 8, 8, 10,10,10,
             16,16,16,16,16,16,16,
             24,24,24,24,24,24,24,
             32,32,32,32,32,32,32),
  reach = c( 1,  1, 2,  1, 3,  1, 3, 5,  1, 3, 5,  1, 3, 5,
             1, 3, 5, 7,10,13,15,
             1, 4, 8,12,16,20,23,
             1, 5,10,15,20,25,31)
)

N_REPS  <- 5L        # microbenchmark repetitions
WARMUP  <- 1L        # warmup runs (discarded)
NCORES  <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", detectCores()))
ABSTOL  <- 1e-6
RELTOL  <- 1e-6
TEND    <- 10.0

# --- Sparsity calculator (for reporting) ---
compute_sparsity <- function(N, reach) {
  nnz <- 0L
  for (i in seq_len(N)) {
    for (j in seq_len(N)) {
      n_nb <- 0L
      for (di in (-reach):reach) {
        for (dj in (-reach):reach) {
          ni <- i + di; nj <- j + dj
          if (ni >= 1 && ni <= N && nj >= 1 && nj <= N) n_nb <- n_nb + 1L
        }
      }
      nnz <- nnz + 2L * (n_nb + 1L)  # u-row + v-row, each: neighbors + 1 cross-coupling
    }
  }
  n <- 2L * N * N
  list(n_states = n, nnz = nnz, sparsity_pct = (1 - nnz / n^2) * 100)
}

# Pre-compute sparsity info
SYSTEM_CONFIGS$n_states <- 2 * SYSTEM_CONFIGS$N^2
for (i in seq_len(nrow(SYSTEM_CONFIGS))) {
  sp <- compute_sparsity(SYSTEM_CONFIGS$N[i], SYSTEM_CONFIGS$reach[i])
  SYSTEM_CONFIGS$nnz[i]          <- sp$nnz
  SYSTEM_CONFIGS$sparsity_pct[i] <- sp$sparsity_pct
}

cat("=== Benchmark config ===\n")
cat(sprintf("Repetitions:  %d (+ %d warmup)\n", N_REPS, WARMUP))
cat(sprintf("Cores:        %d\n", NCORES))
cat(sprintf("Systems:      %d\n", nrow(SYSTEM_CONFIGS)))
cat("\n--- System configurations ---\n")
print(SYSTEM_CONFIGS, row.names = FALSE)
cat("============================\n\n")

# --- Brusselator 2D generator with reach ---
make_brusselator2d <- function(N, reach = 1L) {
  A     <- 1.0
  B     <- 3.0
  alpha <- 0.02
  dx2   <- 1.0 / (N + 1)^2
  coeff <- alpha / dx2

  n_cells <- N * N
  states  <- character(2 * n_cells)
  rhs     <- character(2 * n_cells)

  idx <- function(i, j) (i - 1) * N + j

  for (i in seq_len(N)) {
    for (j in seq_len(N)) {
      k   <- idx(i, j)
      u_k <- sprintf("u%d", k)
      v_k <- sprintf("v%d", k)
      states[2 * k - 1] <- u_k
      states[2 * k]     <- v_k

      # Collect all neighbors within Chebyshev distance 'reach'
      u_terms <- character(0)
      v_terms <- character(0)
      n_nb    <- 0L

      for (di in (-reach):reach) {
        for (dj in (-reach):reach) {
          if (di == 0 && dj == 0) next  # skip self
          ni <- i + di; nj <- j + dj
          if (ni >= 1 && ni <= N && nj >= 1 && nj <= N) {
            nb <- idx(ni, nj)
            u_terms <- c(u_terms, sprintf("u%d", nb))
            v_terms <- c(v_terms, sprintf("v%d", nb))
            n_nb    <- n_nb + 1L
          }
        }
      }

      # Diffusion: coeff * (Sigma u_nb - n_nb * u_k)
      if (n_nb > 0) {
        lap_u <- sprintf("%g*(%s - %d*%s)", coeff,
                         paste(u_terms, collapse = " + "), n_nb, u_k)
        lap_v <- sprintf("%g*(%s - %d*%s)", coeff,
                         paste(v_terms, collapse = " + "), n_nb, v_k)
      } else {
        lap_u <- "0"
        lap_v <- "0"
      }

      # RHS: reaction + diffusion
      rhs[2 * k - 1] <- sprintf("%g + %s^2*%s - %g*%s + %s",
                                A, u_k, v_k, B + 1, u_k, lap_u)
      rhs[2 * k]     <- sprintf("%g*%s - %s^2*%s + %s",
                                B, u_k, u_k, v_k, lap_v)
    }
  }
  names(rhs) <- states
  rhs
}

# --- Build full config table ---
solver_configs <- expand.grid(
  sys_idx = seq_len(nrow(SYSTEM_CONFIGS)),
  method  = c("bdf", "rb4"),
  sparse  = c(FALSE, TRUE),
  deriv   = c(FALSE, TRUE),
  stringsAsFactors = FALSE
)
solver_configs$N            <- SYSTEM_CONFIGS$N[solver_configs$sys_idx]
solver_configs$reach        <- SYSTEM_CONFIGS$reach[solver_configs$sys_idx]
solver_configs$n_states     <- SYSTEM_CONFIGS$n_states[solver_configs$sys_idx]
solver_configs$sparsity_pct <- SYSTEM_CONFIGS$sparsity_pct[solver_configs$sys_idx]

solver_configs$label <- with(solver_configs, sprintf("N%d_r%d_%s_%s_%s",
                                                     N, reach, method,
                                                     ifelse(sparse, "sp", "dn"),
                                                     ifelse(deriv, "AD", "dbl")
))

cat(sprintf("Total solver configurations: %d\n\n", nrow(solver_configs)))

# --- Worker: compile + warmup + microbenchmark one config ---
run_one <- function(cfg) {
  N      <- cfg$N
  reach  <- cfg$reach
  method <- cfg$method
  sp     <- cfg$sparse
  dv     <- cfg$deriv
  lab    <- cfg$label

  tryCatch({
    cat(sprintf("[%s] Compiling (dim=%d, sparsity=%.1f%%) ...\n",
                lab, cfg$n_states, cfg$sparsity_pct))

    rhs      <- make_brusselator2d(N, reach)
    n_states <- length(rhs)

    # Unique modelname per config
    mname <- sprintf("br_%s", gsub("[^a-zA-Z0-9]", "", lab))

    model <- CppODE(
      rhs       = rhs,
      deriv     = dv,
      sparse    = sp,
      method    = method,
      modelname = mname,
      compile   = TRUE,
      verbose   = FALSE
    )

    times <- seq(0, TEND, length.out = 101)
    parms <- numeric(0)

    # Warmup (skip for large systems where each run is expensive)
    if (N < 16) {
      cat(sprintf("[%s] Warmup ...\n", lab))
      for (w in seq_len(WARMUP)) {
        res <- solveODE(model, times, parms, abstol = ABSTOL, reltol = RELTOL)
      }
    }

    # First real solve to capture diagnostics
    res <- solveODE(model, times, parms, abstol = ABSTOL, reltol = RELTOL)

    diag <- res$diagnostics
    if (diag$return_code != 0L) {
      warning(sprintf("[%s] Integration warning: %s", lab, diag$message))
    }

    # Microbenchmark
    cat(sprintf("[%s] Benchmarking %d reps ...\n", lab, N_REPS))
    mb <- microbenchmark(
      solveODE(model, times, parms, abstol = ABSTOL, reltol = RELTOL),
      times = N_REPS,
      unit  = "milliseconds"
    )

    cat(sprintf("[%s] Done: median=%.1f ms\n", lab, median(mb$time) / 1e6))

    data.frame(
      label        = lab,
      N            = N,
      reach        = reach,
      n_states     = n_states,
      sparsity_pct = cfg$sparsity_pct,
      method       = method,
      sparse       = sp,
      deriv        = dv,
      median_ms    = median(mb$time) / 1e6,
      mean_ms      = mean(mb$time) / 1e6,
      min_ms       = min(mb$time) / 1e6,
      max_ms       = max(mb$time) / 1e6,
      sd_ms        = sd(mb$time) / 1e6,
      accepted     = diag$accepted,
      rejected     = diag$rejected,
      fevals       = diag$fevals,
      jevals       = diag$jevals,
      setups       = diag$setups,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    cat(sprintf("[%s] ERROR: %s\n", lab, conditionMessage(e)))
    data.frame(
      label = lab, N = N, reach = reach, n_states = cfg$n_states,
      sparsity_pct = cfg$sparsity_pct,
      method = method, sparse = sp, deriv = dv,
      median_ms = NA, mean_ms = NA, min_ms = NA, max_ms = NA, sd_ms = NA,
      accepted = NA, rejected = NA, fevals = NA, jevals = NA, setups = NA,
      stringsAsFactors = FALSE
    )
  })
}

# --- Run all configs in parallel ---
cat(sprintf("Running %d configs on %d cores ...\n\n", nrow(solver_configs), NCORES))

cfg_list <- split(solver_configs, seq_len(nrow(solver_configs)))
results_list <- mclapply(cfg_list, run_one, mc.cores = NCORES)
results <- do.call(rbind, results_list)
rownames(results) <- NULL

# --- Save raw results ---
outfile <- "benchmark_sparse_threshold_results.csv"
write.csv(results, outfile, row.names = FALSE)
cat(sprintf("\nResults saved to %s\n", outfile))

# --- Summary table ---
cat("\n=== Summary (median ms) ===\n\n")
print(results[order(results$n_states, results$sparsity_pct, results$method,
                    results$sparse, results$deriv),
              c("label", "n_states", "sparsity_pct", "method", "sparse", "deriv",
                "median_ms", "sd_ms", "setups", "jevals")],
      row.names = FALSE)

# --- Speedup analysis: sparse vs dense ---
cat("\n=== Sparse vs Dense speedup (median) ===\n\n")
for (m in c("bdf", "rb4")) {
  for (dv in c(FALSE, TRUE)) {
    type_label <- if (dv) "AD" else "double"
    dense  <- results[results$method == m & !results$sparse & results$deriv == dv, ]
    sparse <- results[results$method == m &  results$sparse & results$deriv == dv, ]
    merged <- merge(dense, sparse, by = c("N", "reach"), suffixes = c(".dense", ".sparse"))
    if (nrow(merged) > 0) {
      merged$speedup <- merged$median_ms.dense / merged$median_ms.sparse
      merged <- merged[order(merged$n_states.dense, merged$sparsity_pct.dense), ]
      cat(sprintf("--- %s / %s ---\n", toupper(m), type_label))
      print(merged[, c("N", "reach", "n_states.dense", "sparsity_pct.dense",
                       "median_ms.dense", "median_ms.sparse", "speedup",
                       "setups.dense", "setups.sparse")],
            row.names = FALSE)
      cat("\n")
    }
  }
}

# ============================================================================
# Visualization
# ============================================================================
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)

  results$type_label   <- ifelse(results$deriv, "F<double>", "double")
  results$solver_label <- with(results, sprintf("%s / %s", method, type_label))
  results$lu_label     <- ifelse(results$sparse, "KLU (sparse)", "LAPACK (dense)")
  results$sparsity_band <- cut(results$sparsity_pct,
                               breaks = c(0, 50, 70, 85, 95, 100),
                               labels = c("<50%", "50-70%", "70-85%", "85-95%", ">95%"),
                               include.lowest = TRUE
  )

  # --- Plot 1: Runtime vs system dimension, colored by LU, shaped by sparsity ---
  p1 <- ggplot(results, aes(x = n_states, y = median_ms,
                            colour = lu_label, shape = sparsity_band)) +
    geom_point(size = 3, alpha = 0.8) +
    geom_errorbar(aes(ymin = pmax(0.1, median_ms - sd_ms),
                      ymax = median_ms + sd_ms),
                  width = 0.04, alpha = 0.4) +
    facet_wrap(~ solver_label, scales = "free_y") +
    scale_x_log10() +
    scale_y_log10() +
    labs(
      title = "Dense vs Sparse LU: Runtime Scaling",
      subtitle = sprintf("Brusselator 2D with variable reach, t=[0,%g], %d reps", TEND, N_REPS),
      x = "System dimension (2N^2)",
      y = "Median runtime (ms)",
      colour = "LU solver",
      shape  = "Sparsity"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")

  ggsave("benchmark_runtime_scaling.pdf", p1, width = 14, height = 9)
  cat("Plot saved: benchmark_runtime_scaling.pdf\n")

  # --- Plot 2: Speedup as function of sparsity ---
  speedup_data <- do.call(rbind, lapply(
    split(results, list(results$method, results$deriv)), function(sub) {
      dense  <- sub[!sub$sparse, c("N", "reach", "n_states", "sparsity_pct",
                                   "median_ms", "method", "type_label")]
      sparse <- sub[ sub$sparse, c("N", "reach", "median_ms")]
      m <- merge(dense, sparse, by = c("N", "reach"), suffixes = c(".dense", ".sparse"))
      if (nrow(m) == 0) return(NULL)
      m$speedup <- m$median_ms.dense / m$median_ms.sparse
      m$solver_label <- sprintf("%s / %s", m$method, m$type_label)
      m
    }
  ))

  if (!is.null(speedup_data) && nrow(speedup_data) > 0) {
    p2 <- ggplot(speedup_data, aes(x = sparsity_pct, y = speedup,
                                   colour = factor(n_states),
                                   shape = solver_label)) +
      geom_point(size = 3.5) +
      geom_hline(yintercept = 1, linetype = "dashed", colour = "grey40") +
      annotate("text", x = 40, y = 1.05,
               label = "breakeven", hjust = 0, colour = "grey40", size = 3.5) +
      scale_colour_viridis_d(name = "Dimension") +
      labs(
        title = "KLU Speedup over LAPACK vs Jacobian Sparsity",
        subtitle = "Values > 1 = KLU wins",
        x = "Jacobian sparsity (%)",
        y = "Speedup (dense_time / sparse_time)",
        shape = "Solver"
      ) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom")

    ggsave("benchmark_speedup_vs_sparsity.pdf", p2, width = 11, height = 7)
    cat("Plot saved: benchmark_speedup_vs_sparsity.pdf\n")

    # --- Plot 3: Speedup vs dimension, faceted by sparsity band ---
    speedup_data$sparsity_band <- cut(speedup_data$sparsity_pct,
                                      breaks = c(0, 50, 70, 85, 95, 100),
                                      labels = c("<50%", "50-70%", "70-85%", "85-95%", ">95%"),
                                      include.lowest = TRUE
    )

    p3 <- ggplot(speedup_data, aes(x = n_states, y = speedup,
                                   colour = solver_label, shape = solver_label)) +
      geom_point(size = 3) +
      geom_line(linewidth = 0.6, alpha = 0.6) +
      geom_hline(yintercept = 1, linetype = "dashed", colour = "grey40") +
      facet_wrap(~ sparsity_band, ncol = 3) +
      scale_x_log10() +
      labs(
        title = "KLU Speedup vs System Size by Sparsity Band",
        x = "System dimension",
        y = "Speedup (dense / sparse)",
        colour = "Solver",
        shape  = "Solver"
      ) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom")

    ggsave("benchmark_speedup_by_band.pdf", p3, width = 13, height = 7)
    cat("Plot saved: benchmark_speedup_by_band.pdf\n")
  }

  # --- Plot 4: LU setups ---
  p4 <- ggplot(results, aes(x = n_states, y = setups,
                            colour = lu_label, shape = sparsity_band)) +
    geom_point(size = 3, alpha = 0.8) +
    facet_wrap(~ solver_label, scales = "free_y") +
    scale_x_log10() +
    labs(
      title = "LU Setups (Factorizations) vs System Size",
      x = "System dimension (2N^2)",
      y = "Number of LU setups",
      colour = "LU solver",
      shape  = "Sparsity"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")

  ggsave("benchmark_setups.pdf", p4, width = 14, height = 9)
  cat("Plot saved: benchmark_setups.pdf\n")
}

cat(sprintf("\nBenchmark completed at %s\n", Sys.time()))
