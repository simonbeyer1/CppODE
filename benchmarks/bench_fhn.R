rm(list = ls(all.names = TRUE))

# ==========================================================================
#  Auto-install missing packages
# ==========================================================================

required_pkgs <- c("CppODE", "parallel", "microbenchmark",
                   "ggplot2", "dplyr", "tidyr", "patchwork")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing missing package: %s\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org",
                     quiet = TRUE)
  }
}

# Working directory
.workingDir <- file.path(tempdir(), "CppODE_bench_fhn")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(parallel)
library(microbenchmark)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

# ============================================================================
# Benchmark: FitzHugh-Nagumo Chain — Dense vs Sparse × BDF vs RB4
#
#   v_i' = v_i - v_i^3/3 - w_i + I_ext_i
#          + D_i * sum_{j in neighbors(i)} (v_j - v_i)
#   w_i' = eps_i * (v_i + a_i - b_i * w_i)
#
#   2*N states, parameters: eps_i, a_i, b_i, D_i, I_ext_i (per neuron)
#
#   Stiffness: controlled by eps (small eps => stiff)
#   Sparsity:  controlled by reach (coupling radius)
# ============================================================================

# --- Parallelization ---
n_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = "1"))
cat(sprintf("Using %d cores for mclapply\n", n_cores))

# --- Microbenchmark settings ---
N_REPS <- as.integer(Sys.getenv("BENCH_NREPS", unset = "5"))
cat(sprintf("Microbenchmark repetitions: %d\n", N_REPS))


# ==========================================================================
#  Model builder
# ==========================================================================

build_fhn <- function(N, reach) {
  v_names <- paste0("v", seq_len(N))
  w_names <- paste0("w", seq_len(N))

  eps_names  <- paste0("eps", seq_len(N))
  a_names    <- paste0("a", seq_len(N))
  b_names    <- paste0("b", seq_len(N))
  D_names    <- paste0("D", seq_len(N))
  Iext_names <- paste0("Iext", seq_len(N))

  rhs <- list()
  for (i in seq_len(N)) {
    v_eq <- sprintf("%s - %s^3/3 - %s + %s",
                    v_names[i], v_names[i], w_names[i], Iext_names[i])

    neighbors <- integer(0)
    for (r in seq_len(reach)) {
      if (i - r >= 1) neighbors <- c(neighbors, i - r)
      if (i + r <= N) neighbors <- c(neighbors, i + r)
    }
    if (length(neighbors) > 0) {
      coupling_terms <- paste0(
        sprintf("%s*(%s - %s)", D_names[i], v_names[neighbors], v_names[i]),
        collapse = " + "
      )
      v_eq <- paste0(v_eq, " + ", coupling_terms)
    }

    w_eq <- sprintf("%s*(%s + %s - %s*%s)",
                    eps_names[i], v_names[i], a_names[i], b_names[i], w_names[i])

    rhs[[v_names[i]]] <- v_eq
    rhs[[w_names[i]]] <- w_eq
  }

  ordered_names <- as.vector(rbind(v_names, w_names))
  rhs <- unlist(rhs[ordered_names])

  all_params <- c(eps_names, a_names, b_names, D_names, Iext_names)
  list(rhs = rhs, params = all_params,
       v_names = v_names, w_names = w_names,
       N = N, reach = reach)
}


# ==========================================================================
#  Compute actual Jacobian sparsity from structure
# ==========================================================================

compute_sparsity <- function(N, reach) {
  n <- 2 * N
  nnz <- 0
  for (i in seq_len(N)) {
    nnz <- nnz + 2  # dv_i/dv_i + dv_i/dw_i
    neighbors <- 0
    for (r in seq_len(reach)) {
      if (i - r >= 1) neighbors <- neighbors + 1
      if (i + r <= N) neighbors <- neighbors + 1
    }
    nnz <- nnz + neighbors  # dv_i/dv_j
    nnz <- nnz + 2          # dw_i/dv_i + dw_i/dw_i
  }
  1 - nnz / n^2
}


# ==========================================================================
#  Parameters and initial conditions
# ==========================================================================

make_fhn_parms <- function(N, eps = 0.01) {
  p <- c(
    setNames(rep(eps,  N), paste0("eps",  seq_len(N))),
    setNames(rep(0.7,  N), paste0("a",    seq_len(N))),
    setNames(rep(0.8,  N), paste0("b",    seq_len(N))),
    setNames(rep(0.5,  N), paste0("D",    seq_len(N))),
    setNames(rep(0.5,  N), paste0("Iext", seq_len(N)))
  )
  set.seed(42)
  p[paste0("Iext", seq_len(N))] <- 0.5 + 0.1 * runif(N, -1, 1)
  p
}

make_fhn_init <- function(N) {
  set.seed(123)
  ini <- numeric(2 * N)
  v_names <- paste0("v", seq_len(N))
  w_names <- paste0("w", seq_len(N))
  names(ini) <- as.vector(rbind(v_names, w_names))
  for (i in seq_len(N)) {
    ini[v_names[i]] <- if (i <= 3) 1.5 else -1.2
    ini[w_names[i]] <- -0.6
  }
  ini
}


# ==========================================================================
#  Benchmark grid
# ==========================================================================

N_values <- if (n_cores > 1) {
  c(2, 5, 10, 20, 50, 100, 200, 500)
} else {
  c(2, 5, 10, 20, 50, 100, 200)
}

eps_values    <- c(0.01, 0.001)
reach_values  <- c(1, 2, 5, 10, 20, 50)
methods       <- c("bdf", "rb4")
lu_types      <- c("dense", "sparse")
deriv_modes   <- c(FALSE, TRUE)

t_end  <- 10
n_tout <- 201
times  <- seq(0, t_end, length.out = n_tout)


# ==========================================================================
#  Build task list
# ==========================================================================

tasks <- expand.grid(
  N       = N_values,
  reach   = reach_values,
  eps     = eps_values,
  method  = methods,
  lu_type = lu_types,
  deriv   = deriv_modes,
  stringsAsFactors = FALSE
)

# reach must be < N
tasks <- tasks[tasks$reach < tasks$N, ]

# Add reach=0 baseline
baseline <- expand.grid(
  N       = N_values,
  reach   = 0,
  eps     = eps_values,
  method  = methods,
  lu_type = lu_types,
  deriv   = deriv_modes,
  stringsAsFactors = FALSE
)
tasks <- rbind(tasks, baseline)
tasks <- unique(tasks)

# Skip sparse for n_states <= 4
tasks <- tasks[!(tasks$lu_type == "sparse" & 2 * tasks$N <= 4), ]

# Skip deriv=TRUE for large systems (N>=100: 500+ AD components, too expensive)
tasks <- tasks[!(tasks$deriv == TRUE & tasks$N >= 100), ]

# Metadata
tasks$n_states <- 2 * tasks$N
tasks$sparsity <- mapply(compute_sparsity, tasks$N, pmax(tasks$reach, 1))

# Sort for efficient compilation (group by model structure)
tasks <- tasks[order(tasks$N, tasks$reach, tasks$deriv, tasks$method, tasks$lu_type), ]
rownames(tasks) <- NULL

cat(sprintf("\n=== Benchmark grid: %d tasks ===\n", nrow(tasks)))
cat(sprintf("N range: %d - %d (states: %d - %d)\n",
            min(tasks$N), max(tasks$N),
            min(tasks$n_states), max(tasks$n_states)))
cat(sprintf("Reach values: %s\n", paste(sort(unique(tasks$reach)), collapse = ", ")))
cat(sprintf("Eps values: %s\n", paste(sort(unique(tasks$eps)), collapse = ", ")))


# ==========================================================================
#  Run a single benchmark task using microbenchmark
# ==========================================================================

# --- Cleanup: unload .so and delete temp files after each task ---
unload_model <- function(model_name) {
  so_ext <- .Platform$dynlib.ext
  loaded <- getLoadedDLLs()
  if (model_name %in% names(loaded)) {
    so_path <- loaded[[model_name]][["path"]]
    try(dyn.unload(so_path), silent = TRUE)
    # Delete .so and .o to free disk space
    try(unlink(so_path), silent = TRUE)
    o_path <- sub(paste0("\\", so_ext, "$"), ".o", so_path)
    try(unlink(o_path), silent = TRUE)
  }
  # Also remove the .cpp source
  cpp_path <- file.path(tempdir(), paste0(model_name, ".cpp"))
  try(unlink(cpp_path), silent = TRUE)
  invisible(NULL)
}


run_single_benchmark <- function(task_row) {
  N       <- task_row$N
  reach   <- task_row$reach
  eps_val <- task_row$eps
  meth    <- task_row$method
  lu      <- task_row$lu_type
  dv      <- task_row$deriv

  actual_reach <- max(reach, 0)
  model_def <- build_fhn(N, actual_reach)

  model_name <- sprintf("fhn_N%d_r%d_e%s_%s_%s_d%d",
                        N, actual_reach,
                        gsub("\\.", "", as.character(eps_val)),
                        meth, lu, as.integer(dv))

  # Guarantee cleanup even on error/early return
  on.exit(unload_model(model_name), add = TRUE)

  # --- Compile ---
  t_compile <- system.time({
    compiled <- tryCatch(
      CppODE(model_def$rhs,
             modelname = model_name,
             outdir    = getwd(),
             sparse    = (lu == "sparse"),
             method    = meth,
             deriv     = dv,
             deriv2    = FALSE,
             compile   = TRUE,
             verbose   = FALSE),
      error = function(e) { message("Compile error: ", e$message); NULL }
    )
  })["elapsed"]

  sparsity_val <- compute_sparsity(N, max(actual_reach, 1))
  fail_row <- data.frame(
    N = N, n_states = 2*N, reach = actual_reach,
    sparsity = sparsity_val,
    eps = eps_val, method = meth, lu_type = lu, deriv = dv,
    t_compile = as.numeric(t_compile),
    t_median = NA_real_, t_mean = NA_real_,
    t_min = NA_real_, t_max = NA_real_, t_sd = NA_real_,
    n_reps = 0L,
    n_steps = NA_integer_, n_fevals = NA_integer_, n_jevals = NA_integer_,
    status = "compile_fail",
    stringsAsFactors = FALSE
  )

  if (is.null(compiled)) return(fail_row)

  parms <- make_fhn_parms(N, eps = eps_val)
  ini   <- make_fhn_init(N)
  parms_ini <- c(ini, parms)

  # --- Warmup run (captures diagnostics + detects solve failures) ---
  warmup <- tryCatch(
    solveODE(compiled, times, parms_ini,
             abstol = 1e-8, reltol = 1e-8, maxsteps = 1e6L),
    error = function(e) { message("Solve error: ", e$message); NULL }
  )

  if (is.null(warmup)) {
    fail_row$status <- "solve_fail"
    return(fail_row)
  }

  diag <- warmup$diagnostics

  # --- Microbenchmark ---
  mb <- microbenchmark(
    solveODE(compiled, times, parms_ini,
             abstol = 1e-8, reltol = 1e-8, maxsteps = 1e6L),
    times = N_REPS,
    unit = "s"
  )

  t_sec <- mb$time / 1e9

  data.frame(
    N = N, n_states = 2*N, reach = actual_reach,
    sparsity = sparsity_val,
    eps = eps_val, method = meth, lu_type = lu, deriv = dv,
    t_compile = as.numeric(t_compile),
    t_median  = median(t_sec),
    t_mean    = mean(t_sec),
    t_min     = min(t_sec),
    t_max     = max(t_sec),
    t_sd      = sd(t_sec),
    n_reps    = N_REPS,
    n_steps   = if (!is.null(diag$accepted))  diag$accepted + diag$rejected else NA_integer_,
    n_fevals  = if (!is.null(diag$fevals)) diag$fevals else NA_integer_,
    n_jevals  = if (!is.null(diag$jevals)) diag$jevals else NA_integer_,
    status    = "ok",
    stringsAsFactors = FALSE
  )
}


# ==========================================================================
#  Execute benchmarks
# ==========================================================================

cat(sprintf("\nRunning benchmarks (%d tasks, %d reps each)...\n",
            nrow(tasks), N_REPS))

t_total <- system.time({
  if (n_cores > 1) {
    task_list <- split(tasks, seq_len(nrow(tasks)))
    results_list <- mclapply(task_list, run_single_benchmark,
                             mc.cores = n_cores)
  } else {
    results_list <- vector("list", nrow(tasks))
    for (i in seq_len(nrow(tasks))) {
      cat(sprintf("  [%3d/%d] N=%3d reach=%2d eps=%.3f %s %6s deriv=%-5s ...",
                  i, nrow(tasks),
                  tasks$N[i], tasks$reach[i], tasks$eps[i],
                  tasks$method[i], tasks$lu_type[i], tasks$deriv[i]))
      results_list[[i]] <- run_single_benchmark(tasks[i, ])
      r <- results_list[[i]]
      if (r$status == "ok") {
        cat(sprintf(" median=%.3fs sd=%.4fs (%s)\n",
                    r$t_median, r$t_sd, r$status))
      } else {
        cat(sprintf(" %s\n", r$status))
      }
      # Free memory between tasks
      gc(verbose = FALSE)

      # Periodic memory report
      if (i %% 20 == 0) {
        mem <- gc(verbose = FALSE)
        n_dlls <- length(getLoadedDLLs())
        cat(sprintf("    [mem] %.0f MB used, %d DLLs loaded\n",
                    sum(mem[, 2]), n_dlls))
      }
    }
  }
})

df <- do.call(rbind, results_list)
rownames(df) <- NULL

cat(sprintf("\n=== Total benchmark time: %.1f s ===\n", t_total["elapsed"]))
cat(sprintf("Successful: %d / %d\n", sum(df$status == "ok"), nrow(df)))

write.csv(df, "benchmark_fhn_results.csv", row.names = FALSE)
cat("Results saved to benchmark_fhn_results.csv\n")


# ==========================================================================
#  Visualization
# ==========================================================================

df_ok <- df[df$status == "ok" & !is.na(df$t_median), ]
if (nrow(df_ok) == 0) {
  cat("No successful runs to plot.\n")
  quit(save = "no")
}

# Labels
df_ok$eps_label   <- ifelse(df_ok$eps == 0.01, "moderate (eps=0.01)",
                            "stiff (eps=0.001)")
df_ok$deriv_label <- ifelse(df_ok$deriv, "deriv=TRUE", "deriv=FALSE")
df_ok$sparsity_pct <- round(df_ok$sparsity * 100)


# --- Sparse/dense ratio per matched pair (using median times) ---
df_wide <- df_ok %>%
  select(N, n_states, reach, sparsity_pct, eps, eps_label,
         method, deriv, deriv_label, lu_type, t_median, t_sd) %>%
  pivot_wider(names_from = lu_type,
              values_from = c(t_median, t_sd),
              names_sep = "_") %>%
  filter(!is.na(t_median_dense), !is.na(t_median_sparse)) %>%
  mutate(
    ratio  = t_median_dense / t_median_sparse,   # >1 = sparse wins
    winner = ifelse(ratio > 1, "sparse", "dense"),
    # Propagate uncertainty: ratio_se via delta method
    ratio_se = ratio * sqrt((t_sd_dense / t_median_dense)^2 +
                              (t_sd_sparse / t_median_sparse)^2)
  )


# -------------------------------------------------------------------------
#  Plot 1: Heatmap — dense/sparse ratio by n_states × sparsity
# -------------------------------------------------------------------------

heat_data <- df_wide %>%
  group_by(n_states, sparsity_pct, method, eps_label, deriv_label) %>%
  summarise(ratio = mean(ratio, na.rm = TRUE), .groups = "drop")

p_heat <- ggplot(heat_data,
                 aes(x = factor(n_states), y = factor(sparsity_pct),
                     fill = log2(ratio))) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f", ratio)),
            size = 2.5, color = "black") +
  scale_fill_gradient2(
    low = "#d73027", mid = "white", high = "#1a9850",
    midpoint = 0, name = "log2(dense/sparse)\n>0 = sparse wins"
  ) +
  facet_grid(eps_label + deriv_label ~ method, scales = "free") +
  labs(x = "Number of states (2N)", y = "Jacobian sparsity (%)",
       title = "Dense vs Sparse: Speed Ratio (dense_time / sparse_time)") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(size = 8))


# -------------------------------------------------------------------------
#  Plot 2: Crossover curves with error bars
# -------------------------------------------------------------------------

df_wide$sparsity_band <- cut(df_wide$sparsity_pct,
                             breaks = c(0, 60, 75, 85, 92, 100),
                             labels = c("<60%", "60-75%", "75-85%",
                                        "85-92%", ">92%"),
                             include.lowest = TRUE)

cross_data <- df_wide %>%
  filter(!is.na(sparsity_band)) %>%
  group_by(n_states, sparsity_band, method, eps_label, deriv_label) %>%
  summarise(ratio_mean = mean(ratio, na.rm = TRUE),
            ratio_lo   = mean(ratio, na.rm = TRUE) - mean(ratio_se, na.rm = TRUE),
            ratio_hi   = mean(ratio, na.rm = TRUE) + mean(ratio_se, na.rm = TRUE),
            .groups = "drop")

p_cross <- ggplot(cross_data,
                  aes(x = n_states, y = ratio_mean,
                      color = sparsity_band, fill = sparsity_band)) +
  geom_ribbon(aes(ymin = ratio_lo, ymax = ratio_hi),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  annotate("text", x = Inf, y = 1.05, label = "sparse wins above",
           hjust = 1.1, vjust = 0, size = 2.5, color = "grey40") +
  scale_x_log10(breaks = sort(unique(cross_data$n_states))) +
  scale_y_log10() +
  facet_grid(eps_label + deriv_label ~ method) +
  labs(x = "Number of states (2N)", y = "Ratio dense/sparse (log scale)",
       color = "Sparsity band", fill = "Sparsity band",
       title = "Dense vs Sparse Crossover (median times, +/- SE)") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")


# -------------------------------------------------------------------------
#  Plot 3: Absolute solve times (median + error bars)
# -------------------------------------------------------------------------

p_abs <- ggplot(df_ok,
                aes(x = n_states, y = t_median,
                    color = lu_type, linetype = method)) +
  geom_ribbon(aes(ymin = t_median - t_sd, ymax = t_median + t_sd,
                  fill = lu_type, group = interaction(lu_type, method)),
              alpha = 0.1, color = NA) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  scale_x_log10(breaks = sort(unique(df_ok$n_states))) +
  scale_y_log10() +
  facet_grid(eps_label ~ deriv_label) +
  labs(x = "Number of states (2N)", y = "Solve time (s, log scale)",
       color = "LU type", linetype = "Method", fill = "LU type",
       title = "Absolute Solve Times (median +/- SD)") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")


# -------------------------------------------------------------------------
#  Plot 4: Compile times
# -------------------------------------------------------------------------

compile_data <- df_ok %>%
  group_by(n_states, method, deriv_label) %>%
  summarise(t_compile = mean(t_compile, na.rm = TRUE), .groups = "drop")

p_compile <- ggplot(compile_data,
                    aes(x = factor(n_states), y = t_compile, fill = method)) +
  geom_col(position = "dodge", width = 0.7) +
  facet_wrap(~ deriv_label) +
  labs(x = "Number of states (2N)", y = "Compile time (s)",
       fill = "Method", title = "Compilation Times") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# -------------------------------------------------------------------------
#  Crossover summary
# -------------------------------------------------------------------------

threshold_data <- df_wide %>%
  group_by(method, eps_label, deriv_label) %>%
  filter(ratio > 1) %>%
  summarise(
    min_n_sparse_wins = min(n_states),
    min_sparsity_sparse_wins = min(sparsity_pct),
    .groups = "drop"
  )

cat("\n========= CROSSOVER SUMMARY ============\n")
if (nrow(threshold_data) > 0) {
  for (i in seq_len(nrow(threshold_data))) {
    r <- threshold_data[i, ]
    cat(sprintf("  %s | %s | %s: sparse wins from n_states >= %d (sparsity >= %d%%)\n",
                r$method, r$eps_label, r$deriv_label,
                r$min_n_sparse_wins, r$min_sparsity_sparse_wins))
  }
} else {
  cat("  Dense wins everywhere in this grid.\n")
}


# -------------------------------------------------------------------------
#  Combine and save
# -------------------------------------------------------------------------

p_page1 <- p_heat / p_cross +
  plot_layout(heights = c(1, 1)) +
  plot_annotation(
    title = "FitzHugh-Nagumo Chain: Dense vs Sparse Benchmark",
    subtitle = sprintf("N = %s neurons | reach = %s | eps = %s | %d reps (microbenchmark)",
                       paste(sort(unique(df_ok$N)), collapse = ", "),
                       paste(sort(unique(df_ok$reach)), collapse = ", "),
                       paste(sort(unique(df_ok$eps)), collapse = ", "),
                       N_REPS),
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

p_page2 <- p_abs / p_compile +
  plot_layout(heights = c(2, 1)) +
  plot_annotation(
    title = "FitzHugh-Nagumo Chain: Solve & Compile Times",
    subtitle = sprintf("%d reps per configuration (microbenchmark)", N_REPS),
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

ggsave("benchmark_fhn_crossover.pdf", p_page1,
       width = 14, height = 16, dpi = 150)
ggsave("benchmark_fhn_times.pdf", p_page2,
       width = 14, height = 12, dpi = 150)

cat("\nPlots saved:\n")
cat("  benchmark_fhn_crossover.pdf  (heatmap + crossover)\n")
cat("  benchmark_fhn_times.pdf      (absolute times + compile)\n")
cat("  benchmark_fhn_results.csv    (raw data)\n")
cat("\nDone.\n")
