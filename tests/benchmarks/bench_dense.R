rm(list = ls(all.names = TRUE))

# ==========================================================================
#  Auto-install missing packages
# ==========================================================================

required_pkgs <- c("CppODE", "parallel", "microbenchmark",
                   "ggplot2", "dplyr", "tidyr", "patchwork", "scales")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing missing package: %s\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org",
                     quiet = TRUE)
  }
}

# Working directory
.workingDir <- file.path(tempdir(), "CppODE_bench_dense")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(parallel)
library(microbenchmark)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(scales)

# ============================================================================
# Benchmark: Dense vs Sparse for LOW-sparsity systems
#
# Two models with tunable Jacobian density:
#
# 1) CT-RNN (Continuous-Time Recurrent Neural Network)
#    dx_i/dt = -alpha_i * x_i + sum_j W_ij * tanh(x_j) + b_i
#    N states, Jacobian density = density of W
#    Stiffness via alpha spread
#
# 2) Coupled Lorenz oscillators
#    dx_i/dt = sigma*(y_i - x_i) + c * sum_j G_ij * (x_j - x_i)
#    dy_i/dt = x_i*(rho - z_i) - y_i
#    dz_i/dt = x_i*y_i - beta*z_i
#    3*N states, Jacobian density controlled by coupling graph G
#    Internally nichtlinear (bilinear terms x*z, x*y)
# ============================================================================

# --- Parallelization ---
n_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = "1"))
cat(sprintf("Using %d cores for mclapply\n", n_cores))

# --- Microbenchmark settings ---
N_REPS <- as.integer(Sys.getenv("BENCH_NREPS", unset = "20"))
cat(sprintf("Microbenchmark repetitions: %d\n", N_REPS))


# ==========================================================================
#  Model builders
# ==========================================================================

# --- CT-RNN ---
# N states, density p in [0,1] controls fraction of nonzero W_ij
# Each W_ij != 0 creates a Jacobian entry: dfi/dxj = W_ij * sech^2(x_j)
# Sparsity = 1 - (N + nnz(W)) / N^2   (diagonal always present)
# Parameters: alpha_i, b_i (per neuron), plus W_ij entries as parameters
# For simplicity: W_ij are hardcoded constants, only alpha_i, b_i are parameters

build_ctrnn <- function(N, density) {
  x_names <- paste0("x", seq_len(N))
  alpha_names <- paste0("alpha", seq_len(N))
  b_names <- paste0("b", seq_len(N))

  # Generate random connectivity mask with given density
  set.seed(17 + N * 1000 + round(density * 100))
  W <- matrix(0, N, N)
  for (i in seq_len(N)) {
    for (j in seq_len(N)) {
      if (runif(1) < density) {
        W[i, j] <- rnorm(1, sd = 1.0 / sqrt(N))
      }
    }
  }
  # Diagonal always present (self-decay)
  diag(W) <- rnorm(N, sd = 0.5 / sqrt(N))

  rhs <- character(N)
  names(rhs) <- x_names

  for (i in seq_len(N)) {
    # Self-decay
    terms <- sprintf("-%s*%s", alpha_names[i], x_names[i])

    # Weighted tanh coupling
    for (j in seq_len(N)) {
      if (abs(W[i, j]) > 1e-15) {
        w_str <- sprintf("%.8f", W[i, j])
        terms <- c(terms, sprintf("%s*tanh(%s)", w_str, x_names[j]))
      }
    }

    # Bias
    terms <- c(terms, b_names[i])

    rhs[x_names[i]] <- paste(terms, collapse = " + ")
  }

  all_params <- c(alpha_names, b_names)

  # Actual sparsity from W structure
  nnz_W <- sum(abs(W) > 1e-15)
  actual_sparsity <- 1 - nnz_W / N^2

  list(rhs = rhs, params = all_params, N = N, density = density,
       n_states = N, actual_sparsity = actual_sparsity, model = "ctrnn")
}


# --- Coupled Lorenz ---
# N oscillators, 3*N states, coupling density p controls G_ij
# Parameters: sigma_i, rho_i, beta_i, c_i (per oscillator)

build_lorenz <- function(N, density) {
  x_names <- paste0("lx", seq_len(N))
  y_names <- paste0("ly", seq_len(N))
  z_names <- paste0("lz", seq_len(N))

  sigma_names <- paste0("sigma", seq_len(N))
  rho_names   <- paste0("rho", seq_len(N))
  beta_names  <- paste0("beta", seq_len(N))
  c_names     <- paste0("coup", seq_len(N))

  # Random coupling graph
  set.seed(31 + N * 1000 + round(density * 100))
  G <- matrix(0, N, N)
  for (i in seq_len(N)) {
    for (j in seq_len(N)) {
      if (i != j && runif(1) < density) {
        G[i, j] <- 1
      }
    }
  }

  n_states <- 3 * N
  rhs <- character(n_states)
  state_names <- character(n_states)

  for (i in seq_len(N)) {
    idx_x <- 3 * (i - 1) + 1
    idx_y <- 3 * (i - 1) + 2
    idx_z <- 3 * (i - 1) + 3

    state_names[idx_x] <- x_names[i]
    state_names[idx_y] <- y_names[i]
    state_names[idx_z] <- z_names[i]

    # dx/dt = sigma*(y - x) + c * sum G_ij * (x_j - x_i)
    x_eq <- sprintf("%s*(%s - %s)", sigma_names[i], y_names[i], x_names[i])
    neighbors <- which(G[i, ] > 0)
    if (length(neighbors) > 0) {
      coupling <- paste0(
        sprintf("%s*(%s - %s)", c_names[i], x_names[neighbors], x_names[i]),
        collapse = " + "
      )
      x_eq <- paste0(x_eq, " + ", coupling)
    }

    # dy/dt = x*(rho - z) - y
    y_eq <- sprintf("%s*(%s - %s) - %s", x_names[i], rho_names[i], z_names[i], y_names[i])

    # dz/dt = x*y - beta*z
    z_eq <- sprintf("%s*%s - %s*%s", x_names[i], y_names[i], beta_names[i], z_names[i])

    rhs[idx_x] <- x_eq
    rhs[idx_y] <- y_eq
    rhs[idx_z] <- z_eq
  }

  names(rhs) <- state_names
  all_params <- c(sigma_names, rho_names, beta_names, c_names)

  # Sparsity: each oscillator has a 3x3 dense internal block = 9*N entries
  # Plus coupling: each G_ij=1 adds dx_i/dx_j entry = nnz(G) entries
  nnz_G <- sum(G > 0)
  nnz_total <- 9 * N + nnz_G
  actual_sparsity <- 1 - nnz_total / n_states^2

  list(rhs = rhs, params = all_params, N = N, density = density,
       n_states = n_states, actual_sparsity = actual_sparsity, model = "lorenz")
}


# ==========================================================================
#  Parameter generators
# ==========================================================================

make_ctrnn_parms <- function(N, stiff = FALSE) {
  set.seed(42)
  if (stiff) {
    alphas <- exp(runif(N, log(0.1), log(100)))  # 3 orders spread
  } else {
    alphas <- runif(N, 0.5, 2.0)
  }
  p <- c(
    setNames(alphas, paste0("alpha", seq_len(N))),
    setNames(rnorm(N, 0, 0.3), paste0("b", seq_len(N)))
  )
  p
}

make_ctrnn_init <- function(N) {
  set.seed(123)
  ini <- rnorm(N, 0, 0.5)
  names(ini) <- paste0("x", seq_len(N))
  ini
}

make_lorenz_parms <- function(N, stiff = FALSE) {
  set.seed(42)
  if (stiff) {
    sigmas <- runif(N, 5, 20)
    rhos   <- runif(N, 20, 35)
    betas  <- runif(N, 1, 5)
  } else {
    sigmas <- rep(10, N)
    rhos   <- rep(28, N)
    betas  <- rep(8/3, N)
  }
  p <- c(
    setNames(sigmas, paste0("sigma", seq_len(N))),
    setNames(rhos,   paste0("rho", seq_len(N))),
    setNames(betas,  paste0("beta", seq_len(N))),
    setNames(rep(0.5, N), paste0("coup", seq_len(N)))
  )
  p
}

make_lorenz_init <- function(N) {
  set.seed(123)
  ini <- numeric(3 * N)
  x_names <- paste0("lx", seq_len(N))
  y_names <- paste0("ly", seq_len(N))
  z_names <- paste0("lz", seq_len(N))
  state_names <- character(3 * N)
  for (i in seq_len(N)) {
    idx <- 3 * (i - 1)
    state_names[idx + 1] <- x_names[i]
    state_names[idx + 2] <- y_names[i]
    state_names[idx + 3] <- z_names[i]
    ini[idx + 1] <- 1.0 + 0.1 * rnorm(1)
    ini[idx + 2] <- 1.0 + 0.1 * rnorm(1)
    ini[idx + 3] <- 25.0 + 0.1 * rnorm(1)
  }
  names(ini) <- state_names
  ini
}


# ==========================================================================
#  Benchmark grid
# ==========================================================================

# CT-RNN: N states directly
ctrnn_N_values <- if (n_cores > 1) {
  c(5, 10, 20, 50, 100, 200, 500, 1000)
} else {
  c(5, 10, 20, 50, 100, 200, 500)
}

# Lorenz: 3*N states, so N_osc values give 3*N states
lorenz_N_values <- if (n_cores > 1) {
  c(2, 5, 10, 20, 50, 100, 200, 333)  # 333*3 = 999 ~ 1000
} else {
  c(2, 5, 10, 20, 50, 100, 200)
}

# Density values — the key parameter for LOW sparsity
density_values <- c(0.1, 0.3, 0.5, 0.7, 0.9)

# Stiffness
stiff_values   <- c(FALSE, TRUE)

# Solver settings
methods   <- c("bdf", "rb4")
lu_types  <- c("dense", "sparse")
deriv_modes <- c(FALSE, TRUE)

t_end  <- 20
n_tout <- 201
times  <- seq(0, t_end, length.out = n_tout)


# ==========================================================================
#  Build task list
# ==========================================================================

# --- CT-RNN tasks ---
ctrnn_tasks <- expand.grid(
  N       = ctrnn_N_values,
  density = density_values,
  stiff   = stiff_values,
  method  = methods,
  lu_type = lu_types,
  deriv   = deriv_modes,
  stringsAsFactors = FALSE
)
ctrnn_tasks$model <- "ctrnn"
ctrnn_tasks$n_states <- ctrnn_tasks$N

# --- Lorenz tasks ---
lorenz_tasks <- expand.grid(
  N       = lorenz_N_values,
  density = density_values,
  stiff   = stiff_values,
  method  = methods,
  lu_type = lu_types,
  deriv   = deriv_modes,
  stringsAsFactors = FALSE
)
lorenz_tasks$model <- "lorenz"
lorenz_tasks$n_states <- 3 * lorenz_tasks$N

# --- Combine ---
tasks <- rbind(ctrnn_tasks, lorenz_tasks)

# Skip sparse for n_states <= 4
tasks <- tasks[!(tasks$lu_type == "sparse" & tasks$n_states <= 4), ]

# Skip deriv=TRUE for n_states > 200
# CT-RNN: N params = 2*N, total sens = N + 2*N = 3*N
# Lorenz: N params = 4*N_osc, total sens = 3*N_osc + 4*N_osc = 7*N_osc
tasks <- tasks[!(tasks$deriv == TRUE & tasks$n_states > 200), ]

# Sort
tasks <- tasks[order(tasks$model, tasks$N, tasks$density, tasks$deriv,
                     tasks$method, tasks$lu_type), ]
rownames(tasks) <- NULL

cat(sprintf("\n=== Benchmark grid: %d tasks ===\n", nrow(tasks)))
cat(sprintf("CT-RNN: N = %s\n", paste(ctrnn_N_values, collapse = ", ")))
cat(sprintf("Lorenz: N_osc = %s (states: %s)\n",
            paste(lorenz_N_values, collapse = ", "),
            paste(3 * lorenz_N_values, collapse = ", ")))
cat(sprintf("Density values: %s\n", paste(density_values, collapse = ", ")))


# ==========================================================================
#  Cleanup helper
# ==========================================================================

unload_model <- function(model_name) {
  so_ext <- .Platform$dynlib.ext
  loaded <- getLoadedDLLs()
  if (model_name %in% names(loaded)) {
    so_path <- loaded[[model_name]][["path"]]
    try(dyn.unload(so_path), silent = TRUE)
    try(unlink(so_path), silent = TRUE)
    o_path <- sub(paste0("\\", so_ext, "$"), ".o", so_path)
    try(unlink(o_path), silent = TRUE)
  }
  cpp_path <- file.path(tempdir(), paste0(model_name, ".cpp"))
  try(unlink(cpp_path), silent = TRUE)
  invisible(NULL)
}


# ==========================================================================
#  Run a single benchmark task
# ==========================================================================

run_single_benchmark <- function(task_row) {
  N       <- task_row$N
  dens    <- task_row$density
  stf     <- task_row$stiff
  meth    <- task_row$method
  lu      <- task_row$lu_type
  dv      <- task_row$deriv
  mdl     <- task_row$model

  # Build model
  if (mdl == "ctrnn") {
    model_def <- build_ctrnn(N, dens)
    parms     <- make_ctrnn_parms(N, stiff = stf)
    ini       <- make_ctrnn_init(N)
    n_st      <- N
  } else {
    model_def <- build_lorenz(N, dens)
    parms     <- make_lorenz_parms(N, stiff = stf)
    ini       <- make_lorenz_init(N)
    n_st      <- 3 * N
  }

  actual_sparsity <- model_def$actual_sparsity

  model_name <- sprintf("%s_N%d_d%d_s%d_%s_%s_d%d",
                        mdl, N, round(dens * 100), as.integer(stf),
                        meth, lu, as.integer(dv))

  on.exit(unload_model(model_name), add = TRUE)

  # --- Compile ---
  t_compile <- system.time({
    compiled <- tryCatch(
      CppODE(model_def$rhs,
             modelname = model_name,
             sparse    = (lu == "sparse"),
             method    = meth,
             deriv     = dv,
             deriv2    = FALSE,
             compile   = TRUE,
             verbose   = FALSE),
      error = function(e) { message("Compile error [", model_name, "]: ", e$message); NULL }
    )
  })["elapsed"]

  fail_row <- data.frame(
    model = mdl, N = N, n_states = n_st, density = dens,
    sparsity = actual_sparsity,
    stiff = stf, method = meth, lu_type = lu, deriv = dv,
    t_compile = as.numeric(t_compile),
    t_median = NA_real_, t_mean = NA_real_,
    t_min = NA_real_, t_max = NA_real_, t_sd = NA_real_,
    n_reps = 0L,
    n_steps = NA_integer_, n_fevals = NA_integer_, n_jevals = NA_integer_,
    status = "compile_fail",
    stringsAsFactors = FALSE
  )

  if (is.null(compiled)) return(fail_row)

  parms_ini <- c(ini, parms)

  # --- Warmup ---
  warmup <- tryCatch(
    solveODE(compiled, times, parms_ini,
             abstol = 1e-8, reltol = 1e-8, maxsteps = 1e6L),
    error = function(e) { message("Solve error [", model_name, "]: ", e$message); NULL }
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
    model = mdl, N = N, n_states = n_st, density = dens,
    sparsity = actual_sparsity,
    stiff = stf, method = meth, lu_type = lu, deriv = dv,
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
      cat(sprintf("  [%3d/%d] %6s N=%3d dens=%.1f stiff=%d %s %6s deriv=%-5s ...",
                  i, nrow(tasks), tasks$model[i],
                  tasks$N[i], tasks$density[i], tasks$stiff[i],
                  tasks$method[i], tasks$lu_type[i], tasks$deriv[i]))
      results_list[[i]] <- run_single_benchmark(tasks[i, ])
      r <- results_list[[i]]
      if (r$status == "ok") {
        cat(sprintf(" median=%.3fs sd=%.4fs (%s)\n",
                    r$t_median, r$t_sd, r$status))
      } else {
        cat(sprintf(" %s\n", r$status))
      }
      gc(verbose = FALSE)
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

write.csv(df, "benchmark_dense_results.csv", row.names = FALSE)
cat("Results saved to benchmark_dense_results.csv\n")


# ==========================================================================
#  Visualization — pseudolog scale to avoid NaN warnings
# ==========================================================================

# Pseudolog: handles values near zero and negative without NaN
pseudo_log_trans <- trans_new(
  name      = "pseudo_log",
  transform = function(x) asinh(x / 2) / log(10),
  inverse   = function(x) 2 * sinh(x * log(10)),
  breaks    = log_breaks(),
  domain    = c(-Inf, Inf)
)

df_ok <- df[df$status == "ok" & !is.na(df$t_median), ]
if (nrow(df_ok) == 0) {
  cat("No successful runs to plot.\n")
  quit(save = "no")
}

# Labels
df_ok$stiff_label <- ifelse(df_ok$stiff, "stiff", "moderate")
df_ok$deriv_label <- ifelse(df_ok$deriv, "deriv=TRUE", "deriv=FALSE")
df_ok$sparsity_pct <- round(df_ok$sparsity * 100)
df_ok$density_pct  <- round(df_ok$density * 100)
df_ok$model_label  <- ifelse(df_ok$model == "ctrnn", "CT-RNN", "Coupled Lorenz")


# --- Sparse/dense ratio per matched pair ---
df_wide <- df_ok %>%
  select(model, model_label, N, n_states, density, density_pct, sparsity_pct,
         stiff, stiff_label, method, deriv, deriv_label, lu_type,
         t_median, t_sd) %>%
  pivot_wider(names_from = lu_type,
              values_from = c(t_median, t_sd),
              names_sep = "_") %>%
  filter(!is.na(t_median_dense), !is.na(t_median_sparse)) %>%
  mutate(
    ratio  = t_median_dense / t_median_sparse,
    winner = ifelse(ratio > 1, "sparse", "dense"),
    ratio_se = ratio * sqrt(
      (t_sd_dense / pmax(t_median_dense, 1e-15))^2 +
        (t_sd_sparse / pmax(t_median_sparse, 1e-15))^2
    )
  )


# -------------------------------------------------------------------------
#  Plot 1: Heatmap — ratio by n_states × density (not sparsity!)
# -------------------------------------------------------------------------

heat_data <- df_wide %>%
  group_by(model_label, n_states, density_pct, method, stiff_label, deriv_label) %>%
  summarise(ratio = mean(ratio, na.rm = TRUE), .groups = "drop")

p_heat <- ggplot(heat_data,
                 aes(x = factor(n_states), y = factor(density_pct),
                     fill = log2(ratio))) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f", ratio)),
            size = 2.2, color = "black") +
  scale_fill_gradient2(
    low = "#d73027", mid = "white", high = "#1a9850",
    midpoint = 0, name = "log2(dense/sparse)\n>0 = sparse wins"
  ) +
  facet_grid(model_label + stiff_label + deriv_label ~ method, scales = "free") +
  labs(x = "Number of states", y = "Jacobian density (%)",
       title = "Dense vs Sparse: Speed Ratio by System Size and Density") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(size = 7))


# -------------------------------------------------------------------------
#  Plot 2: Crossover curves by density
# -------------------------------------------------------------------------

cross_data <- df_wide %>%
  group_by(model_label, n_states, density_pct, method, stiff_label, deriv_label) %>%
  summarise(ratio_mean = mean(ratio, na.rm = TRUE),
            ratio_lo   = pmax(mean(ratio) - mean(ratio_se), 0.01),
            ratio_hi   = mean(ratio) + mean(ratio_se),
            .groups = "drop")

p_cross <- ggplot(cross_data,
                  aes(x = n_states, y = ratio_mean,
                      color = factor(density_pct),
                      fill = factor(density_pct))) +
  geom_ribbon(aes(ymin = ratio_lo, ymax = ratio_hi),
              alpha = 0.12, color = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
  scale_x_continuous(trans = pseudo_log_trans,
                     breaks = sort(unique(cross_data$n_states))) +
  scale_y_continuous(trans = pseudo_log_trans) +
  facet_grid(model_label + stiff_label + deriv_label ~ method,
             scales = "free_y") +
  labs(x = "Number of states", y = "Ratio dense/sparse (pseudo-log)",
       color = "Density %", fill = "Density %",
       title = "Dense vs Sparse Crossover by Jacobian Density") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")


# -------------------------------------------------------------------------
#  Plot 3: Absolute solve times
# -------------------------------------------------------------------------

abs_data <- df_ok %>%
  group_by(model_label, n_states, density_pct, method, lu_type,
           stiff_label, deriv_label) %>%
  summarise(t_med = median(t_median), t_lo = median(t_median - t_sd),
            t_hi = median(t_median + t_sd), .groups = "drop") %>%
  mutate(t_lo = pmax(t_lo, 1e-6))

p_abs <- ggplot(abs_data,
                aes(x = n_states, y = t_med,
                    color = lu_type, linetype = method)) +
  geom_ribbon(aes(ymin = t_lo, ymax = t_hi,
                  fill = lu_type,
                  group = interaction(lu_type, method, density_pct)),
              alpha = 0.08, color = NA) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.0) +
  scale_x_continuous(trans = pseudo_log_trans,
                     breaks = sort(unique(abs_data$n_states))) +
  scale_y_continuous(trans = pseudo_log_trans) +
  facet_grid(model_label + stiff_label ~ deriv_label, scales = "free_y") +
  labs(x = "Number of states", y = "Solve time (s, pseudo-log)",
       color = "LU type", linetype = "Method", fill = "LU type",
       title = "Absolute Solve Times") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")


# -------------------------------------------------------------------------
#  Plot 4: Compile times
# -------------------------------------------------------------------------

compile_data <- df_ok %>%
  group_by(model_label, n_states, method, deriv_label) %>%
  summarise(t_compile = mean(t_compile, na.rm = TRUE), .groups = "drop")

p_compile <- ggplot(compile_data,
                    aes(x = factor(n_states), y = t_compile, fill = method)) +
  geom_col(position = "dodge", width = 0.7) +
  facet_grid(model_label ~ deriv_label, scales = "free_y") +
  labs(x = "Number of states", y = "Compile time (s)",
       fill = "Method", title = "Compilation Times") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# -------------------------------------------------------------------------
#  Crossover summary
# -------------------------------------------------------------------------

threshold_data <- df_wide %>%
  group_by(model_label, method, stiff_label, deriv_label, density_pct) %>%
  filter(ratio > 1) %>%
  summarise(
    min_n_sparse_wins = min(n_states),
    .groups = "drop"
  )

cat("\n========= CROSSOVER SUMMARY ============\n")
if (nrow(threshold_data) > 0) {
  for (i in seq_len(nrow(threshold_data))) {
    r <- threshold_data[i, ]
    cat(sprintf("  %s | %s | %s | %s | density=%d%%: sparse wins from n_states >= %d\n",
                r$model_label, r$method, r$stiff_label, r$deriv_label,
                r$density_pct, r$min_n_sparse_wins))
  }
} else {
  cat("  Dense wins everywhere in this grid.\n")
}

# Also show where dense ALWAYS wins
dense_wins <- df_wide %>%
  group_by(model_label, density_pct) %>%
  summarise(sparse_ever_wins = any(ratio > 1), .groups = "drop") %>%
  filter(!sparse_ever_wins)

if (nrow(dense_wins) > 0) {
  cat("\n  Dense wins at ALL sizes for:\n")
  for (i in seq_len(nrow(dense_wins))) {
    cat(sprintf("    %s, density=%d%%\n",
                dense_wins$model_label[i], dense_wins$density_pct[i]))
  }
}


# -------------------------------------------------------------------------
#  Combine and save
# -------------------------------------------------------------------------

p_page1 <- p_heat +
  plot_annotation(
    title = "Dense Systems Benchmark: CT-RNN & Coupled Lorenz",
    subtitle = sprintf("Density = %s%% | %d reps (microbenchmark)",
                       paste(round(density_values * 100), collapse = ", "),
                       N_REPS),
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

p_page2 <- p_cross +
  plot_annotation(
    title = "Dense vs Sparse Crossover by Jacobian Density",
    subtitle = "Ratio > 1 = sparse wins, < 1 = dense wins",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

p_page3 <- p_abs / p_compile +
  plot_layout(heights = c(2, 1)) +
  plot_annotation(
    title = "Solve & Compile Times: CT-RNN & Coupled Lorenz",
    subtitle = sprintf("%d reps per configuration (microbenchmark)", N_REPS),
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

ggsave("benchmark_dense_heatmap.pdf", p_page1,
       width = 14, height = 20, dpi = 150)
ggsave("benchmark_dense_crossover.pdf", p_page2,
       width = 14, height = 20, dpi = 150)
ggsave("benchmark_dense_times.pdf", p_page3,
       width = 14, height = 14, dpi = 150)

cat("\nPlots saved:\n")
cat("  benchmark_dense_heatmap.pdf   (ratio heatmap)\n")
cat("  benchmark_dense_crossover.pdf (crossover curves)\n")
cat("  benchmark_dense_times.pdf     (absolute + compile times)\n")
cat("  benchmark_dense_results.csv   (raw data)\n")
cat("\nDone.\n")
