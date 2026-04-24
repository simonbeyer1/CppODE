rm(list = ls(all.names = TRUE))
.workingDir <- file.path(tempdir(), "CppODE_example_equilibrate")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)
library(tidyr)

rhs <- c(
  R = "k_act_R_bas-k_deact_R*R",
  A = "-k1*A*R+1*k2*pA",
  pA = "k1*A*R-k2*pA"
)

# Equilibrate - stoppt wenn alle Ableitungen (inkl. Sensitivitäten) < roottol
model <- CppODE(
  rhs = rhs,
  rootfunc = "equilibrate",
  deriv = TRUE,
  deriv2 = FALSE,
  outdir = getwd(),
  modelname = "rootfunc_example"
)

pars <- c(R = 1, A = 1, pA = 1, k_act_R_bas = 0.1, k_deact_R = 0.7, k1 = 0.1, k2 = 0.05)

# Integration - stoppt automatisch bei Steady-State oder Nulldurchgang
res <- solveODE(model, times = seq(0, 1e3, len = 1e3L), parms = pars,
                roottol = 1e-06)


# Access sensitivities res1 + res2 (independent only)
if (!is.null(res$sens1)) {

  n_out    <- length(res$time)
  n_states <- ncol(res$variable)
  n_sens   <- dim(res$sens1)[3]

  dims <- attr(model, "dim_names")

  ## ---------- sens1 ----------
  # sens1 is [n_out, n_states, n_sens] → flatten to [n_out, n_states*n_sens]
  sens1_matrix <- matrix(res$sens1,
                         nrow = n_out,
                         ncol = n_states * n_sens)

  sens1_colnames <-
    as.vector(outer(paste0("∂", dims$variable),
                    paste0("∂", dims$sens),
                    paste, sep = "/"))
  colnames(sens1_matrix) <- sens1_colnames


  ## ---------- sens2 (independent only) ----------
  sens2_matrix <- NULL

  if (!is.null(res$sens2)) {

    # independent (i <= j)
    ind_ij <- which(
      upper.tri(matrix(1, n_sens, n_sens), diag = TRUE),
      arr.ind = TRUE
    )
    n_ind <- nrow(ind_ij)

    sens2_matrix <- matrix(NA_real_,
                           nrow = n_out,
                           ncol = n_states * n_ind)

    sens2_colnames <- character(n_states * n_ind)
    col_idx <- 1

    for (s in seq_len(n_states)) {
      for (k in seq_len(n_ind)) {

        i <- ind_ij[k, 1]
        j <- ind_ij[k, 2]

        # sens2 is [n_out, n_states, n_sens, n_sens] → take time-series for (s, i, j)
        sens2_matrix[, col_idx] <- res$sens2[, s, i, j]

        sens2_colnames[col_idx] <-
          paste0("∂²", dims$variable[s],
                 "/∂", dims$sens[i],
                 "∂", dims$sens[j])

        col_idx <- col_idx + 1
      }
    }

    colnames(sens2_matrix) <- sens2_colnames
  }


  ## ---------- combine everything ----------
  out_full <- cbind(
    time = res$time,
    res$variable,
    sens1_matrix,
    sens2_matrix
  )

  head(out_full)
}


lastidx <- length(res$time)
yini <- res$variable[nrow(res$variable), ]
sensini <- res$sens1[length(res$time), , ]

pars[names(yini)] <- yini

pars
sensini

pars["k1"] = 0.11
pars["k2"] = 0.55

res2 <- solveODE(model, times = seq(0, 1e3, len = 1e3L), parms = pars,
                 sens1ini = sensini, roottol = 1e-06)


# Access sensitivities res1 + res2 (independent only)
if (!is.null(res2$sens1)) {

  n_out    <- length(res2$time)
  n_states <- ncol(res2$variable)
  n_sens   <- dim(res2$sens1)[3]

  dims <- attr(model, "dim_names")

  ## ---------- sens1 ----------
  # sens1 is [n_out, n_states, n_sens] → flatten to [n_out, n_states*n_sens]
  sens1_matrix <- matrix(res2$sens1,
                         nrow = n_out,
                         ncol = n_states * n_sens)

  sens1_colnames <-
    as.vector(outer(paste0("∂", dims$variable),
                    paste0("∂", dims$sens),
                    paste, sep = "/"))
  colnames(sens1_matrix) <- sens1_colnames


  ## ---------- sens2 (independent only) ----------
  sens2_matrix <- NULL

  if (!is.null(res2$sens2)) {

    # independent (i <= j)
    ind_ij <- which(
      upper.tri(matrix(1, n_sens, n_sens), diag = TRUE),
      arr.ind = TRUE
    )
    n_ind <- nrow(ind_ij)

    sens2_matrix <- matrix(NA_real_,
                           nrow = n_out,
                           ncol = n_states * n_ind)

    sens2_colnames <- character(n_states * n_ind)
    col_idx <- 1

    for (s in seq_len(n_states)) {
      for (k in seq_len(n_ind)) {

        i <- ind_ij[k, 1]
        j <- ind_ij[k, 2]

        sens2_matrix[, col_idx] <- res2$sens2[, s, i, j]

        sens2_colnames[col_idx] <-
          paste0("∂²", dims$variable[s],
                 "/∂", dims$sens[i],
                 "∂", dims$sens[j])

        col_idx <- col_idx + 1
      }
    }

    colnames(sens2_matrix) <- sens2_colnames
  }


  ## ---------- combine everything ----------
  out_full2 <- cbind(
    time = res2$time,
    res2$variable,
    sens1_matrix,
    sens2_matrix
  )

  head(out_full2)
}

out_full[nrow(out_full),]


out <- as.data.frame(out_full) %>%
  mutate(time = out_full[,1]) %>%
  pivot_longer(-time, names_to = "name", values_to = "value")


ggplot(out, aes(x = time, y = value)) +
  geom_line() +
  facet_wrap(~ name, scales = "free_y") +
  dMod::theme_dMod() +
  labs(
    title = "res1",
    x = "Time",
    y = "Value"
  )


out2 <- as.data.frame(out_full2) %>%
  mutate(time = out_full2[,1]) %>%
  pivot_longer(-time, names_to = "name", values_to = "value")


ggplot(out2, aes(x = time, y = value)) +
  geom_line() +
  facet_wrap(~ name, scales = "free_y") +
  dMod::theme_dMod() +
  labs(
    title = "res2",
    x = "Time",
    y = "Value"
  )

