rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)
library(tidyr)

# Equilibrate - stoppt wenn alle Ableitungen (inkl. Sensitivitäten) < roottol
model <- CppODE(
  rhs = c(x = "-k * x + d * y", y = "k * x - d * y"),
  rootfunc = "equilibrate",
  deriv = TRUE,
  deriv2 = TRUE,
  outdir = getwd(),
  modelname = "rootfunc_example"
)

pars <- c(x = 1, y = 0, k = 0.1, d = 0.05)

# Integration - stoppt automatisch bei Steady-State oder Nulldurchgang
res <- solveODE(model, times = seq(0, 1e3, len = 1e3L), parms = pars,
                roottol = 1e-05)


# Access sensitivities res1 + res2 (independent only)
if (!is.null(res$sens1)) {

  n_out    <- length(res$time)
  n_states <- ncol(res$variable)
  n_sens   <- dim(res$sens1)[3]

  dims <- attr(model, "dim_names")

  ## ---------- sens1 ----------
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
yini <- res$variable[lastidx,]
sensini <- res$sens1[lastidx,,]

pars[names(yini)] <- yini

pars
sensini

res2 <- solveODE(model, times = seq(0, 1e3, len = 1e3L), parms = pars,
                 sens1ini = sensini, roottol = 1e-05)


# Access sensitivities res1 + res2 (independent only)
if (!is.null(res2$sens1)) {

  n_out    <- length(res2$time)
  n_states <- ncol(res2$variable)
  n_sens   <- dim(res2$sens1)[3]

  dims <- attr(model, "dim_names")

  ## ---------- sens1 ----------
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
