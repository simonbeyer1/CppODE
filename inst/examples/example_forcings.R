rm(list = ls(all.names = TRUE))
.workingDir <- file.path(tempdir(), "CppODE_example_forcings")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)
library(tidyr)
library(data.table)

# ── ODE systems ──────────────────────────────────────────────
eqns <- c(
  x = "v",
  v = "-omega0*x - 2*gamma * v"
)

eqns_forc <- c(
  x = "v",
  v = "-omega0*x - 2*gamma * v + F1"
)

homodel <- CppODE(eqns, outdir = getwd(), modelname = "homodel", compile = TRUE, deriv2 = TRUE)
homodel_forc <- CppODE(eqns_forc, forcings = c("F1"), outdir = getwd(), modelname = "homodel_forc", compile = TRUE, deriv2 = TRUE)

times <- seq(0, 100, length.out = 600)
times_forc <- seq(0, 50, length.out = 300)

forcs <- list(
  F1 = data.frame(time = times_forc, value = sin(times_forc))
)

params <- c(x = 1, v = 0, omega0 = 1, gamma = 0.1)

res <- solveODE(homodel, times, params)
res_forc <- solveODE(homodel_forc, times, params, forcings = forcs)

# ── Helper: extract states, sens1, sens2 into long data.table ──
extract_long <- function(res, system_label) {

  # Layouts: variable [n_out, n_vars]; sens1 [n_out, n_vars, n_sens];
  # sens2 [n_out, n_vars, n_sens, n_sens]
  states <- dimnames(res$sens1)[[2]]  # e.g. c("x", "v")
  pars   <- dimnames(res$sens1)[[3]]  # e.g. c("x", "v", "omega0", "gamma")
  nt     <- length(res$time)

  # — States —
  vars_wide <- as.data.table(cbind(time = res$time, res$variable))
  dt_states <- melt(vars_wide, id.vars = "time", variable.name = "name", value.name = "value")
  dt_states[, type := "state"]

  # — First-order sensitivities: sens1[time, state, par] —
  sens <- res$sens1
  dt_sens1 <- rbindlist(lapply(seq_len(nt), function(i) {
    mat <- sens[i, , ]  # [state x par]
    data.table(
      time = res$time[i],
      name = as.vector(outer(states, pars, function(s, p) paste0("\u2202", s, "/\u2202", p))),
      value = as.vector(mat)
    )
  }))
  dt_sens1[, type := "sens1"]

  # — Second-order sensitivities: sens2[time, state, par, par] —
  sens2 <- res$sens2
  np <- length(pars)
  # Lower-triangular indices (including diagonal)
  idx <- which(lower.tri(matrix(1, np, np), diag = TRUE), arr.ind = TRUE)

  dt_sens2 <- rbindlist(lapply(seq_len(nt), function(i) {
    rbindlist(lapply(states, function(s) {
      mat_i <- sens2[i, s, , ]  # [par x par]
      data.table(
        time  = res$time[i],
        name  = paste0("\u2202\u00B2", s, "/\u2202", pars[idx[, 1]], "\u2202", pars[idx[, 2]]),
        value = mat_i[idx]
      )
    }))
  }))
  dt_sens2[, type := "sens2"]

  # — Combine —
  dt <- rbindlist(list(dt_states, dt_sens1, dt_sens2))
  dt[, system := system_label]
  dt
}

# ── Build combined long table ────────────────────────────────
out_hom  <- extract_long(res,      "free")
out_forc <- extract_long(res_forc, "forced")
out <- rbind(out_hom, out_forc)

# ── Plot ─────────────────────────────────────────────────────

# Separate plots per type for readability
for (tp in c("state", "sens1", "sens2")) {
  p <- ggplot(out[type == tp], aes(x = time, y = value, color = system)) +
    geom_line() +
    facet_wrap(~ name, scales = "free_y") +
    dMod::theme_dMod() +
    dMod::scale_color_dMod() +
    labs(x = "Time", y = "Value", title = tp)
  print(p)
}

# ── Timing ───────────────────────────────────────────────────
system.time({solveODE(homodel, times, params)})
system.time({solveODE(homodel_forc, times, params, forcings = forcs)})
