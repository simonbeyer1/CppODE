rm(list = ls(all.names = TRUE))
.workingDir <- file.path(tempdir(), "CppODE_example_bamodel")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(cOde)
library(ggplot2)
library(dplyr)
library(data.table)
library(tidyr)

f <- c(TCA_buffer = "-k_import * TCA_buffer + k_export_sinus * TCA_cell + k_reflux * TCA_cana",
       TCA_cana = "k_export_cana * TCA_cell - k_reflux * TCA_cana",
       TCA_cell = "k_import * TCA_buffer - k_export_sinus * TCA_cell - k_export_cana * TCA_cell")

f.sens <-sensitivitiesSymb(f)

model_ndf <- CppODE(f, outdir = getwd(), modelname = "bamodel_ndf", compile = T, method = "bdf")
model_bdf <- CppODE(f, outdir = getwd(), modelname = "bamodel_bdf", compile = T, method = "bdf", useNDF = F)
model_rb <- CppODE(f, outdir = getwd(), modelname = "bamodel_rb", compile = T, method = "rb4")
model_cOde_inz <- funC(c(f, f.sens), jacobian = "inz.lsodes", modelname = "bamodel_inzlsodes", compile = T)
model_cOde <- funC(c(f, f.sens), modelname = "bamodel", compile = T)

times <- seq(0,45, len = 300)
params <- c(TCA_buffer = 0.1, TCA_cell = 0.1, TCA_cana = 0.1, k_import = 0.1, k_export_sinus = 0.1, k_reflux = 0.1, k_export_cana = 0.1)
parsC <- params[setdiff(names(params), names(f))]
yini <- c(params[names(f)], attr(f.sens, "yini"))

res_ndf <- solveODE(model_ndf, times, params, abstol = 1e-20, reltol = 1e-20)
res_bdf <- solveODE(model_bdf, times, params, abstol = 1e-20, reltol = 1e-20)
res_rb <- solveODE(model_rb, times, params, abstol = 1e-15, reltol = 1e-15)
res_code_inz <- odeC(yini, times, model_cOde_inz, parsC, atol = 1e-15, rtol = 1e-15)
res_code <- odeC(yini, times, model_cOde, parsC, atol = 1e-15, rtol = 1e-15)

system.time({solveODE(model_ndf, times, params)})
system.time({solveODE(model_bdf, times, params)})
system.time({solveODE(model_rb, times, params)})
system.time({odeC(yini, times, model_cOde_inz, parsC)})
system.time({odeC(yini, times, model_cOde, parsC)})

# --- Helper to reshape results into long data.table ---
# Layouts: variable [n_out, n_vars]; sens1 [n_out, n_vars, n_sens]
reshape_results <- function(res, method_name) {
  vars  <- res$variable
  sens  <- res$sens1
  n_v <- dim(sens)[2]; n_s <- dim(sens)[3]

  # States + first-order sensitivities
  dt <- matrix(sens, nrow = dim(sens)[1],
               dimnames = list(NULL,
                               paste0("∂", rep(dimnames(sens)[[2]], times = n_s),
                                      "/∂", rep(dimnames(sens)[[3]], each = n_v)))) %>%
    cbind(time = res$time, vars, .) %>%
    as.data.table() %>%
    melt(id.vars = 1L)

  dt[, method := method_name]
}

out_rb <- reshape_results(res_rb, "rb4")
out_bdf <- reshape_results(res_bdf, "bdf")
out_ndf <- reshape_results(res_bdf, "ndf")
out   <- rbindlist(list(out_rb, out_bdf, out_ndf))

ggplot(out, aes(x = time, y = value, color = method, linetype = method)) +
  geom_line() +
  facet_wrap(~variable, scales = "free_y") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  labs(
    x = "Time",
    y = "value"
  )

diagnostics(res_ndf)
diagnostics(res_bdf)
#
# M <- matrix(res$sens1, nrow = 300*3, ncol = 7)
