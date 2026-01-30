rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)
library(tidyr)

f <- c(TCA_buffer = "-k_import * TCA_buffer + k_export_sinus * TCA_cell + k_reflux * TCA_cana",
       TCA_cana = "k_export_cana * TCA_cell - k_reflux * TCA_cana",
       TCA_cell = "k_import * TCA_buffer - k_export_sinus * TCA_cell - k_export_cana * TCA_cell")


# Equilibrate - stoppt wenn alle Ableitungen (inkl. Sensitivitäten) < roottol
model <- CppODE(f, outdir = getwd(), modelname = "bamodel_s", compile = T)

times <- seq(0,45, len = 300)
params <- c(TCA_buffer = 0, TCA_cell = 0.1, TCA_cana = 0.1, k_import = 0.1, k_export_sinus = 0.1, k_reflux = 0.1, k_export_cana = 0.1)


res <- solveODE(model, times, params)

M <- matrix(res$sens1, nrow = 300*3, ncol = 7)
