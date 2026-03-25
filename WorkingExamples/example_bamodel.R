rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)
library(data.table)
library(tidyr)

f <- c(TCA_buffer = "-k_import * TCA_buffer + k_export_sinus * TCA_cell + k_reflux * TCA_cana",
       TCA_cana = "k_export_cana * TCA_cell - k_reflux * TCA_cana",
       TCA_cell = "k_import * TCA_buffer - k_export_sinus * TCA_cell - k_export_cana * TCA_cell")


model <- CppODE(f, outdir = getwd(), modelname = "bamodel", compile = T, fixed = c("TCA_buffer"), deriv2 = T)

times <- seq(0,45, len = 300)
params <- c(TCA_buffer = 0, TCA_cell = 0.1, TCA_cana = 0.1, k_import = 0.1, k_export_sinus = 0.1, k_reflux = 0.1, k_export_cana = 0.1)

res <- solveODE(model, times, params)


vars <- res$variable %>% t()
sens <- res$sens1
out <- matrix(aperm(sens, c(3, 1, 2)), nrow = dim(sens)[3],
                    dimnames = list(NULL,
                                    paste0("∂", rep(dimnames(sens)[[1]], each = dim(sens)[2]),
                                           "/∂", dimnames(sens)[[2]]))) %>%
  cbind(time = res$time, vars, .) %>%
  as.data.table() %>%
  melt(id.vars = 1L)

ggplot(out, aes(x = time, y = value)) +
  geom_line() +
  facet_wrap(~variable, scales = "free_y") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  labs(
    x = "Time",
    y = "value"
  )

is.null(res$sens2)
#
# M <- matrix(res$sens1, nrow = 300*3, ncol = 7)
