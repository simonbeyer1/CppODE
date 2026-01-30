rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(dMod)
library(dplyr)


trafo <- c(TCA_buffer = "0",
           TCA_cell = "10^TCA_CELL",
           TCA_cana = "10^TCA_CANA",
           k_import = "10^K_IMPORT",
           k_export_sinus = "10^K_EXPORT_SINUS",
           k_export_cana = "10^K_EXPORT_CANA",
           k_reflux = "10^K_REFLUX",
           s = "10^S")

f <- funCpp(trafo,
            variables  = NULL,
            parameters = getSymbols(trafo),
            fixed = NULL,
            deriv = TRUE,
            deriv2 = TRUE,
            compile = TRUE,
            modelname = "parfn",
            outdir = getwd(),
            convenient = FALSE,
            verbose = FALSE)

pars <- structure(rep(-1, length(getSymbols(trafo))), names = getSymbols(trafo))

f$fun(vars = NULL, params = pars)
f$jac(NULL, pars)[1,,]
f$hess(NULL, pars)[1,"k_import",,]
