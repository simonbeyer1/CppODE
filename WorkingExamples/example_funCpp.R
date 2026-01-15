rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(dMod)
library(dplyr)

eqns <- c(A = "1*(k_p) -1*(k1 * A) +1*(k2 * B)*log(10)",
                B = "1*(k1 * A) -1*(k2 * B) -1*(k_d * B)")
#
#
derivs <- derivSymb(eqns, deriv2 = T, real = T)
derivs$jacobian
derivs$hessian[["A"]]
derivs$hessian[["B"]]
#


trafo <- c(TCA_buffer = "0",
           TCA_cell = "10^TCA_CELL",
           TCA_cana = "10^TCA_CANA",
           k_import = "10^K_IMPORT",
           k_export_sinus = "10^K_EXPORT_SINUS",
           k_export_cana = "10^K_EXPORT_CANA",
           k_reflux = "10^K_REFLUX",
           s = "10^S")



derivs <- derivSymb(unclass(trafo), deriv2 = T, real = T, fixed = NULL, verbose = FALSE)
derivs$jacobian
derivs$hessian[["s"]]
derivs$hessian[["TCA_cell"]]

parnames <- getSymbols(trafo)

f <- funCpp(trafo,
            variables  = NULL,
            parameters = parnames,
            fixed = NULL,
            deriv = TRUE,
            deriv2 = TRUE,
            compile = FALSE,
            modelname = "parfn",
            convenient = FALSE,
            verbose = TRUE)

pars <- structure(rep(-1, length(parnames)), names = parnames)

res <- f(NULL, pars, deriv2 = T)
res$out
# Compile the function
CppODE:::compile(f)
res <- f(NULL, pars, deriv2 = T)
res$out

res$jacobian[ , , 1]
res$hessian["TCA_cell", , , 1]
attributes(f)$jacobian.symb

