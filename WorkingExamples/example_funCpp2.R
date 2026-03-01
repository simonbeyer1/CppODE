rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(dMod)
library(dplyr)


trafo <- c(TCA_cell = "10^TCA_CELL")

f <- funCpp(trafo,
            variables  = NULL,
            parameters = getSymbols(trafo),
            fixed = NULL,
            deriv = TRUE,
            deriv2 = TRUE,
            compile = TRUE,
            modelname = "parfn2",
            outdir = getwd(),
            convenient = FALSE,
            verbose = FALSE)

pars <- structure(rep(-1, length(getSymbols(trafo))), names = getSymbols(trafo))
jac.symb <- attr(f, "jacobian.symb")

fun <- f$fun
jac <- f$jac

pars
out <- fun(vars = NULL, params = pars, attach.input = T)[,]
out
out.jac <- jac(NULL, pars)
out.jac

out.hess <- f$hess(NULL, pars)
out.hess

