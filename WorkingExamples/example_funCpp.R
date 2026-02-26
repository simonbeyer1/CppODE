rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(dMod)
library(dplyr)

observables <- c(obs1 = "scale1*(x1+x2) + offset1",
                 obs2 = "log2(x3) + offset2")

# debugonce(derivSymb)
jactrafo <- derivSymb(observables)


f <- funCpp(observables,
            variables  = c("x1", "x2", "x3"),
            parameters = c("scale1", "offset1", "offset2"),
            fixed = NULL,
            deriv = TRUE,
            deriv2 = TRUE,
            compile = TRUE,
            modelname = "obsfn",
            outdir = getwd(),
            convenient = FALSE,
            verbose = FALSE)

attr(f, "variables")
attr(f, "parameters")



