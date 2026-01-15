rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)
library(tidyr)

# Define ODE system
eqns <- c(
  x = "v",
  v = "-k1*x^2 + F1 + F2"
)

homodel <- CppODE(eqns, forcings = c("F1", "F2"), outdir = getwd(), modelname = "homodel")
