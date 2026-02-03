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
eqns <- c(x = "-k*x")

# Define an event
events <- data.frame(var = "x", time = "te", value = "1", root = NA, method = "add", stringsAsFactors = FALSE)

# # Generate and compile solver
model <- CppODE(eqns, events = events, deriv = T, deriv2 = T, outdir = getwd(),
                modelname = "xmodel_c", compile = T, useDenseOutput = T, verbose = T)


# Example run
params <- c(x=1, k=1, te = 1)
times  <- seq(0, 10, length.out = 300)
res <- solveODE(model, times, params)
cbind(res$time, res$variable[1:40, , drop = F])
res$sens1[1:40, "x", ]

res$sens2[, "x", "te", "te"]
#
