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
  A = "-k1*A^2*time",
  B = "k1*A^2*time - k2*B",
  switch = 0
)

# Define an event
events <- data.frame(
  var   = c("A", "A", "switch"),
  time  = c(NA, 0, NA),
  value = c("B","dose", "B"),
  method= c("replace","add", "replace"),
  root  = c("A-Acrit", NA, "A-Acrit"),
  stringsAsFactors = FALSE
)

# # Generate and compile solver
model_c <- CppODE(eqns, events = events, deriv = T, deriv2 = F, outdir = getwd(), modelname = "Amodel_c", compile = T, useDenseOutput = F, verbose = T)
model_d <- CppODE(eqns, events = events, deriv = T, deriv2 = F, outdir = getwd(), modelname = "Amodel_d", compile = T, useDenseOutput = T)


# Example run
params <- c(A = 0, B = 0, switch = 0, k1 = 0.1, k2 = 0.1, k3 = 0.1, dose = 1, Acrit = 0.25)
times  <- seq(-10, 50, length.out = 200)
res_c <- solveODE(model_c, times, params, maxroot = 4)
res_d <- solveODE(model_d, times, params, maxroot = 4)
head(res_c$variable)
res_c$sens1[, "A", "Acrit"]
res_c$sens2[10, "A", ,]
res_d$sens2[10, "A", ,]

out_c <- as.data.frame(res_c$variable) %>%
  mutate(time = res_c$time) %>%
  pivot_longer(-time, names_to = "name", values_to = "value") %>%
  mutate(method = "controlled")

out_d <- as.data.frame(res_d$variable) %>%
  mutate(time = res_d$time) %>%
  pivot_longer(-time, names_to = "name", values_to = "value") %>%
  mutate(method = "dense")

out <- rbind(out_c, out_d)

ggplot(out, aes(x = time, y = value)) +
  geom_line() +
  facet_wrap(~ name + method, scales = "free_y") +
  dMod::theme_dMod() +
  labs(
    x = "Time",
    y = "Value"
  )

system.time({solveODE(model_c, times, params, maxroot = 4, roottol = 1e-9)})
system.time({solveODE(model_c, times, params, fixed = c("k1", "k2", "Acrit"), maxroot = 4, roottol = 1e-9)})
system.time({solveODE(model_d, times, params, maxroot = 4, roottol = 1e-9)})
system.time({solveODE(model_d, times, params, fixed = c("k1", "k2", "Acrit"), maxroot = 4, roottol = 1e-9)})



