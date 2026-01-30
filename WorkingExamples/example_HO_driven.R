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
  v = "-omega0*x - 2*gamma * v"
)

eqns_forc <- c(
  x = "v",
  v = "-omega0*x - 2*gamma * v + F1"
)

homodel <- CppODE(eqns, outdir = getwd(), modelname = "homodel", compile = T, deriv2 = F)
homodel_forc <- CppODE(eqns_forc, forcings = c("F1"), outdir = getwd(), modelname = "homodel_forc", compile = T, deriv2 = F)


times <- seq(0, 100, length.out = 300)

times_forc <- seq(0, 50, length.out = 100)

# Liste mit Namen
forcs <- list(
  F1 = data.frame(time = times_forc, value = sin(times_forc))
)

params <- c(x = 1, v = 0, omega0 = 1, gamma = 0.1)

res <- solveODE(homodel, times, params)
res_forc <- solveODE(homodel_forc, times, params, forcings = forcs)


out_hom <- as.data.frame(res$variable) %>%
  mutate(time = res$time) %>%
  pivot_longer(-time, names_to = "name", values_to = "value") %>%
  mutate(system = "homogen")

out_forc <- as.data.frame(res_forc$variable) %>%
  mutate(time = res_forc$time) %>%
  pivot_longer(-time, names_to = "name", values_to = "value") %>%
  mutate(system = "getrieben")

out <- rbind(out_hom, out_forc)

ggplot(out, aes(x = time, y = value)) +
  geom_line() +
  facet_wrap(~ name + system, scales = "free_y") +
  dMod::theme_dMod() +
  labs(
    x = "Time",
    y = "Value"
  )

system.time({solveODE(homodel, times, params)})
system.time({solveODE(homodel_forc, times, params, forcings = forcs)})
