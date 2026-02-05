rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)


library(CppODE)
library(ggplot2)
library(dplyr)
library(tidyr)
library(data.table)

# Define ODE system
eqns <- c(x = "-k*x")

# Define an event
events <- data.frame(var = "x", time = "te", value = "v", root = NA, method = "add", stringsAsFactors = FALSE)

# # Generate and compile solver
model <- CppODE(eqns, events = events, deriv = T, deriv2 = F, outdir = getwd(),
                modelname = "model_FTEvent", compile = T, useDenseOutput = T, verbose = T)


# Example run
pars <- c(x=1, k=1, v=2,te = 2)
times  <- seq(0, 10, length.out = 300)
res <- solveODE(model, times, pars, abstol = 1e-10, reltol = 1e-10, roottol = 1e-10)
vars <- res$variable
sens <- res$sens1
out.boost <- matrix(aperm(sens, c(1,2,3)), nrow = dim(sens)[1],
            dimnames = list(NULL,
                            paste0("∂", rep(dimnames(sens)[[2]], each = dim(sens)[3]),
                                   "/∂", dimnames(sens)[[3]]))) %>% cbind(time = res$time, vars, .) %>%
  as.data.table() %>%
  melt(id.vars = 1L) %>%
  mutate(method = "boost")

out.analytical <- data.table(
  time = times,
  x = pars[1]*exp(-pars[2]*times) + pars[3]*exp(-pars[2]*(times-pars[4]))*(times>=pars[4]),
  "∂x/∂x"  = exp(-pars[2]*times),
  "∂x/∂k"  = -times*pars[1]*exp(-pars[2]*times) -
    (times-pars[4])*pars[3]*exp(-pars[2]*(times-pars[4]))*(times>=pars[4]),
  "∂x/∂v"  = exp(-pars[2]*(times-pars[4]))*(times>=pars[4]),
  "∂x/∂te" = pars[2]*pars[3]*exp(-pars[2]*(times-pars[4]))*(times>=pars[4])
) %>% melt(id.vars = 1L) %>%
  mutate(method = "analytical")



ggplot(rbind(out.boost, out.analytical), aes(x = time, y = value, color = method, linetype = method)) +
  geom_line() +
  facet_wrap(~ variable, scales = "free_y") +
  dMod::theme_dMod() +
  labs(
    x = "Time",
    y = "Value"
  )

# res$sens2[, "x", "xc", "xc"]
#
