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

source("../AnalyticSolver.R")
events <- data.frame(var = "x", time = "te", value = "v", root = NA, method = "add", stringsAsFactors = FALSE)

# Define ODE system
eqns <- c(x = "-k*x^2 * time")

# # Generate and compile solver
model <- CppODE(eqns, events = events, deriv = T, deriv2 = F, outdir = getwd(),
                modelname = "model_FTEvent2", compile = T, useDenseOutput = T, verbose = T)

pars <- c(x=1, k=1, v = 2, te = 2)
times  <- seq(0, 10, length.out = 1000)
out.analytical <- solveOdeAnalytic(c(x = "-k*x^2*t"),times,pars, events = events) %>%
  melt(id.vars = 1L) %>%
  mutate(method = "analytical")


# Example run
res <- solveODE(model, times, pars, abstol = 1e-10, reltol = 1e-10, roottol = 1e-10)
vars <- res$variable %>%
sens <- res$sens1
out.boost <- matrix(aperm(sens, c(3, 1, 2)), nrow = dim(sens)[3],
                    dimnames = list(NULL,
                                    paste0("∂", rep(dimnames(sens)[[1]], each = dim(sens)[2]),
                                           "/∂", dimnames(sens)[[2]]))) %>%
  cbind(time = res$time, vars, .) %>%
  as.data.table() %>%
  melt(id.vars = 1L) %>%
  mutate(method = "boost")


out <- rbind(out.boost, out.analytical)
out$variable <- factor(out$variable, levels = unique(as.character(out$variable)))

ggplot(out, aes(x = time, y = value, color = method, linetype = method)) +
  geom_line() +
  facet_wrap(~variable, scales = "free_y") +
  dMod::theme_dMod() +
  dMod::scale_color_dMod() +
  labs(
    x = "Time",
    y = "value"
  )

res$sens2[, "x", "xc", "xc"]



# solveA <- function(times, pars) {
#   x0 <- pars[1]; k <- pars[2]; v <- pars[3]; te <- pars[4]
#   A <- function(t) 0.5*k*t^2
#   xte <- 1/(A(te)+1/x0)+v
#   x <- ifelse(times<te, 1/(A(times)+1/x0), 1/(A(times)+1/xte-A(te)))
#   dx_dx0 <- ifelse(times<te, 1/(x0^2*(A(times)+1/x0)^2), 1/(xte^2*(A(te)+1/x0)^2)/(A(times)+1/xte-A(te))^2)
#   dx_dk <- ifelse(times<te, -0.5*times^2/(A(times)+1/x0)^2, -(0.5*(times^2-te^2)+0.5*te^2/((A(te)+1/x0)^2*xte^2))/(A(times)+1/xte-A(te))^2)
#   dx_dv <- ifelse(times<te, 0, 1/(xte^2*(A(times)+1/xte-A(te))^2))
#   dx_dte <- ifelse(times<te, 0, k*te/(A(times)+1/xte-A(te))^2)
#   data.table(time=times, x=x, "∂x/∂x"=dx_dx0, "∂x/∂k"=dx_dk, "∂x/∂v"=dx_dv, "∂x/∂te"=dx_dte) %>%
#     melt(id.vars=1L) %>% mutate(method="analytical")
# }
#
