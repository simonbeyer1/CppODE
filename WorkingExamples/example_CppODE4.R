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
eqns <- c(x = "-k*x^2*time")

# # Generate and compile solver
model <- CppODE(eqns, events = events, deriv = T, deriv2 = T, outdir = getwd(),
                modelname = "model_FTEvent4", compile = T, useDenseOutput = T)

pars <- c(x=1, k=1, v = 2, te = 4)
times  <- c(4-1e-12,4,4+1e-12, seq(0, 10, length.out = 400)) %>% sort()
out.analytical <- solveOdeAnalytic(c(x = "-k*x^2*t"),times,pars, events = events) %>%
  melt(id.vars = 1L) %>%
  mutate(method = "analytical")


# Example run
res <- solveODE(model, times, pars)
vars <- res$variable %>% t()
sens <- res$sens1
sens2 <- res$sens2
dim(sens2)
out.boost <- matrix(aperm(sens, c(3, 1, 2)), nrow = dim(sens)[3],
                    dimnames = list(NULL,
                                    paste0("∂", rep(dimnames(sens)[[1]], each = dim(sens)[2]),
                                           "/∂", dimnames(sens)[[2]]))) %>%
  cbind(time = res$time, vars, .) %>%
  as.data.table() %>%
  melt(id.vars = 1L) %>%
  (\(dt) rbindlist(list(
    dt,
    {
      p <- dimnames(sens2)[[2]]
      idx <- which(lower.tri(matrix(1, length(p), length(p)), TRUE), arr.ind = TRUE)
      rbindlist(lapply(seq_along(res$time), \(i)
                       data.table(time = res$time[i],
                                  variable = paste0("∂²x/∂", p[idx[,1]], "∂", p[idx[,2]]),
                                  value = sens2[1,, ,i][idx]))
      )
    }
  )))() %>%
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

