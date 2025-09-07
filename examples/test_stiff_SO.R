rm(list = ls(all.names = TRUE))
# Arbeitsverzeichnis auf den Ordner mit integrate_vdp_call.cpp setzen
.workingDir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(.workingDir)

library(ggplot2)
library(dplyr)
library(tidyverse)
library(CppODE)

eqns <- c(x = "v", v = "mu * (1 - x^2) * v - x")
f <- CppODE::CppFun(eqns, modelname = "vdp", deriv = F)


Sys.setenv(
  PKG_CPPFLAGS = "-I/usr/include -I/usr/local/include",
  PKG_CXXFLAGS = "-std=c++17 -O3 -Ofast -march=native -DNDEBUG -fPIC"
)

src <- "vdp.cpp"   # <— WICHTIG: dieser Dateiname!
system2(file.path(R.home("bin"), "R"),
        args = c("CMD","SHLIB","--preclean", src),
        stdout = TRUE, stderr = TRUE)


# Shared Library laden
dyn.load("integrate_vdp_sens_stiff_events_standalone.so")

solve <- function(times, params, abstol = 1e-8, reltol = 1e-6) {
  .Call("solve",
        as.numeric(times),
        as.numeric(params),
        as.numeric(abstol),
        as.numeric(reltol))
}

params <- c(x=2, v=0, mu=2)
times <- seq(0, 10, length.out = 300)

boostCppADtime <- system.time({
  solve(times, params, abstol = 1e-6, reltol = 1e-6)
})


res_CppAD <- solve(times, params, abstol = 1e-6, reltol = 1e-6) %>% as.data.frame() %>%
  pivot_longer(cols = -time, names_to = "name", values_to = "value") %>%
  mutate(solver = "boost::odeint + CppAD")

ggplot(res_CppAD, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  dMod::theme_dMod()


library(dMod)
.dmoddir <- file.path(.workingDir, "wd")
if (!dir.exists(.dmoddir)) dir.create(.dmoddir)
setwd(.dmoddir)
events <- eventlist() %>%
  addEvent(var = "x", time = "time_root", value=2, root = "x", method="replace")
  # addEvent(var = "x", time = 5, value=2, root = NA, method="replace")
odemodel <- odemodel(eqns, events = events, modelname = "vdp")
x <- Xs(odemodel, condition = "Cond1", optionsSens = list(rtol = 1e-6, atol = 1e-6))
setwd(.workingDir)

dModtime <- system.time({
  x(times, c(params, time_root = 0))
})

prd <- x(times, c(params, time_root = 0)) %>% as.data.frame() %>% dplyr::select(time, name , value)
prd_derivs <- x(times, c(params, time_root = 0)) %>% getDerivs() %>% as.data.frame() %>% dplyr::select(time, name , value)



res_dMod <- rbind(prd, prd_derivs) %>%
  mutate(solver = "dMod")
res <- rbind(res_CppAD, res_dMod)


ggplot(res, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  theme_dMod()


