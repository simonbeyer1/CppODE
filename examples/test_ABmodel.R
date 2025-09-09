rm(list = ls(all.names = TRUE))
# Arbeitsverzeichnis auf den Ordner mit integrate_vdp_call.cpp setzen
.workingDir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(.workingDir)

library(ggplot2)
library(dplyr)
library(tidyverse)
library(CppODE)

eqns <- c(A = "-k1*A^2*time", B = "k1*A - k2*B")
events = data.frame(var = "A", time = 5, value=1, method="add")

f <- CppODE::CppFun(eqns, events = events, modelname = "ABmodel_s", secderiv = F)

Sys.setenv(
  PKG_CPPFLAGS = "-I/usr/include -I/usr/local/include",
  PKG_CXXFLAGS = "-std=c++17 -O3 -Ofast -march=native -DNDEBUG -fPIC"
)

src <- "ABmodel_s.cpp"   # <— WICHTIG: dieser Dateiname!
system2(file.path(R.home("bin"), "R"),
        args = c("CMD","SHLIB","--preclean", src),
        stdout = TRUE, stderr = TRUE)


# Shared Library laden
dyn.load("ABmodel_s.so")

solve <- function(times, params, abstol = 1e-8, reltol = 1e-6) {
  params <- params[c("A", "B", "k1", "k2")]
  .Call("solve",
        as.numeric(times),
        as.numeric(params),
        as.numeric(abstol),
        as.numeric(reltol))
}

params <- c(A=1, B=0, k1=0.1, k2=0.1)
times <- seq(0, 10, length.out = 1000)

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
  addEvent(var = "A", time = 5, value=1, method="add")
odemodel <- odemodel(eqns, events = events, modelname = "vdp")
x <- Xs(odemodel, condition = "Cond1", optionsSens = list(rtol = 1e-6, atol = 1e-6))
setwd(.workingDir)

dModtime <- system.time({
  x(times, c(params))
})

prd <- x(times, c(params)) %>% as.data.frame() %>% dplyr::select(time, name , value)
prd_derivs <- x(times, c(params)) %>% getDerivs() %>% as.data.frame() %>% dplyr::select(time, name , value)



res_dMod <- rbind(prd, prd_derivs) %>%
  mutate(solver = "dMod")
res <- rbind(res_CppAD, res_dMod)


ggplot(res, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  theme_dMod()


dA_dt_e <- function(t, t_e, A0, k1) {
  # Vorbereitungen
  denom_e <- 0.5 * k1 * t_e^2 + 1 / A0
  A_e <- 1 / denom_e
  A_e_plus <- A_e + 1

  # Ableitung von A_e nach t_e
  dA_e_dt_e <- -k1 * t_e / denom_e^2

  # Nenner der A(t)
  D <- 0.5 * k1 * (t^2 - t_e^2) + 1 / A_e_plus

  # Ableitung
  dA <- (1 / D^2) * (k1 * t_e - (1 / A_e_plus^2) * dA_e_dt_e)

  # Setze Sensitivität auf 0 für t < t_e
  dA[t < t_e] <- 0

  return(dA)
}

# Parameter setzen
t_e <- 5
A0 <- 1
k1 <- 0.1

# Sensitivitäten berechnen
sensi_vals <- dA_dt_e(t = times, t_e = t_e, A0 = A0, k1 = k1)

# Datenframe zum Plotten
df_analytisch <- data.frame(
  time = times,
  value = sensi_vals,
  name = "A.time_event",
  solver = "analytical"
)

res <- rbind(res, df_analytisch)

ggplot(res, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  theme_dMod()
