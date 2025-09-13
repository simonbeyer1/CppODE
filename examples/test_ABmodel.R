rm(list = ls(all.names = TRUE))
# Arbeitsverzeichnis auf den Ordner mit integrate_vdp_call.cpp setzen
.workingDir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(.workingDir)

library(ggplot2)
library(dplyr)
library(tidyverse)

eqns <- c(A = "-k1*A^2 *time")
events = data.frame(var = c("A","A"), time = c("t_e", "t_e2"), value=c(1,1), method=c("add", "replace"))

f <- CppODE::CppFun(eqns, events = events, modelname = "Amodel_s", secderiv = T)

Sys.setenv(
  PKG_CPPFLAGS = "-I/usr/include -I/usr/local/include",
  PKG_CXXFLAGS = "-std=c++17 -O3 -Ofast -march=native -DNDEBUG -fPIC"
)

src <- "Amodel_s.cpp"   # <— WICHTIG: dieser Dateiname!
system2(file.path(R.home("bin"), "R"),
        args = c("CMD","SHLIB","--preclean", src),
        stdout = TRUE, stderr = TRUE)


# Shared Library laden
dyn.load("Amodel_s.so")

solve <- function(times, params, abstol = 1e-8, reltol = 1e-6) {
  params <- params[c("A", "k1", "t_e")]
  .Call("solve_Amodel_s",
        as.numeric(times),
        as.numeric(params),
        as.numeric(abstol),
        as.numeric(reltol))
}

params <- c(A=1, k1=0.1, t_e=3)
times <- c(seq(0, 10, length.out = 300))

boostCppADtime <- system.time({
  solve(times, params, abstol = 1e-8, reltol = 1e-6)
})

res_CppAD <- solve(times, params, abstol = 1e-8, reltol = 1e-6) %>% as.data.frame() %>%
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
  addEvent(var = "A", time = "t_e", value=1, method="add")
odemodel <- odemodel(eqns, events = events, modelname = "Amodel")
x <- Xs(odemodel, condition = "Cond1", optionsSens = list(rtol = 1e-8, atol = 1e-6))
setwd(.workingDir)

dModtime <- system.time({
  x(times, params)
})

prd <- x(times, params) %>% as.data.frame() %>% dplyr::select(time, name , value)
prd_derivs <- x(times, params) %>% getDerivs() %>% as.data.frame() %>% dplyr::select(time, name , value)



res_dMod <- rbind(prd, prd_derivs) %>%
  mutate(solver = "dMod")
res <- rbind(res_CppAD, res_dMod)


ggplot(res, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  theme_dMod()


# Funktion zur Berechnung von A(t), \partial A / \partial t_e und \partial^2 A / \partial t_e^2
calculate_A <- function(time, A0, k1, t_e) {
  # Initialisieren der Ergebnisvektoren
  A_values <- numeric(length(time))
  A_te_values <- numeric(length(time))
  A_te_te_values <- numeric(length(time))

  # Anfangszeitpunkt t_0 als ersten Zeitpunkt im Vektor
  t_0 <- min(time)

  # Berechnung von A(t), \partial A / \partial t_e und \partial^2 A / \partial t_e^2
  for (i in 1:length(time)) {
    t <- time[i]

    if (t < t_e) {
      # Lösung für t < t_e
      denom <- k1 * (t^2 - t_0^2) + 2 / A0
      A_values[i] <- 2 / denom
      A_te_values[i] <- 0  # Partielle Ableitung ist 0 für t < t_e
      A_te_te_values[i] <- 0  # Zweite partielle Ableitung ist 0 für t < t_e
    } else {
      # Wert bei t = t_e^- und Sprung zu t_e^+
      denom_te_minus <- k1 * (t_e^2 - t_0^2) + 2 / A0
      A_te_minus <- 2 / denom_te_minus
      A_te_plus <- A_te_minus + 1

      # Denominator für t >= t_e
      denom_te_plus <- 2 + k1 * (t_e^2 - t_0^2) + 2 / A0

      if (t == t_e) {
        # Wert bei t = t_e (nach dem Sprung)
        A_values[i] <- A_te_plus
        # Erste partielle Ableitung bei t = t_e
        A_te_values[i] <- -4 * k1 * t_e / (denom_te_minus^2)
        # Zweite partielle Ableitung bei t = t_e
        A_te_te_values[i] <- (-4 * k1 * (-3 * k1 * t_e^2 - k1 * t_0^2 + 2 / A0)) / (denom_te_minus^3)
      } else {
        # Lösung für t > t_e
        denom <- k1 * (t^2 - t_e^2) + 2 / A_te_plus
        A_values[i] <- 2 / denom

        # Erste partielle Ableitung für t > t_e
        d_A_te_plus_inv <- 4 * k1 * t_e / (denom_te_plus^2)
        d_D_te <- -2 * k1 * t_e + 2 * d_A_te_plus_inv
        A_te_values[i] <- -2 / (denom^2) * d_D_te

        # Zweite partielle Ableitung für t > t_e
        d2_A_te_plus_inv <- 4 * k1 / (denom_te_plus^2) - 16 * k1^2 * t_e^2 / (denom_te_plus^3)
        d2_D_te <- -2 * k1 + 2 * d2_A_te_plus_inv
        A_te_te_values[i] <- 4 / (denom^3) * (d_D_te^2) - 2 / (denom^2) * d2_D_te
      }
    }
  }

  # Erstellen des Dataframes
  result <- data.frame(
    time = rep(time, 3),
    name = c(rep("A", length(time)), rep("A.t_e", length(time)), rep("A.t_e.t_e", length(time))),
    value = c(A_values, A_te_values, A_te_te_values),
    solver = "analytical"
  )

  return(result)
}


df_analytical <- calculate_A(c(times, 3), 1, 0.1, 3)

res <- rbind(res_CppAD, df_analytical) %>% filter(name %in% df_analytical$name)

ggplot(res, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  theme_dMod()

