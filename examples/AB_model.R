rm(list = ls(all.names = TRUE))
# Arbeitsverzeichnis auf den Ordner mit integrate_vdp_call.cpp setzen
.workingDir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(.workingDir)

library(ggplot2)
library(dplyr)
library(tidyverse)

eqns <- c(A = "-k1*A^2 * time", B = "k1*A^2 * time - k2 * B")


events = data.frame(var = "A", value = 1, method = "add", time = "te", root = NA)

f <- CppODE::CppFun(eqns, events = events, modelname = "AB_model_s", deriv = T, compile=F)

solve <- function(times, params, abstol = 1e-8, reltol = 1e-6, maxattemps = 5000, maxsteps = 1e6, roottol = 1e-8, maxroots = 1) {
  paramnames <- c(attr(f,"variables"), attr(f,"parameters"))
  # check for missing parameters
  missing <- setdiff(paramnames, names(params))
  if (length(missing) > 0) stop(sprintf("Missing parameters: %s", paste(missing, collapse = ", ")))
  params <- params[paramnames]
  out <- .Call(paste0("solve_",as.character(f)),
        as.numeric(times),
        as.numeric(params),
        as.numeric(abstol),
        as.numeric(reltol),
        as.integer(maxattemps),
        as.integer(maxsteps),
        as.numeric(roottol),
        as.integer(maxroots))
  colnames(out) <- c("time", attr(f,"variables"), attr(f,"sensvariables"))
  return(out)
}
CppODE::compileAndLoad(f)
params <- c(A=1, B=0, k1=0.1, k2=0.2, te=5)

times  <- seq(0, 10, length.out = 300)

boosttime <- system.time({
  solve(times, params, abstol = 1e-6, reltol = 1e-6)
})

res_Cpp <- solve(times, params, abstol = 1e-8, reltol = 1e-6) %>% as.data.frame() %>%
  pivot_longer(cols = -time, names_to = "name", values_to = "value") %>%
  mutate(solver = "CppODE(rosenbrock34)")

ggplot(res_Cpp, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  theme_bw()


library(dMod)
.dmoddir <- file.path(.workingDir, "wd")
if (!dir.exists(.dmoddir)) dir.create(.dmoddir)
eqns <- c(A = "-k1*A^2*time", B = "k1*A^2*time - k2*B")
setwd(.dmoddir)
events <- eventlist() %>%
  addEvent("A", value = "1.0", method = "add", time = "te")
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
res <- rbind(res_Cpp, res_dMod)


ggplot(res, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  theme_dMod()

calculate_AB <- function(time, A0, B0, k1, k2, te, t0 = min(time)) {
  # te IMMER enthalten
  if (!(te %in% time)) time <- sort(c(time, te))
  n <- length(time)

  A_values    <- numeric(n)
  B_values    <- numeric(n)
  A_te_values <- numeric(n)
  B_te_values <- numeric(n)

  # --- geschlossene Lösung für A ---
  A_fun <- function(t, A_init, t_init) {
    denom <- k1 * (t^2 - t_init^2) + 2 / A_init
    2 / denom
  }

  # --- halb-analytische Lösung für B ---
  B_fun <- function(times, A_init, B_init, t_init) {
    sapply(times, function(t) {
      if (t == t_init) return(B_init)
      integrand <- function(s) {
        k1 * (A_fun(s, A_init, t_init)^2) * s * exp(k2 * (s - t_init))
      }
      integral <- integrate(integrand, lower = t_init, upper = t,
                            rel.tol = 1e-9, abs.tol = 0)$value
      exp(-k2 * (t - t_init)) * (B_init + integral)
    })
  }

  # --- Phase 1: t < te ---
  idx1 <- which(time < te)
  if (length(idx1) > 0) {
    A_values[idx1]    <- A_fun(time[idx1], A0, t0)
    B_values[idx1]    <- B_fun(time[idx1], A0, B0, t0)
    A_te_values[idx1] <- 0.0
    B_te_values[idx1] <- 0.0
  }

  # Zustände direkt vor dem Event (ohne Sprung)
  A_minus <- A_fun(te, A0, t0)
  B_minus <- B_fun(te, A0, B0, t0)

  # Zustände nach Sprung
  A_plus <- A_minus + 1
  B_plus <- B_minus

  # Sensitivitäten: Wert GENAU am Sprungpunkt (linke Fluss-Ableitung)
  A_te_at_te <- -k1 * A_minus^2 * te                     # f_A^-(te)
  B_te_at_te <-  k1 * A_minus^2 * te - k2 * B_minus      # f_B^-(te)

  # Startwerte für t > te (rechte Seite, für die Sens.-ODEs)
  A_te_plus_ic <-  k1 * te * (2 * A_minus + 1)           # f^- - f^+
  B_te_plus_ic <- -k1 * te * (2 * A_minus + 1)           # f^- - f^+

  # --- Werte GENAU am Event in den Output schreiben ---
  idxE <- which(time == te)
  A_values[idxE]    <- A_plus
  B_values[idxE]    <- B_plus
  A_te_values[idxE] <- A_te_at_te     # NEGATIV
  B_te_values[idxE] <- B_te_at_te     # (typisch) POSITIV

  # --- Phase 2: t > te ---
  idx2 <- which(time > te)
  if (length(idx2) > 0) {
    t2 <- time[idx2]

    # Zustände
    A_values[idx2] <- A_fun(t2, A_plus, te)
    B_values[idx2] <- B_fun(t2, A_plus, B_plus, te)

    # A.te (geschlossene Form) mit IC am rechten Rand
    A_te_values[idx2] <- A_te_plus_ic * (A_values[idx2] / A_plus)^2

    # B.te via Variation der Konstanten mit IC am rechten Rand
    B_te_values[idx2] <- sapply(t2, function(ti) {
      integrand <- function(s) {
        A_s    <- A_fun(s, A_plus, te)
        A_te_s <- A_te_plus_ic * (A_s / A_plus)^2
        exp(-k2 * (ti - s)) * 2 * k1 * A_s * s * A_te_s
      }
      add <- integrate(integrand, lower = te, upper = ti,
                       rel.tol = 1e-9, abs.tol = 0)$value
      exp(-k2 * (ti - te)) * B_te_plus_ic + add
    })
  }

  # --- Dataframe im long-Format ---
  data.frame(
    time  = rep(time, 4),
    name  = c(rep("A", n), rep("B", n), rep("A.te", n), rep("B.te", n)),
    value = c(A_values, B_values, A_te_values, B_te_values),
    solver = "analytical"
  )
}


df_analytical <- calculate_AB(c(times, 5), 1, 0, 0.1, 0.2, 5)

res <- rbind(res_Cpp, df_analytical) %>% filter(name %in% df_analytical$name)

ggplot(res, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  theme_dMod()
