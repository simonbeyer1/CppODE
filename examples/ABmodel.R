rm(list = ls(all.names = TRUE))
# Arbeitsverzeichnis auf den Ordner mit integrate_vdp_call.cpp setzen
.workingDir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(.workingDir)

library(ggplot2)
library(dplyr)
library(tidyverse)

compileAndLoad <- function(filename, verbose = FALSE) {
  filename_cpp <- paste0(filename, ".cpp")

  cxxflags <- "-std=c++17 -O2 -DNDEBUG -fPIC"
  include_flags <- c("-I/home/simon/Dokumente/Projects/CppODE/inst/include")


  Sys.setenv(
    PKG_CPPFLAGS = paste(include_flags, collapse = " "),
    PKG_CXXFLAGS = cxxflags
  )

  shlibOut <- system2(
    file.path(R.home("bin"), "R"),
    args = c("CMD", "SHLIB", "--preclean", shQuote(filename_cpp)),
    stdout = TRUE, stderr = TRUE
  )

  if (verbose) {
    cat(paste(shlibOut, collapse = "\n"), "\n")
  } else if (length(shlibOut)) {
    cat(paste(shlibOut[1], "\n"))
  }

  soFile <- paste0(filename, .Platform$dynlib.ext)
  if (file.exists(soFile)) {
    try(dyn.unload(soFile), silent = !verbose)
    dyn.load(soFile)
    invisible(soFile)
  } else {
    stop("Compiled shared library not found: ", soFile)
  }
}

compileAndLoad("ABmodel", verbose = F)

eqns <- c(A = "-k1*A^2*time", B = "-k1*A^2*time - k2* B")

dyn.load("ABmodel.so")
solve <- function(times, params, abstol = 1e-8, reltol = 1e-6, maxtrysteps = 1e7, maxsteps = 1e7) {
  paramnames <- c("A", "B", "k1", "k2")
  # check for missing parameters
  missing <- setdiff(paramnames, names(params))
  if (length(missing) > 0) stop(sprintf("Missing parameters: %s", paste(missing, collapse = ", ")))
  params <- params[paramnames]
  .Call("solve",
        as.numeric(times),
        as.numeric(params),
        as.numeric(abstol),
        as.numeric(reltol),
        as.integer(maxtrysteps),
        as.integer(maxsteps))
}


params <- c(A=1, B=0, k1=0.1, k2=0.1)
times <- c(seq(0, 30, length.out = 100))

boostCppADtime <- system.time({
  solve(times, params, abstol = 1e-6, reltol = 1e-6)
})

res_CppAD <- solve(times, params, abstol = 1e-6, reltol = 1e-6) %>% as.data.frame() %>%
  pivot_longer(cols = -time, names_to = "name", values_to = "value") %>%
  mutate(solver = "boost::odeint + FAD")

ggplot(res_CppAD, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  dMod::theme_dMod()


library(dMod)
.dmoddir <- file.path(.workingDir, "wd")
if (!dir.exists(.dmoddir)) dir.create(.dmoddir)
eqns <- c(A = "-k1*A^2*time", B = "k1*A^2*time - k2*B")
setwd(.dmoddir)
events <- eventlist() %>%
  addEvent(var = "A", time = "t_e", value=1, method="add")
odemodel <- odemodel(eqns, modelname = "Amodel")
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

calculate_A <- function(time, A0, k1, t_e) {
  # Ergebnisvektoren
  A_values <- numeric(length(time))
  A_te_values <- numeric(length(time))
  A_te_te_values <- numeric(length(time))
  A_k1_k1_values <- numeric(length(time))   # zweite Ableitung nach k1

  # Anfangszeitpunkt
  t_0 <- min(time)

  for (i in seq_along(time)) {
    t <- time[i]

    if (t < t_e) {
      # t < t_e
      denom <- k1 * (t^2 - t_0^2) + 2 / A0
      A_values[i] <- 2 / denom
      A_te_values[i] <- 0
      A_te_te_values[i] <- 0

      B <- (t^2 - t_0^2)
      A_k1_k1_values[i] <- 4 * B^2 / (denom^3)

    } else {
      # Werte bei t_e
      M <- k1 * (t_e^2 - t_0^2) + 2 / A0       # = denom_te_minus
      A_te_minus <- 2 / M
      A_te_plus  <- A_te_minus + 1
      denom_te_plus <- M + 2                   # = 2 + k1*(t_e^2 - t_0^2) + 2/A0

      if (t == t_e) {
        # t = t_e (nach Sprung)
        A_values[i] <- A_te_plus
        A_te_values[i] <- -4 * k1 * t_e / (M^2)
        A_te_te_values[i] <- (-4 * k1 * (-3 * k1 * t_e^2 - k1 * t_0^2 + 2 / A0)) / (M^3)

        C <- (t_e^2 - t_0^2)
        A_k1_k1_values[i] <- 4 * C^2 / (M^3)

      } else {
        # t > t_e
        # g(k1) = 2/A_te_plus = 2*M/(M+2)
        g      <- 2 * M / denom_te_plus
        gprime <- 4 * (t_e^2 - t_0^2) / (denom_te_plus^2)
        g2     <- -8 * (t_e^2 - t_0^2)^2 / (denom_te_plus^3)

        D  <- k1 * (t^2 - t_e^2) + g
        A_values[i] <- 2 / D

        # A.k1.k1 korrekt mit Kettenregel
        Dprime <- (t^2 - t_e^2) + gprime
        D2     <- g2
        A_k1_k1_values[i] <- 4 * (Dprime^2) / (D^3) - 2 * D2 / (D^2)

        # Die t_e-Ableitungen wie gehabt
        d_A_te_plus_inv <- 4 * k1 * t_e / (denom_te_plus^2)
        d_D_te <- -2 * k1 * t_e + 2 * d_A_te_plus_inv
        A_te_values[i] <- -2 / (D^2) * d_D_te

        d2_A_te_plus_inv <- 4 * k1 / (denom_te_plus^2) - 16 * k1^2 * t_e^2 / (denom_te_plus^3)
        d2_D_te <- -2 * k1 + 2 * d2_A_te_plus_inv
        A_te_te_values[i] <- 4 / (D^3) * (d_D_te^2) - 2 / (D^2) * d2_D_te
      }
    }
  }

  # Dataframe
  result <- data.frame(
    time = rep(time, 4),
    name = c(
      rep("A", length(time)),
      rep("A.t_e", length(time)),
      rep("A.t_e.t_e", length(time)),
      rep("A.k1.k1", length(time))
    ),
    value = c(A_values, A_te_values, A_te_te_values, A_k1_k1_values),
    solver = "analytical"
  )

  return(result)
}


df_analytical <- calculate_A(c(times, 3), 1, 0.1, 3)

res <- rbind(res, df_analytical)

ggplot(res, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") +
  ylab("value") +
  theme_dMod()
