## =================================================================
## Events: time-triggered and root-triggered
##
## Shows both event styles on scalar ODEs, both with first- and
## second-order parameter sensitivities enabled.
## =================================================================
rm(list = ls(all.names = TRUE))
setwd(tempdir())

library(CppODE)

## -----------------------------------------------------------------
## 1. Time-triggered event: replace x with v at t = te
## -----------------------------------------------------------------
## dx/dt = -k * x
## at t = te, x is replaced by the parameter value v.
eqns_time <- c(x = "-k*x")
evt_time  <- data.frame(
  var    = "x",
  time   = "te",
  value  = "v",
  method = "replace",
  root   = NA,
  stringsAsFactors = FALSE
)

model_time <- CppODE(
  eqns_time,
  events    = evt_time,
  deriv     = TRUE,
  deriv2    = TRUE,
  modelname = "event_time_demo"
)

pars_time <- c(x = 1, k = 1, v = 0.75, te = 3)
res_time  <- solveODE(model_time, seq(0, 10, length.out = 300), pars_time)

cat("Time event — state at t just before/after te = 3:\n")
i_before <- max(which(res_time$time <  3))
i_after  <- min(which(res_time$time >= 3))
print(data.frame(
  time = res_time$time[c(i_before, i_after)],
  x    = res_time$variable[c(i_before, i_after), "x"]
))

## -----------------------------------------------------------------
## 2. Root-triggered event: add v each time x drops to xc
## -----------------------------------------------------------------
## dx/dt = -k * x^2 * time
## when xc - x crosses zero, v is added to x. Up to maxroot firings.
eqns_root <- c(x = "-k*x^2*time")
evt_root  <- data.frame(
  var    = "x",
  time   = NA,
  value  = "v",
  method = "add",
  root   = "xc - x",
  stringsAsFactors = FALSE
)

model_root <- CppODE(
  eqns_root,
  events    = evt_root,
  deriv     = TRUE,
  deriv2    = TRUE,
  modelname = "event_root_demo"
)

pars_root <- c(x = 1, k = 1, v = 1, xc = 0.25)
res_root  <- solveODE(model_root, seq(0, 10, length.out = 300), pars_root, maxroot = 3)

cat("\nRoot event — sensitivity of x to v is piecewise but continuous\n")
cat("through each event (saltation-matrix correction applied):\n")
print(summary(res_root$sens1[, "x", "v"]))

## -----------------------------------------------------------------
## 3. Layout of the sensitivity arrays
## -----------------------------------------------------------------
##   res$variable : [n_times, n_states]
##   res$sens1    : [n_times, n_states, n_sens]    (d y / d p)
##   res$sens2    : [n_times, n_states, n_sens, n_sens]  (d^2 y / d p d q)
cat("\nsens1 dims:", paste(dim(res_time$sens1), collapse = " x "), "\n")
cat("sens2 dims:", paste(dim(res_time$sens2), collapse = " x "), "\n")
