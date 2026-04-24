## =================================================================
## funCpp: compile a multivariate function with analytic Jacobian
## (and optional Hessian) from symbolic expressions.
##
## Shows two common use cases:
##   1. Observables -- map internal states to observed quantities
##   2. Parameter transformations -- re-express parameters on a
##      log/linear scale (derivatives propagate through via chain rule)
## =================================================================
rm(list = ls(all.names = TRUE))
setwd(tempdir())

library(CppODE)

## -----------------------------------------------------------------
## 1. Observables with first- and second-order derivatives
## -----------------------------------------------------------------
observables <- c(
  obs1 = "scale1 * (x1 + x2) + offset1",
  obs2 = "log2(x3) + offset2"
)

f_obs <- funCpp(
  observables,
  variables  = c("x1", "x2", "x3"),
  parameters = c("scale1", "offset1", "offset2"),
  deriv      = TRUE,
  deriv2     = TRUE,
  compile    = TRUE,
  modelname  = "obsfn_demo",
  convenient = TRUE
)

## `convenient = TRUE` gives a list with $func / $jac / $hess / $jac_chain,
## each accepting variables and parameters as named arguments.
args_obs <- list(x1 = 1, x2 = 2, x3 = 4,
                 scale1 = 0.5, offset1 = 0.1, offset2 = 0.0)

cat("Observable values:\n");                   print(do.call(f_obs$func, args_obs))
cat("\nJacobian (observable x input):\n");     print(do.call(f_obs$jac,  args_obs)[1, , ])
cat("\nHessian of obs1:\n");                   print(do.call(f_obs$hess, args_obs)[1, "obs1", , ])
cat("\nSymbolic Jacobian:\n");                 print(attr(f_obs, "jacobian.symb"))

## -----------------------------------------------------------------
## 2. Parameter transformation -- log10 trafo
## -----------------------------------------------------------------
## Express parameters on a log10 scale so optimisers see a free
## unconstrained variable. The Jacobian (attached) chains analytically
## into any downstream sensitivity.
trafo <- c(TCA_cell = "10^TCA_CELL")

f_trafo <- funCpp(
  trafo,
  variables  = NULL,
  parameters = "TCA_CELL",
  deriv      = TRUE,
  deriv2     = TRUE,
  compile    = TRUE,
  modelname  = "parfn_demo",
  convenient = TRUE
)

cat("\nTransformed value (10^-1):\n");         print(f_trafo$func(TCA_CELL = -1))
cat("\nJacobian d(TCA_cell)/d(TCA_CELL):\n");  print(f_trafo$jac(TCA_CELL  = -1))
cat("\nHessian:\n");                            print(f_trafo$hess(TCA_CELL = -1))
