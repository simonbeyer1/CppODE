rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(dMod)
library(dplyr)

# kugelkoord <- c(f1 = "r*cos(phi)*sin(theta)",
#                 f2 = "r*sin(phi)*sin(theta)",
#                 f3 = "r*sin(theta)")
#
#
# kugelkoord_derivs <- derivSymb(kugelkoord, deriv2 = T, real = T)
# kugelkoord_derivs$jacobian
# kugelkoord_derivs$hessian[["f1"]]
# kugelkoord_derivs$hessian[["f2"]]
# kugelkoord_derivs$hessian[["f3"]]
#
# eqs <- c(f1 = "a*x^2 + b*y^2",
#          f2 = "x*y + exp(2*c)")
#
# derivs <- derivSymb(eqs, deriv2 = T, real = T, fixed = "c")
# derivs$jacobian
# derivs$hessian[["f1"]]
# derivs$hessian[["f2"]]
#
#
# f <- funCpp(eqs,
#             variables  = NULL,
#             parameters = c("x", "y", "a", "b", "c"),
#             fixed = "c",
#             deriv = TRUE,
#             deriv2 = TRUE,
#             compile = FALSE,
#             modelname = "obsfn",
#             convenient = FALSE,
#             verbose = TRUE)
#
# res <- f(NULL, c(x = 1, y = 1, a = 1, b = 2, c = 0), deriv2 = T)
# res$out
# res$jacobian[ , , 1]
# res$hessian["f1", , , 1]
# attributes(f)$jacobian.symb
# system.time({f(NULL, c(x = 1, y = 1, a = 1, b = 2, c = 0), deriv2 = T)})
#
# # Compile the function
# CppODE:::compile(f)
#
# system.time({res <- f(NULL, c(x = 1, y = 1, a = 1, b = 2, c = 0), deriv2 = T)})
# attributes(res)$jacobian[,,1]
# attributes(f)$jacobian.symb
#
# attributes(f)$hessian.symb
#
# symbolic_jacobian <- attributes(f)$jacobian.symb
# symbolic_jacobian
# symbolic_hessian <- attributes(f)$hessian.symb
# symbolic_hessian$f2
# attributes(res)$hessian["f2", , , 1]

# Check with analytical derivs
trafo <- c(A="k_p * (k2 + k_d) / (k1*k_d)", B = "k_p/k_d")

f <- funCpp(trafo,
            variables  = NULL,
            parameters = c("k_p","k1", "k2", "k_d"),
            fixed = NULL,
            deriv = TRUE,
            deriv2 = TRUE,
            compile = FALSE,
            modelname = "obsfn",
            convenient = FALSE,
            verbose = TRUE)

res <- f(NULL, c(k_p = 0.3, k1 = 0.1, k2 = 0.2, k_d = 0.4), deriv2 = T)
res$out
res$jacobian[,,1]
attributes(f)$jacobian.symb
res$hessian["A",,,1]
attributes(f)$hessian.symb$A

CppODE:::compile(f)
res <- f(NULL, c(k_p = 0.3, k1 = 0.1, k2 = 0.2, k_d = 0.4), deriv2 = T)
res$out
res$jacobian[,,1]
attributes(f)$jacobian.symb
res$hessian["A",,,1]
attributes(f)$hessian.symb$A
