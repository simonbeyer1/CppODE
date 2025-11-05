rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
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

trafo <- c(k0 = "0",
           k1 = "exp(log(10) * K1)",
           k2 = "exp(log(10) * K2)") %>% dMod::as.eqnvec()

f <- funCpp(trafo,
            variables  = NULL,
            parameters = c("K1", "K2"),
            fixed = NULL,
            deriv = TRUE,
            deriv2 = TRUE,
            compile = FALSE,
            modelname = "obsfn",
            convenient = FALSE,
            verbose = TRUE)

res <- f(NULL, c(K2 = -1, K1 = -2), deriv2 = T)
res$out
res$jacobian[,c("K1", "K2"),1]

attributes(f)$jacobian.symb
