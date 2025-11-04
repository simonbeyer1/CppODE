rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)

kugelkoord <- c(f1 = "r*cos(phi)*sin(theta)",
                f2 = "r*sin(phi)*sin(theta)",
                f3 = "r*sin(theta)")


kugelkoord_derivs <- derivSymb(kugelkoord, deriv2 = T, real = T)
kugelkoord_derivs$jacobian
kugelkoord_derivs$hessian[["f1"]]
kugelkoord_derivs$hessian[["f2"]]
kugelkoord_derivs$hessian[["f3"]]

eqs <- c(f1 = "a*x^2 + b*y^2",
         f2 = "x*y + exp(2*c)")

# derivs <- derivSymb(eqs, deriv2 = T, real = T)
# derivs$jacobian
# derivs$hessian[["f1"]]
# derivs$hessian[["f2"]]


f <- funCpp(eqs,
             variables  = c("x", "y"),
             parameters = c("a", "b", "c"),
             deriv = TRUE,
             deriv2 = TRUE,
             compile = FALSE,
             modelname = "obsfn",
             verbose = FALSE)

# res <- f(x = 1:2, y = 1:2, a = 1, b = 2, c = 0)
# res
# attributes(res)$jacobian
# attributes(f)$jacobian.symb

# CppODE:::compile(f)

res <- f(x = 1:2, y = 1:2, a = 1, b = 2, c = 0)
attributes(res)$jacobian
attributes(f)$jacobian.symb

attributes(f)$hessian.symb

symbolic_jacobian <- attributes(f)$jacobian.symb
symbolic_jacobian
symbolic_hessian <- attributes(f)$hessian.symb
symbolic_hessian
symbolic_hessian$f2
attributes(res)$hessian["f2", , , 1]

