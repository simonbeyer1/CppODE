rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)

eqs <- c(f1 = "a*x + b*y^2",
         f2 = "x*y + exp(2*c)")

f <- funCpp0(eqs,
             variables  = c("x", "y"),
             parameters = c("a", "b", "c"),
             deriv = TRUE,
             deriv2 = TRUE,
             compile = FALSE,
             modelname = "algfun",
             verbose = TRUE)

CppODE:::compile(f)

vars   <- cbind(x = 1:4, y = 2:5)
params <- c(a = 1, b = 2, c = 0)

res <- f(vars, params)
res
attributes(res)$jacobian["f1", , 1]
attributes(res)$hessian["f1", , ,2]
attributes(f)$jacobian.symb

symbolic_hessian <- attributes(f)$hessian.symb
symbolic_hessian["f1", , ]
symbolic_hessian["f2", , ]
