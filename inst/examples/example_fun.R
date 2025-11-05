setwd(tempdir())
eqs <- c(f1 = "a*x^2 + b*y^2",
         f2 = "x*y + exp(2*c)")

derivs <- derivSymb(eqs, deriv2 = T, real = T)
derivs$jacobian
derivs$hessian[["f1"]]
derivs$hessian[["f2"]]


f <- funCpp(eqs,
            variables  = c("x", "y"),
            parameters = c("a", "b", "c"),
            fixed = "c",
            deriv = TRUE,
            deriv2 = TRUE,
            compile = FALSE,
            modelname = "obsfn",
            verbose = TRUE)

res <- f(x = 1:2, y = 1:2, a = 1, b = 2, c = 0)
head(res)

CppODE:::compile(f)
res <- f(x = 1:2, y = 1:2, a = 1, b = 2, c = 0, deriv2 = T)

res$jacobian[,,1]
res$jacobian[,,2]
res$hessian["f2", , , 1]

attr(f,"jacobian.symb")
attr(f,"hessian.symb")$f1
attr(f,"hessian.symb")$f2
