setwd(tempdir())
trafo <- c(A="k_p * (k2 + k_d) / (k1*k_d)", B = "k_p/k_d")

f <- funCpp(trafo,
            variables  = NULL,
            parameters = c("k_p","k1", "k2", "k_d"),
            fixed = NULL,
            deriv = TRUE,
            deriv2 = TRUE,
            compile = FALSE,
            modelname = "obsfn",
            convenient = TRUE,
            verbose = TRUE)

res <- f(k_p = 0.3, k1 = 0.1, k2 = 0.2, k_d = 0.4, deriv2 = T)
res$out
res$jacobian[,,1]
attributes(f)$jacobian.symb
res$hessian["A",,,1]
attributes(f)$hessian.symb$A
