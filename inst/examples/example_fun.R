\dontrun{
  oldwd <- getwd()
  on.exit(setwd(oldwd), add = TRUE)
  setwd(tempdir())

  trafo <- c(A = "k_p * (k2 + k_d) / (k1*k_d)", B = "k_p/k_d")

  f <- funCpp(trafo,
              variables  = NULL,
              parameters = c("k_p", "k1", "k2", "k_d"),
              fixed      = NULL,
              deriv      = TRUE,
              deriv2     = TRUE,
              compile    = TRUE,
              modelname  = "obsfn",
              convenient = TRUE,
              verbose    = FALSE)

  res <- f(k_p = 0.3, k1 = 0.1, k2 = 0.2, k_d = 0.4, deriv2 = TRUE)

  # Output values
  res$out

  # Jacobian for first (and only) observation
  res$jacobian[1, , ]

  # Symbolic Jacobian (unchanged)
  attributes(f)$jacobian.symb

  # Hessian for output "A" at first observation
  res$hessian[1, "A", , ]

  # Symbolic Hessian (unchanged)
  attributes(f)$hessian.symb$A
}
