test_that("Example for funCpp() works", {
  skip_on_cran()  # sehr empfohlen für C++ Codegen

  oldwd <- getwd()
  setwd(tempdir())
  on.exit(setwd(oldwd), add = TRUE)

  trafo <- c(
    A = "k_p * (k2 + k_d) / (k1 * k_d)",
    B = "k_p / k_d"
  )

  f <- expect_silent(
    funCpp(
      trafo,
      variables  = NULL,
      parameters = c("k_p", "k1", "k2", "k_d"),
      fixed      = NULL,
      deriv      = TRUE,
      deriv2     = TRUE,
      compile    = FALSE,
      modelname  = "obsfn_test",
      convenient = TRUE,
      verbose    = FALSE
    )
  )

  res <- expect_silent(
    f(
      k_p = 0.3,
      k1  = 0.1,
      k2  = 0.2,
      k_d = 0.4,
      deriv2 = TRUE
    )
  )

  ## ---- Output checks ----

  expect_true(is.matrix(res$out))
  expect_equal(colnames(res$out), c("A", "B"))

  ## ---- Jacobian checks ----

  expect_true(!is.null(res$jacobian))
  expect_equal(dim(res$jacobian)[1:2], c(2, 4))  # A,B × parameters

  expect_true(all(is.finite(res$jacobian)))

  jac_symb <- attr(f, "jacobian.symb")
  expect_true(is.list(jac_symb))
  expect_true(all(c("A", "B") %in% names(jac_symb)))

  ## ---- Hessian checks ----

  expect_true(!is.null(res$hessian))
  expect_equal(dim(res$hessian)[1], 2)  # A,B

  expect_true(all(is.finite(res$hessian)))

  hess_symb <- attr(f, "hessian.symb")
  expect_true(is.list(hess_symb))
  expect_true("A" %in% names(hess_symb))
  expect_true("B" %in% names(hess_symb))

})

})
