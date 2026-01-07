test_that("Example for funCpp() works", {

  skip_on_cran()

  # Check if Python + packages available
  tryCatch({
    CppODE::ensurePythonEnv("CppODE", verbose = FALSE)
  }, error = function(e) {
    skip(paste("Could not setup Python environment:", e$message))
  })

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
      compile    = TRUE,
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
  # Structure: [n_obs, n_outputs, n_params]
  # Expected: [1, 2, 4] for 1 observation, 2 outputs (A,B), 4 parameters

  expect_true(!is.null(res$jacobian))
  expect_true(is.array(res$jacobian))
  expect_equal(length(dim(res$jacobian)), 3)
  expect_equal(dim(res$jacobian), c(1, 2, 4))  # [obs, outputs, params]
  expect_equal(dimnames(res$jacobian)[[2]], c("A", "B"))
  expect_equal(dimnames(res$jacobian)[[3]], c("k_p", "k1", "k2", "k_d"))
  expect_true(all(is.finite(res$jacobian)))

  jac_symb <- attr(f, "jacobian.symb")
  expect_true(!is.null(jac_symb))
  expect_true(is.matrix(jac_symb))
  expect_equal(dim(jac_symb), c(2, 4))  # outputs × params

  ## ---- Hessian checks ----
  # Structure: [n_obs, n_outputs, n_params, n_params]
  # Expected: [1, 2, 4, 4] for 1 observation, 2 outputs, 4×4 parameter Hessian

  expect_true(!is.null(res$hessian))
  expect_true(is.array(res$hessian))
  expect_equal(length(dim(res$hessian)), 4)
  expect_equal(dim(res$hessian), c(1, 2, 4, 4))  # [obs, outputs, params, params]
  expect_equal(dimnames(res$hessian)[[2]], c("A", "B"))
  expect_equal(dimnames(res$hessian)[[3]], c("k_p", "k1", "k2", "k_d"))
  expect_equal(dimnames(res$hessian)[[4]], c("k_p", "k1", "k2", "k_d"))
  expect_true(all(is.finite(res$hessian)))

  hess_symb <- attr(f, "hessian.symb")
  expect_true(!is.null(hess_symb))
  expect_true(is.list(hess_symb))
  expect_equal(length(hess_symb), 2)  # One Hessian matrix per output (A, B)
  expect_equal(names(hess_symb), c("A", "B"))

  # Each Hessian should be 4×4 (params × params)
  expect_true(all(sapply(hess_symb, is.matrix)))
  expect_true(all(sapply(hess_symb, function(h) all(dim(h) == c(4, 4)))))

})
