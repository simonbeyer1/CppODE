test_that("example ODE model runs", {

  skip_on_cran()
  skip_on_ci()

  eqns <- c(
    A = "-k1 * A",
    B = "k1 * A - k2 * B"
  )

  events <- data.frame(
    var    = "A",
    time   = "t_e",
    value  = "dose",
    method = "add",
    root   = NA,
    stringsAsFactors = FALSE
  )

  f <- CppODE(eqns, events = events, modelname = "CppODE_test", deriv2 = TRUE, compile = TRUE)

  solve <- function(times, params) {
    paramnames <- c(attr(f, "variables"), attr(f, "parameters"))
    params <- params[paramnames]

    out <- .Call(
      paste0("solve_", as.character(f)),
      as.numeric(times),
      as.numeric(params),
      1e-6, 1e-6,
      10L, 7e5L,
      0.0, 1e-8, 1L
    )

    dims <- attr(f, "dim_names")
    colnames(out$variable) <- dims$variable

    if (!is.null(out$sens1)) {
      dimnames(out$sens1) <- list(NULL, dims$variable, dims$sens)
    }
    if (!is.null(out$sens2)) {
      dimnames(out$sens2) <- list(NULL, dims$variable, dims$sens, dims$sens)
    }

    out
  }

  params <- c(A = 1, B = 0, k1 = 0.1, k2 = 0.2, t_e = 25, dose = 1)
  times  <- seq(0, 100, length.out = 100)

  res <- expect_silent(solve(times, params))

  ## ---- Assertions ----

  expect_type(res$time, "double")
  expect_true(is.matrix(res$variable))
  expect_equal(colnames(res$variable), c("A", "B"))

  ## ---- Sensitivity checks ----

  expect_true(!is.null(res$sens1))
  expect_true(!is.null(res$sens2))

  expect_equal(dim(res$sens1)[2], 2)  # A, B
  expect_equal(dim(res$sens2)[2], 2)

  expect_true(all(is.finite(res$sens1)))
  expect_true(all(is.finite(res$sens2)))

})
