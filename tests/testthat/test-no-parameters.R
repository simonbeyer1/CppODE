# Test that all three backends accept rhs/eqns with zero parameters.
# Pure-state systems and literal-only expressions must not break codegen,
# AD slab sizing, or sensitivity output shape.

skip_on_cran()
skip_on_ci()

# -- CppODE: pure decay, no parameters ----------------------------------------

test_that("CppODE compiles and solves with zero parameters (deriv = FALSE)", {
  mod <- CppODE(c(x = "-x"), modelname = "noparm_cpp_nd")

  expect_equal(attr(mod, "parameters"), character(0))

  tvec <- seq(0, 2, by = 0.5)
  res  <- solveODE(mod, times = tvec, parms = c(x = 1),
                   abstol = 1e-10, reltol = 1e-10)

  expect_equal(as.numeric(res$variable[, "x"]),
               exp(-tvec), tolerance = 1e-8)
})

test_that("CppODE deriv = TRUE with zero parameters seeds initial-state sens", {
  mod <- CppODE(c(x = "-x"), modelname = "noparm_cpp_d", deriv = TRUE)

  expect_equal(attr(mod, "parameters"), character(0))
  expect_equal(attr(mod, "dimNames")$sens, "x")

  tvec <- seq(0, 2, by = 0.5)
  res  <- solveODE(mod, times = tvec, parms = c(x = 1),
                   abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(res$sens1), c(length(tvec), 1L, 1L))
  # dx(t)/dx0 = exp(-t) for dx/dt = -x
  expect_equal(as.numeric(res$sens1[, 1, 1]),
               exp(-tvec), tolerance = 1e-8)
})

# -- CVODE: same model, same expectations -------------------------------------

test_that("CVODE compiles and solves with zero parameters (deriv = FALSE)", {
  skip_if_not(isTRUE(cvodeConfig$available), "CVODE backend not available")

  mod <- CVODE(c(x = "-x"), modelname = "noparm_cv_nd")

  expect_equal(attr(mod, "parameters"), character(0))

  tvec <- seq(0, 2, by = 0.5)
  res  <- solveODE(mod, times = tvec, parms = c(x = 1),
                   abstol = 1e-10, reltol = 1e-10)

  expect_equal(as.numeric(res$variable[, "x"]),
               exp(-tvec), tolerance = 1e-8)
})

test_that("CVODE deriv = TRUE with zero parameters seeds initial-state sens", {
  skip_if_not(isTRUE(cvodeConfig$available), "CVODE backend not available")

  mod <- CVODE(c(x = "-x"), modelname = "noparm_cv_d", deriv = TRUE)

  expect_equal(attr(mod, "parameters"), character(0))
  expect_equal(attr(mod, "dimNames")$sens, "x")

  tvec <- seq(0, 2, by = 0.5)
  res  <- solveODE(mod, times = tvec, parms = c(x = 1),
                   abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(res$sens1), c(length(tvec), 1L, 1L))
  expect_equal(as.numeric(res$sens1[, 1, 1]),
               exp(-tvec), tolerance = 1e-8)
})

# -- funCpp: literal-only equations (no variables, no parameters) -------------

test_that("funCpp accepts literal-only equations", {
  # convenient = FALSE so we can pass an explicit (n_obs, 0) matrix; the
  # convenient wrapper has no way to express n_obs when there are no vars.
  obj <- funCpp(c(y = "5"), compile = TRUE, modelname = "noparm_fun_lit",
                convenient = FALSE)

  expect_equal(attr(obj, "variables"), character(0))
  expect_null(attr(obj, "parameters"))

  res <- obj$func(matrix(numeric(0), nrow = 3L, ncol = 0L))
  expect_true(is.matrix(res))
  expect_equal(dim(res), c(3L, 1L))
  expect_equal(as.numeric(res[, "y"]), rep(5, 3))
})

# -- funCpp: state variables only, no parameters -------------------------------

test_that("funCpp dual mode evaluates with zero parameters", {
  obj <- funCpp(c(y = "2*x + 3"), compile = TRUE,
                modelname = "noparm_fun_dual", derivMode = "dual")

  expect_equal(attr(obj, "variables"), "x")
  expect_null(attr(obj, "parameters"))

  res <- obj$func(x = c(1, 2, 3))
  expect_equal(as.numeric(res[, "y"]), c(5, 7, 9))

  expect_true(!is.null(obj$jac))
  n_obs <- 3L
  dX <- array(0, c(n_obs, 1L, 1L), list(NULL, "x", "x"))
  dX[, "x", "x"] <- 1
  dP <- matrix(numeric(0), nrow = 0L, ncol = 1L,
               dimnames = list(NULL, "x"))

  jac <- obj$jac(x = c(1, 2, 3), dX = dX, dP = dP)
  # dy/dx = 2 along the lone theta = "x"
  expect_equal(as.numeric(jac[, "y", "x"]), rep(2, n_obs))

  # Combined evaluate() returns y and dy in one nested-dual pass.
  ev <- obj$evaluate(x = c(1, 2, 3), dX = dX, dP = dP)
  expect_equal(as.numeric(ev$y[, "y"]),         c(5, 7, 9))
  expect_equal(as.numeric(ev$dy[, "y", "x"]),   rep(2, n_obs))
})

test_that("funCpp dual mode supports deriv2 with zero parameters", {
  obj <- funCpp(c(y = "x^2 + 3*x"), compile = TRUE,
                modelname = "noparm_fun_dual_d2",
                derivMode = "dual", deriv2 = TRUE)

  # raw hess (identity seed) at x = c(1, 2)
  hess <- obj$hess(x = c(1, 2))
  expect_equal(dim(hess), c(2L, 1L, 1L, 1L))
  # d2y/dx2 = 2
  expect_equal(as.numeric(hess[, "y", "x", "x"]), c(2, 2))

  # explicit identity seed via dX should give the same.
  n_obs <- 2L
  dX <- array(0, c(n_obs, 1L, 1L), list(NULL, "x", "x"))
  dX[, "x", "x"] <- 1
  hess2 <- obj$hess(x = c(1, 2), dX = dX)
  expect_equal(hess, hess2)
})

test_that("funCpp symbolic mode produces correct jac/hess with zero parameters", {
  obj <- funCpp(c(y = "x^2 + 3*x"), compile = TRUE,
                modelname = "noparm_fun_sym",
                derivMode = "symbolic", deriv2 = TRUE)

  expect_equal(attr(obj, "variables"), "x")
  expect_null(attr(obj, "parameters"))

  jac  <- obj$jac(x = c(1, 2))
  hess <- obj$hess(x = c(1, 2))

  expect_equal(dim(jac),  c(2L, 1L, 1L))
  expect_equal(dim(hess), c(2L, 1L, 1L, 1L))
  # dy/dx = 2x + 3
  expect_equal(as.numeric(jac[, "y", "x"]), c(5, 7))
  # d2y/dx2 = 2
  expect_equal(as.numeric(hess[, "y", "x", "x"]), c(2, 2))
})
