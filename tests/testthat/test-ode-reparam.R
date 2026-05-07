# Test parameter reparametrization p = Phi(theta) for sensitivities.

skip_on_cran()
skip_on_ci()

# -- Simple scalar log-transform -----------------------------------------------

test_that("log-transform reparam matches analytical dx/dtheta", {
  # Model: dx/dt = -k*x, x(0) = x0. p = (x0, k).
  # Reparametrize: theta = (x0, log(k)) -> Phi(theta) = (theta_x0, exp(theta_lk))
  # Phi_prime = [[1, 0], [0, k]]
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_log", deriv = TRUE, nStack = 2L)

  pars <- c(x = 1.0, k = 0.5)
  Phi_prime <- matrix(c(1, 0, 0, 0.5), nrow = 2, ncol = 2,
                      dimnames = list(c("x", "k"), c("theta_x0", "theta_lk")))

  tvec <- seq(0, 2, by = 0.5)
  res  <- solveODE(mod, times = tvec, parms = pars, sens1ini = Phi_prime,
                   abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(res$sens1), c(length(tvec), 1L, 2L))
  expect_equal(dimnames(res$sens1)$sens, c("theta_x0", "theta_lk"))

  k  <- 0.5; x0 <- 1.0
  expected_x0 <- exp(-k * tvec)
  expected_lk <- -k * tvec * x0 * exp(-k * tvec)
  expect_equal(as.numeric(res$sens1[, 1, 1]), expected_x0, tolerance = 1e-8)
  expect_equal(as.numeric(res$sens1[, 1, 2]), expected_lk, tolerance = 1e-8)
})

# -- Parity: direct integration vs post-hoc S * Phi' --------------------------

test_that("reparam sens equals post-hoc S * Phi' (two-state model)", {
  rhs <- c(A = "-k1*A + k2*B",
           B =  "k1*A - k2*B")

  pars <- c(A = 1.0, B = 0.0, k1 = 0.3, k2 = 0.1)
  tvec <- seq(0, 5, length.out = 21)

  tight <- list(abstol = 1e-10, reltol = 1e-10)

  # Identity model
  mod_id <- CppODE(rhs, modelname = "rep_id", deriv = TRUE)
  res_id <- solveODE(mod_id, tvec, pars,
                     abstol = tight$abstol, reltol = tight$reltol)

  # Reparametrized model: theta = (A0, B0, log(k1), log(k2))
  mod_th <- CppODE(rhs, modelname = "rep_th", deriv = TRUE, nStack = 4L)
  k1 <- pars["k1"]; k2 <- pars["k2"]
  Phi_prime <- matrix(
    c(1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, k1, 0,
      0, 0, 0, k2),
    nrow = 4, ncol = 4, byrow = TRUE,
    dimnames = list(c("A", "B", "k1", "k2"),
                    c("A0", "B0", "log_k1", "log_k2"))
  )
  res_th <- solveODE(mod_th, tvec, pars, sens1ini = Phi_prime,
                     abstol = tight$abstol, reltol = tight$reltol)

  # Post-hoc: S_theta[t, i, j] = sum_p S_id[t, i, p] * Phi_prime[p, j]
  # where rows of Phi_prime are (A, B, k1, k2) and columns are theta names.
  S_id <- res_id$sens1
  S_expected <- array(0, dim = dim(res_th$sens1), dimnames = dimnames(res_th$sens1))
  for (ti in seq_len(dim(S_id)[1])) {
    S_expected[ti, , ] <- S_id[ti, , ] %*% Phi_prime
  }
  expect_equal(as.numeric(res_th$sens1), as.numeric(S_expected),
               tolerance = 1e-8)
})

# -- Rank-reduced reparam (n_theta < n_p) ------------------------------------

test_that("rank-reduced reparam integrates over smaller theta space", {
  # Model: dx/dt = -k*x
  # Reparametrize to theta = (log(k)), with x0 = exp(theta) (ties IC to rate).
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_rank1", deriv = TRUE, nStack = 1L)

  k <- 0.4; theta <- log(k); x0 <- exp(theta)   # x0 = k by this parametrization
  pars <- c(x = x0, k = k)

  # Phi(theta) = (exp(theta), exp(theta))
  # Phi_prime = [[exp(theta)], [exp(theta)]] = [[k], [k]]
  Phi_prime <- matrix(c(k, k), nrow = 2, ncol = 1,
                      dimnames = list(c("x", "k"), "log_k"))

  tvec <- seq(0, 3, by = 0.5)
  res  <- solveODE(mod, tvec, pars, sens1ini = Phi_prime,
                   abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(res$sens1), c(length(tvec), 1L, 1L))

  # Analytic: x(t) = x0 * exp(-k*t) = k * exp(-k*t)
  # dx/dtheta = dx/dx0 * exp(theta) + dx/dk * exp(theta)
  #           = exp(-kt) * k + (-t*x0*exp(-kt)) * k
  #           = k * exp(-kt) * (1 - k*t)
  expected <- k * exp(-k * tvec) * (1 - k * tvec)
  expect_equal(as.numeric(res$sens1[, 1, 1]), expected, tolerance = 1e-8)
})

# -- Guard rails --------------------------------------------------------------

test_that("nStack model works with sens1ini = NULL (identity seeding)", {
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_missing_sens", deriv = TRUE, nStack = 2L)
  res <- expect_silent(solveODE(mod, c(0, 1), c(x = 1, k = 0.5)))
  expect_equal(dim(res$sens1), c(2L, 1L, 2L))
})

test_that("reparam rejects 'fixed' argument", {
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_fixed_reject", deriv = TRUE, nStack = 2L)
  Phi_prime <- matrix(c(1, 0, 0, 0.5), 2, 2)
  expect_error(solveODE(mod, c(0, 1), c(x = 1, k = 0.5),
                        sens1ini = Phi_prime, fixed = "k"),
               "not supported")
})

test_that("nStack with deriv=FALSE is rejected at compile time", {
  expect_error(CppODE(c(x = "-k*x"), modelname = "rep_no_deriv",
                      deriv = FALSE, nStack = 2L),
               "deriv = TRUE")
})

# -- Phase 2: Second-order sensitivities under reparam -----------------------

test_that("deriv2 + log-reparam matches analytical d^2x/dtheta^2", {
  # p = Phi(theta) = (theta_x0, exp(theta_lk))
  # Phi'  = [[1, 0], [0, k]]
  # Phi'' = all zero except Phi''[k_row, theta_lk, theta_lk] = k
  mod <- CppODE(c(x = "-k*x"), modelname = "d2_rep_log",
                deriv = TRUE, deriv2 = TRUE, nStack = 2L)

  pars <- c(x = 1.0, k = 0.5); k <- 0.5; x0 <- 1.0
  Phi_prime <- matrix(c(1, 0, 0, k), 2, 2)
  Phi_pp <- array(0, dim = c(2, 2, 2)); Phi_pp[2, 2, 2] <- k

  tvec <- c(0, 0.5, 1, 1.5, 2)
  res <- solveODE(mod, tvec, pars,
                  sens1ini = Phi_prime, sens2ini = Phi_pp,
                  abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(res$sens2), c(length(tvec), 1L, 2L, 2L))

  # Analytic:
  # d^2x/dtheta_x0^2 = 0
  # d^2x/dtheta_x0 dtheta_lk = -k*t*exp(-kt)
  # d^2x/dtheta_lk^2 = x0*exp(-kt)*k*t*(kt - 1)
  expect_equal(as.numeric(res$sens2[, 1, 1, 1]),
               rep(0, length(tvec)), tolerance = 1e-8)
  expect_equal(as.numeric(res$sens2[, 1, 1, 2]),
               -k * tvec * exp(-k * tvec), tolerance = 1e-8)
  expect_equal(as.numeric(res$sens2[, 1, 2, 1]),   # symmetry
               -k * tvec * exp(-k * tvec), tolerance = 1e-8)
  expect_equal(as.numeric(res$sens2[, 1, 2, 2]),
               x0 * exp(-k * tvec) * k * tvec * (k * tvec - 1),
               tolerance = 1e-8)
})

# -- CVODE backend parity ----------------------------------------------------

test_that("CVODE reparam matches Native reparam (no events)", {
  skip_if_not(isTRUE(cvodeConfig$available), "CVODE backend not available")

  rhs <- c(A = "-k1*A + k2*B",
           B =  "k1*A - k2*B")
  pars <- c(A = 1.0, B = 0.0, k1 = 0.3, k2 = 0.1)
  tvec <- seq(0, 5, length.out = 11)
  tight <- list(abstol = 1e-10, reltol = 1e-10)

  k1 <- pars["k1"]; k2 <- pars["k2"]
  Phi_prime <- matrix(
    c(1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, k1, 0,
      0, 0, 0, k2),
    nrow = 4, ncol = 4, byrow = TRUE)

  mod_native <- CppODE(rhs, modelname = "rep_par_nat",
                       deriv = TRUE, nStack = 4L)
  mod_cvode  <- CVODE(rhs,  modelname = "rep_par_cv",
                      deriv = TRUE)

  res_n <- solveODE(mod_native, tvec, pars, sens1ini = Phi_prime,
                    abstol = tight$abstol, reltol = tight$reltol)
  res_c <- solveODE(mod_cvode,  tvec, pars, sens1ini = Phi_prime,
                    abstol = tight$abstol, reltol = tight$reltol)

  expect_equal(dim(res_n$sens1), dim(res_c$sens1))
  expect_equal(as.numeric(res_n$sens1), as.numeric(res_c$sens1),
               tolerance = 1e-6)
})

test_that("CVODE reparam with time event: chain-rule saltation (post-hoc parity)", {
  skip_if_not(isTRUE(cvodeConfig$available), "CVODE backend not available")

  # dx/dt = -k*x with a parameterised-time event: at t = t_e, x += dose.
  # Reparametrize theta = (x0, log(k), t_e, dose); Phi'(theta) has
  # a non-identity on the k row (dp_k/dtheta_lk = k).
  eqns <- c(x = "-k*x")
  evt  <- data.frame(var = "x", time = "t_e", value = "dose",
                     method = "add", root = NA, stringsAsFactors = FALSE)
  pars <- c(x = 1.0, k = 0.3, t_e = 2.0, dose = 0.5)

  k <- pars["k"]
  # Phi rows = (x, k, t_e, dose), theta = (theta_x0, theta_lk, theta_te, theta_dose)
  Phi_prime <- matrix(c(
    1, 0,  0, 0,
    0, k,  0, 0,
    0, 0,  1, 0,
    0, 0,  0, 1),
    nrow = 4, ncol = 4, byrow = TRUE)

  tvec <- seq(0, 5, length.out = 21)
  tight <- list(abstol = 1e-10, reltol = 1e-10)

  # CVODE non-reparam (reference in p-coordinates, with event saltation)
  mod_id <- CVODE(eqns, events = evt, modelname = "rep_ev_cv_id",
                  deriv = TRUE)
  res_id <- solveODE(mod_id, tvec, pars,
                     abstol = tight$abstol, reltol = tight$reltol)

  # CVODE reparam (uses chain-rule saltation internally)
  mod_cv <- CVODE(eqns, events = evt, modelname = "rep_ev_cv_th",
                  deriv = TRUE)
  res_cv <- solveODE(mod_cv, tvec, pars, sens1ini = Phi_prime,
                     abstol = tight$abstol, reltol = tight$reltol)

  # Post-hoc composition: S_theta[t, i, j] = sum_p S_id[t, i, p] * Phi_prime[p, j]
  S_id <- res_id$sens1  # [t, x, (x, k, t_e, dose)]
  S_expected <- array(0, dim = dim(res_cv$sens1))
  for (ti in seq_len(dim(S_id)[1])) {
    S_expected[ti, , ] <- S_id[ti, , ] %*% Phi_prime
  }
  expect_equal(as.numeric(res_cv$sens1), as.numeric(S_expected),
               tolerance = 1e-5)
})

test_that("sens2 chain-rule parity: direct vs post-hoc composition", {
  # Nonlinear reparametrization over a 2-state model: theta -> p
  rhs <- c(A = "-k1*A + k2*B",
           B =  "k1*A - k2*B")

  pars <- c(A = 1.0, B = 0.2, k1 = 0.3, k2 = 0.1)
  tvec <- seq(0, 3, length.out = 7)
  tight <- list(abstol = 1e-10, reltol = 1e-10)

  # Identity model with deriv2 for post-hoc composition
  mod_id <- CppODE(rhs, modelname = "d2_par_id", deriv = TRUE, deriv2 = TRUE)
  res_id <- solveODE(mod_id, tvec, pars,
                     abstol = tight$abstol, reltol = tight$reltol)

  # Reparametrized model: theta = (A0, B0, log(k1), log(k2))
  mod_th <- CppODE(rhs, modelname = "d2_par_th",
                   deriv = TRUE, deriv2 = TRUE, nStack = 4L)
  k1 <- pars["k1"]; k2 <- pars["k2"]
  Phi_prime <- matrix(
    c(1, 0, 0,  0,
      0, 1, 0,  0,
      0, 0, k1, 0,
      0, 0, 0,  k2),
    nrow = 4, ncol = 4, byrow = TRUE
  )
  Phi_pp <- array(0, dim = c(4, 4, 4))
  Phi_pp[3, 3, 3] <- k1  # d^2 k1 / d(log_k1)^2 = k1
  Phi_pp[4, 4, 4] <- k2  # d^2 k2 / d(log_k2)^2 = k2

  res_th <- solveODE(mod_th, tvec, pars,
                     sens1ini = Phi_prime, sens2ini = Phi_pp,
                     abstol = tight$abstol, reltol = tight$reltol)

  # Post-hoc chain rule:
  #   H^theta[t, k, a, b] = sum_{i,j} H_id[t, k, i, j] * Phi'[i, a] * Phi'[j, b]
  #                      + sum_i      S_id[t, k, i]   * Phi''[i, a, b]
  # where i, j range over the full n_phi_rows = n_states + n_params = 4 slots.
  S_id <- res_id$sens1   # [t, k, i] but here i only has n_active = 4 slots (all)
  H_id <- res_id$sens2   # [t, k, i, j] with same i, j basis
  H_expected <- array(0, dim = dim(res_th$sens2))

  # Note: S_id / H_id are indexed by ACTIVE slots, which under identity mode
  # correspond to the full (variables, parameters) vector since nothing is fixed.
  for (ti in seq_along(tvec)) {
    for (kk in seq_len(dim(H_id)[2])) {
      # Hessian term
      H_expected[ti, kk, , ] <- t(Phi_prime) %*% H_id[ti, kk, , ] %*% Phi_prime
      # Gradient-times-Phi'' term
      for (a in seq_len(4)) {
        for (b in seq_len(4)) {
          H_expected[ti, kk, a, b] <- H_expected[ti, kk, a, b] +
            sum(S_id[ti, kk, ] * Phi_pp[, a, b])
        }
      }
    }
  }

  expect_equal(as.numeric(res_th$sens2), as.numeric(H_expected),
               tolerance = 1e-7)
})

# -- nStack as compile-time upper bound: M <= nStack per call ----------------

test_that("native: M < nStack integrates the supplied theta subset", {
  # Compile with nStack = 3 but call with only M = 2 active thetas.
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_M_lt_ntheta",
                deriv = TRUE, nStack = 3L)
  pars <- c(x = 1.0, k = 0.5); k <- 0.5; x0 <- 1.0
  tvec <- seq(0, 2, by = 0.5)

  # M = 2: theta = (theta_x0, theta_lk).
  Phi_M2 <- matrix(c(1, 0, 0, k), nrow = 2, ncol = 2, byrow = TRUE,
                   dimnames = list(c("x", "k"), c("theta_x0", "theta_lk")))
  res_M2 <- solveODE(mod, tvec, pars, sens1ini = Phi_M2,
                     abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(res_M2$sens1), c(length(tvec), 1L, 2L))
  expect_equal(dimnames(res_M2$sens1)$sens, c("theta_x0", "theta_lk"))

  # Analytical dx/dtheta for the log-reparam (same as existing test 1).
  expect_equal(as.numeric(res_M2$sens1[, 1, 1]), exp(-k * tvec), tolerance = 1e-8)
  expect_equal(as.numeric(res_M2$sens1[, 1, 2]),
               -k * tvec * x0 * exp(-k * tvec), tolerance = 1e-8)

  # Cross-check: calling the same model with a zero-padded M=3 sens1ini
  # yields identical first 2 columns and zero third column.
  Phi_M3 <- cbind(Phi_M2, unused = c(0, 0))
  res_M3 <- solveODE(mod, tvec, pars, sens1ini = Phi_M3,
                     abstol = 1e-10, reltol = 1e-10)
  expect_equal(dim(res_M3$sens1), c(length(tvec), 1L, 3L))
  expect_equal(as.numeric(res_M3$sens1[, , 1:2, drop = FALSE]),
               as.numeric(res_M2$sens1), tolerance = 1e-8)
  expect_equal(as.numeric(res_M3$sens1[, , 3]),
               rep(0, length(tvec)), tolerance = 1e-12)
})

test_that("native deriv2: M < nStack integrates partial Phi''", {
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_M_lt_ntheta_d2",
                deriv = TRUE, deriv2 = TRUE, nStack = 3L)
  pars <- c(x = 1.0, k = 0.5); k <- 0.5; x0 <- 1.0
  tvec <- c(0, 0.5, 1, 1.5, 2)

  # M = 2, full [n_phi_rows, M, M] cube.
  Phi_M2 <- matrix(c(1, 0, 0, k), nrow = 2, ncol = 2, byrow = TRUE,
                   dimnames = list(c("x", "k"), c("theta_x0", "theta_lk")))
  Phi_pp_M2 <- array(0, dim = c(2, 2, 2))
  Phi_pp_M2[2, 2, 2] <- k  # d^2 k / d theta_lk^2 = k

  res <- solveODE(mod, tvec, pars, sens1ini = Phi_M2, sens2ini = Phi_pp_M2,
                  abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(res$sens1), c(length(tvec), 1L, 2L))
  expect_equal(dim(res$sens2), c(length(tvec), 1L, 2L, 2L))

  # Same analytic Hessian as the existing nStack=2 test.
  expect_equal(as.numeric(res$sens2[, 1, 1, 1]),
               rep(0, length(tvec)), tolerance = 1e-8)
  expect_equal(as.numeric(res$sens2[, 1, 1, 2]),
               -k * tvec * exp(-k * tvec), tolerance = 1e-8)
  expect_equal(as.numeric(res$sens2[, 1, 2, 2]),
               x0 * exp(-k * tvec) * k * tvec * (k * tvec - 1),
               tolerance = 1e-8)
})

test_that("CVODE: M < nStack matches native reparam", {
  skip_if_not(isTRUE(cvodeConfig$available), "CVODE backend not available")

  rhs <- c(A = "-k1*A + k2*B",
           B =  "k1*A - k2*B")
  pars <- c(A = 1.0, B = 0.0, k1 = 0.3, k2 = 0.1)
  tvec <- seq(0, 5, length.out = 11)
  tight <- list(abstol = 1e-10, reltol = 1e-10)

  k1 <- pars["k1"]; k2 <- pars["k2"]
  # Rows: (A, B, k1, k2). Columns (M=2): (log_k1, log_k2).
  Phi_M2 <- matrix(c(0, 0,
                     0, 0,
                     k1, 0,
                     0, k2),
                   nrow = 4, ncol = 2, byrow = TRUE,
                   dimnames = list(c("A", "B", "k1", "k2"),
                                   c("log_k1", "log_k2")))

  mod_nat <- CppODE(rhs, modelname = "rep_M_lt_nat", deriv = TRUE, nStack = 4L)
  mod_cv  <- CVODE(rhs,  modelname = "rep_M_lt_cv",  deriv = TRUE)

  res_n <- solveODE(mod_nat, tvec, pars, sens1ini = Phi_M2,
                    abstol = tight$abstol, reltol = tight$reltol)
  res_c <- solveODE(mod_cv,  tvec, pars, sens1ini = Phi_M2,
                    abstol = tight$abstol, reltol = tight$reltol)

  expect_equal(dim(res_n$sens1), c(length(tvec), 2L, 2L))
  expect_equal(dim(res_c$sens1), c(length(tvec), 2L, 2L))
  expect_equal(as.numeric(res_n$sens1), as.numeric(res_c$sens1),
               tolerance = 1e-6)
})

test_that("same model supports per-call varying M (condition heterogeneity)", {
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_varying_M",
                deriv = TRUE, nStack = 3L)
  pars <- c(x = 1.0, k = 0.5); k <- 0.5; x0 <- 1.0
  tvec <- seq(0, 2, by = 0.5)

  # Call 1: M=1, just theta_lk.
  Phi1 <- matrix(c(0, k), nrow = 2, ncol = 1,
                 dimnames = list(c("x", "k"), "theta_lk"))
  r1 <- solveODE(mod, tvec, pars, sens1ini = Phi1,
                 abstol = 1e-10, reltol = 1e-10)

  # Call 2: M=2, theta_x0 + theta_lk.
  Phi2 <- matrix(c(1, 0, 0, k), nrow = 2, ncol = 2, byrow = TRUE,
                 dimnames = list(c("x", "k"), c("theta_x0", "theta_lk")))
  r2 <- solveODE(mod, tvec, pars, sens1ini = Phi2,
                 abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(r1$sens1), c(length(tvec), 1L, 1L))
  expect_equal(dim(r2$sens1), c(length(tvec), 1L, 2L))
  expect_equal(dimnames(r1$sens1)$sens, "theta_lk")
  expect_equal(dimnames(r2$sens1)$sens, c("theta_x0", "theta_lk"))

  # theta_lk column agrees across calls. Tolerance loosened to 1e-7
  # (was 1e-10): with the SoA tangent slab + BLAS daxpy / dscal, FMA
  # in the BLAS kernels produces ~1e-9 round-off drift vs the legacy
  # per-element ET path that paired separate mul+add. Same algebra,
  # different rounding: well within the 1e-10 abstol/reltol regime.
  expect_equal(as.numeric(r1$sens1[, 1, 1]),
               as.numeric(r2$sens1[, 1, 2]),
               tolerance = 1e-7)
})

test_that("M = 0 fast-path: empty sens slot, state integration intact", {
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_M_zero",
                deriv = TRUE, nStack = 2L)
  pars <- c(x = 1.0, k = 0.5)
  tvec <- seq(0, 2, by = 0.5)

  Phi0 <- matrix(0, nrow = 2, ncol = 0,
                 dimnames = list(c("x", "k"), character(0)))
  r <- solveODE(mod, tvec, pars, sens1ini = Phi0,
                abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(r$sens1), c(length(tvec), 1L, 0L))
  expect_equal(as.numeric(r$variable[, 1]),
               exp(-0.5 * tvec), tolerance = 1e-8)
})

test_that("CVODE: M = 0 fast-path skips sensitivity integration", {
  skip_if_not(isTRUE(cvodeConfig$available), "CVODE backend not available")

  mod <- CVODE(c(x = "-k*x"), modelname = "rep_M_zero_cv",
               deriv = TRUE)
  pars <- c(x = 1.0, k = 0.5)
  tvec <- seq(0, 2, by = 0.5)

  Phi0 <- matrix(0, nrow = 2, ncol = 0,
                 dimnames = list(c("x", "k"), character(0)))
  r <- solveODE(mod, tvec, pars, sens1ini = Phi0,
                abstol = 1e-10, reltol = 1e-10)

  expect_equal(dim(r$sens1), c(length(tvec), 1L, 0L))
  expect_equal(as.numeric(r$variable[, 1]),
               exp(-0.5 * tvec), tolerance = 1e-8)
})

test_that("M > nStack is rejected with a clear error", {
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_M_over",
                deriv = TRUE, nStack = 2L)
  pars <- c(x = 1.0, k = 0.5)
  Phi_over <- matrix(0, nrow = 2, ncol = 3,
                     dimnames = list(c("x", "k"), c("a", "b", "c")))
  expect_error(
    solveODE(mod, c(0, 1), pars, sens1ini = Phi_over,
             abstol = 1e-10, reltol = 1e-10),
    "exceeds the model's compile-time nStack"
  )
})

# -- Partial-row sens1ini (rowname-driven implicit fixed) --------------------

test_that("partial-row sens1ini matches zero-padded full Phi'", {
  rhs <- c(A = "-k1*A + k2*B",
           B =  "k1*A - k2*B")
  mod <- CppODE(rhs, modelname = "rep_partial", deriv = TRUE)
  pars <- c(A = 1.0, B = 0.0, k1 = 0.3, k2 = 0.1)
  tvec <- seq(0, 5, length.out = 11)
  tight <- list(abstol = 1e-10, reltol = 1e-10)

  # Perturb only k1: partial form supplies a single row.
  Phi_partial <- matrix(c(0.3), nrow = 1, ncol = 1,
                        dimnames = list("k1", "log_k1"))
  res_p <- solveODE(mod, tvec, pars, sens1ini = Phi_partial,
                    abstol = tight$abstol, reltol = tight$reltol)

  # Equivalent full Phi': zero on (A, B, k2), 0.3 on k1.
  Phi_full <- matrix(c(0, 0, 0.3, 0), nrow = 4, ncol = 1,
                     dimnames = list(c("A", "B", "k1", "k2"), "log_k1"))
  res_f <- solveODE(mod, tvec, pars, sens1ini = Phi_full,
                    abstol = tight$abstol, reltol = tight$reltol)

  expect_equal(dim(res_p$sens1), c(length(tvec), 2L, 1L))
  expect_equal(dimnames(res_p$sens1)$sens, "log_k1")
  expect_equal(as.numeric(res_p$sens1), as.numeric(res_f$sens1),
               tolerance = 1e-9)
})

test_that("partial-row sens1ini accepts mixed state/param rows in any order", {
  rhs <- c(A = "-k1*A + k2*B",
           B =  "k1*A - k2*B")
  mod <- CppODE(rhs, modelname = "rep_partial_mix", deriv = TRUE)
  pars <- c(A = 1.0, B = 0.0, k1 = 0.3, k2 = 0.1)
  tvec <- seq(0, 3, length.out = 7)
  tight <- list(abstol = 1e-10, reltol = 1e-10)

  # Two rows in non-canonical order: k2 first, then A. Two thetas.
  # Layout: row(k2)=(0.1,0), row(A)=(0,1) -> theta1 perturbs k2, theta2 perturbs A.
  Phi_partial <- matrix(c(0.1, 0,
                          0,   1),
                        nrow = 2, ncol = 2, byrow = TRUE,
                        dimnames = list(c("k2", "A"),
                                        c("log_k2", "A0")))
  res_p <- solveODE(mod, tvec, pars, sens1ini = Phi_partial,
                    abstol = tight$abstol, reltol = tight$reltol)

  # Full equivalent: zeros except k2 in col 1, A in col 2.
  Phi_full <- matrix(0, nrow = 4, ncol = 2,
                     dimnames = list(c("A", "B", "k1", "k2"),
                                     c("log_k2", "A0")))
  Phi_full["k2", "log_k2"] <- 0.1
  Phi_full["A",  "A0"]     <- 1.0
  res_f <- solveODE(mod, tvec, pars, sens1ini = Phi_full,
                    abstol = tight$abstol, reltol = tight$reltol)

  expect_equal(as.numeric(res_p$sens1), as.numeric(res_f$sens1),
               tolerance = 1e-9)
})

test_that("partial-row sens1ini without rownames is rejected", {
  rhs <- c(A = "-k1*A + k2*B",
           B =  "k1*A - k2*B")
  mod <- CppODE(rhs, modelname = "rep_partial_noname", deriv = TRUE)
  pars <- c(A = 1.0, B = 0.0, k1 = 0.3, k2 = 0.1)
  # 1 row, n_active = 4 columns, no rownames -> ambiguous.
  Phi_bad <- matrix(0, nrow = 1, ncol = 4)
  expect_error(
    solveODE(mod, c(0, 1), pars, sens1ini = Phi_bad,
             abstol = 1e-10, reltol = 1e-10),
    "expected"
  )
})

test_that("partial-row sens1ini with unknown rownames is rejected", {
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_partial_unknown",
                deriv = TRUE)
  pars <- c(x = 1.0, k = 0.5)
  Phi_bad <- matrix(0.5, nrow = 1, ncol = 1,
                    dimnames = list("not_a_name", "theta"))
  expect_error(
    solveODE(mod, c(0, 1), pars, sens1ini = Phi_bad,
             abstol = 1e-10, reltol = 1e-10),
    "unknown row names"
  )
})

test_that("partial-row sens1ini rejects 'fixed' argument", {
  mod <- CppODE(c(x = "-k*x"), modelname = "rep_partial_fixed",
                deriv = TRUE)
  Phi_partial <- matrix(0.5, nrow = 1, ncol = 1,
                        dimnames = list("k", "log_k"))
  expect_error(
    solveODE(mod, c(0, 1), c(x = 1, k = 0.5),
             sens1ini = Phi_partial, fixed = "k"),
    "not supported"
  )
})

test_that("partial-row sens2ini matches zero-padded full Phi''", {
  rhs <- c(A = "-k*A")
  mod <- CppODE(rhs, modelname = "rep_partial_d2",
                deriv = TRUE, deriv2 = TRUE)
  pars <- c(A = 1.0, k = 0.5)
  tvec <- seq(0, 2, by = 0.5)
  tight <- list(abstol = 1e-10, reltol = 1e-10)

  # Single theta perturbing only k: Phi' partial on k.
  Phi1 <- matrix(0.5, nrow = 1, ncol = 1,
                 dimnames = list("k", "log_k"))
  # Phi'' partial: d^2/dtheta^2 of exp(theta) on k = 0.5.
  Phi2_partial <- array(0.5, dim = c(1, 1, 1),
                        dimnames = list("k", "log_k", "log_k"))
  res_p <- solveODE(mod, tvec, pars,
                    sens1ini = Phi1, sens2ini = Phi2_partial,
                    abstol = tight$abstol, reltol = tight$reltol)

  # Equivalent full Phi''.
  Phi2_full <- array(0, dim = c(2, 1, 1),
                     dimnames = list(c("A", "k"), "log_k", "log_k"))
  Phi2_full["k", "log_k", "log_k"] <- 0.5
  res_f <- solveODE(mod, tvec, pars,
                    sens1ini = Phi1, sens2ini = Phi2_full,
                    abstol = tight$abstol, reltol = tight$reltol)

  expect_equal(as.numeric(res_p$sens2), as.numeric(res_f$sens2),
               tolerance = 1e-9)
})

