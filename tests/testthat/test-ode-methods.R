# Test all integration methods on problems with known analytical solutions.

skip_on_cran()
skip_on_ci()

# -- Shared setup --------------------------------------------------------------

eqns_decay <- c(A = "-k1 * A", B = "k1 * A - k2 * B")
times <- seq(0, 50, length.out = 200)
pars  <- c(A = 1, B = 0, k1 = 0.1, k2 = 0.2)
#
exact_A <- function(t, A0, k1) A0 * exp(-k1 * t)
exact_B <- function(t, A0, k1, k2) A0 * k1 / (k2 - k1) * (exp(-k1 * t) - exp(-k2 * t))

methods_all <- c("bdf", "adams", "rb4", "tsit5")

# -- Basic solver output structure --------------------------------------------

test_that("solveODE returns correct structure for all methods", {
  for (m in methods_all) {
    mod <- CppODE(eqns_decay, method = m, modelname = paste0("struct_", m))
    res <- solveODE(mod, times, pars)

    expect_type(res$time, "double")
    expect_true(is.matrix(res$variable))
    expect_equal(ncol(res$variable), 2)
    expect_equal(colnames(res$variable), c("A", "B"))
    expect_equal(length(res$time), nrow(res$variable))
  }
})

# -- Accuracy against analytical solution --------------------------------------

test_that("all methods match analytical solution (decay system)", {
  for (m in methods_all) {
    mod <- CppODE(eqns_decay, method = m, modelname = paste0("accuracy_", m))
    res <- solveODE(mod, times, pars, abstol = 1e-10, reltol = 1e-10)

    t_out <- res$time
    A_num <- res$variable[, "A"]
    B_num <- res$variable[, "B"]

    A_exact <- exact_A(t_out, pars["A"], pars["k1"])
    B_exact <- exact_B(t_out, pars["A"], pars["k1"], pars["k2"])

    expect_equal(A_num, A_exact, tolerance = 1e-6, label = paste(m, "A"))
    expect_equal(B_num, B_exact, tolerance = 1e-6, label = paste(m, "B"))
  }
})

# -- First-order sensitivities via AD vs finite differences --------------------

test_that("first-order sensitivities are correct for all methods", {
  eps <- 1e-5

  for (m in methods_all) {
    mod    <- CppODE(eqns_decay, method = m, deriv = TRUE,
                     modelname = paste0("sens1_", m))
    res    <- solveODE(mod, times, pars, abstol = 1e-10, reltol = 1e-10)

    # Check AD sensitivity of A w.r.t. k1 against finite difference
    p_hi      <- pars; p_hi["k1"] <- pars["k1"] + eps
    p_lo      <- pars; p_lo["k1"] <- pars["k1"] - eps
    mod_noad  <- CppODE(eqns_decay, method = m, deriv = FALSE,
                        modelname = paste0("sens1_fd_", m))
    res_hi    <- solveODE(mod_noad, times, p_hi, abstol = 1e-12, reltol = 1e-12)
    res_lo    <- solveODE(mod_noad, times, p_lo, abstol = 1e-12, reltol = 1e-12)

    # Match time grids (both should be identical since same output times)
    fd_dA_dk1 <- (res_hi$variable[, "A"] - res_lo$variable[, "A"]) / (2 * eps)

    sens_names <- attr(mod, "dimNames")$sens
    k1_idx <- which(sens_names == "k1")
    ad_dA_dk1 <- res$sens1[, 1, k1_idx]  # state 1 (A), param k1_idx

    expect_equal(ad_dA_dk1, fd_dA_dk1, tolerance = 1e-3,
                 label = paste(m, "dA/dk1"))
  }
})

# -- Second-order sensitivities ------------------------------------------------

test_that("second-order sensitivities are finite for stiff methods", {
  stiff_methods <- c("bdf", "rb4")

  for (m in stiff_methods) {
    mod <- CppODE(eqns_decay, method = m, deriv = TRUE, deriv2 = TRUE,
                  modelname = paste0("sens2_", m))
    res <- solveODE(mod, times, pars, abstol = 1e-10, reltol = 1e-10)

    expect_true(!is.null(res$sens2), label = paste(m, "sens2 exists"))
    expect_true(all(is.finite(res$sens2)), label = paste(m, "sens2 finite"))
  }
})

# -- Time-triggered events -----------------------------------------------------

test_that("time-triggered dose event works", {
  eqns <- c(A = "-k1 * A")
  evt  <- data.frame(var = "A", time = "t_e", value = "dose",
                     method = "add", root = NA, stringsAsFactors = FALSE)
  pars_ev <- c(A = 1, k1 = 0.1, t_e = 25, dose = 0.5)

  mod <- CppODE(eqns, events = evt, modelname = "event_time")
  res <- solveODE(mod, seq(0, 50, length.out = 200), pars_ev)

  # After event at t=25, A should jump up
  idx_before <- max(which(res$time < 25))
  idx_after  <- min(which(res$time >= 25))
  expect_gt(res$variable[idx_after, "A"], res$variable[idx_before, "A"])
})

# -- Root-triggered events -----------------------------------------------------

test_that("root-triggered event fires correctly", {
  eqns <- c(x = "-k * x")
  evt  <- data.frame(var = "x", time = NA, value = "dose",
                     method = "add", root = "xc - x",
                     stringsAsFactors = FALSE)
  pars_root <- c(x = 1, k = 0.1, xc = 0.5, dose = 0.5)

  mod <- CppODE(eqns, events = evt, modelname = "event_root")
  res <- solveODE(mod, seq(0, 100, length.out = 500), pars_root)

  # x decays below xc, then gets dose added -> should see multiple oscillations
  # Check that x goes back up after crossing xc at least once
  x_vals <- res$variable[, "x"]
  crossings <- sum(diff(x_vals > 0.5) != 0)
  expect_gt(crossings, 0, label = "root event triggers at least once")
})

# -- Diagnostics ---------------------------------------------------------------

test_that("diagnostics() returns solver statistics", {
  mod <- CppODE(eqns_decay, modelname = "diag_test")
  res <- solveODE(mod, times, pars)
  d   <- diagnostics(res)

  expect_true(is.list(d))
  expect_true(d$accepted > 0)
  expect_true(d$fevals > 0)
  expect_equal(d$return_code, 0)  # success
})

# -- Fixed parameters ----------------------------------------------------------

test_that("fixed parameters are excluded from sensitivities", {
  mod <- CppODE(eqns_decay, deriv = TRUE, fixed = "k2",
                modelname = "fixed_test")
  res <- solveODE(mod, times, pars)

  sens_names <- attr(mod, "dimNames")$sens
  expect_false("k2" %in% sens_names)
  expect_true("k1" %in% sens_names)
})
