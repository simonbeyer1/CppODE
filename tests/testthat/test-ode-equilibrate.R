# Test steady-state equilibration (rootfunc = "equilibrate") across methods
# and warm-start behaviour.

skip_on_cran()
skip_on_ci()

# -- Shared setup --------------------------------------------------------------
# System with known analytical steady state:
#   R' = k_act - k_deact * R   ->  R_ss = k_act / k_deact
#   A' = -k1*A*R + k2*pA       ->  conservation: A + pA = const
#   pA'=  k1*A*R - k2*pA       ->  pA_ss = k1*R_ss / (k2 + k1*R_ss) * (A0+pA0)

rhs <- c(
  R  = "k_act - k_deact * R",
  A  = "-k1 * A * R + k2 * pA",
  pA = " k1 * A * R - k2 * pA"
)

pars <- c(R = 1, A = 1, pA = 0, k_act = 0.1, k_deact = 0.7,
          k1 = 0.1, k2 = 0.05)
times <- seq(0, 1e3, length.out = 500)

ss_R  <- unname(pars["k_act"] / pars["k_deact"])
total <- unname(pars["A"] + pars["pA"])
ss_pA <- unname(pars["k1"]) * ss_R / (unname(pars["k2"]) + unname(pars["k1"]) * ss_R) * total
ss_A  <- total - ss_pA

stiff_methods   <- c("bdf", "rb4")
all_methods     <- c("bdf", "adams", "msoda", "rb4", "tsit5")

# -- Basic equilibrate: reaches correct steady state ---------------------------

test_that("equilibrate finds the analytical steady state", {
  for (m in all_methods) {
    mod <- CppODE(rhs, rootfunc = "equilibrate", method = m,
                  modelname = paste0("eq_ss_", m))
    res <- solveODE(mod, times, pars, roottol = 1e-06)

    y_final <- res$variable[nrow(res$variable), ]
    expect_equal(unname(y_final["R"]),  ss_R,  tolerance = 1e-4,
                 label = paste(m, "R steady state"))
    expect_equal(unname(y_final["A"]),  ss_A,  tolerance = 1e-4,
                 label = paste(m, "A steady state"))
    expect_equal(unname(y_final["pA"]), ss_pA, tolerance = 1e-4,
                 label = paste(m, "pA steady state"))
  }
})

# -- Equilibrate terminates early ----------------------------------------------

test_that("equilibrate stops integration before final time", {
  for (m in all_methods) {
    mod <- CppODE(rhs, rootfunc = "equilibrate", method = m,
                  modelname = paste0("eq_early_", m))
    res <- solveODE(mod, times, pars, roottol = 1e-04)

    # Should stop well before t = 1000
    expect_lt(max(res$time), 500,
              label = paste(m, "early termination"))
    # But should have progressed past t = 0
    expect_gt(max(res$time), 1,
              label = paste(m, "nontrivial progress"))
  }
})

# -- Equilibrate with sensitivities --------------------------------------------

test_that("equilibrate works with first-order sensitivities", {
  for (m in stiff_methods) {
    mod <- CppODE(rhs, rootfunc = "equilibrate", method = m,
                  deriv = TRUE, modelname = paste0("eq_sens_", m))
    res <- solveODE(mod, times, pars, roottol = 1e-05)

    expect_true(!is.null(res$sens1), label = paste(m, "sens1 present"))
    expect_true(all(is.finite(res$sens1)), label = paste(m, "sens1 finite"))

    # At steady state, sensitivity derivatives should be near zero
    # (that's what the termination checks)
    y_final <- res$variable[nrow(res$variable), ]
    expect_equal(unname(y_final["R"]), ss_R, tolerance = 1e-3,
                 label = paste(m, "R with sens"))
  }
})

# -- Warm start is faster than cold start --------------------------------------

test_that("warm start converges in fewer steps than cold start", {
  for (m in stiff_methods) {
    mod <- CppODE(rhs, rootfunc = "equilibrate", method = m,
                  deriv = TRUE, modelname = paste0("eq_warm_", m))

    # Cold start -> equilibrium
    res1 <- solveODE(mod, times, pars, roottol = 1e-05)
    yini    <- res1$variable[nrow(res1$variable), ]
    sensini <- res1$sens1[length(res1$time), , ]

    # Small parameter perturbation
    pars2 <- pars
    pars2[names(yini)] <- yini
    pars2["k1"] <- pars["k1"] * 1.01   # +1%
    pars2["k2"] <- pars["k2"] * 0.99   # -1%

    # Warm start (with sensitivity initial values)
    res_warm <- solveODE(mod, times, pars2, sens1ini = sensini,
                         roottol = 1e-05)

    # Cold start (same perturbed params, no sens1ini)
    res_cold <- solveODE(mod, times, pars2, roottol = 1e-05)

    d_warm <- diagnostics(res_warm)
    d_cold <- diagnostics(res_cold)

    expect_lt(d_warm$accepted, d_cold$accepted,
              label = paste(m, "warm < cold steps"))
  }
})

# -- Warm start reaches correct steady state -----------------------------------

test_that("warm start converges to correct new steady state", {
  mod <- CppODE(rhs, rootfunc = "equilibrate", deriv = TRUE,
                modelname = "eq_warm_correct")

  # First equilibration
  res1 <- solveODE(mod, times, pars, roottol = 1e-06)
  yini    <- res1$variable[nrow(res1$variable), ]
  sensini <- res1$sens1[length(res1$time), , ]

  # Perturb parameters
  pars2 <- pars
  pars2[names(yini)] <- yini
  pars2["k_act"] <- 0.15  # changed from 0.1

  res2 <- solveODE(mod, times, pars2, sens1ini = sensini, roottol = 1e-06)

  # New analytical steady state
  ss_R2  <- unname(0.15 / pars["k_deact"])
  total2 <- unname(pars2["A"] + pars2["pA"])
  ss_pA2 <- unname(pars2["k1"]) * ss_R2 / (unname(pars2["k2"]) + unname(pars2["k1"]) * ss_R2) * total2
  ss_A2  <- total2 - ss_pA2

  y_final <- res2$variable[nrow(res2$variable), ]
  expect_equal(unname(y_final["R"]), ss_R2, tolerance = 1e-4)
  expect_equal(unname(y_final["A"]), ss_A2, tolerance = 1e-4)
})

# -- Tight tolerance equilibrate -----------------------------------------------

test_that("equilibrate respects tight roottol", {
  mod <- CppODE(rhs, rootfunc = "equilibrate", modelname = "eq_tight")

  res_loose <- solveODE(mod, times, pars, roottol = 1e-02)
  res_tight <- solveODE(mod, times, pars, roottol = 1e-08)

  # Tight tolerance should require more time / steps
  expect_gt(max(res_tight$time), max(res_loose$time),
            label = "tight tol needs more integration time")
})

# -- Already at steady state: immediate termination (no sensitivities) ---------

test_that("equilibrate terminates immediately when starting at SS (deriv=FALSE)", {
  mod <- CppODE(rhs, rootfunc = "equilibrate", deriv = FALSE,
                modelname = "eq_at_ss")

  # Start exactly at the analytical steady state
  pars_ss <- pars
  pars_ss["R"]  <- ss_R
  pars_ss["A"]  <- ss_A
  pars_ss["pA"] <- ss_pA

  res <- solveODE(mod, times, pars_ss, roottol = 1e-04)

  # Without sensitivities, should terminate after very few steps
  d <- diagnostics(res)
  expect_lt(d$accepted, 5, label = "immediate SS detection")
})

# -- At SS with sensitivities: needs to equilibrate sens -----------------------

test_that("equilibrate at SS with deriv=TRUE still needs sensitivity equilibration", {
  mod <- CppODE(rhs, rootfunc = "equilibrate", deriv = TRUE,
                modelname = "eq_at_ss_sens")

  pars_ss <- pars
  pars_ss["R"]  <- ss_R
  pars_ss["A"]  <- ss_A
  pars_ss["pA"] <- ss_pA

  res <- solveODE(mod, times, pars_ss, roottol = 1e-04)

  # States are at SS but sensitivities start at 0 -> need integration
  d <- diagnostics(res)
  expect_gt(d$accepted, 5, label = "sens equilibration takes steps")

  # But should still reach the correct state SS
  y_final <- res$variable[nrow(res$variable), ]
  expect_equal(unname(y_final["R"]), ss_R, tolerance = 1e-4)
})
