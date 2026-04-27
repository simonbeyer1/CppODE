## =================================================================
## Robertson test: all CppODE methods vs deSolve (bdf, vode, lsode)
## =================================================================
rm(list = ls(all.names = TRUE))

.workingDir <- file.path(tempdir(), "CppODE_bench_robertson")
dir.create(.workingDir, showWarnings = FALSE, recursive = TRUE)
setwd(.workingDir)

library(CppODE)
library(deSolve)

# --- Robertson system --------------------------------------------------------
rob <- c(y1 = "-0.04*y1 + 1e4*y2*y3",
         y2 = "0.04*y1 - 1e4*y2*y3 - 3e7*y2*y2",
         y3 = "3e7*y2*y2")

p0    <- c(y1 = 1, y2 = 0, y3 = 0)
times <- c(0, 10^seq(-5, 5, length.out = 100))   # 0 + log-spaced up to 1e5

ATOL  <- 1e-8
RTOL  <- 1e-6

# --- deSolve function --------------------------------------------------------
rob_fun <- function(t, y, p) {
  with(as.list(y), list(c(
    -0.04*y1 + 1e4*y2*y3,
     0.04*y1 - 1e4*y2*y3 - 3e7*y2*y2,
     3e7*y2*y2
  )))
}

rob_jac <- function(t, y, p) {
  with(as.list(y), matrix(c(
    -0.04,       1e4*y3,        1e4*y2,
     0.04, -1e4*y3 - 6e7*y2, -1e4*y2,
     0,          6e7*y2,        0
  ), nrow = 3, byrow = TRUE))
}

# --- deSolve references ------------------------------------------------------
cat("============================================================\n")
cat("deSolve references (bdf, vode, lsode)\n")
cat("============================================================\n")

ref_bdf <- ode(y = p0[1:3], times = times, func = rob_fun, parms = NULL,
               method = "bdf", jacfunc = rob_jac,
               rtol = RTOL, atol = ATOL)
cat("deSolve bdf:   OK (", nrow(ref_bdf), "rows )\n")

ref_vode <- ode(y = p0[1:3], times = times, func = rob_fun, parms = NULL,
                method = "vode", jacfunc = rob_jac,
                rtol = RTOL, atol = ATOL)
cat("deSolve vode:  OK (", nrow(ref_vode), "rows )\n")

ref_lsode <- ode(y = p0[1:3], times = times, func = rob_fun, parms = NULL,
                 method = "lsode", jacfunc = rob_jac,
                 rtol = RTOL, atol = ATOL)
cat("deSolve lsode: OK (", nrow(ref_lsode), "rows )\n")

ref_lsoda <- lsoda(y = p0[1:3], times = times, func = rob_fun, parms = NULL,
                   jacfunc = rob_jac, rtol = RTOL, atol = ATOL)
cat("deSolve lsoda: OK (", nrow(ref_lsoda), "rows )\n")

# Use bdf as "truth" for comparisons
ref <- ref_vode

# --- CppODE methods to test --------------------------------------------------
methods <- c("bdf", "rb4")

cat("\n============================================================\n")
cat("Compiling CppODE models for all methods\n")
cat("============================================================\n")

results <- list()
for (meth in methods) {
  mname <- paste0("rob_", gsub("\\+", "p", meth))
  cat(sprintf("\n--- Method: %s ---\n", meth))

  # Adams is non-stiff only; Robertson is stiff => expect failure
  # ndf++/bdf++ should handle it (switch to stiff mode)
  skip <- FALSE

  tryCatch({
    m <- CppODE(rob, deriv = FALSE, deriv2 = FALSE, outdir = getwd(),
                method = meth, modelname = mname, compile = TRUE,
                includeTimeZero = TRUE, verbose = FALSE)
    cat("  Compiled OK\n")

    tryCatch({
      r <- solveODE(m, times, p0, abstol = ATOL, reltol = RTOL)
      nout <- nrow(r$variable)  # time-first layout
      cat(sprintf("  Solved OK: %d / %d output points\n", nout, length(times)))

      if (nout == length(times)) {
        # Compare with deSolve bdf reference
        err_y1 <- max(abs(as.numeric(r$variable[, 1]) - ref[, "y1"]))
        err_y2 <- max(abs(as.numeric(r$variable[, 2]) - ref[, "y2"]))
        err_y3 <- max(abs(as.numeric(r$variable[, 3]) - ref[, "y3"]))
        mass   <- as.numeric(r$variable[, 1] + r$variable[, 2] + r$variable[, 3])
        mass_err <- max(abs(mass - 1))

        cat(sprintf("  max|y1 - ref|: %.4e\n", err_y1))
        cat(sprintf("  max|y2 - ref|: %.4e\n", err_y2))
        cat(sprintf("  max|y3 - ref|: %.4e\n", err_y3))
        cat(sprintf("  mass conservation error: %.4e\n", mass_err))

        results[[meth]] <- list(ok = TRUE, y = r$variable, err = c(err_y1, err_y2, err_y3),
                                mass_err = mass_err, nout = nout)
      } else {
        cat("  WARNING: fewer output points than requested!\n")
        results[[meth]] <- list(ok = FALSE, reason = "incomplete output", nout = nout)
      }
    }, error = function(e) {
      cat(sprintf("  SOLVE FAILED: %s\n", conditionMessage(e)))
      results[[meth]] <<- list(ok = FALSE, reason = conditionMessage(e))
    })
  }, error = function(e) {
    cat(sprintf("  COMPILE FAILED: %s\n", conditionMessage(e)))
    results[[meth]] <<- list(ok = FALSE, reason = conditionMessage(e))
  })
}

# --- deSolve cross-comparison ------------------------------------------------
cat("\n============================================================\n")
cat("deSolve cross-comparison (all vs deSolve::vode reference)\n")
cat("============================================================\n")
for (nm in c("vode", "lsode", "lsoda")) {
  r <- switch(nm, vode = ref_vode, lsode = ref_lsode, lsoda = ref_lsoda)
  cat(sprintf("  %s: max|y1|=%.4e  max|y2|=%.4e  max|y3|=%.4e\n", nm,
              max(abs(r[, "y1"] - ref[, "y1"])),
              max(abs(r[, "y2"] - ref[, "y2"])),
              max(abs(r[, "y3"] - ref[, "y3"]))))
}

# --- Summary table -----------------------------------------------------------
cat("\n============================================================\n")
cat("SUMMARY\n")
cat("============================================================\n")
cat(sprintf("%-8s  %6s  %12s  %12s  %12s  %12s\n",
            "Method", "OK?", "max|dy1|", "max|dy2|", "max|dy3|", "mass_err"))
cat(strrep("-", 72), "\n")
for (meth in methods) {
  r <- results[[meth]]
  if (!is.null(r) && isTRUE(r$ok)) {
    cat(sprintf("%-8s  %6s  %12.4e  %12.4e  %12.4e  %12.4e\n",
                meth, "YES", r$err[1], r$err[2], r$err[3], r$mass_err))
  } else {
    reason <- if (!is.null(r)) r$reason else "not run"
    cat(sprintf("%-8s  %6s  %s\n", meth, "NO", substr(reason, 1, 50)))
  }
}
