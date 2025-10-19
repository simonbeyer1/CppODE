rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)
library(cOde)

# Define ODE system
eqns <- c(
  STATE = "-k*STATE*time"
)

# # Define an event
# events <- data.frame(
#   var   = "X",
#   time  = "t_e",
#   value = 1,
#   method= "add",
#   root  = NA
# )

# Generate and compile solver
f <- CppODE::CppFun(eqns, modelname = "example2",
                    deriv = T, deriv2 = T, useDenseOutput=TRUE, verbose = T)

# Wrap in an R solver function
solve <- function(times, params,
                  abstol = 1e-6, reltol = 1e-6,
                  maxattemps = 5000, maxsteps = 1e6,
                  roottol = 1e-8, maxroot = 1) {

  paramnames <- c(attr(f, "variables"), attr(f, "parameters"))
  missing <- setdiff(paramnames, names(params))
  if (length(missing) > 0) stop("Missing parameters: ", paste(missing, collapse = ", "))

  params <- params[paramnames]

  # Call C++ solver - returns a list now
  result <- .Call(paste0("solve_", as.character(f)),
                  as.numeric(times),
                  as.numeric(params),
                  as.numeric(abstol),
                  as.numeric(reltol),
                  as.integer(maxattemps),
                  as.integer(maxsteps),
                  as.numeric(roottol),
                  as.integer(maxroot))

  # Extract dimension names
  dims <- attr(f, "dim_names")

  # Add column names to state matrix
  colnames(result$state) <- dims$state

  # Add dimension names to sens1 array if present
  if (!is.null(result$sens1)) {
    dimnames(result$sens1) <- list(NULL, dims$state, dims$sens)
  }

  return(result)
}

# Example run
params <- c(STATE=1, k=1)
times  <- seq(0, 10, length.out = 300)
res <- solve(times, params)

# Access sensitivities
if (!is.null(res$sens1)) {
  # First-order sensitivities for state A at time index 10
  # res$sens1[10, "A", ]

  # Flatten to matrix format if needed (like old output)
  n_out <- length(res$time)
  n_states <- ncol(res$state)
  n_sens <- dim(res$sens1)[3]

  sens_matrix <- matrix(res$sens1, nrow = n_out, ncol = n_states * n_sens)

  # Generate column names (state.param format)
  dims <- attr(f, "dim_names")
  sens_colnames <- as.vector(outer(dims$state, dims$sens, paste, sep = "."))
  colnames(sens_matrix) <- sens_colnames

  # Combine everything into old-style matrix if needed
  out_full <- cbind(time = res$time, res$state, sens_matrix)
  # head(out_full)
}

out_wide <- out_full %>% dMod::wide2long() %>% mutate(solver = "CppODE")
# odemodel0 <- odemodel(eqns)

f.sens <- sensitivitiesSymb(eqns)
outfun <- cOde::funC(c(eqns, f.sens), modelname = "example2cOde")

outC_lsoda <- cOde::odeC(y=c(STATE=1, attr(f.sens, "yini")),
                   times = times ,
                   func= outfun,
                   parms = c(k=1),
                   method = "lsoda")

outC_lsodes <- cOde::odeC(y=c(STATE=1, attr(f.sens, "yini")),
                         times = times ,
                         func= outfun,
                         parms = c(k=1),
                         method = "lsodes")

outC_wide_lsoda <- outC_lsoda %>% dMod::wide2long() %>% mutate(solver = "cOde_lsoda")
outC_wide_lsodes <- outC_lsodes %>% dMod::wide2long() %>% mutate(solver = "cOde_lsodes")
# x <- odemodel0 %>% Xs()
# out_dMod <- x(times, params)

# getDerivs(out_dMod) %>% plot()

# out_dMod_wide <- getDerivs(out_dMod)[[1]] %>% wide2long() %>% mutate(solver = "dMod")
# out_dMod_wide <- rbind(out_dMod_wide, out_dMod[[1]] %>% wide2long() %>% mutate(solver = "dMod"))
Xanalytical <- function(time, pars) {
  STATE0 <- pars[["STATE"]]
  k  <- pars[["k"]]

  # analytische Lösung
  STATE   <- STATE0 * exp(-0.5 * k * time^2)
  STATE.STATE <- exp(-0.5 * k * time^2)         # dX/dX0
  STATE.k <- -0.5 * time^2 * STATE              # dX/dk

  data.frame(
    time  = rep(time, times = 3),
    name  = rep(c("STATE", "STATE.STATE", "STATE.k"), each = length(time)),
    value = c(STATE, STATE.STATE, STATE.k)
  )
}


out_analytical <- Xanalytical(times, params) %>% mutate(solver = "analytical")

out_wide <- rbind(out_wide, out_analytical, outC_wide_lsoda, outC_wide_lsodes)

ggplot(out_wide, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line(linewidth = 1) +
  facet_wrap(~name, scales = "free") +
  dMod::theme_dMod()
