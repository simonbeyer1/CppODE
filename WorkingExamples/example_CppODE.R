rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)
library(tidyr)

# Define ODE system
eqns <- c(
  A = "-k1*A^2*time + k2*B - k3 * A",
  B = "k1*A^2*time - k2*B"
)

# Define an event
events <- data.frame(
  var   = c("A", "A"),
  time  = c(NA, 0),
  value = c("dose","dose"),
  method= c("replace","add"),
  root  = c("A-Acrit", NA),
  stringsAsFactors = FALSE
)

# # Generate and compile solver
f_controlled <- CppODE(eqns, events = events, modelname = "Amodel_c", compile = T, useDenseOutput = F, fullErr = F)
f_dense <- CppODE(eqns, events = events, modelname = "Amodel_d", compile = T, useDenseOutput = T, fullErr = F)

# Wrap in an R solver function
solve_c <- function(times, params,
                    abstol = 1e-6, reltol = 1e-6,
                    maxattemps = 100L, maxsteps = 1e6L,
                    hini = 0.1, roottol = 1e-10, maxroot = 4L) {
  paramnames <- c(attr(f_controlled, "variables"), attr(f_controlled, "parameters"))
  missing <- setdiff(paramnames, names(params))
  if (length(missing) > 0) stop("Missing parameters: ", paste(missing, collapse = ", "))
  params <- params[paramnames]
  out <- .Call(paste0("solve_", as.character(f_controlled)),
               as.numeric(times),
               as.numeric(params),
               as.numeric(abstol),
               as.numeric(reltol),
               as.integer(maxattemps),
               as.integer(maxsteps),
               as.numeric(hini),
               as.numeric(roottol),
               as.integer(maxroot))

  # Extract dimension names
  dims <- attr(f_controlled, "dim_names")

  # Add column names to state matrix
  colnames(out$variable) <- dims$variable

  # Add dimension names to sens1 array if present
  if (!is.null(out$sens1)) {
    dimnames(out$sens1) <- list(NULL, dims$variable, dims$sens)
  }

  if (!is.null(out$sens2)) {
    dimnames(out$sens2) <- list(NULL, dims$variable, dims$sens, dims$sens)
  }

  return(out)
}

solve_d <- function(times, params,
                    abstol = 1e-6, reltol = 1e-6,
                    maxattemps = 100L, maxsteps = 1e6L,
                    hini = 0.1, roottol = 1e-10, maxroot = 4L) {
  paramnames <- c(attr(f_dense, "variables"), attr(f_dense, "parameters"))
  missing <- setdiff(paramnames, names(params))
  if (length(missing) > 0) stop("Missing parameters: ", paste(missing, collapse = ", "))
  params <- params[paramnames]
  out <- .Call(paste0("solve_", as.character(f_dense)),
               as.numeric(times),
               as.numeric(params),
               as.numeric(abstol),
               as.numeric(reltol),
               as.integer(maxattemps),
               as.integer(maxsteps),
               as.numeric(hini),
               as.numeric(roottol),
               as.integer(maxroot))

  # Extract dimension names
  dims <- attr(f_dense, "dim_names")

  # Add column names to state matrix
  colnames(out$variable) <- dims$variable

  # Add dimension names to sens1 array if present
  if (!is.null(out$sens1)) {
    dimnames(out$sens1) <- list(NULL, dims$variable, dims$sens)
  }

  if (!is.null(out$sens2)) {
    dimnames(out$sens2) <- list(NULL, dims$variable, dims$sens, dims$sens)
  }

  return(out)
}

# Example run
params <- c(A = 0, B = 0, k1 = 0.1, k2 = 0.2, k3 = 0.1, dose = 1, Acrit = 0.25)
times  <- seq(-10, 400, length.out = 1000)
res_c <- solve_c(times, params)
res_d <- solve_d(times, params)
res_c$variable
head(res_c$sens1[, "A", ])

res_d$sens2[10, "A", , ]

out_c <- as.data.frame(res_c$variable) %>%
  mutate(time = res_c$time) %>%
  pivot_longer(-time, names_to = "name", values_to = "value") %>%
  mutate(method = "controlled")

out_d <- as.data.frame(res_d$variable) %>%
  mutate(time = res_d$time) %>%
  pivot_longer(-time, names_to = "name", values_to = "value") %>%
  mutate(method = "dense")

out <- rbind(out_c, out_d)

ggplot(out, aes(x = time, y = value)) +
  geom_line() +
  facet_wrap(~ name + method, scales = "free_y") +
  dMod::theme_dMod() +
  labs(
    x = "Time",
    y = "Value"
  )

system.time({solve_c(times, params)})
system.time({solve_d(times, params)})
