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
  A = "-k1*A + k2*B",
  B = "k1*A - k2*B"
)

# Define an event
events <- data.frame(
  var   = "A",
  time  = 0,
  value = "dose",
  method= "add",
  root  = NA
)

# Generate and compile solver
f <- CppODE(eqns, events = events, modelname = "Amodel_s", deriv2 = T)

# Wrap in an R solver function
solve <- function(times, params,
                  abstol = 1e-6, reltol = 1e-6,
                  maxattemps = 5000L, maxsteps = 1e6L,
                  roottol = 1e-6, maxroot = 1L) {
  paramnames <- c(attr(f, "variables"), attr(f, "parameters"))
  missing <- setdiff(paramnames, names(params))
  if (length(missing) > 0) stop("Missing parameters: ", paste(missing, collapse = ", "))
  params <- params[paramnames]
  out <- .Call(paste0("solve_", as.character(f)),
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
params <- c(A = 0, B = 0, k1 = 0.1, k2 = 0.2, dose = 1)
times  <- seq(-10, 100, length.out = 400)
res <- solve(times, params)
res$variable
head(res$sens1[, "A", ])

res$sens2[10, "A", , ]

out <- as.data.frame(res$variable) %>%
  mutate(time = res$time) %>%
  pivot_longer(-time, names_to = "name", values_to = "value")

ggplot(out, aes(x = time, y = value)) +
  geom_line() +
  facet_wrap(~ name, scales = "free_y") +
  dMod::theme_dMod() +
  labs(
    x = "Time",
    y = "Value"
  )

