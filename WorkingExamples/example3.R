rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(ggplot2)
library(dplyr)
library(dMod)

# Define ODE system
r <- eqnlist() %>%
  addReaction("STAT"       , "pSTAT"      , "p1*pEpoR*STAT" , "STAT phosphoyrl.") %>%
  addReaction("2*pSTAT"    , "pSTATdimer" , "p2*pSTAT^2"    , "pSTAT dimerization") %>%
  addReaction("pSTATdimer" , "npSTATdimer", "p3*pSTATdimer" , "pSTAT dimer import") %>%
  addReaction("npSTATdimer", "2*nSTAT"    , "p4*npSTATdimer", "dimer dissociation") %>%
  addReaction("nSTAT"      , "STAT"       , "p5*nSTAT"      , "nuclear export")

print(r)

# Parameterize the receptor phosphorylation
receptor <- "((1 - exp(-time*lambda1))*exp(-time*lambda2))^3"
r$rates <- r$rates %>%
  insert("pEpoR ~ pEpoR*rec", rec = receptor)

eqns <- as.eqnvec(r)

# Generate odemodel
model0 <- odemodel(r, modelname = "jakstat", compile = TRUE)

### Prediction and observation functions
# Generate a prediction function
x <- Xs(model0, optionsSens = list(method = "lsoda"))

# Make a prediction based on random parameter values
parameters <- getParameters(x)
pars <- structure(runif(length(parameters), 0, 1), names = parameters)
times <- seq(0, 60, len = 100)
prediction <- x(times, pars)
plot(prediction)


# Define observables like total STAT, total phosphorylated STAT, etc.
observables <- eqnvec(
  tSTAT = "s_tSTAT*(STAT + pSTAT + 2*pSTATdimer)",
  tpSTAT = "s_tpSTAT*(pSTAT + 2*pSTATdimer) + off_tpSTAT",
  pEpoR = paste0("s_EpoR * pEpoR *", receptor)
)

# Define the observation function. Information about states and dynamic parameters
# is contained in reactions
g <- Y(observables, r, modelname = "obsfn", compile = TRUE, attach.input = FALSE)

parameters <- getParameters(x, g)
pars <- structure(runif(length(parameters), 0, 1), names = parameters)
times <- seq(0, 10, len = 100)
prediction <- (g*x)(times, pars)
plot(prediction)

### Parameter transformations
p <- eqnvec() %>%
  # Start with the identity transformation
  define("x~x", x = getParameters(x, g)) %>%
  # Fix some initial values
  define("x~0", x = c("pSTAT", "pSTATdimer", "npSTATdimer", "nSTAT")) %>%
  # Log-transform all current symbols found in the equations
  insert("x~exp(x)", x = .currentSymbols) %>%
  # Generate parameter transformation function
  P(condition = "Epo")

print(getEquations(p))

# Add another parameter transformation
# p <- p +
#   # Start with the current transformation
#   getEquations(p, conditions = "Epo") %>%
#   # Insert multiple of pEpoR everywhere where we finde pEpoR
#   define("pEpoR ~ multiple*exp(pEpoR)") %>%
#   # Generate parameter transformation function with another condition name
#   P(condition = "Epo prediction")
#
# print(getEquations(p))

# # Make a prediction of the observables based on random parameter values
# parameters <- getParameters(p)
# pars <- structure(runif(length(parameters), 0, 1), names = parameters)
# pars["multiple"] <- 2
# times <- seq(0, 10, len = 100)
# prediction <- (g*x*p)(times, pars)
# plot(prediction)


## Parameter estimation

### Preparing the data
data(jakstat)
data <- as.datalist(jakstat, split.by = "condition")
plot(data)

### Objective function
obj <- normL2(data, g*x*p)


outms <- mstrust(obj, pars, fits = 50, iterlim = 1e3, studyname = "jackstatms", cores=10)
out_frame <- outms %>% as.parframe()
plotValues(out_frame)
bestfit <- as.parvec(out_frame)

plot((g*x*p)(times, bestfit), data)

bestfit_params <- p(bestfit)$Epo

f <- CppODE::CppFun(eqns, modelname = "jacstatcpp",
                    deriv = T, deriv2 = F,
                    useDenseOutput=TRUE, verbose = T)

# Wrap in an R solver function
solve <- function(times, params,
                  abstol = 1e-3, reltol = 1e-3,
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


res <- solve(times, bestfit_params)

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
  head(out_full)
}

out_wide <- out_full %>% wide2long() %>% mutate(solver = "CppODE")
out_wide_dM <- x(times, bestfit_params)
oit_wide_dM_derivs <- getDerivs(out_wide_dM)[[1]] %>% wide2long() %>% mutate(solver = "dMod")
out_wide_dM <- out_wide_dM[[1]] %>% wide2long() %>% mutate(solver = "dMod")


states_cpp <- cbind(time = res$time, res$state) %>% wide2long() %>% mutate(solver = "CppODE")
states_wide <- rbind(states_cpp, out_wide_dM)
ggplot(states_wide, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line(linewidth = 1) +
  facet_wrap(~name, scales = "free") +
  theme_dMod()


out_all <- rbind(out_wide, out_wide_dM, oit_wide_dM_derivs) %>%
  filter(stringr::str_detect(name, "p1"))
ggplot(out_all, aes(x = time, y = value, color = solver, linetype = solver)) +
  geom_line(linewidth = 1) +
  facet_wrap(~name, scales = "free") +
  theme_dMod()

