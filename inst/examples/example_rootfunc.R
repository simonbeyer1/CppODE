library(CppODE)
library(tidyr)
library(dplyr)
library(ggplot2)

rhs <- c(
  O   = "2 * J1 * O2 + J3 * O3 + J_NO2 * NO2 - k2 * O * O2 - k4 * O * O3",
  O2  = "-J1 * O2 + J3 * O3 + 2 * k4 * O * O3 + k_NO * NO * O3 - k2 * O * O2",
  O3  = "k2 * O * O2 - J3 * O3 - k4 * O * O3 - k_NO * NO * O3",
  NO  = "-k_NO * NO * O3 + J_NO2 * NO2",
  NO2 = "k_NO * NO * O3 - J_NO2 * NO2"
)


model <- CppODE(rhs)

parms <- c(
  O   = 0.0001,   # ppm
  O2  = 210000,   # ppm (≈ 21 %)
  O3  = 5,        # ppm
  NO  = 0.02,     # ppm
  NO2 = 0.01,     # ppm
  J1     = 0.00001,   # O2-Photolyse
  J3     = 0.001,     # O3-Photolyse
  J_NO2  = 0.01,      # NO2-Photolyse
  k2     = 0.02,      # O + O2 -> O3
  k4     = 0.01,      # O + O3 -> 2 O2
  k_NO   = 0.05       # NO + O3 -> NO2 + O2
)


times <- seq(0,1e2, len = 300)


res <- solveODE(model, times, parms)

n_out    <- length(res$time)
n_states <- ncol(res$variable)
n_sens   <- dim(res$sens1)[3]

dims <- attr(model, "dim_names")

## ---------- sens1 ----------
sens1_matrix <- matrix(res$sens1,
                       nrow = n_out,
                       ncol = n_states * n_sens)

sens1_colnames <-
  as.vector(outer(paste0("∂", dims$variable),
                  paste0("∂", dims$sens),
                  paste, sep = "/"))
colnames(sens1_matrix) <- sens1_colnames

levels <- c(colnames(res$variable), sens1_colnames)

## ---------- combine everything ----------
out <- cbind(time = res$time, res$variable, sens1_matrix) %>%
  as.data.frame() %>%
  pivot_longer(
    cols = -time,
    names_to = "name",
    values_to = "value"
  ) %>%
  mutate(name = factor(name, levels = levels)) %>%
  arrange(name, time)

ggplot(out, aes(x = time, y = value)) +
  geom_line() +
  facet_wrap(~name, scales = "free") +
  theme_bw()


