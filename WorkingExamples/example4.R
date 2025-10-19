rm(list = ls(all.names = TRUE))
# Create and set a specific working directory inside your project folder
.workingDir <- file.path(purrr::reduce(1:1, ~dirname(.x), .init = rstudioapi::getSourceEditorContext()$path), "wd")
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

# --- Libraries ---------------------------------------------------------------
library(deSolve)
library(ggplot2)
library(dplyr)
library(tidyr)

# --- Model -------------------------------------------------------------------
model <- function(t, y, parms) {
  k   <- parms["k"]
  x   <- y[1]
  sx0 <- y[2]
  sk  <- y[3]

  dx   <- -k * t * x
  dsx0 <- -k * t * sx0
  dsk  <- -k * t * sk - t * x

  list(c(dx, dsx0, dsk))
}

# --- Parameters, initial conditions, and time grid ---------------------------
parms <- c(k = 0.5)
yini  <- c(x = 1, sx0 = 1, sk = 0)
times <- seq(0, 10, by = 0.1)

# --- Numerical integration ---------------------------------------------------
out_lsoda  <- ode(y = yini, times = times, func = model, parms = parms, method = "lsoda")
out_lsodes <- ode(y = yini, times = times, func = model, parms = parms, method = "lsodes")

# --- Convert to tidy data ----------------------------------------------------
df_lsoda  <- as.data.frame(out_lsoda)  %>% mutate(method = "lsoda")
df_lsodes <- as.data.frame(out_lsodes) %>% mutate(method = "lsodes")

df <- bind_rows(df_lsoda, df_lsodes) %>%
  pivot_longer(cols = -c(time, method), names_to = "variable", values_to = "value")

# --- Analytical solution -----------------------------------------------------
k <- parms["k"]

df_exact <- tibble(
  time = times,
  x   = exp(-0.5 * k * times^2),
  sx0 = exp(-0.5 * k * times^2),
  sk  = -0.5 * times^2 * exp(-0.5 * k * times^2)
) %>%
  pivot_longer(cols = -time, names_to = "variable", values_to = "value") %>%
  mutate(method = "analytical")

# --- Combine all results -----------------------------------------------------
df_all <- bind_rows(df, df_exact)

# --- Plot comparison ---------------------------------------------------------
ggplot(df_all, aes(x = time, y = value, color = method, linetype = method)) +
  geom_line(linewidth = 1) +
  facet_wrap(~ variable, scales = "free_y") +
  dMod::theme_dMod() +
  labs(
    title = "Comparison of lsoda vs. lsodes (deSolve)",
    subtitle = "System: dx/dt = -k * t * x  with sensitivities",
    x = "time t",
    y = "value",
    color = "method",
    linetype = "method"
  ) +
  scale_color_manual(values = c(
    "analytical" = "black",
    "lsoda" = "steelblue",
    "lsodes" = "firebrick"
  ))
