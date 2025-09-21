# CppODE

**Automated C++ code generation for ODE integration with Boost.Odeint solvers and sensitivity calculation using FADBAD++.**

This package provides a convenient R interface to generate, compile, and run C++ code for ODE models with optional automatic differentiation support.  
Internally, it uses:

- [Boost.Odeint](https://www.boost.org/doc/libs/release/libs/numeric/odeint/) for stiff ODE integration with the Rosenbrock 3(4) method (error-controlled, dense output).
- [FADBAD++](https://github.com/stefano-lake/fadbad) for forward- and backward-mode automatic differentiation.  

---

## Installation

Install the R package directly from GitHub:

```r
install.packages("devtools") # if needed
devtools::install_github("simonbeyer1/CppODE")
```

---

## Example

```r
library(ggplot2)
library(dplyr)
library(tidyverse)

eqns <- c(
  A = "-k1*A^2 * time",
  B = "k1*A^2 * time - k2*B"
)

events <- data.frame(
  var   = "A",
  time  = "t_e",
  value = 1,
  method= "add"
)

f <- CppODE::CppFun(eqns, events = events, modelname = "Amodel_s",
                    deriv = TRUE, compile = FALSE)
CppODE::compileAndLoad(f, verbose = FALSE)

solve <- function(times, params,
                  abstol = 1e-8, reltol = 1e-6,
                  maxtrysteps = 500, maxsteps = 1e6) {
  paramnames <- c(attr(f,"variables"), attr(f,"parameters"))
  params <- params[paramnames]
  .Call("solve_Amodel_s",
        as.numeric(times),
        as.numeric(params),
        as.numeric(abstol),
        as.numeric(reltol),
        as.integer(maxtrysteps),
        as.integer(maxsteps))
}

params <- c(A = 1, k1 = 0.1, t_e = 3)
times  <- seq(0, 10, length.out = 300)

res_FADBAD <- solve(times, params, abstol = 1e-6, reltol = 1e-6) %>%
  as.data.frame() %>%
  tidyr::pivot_longer(cols = -time, names_to = "name", values_to = "value") %>%
  mutate(solver = "boost::odeint + FADBAD++")

ggplot(res_FADBAD, aes(x = time, y = value,
                       color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") + ylab("value") +
  dMod::theme_bw()
```

---

## Notes
- Future work: extend support to additional solvers, including adaptive schemes with automatic stiffness detection.  
- Contributions, issues and feedback are welcome!  

---

## License

This package is licensed under the MIT License – see the [LICENSE.md](LICENSE.md) file for details.

