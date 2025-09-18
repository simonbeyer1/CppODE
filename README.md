# CppODE

**Automated C++ code generation for ODE integration with Boost.Odeint solvers and first- and second-order sensitivity calculation using CppAD.**

This package provides a convenient R interface to generate, compile, and run C++ code for ODE models with automatic differentiation support.  
Internally, it uses:

- [Boost.Odeint](https://www.boost.org/doc/libs/release/libs/numeric/odeint/) for efficient ODE integration  
- [CppAD](https://coin-or.github.io/CppAD/doc/cppad.htm) for automatic differentiation (first- and second-order)  

---

## Installation

### System requirements

Before installing the R package, you need the development libraries for **Boost** and **CppAD**:

**Ubuntu/Debian**
```bash
sudo apt update
sudo apt install libboost-all-dev libcppad-dev
```

**macOS (Homebrew)**
```bash
brew install boost cppad
```

**Windows**
- Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/)  
- Download and install Boost + CppAD manually, and make sure the headers are visible in your `INCLUDE` path.  

---

### R package installation

Once the system libraries are available, you can install the package from GitHub:

```r
install.packages("devtools")
devtools::install_github("simonbeyer1/CppODE")
```

---

## Example

A minimal example with events, automatic code generation and solving:

```r
library(ggplot2)
library(dplyr)
library(tidyverse)

eqns <- c(
  A = "-k1*A^2 *time",
  B = "k1*A^2 *time - k2*B"
)

events <- data.frame(
  var   = "A",
  time  = "t_e",
  value = 1,
  method= "add"
)

f <- CppODE::CppFun(eqns, events = events, modelname = "Amodel_s",
                    secderiv = TRUE, compile = FALSE)
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

boostCppADtime <- system.time({
  solve(times, params, abstol = 1e-6, reltol = 1e-6)
})

res_CppAD <- solve(times, params, abstol = 1e-6, reltol = 1e-6) %>%
  as.data.frame() %>%
  pivot_longer(cols = -time, names_to = "name", values_to = "value") %>%
  mutate(solver = "boost::odeint + CppAD")

ggplot(res_CppAD, aes(x = time, y = value,
                      color = solver, linetype = solver)) +
  geom_line() +
  facet_wrap(~name, scales = "free_y") +
  xlab("time") + ylab("value") +
  dMod::theme_dMod()
```

---

## Notes

- The package is currently in development and hosted on GitHub.  
- CRAN submission might require vendoring of headers or custom build scripts.  
- Contributions, issues and feedback are welcome!  

---

## License

GPL-3
