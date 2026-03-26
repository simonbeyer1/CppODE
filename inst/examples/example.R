rm(list = ls(all.names = TRUE))

# Create and set working directory
.workingDir <- file.path(
  dirname(rstudioapi::getSourceEditorContext()$path),
  "wd"
)
if (!dir.exists(.workingDir)) dir.create(.workingDir)
setwd(.workingDir)

library(CppODE)
library(readxl)
library(data.table)
library(ggplot2)

data(jakstat)
head(jakstat)

ggplot(jakstat, aes(x = time, y = value)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = value - sigma, ymax = value + sigma),
                width = 0) +        # Error Bars
  facet_wrap(~ name, scales = "free_y") +            # Facets pro Variable
  theme_bw() +                                       # Klassisches Theme
  labs(
    x = "time",
    y = "value",
  ) +
  theme_bw()


rhs <- c(
  npSTATdimer = "p3*pSTATdimer-p4*npSTATdimer",
  nSTAT = "2*p4*npSTATdimer-p5*nSTAT",
  pSTAT = "p1*pEpoR*STAT-2*p2*pSTAT^2",
  pSTATdimer = "p2*pSTAT^2-p3*pSTATdimer",
  STAT = "-p1*pEpoR*STAT + p5*nSTAT"
)

rhs <- gsub("pEpoR", "pEpoR*((1 - exp(-time*lambda1))*exp(-time*lambda2))^3", rhs)

jakstatmodel <- CppODE(rhs, modelname = "jakstatmodel", deriv = TRUE)

observables <- c(
  tSTAT = "s_tSTAT*(STAT + pSTAT + 2*pSTATdimer)",
  tpSTAT = "s_tpSTAT*(pSTAT + 2*pSTATdimer) + off_tpSTAT",
  pEpoR = "s_EpoR * pEpoR*((1 - exp(-time*lambda1))*exp(-time*lambda2))^3"
)

gpred <- funCpp(observables, parameters = c(attr(jakstatmodel, "parameters"), "s_tSTAT", "s_tpSTAT", "off_tpSTAT", "s_EpoR"),
                modelname = "jakstatobsfn", deriv = TRUE, compile = TRUE)


parnames <- setdiff(c(attr(gpred, "variables"), attr(gpred, "parameters")), "time")

trafo <- structure(paste0("exp10(", parnames,")"), names = parnames)

ptrafo <- funCpp(trafo, parameters = parnames,
                 modelname = "jakstatparfn", deriv = TRUE, compile = TRUE,
                 convenient = TRUE)


pini <- structure(rep(-1, length(parnames)), names = parnames)
