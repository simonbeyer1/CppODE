#!/usr/bin/env Rscript
# Local R CMD check that mimics the GitHub Actions r-lib/actions/check-r-package
# step (build_args + --as-cran). Aborts on warnings, not just on errors.
#
# Requirements:
#   * pandoc on PATH or reachable via RSTUDIO_PANDOC (for CppODE_LotkaVolterra.Rmd)
#   * vignettes/Methods.pdf exists (otherwise run dev/render-methods.R first)
#   * ghostscript optional — without it --compact-vignettes=gs+qpdf simply
#     skips compaction instead of failing.

# On Windows without pandoc on PATH: try the usual RStudio bundle locations.
if (!nzchar(Sys.which("pandoc")) && !nzchar(Sys.getenv("RSTUDIO_PANDOC"))) {
  rstudio_pandoc_candidates <- c(
    "C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools",
    "C:/Program Files/RStudio/bin/pandoc",
    "C:/Program Files/Quarto/bin/tools"
  )
  hit <- rstudio_pandoc_candidates[file.exists(file.path(rstudio_pandoc_candidates, "pandoc.exe"))][1]
  if (!is.na(hit)) {
    Sys.setenv(RSTUDIO_PANDOC = hit)
    message("Setting RSTUDIO_PANDOC = ", hit)
  }
}

devtools::check(
  error_on   = "warning",
  args       = c("--no-manual", "--as-cran"),
  build_args = c("--no-manual", "--compact-vignettes=gs+qpdf")
)
