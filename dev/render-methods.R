#!/usr/bin/env Rscript
# Re-render dev/methods/Methods.Rmd -> vignettes/Methods.pdf
#
# Methods.pdf is shipped as a *static* (pre-rendered) vignette via
# R.rsp::asis, because building it requires lualatex + CMU/New Computer
# Modern fonts that are not available on standard CI runners. The source
# Bundle (Rmd + bibliography + CSL) lives under dev/methods/ — this whole
# directory is .Rbuildignore'd, so the package tarball contains only the
# rendered PDF and the .asis stub. CRAN/CI never sees a LaTeX dependency.
#
# Workflow: edit dev/methods/Methods.Rmd, run this script locally, commit
# the regenerated vignettes/Methods.pdf alongside the source change.
#
# Requirements (only on the maintainer's machine, not on CI):
#   lualatex, fontspec, unicode-math, CMU Serif, NewCMMath.

src <- "dev/methods/Methods.Rmd"
out <- "vignettes/Methods.pdf"

if (!file.exists(src))
  stop("Cannot find ", src, " — run this from the package root.")

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

# On Windows without lualatex on PATH: try the standard TeX Live install paths.
if (!nzchar(Sys.which("lualatex"))) {
  texlive_candidates <- Sys.glob(c(
    "C:/texlive/*/bin/windows",
    "C:/texlive/*/bin/win32"
  ))
  hit <- texlive_candidates[file.exists(file.path(texlive_candidates, "lualatex.exe"))][1]
  if (!is.na(hit)) {
    Sys.setenv(PATH = paste(hit, Sys.getenv("PATH"), sep = .Platform$path.sep))
    message("Prepended TeX Live to PATH: ", hit)
  }
}

rmarkdown::render(
  input       = src,
  output_file = "Methods.pdf",
  output_dir  = normalizePath("vignettes", mustWork = TRUE),
  quiet       = FALSE
)

if (!file.exists(out))
  stop("Render finished but ", out, " was not produced.")

message("Wrote ", out, " (", format(file.size(out) / 1024, digits = 1), " KiB)")
