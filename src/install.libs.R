# =========================================================================
#  CppODE — custom library installer
#
#  Installs the pre-built KLU static archive (libcppode_ss.a) into the
#  package.  No shared object (.so/.dll) is installed because CppODE
#  generates and compiles model code at runtime.
#
#  R provides: R_PACKAGE_DIR, R_ARCH
# =========================================================================

dest <- file.path(R_PACKAGE_DIR, paste0("lib", R_ARCH))
dir.create(dest, recursive = TRUE, showWarnings = FALSE)

sslib <- "libcppode_ss.a"
if (file.exists(sslib)) {
  file.copy(sslib, dest, overwrite = TRUE)
  cat(sprintf("  KLU static library installed to %s\n", dest))
} else {
  warning("libcppode_ss.a not found — sparse LU will be unavailable")
}
