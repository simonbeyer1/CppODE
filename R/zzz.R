#' Package-internal CVODE backend configuration
#'
#' Populated at load time from `inst/cvodeConfig.dcf` (written by
#' `./configure` or `./configure.win` at install time).  Defaults mark
#' the backend as disabled so the package loads cleanly even if the DCF
#' is missing; in that case `CVODE()` errors with a clear hint.
#'
#' @keywords internal
#' @noRd
cvodeConfig <- new.env(parent = emptyenv())

#' Package initialization
#'
#' @keywords internal
#' @importFrom reticulate py_require
#' @noRd
.onLoad <- function(libname, pkgname) {
  reticulate::py_require("sympy")

  cvodeConfig$available     <- FALSE
  cvodeConfig$cflags        <- ""
  cvodeConfig$libs          <- ""
  cvodeConfig$klu_available <- FALSE
  cvodeConfig$klu_cflags    <- ""
  cvodeConfig$klu_libs      <- ""

  file <- system.file("cvodeConfig.dcf", package = pkgname)
  if (nzchar(file) && file.exists(file)) {
    d <- tryCatch(read.dcf(file), error = function(e) NULL)
    if (!is.null(d) && nrow(d) >= 1L) {
      get_str <- function(k) {
        if (!(k %in% colnames(d))) return("")
        v <- d[1L, k]
        if (is.na(v)) "" else as.character(v)
      }
      cvodeConfig$available     <- identical(get_str("available"), "TRUE")
      cvodeConfig$cflags        <- get_str("cflags")
      cvodeConfig$libs          <- get_str("libs")
      cvodeConfig$klu_available <- identical(get_str("klu_available"), "TRUE")
      cvodeConfig$klu_cflags    <- get_str("klu_cflags")
      cvodeConfig$klu_libs      <- get_str("klu_libs")
    }
  }
}

#' Lazy import of internal Python modules
#'
#' @keywords internal
#' @noRd
.cppode_py_cache <- new.env(parent = emptyenv())

#' @keywords internal
#' @importFrom reticulate import_from_path
#' @noRd
get_codegenCppODE_py <- function() {
  if (!exists("codegenCppODE", envir = .cppode_py_cache, inherits = FALSE)) {
    .cppode_py_cache$codegenCppODE <-
      reticulate::import_from_path(
        "codegenCppODE",
        path = system.file("python", package = "CppODE"),
        delay_load = TRUE
      )
  }
  .cppode_py_cache$codegenCppODE
}

#' @keywords internal
#' @importFrom reticulate import_from_path
#' @noRd
get_codegenfunCpp_py <- function() {
  if (!exists("codegenfunCpp", envir = .cppode_py_cache, inherits = FALSE)) {
    .cppode_py_cache$codegenfunCpp <-
      reticulate::import_from_path(
        "codegenfunCpp",
        path = system.file("python", package = "CppODE"),
        delay_load = TRUE
      )
  }
  .cppode_py_cache$codegenfunCpp
}

#' @keywords internal
#' @importFrom reticulate import_from_path
#' @noRd
get_codegenCVODE_py <- function() {
  if (!exists("codegenCVODE", envir = .cppode_py_cache, inherits = FALSE)) {
    .cppode_py_cache$codegenCVODE <-
      reticulate::import_from_path(
        "codegenCVODE",
        path = system.file("python", package = "CppODE"),
        delay_load = TRUE
      )
  }
  .cppode_py_cache$codegenCVODE
}

#' @keywords internal
#' @importFrom reticulate import_from_path
#' @noRd
get_derivSymb_py <- function() {
  if (!exists("derivSymb", envir = .cppode_py_cache, inherits = FALSE)) {
    .cppode_py_cache$derivSymb <-
      reticulate::import_from_path(
        "derivSymb",
        path = system.file("python", package = "CppODE"),
        delay_load = TRUE
      )
  }
  .cppode_py_cache$derivSymb
}
