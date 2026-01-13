#' Package initialization
#'
#' @keywords internal
#' @importFrom reticulate py_require
#' @noRd
.onLoad <- function(libname, pkgname) {
  reticulate::py_require("sympy")
}

#' Lazy import of internal Python modules
#'
#' @keywords internal
#' @noRd
.cppode_py_cache <- new.env(parent = emptyenv())
.codegenCppODE_py <- NULL
.codegenfunCpp_py <- NULL
.derivSymb_py <- NULL

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
