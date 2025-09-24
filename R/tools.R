#' Ensure availability of a dedicated Python environment with SymPy
#'
#' This helper function checks whether a Python virtual environment with a
#' given name (default: `"CppODE"`) exists and contains the package
#' **SymPy**. If not, it will automatically create the virtual environment
#' using `reticulate::virtualenv_create()` and install SymPy into it.
#'
#' After ensuring availability, the environment is activated via
#' `reticulate::use_virtualenv()`, so that subsequent calls to
#' `reticulate::import("sympy")` work reliably, independent of the user's
#' global Python installation.
#'
#' ## Typical usage
#' Call this function at the beginning of any code that requires SymPy,
#' e.g. inside your solver generator:
#' \preformatted{
#' ensurePythonEnv("CppODE")
#' sympy  <- reticulate::import("sympy")
#' parser <- reticulate::import("sympy.parsing.sympy_parser")
#' }
#'
#' ## Notes
#' - Requires the R package **reticulate**.
#' - Uses Python 3 if available on the system.
#' - If you prefer Conda environments instead of virtualenvs, you can
#'   adapt this function to use `reticulate::conda_create()` and
#'   `reticulate::use_condaenv()`.
#'
#' @param envname Character scalar. Name of the Python virtual environment
#'   to check or create. Default is `"CppODE"`.
#'
#' @return Invisibly returns the name of the environment that was ensured
#'   and activated.
#' @keywords internal
ensurePythonEnv <- function(envname = "CppODE") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The package 'reticulate' is required but not installed.")
  }

  # check if virtualenv exists
  venvs <- reticulate::virtualenv_list()
  if (!(envname %in% venvs)) {
    message("Creating Python virtualenv '", envname, "' with SymPy ...")
    reticulate::virtualenv_create(envname = envname, python = "python3")
    reticulate::virtualenv_install(envname, packages = c("sympy"))
  } else {
    # ensure sympy is available
    mods <- reticulate::py_list_packages(envname = envname)
    if (!"sympy" %in% mods$package) {
      message("Installing 'sympy' into '", envname, "' ...")
      reticulate::virtualenv_install(envname, packages = c("sympy"))
    }
  }

  # activate the environment for reticulate
  reticulate::use_virtualenv(envname, required = TRUE)

  invisible(envname)
}
