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
#' ## Requirements
#' - A working `python3` executable on your system.
#'   On Linux, install via your package manager, e.g.:
#'   \preformatted{
#'   sudo apt-get install python3 python3-venv python3-pip
#'   }
#'
#' ## Usage
#' \preformatted{
#' ensurePythonEnv("CppODE")
#' sympy  <- reticulate::import("sympy")
#' parser <- reticulate::import("sympy.parsing.sympy_parser")
#' }
#'
#' @param envname Character scalar. Name of the Python virtual environment
#'   to check or create. Default is `"CppODE"`.
#'
#' @return Invisibly returns the name of the environment that was ensured
#'   and activated.
#' @keywords internal
ensurePythonEnv <- function(envname = "CppODE", verbose = F) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The package 'reticulate' is required but not installed.")
  }

  # Try to locate python3
  py <- Sys.which("python3")
  if (py == "") {
    stop(
      "No 'python3' found on your system. Please install Python 3.\n",
      "On Debian/Ubuntu: sudo apt-get install python3 python3-venv python3-pip\n",
      "On Windows: please install Python from https://www.python.org/downloads/ and ensure 'python3' is on PATH\n",
      "On macOS: you can install via Homebrew: brew install python3\n",
      "Or use reticulate::install_python() to get a managed Python."
    )
  }

  # Check if virtualenv already exists
  venvs <- reticulate::virtualenv_list()
  if (!(envname %in% venvs)) {
    if (verbose) message("Creating Python virtualenv '", envname, "' with SymPy ...")
    reticulate::virtualenv_create(envname = envname, python = py)
    reticulate::virtualenv_install(envname, packages = c("sympy"))
  } else {
    # Ensure sympy is available
    mods <- reticulate::py_list_packages(envname = envname)
    if (!"sympy" %in% mods$package) {
      if (verbose) message("Installing 'sympy' into '", envname, "' ...")
      reticulate::virtualenv_install(envname, packages = c("sympy"))
    }
  }

  # Activate environment
  reticulate::use_virtualenv(envname)

  invisible(envname)
}
