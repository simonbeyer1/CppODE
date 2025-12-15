#' Ensure availability of a dedicated Python environment with required packages
#'
#' This helper function checks whether a Python virtual environment with a
#' given name (default: `"CppODE"`) exists and contains the required
#' Python packages (default: c("numpy", "sympy")). If not, it will create
#' the virtual environment and install the missing packages.
#'
#' After ensuring availability, the environment is activated via
#' `reticulate::use_virtualenv()`.
#'
#' @param envname Character scalar. Name of the Python virtual environment
#'   to check or create. Default is `"CppODE"`.
#' @param packages Character vector. Python packages to ensure are installed.
#'   Default: c("numpy", "sympy").
#' @param verbose Logical. If TRUE, prints progress messages.
#'
#' @return Invisibly returns the name of the environment that was ensured.
#' @export
ensurePythonEnv <- function(envname = "CppODE",
                            packages = c("numpy", "sympy"),
                            verbose = FALSE) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The package 'reticulate' is required but not installed.")
  }

  # Locate python3
  py <- Sys.which("python3")
  if (py == "") {
    stop(
      "No 'python3' found on your system. Please install Python 3.\n",
      "On Debian/Ubuntu: sudo apt-get install python3 python3-venv python3-pip\n",
      "On Windows: install from https://www.python.org/downloads/ and ensure PATH\n",
      "On macOS: brew install python3\n",
      "Or use reticulate::install_python()"
    )
  }

  # Check if virtualenv exists
  venvs <- reticulate::virtualenv_list()
  if (!(envname %in% venvs)) {
    if (verbose) message("Creating Python virtualenv '", envname, "' ...")
    reticulate::virtualenv_create(envname = envname, python = py)

    if (verbose) message("Installing required packages into '", envname, "': ",
                         paste(packages, collapse = ", "))
    reticulate::virtualenv_install(envname, packages = packages)
  } else {
    # Ensure all requested packages are installed
    mods <- reticulate::py_list_packages(envname = envname)
    missing <- setdiff(packages, mods$package)

    if (length(missing) > 0) {
      if (verbose) {
        message("Installing missing packages into '", envname, "': ",
                paste(missing, collapse = ", "))
      }
      reticulate::virtualenv_install(envname, packages = missing)
    }
  }

  # Activate environment
  reticulate::use_virtualenv(envname)

  invisible(envname)
}
