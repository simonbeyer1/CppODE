# derivSymb.py
# Author: Simon Beyer
import sympy as sp

def jac_hess_symb(exprs, variables, deriv2=False):
    """
    Compute the symbolic Jacobian matrix and optionally the Hessian tensor
    for a system of algebraic expressions using SymPy.

    This function is designed for interoperability with R via reticulate.
    It accepts either a dict of named expressions or a list of expressions
    (strings), computes first and, optionally, second derivatives with
    respect to a list of variables, and returns the results as nested lists
    of strings.

    Parameters
    ----------
    exprs : dict or list of str
        The expressions to differentiate.
        - If a dict, keys correspond to the function names (e.g. inner parameters)
          and values are algebraic expressions as strings.
        - If a list, entries are expressions as strings; names will be generated
          automatically ("f1", "f2", ...).
        Example: {"y": "a*x**2 + b"}

    variables : list of str
        The variables with respect to which the derivatives are computed
        (e.g. outer parameters).

    deriv2 : bool, optional (default: False)
        If True, compute also the Hessian tensor (second derivatives).
        The Hessian is returned as a 3-D list with dimensions
        [n_functions][n_variables][n_variables], where
        H[i][j][k] = ∂²f_i / (∂var_j ∂var_k).

    Returns
    -------
    dict
        A dictionary with the following entries:
        - "jacobian" : list of str
              Flattened list of first derivatives in row-major order,
              i.e. for each function f_i and variable v_j.
        - "hessian" : list or None
              3-D list of strings with second derivatives
              [n_functions][n_variables][n_variables],
              or None if `deriv2` is False.
        - "names" : list of str
              Names of the functions (inner parameters).

    Examples
    --------
    >>> jac_hess_symb({"y": "a*x**2 + b"}, ["x", "a", "b"], deriv2=False)
    {'jacobian': ['2*a*x', 'x**2', '1'], 'hessian': None, 'names': ['y']}

    >>> jac_hess_symb({"y": "a*x**2 + b"}, ["x", "a", "b"], deriv2=True)["hessian"][0]
    [['2*a', '2*x', '0'],
     ['2*x', '0', '0'],
     ['0', '0', '0']]

    Notes
    -----
    - SymPy supports differentiation of non-analytic operations such as
      `abs`, `min`, `max`, and `sign`; in these cases, the result may
      contain Heaviside or sign functions.
    - Returned expressions are simplified using `sympy.simplify()`.
    - Designed to be called from R via reticulate and combined with
      higher-level wrapper functions such as `derivSymb()`.

    Author
    ------
    Simon Beyer, 2025
    """

    # convert variable names to sympy Symbols
    vars_syms = sp.symbols(variables)

    # parse expressions
    if isinstance(exprs, dict):
        fnames = list(exprs.keys())
        exprs_syms = [sp.sympify(exprs[k]) for k in fnames]
    elif isinstance(exprs, (list, tuple)):
        fnames = [f"f{i+1}" for i in range(len(exprs))]
        exprs_syms = [sp.sympify(e) for e in exprs]
    else:
        raise TypeError("exprs must be a dict or list of strings")

    # Jacobian
    J = sp.Matrix(exprs_syms).jacobian(vars_syms)
    jac_list = [str(sp.simplify(J[i, j]))
                for i in range(len(exprs_syms))
                for j in range(len(vars_syms))]

    # Hessian (3D list)
    H = None
    if deriv2:
        H = []
        for f_expr in exprs_syms:
            Hi = []
            for v1 in vars_syms:
                row = []
                for v2 in vars_syms:
                    d2 = sp.simplify(sp.diff(f_expr, v1, v2))
                    row.append(str(d2))
                Hi.append(row)
            H.append(Hi)

    return {"jacobian": jac_list, "hessian": H, "names": fnames}
