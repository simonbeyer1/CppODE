# =============================================================================
# derivSymb.py
# Symbolic differentiation utilities for CppODE
#
# Provides safe symbolic differentiation (Jacobian + Hessian)
# using SymPy, with optional post-hoc enforcement of real-valued
# simplifications. Non-analytic expressions like abs(), max(), or
# sign() are supported without recursion errors.
# =============================================================================

import sympy as sp


# -----------------------------------------------------------------------------
# Helper: Enforce real-valued simplification
# -----------------------------------------------------------------------------
def _make_real_and_simplify(expr):
    """
    Aggressively strip complex parts from an expression by
    replacing:
      re(Z) -> Z
      im(Z) -> 0
      Abs(re(Z)) -> Abs(Z)
      Derivative(re(Z), V) -> Derivative(Z, V)
      Derivative(im(Z), V) -> 0

    …und zwar iterativ, bis keine Änderung mehr passiert.
    Danach wird simplify() aufgerufen.
    """
    Z = sp.Wild('Z')
    V = sp.Wild('V')

    try:
        old = None
        while expr != old:
            old = expr
            # direkte re/im
            expr = expr.replace(sp.re(Z), Z)
            expr = expr.replace(sp.im(Z), 0)

            # Abs(re(.)) -> Abs(.)
            expr = expr.replace(sp.Abs(sp.re(Z)), sp.Abs(Z))

            # Ableitungen von re/im
            expr = expr.replace(sp.Derivative(sp.re(Z), V), sp.Derivative(Z, V))
            expr = expr.replace(sp.Derivative(sp.im(Z), V), 0)

            # manchmal entstehen danach noch triviale Derivates: d/dx(x) -> 1 etc.
            try:
                expr = sp.simplify(expr.doit())
            except Exception:
                # wenn .doit() mal nicht evaluieren kann – okay, wir lassen es
                pass

        # abschließende Vereinfachung
        expr = sp.simplify(expr)
    except Exception:
        # konservativer Fallback
        pass

    return expr

# -----------------------------------------------------------------------------
# Helper: Parse expressions and infer variables
# -----------------------------------------------------------------------------
def _prepare_expressions(exprs, variables=None):
    """
    Parse expressions from strings (or dict) and infer variable symbols.

    Parameters
    ----------
    exprs : dict or list of str
        Either a dict {name: expression_string} or a list of expression strings.
    variables : list of str or None, optional
        If provided, use these variable names. Otherwise, infer them
        automatically from all expressions' free symbols.

    Returns
    -------
    tuple
        (fnames, exprs_syms, vars_syms)
        - fnames: list of expression names
        - exprs_syms: list of sympy expressions
        - vars_syms: list of sympy Symbols (sorted by name)
    """
    # Handle dict or list input
    if isinstance(exprs, dict):
        fnames = list(exprs.keys())
        expr_strs = [exprs[k] for k in fnames]
    elif isinstance(exprs, (list, tuple)):
        fnames = [f"f{i+1}" for i in range(len(exprs))]
        expr_strs = list(exprs)
    else:
        raise TypeError("exprs must be a dict or list of expression strings")

    # Parse to SymPy
    exprs_syms = [sp.sympify(e) for e in expr_strs]

    # Determine variables
    if variables is None:
        vars_syms = sorted(
            set().union(*(e.free_symbols for e in exprs_syms)),
            key=lambda s: s.name
        )
    else:
        vars_syms = [sp.Symbol(v) for v in variables]

    return fnames, exprs_syms, vars_syms


# -----------------------------------------------------------------------------
# Main function: Compute Jacobian and optional Hessians
# -----------------------------------------------------------------------------
def jac_hess_symb(exprs, variables=None, deriv2=False, real=False):
    """
    Compute symbolic Jacobian and optionally Hessian for one or more expressions.

    Differentiation is always performed without assumptions for stability.
    If `real=True`, all resulting expressions are post-processed by
    `_make_real_and_simplify()` to remove imaginary parts and simplify safely.

    Parameters
    ----------
    exprs : dict or list of str
        Dictionary or list of symbolic expressions.
        Example: {"f1": "a*x**2 + b*y**2", "f2": "x*y + exp(2*c) + abs(max(x,y))"}
    variables : list of str or None, optional
        Variable names to differentiate with respect to. If None, automatically
        inferred using SymPy’s free symbol detection.
    deriv2 : bool, optional
        If True, compute second derivatives (Hessians) for each expression. Default: False.
    real : bool, optional
        If True, enforce real-valued simplification post-hoc. Default: False.

    Returns
    -------
    dict
        A dictionary with:
          - "jacobian": list[list[str]] — Jacobian entries as strings
          - "hessian": list[list[list[str]]] or None — Hessians (if deriv2=True)
          - "names": list[str] — expression names
          - "vars": list[str] — variable names (in order)

    Example
    -------
    >>> exprs = {
    ...   "f1": "a*x**2 + b*y**2",
    ...   "f2": "x*y + exp(2*c) + abs(max(x,y))"
    ... }
    >>> jac_hess_symb(exprs, real=True)
    {
        "jacobian": [
            ["x**2", "2*a*x", "y**2", "2*b*y", "0"],
            ["0", "y + Heaviside(x - y)", "0", "x - Heaviside(x - y)", "2*exp(2*c)"]
        ],
        "hessian": None,
        "names": ["f1", "f2"],
        "vars": ["a", "b", "c", "x", "y"]
    }
    """
    fnames, exprs_syms, vars_syms = _prepare_expressions(exprs, variables)

    # --- Compute Jacobian ---
    J = sp.Matrix(exprs_syms).jacobian(vars_syms)
    jac = []
    for i in range(len(exprs_syms)):
        row = []
        for j in range(len(vars_syms)):
            d = J[i, j]
            d = _make_real_and_simplify(d) if real else sp.simplify(d)
            row.append(str(d))
        jac.append(row)

    # --- Compute Hessian if requested ---
    H = None
    if deriv2:
        H = []
        for f_expr in exprs_syms:
            Hi = []
            for v1 in vars_syms:
                row = []
                for v2 in vars_syms:
                    d2 = sp.diff(f_expr, v1, v2)
                    d2 = _make_real_and_simplify(d2) if real else sp.simplify(d2)
                    row.append(str(d2))
                Hi.append(row)
            H.append(Hi)

    return {
        "jacobian": jac,
        "hessian": H,
        "names": fnames,
        "vars": [v.name for v in vars_syms]
    }
