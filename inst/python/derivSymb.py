"""
Symbolic differentiation utilities for CppODE
=============================================
Provides safe symbolic differentiation (Jacobian + Hessian)
using SymPy, with optional post-hoc enforcement of real-valued
simplifications. Non-analytic expressions like abs(), max(), or
sign() are supported without recursion errors.

Author: Simon Beyer
Updated: 2025-11-04
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor
)


# -----------------------------------------------------------------------------
# Safe parsing configuration
# -----------------------------------------------------------------------------
def _get_safe_parse_dict():
    """
    Create safe local_dict for parsing that avoids SymPy singleton conflicts.
    """
    safe_local_dict = {
        # Override problematic SymPy singletons - treat as regular symbols
        'S': sp.Symbol('S'),
        'I': sp.Symbol('I'),
        'N': sp.Symbol('N'),
        'O': sp.Symbol('O'),
        'Q': sp.Symbol('Q'),
        'C': sp.Symbol('C'),
        
        # Exponential and logarithmic functions
        'exp': sp.exp,
        'exp10': lambda x: sp.Pow(10, x),
        'exp2': lambda x: sp.Pow(2, x),
        'log': sp.log,
        'ln': sp.log,
        'log10': lambda x: sp.log(x, 10),
        'log2': lambda x: sp.log(x, 2),
        
        # Trigonometric functions
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'cot': sp.cot,
        'sec': sp.sec,
        'csc': sp.csc,
        
        # Inverse trigonometric functions
        'asin': sp.asin,
        'acos': sp.acos,
        'atan': sp.atan,
        'acot': sp.acot,
        'asec': sp.asec,
        'acsc': sp.acsc,
        'atan2': sp.atan2,
        
        # Hyperbolic functions
        'sinh': sp.sinh,
        'cosh': sp.cosh,
        'tanh': sp.tanh,
        'coth': sp.coth,
        'sech': sp.sech,
        'csch': sp.csch,
        
        # Inverse hyperbolic functions
        'asinh': sp.asinh,
        'acosh': sp.acosh,
        'atanh': sp.atanh,
        'acoth': sp.acoth,
        'asech': sp.asech,
        'acsch': sp.acsch,
        
        # Power and root functions
        'sqrt': sp.sqrt,
        'cbrt': sp.cbrt,
        'root': sp.root,
        'pow': sp.Pow,
        
        # Absolute value and sign
        'abs': sp.Abs,
        'sign': sp.sign,
        
        # Rounding functions
        'floor': sp.floor,
        'ceiling': sp.ceiling,
        
        # Min/Max
        'min': sp.Min,
        'max': sp.Max,
        
        # Factorial and gamma functions
        'factorial': sp.factorial,
        'gamma': sp.gamma,
        'loggamma': sp.loggamma,
        'digamma': sp.digamma,
        
        # Error functions
        'erf': sp.erf,
        'erfc': sp.erfc,
        
        # Bessel functions
        'besselj': sp.besselj,
        'bessely': sp.bessely,
        'besseli': sp.besseli,
        'besselk': sp.besselk,
        
        # Special functions
        'Heaviside': sp.Heaviside,
        'DiracDelta': sp.DiracDelta,
        
        # Piecewise
        'Piecewise': sp.Piecewise,
        
        # Constants
        'pi': sp.pi,
        'E': sp.E,
        'oo': sp.oo,
    }
    
    return safe_local_dict


def _safe_sympify(expr_str, local_symbols=None):
    """
    Safely parse a string expression to SymPy, avoiding singleton conflicts.
    
    Parameters
    ----------
    expr_str : str
        Expression string to parse
    local_symbols : dict, optional
        Additional local symbols (variables/parameters)
    
    Returns
    -------
    sp.Expr
        Parsed SymPy expression
    """
    expr_str = str(expr_str).strip()
    if expr_str == "0":
        return sp.Integer(0)
    
    safe_local = _get_safe_parse_dict()
    
    # Merge with user-provided symbols
    if local_symbols:
        safe_local = {**safe_local, **local_symbols}
    
    transformations = standard_transformations + (
        convert_xor,
    )
    
    return parse_expr(
        expr_str,
        local_dict=safe_local,
        transformations=transformations,
        evaluate=True,
    )


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

    # Parse to SymPy using safe parsing
    exprs_syms = [_safe_sympify(e) for e in expr_strs]

    # Determine variables
    if variables is None:
        vars_syms = sorted(
            set().union(*(e.free_symbols for e in exprs_syms)),
            key=lambda s: s.name
        )
    else:
        # Create symbols for provided variable names
        vars_syms = [sp.Symbol(v) for v in variables]

    return fnames, exprs_syms, vars_syms


# -----------------------------------------------------------------------------
# Main function: Compute Jacobian and optional Hessians
# -----------------------------------------------------------------------------
def jac_hess_symb(exprs, variables=None, fixed=None, deriv2=False, real=False):
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
        inferred using SymPy's free symbol detection.
    fixed : list of str or None
        Variable names to treat as fixed (excluded from differentiation).
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

    # --- remove fixed symbols if any ---
    if fixed is not None:
        fixed_set = set(fixed)
        vars_syms = [v for v in vars_syms if v.name not in fixed_set]

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
