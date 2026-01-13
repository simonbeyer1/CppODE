"""
Symbolic differentiation utilities for CppODE
=========================================================
Provides safe symbolic differentiation (Jacobian + Hessian)
using SymPy, with optional post-hoc enforcement of real-valued
simplifications. Non-analytic expressions like abs(), max(), or
sign() are supported.

Author: Simon Beyer
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor
)
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


# -----------------------------------------------------------------------------
# Safe parsing configuration (cached)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_safe_parse_dict_cached():
    """
    Create safe local_dict for parsing that avoids SymPy singleton conflicts.
    Cached to avoid repeated dict creation.
    """
    return {
        # Override problematic SymPy singletons
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
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'cot': sp.cot, 'sec': sp.sec, 'csc': sp.csc,
        
        # Inverse trigonometric functions
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'acot': sp.acot, 'asec': sp.asec, 'acsc': sp.acsc,
        'atan2': sp.atan2,
        
        # Hyperbolic functions
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'coth': sp.coth, 'sech': sp.sech, 'csch': sp.csch,
        
        # Inverse hyperbolic functions
        'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
        'acoth': sp.acoth, 'asech': sp.asech, 'acsch': sp.acsch,
        
        # Power and root functions
        'sqrt': sp.sqrt, 'cbrt': sp.cbrt, 'root': sp.root, 'pow': sp.Pow,
        
        # Absolute value and sign
        'abs': sp.Abs, 'sign': sp.sign,
        
        # Rounding functions
        'floor': sp.floor, 'ceiling': sp.ceiling,
        
        # Min/Max
        'min': sp.Min, 'max': sp.Max,
        
        # Factorial and gamma functions
        'factorial': sp.factorial, 'gamma': sp.gamma,
        'loggamma': sp.loggamma, 'digamma': sp.digamma,
        
        # Error functions
        'erf': sp.erf, 'erfc': sp.erfc,
        
        # Bessel functions
        'besselj': sp.besselj, 'bessely': sp.bessely,
        'besseli': sp.besseli, 'besselk': sp.besselk,
        
        # Special functions
        'Heaviside': sp.Heaviside, 'DiracDelta': sp.DiracDelta,
        
        # Piecewise
        'Piecewise': sp.Piecewise,
        
        # Constants
        'pi': sp.pi, 'E': sp.E, 'oo': sp.oo,
    }


# Precompiled transformations (module-level constant)
_TRANSFORMATIONS = standard_transformations + (convert_xor,)


def _safe_sympify(expr_str, local_symbols=None):
    """
    Safely parse a string expression to SymPy, avoiding singleton conflicts.
    """
    expr_str = str(expr_str).strip()
    if expr_str == "0":
        return sp.Integer(0)
    
    # Start with cached base dict
    safe_local = dict(_get_safe_parse_dict_cached())
    
    # Merge with user-provided symbols
    if local_symbols:
        safe_local.update(local_symbols)
    
    return parse_expr(
        expr_str,
        local_dict=safe_local,
        transformations=_TRANSFORMATIONS,
        evaluate=True,
    )


# -----------------------------------------------------------------------------
# Helper: Enforce real-valued simplification (optimized)
# -----------------------------------------------------------------------------

# Precompute wildcards once at module level
_Z = sp.Wild('Z')
_V = sp.Wild('V')


def _apply_simplify(expr, simplify_func="powsimp"):
    """
    Apply the specified simplification function.
    """
    if simplify_func == "powsimp":
        return sp.powsimp(expr)
    elif simplify_func == "cancel":
        return sp.cancel(expr)
    elif simplify_func == "ratsimp":
        return sp.ratsimp(expr)
    elif simplify_func == "simplify":
        return sp.simplify(expr)
    else:
        return sp.powsimp(expr)  # default fallback


def _make_real_and_simplify(expr, max_iterations=5, simplify_func="powsimp"):
    """
    Aggressively strip complex parts from an expression.
    
    Optimized version with:
    - Precomputed wildcards
    - Early exit when no change
    - Limited iterations to prevent infinite loops
    - Batched replacements
    """
    if expr == 0 or expr.is_number:
        return expr
    
    try:
        for _ in range(max_iterations):
            old = expr
            
            # Batch all replacements
            expr = expr.replace(sp.re(_Z), _Z)
            expr = expr.replace(sp.im(_Z), 0)
            expr = expr.replace(sp.Abs(sp.re(_Z)), sp.Abs(_Z))
            expr = expr.replace(sp.Derivative(sp.re(_Z), _V), sp.Derivative(_Z, _V))
            expr = expr.replace(sp.Derivative(sp.im(_Z), _V), 0)
            
            # Early exit if no change
            if expr == old:
                break
            
            # Try to evaluate derivatives
            try:
                expr = expr.doit()
            except Exception:
                pass
        
        # Final simplification
        return _apply_simplify(expr, simplify_func)
    
    except Exception:
        # Conservative fallback
        return expr


def _simplify_derivative(deriv_expr, use_real=False, simplify_func="powsimp"):
    """
    Simplify a single derivative expression.
    Factored out for potential parallelization.
    """
    if deriv_expr == 0:
        return "0"
    
    if use_real:
        result = _make_real_and_simplify(deriv_expr, simplify_func=simplify_func)
    else:
        result = _apply_simplify(deriv_expr, simplify_func)
    
    return str(result)


# -----------------------------------------------------------------------------
# Helper: Parse expressions and infer variables (optimized)
# -----------------------------------------------------------------------------

def _prepare_expressions(exprs, variables=None):
    """
    Parse expressions from strings (or dict) and infer variable symbols.
    
    Optimized: Creates symbols dict once and reuses it.
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

    # Pre-create symbols if variables are provided
    if variables is not None:
        local_symbols = {v: sp.Symbol(v, real=True) for v in variables}
    else:
        local_symbols = None

    # Parse all expressions with shared symbol dict
    exprs_syms = [_safe_sympify(e, local_symbols) for e in expr_strs]

    # Determine variables
    if variables is None:
        vars_syms = sorted(
            set().union(*(e.free_symbols for e in exprs_syms)),
            key=lambda s: s.name
        )
    else:
        vars_syms = [local_symbols[v] for v in variables]

    return fnames, exprs_syms, vars_syms


# -----------------------------------------------------------------------------
# Optimized Jacobian computation
# -----------------------------------------------------------------------------

def _compute_jacobian_row(expr, vars_syms, use_real, simplify_func):
    """Compute one row of the Jacobian (for parallelization)."""
    row = []
    for v in vars_syms:
        d = sp.diff(expr, v)
        row.append(_simplify_derivative(d, use_real, simplify_func))
    return row


def _compute_hessian_matrix(expr, vars_syms, use_real, simplify_func):
    """Compute Hessian matrix for one expression."""
    n = len(vars_syms)
    H = []
    
    # Compute first derivatives once
    first_derivs = [sp.diff(expr, v) for v in vars_syms]
    
    for i, v1 in enumerate(vars_syms):
        row = []
        for j, v2 in enumerate(vars_syms):
            # Use symmetry: H[i,j] = H[j,i]
            if j < i:
                row.append(H[j][i])  # Reference already computed
            else:
                d2 = sp.diff(first_derivs[i], v2)
                row.append(_simplify_derivative(d2, use_real, simplify_func))
        H.append(row)
    
    return H


# -----------------------------------------------------------------------------
# Main function: Compute Jacobian and optional Hessians
# -----------------------------------------------------------------------------

def jac_hess_symb(exprs, variables=None, fixed=None, deriv2=False, real=False,
                  parallel=False, n_workers=None, simplify_func="powsimp"):
    """
    Compute symbolic Jacobian and optionally Hessian for one or more expressions.

    Parameters
    ----------
    exprs : dict or list of str
        Dictionary or list of symbolic expressions.
    variables : list of str or None, optional
        Variable names to differentiate with respect to.
    fixed : list of str or None
        Variable names to treat as fixed (excluded from differentiation).
    deriv2 : bool, optional
        If True, compute second derivatives (Hessians). Default: False.
    real : bool, optional
        If True, enforce real-valued simplification. Default: False.
    parallel : bool, optional
        If True, use parallel processing for large models. Default: False.
    n_workers : int or None, optional
        Number of worker threads. Default: CPU count.
    simplify_func : str, optional
        Simplification method: "powsimp" (fast, default), "cancel", "ratsimp", 
        or "simplify" (slow but thorough).

    Returns
    -------
    dict
        {
            "jacobian": dict[str, list[str]] — Jacobian rows keyed by expression name
            "hessian": dict[str, list[list[str]]] or None — Hessians if deriv2=True
            "names": list[str] — expression names
            "vars": list[str] — variable names (in order)
        }
    """
    fnames, exprs_syms, vars_syms = _prepare_expressions(exprs, variables)

    # Remove fixed symbols if any
    if fixed is not None:
        fixed_set = set(fixed)
        vars_syms = [v for v in vars_syms if v.name not in fixed_set]

    n_exprs = len(exprs_syms)
    n_vars = len(vars_syms)
    
    # Decide on parallelization
    use_parallel = parallel and n_exprs * n_vars > 100
    
    if use_parallel:
        n_workers = n_workers or os.cpu_count() or 4
        jac = _compute_jacobian_parallel(fnames, exprs_syms, vars_syms, real, n_workers, simplify_func)
    else:
        jac = _compute_jacobian_serial(fnames, exprs_syms, vars_syms, real, simplify_func)

    # Compute Hessian if requested
    hess = None
    if deriv2:
        if use_parallel:
            hess = _compute_hessian_parallel(fnames, exprs_syms, vars_syms, real, n_workers, simplify_func)
        else:
            hess = _compute_hessian_serial(fnames, exprs_syms, vars_syms, real, simplify_func)

    return {
        "jacobian": jac,
        "hessian": hess,
        "names": fnames,
        "vars": [v.name for v in vars_syms]
    }


def _compute_jacobian_serial(fnames, exprs_syms, vars_syms, use_real, simplify_func):
    """Compute Jacobian serially."""
    jac = {}
    for fname, expr in zip(fnames, exprs_syms):
        jac[fname] = _compute_jacobian_row(expr, vars_syms, use_real, simplify_func)
    return jac


def _compute_jacobian_parallel(fnames, exprs_syms, vars_syms, use_real, n_workers, simplify_func):
    """Compute Jacobian in parallel."""
    jac = {}
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_compute_jacobian_row, expr, vars_syms, use_real, simplify_func): fname
            for fname, expr in zip(fnames, exprs_syms)
        }
        
        for future in as_completed(futures):
            fname = futures[future]
            jac[fname] = future.result()
    
    return jac


def _compute_hessian_serial(fnames, exprs_syms, vars_syms, use_real, simplify_func):
    """Compute Hessians serially."""
    hess = {}
    for fname, expr in zip(fnames, exprs_syms):
        hess[fname] = _compute_hessian_matrix(expr, vars_syms, use_real, simplify_func)
    return hess


def _compute_hessian_parallel(fnames, exprs_syms, vars_syms, use_real, n_workers, simplify_func):
    """Compute Hessians in parallel."""
    hess = {}
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_compute_hessian_matrix, expr, vars_syms, use_real, simplify_func): fname
            for fname, expr in zip(fnames, exprs_syms)
        }
        
        for future in as_completed(futures):
            fname = futures[future]
            hess[fname] = future.result()
    
    return hess


# -----------------------------------------------------------------------------
# Convenience wrapper (original API compatibility)
# -----------------------------------------------------------------------------

def derivSymb(exprs, variables=None, fixed=None, deriv2=False, real=False, simplify_func="powsimp"):
    """
    Legacy wrapper for backward compatibility.
    Returns jacobian/hessian in the format expected by generate_fun_cpp().
    """
    result = jac_hess_symb(exprs, variables, fixed, deriv2, real, simplify_func=simplify_func)
    
    # Return in format compatible with generate_fun_cpp
    return {
        "jacobian": result["jacobian"],
        "hessian": result["hessian"],
        "names": result["names"],
        "vars": result["vars"]
    }
