"""
Algebraic Function C++ Code Generator for CppODE
================================================
Generates C++ source code for evaluating algebraic functions and their
symbolic Jacobians and Hessians. Supports fixed symbols that are treated
as constants (no differentiation) but still appear as runtime parameters.

Author: Simon Beyer
Updated: 2025-11-04
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor
)
import os
import re


# =====================================================================
# Safe parsing configuration
# =====================================================================

def _get_safe_parse_dict():
    """
    Create safe local_dict for parsing that avoids SymPy singleton conflicts.
    Allows all standard mathematical functions but treats S, I, N, etc. as regular symbols.
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
        'ln': sp.log,  # alias for natural log
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
        'cbrt': sp.cbrt,  # cube root
        'root': sp.root,
        'pow': sp.Pow,
        
        # Absolute value and sign
        'abs': sp.Abs,
        'sign': sp.sign,
        
        # Rounding functions
        'floor': sp.floor,
        'ceiling': sp.ceiling,
        'round': lambda x: sp.floor(x + sp.Rational(1, 2)),
        
        # Min/Max
        'min': sp.Min,
        'max': sp.Max,
        
        # Factorial and gamma functions
        'factorial': sp.factorial,
        'gamma': sp.gamma,
        'loggamma': sp.loggamma,
        'digamma': sp.digamma,
        'polygamma': sp.polygamma,
        'beta': sp.beta,
        
        # Error functions
        'erf': sp.erf,
        'erfc': sp.erfc,
        'erfi': sp.erfi,
        
        # Bessel functions
        'besselj': sp.besselj,
        'bessely': sp.bessely,
        'besseli': sp.besseli,
        'besselk': sp.besselk,
        
        # Special functions
        'Heaviside': sp.Heaviside,
        'DiracDelta': sp.DiracDelta,
        'KroneckerDelta': sp.KroneckerDelta,
        
        # Piecewise
        'Piecewise': sp.Piecewise,
        
        # Constants (if you want to allow them explicitly)
        'pi': sp.pi,
        'E': sp.E,
        'euler_gamma': sp.EulerGamma,
        'oo': sp.oo,  # infinity
        
        # Complex functions (optional, depending on your use case)
        're': sp.re,
        'im': sp.im,
        'conjugate': sp.conjugate,
        'arg': sp.arg,
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


# =====================================================================
# Public interface
# =====================================================================

def generate_fun_cpp(exprs, variables, parameters=None,
                     jacobian=None, hessian=None, modelname="model"):
    """
    Generate C++ source code for algebraic model evaluation.

    Parameters
    ----------
    exprs : dict[str, str] or list[str]
        Algebraic model expressions. Can be a dict {"f1": "a*x + b*y"} or a list.
    variables : list[str]
        Variable symbols (vary by observation).
    parameters : list[str], optional
        Parameter symbols (constant across observations). Includes fixed ones.
    jacobian : dict[str, list[str]], optional
        Symbolic Jacobian from derivSymb().
    hessian : dict[str, list[list[str]]], optional
        Symbolic Hessian from derivSymb().
    modelname : str, optional
        Base name for the generated model.

    Returns
    -------
    dict
        {"filename": absolute path, "modelname": model name}
    """
    # variables: String -> [String]
    if isinstance(variables, str):
        variables = [variables]
    elif variables is None:
        variables = []

    # parameters: None / String -> Liste
    if parameters is None:
        parameters = []
    elif isinstance(parameters, str):
        parameters = [parameters]
        
        
    variables = variables or []
    parameters = parameters or []

    # Ensure exprs is a dictionary
    if isinstance(exprs, list):
        exprs = {f"f{i+1}": expr for i, expr in enumerate(exprs)}
    elif not isinstance(exprs, dict):
        raise TypeError("exprs must be a dict or list")

    # Parse to SymPy expressions
    parsed_exprs = _parse_expressions(exprs, variables, parameters)

    # Generate full C++ source code
    cpp_code = _generate_cpp_code(
        parsed_exprs, variables, parameters,
        jacobian=jacobian, hessian=hessian,
        modelname=modelname
    )

    filename = f"{modelname}.cpp"
    with open(filename, "w") as f:
        f.write(cpp_code)

    return {"filename": os.path.abspath(filename), "modelname": modelname}


# =====================================================================
# Parsing and preprocessing
# =====================================================================

def _parse_expressions(exprs, variables, parameters):
    """Parse algebraic expressions into SymPy objects."""
    vars_syms = {v: sp.Symbol(v, real=True) for v in variables}
    pars_syms = {p: sp.Symbol(p, real=True) for p in parameters}
    local_symbols = {**vars_syms, **pars_syms}

    parsed = {}
    for name, expr_str in exprs.items():
        try:
            expr_str = str(expr_str)
            parsed[name] = _safe_sympify(expr_str, local_symbols)
        except Exception as e:
            raise ValueError(
                f"Failed to parse expression '{name}': {expr_str}\nError: {e}"
            )
    return parsed


# =====================================================================
# Utility helpers
# =====================================================================

def _replace_dirac_delta(expr):
    """Replace DiracDelta(x) with a discrete equivalent Piecewise form."""
    if expr == 0:
        return expr

    def dirac_to_piecewise(d):
        if isinstance(d, sp.DiracDelta):
            arg = d.args[0]
            return sp.Piecewise(
                (sp.Float(1.0), sp.Eq(arg, 0)), (sp.Float(0.0), True)
            )
        return d

    return expr.replace(lambda x: isinstance(x, sp.DiracDelta), dirac_to_piecewise)


def _safe_replace(text, symbol, replacement):
    """Replace symbol in code safely using regex word boundaries."""
    pattern = r"\b" + re.escape(symbol) + r"\b"
    return re.sub(pattern, replacement, str(text))


def _to_cpp(expr, variables, parameters):
    """
    Convert a SymPy expression or string to valid C++ code.

    * Variables → x_obs[i]
    * Parameters (including fixed) → p[i]
    """
    # Convert string to SymPy expression if needed
    if isinstance(expr, str):
        if expr.strip() == "0":
            return "0.0"
        vars_syms = {v: sp.Symbol(v, real=True) for v in variables}
        pars_syms = {p: sp.Symbol(p, real=True) for p in parameters}
        local_symbols = {**vars_syms, **pars_syms}
        
        expr = _safe_sympify(expr, local_symbols)

    if expr == 0:
        return "0.0"

    expr = _replace_dirac_delta(expr)
    cpp_code = sp.cxxcode(expr, standard="c++17", strict=False)

    # Perform safe replacements for all declared symbols
    all_symbols = list(variables) + list(parameters)
    sorted_symbols = sorted(all_symbols, key=len, reverse=True)

    # Replace variables with x_obs[i]
    for sym in sorted_symbols:
        if sym in variables:
            idx = variables.index(sym)
            cpp_code = _safe_replace(cpp_code, sym, f"x_obs[{idx}]")

    # Replace all parameters (including fixed) with p[i]
    for sym in sorted_symbols:
        if sym in parameters:
            idx = parameters.index(sym)
            cpp_code = _safe_replace(cpp_code, sym, f"p[{idx}]")

    return cpp_code


# =====================================================================
# C++ source assembly
# =====================================================================

def _generate_cpp_code(exprs, variables, parameters, jacobian, hessian, modelname):
    """Assemble the complete C++ source for the model."""
    out_names = list(exprs.keys())

    lines = [
        f"// Auto-generated C++ code for {modelname}",
        "// Generated by CppODE",
        "// ------------------------------------------------------------",
        "#include <cmath>",
        "#include <algorithm>",
        "",
        "extern \"C\" {",
        "",
        f"// Variables: {', '.join(variables) if variables else 'none'}",
        f"// Parameters: {', '.join(parameters) if parameters else 'none'}",
        f"// Outputs: {', '.join(out_names)}",
        ""
    ]

    lines.extend(_generate_eval_function(exprs, out_names, variables, parameters, modelname))

    if jacobian is not None:
        lines.extend(_generate_jacobian_function(jacobian, out_names, variables, parameters, modelname))

    if hessian is not None:
        lines.extend(_generate_hessian_function(hessian, out_names, variables, parameters, modelname))

    lines.append("} // extern \"C\"")
    lines.append("")
    return "\n".join(lines)


# =====================================================================
# Evaluation function
# =====================================================================

def _generate_eval_function(exprs, out_names, variables, parameters, modelname):
    """Generate main evaluation loop in C++."""
    n_out = len(out_names)
    n_vars = len(variables)

    lines = [
        f"void {modelname}_eval(double* x, double* y, double* p, int* n, int* k, int* l) {{",
        "    const int n_obs = *n;",
        "    const int n_vars = *k;",
        "    const int n_out  = *l;",
        ""
    ]

    lines.append("    for (int obs = 0; obs < n_obs; obs++) {")
    if n_vars > 0:
        lines.append("        const double* x_obs = x + obs * n_vars;")
    lines.append("")

    for i, out_name in enumerate(out_names):
        cpp_code = _to_cpp(exprs[out_name], variables, parameters)
        lines.append(f"        // {out_name}")
        lines.append(f"        y[obs * n_out + {i}] = {cpp_code};")
        lines.append("")

    lines.append("    }")
    lines.extend(["}", ""])
    return lines


# =====================================================================
# Jacobian generation
# =====================================================================

def _generate_jacobian_function(jacobian, out_names, variables, parameters, modelname):
    """Generate Jacobian evaluation function (robust to missing fixed symbols)."""
    first_fn = next(iter(jacobian))
    n_symbols = len(jacobian[first_fn])
    n_vars = len(variables)
    n_out = len(out_names)

    lines = [
        f"void {modelname}_jacobian(double* x, double* jac, double* p, int* n, int* k, int* l) {{",
        "    const int n_obs = *n;",
        "    const int n_vars = *k;",
        f"    const int n_out = {n_out};",
        f"    const int n_symbols = {n_symbols};",
        "",
        "    for (int obs = 0; obs < n_obs; obs++) {"
    ]
    if n_vars > 0:
        lines.append("        const double* x_obs = x + obs * n_vars;")
        lines.append("")

    for i, out_name in enumerate(out_names):
        if out_name not in jacobian:
            for j in range(n_symbols):
                lines.append(
                    f"        jac[{i} + {j} * n_out + obs * (n_out * n_symbols)] = 0.0;"
                )
            continue

        jac_exprs = jacobian[out_name]
        for j, expr in enumerate(jac_exprs):
            cpp_code = _to_cpp(expr, variables, parameters)
            lines.append(
                f"        jac[{i} + {j} * n_out + obs * (n_out * n_symbols)] = {cpp_code};"
            )
        lines.append("")

    lines.extend(["    }", "}", ""])
    return lines


# =====================================================================
# Hessian generation
# =====================================================================

def _generate_hessian_function(hessian, out_names, variables, parameters, modelname):
    """Generate Hessian evaluation function (robust to missing fixed symbols)."""
    first_fn = next(iter(hessian))
    n_symbols = len(hessian[first_fn])
    n_vars = len(variables)
    n_out = len(out_names)

    lines = [
        f"void {modelname}_hessian(double* x, double* hess, double* p, int* n, int* k, int* l) {{",
        "    const int n_obs = *n;",
        "    const int n_vars = *k;",
        f"    const int n_out = {n_out};",
        f"    const int n_symbols = {n_symbols};",
        "",
        "    for (int obs = 0; obs < n_obs; obs++) {"
    ]
    if n_vars > 0:
        lines.append("        const double* x_obs = x + obs * n_vars;")
        lines.append("")

    for i, out_name in enumerate(out_names):
        if out_name not in hessian:
            for j in range(n_symbols):
                for k in range(n_symbols):
                    lines.append(
                        f"        hess[{i} + {j} * n_out + {k} * (n_out * n_symbols) + "
                        f"obs * (n_out * n_symbols * n_symbols)] = 0.0;"
                    )
            continue

        hess_matrix = hessian[out_name]
        for j, hess_row in enumerate(hess_matrix):
            for k, expr in enumerate(hess_row):
                cpp_code = _to_cpp(expr, variables, parameters)
                lines.append(
                    f"        hess[{i} + {j} * n_out + {k} * (n_out * n_symbols) + "
                    f"obs * (n_out * n_symbols * n_symbols)] = {cpp_code};"
                )
        lines.append("")

    lines.extend(["    }", "}", ""])
    return lines
