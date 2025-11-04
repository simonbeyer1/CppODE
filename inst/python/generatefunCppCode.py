"""
Algebraic function C++ code generator for CppODE
================================================
Generates C++ source code for evaluating algebraic functions
and their symbolic Jacobians and Hessians.
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
    implicit_multiplication_application
)
import os
import re


# =====================================================================
# Main interface
# =====================================================================

def generate_fun_cpp(exprs, variables, parameters=None,
                     jacobian=None, hessian=None, modelname="model"):
    """
    Generate C++ source code for algebraic function evaluation.

    Parameters
    ----------
    exprs : dict or list
        Named expressions to evaluate.
        Can be a dict {"f1": "a*x + b*y^2"} or a list ["a*x + b*y^2", ...].
    variables : list of str
        Names of variable symbols (inputs).
    parameters : list of str, optional
        Names of parameter symbols.
    jacobian : dict, optional
        Symbolic Jacobian from derivSymb().
        Format: {"f1": ["df1/dx", "df1/da", ...], "f2": [...]}
    hessian : dict, optional
        Symbolic Hessian from derivSymb().
        Format: {"f1": [["d2f1/dx2", ...], [...]], "f2": [...]}
    modelname : str
        Base name for the generated C++ model.

    Returns
    -------
    dict
        Dictionary with keys:
        - "filename": path to generated C++ file
        - "modelname": name used in code generation
    """

    if parameters is None:
        parameters = []

    # Handle empty inputs
    if not variables:
        variables = []
    if not parameters:
        parameters = []

    # Ensure exprs is a dict
    if isinstance(exprs, list):
        exprs = {f"f{i+1}": expr for i, expr in enumerate(exprs)}
    elif not isinstance(exprs, dict):
        raise TypeError("exprs must be a dict or list")

    # Parse all expressions into SymPy
    parsed_exprs = _parse_expressions(exprs, variables, parameters)

    # Generate C++ code
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
# Expression parsing and conversion
# =====================================================================

def _parse_expressions(exprs, variables, parameters):
    """Parse all expressions robustly using SymPy."""
    vars_syms = {v: sp.Symbol(v, real=True) for v in variables}
    pars_syms = {p: sp.Symbol(p, real=True) for p in parameters}
    local_dict = {**vars_syms, **pars_syms}

    transformations = standard_transformations + (
        convert_xor,
        implicit_multiplication_application
    )

    parsed = {}
    for name, expr_str in exprs.items():
        try:
            # Handle "0" specially
            if str(expr_str).strip() == "0":
                parsed[name] = sp.sympify(0)
            else:
                parsed[name] = parse_expr(str(expr_str), local_dict=local_dict,
                                          transformations=transformations,
                                          evaluate=True)
        except Exception as e:
            raise ValueError(f"Failed to parse expression '{name}': {expr_str}\nError: {e}")
    return parsed


def _replace_dirac_delta(expr):
    """
    Replace DiracDelta(x) with discrete delta function:
    delta(x) = 1 if x == 0, else 0
    """
    if expr == 0:
        return expr
    
    def dirac_to_discrete(d):
        if isinstance(d, sp.DiracDelta):
            arg = d.args[0]
            # Exact: 1 if arg == 0, else 0
            return sp.Piecewise((sp.Float(1.0), sp.Eq(arg, 0)), (sp.Float(0.0), True))
        return d
    
    expr = expr.replace(lambda x: isinstance(x, sp.DiracDelta), dirac_to_discrete)
    return expr


def _to_cpp(expr, variables, parameters):
    """
    Convert SymPy expression or string to C++ code.
    Handles symbol replacement with proper ordering to avoid conflicts.
    """
    # Parse string expressions if necessary
    if isinstance(expr, str):
        if expr.strip() == "0":
            return "0.0"
        
        vars_syms = {v: sp.Symbol(v, real=True) for v in variables}
        pars_syms = {p: sp.Symbol(p, real=True) for p in parameters}
        local_dict = {**vars_syms, **pars_syms}
        transformations = standard_transformations + (
            convert_xor,
            implicit_multiplication_application
        )
        expr = parse_expr(str(expr), local_dict=local_dict,
                          transformations=transformations, evaluate=True)

    # Handle zero expression
    if expr == 0:
        return "0.0"
    
    # Replace DiracDelta with discrete delta function (exact comparison)
    expr = _replace_dirac_delta(expr)

    # Generate C++ code using SymPy
    cpp_code = sp.cxxcode(expr, standard='c++17', strict=False)

    # Sort symbols by length (longest first) to avoid partial replacements
    # e.g., replace "xx" before "x"
    all_symbols = list(variables) + list(parameters)
    sorted_symbols = sorted(all_symbols, key=len, reverse=True)

    # Replace variables with x_obs[i]
    for sym in sorted_symbols:
        if sym in variables:
            idx = variables.index(sym)
            cpp_code = _safe_replace(cpp_code, sym, f"x_obs[{idx}]")

    # Replace parameters with p[i]
    for sym in sorted_symbols:
        if sym in parameters:
            idx = parameters.index(sym)
            cpp_code = _safe_replace(cpp_code, sym, f"p[{idx}]")

    return cpp_code


def _safe_replace(text, symbol, replacement):
    """Replace symbols only as full words using regex word boundaries."""
    pattern = r'\b' + re.escape(symbol) + r'\b'
    return re.sub(pattern, replacement, str(text))


# =====================================================================
# C++ code generation
# =====================================================================

def _generate_cpp_code(exprs, variables, parameters, jacobian, hessian, modelname):
    """Generate the full C++ source code for model evaluation."""
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

    # Main evaluation function
    lines.extend(_generate_eval_function(exprs, out_names, variables, parameters, modelname))

    # Jacobian if provided
    if jacobian is not None:
        lines.extend(_generate_jacobian_function(jacobian, out_names, variables, parameters, modelname))

    # Hessian if provided
    if hessian is not None:
        lines.extend(_generate_hessian_function(hessian, out_names, variables, parameters, modelname))

    lines.append("} // extern \"C\"")
    lines.append("")
    return "\n".join(lines)


def _generate_eval_function(exprs, out_names, variables, parameters, modelname):
    """Generate C++ code for the main evaluation loop."""
    n_out = len(out_names)
    n_vars = len(variables)

    lines = [
        f"void {modelname}_eval(double* x, double* y, double* p, int* n, int* k, int* l) {{",
        "    const int n_obs = *n;",
        "    const int n_vars = *k;",
        "    const int n_out = *l;",
        ""
    ]

    if n_vars == 0:
        # No variables case
        for i, out_name in enumerate(out_names):
            cpp_code = _to_cpp(exprs[out_name], variables, parameters)
            lines.append(f"    // {out_name}")
            lines.append(f"    double val_{i} = {cpp_code};")
        lines.append("")
        lines.append("    for (int obs = 0; obs < n_obs; obs++) {")
        for i in range(n_out):
            lines.append(f"        y[obs * n_out + {i}] = val_{i};")
        lines.append("    }")
    else:
        # Normal case with variables
        lines.append("    for (int obs = 0; obs < n_obs; obs++) {")
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


def _generate_jacobian_function(jacobian, out_names, variables, parameters, modelname):
    """
    Generate Jacobian evaluation function.
    Output format: R column-major [n_out, n_sym, n_obs]
    Index: out + sym * n_out + obs * (n_out * n_sym)
    """
    all_symbols = variables + parameters
    n_symbols = len(all_symbols)
    n_vars = len(variables)
    n_out = len(out_names)

    lines = [
        f"void {modelname}_jacobian(double* x, double* jac, double* p, int* n, int* k, int* l) {{",
        "    const int n_obs = *n;",
        "    const int n_vars = *k;",
        f"    const int n_out = {n_out};",
        f"    const int n_symbols = {n_symbols};",
        ""
    ]

    if n_vars == 0:
        # No variables case
        lines.append("    for (int obs = 0; obs < n_obs; obs++) {")
        lines.append("")
    else:
        lines.append("    for (int obs = 0; obs < n_obs; obs++) {")
        lines.append("        const double* x_obs = x + obs * n_vars;")
        lines.append("")

    # Write in R column-major order: out varies fastest
    for i, out_name in enumerate(out_names):
        if out_name not in jacobian:
            for j in range(n_symbols):
                lines.append(f"        jac[{i} + {j} * n_out + obs * (n_out * n_symbols)] = 0.0;  // d({out_name})/d({all_symbols[j]})")
            continue
        
        jac_exprs = jacobian[out_name]
        
        if len(jac_exprs) != n_symbols:
            raise ValueError(f"Jacobian for '{out_name}' has {len(jac_exprs)} entries but expected {n_symbols}")
        
        for j in range(n_symbols):
            sym = all_symbols[j]
            expr = jac_exprs[j]
            cpp_code = _to_cpp(expr, variables, parameters)
            lines.append(f"        // d({out_name})/d({sym})")
            lines.append(f"        jac[{i} + {j} * n_out + obs * (n_out * n_symbols)] = {cpp_code};")
        lines.append("")

    lines.extend(["    }", "}", ""])
    return lines


def _generate_hessian_function(hessian, out_names, variables, parameters, modelname):
    """
    Generate Hessian evaluation function.
    Output format: R column-major [n_out, n_sym, n_sym, n_obs]
    Index: out + sym1 * n_out + sym2 * (n_out * n_sym) + obs * (n_out * n_sym * n_sym)
    """
    all_symbols = variables + parameters
    n_symbols = len(all_symbols)
    n_vars = len(variables)
    n_out = len(out_names)

    lines = [
        f"void {modelname}_hessian(double* x, double* hess, double* p, int* n, int* k, int* l) {{",
        "    const int n_obs = *n;",
        "    const int n_vars = *k;",
        f"    const int n_out = {n_out};",
        f"    const int n_symbols = {n_symbols};",
        ""
    ]

    if n_vars == 0:
        lines.append("    for (int obs = 0; obs < n_obs; obs++) {")
        lines.append("")
    else:
        lines.append("    for (int obs = 0; obs < n_obs; obs++) {")
        lines.append("        const double* x_obs = x + obs * n_vars;")
        lines.append("")

    # Write in R column-major order: out varies fastest, then sym1, then sym2, then obs
    for i, out_name in enumerate(out_names):
        if out_name not in hessian:
            for j in range(n_symbols):
                for k in range(n_symbols):
                    lines.append(
                        f"        hess[{i} + {j} * n_out + {k} * (n_out * n_symbols) + obs * (n_out * n_symbols * n_symbols)] = 0.0;  "
                        f"// d²({out_name})/d({all_symbols[j]})d({all_symbols[k]})"
                    )
            continue
        
        lines.append(f"        // Hessian for {out_name}")
        
        hess_matrix = hessian[out_name]
        
        if len(hess_matrix) != n_symbols:
            raise ValueError(f"Hessian for '{out_name}' has {len(hess_matrix)} rows but expected {n_symbols}")
        
        for j in range(n_symbols):
            hess_row = hess_matrix[j]
            if len(hess_row) != n_symbols:
                raise ValueError(f"Hessian for '{out_name}' row {j} has {len(hess_row)} entries but expected {n_symbols}")
            
            for k in range(n_symbols):
                expr = hess_row[k]
                cpp_code = _to_cpp(expr, variables, parameters)
                lines.append(
                    f"        hess[{i} + {j} * n_out + {k} * (n_out * n_symbols) + obs * (n_out * n_symbols * n_symbols)] = {cpp_code};  "
                    f"// d²({out_name})/d({all_symbols[j]})d({all_symbols[k]})"
                )
        lines.append("")

    lines.extend(["    }", "}", ""])
    return lines
