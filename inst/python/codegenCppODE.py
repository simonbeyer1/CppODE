"""
ODE and Jacobian C++ code generator for CppODE
==============================================

This module provides entry points for generating C++ code:

- generate_ode_cpp(...)
    Generates C++ code for the ODE right-hand side and its Jacobian,
    using SymPy for parsing and symbolic differentiation.

- generate_event_code(...)
    Generates C++ code for fixed-time and root-triggered events,
    including analytical gradients for saltation matrix corrections.

- generate_rootfunc_code(...)
    Generates C++ code for root function based termination.

Author: Simon Beyer
"""

import re
import numbers
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
)

# =====================================================================
# Safe parsing configuration
# =====================================================================

def _get_safe_parse_dict():
    """Construct a local dictionary for SymPy parsing."""
    return {
        "exp": sp.exp,
        "exp10": lambda x: sp.Pow(10, x),
        "exp2": lambda x: sp.Pow(2, x),
        "log": sp.log,
        "ln": sp.log,
        "log10": lambda x: sp.log(x, 10),
        "log2": lambda x: sp.log(x, 2),
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "acot": sp.acot, "asec": sp.asec, "acsc": sp.acsc,
        "atan2": sp.atan2,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "coth": sp.coth, "sech": sp.sech, "csch": sp.csch,
        "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
        "acoth": sp.acoth, "asech": sp.asech, "acsch": sp.acsch,
        "sqrt": sp.sqrt, "cbrt": sp.cbrt, "root": sp.root, "pow": sp.Pow,
        "abs": sp.Abs, "sign": sp.sign,
        "floor": sp.floor, "ceiling": sp.ceiling,
        "min": sp.Min, "max": sp.Max,
        "factorial": sp.factorial, "gamma": sp.gamma,
        "loggamma": sp.loggamma, "digamma": sp.digamma,
        "erf": sp.erf, "erfc": sp.erfc,
        "besselj": sp.besselj, "bessely": sp.bessely,
        "besseli": sp.besseli, "besselk": sp.besselk,
        "Heaviside": sp.Heaviside, "DiracDelta": sp.DiracDelta,
        "Piecewise": sp.Piecewise,
        "pi": sp.pi, "E": sp.E, "oo": sp.oo,
    }


def _safe_sympify(expr_str, local_symbols=None):
    """Safely parse a string expression to SymPy."""
    expr_str = str(expr_str).strip()
    if expr_str == "0":
        return sp.Integer(0)

    safe_local = _get_safe_parse_dict()
    if local_symbols:
        safe_local = {**safe_local, **local_symbols}

    transformations = standard_transformations + (convert_xor,)
    return parse_expr(
        expr_str,
        local_dict=safe_local,
        transformations=transformations,
        evaluate=True,
    )


def _safe_replace(text, symbol, replacement):
    """Replace symbol in text avoiding already-indexed variables."""
    pattern = r"(?<![a-zA-Z0-9_])" + re.escape(symbol) + r"(?![a-zA-Z0-9_\[])"
    return re.sub(pattern, replacement, str(text))


def _ensure_double_literals(cpp_code):
    """Convert integer literals to double literals in C++ code."""
    sci_pattern = r'(\d+\.?\d*[eE][+-]?\d+)'
    
    sci_numbers = []
    def store_sci(match):
        sci_numbers.append(match.group(0))
        return f'__SCI_PLACEHOLDER_{len(sci_numbers)-1}__'
    
    temp = re.sub(sci_pattern, store_sci, cpp_code)
    int_pattern = r'(?<![a-zA-Z0-9_.\[])(\d+)(?![0-9.\]])'
    temp = re.sub(int_pattern, lambda m: m.group(1) + '.0', temp)
    
    for i, sci in enumerate(sci_numbers):
        temp = temp.replace(f'__SCI_PLACEHOLDER_{i}__', sci)
    
    return temp


# =====================================================================
# Math macro replacement map
# =====================================================================

_MATH_MACRO_MAP = {
    "M_E": "std::exp(1.0)",
    "M_LOG2E": "1.0 / std::log(2.0)",
    "M_LOG10E": "1.0 / std::log(10.0)",
    "M_LN2": "std::log(2.0)",
    "M_LN10": "std::log(10.0)",
    "M_PI": "std::acos(-1.0)",
    "M_PI_2": "(std::acos(-1.0) * 0.5)",
    "M_PI_4": "(std::acos(-1.0) * 0.25)",
    "M_1_PI": "(1.0 / std::acos(-1.0))",
    "M_2_PI": "(2.0 / std::acos(-1.0))",
    "M_2_SQRTPI": "(2.0 / std::sqrt(std::acos(-1.0)))",
    "M_SQRT2": "std::sqrt(2.0)",
    "M_SQRT1_2": "std::sqrt(0.5)",
}


# =====================================================================
# _to_cpp - Convert SymPy expression to C++ code
# =====================================================================

def _to_cpp(expr, states, params, n_states, num_type, forcings=None, use_initial_states=False):
    """Convert a SymPy expression to C++ code."""
    if forcings is None:
        forcings = []
    
    cpp_code = str(sp.cxxcode(expr, standard="c++17", strict=False)).replace("\n", " ")
    
    # Replace non-standard math macros
    for macro, repl in _MATH_MACRO_MAP.items():
        cpp_code = cpp_code.replace(macro, repl)
    
    # Use FADBAD++ math functions for AD types
    if num_type in ("AD", "AD2"):
        for fn in [
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
            "exp", "log", "sqrt", "pow", "abs", "min", "max"
        ]:
            cpp_code = cpp_code.replace(f"std::{fn}", f"fadbad::{fn}")
    
    # Replace forcings
    for i, forcing in enumerate(forcings):
        cpp_code = _safe_replace(cpp_code, forcing, f"(*F[{i}])(t)")
    
    # Replace params
    for j, param in enumerate(params):
        cpp_code = _safe_replace(cpp_code, param, f"params[{n_states + j}]")
    
    # Replace states
    if use_initial_states:
        for i, state in enumerate(states):
            cpp_code = _safe_replace(cpp_code, state, f"params[{i}]")
    else:
        for i, state in enumerate(states):
            cpp_code = _safe_replace(cpp_code, state, f"x[{i}]")
    
    # Explicit notation for initial values
    for i, state in enumerate(states):
        cpp_code = _safe_replace(cpp_code, f"{state}_0", f"params[{i}]")
    
    cpp_code = _safe_replace(cpp_code, "time", "t")
    cpp_code = re.sub(r"\s+", "", cpp_code)
    cpp_code = _ensure_double_literals(cpp_code)
    
    return cpp_code


# =====================================================================
# Generate gradient code for a scalar expression
# =====================================================================

def _generate_gradient_cpp(expr, states_list, params_list, n_states, num_type,
                           forcings_list, local_symbols, states_syms):
    """
    Generate C++ code for the gradient ∂expr/∂x of a scalar expression.
    
    Returns a list of C++ expressions, one for each state variable.
    """
    gradient = []
    for state_name in states_list:
        state_sym = states_syms[state_name]
        deriv = sp.diff(expr, state_sym)
        deriv_cpp = _to_cpp(deriv, states_list, params_list, n_states,
                           num_type, forcings_list, use_initial_states=False)
        deriv_cpp = str(deriv_cpp).replace("params[", "full_params[")
        gradient.append(deriv_cpp)
    return gradient


def _generate_time_deriv_cpp(expr, t_sym, states_list, params_list, n_states,
                             num_type, forcings_list):
    """
    Generate C++ code for ∂expr/∂t.
    
    Returns the C++ expression as a string.
    """
    deriv = sp.diff(expr, t_sym)
    deriv_cpp = _to_cpp(deriv, states_list, params_list, n_states,
                       num_type, forcings_list, use_initial_states=False)
    return str(deriv_cpp).replace("params[", "full_params[")


def _generate_gradient_lambda(gradient_cpp, n_states, num_type):
    """
    Generate a C++ lambda that returns a gradient vector.
    
    Parameters:
    -----------
    gradient_cpp : list of str
        C++ expressions for each gradient component
    n_states : int
        Number of state variables
    num_type : str
        Numeric type (e.g., "AD", "AD2", "double")
    
    Returns:
    --------
    str : C++ lambda expression
    """
    state_type = f"ublas::vector<{num_type}>"
    
    # Build the lambda body
    lines = [f"[full_params, &F](const {state_type}& x, const {num_type}& t) -> {state_type} {{"]
    lines.append(f"      {state_type} grad({n_states});")
    
    for i, grad_expr in enumerate(gradient_cpp):
        lines.append(f"      grad[{i}] = {grad_expr};")
    
    lines.append("      return grad;")
    lines.append("    }")
    
    return "\n".join(lines)


# =====================================================================
# Main ODE generator
# =====================================================================

def generate_ode_cpp(
    rhs_dict,
    params_list,
    num_type="AD",
    fixed_states=None,
    fixed_params=None,
    forcings_list=None,
):
    """
    Generate C++ code for ODE system and Jacobian.
    
    Parameters:
    -----------
    rhs_dict : dict
        Dictionary mapping state names to RHS expressions
    params_list : list of str
        Parameter names
    num_type : str
        Numeric type ("AD", "AD2", "double")
    fixed_states : list of str, optional
        States to exclude from sensitivity
    fixed_params : list of str, optional
        Parameters to exclude from sensitivity
    forcings_list : list of str, optional
        Forcing function names
    
    Returns:
    --------
    dict with keys: "ode_code", "jac_code", "jac_matrix", "time_derivs",
                    "states", "params", "forcings"
    """
    # Normalize inputs
    if fixed_states is None:
        fixed_states = []
    if fixed_params is None:
        fixed_params = []
    if forcings_list is None:
        forcings_list = []
    if params_list is None:
        params_list = []

    if isinstance(params_list, str):
        params_list = [params_list]
    else:
        params_list = list(params_list)
    
    if isinstance(forcings_list, str):
        forcings_list = [forcings_list]
    else:
        forcings_list = list(forcings_list)
    
    if isinstance(fixed_states, str):
        fixed_states = [fixed_states]
    else:
        fixed_states = list(fixed_states)
    
    if isinstance(fixed_params, str):
        fixed_params = [fixed_params]
    else:
        fixed_params = list(fixed_params)

    states_list = list(rhs_dict.keys())
    odes_list = list(rhs_dict.values())
    n_states = len(states_list)

    # Define symbols
    states_syms = {n: sp.Symbol(n, real=True) for n in states_list}
    params_syms = {n: sp.Symbol(n, real=True) for n in params_list}
    forcing_syms = {n: sp.Symbol(n, real=True) for n in forcings_list}
    t = sp.Symbol("time", real=True)

    local_symbols = {}
    local_symbols.update(states_syms)
    local_symbols.update(params_syms)
    local_symbols.update(forcing_syms)
    local_symbols["time"] = t

    # Parse RHS expressions
    exprs = [_safe_sympify(expr, local_symbols) for expr in odes_list]

    states_syms_list = [states_syms[s] for s in states_list]

    # Compute Jacobian matrix
    jac_matrix = [
        [sp.diff(expr, s) for s in states_syms_list]
        for expr in exprs
    ]

    # Compute time derivatives
    time_derivs = [sp.diff(expr, t) for expr in exprs]

    # Generate ODE system code
    ode_cpp_lines = [
        "// ODE system",
        "struct ode_system {",
        f"  ublas::vector<{num_type}> params;",
        f"  std::vector<const cppode::PchipForcing<{num_type}>*> F;",
        "",
        f"  ode_system(const ublas::vector<{num_type}>& p_,",
        f"             const std::vector<const cppode::PchipForcing<{num_type}>*>& F_)",
        "    : params(p_), F(F_) {}",
        "",
        f"  void operator()(const ublas::vector<{num_type}>& x,",
        f"                  ublas::vector<{num_type}>& dxdt,",
        f"                  const {num_type}& t) {{",
    ]

    for i, expr in enumerate(exprs):
        ode_cpp_lines.append(
            f"    dxdt[{i}] = {_to_cpp(expr, states_list, params_list, n_states, num_type, forcings_list)};"
        )

    ode_cpp_lines += ["  }", "};"]

    # Generate Jacobian code
    jac_cpp_lines = [
        "// Jacobian for stiff solver",
        "struct jacobian {",
        f"  ublas::vector<{num_type}> params;",
        f"  std::vector<const cppode::PchipForcing<{num_type}>*> F;",
        "",
        f"  jacobian(const ublas::vector<{num_type}>& p_,",
        f"           const std::vector<const cppode::PchipForcing<{num_type}>*>& F_)",
        "    : params(p_), F(F_) {}",
        "",
        f"  void operator()(const ublas::vector<{num_type}>& x,",
        f"                  ublas::matrix<{num_type}>& J,",
        f"                  const {num_type}& t,",
        f"                  ublas::vector<{num_type}>& dfdt) {{",
    ]

    # J = df/dx (no forcings in Jacobian)
    for i in range(n_states):
        for j in range(n_states):
            jac_cpp_lines.append(
                f"    J({i},{j}) = {_to_cpp(jac_matrix[i][j], states_list, params_list, n_states, num_type, forcings=[])};"
            )

    # dfdt with forcing chain rule
    for i, expr in enumerate(exprs):
        cpp_code = _to_cpp(time_derivs[i], states_list, params_list, n_states, num_type, forcings_list)

        forcing_terms = []
        for j, fname in enumerate(forcings_list):
            df_dF = sp.diff(expr, forcing_syms[fname])
            if df_dF != 0:
                forcing_terms.append(
                    f"({_to_cpp(df_dF, states_list, params_list, n_states, num_type, forcings_list)})*F[{j}]->derivative(t)"
                )

        if forcing_terms:
            cpp_code = " + ".join([cpp_code] + forcing_terms) if cpp_code != "0" else " + ".join(forcing_terms)

        jac_cpp_lines.append(f"    dfdt[{i}] = {cpp_code};")

    jac_cpp_lines += ["  }", "};"]

    return {
        "ode_code": "\n".join(ode_cpp_lines),
        "jac_code": "\n".join(jac_cpp_lines),
        "jac_matrix": [[str(e) for e in row] for row in jac_matrix],
        "time_derivs": [str(d) for d in time_derivs],
        "states": states_list,
        "params": params_list,
        "forcings": forcings_list,
    }


# =====================================================================
# Forcing initialization code generation
# =====================================================================

def generate_forcing_init_code(n_forcings, num_type="AD"):
    """Generate C++ code to initialize PchipForcing objects from R raw data."""
    return [
        "",
        "  // --- Initialize forcings (PCHIP interpolation) ---",
        f"  int n_forcings = Rf_length(forcingTimesSEXP);",
        f"  std::vector<cppode::PchipForcing<{num_type}>> forcing_storage(n_forcings);",
        f"  std::vector<const cppode::PchipForcing<{num_type}>*> F(n_forcings);",
        "",
        "  for (int fi = 0; fi < n_forcings; ++fi) {",
        "    SEXP times_i = VECTOR_ELT(forcingTimesSEXP, fi);",
        "    SEXP values_i = VECTOR_ELT(forcingValuesSEXP, fi);",
        "    int n_points = Rf_length(times_i);",
        "",
        "    std::vector<double> ftimes(REAL(times_i), REAL(times_i) + n_points);",
        "    std::vector<double> fvalues(REAL(values_i), REAL(values_i) + n_points);",
        "",
        "    forcing_storage[fi].initialize(ftimes, fvalues);",
        "    F[fi] = &forcing_storage[fi];",
        "  }",
        "",
    ]


# =====================================================================
# Event code generation with analytical gradients
# =====================================================================

def _get_list_value(dict_or_df, key, index, n_events):
    """Extract a value from a dict-of-lists."""
    if key not in dict_or_df:
        return None
    value = dict_or_df[key]
    if isinstance(value, (list, tuple)):
        if index < len(value):
            return value[index]
        return None
    return value


def _is_valid_value(value):
    """Check whether a value is meaningful (not NA/None/boolean)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Number):
        try:
            if value != value:  # NaN check
                return False
        except Exception:
            pass
        return True
    str_val = str(value).lower().strip()
    if str_val in {"none", "nan", "na", ""}:
        return False
    if str_val in {"true", "false"}:
        return False
    return True


def _parse_value_or_expression(value, local_symbols, states_list, params_list, 
                                n_states, num_type, forcings_list=None):
    """Interpret a value as numeric literal or symbolic expression."""
    if forcings_list is None:
        forcings_list = []
    
    if not _is_valid_value(value):
        return None

    value_str = str(value).strip()

    # Try numeric literal first
    try:
        float(value_str)
        return value_str
    except (ValueError, TypeError):
        pass

    # Parse as symbolic expression
    try:
        value_expr = _safe_sympify(value_str, local_symbols)
        value_code = _to_cpp(value_expr, states_list, params_list, n_states, 
                            num_type, forcings=forcings_list)
        return str(value_code)
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{value_str}': {e}")


def generate_event_code(events_df, states_list, params_list, n_states, 
                        num_type="AD", forcings_list=None):
    """
    Generate C++ initialization lines for events with analytical gradients.
    
    This generates FixedEvent and RootEvent structures including:
    - dg_dx_func: gradient of root function w.r.t. states
    - dg_dt_func: time derivative of root function
    - dh_dx_func: gradient of value function w.r.t. states
    
    These analytical derivatives are required for correct saltation
    matrix corrections when computing sensitivities (deriv=TRUE).
    """
    if forcings_list is None:
        forcings_list = []
    if states_list is None:
        states_list = []
    if params_list is None:
        params_list = []
    
    # Normalize inputs
    if isinstance(states_list, str):
        states_list = [states_list]
    else:
        states_list = list(states_list)
    if isinstance(params_list, str):
        params_list = [params_list]
    else:
        params_list = list(params_list)
    if isinstance(forcings_list, str):
        forcings_list = [forcings_list]
    else:
        forcings_list = list(forcings_list)
    
    if events_df is None or len(events_df) == 0:
        return []

    if hasattr(events_df, "to_dict"):
        events_dict = events_df.to_dict("list")
    else:
        events_dict = events_df

    # Create symbols
    states_syms = {name: sp.Symbol(name, real=True) for name in states_list}
    params_syms = {name: sp.Symbol(name, real=True) for name in params_list}
    t = sp.Symbol("time", real=True)

    local_symbols = {}
    local_symbols.update(states_syms)
    local_symbols.update(params_syms)
    for name in forcings_list:
        local_symbols[name] = sp.Symbol(name, real=True)
    local_symbols["time"] = t

    event_lines = []
    state_type = f"ublas::vector<{num_type}>"

    # Determine number of events
    list_lengths = []
    for v in events_dict.values():
        if isinstance(v, (list, tuple)):
            list_lengths.append(len(v))
    n_events = max(list_lengths) if list_lengths else 1

    for i in range(n_events):
        var_raw = _get_list_value(events_dict, "var", i, n_events)
        if var_raw is None:
            continue

        var_name = str(var_raw)
        if var_name not in states_list:
            raise ValueError(f"Event {i}: unknown state variable '{var_name}'")
        var_idx = states_list.index(var_name)

        value_raw = _get_list_value(events_dict, "value", i, n_events)
        value_code = _parse_value_or_expression(
            value_raw, local_symbols, states_list, params_list, n_states, 
            num_type, forcings_list
        )
        if value_code is None:
            raise ValueError(f"Event {i}: 'value' is required but is NA/None")
        value_code = str(value_code).replace("params[", "full_params[")

        # Parse value expression for gradient computation
        value_expr = _safe_sympify(str(value_raw), local_symbols)

        method_raw = _get_list_value(events_dict, "method", i, n_events)
        method = str(method_raw).lower() if method_raw is not None else "replace"
        method_map = {
            "replace": "EventMethod::Replace",
            "add": "EventMethod::Add",
            "multiply": "EventMethod::Multiply",
        }
        method_code = method_map.get(method, "EventMethod::Replace")

        time_raw = _get_list_value(events_dict, "time", i, n_events)
        time_code = _parse_value_or_expression(
            time_raw, local_symbols, states_list, params_list, n_states, 
            num_type, forcings_list
        )

        root_raw = _get_list_value(events_dict, "root", i, n_events)
        root_code = _parse_value_or_expression(
            root_raw, local_symbols, states_list, params_list, n_states, 
            num_type, forcings_list
        )

        if time_code is not None:
            # ============================================================
            # Fixed-time event
            # ============================================================
            time_code = str(time_code).replace("params[", "full_params[")
            
            # Compute value function gradient: ∂h/∂x
            dh_dx_cpp = _generate_gradient_cpp(
                value_expr, states_list, params_list, n_states,
                num_type, forcings_list, local_symbols, states_syms
            )
            dh_dx_lambda = _generate_gradient_lambda(dh_dx_cpp, n_states, num_type)
            
            event_lines.append(f"  // Fixed event {i}: {var_name} at t = {time_raw}")
            event_lines.append(f"  fixed_events.emplace_back(FixedEvent<{state_type}, {num_type}>{{")
            event_lines.append(f"    {time_code},  // time")
            event_lines.append(f"    {var_idx},    // state_index")
            event_lines.append(f"    [full_params, &F](const {state_type}& x, const {num_type}& t) -> {num_type} {{")
            event_lines.append(f"      return {value_code};")
            event_lines.append(f"    }},  // value_func")
            event_lines.append(f"    {method_code},  // method")
            event_lines.append(f"    {dh_dx_lambda}  // dh_dx_func")
            event_lines.append(f"  }});")
            event_lines.append("")
            
        elif root_code is not None:
            # ============================================================
            # Root-finding event
            # ============================================================
            root_code = str(root_code).replace("params[", "full_params[")
            
            # Parse root expression
            root_expr = _safe_sympify(str(root_raw), local_symbols)
            
            # Compute root function gradient: ∂g/∂x
            dg_dx_cpp = _generate_gradient_cpp(
                root_expr, states_list, params_list, n_states,
                num_type, forcings_list, local_symbols, states_syms
            )
            dg_dx_lambda = _generate_gradient_lambda(dg_dx_cpp, n_states, num_type)
            
            # Compute root function time derivative: ∂g/∂t
            dg_dt_cpp = _generate_time_deriv_cpp(
                root_expr, t, states_list, params_list, n_states,
                num_type, forcings_list
            )
            
            # Compute value function gradient: ∂h/∂x
            dh_dx_cpp = _generate_gradient_cpp(
                value_expr, states_list, params_list, n_states,
                num_type, forcings_list, local_symbols, states_syms
            )
            dh_dx_lambda = _generate_gradient_lambda(dh_dx_cpp, n_states, num_type)
            
            # Get optional parameters
            terminal_raw = _get_list_value(events_dict, "terminal", i, n_events)
            terminal = "true" if terminal_raw and str(terminal_raw).lower() == "true" else "false"
            
            direction_raw = _get_list_value(events_dict, "direction", i, n_events)
            try:
                direction = int(direction_raw) if direction_raw is not None else 0
            except (ValueError, TypeError):
                direction = 0
            
            event_lines.append(f"  // Root event {i}: {var_name} when {root_raw} = 0")
            event_lines.append(f"  root_events.push_back(RootEvent<{state_type}, {num_type}>{{")
            event_lines.append(f"    [full_params, &F](const {state_type}& x, const {num_type}& t) -> {num_type} {{")
            event_lines.append(f"      return {root_code};")
            event_lines.append(f"    }},  // func (root condition g)")
            event_lines.append(f"    {var_idx},  // state_index")
            event_lines.append(f"    [full_params, &F](const {state_type}& x, const {num_type}& t) -> {num_type} {{")
            event_lines.append(f"      return {value_code};")
            event_lines.append(f"    }},  // value_func (h)")
            event_lines.append(f"    {method_code},  // method")
            event_lines.append(f"    {terminal},     // terminal")
            event_lines.append(f"    {direction},    // direction")
            event_lines.append(f"    {dg_dx_lambda},  // dg_dx_func")
            event_lines.append(f"    [full_params, &F](const {state_type}& x, const {num_type}& t) -> {num_type} {{")
            event_lines.append(f"      return {num_type}({dg_dt_cpp});")
            event_lines.append(f"    }},  // dg_dt_func")
            event_lines.append(f"    {dh_dx_lambda}  // dh_dx_func")
            event_lines.append(f"  }});")
            event_lines.append("")
            
        else:
            raise ValueError(f"Event {i}: must specify either 'time' or 'root'")

    return event_lines


# =====================================================================
# Root function code generation
# =====================================================================

def generate_rootfunc_code(rootfunc, states_list, params_list, n_states,
                           num_type="AD", forcings_list=None):
    """
    Generate C++ code for root function based termination.
    
    Handles two cases:
    1. rootfunc = "equilibrate": steady-state detection
    2. rootfunc = list of expressions: stop when any crosses zero
    """
    if forcings_list is None:
        forcings_list = []
    
    if rootfunc is None:
        return []
    
    # Handle single string vs list
    if isinstance(rootfunc, str):
        if rootfunc.strip().lower() == "equilibrate":
            return _generate_equilibrate_code(num_type)
        else:
            rootfunc = [rootfunc]
    
    if not isinstance(rootfunc, (list, tuple)):
        raise ValueError(f"rootfunc must be 'equilibrate' or a list of expressions, got: {type(rootfunc)}")
    
    return _generate_user_rootfunc_code(
        rootfunc, states_list, params_list, n_states, num_type, forcings_list
    )


def _generate_equilibrate_code(num_type):
    """Generate C++ code for steady-state detection."""
    state_type = f"ublas::vector<{num_type}>"
    
    return [
        "",
        "  // --- Steady-state termination (rootfunc = 'equilibrate') ---",
        f"  auto ss_root_func = make_steady_state_root_func<ode_system, {state_type}, {num_type}>(sys, root_tol);",
        f"  root_events.push_back(RootEvent<{state_type}, {num_type}>{{",
        f"    ss_root_func,",
        f"    0,            // state_index (ignored for terminal)",
        f"    [](const {state_type}&, const {num_type}&) {{ return {num_type}(0.0); }},  // value_func",
        f"    EventMethod::Replace,  // method (ignored for terminal)",
        f"    true,         // terminal = true",
        f"    -1,           // direction = -1",
        f"    nullptr,      // dg_dx_func (not needed for terminal)",
        f"    nullptr,      // dg_dt_func (not needed for terminal)",
        f"    nullptr       // dh_dx_func (not needed for terminal)",
        f"  }});",
        ""
    ]


def _generate_user_rootfunc_code(rootfunc_list, states_list, params_list, 
                                  n_states, num_type, forcings_list):
    """Generate C++ code for user-defined root expressions (terminal events)."""
    states_syms = {name: sp.Symbol(name, real=True) for name in states_list}
    params_syms = {name: sp.Symbol(name, real=True) for name in params_list}
    t = sp.Symbol("time", real=True)
    
    local_symbols = {}
    local_symbols.update(states_syms)
    local_symbols.update(params_syms)
    for name in forcings_list:
        local_symbols[name] = sp.Symbol(name, real=True)
    local_symbols["time"] = t
    
    state_type = f"ublas::vector<{num_type}>"
    lines = [
        "",
        "  // --- User-defined root function termination ---"
    ]
    
    for i, expr_str in enumerate(rootfunc_list):
        expr_str = str(expr_str).strip()
        if not expr_str:
            continue
        
        try:
            expr = _safe_sympify(expr_str, local_symbols)
            root_code = _to_cpp(expr, states_list, params_list, n_states, 
                               num_type, forcings_list)
            root_code = str(root_code).replace("params[", "full_params[")
        except Exception as e:
            raise ValueError(f"Failed to parse rootfunc expression '{expr_str}': {e}")
        
        # Generate analytical gradients for terminal root functions
        dg_dx_cpp = _generate_gradient_cpp(
            expr, states_list, params_list, n_states,
            num_type, forcings_list, local_symbols, states_syms
        )
        dg_dx_lambda = _generate_gradient_lambda(dg_dx_cpp, n_states, num_type)
        
        dg_dt_cpp = _generate_time_deriv_cpp(
            expr, t, states_list, params_list, n_states,
            num_type, forcings_list
        )
        
        lines.extend([
            f"  // rootfunc[{i}]: {expr_str}",
            f"  root_events.push_back(RootEvent<{state_type}, {num_type}>{{",
            f"    [full_params, &F](const {state_type}& x, const {num_type}& t) -> {num_type} {{",
            f"      return {root_code};",
            f"    }},  // func",
            f"    0,            // state_index (ignored for terminal)",
            f"    [](const {state_type}&, const {num_type}&) {{ return {num_type}(0.0); }},  // value_func",
            f"    EventMethod::Replace,  // method",
            f"    true,         // terminal = true",
            f"    0,            // direction = 0 (any crossing)",
            f"    {dg_dx_lambda},  // dg_dx_func",
            f"    [full_params, &F](const {state_type}& x, const {num_type}& t) -> {num_type} {{",
            f"      return {num_type}({dg_dt_cpp});",
            f"    }},  // dg_dt_func",
            f"    nullptr       // dh_dx_func (not needed for terminal)",
            f"  }});",
        ])
    
    lines.append("")
    return lines
