"""
ODE and Jacobian C++ code generator for CppODE
==============================================
Handles symbolic differentiation for the jacobian and C++ code generation

Author: Simon Beyer
Updated: 2025-11-04
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    convert_xor, 
    implicit_multiplication_application
)
import re


# =====================================================================
# Safe parsing configuration
# =====================================================================

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
        Additional local symbols (variables/parameters/time)
    
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
        implicit_multiplication_application,
    )
    
    return parse_expr(
        expr_str,
        local_dict=safe_local,
        transformations=transformations,
        evaluate=True,
    )


# =====================================================================
# Main ODE generation
# =====================================================================

def generate_ode_cpp(rhs_dict, params_list, num_type="AD", 
                     fixed_states=None, fixed_params=None):
    """
    Generate complete ODE system and Jacobian C++ code.
    
    Parameters
    ----------
    rhs_dict : dict
        Dictionary mapping state variable names to their RHS expressions (strings)
        Example: {"x": "v", "v": "mu*(1 - x**2)*v - x"}
    params_list : list of str
        List of parameter names
    num_type : str
        Numerical type for C++ code: "AD" or "double"
    fixed_states : list of str or None
        State variables excluded from sensitivity system
    fixed_params : list of str or None
        Parameters excluded from sensitivity system
        
    Returns
    -------
    dict with keys:
        - "ode_code": C++ code for ODE system struct
        - "jac_code": C++ code for Jacobian struct
        - "jac_matrix": Symbolic Jacobian (for R inspection)
        - "time_derivs": Symbolic time derivatives (for R inspection)
    """
    
    if fixed_states is None:
        fixed_states = []
    if fixed_params is None:
        fixed_params = []
    
    states_list = list(rhs_dict.keys())
    odes_list = list(rhs_dict.values())
    
    n_states = len(states_list)
    n_params = len(params_list)
    
    # --- Create SymPy symbols ---
    states_syms = {name: sp.Symbol(name, real=True) for name in states_list}
    params_syms = {name: sp.Symbol(name, real=True) for name in params_list}
    t = sp.Symbol('time', real=True)
    
    # Build local dictionary for parsing
    local_symbols = {}
    local_symbols.update(states_syms)
    local_symbols.update(params_syms)
    local_symbols['time'] = t
    
    # --- Parse all ODE expressions with safe parsing ---
    exprs = []
    for ode_str in odes_list:
        try:
            expr = _safe_sympify(ode_str, local_symbols)
            exprs.append(expr)
        except Exception as e:
            raise ValueError(f"Failed to parse ODE expression '{ode_str}': {e}")
    
    # --- Compute Jacobian matrix (all at once) ---
    states_syms_list = [states_syms[s] for s in states_list]
    jac_matrix = [[sp.diff(expr, state_sym) for state_sym in states_syms_list] 
                  for expr in exprs]
    
    # --- Compute time derivatives ---
    time_derivs = [sp.diff(expr, t) for expr in exprs]
    
    # --- Generate C++ code for ODE system ---
    ode_cpp_lines = [
        "// ODE system",
        "struct ode_system {",
        f"  ublas::vector<{num_type}> params;",
        f"  explicit ode_system(const ublas::vector<{num_type}>& p_) : params(p_) {{}}",
        f"  void operator()(const ublas::vector<{num_type}>& x, ublas::vector<{num_type}>& dxdt, const {num_type}& t) {{"
    ]
    
    for i, expr in enumerate(exprs):
        cpp_code = _to_cpp(expr, states_list, params_list, n_states)
        ode_cpp_lines.append(f"    dxdt[{i}] = {cpp_code};")
    
    ode_cpp_lines.extend(["  }", "};"])
    ode_code = "\n".join(ode_cpp_lines)
    
    # --- Generate C++ code for Jacobian ---
    jac_cpp_lines = [
        "// Jacobian for stiff solver",
        "struct jacobian {",
        f"  ublas::vector<{num_type}> params;",
        f"  explicit jacobian(const ublas::vector<{num_type}>& p_) : params(p_) {{}}",
        f"  void operator()(const ublas::vector<{num_type}>& x, ublas::matrix<{num_type}>& J, const {num_type}& t, ublas::vector<{num_type}>& dfdt) {{"
    ]
    
    # Fill Jacobian matrix
    for i in range(n_states):
        for j in range(n_states):
            cpp_code = _to_cpp(jac_matrix[i][j], states_list, params_list, n_states)
            jac_cpp_lines.append(f"    J({i},{j}) = {cpp_code};")
    
    # Fill time derivatives
    for i in range(n_states):
        cpp_code = _to_cpp(time_derivs[i], states_list, params_list, n_states)
        jac_cpp_lines.append(f"    dfdt[{i}] = {cpp_code};")
    
    jac_cpp_lines.extend(["  }", "};"])
    jac_code = "\n".join(jac_cpp_lines)
    
    # --- Return results ---
    return {
        "ode_code": ode_code,
        "jac_code": jac_code,
        "jac_matrix": [[str(entry) for entry in row] for row in jac_matrix],
        "time_derivs": [str(deriv) for deriv in time_derivs],
        "states": states_list,
        "params": params_list
    }


# =====================================================================
# Helper functions
# =====================================================================

def _to_cpp(expr, states, params, n_states):
    """
    Convert SymPy expression to C++ code with proper variable mapping.
    Always returns a string.
    """
    cpp_code = sp.cxxcode(expr, standard='c++17', strict=False)
    cpp_code = str(cpp_code)
    
    if 'Not supported in C++' in cpp_code:
        raise ValueError(f"Expression contains C++-unsupported constructs: {expr}")
    
    # Replace initial values (state_0 -> params[i])
    for i, state in enumerate(states):
        cpp_code = _safe_replace(cpp_code, f"{state}_0", f"params[{i}]")
    
    # Replace state variables (state -> x[i])
    for i, state in enumerate(states):
        cpp_code = _safe_replace(cpp_code, state, f"x[{i}]")
    
    # Replace parameters (param -> params[n_states + j])
    for j, param in enumerate(params):
        cpp_code = _safe_replace(cpp_code, param, f"params[{n_states + j}]")
    
    # Replace time
    cpp_code = _safe_replace(cpp_code, "time", "t")
    
    # Cleanup whitespace
    cpp_code = cpp_code.replace(" ", "")
    
    return cpp_code


def _safe_replace(text, symbol, replacement):
    """Replace symbol as whole word only"""
    text = str(text)
    pattern = r'\b' + re.escape(symbol) + r'\b'
    return re.sub(pattern, replacement, text)


def _get_list_value(dict_or_df, key, index, n_events):
    """
    Safely extract value from dict/DataFrame at given index.
    Handles both list-like and scalar values from R.
    """
    if key not in dict_or_df:
        return None
    
    value = dict_or_df[key]
    
    # If it's a list/array, get the indexed value
    if isinstance(value, (list, tuple)):
        if index < len(value):
            return value[index]
        return None
    
    # If it's a scalar (from R single-row df), return it for index 0
    if index == 0:
        return value
    
    # Otherwise assume it's the same for all events
    return value


def _is_valid_value(value):
    """Check if value is valid (not None/NaN/NA)"""
    if value is None:
        return False
    
    str_val = str(value).lower().strip()
    return str_val not in ['none', 'nan', 'na', '']


def _parse_value_or_expression(value, local_symbols, states_list, params_list, n_states):
    """
    Parse a value that can be either numeric or an expression.
    Always returns a string of C++ code or None.
    
    Parameters
    ----------
    value : str or numeric
        Value to parse
    local_symbols : dict
        Dictionary of SymPy symbols for parsing
    states_list : list of str
    params_list : list of str
    n_states : int
    
    Returns
    -------
    str or None
        C++ code string or None if invalid
    """
    if not _is_valid_value(value):
        return None
    
    value_str = str(value).strip()
    
    # Check if it's a simple number
    try:
        float(value_str)
        return value_str
    except (ValueError, TypeError):
        pass
    
    # It's an expression - parse it with safe parsing
    try:
        value_expr = _safe_sympify(value_str, local_symbols)
        value_code = _to_cpp(value_expr, states_list, params_list, n_states)
        return str(value_code)
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{value_str}': {e}")


# =====================================================================
# Event code generation
# =====================================================================

def generate_event_code(events_df, states_list, params_list, n_states, num_type="AD"):
    """
    Generate C++ code for events (fixed-time and root-triggered).
    
    Parameters
    ----------
    events_df : pandas.DataFrame or dict of lists
        Event specifications with columns: var, value, method, time (optional), root (optional)
    states_list : list of str
    params_list : list of str
    n_states : int
    num_type : str
        "AD" or "double"
        
    Returns
    -------
    list of str
        C++ code lines for event initialization
    """
    
    if events_df is None or len(events_df) == 0:
        return []
    
    # Convert pandas DataFrame to dict if needed
    if hasattr(events_df, 'to_dict'):
        events_dict = events_df.to_dict('list')
    else:
        events_dict = events_df
    
    # Create SymPy symbols for parsing
    states_syms = {name: sp.Symbol(name, real=True) for name in states_list}
    params_syms = {name: sp.Symbol(name, real=True) for name in params_list}
    t = sp.Symbol('time', real=True)
    
    local_symbols = {}
    local_symbols.update(states_syms)
    local_symbols.update(params_syms)
    local_symbols['time'] = t
    
    event_lines = []
    
    # Determine number of events - robust handling
    if 'var' in events_dict:
        var_val = events_dict['var']
        if isinstance(var_val, (list, tuple)):
            n_events = len(var_val)
        else:
            n_events = 1
    else:
        return []
    
    for i in range(n_events):
        # Get var name
        var_raw = _get_list_value(events_dict, 'var', i, n_events)
        if var_raw is None:
            continue
        var_name = str(var_raw)
        
        if var_name not in states_list:
            raise ValueError(f"Event {i}: unknown state variable '{var_name}'")
        
        var_idx = states_list.index(var_name)
        
        # Parse value expression
        value_raw = _get_list_value(events_dict, 'value', i, n_events)
        value_code = _parse_value_or_expression(
            value_raw, local_symbols, states_list, params_list, n_states
        )
        
        if value_code is None:
            raise ValueError(f"Event {i}: value is required but was None/NA")
        
        value_code = str(value_code).replace("params[", "full_params[")
        
        # Determine method
        method_raw = _get_list_value(events_dict, 'method', i, n_events)
        method = str(method_raw).lower() if method_raw else 'replace'
        method_map = {
            'replace': 'EventMethod::Replace',
            'add': 'EventMethod::Add',
            'multiply': 'EventMethod::Multiply'
        }
        method_code = method_map.get(method, 'EventMethod::Replace')
        
        # Parse time if present
        time_raw = _get_list_value(events_dict, 'time', i, n_events)
        time_code = _parse_value_or_expression(
            time_raw, local_symbols, states_list, params_list, n_states
        )
        
        # Parse root if present
        root_raw = _get_list_value(events_dict, 'root', i, n_events)
        root_code = _parse_value_or_expression(
            root_raw, local_symbols, states_list, params_list, n_states
        )
        
        # Generate C++ code based on event type
        if time_code is not None:
            # Fixed-time event
            time_code = str(time_code).replace("params[", "full_params[")
            event_lines.append(
                f"  fixed_events.emplace_back(FixedEvent<{num_type}>{{{time_code}, {var_idx}, {value_code}, {method_code}}});"
            )
        
        elif root_code is not None:
            # Root-triggered event
            root_code = str(root_code).replace("params[", "full_params[")
            event_lines.append(
                f"  root_events.push_back(RootEvent<ublas::vector<{num_type}>, {num_type}>{{"
            )
            event_lines.append(
                f"    [](const ublas::vector<{num_type}>& x, const {num_type}& t){{ return {root_code}; }},"
            )
            event_lines.append(
                f"    {var_idx}, {value_code}, {method_code}}});"
            )
        else:
            raise ValueError(f"Event {i}: must specify either 'time' or 'root'")
    
    return event_lines
