"""
ODE and Jacobian C++ code generator for CppODE

This module generates optimized C++ code for ODE systems with automatic
differentiation support. It handles symbolic differentiation via SymPy
and produces C++ code compatible with Boost.Odeint and FADBAD++.

Author: CppODE Development Team
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    convert_xor, 
    implicit_multiplication_application
)
import re


def generate_ode_cpp(odes_dict, params_list, num_type="AD", 
                     fixed_states=None, fixed_params=None):
    """
    Generate complete ODE system and Jacobian C++ code.
    
    Parameters
    ----------
    odes_dict : dict
        Dictionary mapping state variable names to their RHS expressions (strings).
        Example: {"x": "v", "v": "mu*(1 - x**2)*v - x"}
    params_list : list of str
        List of parameter names extracted from the ODE expressions.
    num_type : str, optional
        Numerical type for C++ code generation. Options:
        - "double": Standard floating point (no sensitivities)
        - "AD": First-order automatic differentiation (F<double>)
        - "AD2": Second-order automatic differentiation (F<F<double>>)
        Default is "AD".
    fixed_states : list of str, optional
        State variables to exclude from sensitivity calculations.
        Default is None (all states included).
    fixed_params : list of str, optional
        Parameters to exclude from sensitivity calculations.
        Default is None (all parameters included).
        
    Returns
    -------
    dict
        Dictionary containing:
        - "ode_code" : str
            C++ code for the ODE system struct
        - "jac_code" : str
            C++ code for the Jacobian struct
        - "jac_matrix" : list of list of str
            Symbolic Jacobian matrix entries (for R inspection)
        - "time_derivs" : list of str
            Symbolic time derivatives (for R inspection)
        - "states" : list of str
            State variable names (echoed back)
        - "params" : list of str
            Parameter names (echoed back)
    """
    
    # Initialize optional parameters
    if fixed_states is None:
        fixed_states = []
    if fixed_params is None:
        fixed_params = []
    
    # Extract ordered lists from input dictionary
    states_list = list(odes_dict.keys())
    odes_list = list(odes_dict.values())
    
    n_states = len(states_list)
    n_params = len(params_list)
    
    # Create SymPy symbols for all variables
    states_syms = {name: sp.Symbol(name, real=True) for name in states_list}
    params_syms = {name: sp.Symbol(name, real=True) for name in params_list}
    t = sp.Symbol('time', real=True)
    
    # Build local namespace for expression parsing
    local_dict = {}
    local_dict.update(states_syms)
    local_dict.update(params_syms)
    local_dict['time'] = t
    
    # Configure SymPy parser transformations
    transformations = standard_transformations + (
        convert_xor,  # Convert ^ to ** for exponentiation
        implicit_multiplication_application  # Allow implicit multiplication
    )
    
    # Parse all ODE right-hand sides into SymPy expressions
    exprs = []
    for ode_str in odes_list:
        try:
            expr = parse_expr(
                ode_str, 
                local_dict=local_dict, 
                transformations=transformations,
                evaluate=True
            )
            exprs.append(expr)
        except Exception as e:
            raise ValueError(f"Failed to parse ODE expression '{ode_str}': {e}")
    
    # Compute Jacobian matrix: J[i,j] = d(f_i)/d(x_j)
    states_syms_list = [states_syms[s] for s in states_list]
    jac_matrix = [[sp.diff(expr, state_sym) for state_sym in states_syms_list] 
                  for expr in exprs]
    
    # Compute time derivatives: df/dt for implicit methods
    time_derivs = [sp.diff(expr, t) for expr in exprs]
    
    # Generate C++ code for ODE system functor
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
    
    # Generate C++ code for Jacobian functor
    jac_cpp_lines = [
        "// Jacobian for stiff solver",
        "struct jacobian {",
        f"  ublas::vector<{num_type}> params;",
        f"  explicit jacobian(const ublas::vector<{num_type}>& p_) : params(p_) {{}}",
        f"  void operator()(const ublas::vector<{num_type}>& x, ublas::matrix<{num_type}>& J, const {num_type}& t, ublas::vector<{num_type}>& dfdt) {{"
    ]
    
    # Fill Jacobian matrix entries
    for i in range(n_states):
        for j in range(n_states):
            cpp_code = _to_cpp(jac_matrix[i][j], states_list, params_list, n_states)
            jac_cpp_lines.append(f"    J({i},{j}) = {cpp_code};")
    
    # Fill time derivative vector
    for i in range(n_states):
        cpp_code = _to_cpp(time_derivs[i], states_list, params_list, n_states)
        jac_cpp_lines.append(f"    dfdt[{i}] = {cpp_code};")
    
    jac_cpp_lines.extend(["  }", "};"])
    jac_code = "\n".join(jac_cpp_lines)
    
    # Return all generated code and metadata
    return {
        "ode_code": ode_code,
        "jac_code": jac_code,
        "jac_matrix": [[str(entry) for entry in row] for row in jac_matrix],
        "time_derivs": [str(deriv) for deriv in time_derivs],
        "states": states_list,
        "params": params_list
    }


def _to_cpp(expr, states, params, n_states):
    """
    Convert a SymPy expression to C++ code with proper variable indexing.
    
    This function maps symbolic variables to their C++ representations:
    - State variables (A, B, ...) -> x[0], x[1], ...
    - Initial conditions (A_0, B_0, ...) -> params[0], params[1], ...
    - Parameters (k1, k2, ...) -> params[n_states], params[n_states+1], ...
    - Time variable (time) -> t
    
    The implementation uses a two-phase approach to prevent double-replacement:
    1. Replace all symbols with unique temporary placeholders
    2. Replace placeholders with final C++ code
    
    Parameters
    ----------
    expr : sympy.Basic or numeric
        SymPy expression to convert
    states : list of str
        Ordered list of state variable names
    params : list of str
        Ordered list of parameter names
    n_states : int
        Number of state variables (used for parameter indexing offset)
        
    Returns
    -------
    str
        C++ code string with proper variable indexing and whitespace removed
    """
    
    # Handle trivial cases
    if not isinstance(expr, sp.Basic):
        return str(expr)
    
    if expr.is_number:
        return str(expr)
    
    # Convert SymPy expression to C++ syntax
    cpp_code = sp.cxxcode(expr, standard='c++17', strict=False)
    cpp_code = str(cpp_code)
    
    if 'Not supported in C++' in cpp_code:
        raise ValueError(f"Expression contains C++-unsupported constructs: {expr}")
    
    # Build symbol mapping with temporary placeholders
    # Format: (pattern, temporary_placeholder, final_code)
    symbol_map = []
    
    # Initial state values: A_0 -> params[0]
    for i, state in enumerate(states):
        pattern = f"{state}_0"
        placeholder = f"__XINIT{i}__"
        final_code = f"params[{i}]"
        symbol_map.append((pattern, placeholder, final_code))
    
    # Current state values: A -> x[0]
    for i, state in enumerate(states):
        pattern = state
        placeholder = f"__XSTATE{i}__"
        final_code = f"x[{i}]"
        symbol_map.append((pattern, placeholder, final_code))
    
    # Parameters: k1 -> params[n_states + 0]
    for j, param in enumerate(params):
        pattern = param
        placeholder = f"__XPARAM{j}__"
        final_code = f"params[{n_states + j}]"
        symbol_map.append((pattern, placeholder, final_code))
    
    # Time variable: time -> t
    symbol_map.append(("time", "__XTIME__", "t"))
    
    # Sort by pattern length (longest first) to handle overlapping symbol names
    # Example: replace "k10" before "k1" to avoid partial matches
    symbol_map.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Phase 1: Replace symbols with temporary placeholders using regex word boundaries
    for pattern, placeholder, _ in symbol_map:
        regex_pattern = r'\b' + re.escape(pattern) + r'\b'
        cpp_code = re.sub(regex_pattern, placeholder, cpp_code)
    
    # Phase 2: Replace placeholders with final C++ code (simple string replacement)
    for _, placeholder, final_code in symbol_map:
        cpp_code = cpp_code.replace(placeholder, final_code)
    
    # Remove all whitespace for compact code
    cpp_code = cpp_code.replace(" ", "")
    
    return cpp_code


def _get_list_value(dict_or_df, key, index, n_events):
    """
    Extract a value from a dictionary or DataFrame at a specific index.
    
    Handles both R-style DataFrames (converted to dict of lists) and
    single-row DataFrames (converted to dict of scalars).
    
    Parameters
    ----------
    dict_or_df : dict
        Dictionary with keys mapping to lists or scalar values
    key : str
        Key to extract
    index : int
        Index within the list (ignored for scalar values)
    n_events : int
        Total number of events (unused, kept for API consistency)
        
    Returns
    -------
    value or None
        The extracted value, or None if key doesn't exist or index out of bounds
    """
    if key not in dict_or_df:
        return None
    
    value = dict_or_df[key]
    
    # Handle list-like values
    if isinstance(value, (list, tuple)):
        if index < len(value):
            return value[index]
        return None
    
    # Handle scalar values (single-row DataFrame case)
    if index == 0:
        return value
    
    # Default: assume scalar applies to all events
    return value


def _is_valid_value(value):
    """
    Check if a value is valid (not None, NaN, or NA).
    
    Parameters
    ----------
    value : any
        Value to check
        
    Returns
    -------
    bool
        True if value is valid, False otherwise
    """
    if value is None:
        return False
    
    str_val = str(value).lower().strip()
    return str_val not in ['none', 'nan', 'na', '']


def _parse_value_or_expression(value, local_dict, transformations, 
                                states_list, params_list, n_states):
    """
    Parse a value that can be either numeric or a symbolic expression.
    
    Handles three cases:
    1. Numeric literals: "5.0" -> "5.0"
    2. Parameter references: "k1" -> "params[1]"
    3. Expressions: "2*k1 + A" -> "2*params[1]+x[0]"
    
    Parameters
    ----------
    value : any
        Value to parse (typically string, but can be numeric)
    local_dict : dict
        SymPy symbol namespace for expression parsing
    transformations : tuple
        SymPy parser transformations
    states_list : list of str
        Ordered state variable names
    params_list : list of str
        Ordered parameter names
    n_states : int
        Number of state variables
        
    Returns
    -------
    str or None
        C++ code string, or None if value is invalid
    """
    if not _is_valid_value(value):
        return None
    
    value_str = str(value).strip()
    
    # Check if value is a simple numeric literal
    try:
        float(value_str)
        return value_str
    except (ValueError, TypeError):
        pass
    
    # Parse as symbolic expression and convert to C++
    try:
        value_expr = parse_expr(
            value_str, 
            local_dict=local_dict, 
            transformations=transformations
        )
        value_code = _to_cpp(value_expr, states_list, params_list, n_states)
        return str(value_code)
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{value_str}': {e}")


def generate_event_code(events_df, states_list, params_list, n_states, num_type="AD"):
    """
    Generate C++ code for fixed-time and root-triggered events.
    
    Events can modify state variables at specific times or when a condition
    (root function) crosses zero.
    
    Parameters
    ----------
    events_df : pandas.DataFrame or dict of lists
        Event specifications with columns:
        - var : str
            Name of state variable to modify
        - value : numeric or str
            New value or expression to apply
        - method : str
            How to apply value: "replace", "add", or "multiply"
        - time : numeric or str, optional
            Fixed time point for event (mutually exclusive with root)
        - root : str, optional
            Root function expression (mutually exclusive with time)
    states_list : list of str
        Ordered state variable names
    params_list : list of str
        Ordered parameter names
    n_states : int
        Number of state variables
    num_type : str, optional
        Numerical type ("double", "AD", or "AD2"). Default is "AD".
        
    Returns
    -------
    list of str
        C++ code lines for event initialization (to be inserted in solver)
    """
    
    if events_df is None or len(events_df) == 0:
        return []
    
    # Convert pandas DataFrame to dict of lists if necessary
    if hasattr(events_df, 'to_dict'):
        events_dict = events_df.to_dict('list')
    else:
        events_dict = events_df
    
    # Create SymPy symbol namespace for expression parsing
    states_syms = {name: sp.Symbol(name, real=True) for name in states_list}
    params_syms = {name: sp.Symbol(name, real=True) for name in params_list}
    t = sp.Symbol('time', real=True)
    
    local_dict = {}
    local_dict.update(states_syms)
    local_dict.update(params_syms)
    local_dict['time'] = t
    
    transformations = standard_transformations + (
        convert_xor, 
        implicit_multiplication_application
    )
    
    event_lines = []
    
    # Determine number of events from DataFrame structure
    if 'var' in events_dict:
        var_val = events_dict['var']
        if isinstance(var_val, (list, tuple)):
            n_events = len(var_val)
        else:
            n_events = 1
    else:
        return []
    
    # Generate code for each event
    for i in range(n_events):
        # Extract and validate target variable
        var_raw = _get_list_value(events_dict, 'var', i, n_events)
        if var_raw is None:
            continue
        var_name = str(var_raw)
        
        if var_name not in states_list:
            raise ValueError(f"Event {i}: unknown state variable '{var_name}'")
        
        var_idx = states_list.index(var_name)
        
        # Parse value expression or literal
        value_raw = _get_list_value(events_dict, 'value', i, n_events)
        value_code = _parse_value_or_expression(
            value_raw, local_dict, transformations, 
            states_list, params_list, n_states
        )
        
        if value_code is None:
            raise ValueError(f"Event {i}: value is required but was None/NA")
        
        # Replace params[] with full_params[] for event context
        value_code = str(value_code).replace("params[", "full_params[")
        
        # Map method string to C++ enum
        method_raw = _get_list_value(events_dict, 'method', i, n_events)
        method = str(method_raw).lower() if method_raw else 'replace'
        method_map = {
            'replace': 'EventMethod::Replace',
            'add': 'EventMethod::Add',
            'multiply': 'EventMethod::Multiply'
        }
        method_code = method_map.get(method, 'EventMethod::Replace')
        
        # Parse optional time specification (for fixed-time events)
        time_raw = _get_list_value(events_dict, 'time', i, n_events)
        time_code = _parse_value_or_expression(
            time_raw, local_dict, transformations, 
            states_list, params_list, n_states
        )
        
        # Parse optional root specification (for root-triggered events)
        root_raw = _get_list_value(events_dict, 'root', i, n_events)
        root_code = _parse_value_or_expression(
            root_raw, local_dict, transformations, 
            states_list, params_list, n_states
        )
        
        # Generate C++ code based on event type
        if time_code is not None:
            # Fixed-time event: occurs at specified time point
            time_code = str(time_code).replace("params[", "full_params[")
            event_lines.append(
                f"  fixed_events.emplace_back(FixedEvent<{num_type}>{{{time_code}, {var_idx}, {value_code}, {method_code}}});"
            )
        
        elif root_code is not None:
            # Root-triggered event: occurs when root function crosses zero
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
