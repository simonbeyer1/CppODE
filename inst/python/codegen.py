"""
ODE and Jacobian C++ code generator for CppODE
==============================================

This module provides two main entry points:

- generate_ode_cpp(...)
    Generates C++ code for the ODE right-hand side and its Jacobian,
    using SymPy for parsing and symbolic differentiation.

- generate_event_code(...)
    Generates C++ code for fixed-time and root-triggered events,
    based on an R-style events data frame (converted to a dict).

Both functions are used from the R side by CppODE to generate
self-contained C++ solvers with Boost.Odeint and FADBAD++.
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
    """
    Construct a local dictionary for SymPy parsing that avoids conflicts
    with SymPy singletons and exposes a controlled set of functions.

    This ensures that expression strings originating from R are parsed
    into SymPy expressions in a predictable and safe way.
    """
    safe_local_dict = {
        # Override problematic SymPy singletons: treat as ordinary symbols
        "S": sp.Symbol("S"),
        "I": sp.Symbol("I"),
        "N": sp.Symbol("N"),
        "O": sp.Symbol("O"),
        "Q": sp.Symbol("Q"),
        "C": sp.Symbol("C"),
        # Exponential and logarithmic functions
        "exp": sp.exp,
        "exp10": lambda x: sp.Pow(10, x),
        "exp2": lambda x: sp.Pow(2, x),
        "log": sp.log,
        "ln": sp.log,
        "log10": lambda x: sp.log(x, 10),
        "log2": lambda x: sp.log(x, 2),
        # Trigonometric functions
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "cot": sp.cot,
        "sec": sp.sec,
        "csc": sp.csc,
        # Inverse trigonometric functions
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "acot": sp.acot,
        "asec": sp.asec,
        "acsc": sp.acsc,
        "atan2": sp.atan2,
        # Hyperbolic functions
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "coth": sp.coth,
        "sech": sp.sech,
        "csch": sp.csch,
        # Inverse hyperbolic functions
        "asinh": sp.asinh,
        "acosh": sp.acosh,
        "atanh": sp.atanh,
        "acoth": sp.acoth,
        "asech": sp.asech,
        "acsch": sp.acsch,
        # Power and root functions
        "sqrt": sp.sqrt,
        "cbrt": sp.cbrt,
        "root": sp.root,
        "pow": sp.Pow,
        # Absolute value and sign
        "abs": sp.Abs,
        "sign": sp.sign,
        # Rounding functions
        "floor": sp.floor,
        "ceiling": sp.ceiling,
        # Min/Max
        "min": sp.Min,
        "max": sp.Max,
        # Factorial and gamma functions
        "factorial": sp.factorial,
        "gamma": sp.gamma,
        "loggamma": sp.loggamma,
        "digamma": sp.digamma,
        # Error functions
        "erf": sp.erf,
        "erfc": sp.erfc,
        # Bessel functions
        "besselj": sp.besselj,
        "bessely": sp.bessely,
        "besseli": sp.besseli,
        "besselk": sp.besselk,
        # Special functions
        "Heaviside": sp.Heaviside,
        "DiracDelta": sp.DiracDelta,
        # Piecewise
        "Piecewise": sp.Piecewise,
        # Constants
        "pi": sp.pi,
        "E": sp.E,
        "oo": sp.oo,
    }
    return safe_local_dict


def _safe_sympify(expr_str, local_symbols=None):
    """
    Safely parse a string expression into a SymPy expression.

    This uses a restricted local dictionary to prevent conflicts with
    SymPy singletons and applies a standard set of transformations
    (e.g. handling "^" vs "**").

    Parameters
    ----------
    expr_str : str
        Expression string from R.
    local_symbols : dict, optional
        Additional symbols (states, parameters, time).

    Returns
    -------
    sp.Expr
        Parsed expression.
    """
    expr_str = str(expr_str).strip()

    # Fast-path for the zero literal
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


# =====================================================================
# Main ODE generation
# =====================================================================


def generate_ode_cpp(
    rhs_dict,
    params_list,
    num_type="AD",
    fixed_states=None,
    fixed_params=None,
):
    """
    Generate C++ code for the ODE system and its Jacobian.

    Parameters
    ----------
    rhs_dict : dict
        Mapping from state names to RHS strings, e.g.
        {"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}.
    params_list : list of str
        Names of model parameters (order matches R).
    num_type : str
        Numeric type used in C++ code: "double", "AD", or "AD2".
        The caller chooses consistent with how the solver is generated.
    fixed_states : list of str, optional
        Names of states excluded from sensitivity seeding (not used
        directly here, but passed in for potential future extensions).
    fixed_params : list of str, optional
        Names of parameters excluded from sensitivities.

    Returns
    -------
    dict
        {
          "ode_code": C++ code for the ODE struct,
          "jac_code": C++ code for the Jacobian struct,
          "jac_matrix": [[str, ...], ...] symbolic Jacobian entries,
          "time_derivs": [str, ...] time derivatives of each RHS,
          "states": [state names],
          "params": [parameter names],
        }
    """
    if fixed_states is None:
        fixed_states = []
    if fixed_params is None:
        fixed_params = []

    states_list = list(rhs_dict.keys())
    odes_list = list(rhs_dict.values())

    n_states = len(states_list)
    n_params = len(params_list)

    # SymPy symbols for states/parameters/time
    states_syms = {name: sp.Symbol(name, real=True) for name in states_list}
    params_syms = {name: sp.Symbol(name, real=True) for name in params_list}
    t = sp.Symbol("time", real=True)

    local_symbols = {}
    local_symbols.update(states_syms)
    local_symbols.update(params_syms)
    local_symbols["time"] = t

    # Parse RHS expressions
    exprs = []
    for ode_str in odes_list:
        try:
            expr = _safe_sympify(ode_str, local_symbols)
            exprs.append(expr)
        except Exception as e:
            raise ValueError(f"Failed to parse ODE expression '{ode_str}': {e}")

    states_syms_list = [states_syms[s] for s in states_list]

    # Symbolic Jacobian
    jac_matrix = [
        [sp.diff(expr, state_sym) for state_sym in states_syms_list]
        for expr in exprs
    ]

    # Time derivatives of RHS
    time_derivs = [sp.diff(expr, t) for expr in exprs]

    # -----------------------------------------------------------------
    # Generate C++: ODE system
    # -----------------------------------------------------------------
    ode_cpp_lines = [
        "// ODE system",
        "struct ode_system {",
        f"  ublas::vector<{num_type}> params;",
        f"  explicit ode_system(const ublas::vector<{num_type}>& p_) : params(p_) {{}}",
        f"  void operator()(const ublas::vector<{num_type}>& x, "
        f"ublas::vector<{num_type}>& dxdt, const {num_type}& t) {{",
    ]

    for i, expr in enumerate(exprs):
        cpp_code = _to_cpp(expr, states_list, params_list, n_states)
        ode_cpp_lines.append(f"    dxdt[{i}] = {cpp_code};")

    ode_cpp_lines.extend(["  }", "};"])
    ode_code = "\n".join(ode_cpp_lines)

    # -----------------------------------------------------------------
    # Generate C++: Jacobian
    # -----------------------------------------------------------------
    jac_cpp_lines = [
        "// Jacobian for stiff solver",
        "struct jacobian {",
        f"  ublas::vector<{num_type}> params;",
        f"  explicit jacobian(const ublas::vector<{num_type}>& p_) : params(p_) {{}}",
        f"  void operator()(const ublas::vector<{num_type}>& x, "
        f"ublas::matrix<{num_type}>& J, const {num_type}& t, "
        f"ublas::vector<{num_type}>& dfdt) {{",
    ]

    # Fill Jacobian entries
    for i in range(n_states):
        for j in range(n_states):
            cpp_code = _to_cpp(jac_matrix[i][j], states_list, params_list, n_states)
            jac_cpp_lines.append(f"    J({i},{j}) = {cpp_code};")

    # Fill df/dt
    for i in range(n_states):
        cpp_code = _to_cpp(time_derivs[i], states_list, params_list, n_states)
        jac_cpp_lines.append(f"    dfdt[{i}] = {cpp_code};")

    jac_cpp_lines.extend(["  }", "};"])
    jac_code = "\n".join(jac_cpp_lines)

    return {
        "ode_code": ode_code,
        "jac_code": jac_code,
        "jac_matrix": [[str(entry) for entry in row] for row in jac_matrix],
        "time_derivs": [str(deriv) for deriv in time_derivs],
        "states": states_list,
        "params": params_list,
    }


# =====================================================================
# Helper functions
# =====================================================================


def _to_cpp(expr, states, params, n_states):
    """
    Convert a SymPy expression into a C++ expression string.

    This uses SymPy's cxxcode generator and then rewrites symbolic
    names into the solver's C++ memory layout:

    - parameters:  params[n_states + j]
    - states:      x[i]
    - initial values (state_0): params[i]
    - time:        t

    The order of replacements is important to avoid accidental
    partial matches (e.g. "A" inside "A0").
    """
    cpp_code = sp.cxxcode(expr, standard="c++17", strict=False)
    cpp_code = str(cpp_code)

    if "Not supported in C++" in cpp_code:
        raise ValueError(f"Expression contains C++-unsupported constructs: {expr}")

    # 1) Replace parameters first
    for j, param in enumerate(params):
        cpp_code = _safe_replace(cpp_code, param, f"params[{n_states + j}]")

    # 2) Replace state variables (A, B, ...) with x[i]
    for i, state in enumerate(states):
        cpp_code = _safe_replace(cpp_code, state, f"x[{i}]")

    # 3) Replace initial values (A_0, B_0, ...) with params[i]
    for i, state in enumerate(states):
        cpp_code = _safe_replace(cpp_code, f"{state}_0", f"params[{i}]")

    # 4) Replace time
    cpp_code = _safe_replace(cpp_code, "time", "t")

    # Remove superfluous spaces
    cpp_code = cpp_code.replace(" ", "")

    return cpp_code


def _safe_replace(text, symbol, replacement):
    """
    Replace `symbol` in `text` only when it appears as a full token.

    Implemented via a word-boundary regular expression:
    - avoids replacing substrings inside other identifiers.
    """
    text = str(text)
    pattern = r"\b" + re.escape(symbol) + r"\b"
    return re.sub(pattern, replacement, text)


def _get_list_value(dict_or_df, key, index, n_events):
    """
    Extract a value from a dict-of-lists that mirrors an R data frame.

    Parameters
    ----------
    dict_or_df : dict
        Typically events_df.to_dict('list') from pandas, or a plain dict.
    key : str
        Column name.
    index : int
        Row index (0-based).
    n_events : int
        Total number of events.

    Returns
    -------
    Any or None
        The value at (key, index), or None if not present.
    """
    if key not in dict_or_df:
        return None

    value = dict_or_df[key]

    # True list-like column
    if isinstance(value, (list, tuple)):
        if index < len(value):
            return value[index]
        return None

    # Scalar column: same for all events
    return value


def _is_valid_value(value):
    """
    Check whether a value from the events table is a meaningful numeric
    or expression, as opposed to NA/None/boolean placeholders.

    Returns False for:
    - None
    - booleans (True/False)
    - textual NA-like markers ("NA", "NaN", "")
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return False

    # If it's a numeric type, accept unless NaN
    if isinstance(value, numbers.Number):
        try:
            # NaN check: NaN != NaN
            if value != value:
                return False
        except Exception:
            pass
        return True

    # String-like handling
    str_val = str(value).lower().strip()
    if str_val in {"none", "nan", "na", ""}:
        return False
    if str_val in {"true", "false"}:
        return False

    return True


def _parse_value_or_expression(value, local_symbols, states_list, params_list, n_states):
    """
    Interpret a value from the events data as either a numeric literal
    or a symbolic expression.

    Parameters
    ----------
    value : Any
        Source value (string, numeric, etc.) from events_df.
    local_symbols : dict
        Symbol table for SymPy parsing (states, params, time).
    states_list : list of str
        State variable names.
    params_list : list of str
        Parameter names.
    n_states : int
        Number of states.

    Returns
    -------
    str or None
        C++ expression as a string, or None if value is not valid.
    """
    if not _is_valid_value(value):
        return None

    value_str = str(value).strip()

    # Fast-path: numeric literal (including scientific notation)
    try:
        float(value_str)
        return value_str
    except (ValueError, TypeError):
        pass

    # Otherwise treat as expression
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
    Generate C++ initialization lines for fixed-time and root-triggered events.

    The input mirrors an R data.frame with columns (as strings):

      var   : affected state variable name (required)
      value : value or expression applied at the event (required)
      method: "replace", "add", or "multiply" (optional, default "replace")
      time  : numeric time or expression (optional)
      root  : root expression in states/time (optional)

    For each row i:

      - if time_i is non-NA/non-empty, a fixed-time event is generated.
      - else if root_i is non-NA/non-empty, a root-triggered event is generated.
      - else an error is raised.

    Parameters
    ----------
    events_df : pandas.DataFrame or dict-like
        Event specifications.
    states_list : list of str
        Names of state variables (matching rhs_dict).
    params_list : list of str
        Names of parameters (matching params_list in generate_ode_cpp).
    n_states : int
        Number of states.
    num_type : str
        Numeric type used in C++ (e.g., "double", "AD", "AD2").

    Returns
    -------
    list of str
        C++ code lines that append to `fixed_events` and `root_events`.
    """
    if events_df is None or len(events_df) == 0:
        return []

    # Convert pandas DataFrame to dict of lists, if necessary
    if hasattr(events_df, "to_dict"):
        events_dict = events_df.to_dict("list")
    else:
        events_dict = events_df

    states_syms = {name: sp.Symbol(name, real=True) for name in states_list}
    params_syms = {name: sp.Symbol(name, real=True) for name in params_list}
    t = sp.Symbol("time", real=True)

    local_symbols = {}
    local_symbols.update(states_syms)
    local_symbols.update(params_syms)
    local_symbols["time"] = t

    event_lines = []

    # Determine number of events robustly: maximum list-length over columns
    list_lengths = []
    for v in events_dict.values():
        if isinstance(v, (list, tuple)):
            list_lengths.append(len(v))
    n_events = max(list_lengths) if list_lengths else 1

    for i in range(n_events):
        # --------------------------------------------------------------
        # Affected variable
        # --------------------------------------------------------------
        var_raw = _get_list_value(events_dict, "var", i, n_events)
        if var_raw is None:
            # silently skip malformed rows
            continue

        var_name = str(var_raw)
        if var_name not in states_list:
            raise ValueError(f"Event {i}: unknown state variable '{var_name}'")
        var_idx = states_list.index(var_name)

        # --------------------------------------------------------------
        # Value to apply
        # --------------------------------------------------------------
        value_raw = _get_list_value(events_dict, "value", i, n_events)
        value_code = _parse_value_or_expression(
            value_raw, local_symbols, states_list, params_list, n_states
        )
        if value_code is None:
            raise ValueError(f"Event {i}: 'value' is required but is NA/None")

        # Map params[...] (symbolic) to full_params[...] (runtime C++ vector)
        value_code = str(value_code).replace("params[", "full_params[")

        # --------------------------------------------------------------
        # Method (replace/add/multiply)
        # --------------------------------------------------------------
        method_raw = _get_list_value(events_dict, "method", i, n_events)
        method = str(method_raw).lower() if method_raw is not None else "replace"
        method_map = {
            "replace": "EventMethod::Replace",
            "add": "EventMethod::Add",
            "multiply": "EventMethod::Multiply",
        }
        method_code = method_map.get(method, "EventMethod::Replace")

        # --------------------------------------------------------------
        # Time and root expressions
        # --------------------------------------------------------------
        time_raw = _get_list_value(events_dict, "time", i, n_events)
        time_code = _parse_value_or_expression(
            time_raw, local_symbols, states_list, params_list, n_states
        )
        if time_raw is not None and time_code is None:
            # A column existed but contained NA/boolean/empty -> treat as absent
            time_code = None

        root_raw = _get_list_value(events_dict, "root", i, n_events)
        root_code = _parse_value_or_expression(
            root_raw, local_symbols, states_list, params_list, n_states
        )
        if root_raw is not None and root_code is None:
            root_code = None

        # --------------------------------------------------------------
        # Generate C++ for the event
        #   - fixed-time if a valid time is present
        #   - root-triggered if time is absent but root is present
        # --------------------------------------------------------------
        if time_code is not None:
            # Fixed-time event
            time_code = str(time_code).replace("params[", "full_params[")
            event_lines.append(
                f"  fixed_events.emplace_back(FixedEvent<{num_type}>{{"
                f"{time_code}, {var_idx}, {value_code}, {method_code}}});"
            )

        elif root_code is not None:
            # Root-triggered event
            root_code = str(root_code).replace("params[", "full_params[")
            event_lines.append(
                f"  root_events.push_back("
                f"RootEvent<ublas::vector<{num_type}>, {num_type}>{{"
            )
            event_lines.append(
                f"    [](const ublas::vector<{num_type}>& x, "
                f"const {num_type}& t){{ return {root_code}; }},"
            )
            event_lines.append(
                f"    {var_idx}, {value_code}, {method_code}}});"
            )

        else:
            # Neither time nor root provided in a meaningful way
            raise ValueError(f"Event {i}: must specify either 'time' or 'root'")

    return event_lines
