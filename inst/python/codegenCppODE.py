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

# =====================================================================
# Safe parsing configuration
# =====================================================================

def _get_safe_parse_dict():
    """Construct a local dictionary for SymPy parsing."""
    safe_local_dict = {
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
    return safe_local_dict


def _safe_sympify(expr_str, local_symbols=None):
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
    pattern = r"\b" + re.escape(symbol) + r"\b"
    return re.sub(pattern, replacement, str(text))


# =====================================================================
# _to_cpp
# =====================================================================

def _to_cpp(expr, states, params, n_states, num_type, forcings=None):
    if forcings is None:
        forcings = []

    cpp_code = str(sp.cxxcode(expr, standard="c++17", strict=False)).replace("\n", " ")

    if num_type in ("AD", "AD2"):
        for fn in [
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
            "exp", "log", "sqrt", "pow", "abs", "min", "max"
        ]:
            cpp_code = cpp_code.replace(f"std::{fn}", f"fadbad::{fn}")

    # forcings
    for i, forcing in enumerate(forcings):
        cpp_code = _safe_replace(cpp_code, forcing, f"(*F[{i}])(t)")

    # params
    for j, param in enumerate(params):
        cpp_code = _safe_replace(cpp_code, param, f"params[{n_states + j}]")

    # states
    for i, state in enumerate(states):
        cpp_code = _safe_replace(cpp_code, state, f"x[{i}]")

    for i, state in enumerate(states):
        cpp_code = _safe_replace(cpp_code, f"{state}_0", f"params[{i}]")

    cpp_code = _safe_replace(cpp_code, "time", "t")
    cpp_code = re.sub(r"\s+", "", cpp_code)

    return cpp_code


# =====================================================================
# Main generator
# =====================================================================

def generate_ode_cpp(
    rhs_dict,
    params_list,
    num_type="AD",
    fixed_states=None,
    fixed_params=None,
    forcings_list=None,
):

    if fixed_states is None:
        fixed_states = []
    if fixed_params is None:
        fixed_params = []
    if forcings_list is None:
        forcings_list = []

    # --- Ensure all list arguments are proper lists (not strings) ---
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

    # ------------------------------------------------------------------
    # STEP 1+2: define ALL symbols once
    # ------------------------------------------------------------------
    states_syms  = {n: sp.Symbol(n, real=True) for n in states_list}
    params_syms  = {n: sp.Symbol(n, real=True) for n in params_list}
    forcing_syms = {n: sp.Symbol(n, real=True) for n in forcings_list}
    t = sp.Symbol("time", real=True)

    local_symbols = {}
    local_symbols.update(states_syms)
    local_symbols.update(params_syms)
    local_symbols.update(forcing_syms)
    local_symbols["time"] = t

    # ------------------------------------------------------------------
    # parse RHS
    # ------------------------------------------------------------------
    exprs = [
        _safe_sympify(expr, local_symbols)
        for expr in odes_list
    ]

    states_syms_list = [states_syms[s] for s in states_list]

    jac_matrix = [
        [sp.diff(expr, s) for s in states_syms_list]
        for expr in exprs
    ]

    time_derivs = [sp.diff(expr, t) for expr in exprs]

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------
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

    # STEP 3: J = df/dx, NO forcings
    for i in range(n_states):
        for j in range(n_states):
            jac_cpp_lines.append(
                f"    J({i},{j}) = {_to_cpp(jac_matrix[i][j], states_list, params_list, n_states, num_type, forcings=[])};"
            )

    # STEP 4: dfdt with forcing chain rule
    for i, expr in enumerate(exprs):
        cpp_code = _to_cpp(time_derivs[i], states_list, params_list, n_states, num_type, forcings_list)

        forcing_terms = []
        for j, fname in enumerate(forcings_list):
            df_dF = sp.diff(expr, forcing_syms[fname])   # <<< FIX
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
    """
    Generate C++ code to initialize PchipForcing objects from R raw data.
    
    Always generates the forcing infrastructure, even for n_forcings=0.
    This keeps the solve_* signature uniform.
    """
    lines = [
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
    return lines


# =====================================================================
# Event code generation
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
            if value != value:
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

    try:
        float(value_str)
        return value_str
    except (ValueError, TypeError):
        pass

    try:
        value_expr = _safe_sympify(value_str, local_symbols)
        value_code = _to_cpp(value_expr, states_list, params_list, n_states, 
                            num_type, forcings=forcings_list)
        return str(value_code)
    except Exception as e:
        raise ValueError(f"Failed to parse expression '{value_str}': {e}")


def generate_event_code(events_df, states_list, params_list, n_states, 
                        num_type="AD", forcings_list=None):
    """Generate C++ initialization lines for events."""
    if forcings_list is None:
        forcings_list = []
    
    if events_df is None or len(events_df) == 0:
        return []

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
    for name in forcings_list:
        local_symbols[name] = sp.Symbol(name, real=True)
    local_symbols["time"] = t

    event_lines = []

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
        if time_raw is not None and time_code is None:
            time_code = None

        root_raw = _get_list_value(events_dict, "root", i, n_events)
        root_code = _parse_value_or_expression(
            root_raw, local_symbols, states_list, params_list, n_states, 
            num_type, forcings_list
        )
        if root_raw is not None and root_code is None:
            root_code = None

        if time_code is not None:
            time_code = str(time_code).replace("params[", "full_params[")
            event_lines.append(
                f"  fixed_events.emplace_back(FixedEvent<{num_type}>{{"
                f"{time_code}, {var_idx}, {value_code}, {method_code}}});"
            )
        elif root_code is not None:
            root_code = str(root_code).replace("params[", "full_params[")
            event_lines.append(
                f"  root_events.push_back("
                f"RootEvent<ublas::vector<{num_type}>, {num_type}>{{"
            )
            event_lines.append(
                f"    [full_params](const ublas::vector<{num_type}>& x, "
                f"const {num_type}& t){{ return {root_code}; }},"
            )
            event_lines.append(
                f"    {var_idx}, {value_code}, {method_code}}});"
            )
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
    
    Parameters
    ----------
    rootfunc : str or list
        Either "equilibrate" for steady-state detection, or a list/vector
        of expressions that trigger termination when crossing zero.
    states_list : list
        List of state variable names.
    params_list : list
        List of parameter names.
    n_states : int
        Number of state variables.
    num_type : str
        Numeric type ("double", "AD", or "AD2").
    forcings_list : list, optional
        List of forcing function names.
    
    Returns
    -------
    list
        Lines of C++ code to initialize root events for the root function.
    """
    if forcings_list is None:
        forcings_list = []
    
    if rootfunc is None:
        return []
    
    # Handle single string vs list
    if isinstance(rootfunc, str):
        rootfunc_list = [rootfunc]
    else:
        rootfunc_list = list(rootfunc)
    
    # Check for "equilibrate" - special case
    if len(rootfunc_list) == 1 and rootfunc_list[0].lower().strip() == "equilibrate":
        return _generate_equilibrate_code(num_type)
    
    # Otherwise, treat as user-defined root expressions
    return _generate_user_rootfunc_code(
        rootfunc_list, states_list, params_list, n_states, 
        num_type, forcings_list
    )


def _generate_equilibrate_code(num_type):
    """
    Generate C++ code for steady-state detection using make_steady_state_root_func.
    
    The generated code creates a terminal root event that fires when all derivatives
    (including sensitivities for AD types) fall below the root tolerance.
    """
    state_type = f"ublas::vector<{num_type}>"
    
    lines = [
        "",
        "  // --- Steady-state termination (rootfunc = 'equilibrate') ---",
        f"  auto ss_root_func = make_steady_state_root_func<ode_system, {state_type}, {num_type}>(sys, root_tol);",
        f"  root_events.push_back(RootEvent<{state_type}, {num_type}>{{",
        f"    ss_root_func,",
        f"    0,            // state_index (ignored for terminal)",
        f"    {num_type}(0.0),  // value (ignored for terminal)",
        f"    EventMethod::Replace,  // method (ignored for terminal)",
        f"    true,         // terminal = true: stop integration",
        f"    -1            // direction = -1: trigger when rate falls below tol",
        f"  }});",
        ""
    ]
    return lines


def _generate_user_rootfunc_code(rootfunc_list, states_list, params_list, 
                                  n_states, num_type, forcings_list):
    """
    Generate C++ code for user-defined root expressions.
    
    Each expression in rootfunc_list becomes a terminal root event that
    fires when the expression crosses zero (in either direction).
    """
    # Build symbol table for parsing
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
        
        # Parse the expression
        try:
            expr = _safe_sympify(expr_str, local_symbols)
            root_code = _to_cpp(expr, states_list, params_list, n_states, 
                               num_type, forcings_list)
            root_code = str(root_code).replace("params[", "full_params[")
        except Exception as e:
            raise ValueError(f"Failed to parse rootfunc expression '{expr_str}': {e}")
        
        lines.extend([
            f"  // rootfunc[{i}]: {expr_str}",
            f"  root_events.push_back(RootEvent<{state_type}, {num_type}>{{",
            f"    [full_params, &F](const {state_type}& x, const {num_type}& t) -> {num_type} {{",
            f"      return {root_code};",
            f"    }},",
            f"    0,            // state_index (ignored for terminal)",
            f"    {num_type}(0.0),  // value (ignored for terminal)",
            f"    EventMethod::Replace,  // method (ignored for terminal)",
            f"    true,         // terminal = true: stop integration",
            f"    0             // direction = 0: any crossing",
            f"  }});",
        ])
    
    lines.append("")
    return lines


# =====================================================================
# Root function code generation (for integration termination)
# =====================================================================

def generate_rootfunc_code(rootfunc, states_list, params_list, n_states,
                           num_type="AD", forcings_list=None):
    """
    Generate C++ initialization lines for terminal root events.
    
    This function handles two cases:
    
    1. rootfunc = "equilibrate": 
       Uses make_steady_state_root_func to stop integration when the system
       reaches steady state (all derivatives including sensitivities < roottol).
       
    2. rootfunc = list of expressions (e.g., ["x - 0.5", "y - 1.0"]):
       Similar to deSolve's rootfunc - stops integration when any expression
       crosses zero.
    
    Parameters
    ----------
    rootfunc : str or list
        Either "equilibrate" or a list of expressions
    states_list : list
        List of state variable names
    params_list : list
        List of parameter names
    n_states : int
        Number of state variables
    num_type : str
        Numeric type ("double", "AD", or "AD2")
    forcings_list : list, optional
        List of forcing function names
        
    Returns
    -------
    list of str
        C++ code lines to be inserted after event initialization
    """
    if forcings_list is None:
        forcings_list = []
    
    if rootfunc is None:
        return []
    
    # Handle single string that might be "equilibrate" or a single expression
    if isinstance(rootfunc, str):
        rootfunc_lower = rootfunc.strip().lower()
        if rootfunc_lower == "equilibrate":
            # Use make_steady_state_root_func
            return _generate_equilibrate_code(num_type)
        else:
            # Single expression - wrap in list
            rootfunc = [rootfunc]
    
    # At this point, rootfunc should be a list of expressions
    if not isinstance(rootfunc, (list, tuple)):
        raise ValueError(f"rootfunc must be 'equilibrate' or a list of expressions, got: {type(rootfunc)}")
    
    return _generate_rootfunc_expressions_code(
        rootfunc, states_list, params_list, n_states, num_type, forcings_list
    )


def _generate_equilibrate_code(num_type):
    """
    Generate C++ code for steady-state detection using make_steady_state_root_func.
    
    The generated code creates a terminal root event that fires when all
    derivatives (including sensitivities for AD types) fall below root_tol.
    """
    lines = [
        "",
        "  // --- Steady-state detection (equilibrate) ---",
        "  {",
        f"    auto ss_root_func = make_steady_state_root_func<",
        f"        decltype(sys), ublas::vector<{num_type}>, {num_type}>(sys, root_tol);",
        "",
        f"    root_events.push_back(RootEvent<ublas::vector<{num_type}>, {num_type}>{{",
        "        ss_root_func,",
        "        0,                      // state_index (ignored for terminal)",
        f"        {num_type}(0),          // value (ignored for terminal)",
        "        EventMethod::Replace,   // method (ignored for terminal)",
        "        true,                   // terminal = true: stop integration",
        "        -1                      // direction = -1: trigger when rate falls below tol",
        "    });",
        "  }",
        "",
    ]
    return lines


def _generate_rootfunc_expressions_code(expressions, states_list, params_list, 
                                         n_states, num_type, forcings_list):
    """
    Generate C++ code for user-specified root function expressions.
    
    Each expression creates a terminal root event that fires when
    the expression crosses zero (any direction).
    """
    # Create symbol tables for parsing
    states_syms = {name: sp.Symbol(name, real=True) for name in states_list}
    params_syms = {name: sp.Symbol(name, real=True) for name in params_list}
    t = sp.Symbol("time", real=True)
    
    local_symbols = {}
    local_symbols.update(states_syms)
    local_symbols.update(params_syms)
    for name in forcings_list:
        local_symbols[name] = sp.Symbol(name, real=True)
    local_symbols["time"] = t
    
    lines = [
        "",
        "  // --- User-specified root functions (terminal) ---",
    ]
    
    for i, expr_str in enumerate(expressions):
        if not _is_valid_value(expr_str):
            continue
            
        expr_str = str(expr_str).strip()
        
        try:
            # Parse and convert to C++
            expr = _safe_sympify(expr_str, local_symbols)
            cpp_code = _to_cpp(expr, states_list, params_list, n_states, 
                              num_type, forcings=forcings_list)
            cpp_code = str(cpp_code).replace("params[", "full_params[")
        except Exception as e:
            raise ValueError(f"Failed to parse rootfunc expression '{expr_str}': {e}")
        
        lines.extend([
            f"  // Root expression {i}: {expr_str}",
            f"  root_events.push_back(RootEvent<ublas::vector<{num_type}>, {num_type}>{{",
            f"      [full_params](const ublas::vector<{num_type}>& x, const {num_type}& t) {{",
            f"          return {cpp_code};",
            "      },",
            "      0,                      // state_index (ignored for terminal)",
            f"      {num_type}(0),          // value (ignored for terminal)",
            "      EventMethod::Replace,   // method (ignored for terminal)",
            "      true,                   // terminal = true: stop integration",
            "      0                       // direction = 0: any direction",
            "  });",
            "",
        ])
    
    return lines
