"""
Algebraic Function C++ Code Generator for CppODE
============================================================
Generates C++ source code for evaluating algebraic functions and their
symbolic Jacobians and Hessians. Supports fixed symbols that are treated
as constants (no differentiation) but still appear as runtime parameters.

Author: Simon Beyer
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor
)
from sympy.printing.cxx import CXX17CodePrinter
import os
import re
from functools import lru_cache
from io import StringIO
# =====================================================================
# Safe parsing configuration (cached)
# =====================================================================

def _ensure_double_literals(cpp_code):
    """
    Convert integer literals to double literals in C++ code.
    
    This fixes issues with std::max/std::min template argument deduction
    where mixing int and double arguments causes compilation errors.
    E.g., std::max(0, x[1]) fails but std::max(0.0, x[1]) works.
    
    The function uses a two-pass approach:
    1. Temporarily replace scientific notation with placeholders
    2. Convert remaining integers to doubles
    3. Restore scientific notation
    
    This avoids converting numbers like 1e-5 incorrectly.
    """
    # Match scientific notation: digits, optionally with decimal, 
    # followed by e/E and optional sign and digits
    sci_pattern = r'(\d+\.?\d*[eE][+-]?\d+)'
    
    # Store all scientific notation numbers
    sci_numbers = []
    def store_sci(match):
        sci_numbers.append(match.group(0))
        return f'__SCI_PLACEHOLDER_{len(sci_numbers)-1}__'
    
    # Replace scientific notation with placeholders
    temp = re.sub(sci_pattern, store_sci, cpp_code)
    
    # Convert integers (not preceded by identifier chars or '[', not followed by digits, '.', or ']')
    # This converts: 0, 1, 42
    # But NOT: x[0], 3.14, array[123]
    int_pattern = r'(?<![a-zA-Z0-9_.\[])(\d+)(?![0-9.\]])'
    temp = re.sub(int_pattern, lambda m: m.group(1) + '.0', temp)
    
    # Restore scientific notation
    for i, sci in enumerate(sci_numbers):
        temp = temp.replace(f'__SCI_PLACEHOLDER_{i}__', sci)
    
    return temp
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
        'exp10': lambda x: sp.exp(x * sp.log(10)),
        'exp2': lambda x: sp.exp(x * sp.log(2)),
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
        'round': lambda x: sp.floor(x + sp.Rational(1, 2)),
        
        # Min/Max
        'min': sp.Min, 'max': sp.Max,
        
        # Factorial and gamma functions
        'factorial': sp.factorial, 'gamma': sp.gamma,
        'loggamma': sp.loggamma, 'digamma': sp.digamma,
        'polygamma': sp.polygamma, 'beta': sp.beta,
        
        # Error functions
        'erf': sp.erf, 'erfc': sp.erfc, 'erfi': sp.erfi,
        
        # Bessel functions
        'besselj': sp.besselj, 'bessely': sp.bessely,
        'besseli': sp.besseli, 'besselk': sp.besselk,
        
        # Special functions
        'Heaviside': sp.Heaviside, 'DiracDelta': sp.DiracDelta,
        'KroneckerDelta': sp.KroneckerDelta, 'Piecewise': sp.Piecewise,
        
        # Constants
        'pi': sp.pi, 'E': sp.E, 'euler_gamma': sp.EulerGamma, 'oo': sp.oo,
        
        # Complex functions
        're': sp.re, 'im': sp.im, 'conjugate': sp.conjugate, 'arg': sp.arg,
    }
# =====================================================================
# Code Generation Context
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
class CodeGenContext:
    """
    Holds precomputed state for efficient code generation.
    Avoids repeated symbol creation and regex compilation.
    """
    
    def __init__(self, variables, parameters):
        self.variables = list(variables) if variables else []
        self.parameters = list(parameters) if parameters else []
        
        # Create symbols once
        self.var_symbols = {v: sp.Symbol(v, real=True) for v in self.variables}
        self.par_symbols = {p: sp.Symbol(p, real=True) for p in self.parameters}
        self.all_symbols = {**self.var_symbols, **self.par_symbols}
        
        # Build replacement mapping: symbol_name -> replacement string
        self.replacements = {}
        for i, v in enumerate(self.variables):
            self.replacements[v] = f"x_obs[{i}]"
        for i, p in enumerate(self.parameters):
            self.replacements[p] = f"p[{i}]"
        
        # Precompile regex patterns (sorted by length, longest first)
        all_names = sorted(self.replacements.keys(), key=len, reverse=True)
        self.compiled_patterns = [
            (re.compile(r"\b" + re.escape(name) + r"\b"), self.replacements[name])
            for name in all_names
        ]
        
        # Custom printer for faster code generation
        self.printer = CXX17CodePrinter()
        
        # Cache for converted expressions
        self._expr_cache = {}
    
    def to_cpp(self, expr):
        """
        Convert a SymPy expression or string to valid C++ code.
        Uses caching for repeated expressions.
        """
        # Create a hashable key
        if isinstance(expr, str):
            cache_key = expr.strip()
            if cache_key == "0":
                return "0.0"
        else:
            cache_key = expr
            if expr == 0:
                return "0.0"
        
        # Check cache
        if cache_key in self._expr_cache:
            return self._expr_cache[cache_key]
        
        # Parse string if needed
        if isinstance(expr, str):
            expr = self._parse_expr(cache_key)
        
        if expr == 0:
            self._expr_cache[cache_key] = "0.0"
            return "0.0"
        
        # Replace DiracDelta
        expr = self._replace_dirac_delta(expr)
        
        # Generate C++ code
        cpp_code = self.printer.doprint(expr)
        
        # replace non-standard math macros
        for macro, repl in _MATH_MACRO_MAP.items():
            cpp_code = cpp_code.replace(macro, repl)
        
        # Apply all replacements using precompiled patterns
        for pattern, replacement in self.compiled_patterns:
            cpp_code = pattern.sub(replacement, cpp_code)
        
        # Convert integer literals to double literals to avoid C++ template deduction issues
        cpp_code = _ensure_double_literals(cpp_code)
        
        self._expr_cache[cache_key] = cpp_code
        return cpp_code
    
    def _parse_expr(self, expr_str):
        """Parse a string expression to SymPy."""
        safe_local = dict(_get_safe_parse_dict_cached())
        safe_local.update(self.all_symbols)
        
        transformations = standard_transformations + (convert_xor,)
        
        return parse_expr(
            expr_str,
            local_dict=safe_local,
            transformations=transformations,
            evaluate=True,
        )
    
    @staticmethod
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
# =====================================================================
# Public interface
# =====================================================================

def generate_fun_cpp(exprs, variables, parameters=None,
                     jacobian=None, hessian=None, ad=False,
                     modelname="model", outdir=None, version = "1.0.0"):
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
    ad : bool, optional
        If True, emit an extra entry <modelname>_eval_ad that, via forward-mode
        AD on cppode::dual<double, 0>, returns value and chain-ruled
        dY/dtheta for given upstream seeds dX (per obs) and dP (const).
    modelname : str, optional
        Base name for the generated model.

    Returns
    -------
    dict
        {"filename": absolute path, "modelname": model name}
    """
    # Normalize inputs
    if isinstance(variables, str):
        variables = [variables]
    variables = variables or []

    if parameters is None:
        parameters = []
    elif isinstance(parameters, str):
        parameters = [parameters]

    if isinstance(exprs, list):
        exprs = {f"f{i+1}": expr for i, expr in enumerate(exprs)}
    elif not isinstance(exprs, dict):
        raise TypeError("exprs must be a dict or list")

    # Create context with precomputed state
    ctx = CodeGenContext(variables, parameters)

    # Parse expressions
    parsed_exprs = _parse_expressions(exprs, ctx)

    # Generate C++ code using StringIO for efficient string building
    cpp_code = _generate_cpp_code(
        parsed_exprs, ctx, jacobian, hessian, ad, modelname, version
    )
    
    if outdir is None:
        raise ValueError("outdir must be provided explicitly")
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f"{modelname}.cpp")
    
    with open(filename, "w") as f:
        f.write(cpp_code)

    return {"filename": os.path.abspath(filename), "modelname": modelname}
# =====================================================================
# Parsing
# =====================================================================

def _parse_expressions(exprs, ctx):
    """Parse algebraic expressions into SymPy objects."""
    safe_local = dict(_get_safe_parse_dict_cached())
    safe_local.update(ctx.all_symbols)
    
    transformations = standard_transformations + (convert_xor,)
    
    parsed = {}
    for name, expr_str in exprs.items():
        try:
            expr_str = str(expr_str).strip()
            if expr_str == "0":
                parsed[name] = sp.Integer(0)
            else:
                parsed[name] = parse_expr(
                    expr_str,
                    local_dict=safe_local,
                    transformations=transformations,
                    evaluate=True,
                )
        except Exception as e:
            raise ValueError(
                f"Failed to parse expression '{name}': {expr_str}\nError: {e}"
            )
    return parsed
# =====================================================================
# C++ source assembly (using StringIO)
# =====================================================================

def _strip_std_prefix(cpp):
    """
    Strip `std::` prefix from math functions and rename fabs -> abs so the
    same expression compiles for both T=double and T=cppode::dual<double, 0>.

    For T=double, `using std::{exp,log,...,abs,max,min};` resolves the call.
    For T=dual<...>, the cppode:: overloads defined in cppode_dual_math.hpp
    are picked by ADL.
    """
    cpp = re.sub(r'\bstd::fabs\b', 'abs', cpp)
    cpp = re.sub(r'\bstd::', '', cpp)
    return cpp


def _generate_cpp_code(exprs, ctx, jacobian, hessian, ad, modelname, version):
    """Assemble the complete C++ source for the model."""
    out_names = list(exprs.keys())
    buf = StringIO()

    # Header
    buf.write(f"/** Code auto-generated by CppODE {version} **/\n\n")
    buf.write("#include <cmath>\n")
    buf.write("#include <algorithm>\n")
    if ad:
        buf.write("#include <vector>\n")
        buf.write("#include <cppode/cppode_dual_math.hpp>\n")
        buf.write("#include <cppode/cppode_dual_expr.hpp>\n")
    buf.write("\n")
    buf.write(f"// Modelname: {modelname}\n")
    buf.write(f"// Variables: {', '.join(ctx.variables) if ctx.variables else 'none'}\n")
    buf.write(f"// Parameters: {', '.join(ctx.parameters) if ctx.parameters else 'none'}\n")
    buf.write(f"// Outputs: {', '.join(out_names)}\n\n")

    # Template helper shared by _eval and (if ad) _eval_ad.
    _write_eval_template(buf, exprs, out_names, ctx, modelname, ad)

    buf.write("extern \"C\" {\n\n")

    # Eval function (always, uses template helper with T=double).
    _write_eval_function(buf, out_names, ctx, modelname)

    # AD function (forward-mode, chain-ruled derivatives).
    if ad:
        _write_eval_ad_function(buf, out_names, ctx, modelname)

    # Jacobian (symbolic, double-only).
    if jacobian is not None:
        _write_jacobian_function(buf, jacobian, out_names, ctx, modelname)

    # Hessian (symbolic, double-only).
    if hessian is not None:
        _write_hessian_function(buf, hessian, out_names, ctx, modelname)

    buf.write("} // extern \"C\"\n")

    return buf.getvalue()


def _write_eval_template(buf, exprs, out_names, ctx, modelname, ad):
    """
    Emit an inline template helper that evaluates the expressions for a single
    observation. Templated on scalar type T so it instantiates for `double` and,
    when `ad=True`, for cppode::dual<double, 0>.
    """
    n_vars = len(ctx.variables)

    buf.write("namespace {\n")
    buf.write("template <typename T>\n")
    buf.write(f"inline void {modelname}_eval_one(const T* x_obs, const T* p, T* y_local) {{\n")
    # Bring math functions into scope so expressions resolve for both double
    # and the AD scalar via overload resolution.
    buf.write("    using std::exp; using std::log; using std::sqrt; using std::pow;\n")
    buf.write("    using std::sin; using std::cos; using std::tan;\n")
    buf.write("    using std::asin; using std::acos; using std::atan; using std::atan2;\n")
    buf.write("    using std::sinh; using std::cosh; using std::tanh;\n")
    buf.write("    using std::asinh; using std::acosh; using std::atanh;\n")
    buf.write("    using std::floor; using std::ceil;\n")
    buf.write("    using std::abs; using std::max; using std::min;\n")
    if ad:
        # cppode::dual overloads live in namespace cppode; ADL picks them up
        # for dual<...> arguments. Pull them in by name to also make
        # double-or-dual mixed expressions resolve consistently.
        buf.write("    using cppode::exp; using cppode::log; using cppode::sqrt; using cppode::pow;\n")
        buf.write("    using cppode::sin; using cppode::cos; using cppode::tan;\n")
        buf.write("    using cppode::asin; using cppode::acos; using cppode::atan;\n")
        buf.write("    using cppode::sinh; using cppode::cosh; using cppode::tanh;\n")
        buf.write("    using cppode::asinh; using cppode::acosh; using cppode::atanh;\n")
        buf.write("    using cppode::abs; using cppode::max; using cppode::min;\n")
    if n_vars == 0:
        buf.write("    (void)x_obs;\n")
    if not ctx.parameters:
        buf.write("    (void)p;\n")
    buf.write("\n")

    for i, out_name in enumerate(out_names):
        cpp_code = _strip_std_prefix(ctx.to_cpp(exprs[out_name]))
        buf.write(f"    y_local[{i}] = {cpp_code};\n")

    buf.write("}\n")
    buf.write("} // anonymous namespace\n\n")


def _write_eval_function(buf, out_names, ctx, modelname):
    """Generate extern-C eval entry point (double instantiation of the helper).

    Layouts (R column-major):
      x [n_vars, n_obs]  -> j + n_vars * obs   (unchanged input)
      y [n_obs, n_out]   -> obs + n_obs * i    (time-first output)
    """
    n_vars = len(ctx.variables)
    n_out = len(out_names)

    buf.write(f"void {modelname}_eval(double* x, double* y, double* p, int* n, int* k, int* l) {{\n")
    buf.write("    const int n_obs = *n;\n")
    buf.write("    const int n_vars = *k;\n")
    buf.write("    const int n_out  = *l;\n")
    buf.write("    (void)n_vars;  // suppress unused warning\n\n")
    buf.write(f"    double y_local[{max(n_out, 1)}];\n")
    buf.write("    for (int obs = 0; obs < n_obs; obs++) {\n")

    if n_vars > 0:
        buf.write("        const double* x_obs = x + obs * n_vars;\n")
    else:
        buf.write("        const double* x_obs = nullptr;\n")

    buf.write(f"        {modelname}_eval_one<double>(x_obs, p, y_local);\n")
    buf.write("        for (int i = 0; i < n_out; ++i)\n")
    buf.write("            y[obs + (size_t)n_obs * i] = y_local[i];\n")
    buf.write("    }\n}\n\n")


def _write_eval_ad_function(buf, out_names, ctx, modelname):
    """
    Generate extern-C AD entry point. Takes upstream seeds dX (per-obs state
    sensitivities) and dP (parameter Jacobian) and fills value + dY/dtheta
    in a single forward-mode AD pass per observation.

    Layouts (R column-major):
      x   [n_vars, n_obs]                    -> j + n_vars * obs     (input, unchanged)
      p   [n_params]                         -> j
      dX  [n_obs, n_vars, n_theta]           -> obs + n_obs * (j + n_vars * k)
      dP  [n_params, n_theta]                -> j + n_params * k
      y   [n_obs, n_out]                     -> obs + n_obs * i
      dy  [n_obs, n_out, n_theta]            -> obs + n_obs * (i + n_out * k)
    """
    n_vars = len(ctx.variables)
    n_params = len(ctx.parameters)
    n_out = len(out_names)

    ad_t = "cppode::dual<double, 0>"

    buf.write(
        f"void {modelname}_eval_ad(double* x, double* p, double* dX, double* dP,\n"
        f"                         double* y, double* dy,\n"
        f"                         int* n_obs_p, int* n_vars_p, int* n_params_p,\n"
        f"                         int* n_out_p, int* n_theta_p) {{\n"
    )
    buf.write(f"    using AD = {ad_t};\n")
    buf.write("    const int n_obs    = *n_obs_p;\n")
    buf.write("    const int n_vars   = *n_vars_p;\n")
    buf.write("    const int n_params = *n_params_p;\n")
    buf.write("    const int n_out    = *n_out_p;\n")
    buf.write("    const int n_theta  = *n_theta_p;\n")
    buf.write("    (void)n_vars; (void)n_params; (void)n_out;\n")
    buf.write("    (void)dX; (void)dP;\n\n")

    # dual<double, 0> tangent buffers come from the thread-local arena.
    # One outer scope guard frees every per-obs CSE temp at function exit.
    # Per-obs scopes are unsafe here because the AD vectors persist across
    # iterations and would hold dangling tan_ pointers after a pop.
    buf.write("    cppode::dual_arena::scope _eval_ad_scope;\n")

    buf.write(f"    std::vector<AD> x_ad({n_vars});\n")
    buf.write(f"    std::vector<AD> p_ad({n_params});\n")
    buf.write(f"    std::vector<AD> y_ad({n_out});\n\n")

    # Seed parameters once (constant across obs).
    if n_params > 0:
        buf.write("    // Seed parameters (time-invariant).\n")
        buf.write("    for (int j = 0; j < n_params; ++j) {\n")
        buf.write("        p_ad[j].x() = p[j];\n")
        buf.write("        if (n_theta > 0) {\n")
        buf.write("            p_ad[j].diff(0, n_theta);\n")
        buf.write("            for (int k = 0; k < n_theta; ++k) {\n")
        buf.write("                p_ad[j][k] = dP[j + n_params * k];\n")
        buf.write("            }\n")
        buf.write("        }\n")
        buf.write("    }\n\n")

    buf.write("    for (int obs = 0; obs < n_obs; ++obs) {\n")
    if n_vars > 0:
        buf.write("        for (int j = 0; j < n_vars; ++j) {\n")
        buf.write("            x_ad[j].x() = x[j + n_vars * obs];\n")
        buf.write("            if (n_theta > 0) {\n")
        buf.write("                x_ad[j].diff(0, n_theta);\n")
        buf.write("                for (int k = 0; k < n_theta; ++k) {\n")
        buf.write("                    x_ad[j][k] = dX[obs + (size_t)n_obs * (j + (size_t)n_vars * k)];\n")
        buf.write("                }\n")
        buf.write("            }\n")
        buf.write("        }\n")
    buf.write(f"        {modelname}_eval_one<AD>(x_ad.data(), p_ad.data(), y_ad.data());\n")
    buf.write("        for (int i = 0; i < n_out; ++i) {\n")
    buf.write("            y[obs + (size_t)n_obs * i] = y_ad[i].val();\n")
    buf.write("            for (int k = 0; k < n_theta; ++k) {\n")
    # operator[] on cppode::dual<double, 0> exposes the tangent component.
    buf.write("                dy[obs + (size_t)n_obs * (i + (size_t)n_out * k)] = y_ad[i][k];\n")
    buf.write("            }\n")
    buf.write("        }\n")
    buf.write("    }\n}\n\n")
def _write_jacobian_function(buf, jacobian, out_names, ctx, modelname):
    """
    Generate Jacobian evaluation function (sparse: only non-zero entries).

    Array layout for R (column-major): jac[n_obs, n_out, n_symbols]
    Linear index: obs + n_obs * (output + n_out * symbol)
    """
    first_fn = next(iter(jacobian))
    n_symbols = len(jacobian[first_fn])
    n_vars = len(ctx.variables)
    n_out = len(out_names)

    buf.write(f"void {modelname}_jacobian(double* x, double* jac, double* p, int* n, int* k, int* l) {{\n")
    buf.write("    const int n_obs = *n;\n")
    buf.write("    const int n_vars = *k;\n")
    buf.write("    (void)n_vars;  // suppress unused warning\n")
    buf.write(f"    const int n_out = {n_out};\n")
    buf.write(f"    const int n_symbols = {n_symbols};\n\n")

    # Zero-initialize entire jacobian array
    buf.write("    // Zero-initialize\n")
    buf.write("    std::fill(jac, jac + (size_t)n_obs * n_out * n_symbols, 0.0);\n\n")

    buf.write("    // Layout: jac[obs, output, symbol] (R column-major)\n")
    buf.write("    // Linear index: obs + n_obs * (output + n_out * symbol)\n")
    buf.write("    for (int obs = 0; obs < n_obs; obs++) {\n")

    if n_vars > 0:
        buf.write("        const double* x_obs = x + obs * n_vars;\n")
        buf.write("        (void)x_obs;  // suppress unused warning\n")

    for i, out_name in enumerate(out_names):
        if out_name not in jacobian:
            continue

        jac_exprs = jacobian[out_name]
        has_nonzero = False
        for j, expr in enumerate(jac_exprs):
            cpp_code = ctx.to_cpp(expr)
            if cpp_code == "0.0" or cpp_code == "0":
                continue
            has_nonzero = True
            # R column-major: obs + n_obs * (output + n_out * symbol)
            buf.write(f"        jac[obs + (size_t)n_obs * ({i} + n_out * {j})] = {cpp_code};\n")

        # Blank line between output blocks (not after the last one)
        if has_nonzero and i < len(out_names) - 1:
            buf.write("\n")

    buf.write("    }\n}\n\n")
def _write_hessian_function(buf, hessian, out_names, ctx, modelname):
    """
    Generate Hessian evaluation function (sparse: only non-zero entries).

    Array layout for R (column-major): hess[n_obs, n_out, n_symbols, n_symbols]
    Linear index: obs + n_obs * (output + n_out * (sym1 + n_symbols * sym2))
    """
    first_fn = next(iter(hessian))
    n_symbols = len(hessian[first_fn])
    n_vars = len(ctx.variables)
    n_out = len(out_names)

    buf.write(f"void {modelname}_hessian(double* x, double* hess, double* p, int* n, int* k, int* l) {{\n")
    buf.write("    const int n_obs = *n;\n")
    buf.write("    const int n_vars = *k;\n")
    buf.write(f"    const int n_out = {n_out};\n")
    buf.write("    (void)n_vars;  // suppress unused warning\n")
    buf.write(f"    const int n_symbols = {n_symbols};\n\n")

    buf.write("    // Zero-initialize\n")
    buf.write("    std::fill(hess, hess + (size_t)n_obs * n_out * n_symbols * n_symbols, 0.0);\n\n")

    buf.write("    // Layout: hess[obs, output, sym1, sym2] (R column-major)\n")
    buf.write("    // Linear index: obs + n_obs * (output + n_out * (sym1 + n_symbols * sym2))\n")
    buf.write("    for (int obs = 0; obs < n_obs; obs++) {\n")

    if n_vars > 0:
        buf.write("        const double* x_obs = x + obs * n_vars;\n")
        buf.write("        (void)x_obs;  // suppress unused warning\n")

    for i, out_name in enumerate(out_names):
        if out_name not in hessian:
            continue

        hess_matrix = hessian[out_name]
        has_nonzero = False
        for j, hess_row in enumerate(hess_matrix):
            for k, expr in enumerate(hess_row):
                cpp_code = ctx.to_cpp(expr)
                if cpp_code == "0.0" or cpp_code == "0":
                    continue
                has_nonzero = True
                # R column-major: obs + n_obs * (output + n_out * (sym1 + n_symbols * sym2))
                buf.write(
                    f"        hess[obs + (size_t)n_obs * ({i} + n_out * ({j} + n_symbols * {k}))] = {cpp_code};\n"
                )

        # Blank line between output blocks (not after the last one)
        if has_nonzero and i < len(out_names) - 1:
            buf.write("\n")

    buf.write("    }\n}\n\n")
