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
import keyword
from functools import lru_cache
from io import StringIO

_IDENT_RE = re.compile(r'(?<![\.\w])[A-Za-z_][A-Za-z0-9_]*')
_PY_RESERVED = frozenset(keyword.kwlist) | {'True', 'False', 'None'}
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

        # Pre-declare bare identifiers as Symbols so user names like beta,
        # gamma, zeta, etc. don't resolve to SymPy FunctionClass globals.
        for name in _IDENT_RE.findall(expr_str):
            if name not in safe_local and name not in _PY_RESERVED:
                safe_local[name] = sp.Symbol(name, real=True)

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
                     jacobian=None, hessian=None, ad=False, deriv2=False,
                     modelname="model", outdir=None, version = "1.0.0"):
    """
    Generate C++ source code for algebraic model evaluation.

    Two disjoint emission paths driven by `ad`:

      ad = False (symbolic mode): emit `_eval`; emit `_jacobian` if `jacobian`
        is given; emit `_hessian` if `hessian` is given. Whenever a Jacobian
        or Hessian C entry is emitted, also emit a thin `_chain_jac` /
        `_chain_hess` wrapper around the BLAS-3 helpers in
        `cppode/cppode_chain_blas.hpp`.

      ad = True (dual mode): emit `_eval` plus `_eval_ad` (forward-mode AD on
        `cppode::dual<double, 0>`); if `deriv2` is also True, additionally
        emit `_eval_ad2` (nested dual `cppode::dual2nd<double, 0>`) for
        first- and second-order chain-rule sensitivities in one pass. No
        symbolic Jacobian/Hessian or chain wrappers are emitted in this
        mode.

    Parameters
    ----------
    exprs : dict[str, str] or list[str]
        Algebraic model expressions. Dict {"f1": "a*x + b*y"} or list.
    variables : list[str]
        Variable symbols (vary by observation).
    parameters : list[str], optional
        Parameter symbols (constant across observations). Includes fixed ones.
    jacobian : dict[str, list[str]], optional
        Symbolic Jacobian from derivSymb(). Only consulted when `ad = False`.
    hessian : dict[str, list[list[str]]], optional
        Symbolic Hessian from derivSymb(). Only consulted when `ad = False`.
    ad : bool, optional
        If True, select the dual-number AD emission path (no symbolic
        Jacobian/Hessian even if supplied).
    deriv2 : bool, optional
        Only meaningful with `ad = True`. Triggers emission of `_eval_ad2`.
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
        parsed_exprs, ctx, jacobian, hessian, ad, deriv2, modelname, version
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
    base_local = dict(_get_safe_parse_dict_cached())
    base_local.update(ctx.all_symbols)

    transformations = standard_transformations + (convert_xor,)

    parsed = {}
    for name, expr_str in exprs.items():
        try:
            expr_str = str(expr_str).strip()
            if expr_str == "0":
                parsed[name] = sp.Integer(0)
            else:
                # Pre-declare bare identifiers per-expression so user names
                # like beta/gamma/zeta don't resolve to SymPy FunctionClass
                # globals via parse_expr's default global_dict.
                local = dict(base_local)
                for nm in _IDENT_RE.findall(expr_str):
                    if nm not in local and nm not in _PY_RESERVED:
                        local[nm] = sp.Symbol(nm, real=True)
                parsed[name] = parse_expr(
                    expr_str,
                    local_dict=local,
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


# =====================================================================
# Common subexpression elimination
#
# funCpp expressions produced by upstream symbolic pipelines can be enormous
# with heavy internal repetition (the same sub-tree appearing dozens of times
# across outputs and within a single output). Without CSE every occurrence is
# re-emitted verbatim, which blows up C++ compile time and, in AD mode, the
# size of the dual expression-template tree walked once per observation.
#
# sp.cse lifts subexpressions shared across (or within) a group of expressions
# into numbered temporaries materialised once. In the AD-templated `_eval_one`
# helper the temp is stored into a `const T` local, which forces the dual ET to
# evaluate at the temp boundary so later references are a scalar dual load
# instead of a re-walk of the whole tree (mirrors codegenCppODE.py::_cse_temps).
# =====================================================================

def _cse_exprs(exprs, prefix='_cse_t'):
    """Apply sympy CSE across a list of parsed sympy expressions.

    Returns (temps, simplified): temps is a list of (Symbol, subexpr) in
    dependency order; simplified is the rewritten expression list (same length
    and order as the input). When sympy finds no shared structure temps is
    empty and simplified == list(exprs), so callers fall back to direct
    emission with zero overhead. No gating on expression count: a single giant
    expression with internal repetition is the primary case CSE targets here.
    """
    syms = sp.numbered_symbols(prefix=prefix, cls=sp.Symbol)
    return sp.cse(list(exprs), symbols=syms, optimizations='basic')


def _parse_entry(expr, ctx):
    """Parse a Jacobian/Hessian entry to a sympy expression.

    Entries arrive as strings from derivSymb() (or already-parsed sympy).
    Returns sp.Integer(0) for entries that are exactly zero so callers can
    skip them before CSE, preserving the original sparse-emission behaviour.
    """
    if not isinstance(expr, str):
        return expr
    s = expr.strip()
    if s == "0" or s == "0.0":
        return sp.Integer(0)
    return ctx._parse_expr(s)


def _generate_cpp_code(exprs, ctx, jacobian, hessian, ad, deriv2, modelname, version):
    """Assemble the complete C++ source for the model.

    Modes are disjoint: when `ad=True` the symbolic jacobian/hessian arguments
    are ignored even if provided, and no `_jacobian`/`_hessian`/`_chain_*`
    entries are emitted. When `ad=False` the AD entries `_eval_ad`/`_eval_ad2`
    are not emitted.
    """
    out_names = list(exprs.keys())
    buf = StringIO()

    if ad:
        emit_jacobian = False
        emit_hessian  = False
    else:
        emit_jacobian = jacobian is not None
        emit_hessian  = hessian  is not None

    emit_chain = emit_jacobian or emit_hessian
    emit_ad    = ad
    emit_ad2   = ad and deriv2

    # Header
    buf.write(f"/** Code auto-generated by CppODE {version} **/\n\n")
    buf.write("#include <cmath>\n")
    buf.write("#include <algorithm>\n")
    if emit_ad:
        buf.write("#include <vector>\n")
        buf.write("#include <cppode/cppode_dual_math.hpp>\n")
        buf.write("#include <cppode/cppode_dual_expr.hpp>\n")
    if emit_ad2:
        buf.write("#include <cppode/cppode_dual2nd.hpp>\n")
        buf.write("#include <cppode/cppode_dual2nd_math.hpp>\n")
        buf.write("#include <cppode/cppode_dual2nd_expr.hpp>\n")
    if emit_chain:
        buf.write("#include <cppode/cppode_chain_blas.hpp>\n")
    buf.write("\n")
    buf.write(f"// Modelname: {modelname}\n")
    buf.write(f"// Variables: {', '.join(ctx.variables) if ctx.variables else 'none'}\n")
    buf.write(f"// Parameters: {', '.join(ctx.parameters) if ctx.parameters else 'none'}\n")
    buf.write(f"// Outputs: {', '.join(out_names)}\n\n")

    # Template helper shared by _eval and (if ad) _eval_ad / _eval_ad2.
    _write_eval_template(buf, exprs, out_names, ctx, modelname, emit_ad)

    buf.write("extern \"C\" {\n\n")

    # Eval function (always, uses template helper with T=double).
    _write_eval_function(buf, out_names, ctx, modelname)

    # AD entries (dual mode).
    if emit_ad:
        _write_eval_ad_function(buf, out_names, ctx, modelname)
    if emit_ad2:
        _write_eval_ad2_function(buf, out_names, ctx, modelname)

    # Jacobian (symbolic, double-only) and its BLAS chain-rule wrapper.
    if emit_jacobian:
        _write_jacobian_function(buf, jacobian, out_names, ctx, modelname)
        _write_chain_jac_wrapper(buf, modelname)

    # Hessian (symbolic, double-only) and its BLAS chain-rule wrapper.
    if emit_hessian:
        _write_hessian_function(buf, hessian, out_names, ctx, modelname)
        _write_chain_hess_wrapper(buf, modelname)

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

    # CSE across all outputs: shared subexpressions are lifted into `const T`
    # temps materialised once per call. For T=dual this collapses the ET tree
    # walked per observation; for T=double it just shrinks the emitted source.
    ordered = [exprs[nm] for nm in out_names]
    temps, simplified = _cse_exprs(ordered, prefix='_cse_t')
    for sym, sub in temps:
        sub_cpp = _strip_std_prefix(ctx.to_cpp(sub))
        buf.write(f"    const T {sym.name} = {sub_cpp};\n")
    if temps:
        buf.write("\n")

    for i, expr in enumerate(simplified):
        cpp_code = _strip_std_prefix(ctx.to_cpp(expr))
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
    # .d(k) on cppode::dual<double, 0> returns the k-th tangent, or a
    # thread-local zero when the dual is non-depend (size_ == 0). Using the
    # raw operator[] would dereference tan_ unconditionally and segfault for
    # outputs that are constant in all parameters (e.g. y_i = "0"), since
    # operator=(double) leaves tan_ == nullptr.
    buf.write("                dy[obs + (size_t)n_obs * (i + (size_t)n_out * k)] = y_ad[i].d(k);\n")
    buf.write("            }\n")
    buf.write("        }\n")
    buf.write("    }\n}\n\n")


def _write_eval_ad2_function(buf, out_names, ctx, modelname):
    """
    Generate extern-C nested-dual AD entry. Computes y, dy, d2y in one pass on
    cppode::dual<cppode::dual<double, 0>, 0>. Accepts first-order seeds dX,
    dP and optional second-order seeds dX2, dP2 (controlled by has_dX2,
    has_dP2 flags so length-zero R vectors stay safe).

    Layouts (R column-major):
      x   [n_vars, n_obs]
      p   [n_params]
      dX  [n_obs, n_vars, n_theta]
      dP  [n_params, n_theta]
      dX2 [n_obs, n_vars, n_theta, n_theta]   (skipped if has_dX2 == 0)
      dP2 [n_params, n_theta, n_theta]        (skipped if has_dP2 == 0)
      y   [n_obs, n_out]
      dy  [n_obs, n_out, n_theta]
      d2y [n_obs, n_out, n_theta, n_theta]
    """
    n_vars = len(ctx.variables)
    n_params = len(ctx.parameters)
    n_out = len(out_names)

    inner_t = "cppode::dual<double, 0>"
    outer_t = "cppode::dual2nd<double, 0>"

    buf.write(
        f"void {modelname}_eval_ad2(double* x, double* p,\n"
        f"                          double* dX, double* dP,\n"
        f"                          double* dX2_in, double* dP2_in,\n"
        f"                          int* has_dX2, int* has_dP2,\n"
        f"                          double* y, double* dy, double* d2y,\n"
        f"                          int* n_obs_p, int* n_vars_p, int* n_params_p,\n"
        f"                          int* n_out_p, int* n_theta_p) {{\n"
    )
    buf.write(f"    using inner_t = {inner_t};\n")
    buf.write(f"    using AD      = {outer_t};\n")
    buf.write("    const int n_obs    = *n_obs_p;\n")
    buf.write("    const int n_vars   = *n_vars_p;\n")
    buf.write("    const int n_params = *n_params_p;\n")
    buf.write("    const int n_out    = *n_out_p;\n")
    buf.write("    const int n_theta  = *n_theta_p;\n")
    buf.write("    (void)n_vars; (void)n_params; (void)n_out;\n")
    buf.write("    (void)dX; (void)dP;\n")
    buf.write("    const double* dX2 = (*has_dX2) ? dX2_in : nullptr;\n")
    buf.write("    const double* dP2 = (*has_dP2) ? dP2_in : nullptr;\n")
    buf.write("    (void)dX2; (void)dP2;\n\n")

    buf.write("    cppode::dual_arena::scope _eval_ad2_scope;\n")
    buf.write(f"    std::vector<AD> x_ad({n_vars});\n")
    buf.write(f"    std::vector<AD> p_ad({n_params});\n")
    buf.write(f"    std::vector<AD> y_ad({n_out});\n\n")

    if n_params > 0:
        buf.write("    // Seed parameters once (constant across obs).\n")
        buf.write("    for (int j = 0; j < n_params; ++j) {\n")
        buf.write("        // Innermost scalar value\n")
        buf.write("        p_ad[j].scalar() = p[j];\n")
        buf.write("        if (n_theta > 0) {\n")
        buf.write("            // Outer tangent layer: inline gradient slot k = dP[j, k];\n")
        buf.write("            // inner tangent slot m = dP2[j, k, m] (Hessian row k).\n")
        buf.write("            p_ad[j].diff(0, n_theta);\n")
        buf.write("            for (int k = 0; k < n_theta; ++k) {\n")
        buf.write("                p_ad[j][k].x() = dP[j + (size_t)n_params * k];\n")
        buf.write("                if (dP2 != nullptr) {\n")
        buf.write("                    p_ad[j][k].diff(0, n_theta);\n")
        buf.write("                    for (int m = 0; m < n_theta; ++m)\n")
        buf.write("                        p_ad[j][k][m] = dP2[j + (size_t)n_params * (k + (size_t)n_theta * m)];\n")
        buf.write("                }\n")
        buf.write("            }\n")
        buf.write("        }\n")
        buf.write("    }\n\n")

    buf.write("    for (int obs = 0; obs < n_obs; ++obs) {\n")
    if n_vars > 0:
        buf.write("        for (int j = 0; j < n_vars; ++j) {\n")
        buf.write("            // Innermost scalar value\n")
        buf.write("            x_ad[j].scalar() = x[j + (size_t)n_vars * obs];\n")
        buf.write("            if (n_theta > 0) {\n")
        buf.write("                // Outer tangent layer: inline gradient slot k = dX[obs, j, k];\n")
        buf.write("                // inner tangent slot m = dX2[obs, j, k, m] (Hessian row k)\n")
        buf.write("                x_ad[j].diff(0, n_theta);\n")
        buf.write("                for (int k = 0; k < n_theta; ++k) {\n")
        buf.write("                    x_ad[j][k].x() = dX[obs + (size_t)n_obs * (j + (size_t)n_vars * k)];\n")
        buf.write("                    if (dX2 != nullptr) {\n")
        buf.write("                        x_ad[j][k].diff(0, n_theta);\n")
        buf.write("                        for (int m = 0; m < n_theta; ++m)\n")
        buf.write("                            x_ad[j][k][m] = dX2[obs + (size_t)n_obs * (j + (size_t)n_vars * (k + (size_t)n_theta * m))];\n")
        buf.write("                    }\n")
        buf.write("                }\n")
        buf.write("            }\n")
        buf.write("        }\n")
    buf.write(f"        {modelname}_eval_one<AD>(x_ad.data(), p_ad.data(), y_ad.data());\n")
    buf.write("        for (int i = 0; i < n_out; ++i) {\n")
    # Read through a const reference so d1_at / dd_at pick the bounds-safe
    # const overloads (which route through .d(.).d(.) and return a
    # thread-local zero when the outer or inner tangent layer is non-depend).
    # The non-const overloads use raw operator[] for armed-storage write
    # semantics: they segfault for outputs that are constant in some/all
    # parameters (e.g. `y = la` identity pass-through, where the inner
    # tangent layer is never seeded because dP2 may be NULL) by dereferencing
    # the inner tan_ == nullptr.
    buf.write("            const AD& y_const = y_ad[i];\n")
    buf.write("            y[obs + (size_t)n_obs * i] = y_const.scalar();\n")
    buf.write("            for (int k = 0; k < n_theta; ++k) {\n")
    buf.write("                // dy / d2y read from outer tangent slots (inline gradient and\n")
    buf.write("                // Hessian rows). Const overloads of d1_at / dd_at are bounds-safe;\n")
    buf.write("                // the non-const ones assume armed storage and would deref the\n")
    buf.write("                // inner tan_ for outputs constant in all parameters.\n")
    buf.write("                dy[obs + (size_t)n_obs * (i + (size_t)n_out * k)] = y_const.d1_at(k);\n")
    buf.write("                for (int m = 0; m < n_theta; ++m) {\n")
    buf.write("                    d2y[obs + (size_t)n_obs * (i + (size_t)n_out * (k + (size_t)n_theta * m))]\n")
    buf.write("                        = y_const.dd_at(k, m);\n")
    buf.write("                }\n")
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

    # Collect non-zero (output, symbol) entries, then CSE across them: shared
    # denominators / factors across df/dsym entries lift into `const double`
    # temps materialised once per obs (temps depend on x_obs/p, so they live
    # inside the obs loop).
    entries = []  # (output_i, symbol_j, sympy_expr)
    for i, out_name in enumerate(out_names):
        if out_name not in jacobian:
            continue
        for j, expr in enumerate(jacobian[out_name]):
            e = _parse_entry(expr, ctx)
            if e == 0:
                continue
            entries.append((i, j, e))

    temps, simplified = _cse_exprs([e for _, _, e in entries], prefix='_cse_jt')
    for sym, sub in temps:
        buf.write(f"        const double {sym.name} = {ctx.to_cpp(sub)};\n")
    if temps:
        buf.write("\n")

    for (i, j, _), e in zip(entries, simplified):
        # R column-major: obs + n_obs * (output + n_out * symbol)
        buf.write(f"        jac[obs + (size_t)n_obs * ({i} + n_out * {j})] = {ctx.to_cpp(e)};\n")

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

    # Collect non-zero (output, sym1, sym2) entries, then CSE across them.
    entries = []  # (output_i, sym1_j, sym2_k, sympy_expr)
    for i, out_name in enumerate(out_names):
        if out_name not in hessian:
            continue
        for j, hess_row in enumerate(hessian[out_name]):
            for k, expr in enumerate(hess_row):
                e = _parse_entry(expr, ctx)
                if e == 0:
                    continue
                entries.append((i, j, k, e))

    temps, simplified = _cse_exprs([e for _, _, _, e in entries], prefix='_cse_ht')
    for sym, sub in temps:
        buf.write(f"        const double {sym.name} = {ctx.to_cpp(sub)};\n")
    if temps:
        buf.write("\n")

    for (i, j, k, _), e in zip(entries, simplified):
        # R column-major: obs + n_obs * (output + n_out * (sym1 + n_symbols * sym2))
        buf.write(
            f"        hess[obs + (size_t)n_obs * ({i} + n_out * ({j} + n_symbols * {k}))] = {ctx.to_cpp(e)};\n"
        )

    buf.write("    }\n}\n\n")


def _write_chain_jac_wrapper(buf, modelname):
    """
    Thin extern-C wrapper around cppode::chain_jac. Performs the per-obs
    DGEMM contraction J_theta[obs, ., .] = J[obs, ., .] %*% S[obs, ., .].
    """
    buf.write(f"void {modelname}_chain_jac(double* J, double* S, double* J_theta,\n")
    buf.write("                          int* n_obs, int* n_out, int* n_diff, int* n_theta) {\n")
    buf.write("    cppode::chain_jac(J, S, J_theta, *n_obs, *n_out, *n_diff, *n_theta);\n")
    buf.write("}\n\n")


def _write_chain_hess_wrapper(buf, modelname):
    """
    Thin extern-C wrapper around cppode::chain_hess. Performs per-obs and
    per-output H_theta[obs, o, ., .] = S' H[obs, o, ., .] S
                                     + sum_i J[obs, o, i] S2[obs, i, ., .]
    where the J*S2 term is skipped when has_S2 == 0.
    """
    buf.write(f"void {modelname}_chain_hess(double* H, double* J,\n")
    buf.write("                           double* S, double* S2_in, double* H_theta,\n")
    buf.write("                           int* has_S2,\n")
    buf.write("                           int* n_obs, int* n_out, int* n_diff, int* n_theta) {\n")
    buf.write("    const double* S2 = (*has_S2) ? S2_in : nullptr;\n")
    buf.write("    cppode::chain_hess(H, J, S, S2, H_theta,\n")
    buf.write("                       *n_obs, *n_out, *n_diff, *n_theta);\n")
    buf.write("}\n\n")
