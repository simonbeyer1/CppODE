"""
CVODE / CVODES C++ code generator for CppODE
=============================================

Generates a self-contained C++ source file that compiles against SUNDIALS
(CVODE / CVODES) and exposes an `extern "C" SEXP solve_<name>(...)` entry
point with the same 15-SEXP signature as CppODE's generated models, so that
`solveODE()` can call it unchanged.

Scope of the generator:

  * method : "bdf" or "adams"          (CVODE's two multistep families)
  * dense or sparse (KLU) linear solver with analytic Jacobian
  * forward sensitivities (CVODES) with ANALYTIC sens RHS:
        dS_iS/dt = J(x,p) * S_iS + df/dp_k
    where k is the global parameter index corresponding to compile-time
    sensitivity slot iS.  For IC-typed sens slots the df/dp_k term is zero.
  * PCHIP forcings (time-dependent data inputs).  Forcings do not
    contribute to sensitivities (df/dforcing = 0 w.r.t. params).
  * rootfunc termination via CVodeRootInit
  * time- and root-triggered events via CVodeReInit

Out of scope (errored out by the R wrapper):
  * second-order sensitivities (CVODES forward mode is first-order only)

Author: Simon Beyer
"""

import os
import sympy as sp

from codegenCppODE import _safe_sympify, _to_cpp


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------

def generate_cvode_cpp(
    rhs_dict,
    params_list,
    modelname,
    outdir,
    deriv=False,
    fixed_states=None,
    fixed_params=None,
    sparse=None,
    method="bdf",
    forcings_list=None,
    events=None,
    rootfunc=None,
    ntheta=None,
    has_reparam=False,
    version="unknown",
):
    fixed_states = _as_list(fixed_states)
    fixed_params = _as_list(fixed_params)
    params_list = _as_list(params_list)
    forcings_list = _as_list(forcings_list)

    states_list = list(rhs_dict.keys())
    odes_list = list(rhs_dict.values())
    n_states = len(states_list)
    n_params = len(params_list)
    n_global = n_states + n_params
    n_forcings = len(forcings_list)

    # --- parse symbolic ---
    states_syms = {n: sp.Symbol(n, real=True) for n in states_list}
    params_syms = {n: sp.Symbol(n, real=True) for n in params_list}
    forcing_syms = {n: sp.Symbol(n, real=True) for n in forcings_list}
    t_sym = sp.Symbol("time", real=True)
    local_syms = dict(states_syms)
    local_syms.update(params_syms)
    local_syms.update(forcing_syms)
    local_syms["time"] = t_sym

    rhs_exprs = [_safe_sympify(e, local_syms) for e in odes_list]

    # --- symbolic Jacobian df/dx ---
    sym_states = [states_syms[n] for n in states_list]
    jac_mat = [[sp.Integer(0)] * n_states for _ in range(n_states)]
    for i, e in enumerate(rhs_exprs):
        free = e.free_symbols
        for j, s in enumerate(sym_states):
            if s in free:
                jac_mat[i][j] = sp.diff(e, s)

    time_derivs = [sp.diff(e, t_sym) for e in rhs_exprs]

    jac_nnz = []
    for i in range(n_states):
        for j in range(n_states):
            e = jac_mat[i][j]
            if e != 0:
                jac_nnz.append((i, j, e))

    # --- sparsity decision ---
    density = (len(jac_nnz) / (n_states * n_states)) if n_states else 1.0
    if sparse is None:
        use_sparse = (n_states >= 50) and (density <= 0.2)
    else:
        use_sparse = bool(sparse)

    # --- sens layout ---
    sens_ic_names = [s for s in states_list if s not in fixed_states]
    sens_pr_names = [p for p in params_list if p not in fixed_params]
    sens_names = sens_ic_names + sens_pr_names
    n_sens_compile = len(sens_names)

    # --- ntheta / reparametrization resolution ---
    # has_reparam=True means sens columns are theta slots; sens1ini at solve()
    # time carries the full Phi'(theta) of shape [n_states + n_params, ntheta].
    # Runtime `fixed` is not supported with reparam.
    has_reparam = bool(has_reparam)
    if ntheta is None:
        ntheta_resolved = n_sens_compile
    else:
        ntheta_resolved = int(ntheta)
    if has_reparam and not deriv:
        raise ValueError("ntheta is only meaningful with deriv=TRUE")

    sens_to_global = []
    sens_to_param_idx = []
    for nm in sens_ic_names:
        sens_to_global.append(states_list.index(nm))
        sens_to_param_idx.append(-1)
    for nm in sens_pr_names:
        pk = params_list.index(nm)
        sens_to_global.append(n_states + pk)
        sens_to_param_idx.append(pk)

    # df/dp_k only for non-fixed parameters
    df_dp_by_pk = {}
    for nm in sens_pr_names:
        pk = params_list.index(nm)
        sym = params_syms[nm]
        col = []
        for i, e in enumerate(rhs_exprs):
            if sym in e.free_symbols:
                d = sp.diff(e, sym)
                if d != 0:
                    col.append((i, d))
        df_dp_by_pk[pk] = col

    # --- method ---
    if method not in ("bdf", "adams"):
        raise ValueError(f"method must be 'bdf' or 'adams', got {method!r}")
    cv_method = "CV_BDF" if method == "bdf" else "CV_ADAMS"

    # --- rootfunc parsing ---
    # Three cases:
    #   (a) rootfunc is None                     → no root handling
    #   (b) rootfunc == "equilibrate"            → post-step threshold check
    #   (c) rootfunc is a list of expressions    → CVodeRootInit + CV_ROOT_RETURN
    rootfunc_mode = "none"
    rootfunc_exprs_cpp = []
    if rootfunc is not None:
        if isinstance(rootfunc, str):
            if rootfunc.strip().lower() == "equilibrate":
                rootfunc_mode = "equilibrate"
            else:
                rootfunc_list = [rootfunc]
                rootfunc_mode = "user"
        elif isinstance(rootfunc, (list, tuple)):
            rootfunc_list = list(rootfunc)
            rootfunc_mode = "user"
        else:
            raise ValueError(
                f"rootfunc must be 'equilibrate' or a character vector, got {type(rootfunc)}")

        if rootfunc_mode == "user":
            for expr_str in rootfunc_list:
                s = str(expr_str).strip()
                if not s:
                    continue
                e = _safe_sympify(s, local_syms)
                rootfunc_exprs_cpp.append(
                    _to_cpp(e, states_list, params_list, n_states,
                            "double", forcings_list))

    # --- Events (time- and root-triggered) ---
    # For each event we compute at codegen time:
    #   g_cpp      : C++ expression for the new value of the affected state,
    #                i.e. the full post-event value x[var_idx]_new = g(x, t, p, F)
    #                already incorporating the method (replace/add/multiply).
    #   dg_dx[i]   : C++ expression for partial g / partial x_i  (sensitivity chain rule)
    #   dg_dp[k]   : C++ expression for partial g / partial p_k
    #   dg_dt      : explicit time partial of g
    # Time events additionally get:
    #   t_expr_cpp : C++ expression for event time (may reference params)
    #   dt_dp[k]   : C++ expression for ∂t_e/∂p_k  (saltation of parameterised times)
    # Root events additionally get:
    #   r_cpp      : C++ expression for the root condition g(x, t, p)
    #   dr_dx[i]   : partial of root w.r.t. each state
    #   dr_dp[k]   : partial of root w.r.t. each param
    #   dr_dt      : partial of root w.r.t. explicit time
    #   direction  : -1 / 0 / +1
    #   terminal   : stop integration after applying, yes/no
    time_events = []
    root_events = []
    if events is not None:
        if hasattr(events, "to_dict"):
            ev_dict = events.to_dict("list")
        else:
            ev_dict = events
        list_lens = [len(v) for v in ev_dict.values()
                     if isinstance(v, (list, tuple))]
        n_ev = max(list_lens) if list_lens else 1

        def _get(key, i):
            if key not in ev_dict:
                return None
            v = ev_dict[key]
            if isinstance(v, (list, tuple)):
                return v[i] if i < len(v) else None
            return v

        def _valid(val):
            if val is None:
                return False
            # R's NA_logical crosses the reticulate bridge as a Python bool;
            # treat any bool here as "missing" since a literal True/False
            # in an event table would never be meaningful as time/value/root.
            if isinstance(val, bool):
                return False
            # pandas NaN / R NA-carrying floats come through as NaN
            try:
                if isinstance(val, float) and val != val:
                    return False
            except Exception:
                pass
            # NA_character_ in a character column reaches us as the string
            # "NA"; normalise the same set of placeholders codegenCppODE uses.
            if isinstance(val, str):
                if val.strip().lower() in {"", "na", "nan", "none"}:
                    return False
            return True

        for i in range(n_ev):
            var_raw = _get("var", i)
            if var_raw is None:
                continue
            var_name = str(var_raw)
            if var_name not in states_list:
                raise ValueError(f"Event {i}: unknown state variable '{var_name}'")
            var_idx = states_list.index(var_name)

            time_raw = _get("time", i)
            root_raw = _get("root", i)
            if not _valid(time_raw) and not _valid(root_raw):
                raise ValueError(f"Event {i}: either 'time' or 'root' is required")
            if _valid(time_raw) and _valid(root_raw):
                raise ValueError(f"Event {i}: specify exactly one of 'time' or 'root'")

            value_raw = _get("value", i)
            if not _valid(value_raw):
                raise ValueError(f"Event {i}: 'value' is required")

            method_raw = _get("method", i)
            method = str(method_raw).lower() if _valid(method_raw) else "replace"
            if method not in ("replace", "add", "multiply"):
                raise ValueError(
                    f"Event {i}: method must be replace/add/multiply, got {method!r}")

            # Value is an expression of (x, t, p, forcings) shared by time and
            # root events; we derive g(x, t, p) = new x[var_idx] from it.
            value_sym = _safe_sympify(str(value_raw), local_syms)
            state_sym = states_syms[var_name]
            if method == "replace":
                g_sym = value_sym
            elif method == "add":
                g_sym = state_sym + value_sym
            else:  # multiply
                g_sym = state_sym * value_sym

            g_cpp = _to_cpp(g_sym, states_list, params_list, n_states,
                            "double", forcings_list)
            dg_dx_cpp = [
                _to_cpp(sp.diff(g_sym, states_syms[s]),
                        states_list, params_list, n_states,
                        "double", forcings_list)
                for s in states_list
            ]
            dg_dp_cpp = [
                _to_cpp(sp.diff(g_sym, params_syms[p]),
                        states_list, params_list, n_states,
                        "double", forcings_list)
                for p in params_list
            ]
            dg_dt_cpp = _to_cpp(sp.diff(g_sym, t_sym),
                                states_list, params_list, n_states,
                                "double", forcings_list)

            if _valid(time_raw):
                # --- Time-triggered ---
                # t_e may be an expression of params; ∂t_e/∂p_k drives the
                # saltation for parameterised event times.
                t_event_sym = _safe_sympify(str(time_raw), local_syms)
                t_expr_cpp = _to_cpp(t_event_sym, states_list, params_list,
                                     n_states, "double", forcings_list)
                dt_dp_cpp = [
                    _to_cpp(sp.diff(t_event_sym, params_syms[p]),
                            states_list, params_list, n_states,
                            "double", forcings_list)
                    for p in params_list
                ]
                time_events.append({
                    "t_expr_cpp": t_expr_cpp,
                    "var_idx": var_idx,
                    "g_cpp": g_cpp,
                    "dg_dx_cpp": dg_dx_cpp,
                    "dg_dp_cpp": dg_dp_cpp,
                    "dg_dt_cpp": dg_dt_cpp,
                    "dt_dp_cpp": dt_dp_cpp,
                })
            else:
                # --- Root-triggered ---
                # The event fires when r(x, t, p) crosses zero.  Saltation uses
                # the IFT: dt_root/dp_k = -(dr/dp_k + Σ dr/dx_i · S_i[k]) / ġ,
                # where ġ = Σ dr/dx_i · f_i + dr/dt  (the total time derivative
                # of r along the trajectory).
                r_sym = _safe_sympify(str(root_raw), local_syms)
                r_cpp = _to_cpp(r_sym, states_list, params_list, n_states,
                                "double", forcings_list)
                dr_dx_cpp = [
                    _to_cpp(sp.diff(r_sym, states_syms[s]),
                            states_list, params_list, n_states,
                            "double", forcings_list)
                    for s in states_list
                ]
                dr_dp_cpp = [
                    _to_cpp(sp.diff(r_sym, params_syms[p]),
                            states_list, params_list, n_states,
                            "double", forcings_list)
                    for p in params_list
                ]
                dr_dt_cpp = _to_cpp(sp.diff(r_sym, t_sym),
                                    states_list, params_list, n_states,
                                    "double", forcings_list)

                direction_raw = _get("direction", i)
                try:
                    direction = int(direction_raw) if _valid(direction_raw) else 0
                except (ValueError, TypeError):
                    direction = 0
                if direction not in (-1, 0, 1):
                    raise ValueError(
                        f"Event {i}: direction must be -1, 0, or 1, got {direction}")

                # Terminal is a bool column — accept Python bool directly
                # (don't funnel through `_valid`, which rejects bools since
                # they stand in for R NA_logical in the time/root columns).
                terminal_raw = _get("terminal", i)
                if isinstance(terminal_raw, bool):
                    terminal = terminal_raw
                elif terminal_raw is None:
                    terminal = False
                else:
                    try:
                        terminal = bool(terminal_raw)
                    except Exception:
                        terminal = False

                root_events.append({
                    "var_idx": var_idx,
                    "g_cpp": g_cpp,
                    "dg_dx_cpp": dg_dx_cpp,
                    "dg_dp_cpp": dg_dp_cpp,
                    "dg_dt_cpp": dg_dt_cpp,
                    "r_cpp": r_cpp,
                    "dr_dx_cpp": dr_dx_cpp,
                    "dr_dp_cpp": dr_dp_cpp,
                    "dr_dt_cpp": dr_dt_cpp,
                    "direction": direction,
                    "terminal": terminal,
                })

    # --- CSC ordering for sparse ---
    csc_entries = sorted(jac_nnz, key=lambda rce: (rce[1], rce[0]))
    colptr = [0] * (n_states + 1)
    for _, c, _ in csc_entries:
        colptr[c + 1] += 1
    for k in range(1, len(colptr)):
        colptr[k] += colptr[k - 1]
    rowval = [r for (r, _, _) in csc_entries]
    nnz_total = len(csc_entries)

    # --- cpp expression helper (uses CppODE's _to_cpp: x[i], params[j],
    #     (*F[k])(t) for forcings) ---
    def cpp_of(expr):
        return _to_cpp(expr, states_list, params_list, n_states, "double",
                       forcings_list)

    # ---------------------------------------------------------------
    # Body snippets
    # ---------------------------------------------------------------

    ode_body = "\n".join(f"    ydot_arr[{i}] = {cpp_of(e)};"
                         for i, e in enumerate(rhs_exprs))

    jac_body_dense = "\n".join(f"    SM_ELEMENT_D(J, {i}, {j}) = {cpp_of(e)};"
                               for (i, j, e) in jac_nnz)
    if not jac_body_dense:
        jac_body_dense = "    // zero Jacobian"

    # Sparse: always write the pattern (cheap) + data.
    sp_lines = []
    sp_lines.append("    sunindextype* indexptrs = SUNSparseMatrix_IndexPointers(J);")
    sp_lines.append("    sunindextype* indexvals = SUNSparseMatrix_IndexValues(J);")
    sp_lines.append("    sunrealtype*  data      = SUNSparseMatrix_Data(J);")
    colptr_str = ", ".join(str(v) for v in colptr)
    rowval_str = ", ".join(str(v) for v in rowval) if rowval else "0"
    sp_lines.append(f"    static const sunindextype _colptr[{len(colptr)}] = {{{colptr_str}}};")
    sp_lines.append(f"    static const sunindextype _rowval[{max(1, len(rowval))}] = {{{rowval_str}}};")
    sp_lines.append(f"    for (int k = 0; k <= {n_states}; ++k) indexptrs[k] = _colptr[k];")
    sp_lines.append(f"    for (int k = 0; k < {len(rowval)}; ++k) indexvals[k] = _rowval[k];")
    for ax_k, (r, c, e) in enumerate(csc_entries):
        sp_lines.append(f"    data[{ax_k}] = {cpp_of(e)};")
    jac_body_sparse = "\n".join(sp_lines)

    # Sens: J*yS
    jyS_lines = [f"    ySdot_arr[{i}] = 0.0;" for i in range(n_states)]
    for i, j, e in jac_nnz:
        jyS_lines.append(f"    ySdot_arr[{i}] += ({cpp_of(e)}) * yS_arr[{j}];")
    jac_times_yS_body = "\n".join(jyS_lines)

    # Sens: df/dp switch
    sw_lines = []
    for pk in sorted(df_dp_by_pk.keys()):
        sw_lines.append(f"        case {pk}: {{")
        for (i, d_expr) in df_dp_by_pk[pk]:
            sw_lines.append(f"          ySdot_arr[{i}] += {cpp_of(d_expr)};")
        sw_lines.append("          break;")
        sw_lines.append("        }")
    df_dp_switch_body = "\n".join(sw_lines)

    # Constant tables
    n_sens_arr = max(1, n_sens_compile)
    s2g_str = ", ".join(str(v) for v in sens_to_global) if sens_to_global else "-1"
    s2pk_str = ", ".join(str(v) for v in sens_to_param_idx) if sens_to_param_idx else "-1"

    # --- Compile-time preprocessor defs (only codegen-owned).
    # Linker flags are supplied by the R wrapper from the configure-time
    # detection (the `cvodeConfig` list at the top of R/CVODE.R, patched
    # in place by ./configure), not by codegen, so that the package
    # stays portable across Linux distros and macOS/Homebrew.
    compile_defs = []
    if use_sparse:
        compile_defs.append("-DCVODE_KLU")

    # Emit df/dp contributions indexed by (state_row, param_index, expr) — used
    # under reparam to accumulate (df/dp) * M[NEQ + pk, iS] for each iS.
    df_dp_entries = []  # list of (state_i, param_pk, sympy_expr)
    for pk in sorted(df_dp_by_pk.keys()):
        for (i, d_expr) in df_dp_by_pk[pk]:
            df_dp_entries.append((i, pk, d_expr))

    # --- Assemble source ---
    src = _render_source(
        modelname=modelname, version=version,
        n_states=n_states, n_global=n_global,
        n_sens_compile=n_sens_compile, n_sens_arr=n_sens_arr,
        s2g_str=s2g_str, s2pk_str=s2pk_str,
        ode_body=ode_body,
        jac_body_dense=jac_body_dense,
        jac_body_sparse=jac_body_sparse,
        nnz_total=nnz_total,
        jac_times_yS_body=jac_times_yS_body,
        df_dp_switch_body=df_dp_switch_body,
        df_dp_entries=df_dp_entries,
        cv_method=cv_method,
        deriv=deriv,
        use_sparse=use_sparse,
        n_forcings=n_forcings,
        rootfunc_mode=rootfunc_mode,
        rootfunc_exprs_cpp=rootfunc_exprs_cpp,
        time_events=time_events,
        root_events=root_events,
        n_params=n_params,
        ntheta=ntheta_resolved,
        has_reparam=has_reparam,
        states_list=states_list,
        params_list=params_list,
        forcings_list=forcings_list,
    )

    srcfile = os.path.join(outdir, f"{modelname}.cpp")
    if os.path.exists(srcfile):
        print(f"Overwriting existing file: {srcfile}")
    with open(srcfile, "w") as f:
        f.write(src)

    return {
        "srcfile": srcfile,
        "variables": states_list,
        "parameters": params_list,
        "forcings": forcings_list,
        "sens_names": sens_names,
        "jac_nnz_rows": [r for (r, _, _) in jac_nnz],
        "jac_nnz_cols": [c for (_, c, _) in jac_nnz],
        "jac_nnz_exprs": [str(e) for (_, _, e) in jac_nnz],
        "time_derivs": [str(td) if td != 0 else "0" for td in time_derivs],
        "use_sparse": use_sparse,
        "compile_defs": compile_defs,
    }


# =====================================================================
# C++ source template
# =====================================================================

def _render_source(
    modelname, version,
    n_states, n_global,
    n_sens_compile, n_sens_arr,
    s2g_str, s2pk_str,
    ode_body, jac_body_dense, jac_body_sparse, nnz_total,
    jac_times_yS_body, df_dp_switch_body,
    cv_method, deriv, use_sparse,
    n_forcings=0,
    rootfunc_mode="none",
    rootfunc_exprs_cpp=None,
    time_events=None,
    root_events=None,
    n_params=0,
    df_dp_entries=None,
    ntheta=0,
    has_reparam=False,
    states_list=None,
    params_list=None,
    forcings_list=None,
):
    deriv_flag = "true" if deriv else "false"
    has_reparam_cpp = "true" if has_reparam else "false"
    has_forcings = n_forcings > 0
    if rootfunc_exprs_cpp is None:
        rootfunc_exprs_cpp = []
    if time_events is None:
        time_events = []
    if root_events is None:
        root_events = []
    has_time_events = len(time_events) > 0
    has_root_events = len(root_events) > 0
    has_events      = has_time_events or has_root_events

    # Forcings/events: PchipForcing storage lives in UserData so the
    # SUNDIALS callbacks (and event lambdas) can read it.  The `F` vector
    # is always present so event lambdas can uniformly capture it —
    # empty if the model has no forcings.  Forcings contribute to df/dx
    # (and to rhs/sens rhs via evaluation), but NOT to df/dp.
    need_pchip = has_forcings  # include the PCHIP header
    # Event lambdas always capture F even when no forcings, so UserData
    # must have F available.  With events-only we still include pchip.
    if has_events and not has_forcings:
        need_pchip = True

    if need_pchip:
        forcing_include = "#include <cppode/cppode_pchip_forcing.hpp>\n"
        forcing_ud_members = (
            "  std::vector<cppode::PchipForcing<double>> forcing_storage;\n"
            "  std::vector<const cppode::PchipForcing<double>*> F;\n"
        )
        forcing_local = "  const auto& F = ud->F;\n  (void)F;"
    else:
        forcing_include = ""
        forcing_ud_members = ""
        forcing_local = ""

    # Root-event fire counters live in UserData so the C-style root_fn
    # callback can see them (they're modified from the main loop).  Kept
    # small so unused models pay nothing extra.
    events_ud_members = (
        "  std::vector<int> root_fired;           // per event-root fire count\n"
        "  int              maxroot = 1;          // cap from solveODE(maxroot=)\n"
    )

    # Reparam: UserData carries Phi_prime (Phi' flat, column-major) so the
    # sens rhs and event saltation lambdas can apply the chain rule.
    if has_reparam:
        reparam_ud_members = (
            "  std::vector<double> Phi_prime;         // (NEQ + n_params) * NTHETA, column-major\n"
        )
    else:
        reparam_ud_members = ""

    if has_forcings:
        forcing_init_block = f"""  // --- Initialize forcings (PCHIP interpolation) ---
  {{
    if (Rf_isNull(forcingTimesSEXP) || Rf_isNull(forcingValuesSEXP))
      Rf_error("Model requires %d forcing(s) but none supplied", {n_forcings});
    int n_f_in = Rf_length(forcingTimesSEXP);
    if (n_f_in != {n_forcings})
      Rf_error("Forcing count mismatch: model expects %d, got %d", {n_forcings}, n_f_in);
    ud.forcing_storage.resize({n_forcings});
    ud.F.resize({n_forcings});
    for (int fi = 0; fi < {n_forcings}; ++fi) {{
      SEXP times_i  = VECTOR_ELT(forcingTimesSEXP,  fi);
      SEXP values_i = VECTOR_ELT(forcingValuesSEXP, fi);
      int np = Rf_length(times_i);
      std::vector<double> ft(REAL(times_i),  REAL(times_i)  + np);
      std::vector<double> fv(REAL(values_i), REAL(values_i) + np);
      ud.forcing_storage[fi].initialize(ft, fv);
      ud.F[fi] = &ud.forcing_storage[fi];
    }}
  }}
"""
    else:
        forcing_init_block = ""

    # --- Rootfunc / event-root registration ---
    # CVodeRootInit is driven by the combined set of user rootfunc
    # expressions and any root-triggered events.  The root_fn writes
    # gout[0..n_user-1] from user rootfunc, then gout[n_user..] from
    # event roots.  On CV_ROOT_RETURN the integration loop calls
    # CVodeGetRootInfo and dispatches: user roots terminate (like the
    # standalone rootfunc path); event roots apply their state change
    # and saltation, then reinit and continue (unless marked terminal).
    n_user_rootfunc = len(rootfunc_exprs_cpp) if rootfunc_mode == "user" else 0
    n_event_roots   = len(root_events)
    n_total_roots   = n_user_rootfunc + n_event_roots
    has_user_root   = rootfunc_mode == "user"
    need_cvode_root = (n_total_roots > 0)

    if need_cvode_root:
        root_gout_lines = []
        for i, expr_cpp in enumerate(rootfunc_exprs_cpp if has_user_root else []):
            root_gout_lines.append(f"  gout[{i}] = {expr_cpp};")
        # Exhausted event roots emit a constant +1 so no further sign change is
        # ever seen by CVode's rootfinder.  Between the change-over step and
        # the CVodeReInit that follows event application the root history is
        # reset, so no phantom crossing is registered from the -ε → +1 jump.
        for j, e in enumerate(root_events):
            root_gout_lines.append(
                f"  gout[{n_user_rootfunc + j}] = "
                f"(ud->root_fired[{j}] >= ud->maxroot) ? 1.0 : ({e['r_cpp']});")
        root_gout_body = "\n".join(root_gout_lines) if root_gout_lines else "  (void)gout;"
        rootfunc_decl = ("static int root_fn(sunrealtype t, N_Vector y, "
                         "sunrealtype* gout, void* ud_vp);")
        rootfunc_impl = f"""
static int root_fn(sunrealtype t, N_Vector y, sunrealtype* gout, void* ud_vp) {{
  (void)t;
  UserData* ud = static_cast<UserData*>(ud_vp);
  const double* params = ud->params.data();
  const double* x = N_VGetArrayPointer(y);
  (void)x; (void)params;
{forcing_local}
{root_gout_body}
  return 0;
}}
"""
        # Build direction array for CVodeSetRootDirection.  User roots
        # get direction 0 (any crossing); event roots use their spec.
        direction_entries = ["0"] * n_user_rootfunc + [
            str(e["direction"]) for e in root_events
        ]
        direction_list = ", ".join(direction_entries)
        rootfunc_init_block = (
            f"  {{ int rd[{n_total_roots}] = {{ {direction_list} }};\n"
            f"    if (CVodeRootInit(cvode_mem, {n_total_roots}, root_fn) < 0) "
            "{ cleanup(); Rf_error(\"CVodeRootInit failed\"); }\n"
            f"    if (CVodeSetRootDirection(cvode_mem, rd) < 0) "
            "{ cleanup(); Rf_error(\"CVodeSetRootDirection failed\"); } }\n"
        )
    else:
        rootfunc_decl = ""
        rootfunc_impl = ""
        rootfunc_init_block = ""

    # --- Events (time and root) ---
    # We emit structs + builders conditional on what is present in the
    # model.  Time events land in a sorted std::vector<TimeEvent>,
    # traversed in the main loop; root events land in std::vector<RootEvent>,
    # triggered by CVodeRootInit / CV_ROOT_RETURN dispatch.
    event_block_includes = "#include <functional>\n" if has_events else ""
    event_struct = ""
    if has_time_events:
        event_struct += f"""
struct TimeEvent {{
  double time;
  int    var_idx;
  // All lambdas capture `params` (pointer) and `F` (ref) from the builder.
  std::function<double(const double* x, double t)> g_fn;
  // dg/dx_i as a single dispatch lambda — returns 0 outside range.
  std::function<double(const double* x, double t, int i)> dg_dx_fn;
  // dg/dp_k (explicit partial, at fixed t_e).
  std::function<double(const double* x, double t, int k)> dg_dp_fn;
  // dg/dt (explicit time partial of g, at fixed x/p).
  std::function<double(const double* x, double t)> dg_dt_fn;
  // dt_e/dp_k (saltation correction for parameterized event times).
  std::function<double(const double* x, double t, int k)> dt_dp_fn;
}};

static std::vector<TimeEvent> build_time_events(const double* params,
                                                const std::vector<const cppode::PchipForcing<double>*>& F) {{
  (void)params; (void)F;
  std::vector<TimeEvent> ev;
"""
        ev_body = []
        for i, e in enumerate(time_events):
            t_cpp = e["t_expr_cpp"]
            v_idx = e["var_idx"]
            g_cpp = e["g_cpp"]
            dg_dt_cpp = e["dg_dt_cpp"]
            dx_cases = "\n".join(
                f"        case {j}: return {dx};"
                for j, dx in enumerate(e["dg_dx_cpp"])
            )
            dp_cases = "\n".join(
                f"        case {k}: return {dp};"
                for k, dp in enumerate(e["dg_dp_cpp"])
            )
            dt_dp_cases = "\n".join(
                f"        case {k}: return {dtp};"
                for k, dtp in enumerate(e["dt_dp_cpp"])
            )
            # Inside lambdas, `x` is already a parameter, `t` is a parameter,
            # `params` and `F` are captures.  _to_cpp emitted `x[i]`,
            # `params[...]`, `(*F[...])(t)`, `t` — all resolve here.
            ev_body.append(f"""  {{
    TimeEvent e;
    e.time    = {t_cpp};
    e.var_idx = {v_idx};
    e.g_fn    = [params, &F](const double* x, double t) -> double {{
      return {g_cpp};
    }};
    e.dg_dx_fn = [params, &F](const double* x, double t, int i) -> double {{
      switch (i) {{
{dx_cases}
        default: return 0.0;
      }}
    }};
    e.dg_dp_fn = [params, &F](const double* x, double t, int k) -> double {{
      switch (k) {{
{dp_cases}
        default: return 0.0;
      }}
    }};
    e.dg_dt_fn = [params, &F](const double* x, double t) -> double {{
      return {dg_dt_cpp};
    }};
    e.dt_dp_fn = [params, &F](const double* x, double t, int k) -> double {{
      switch (k) {{
{dt_dp_cases}
        default: return 0.0;
      }}
    }};
    ev.push_back(std::move(e));
  }}""")
        event_struct += "\n".join(ev_body) + """
  std::sort(ev.begin(), ev.end(),
            [](const TimeEvent& a, const TimeEvent& b) { return a.time < b.time; });
  return ev;
}
"""

    if has_root_events:
        event_struct += f"""
struct RootEvent {{
  int  var_idx;
  int  direction;
  bool terminal;
  // Event map g(x, t, p) and its partials (for saltation)
  std::function<double(const double* x, double t)> g_fn;
  std::function<double(const double* x, double t, int i)> dg_dx_fn;
  std::function<double(const double* x, double t, int k)> dg_dp_fn;
  std::function<double(const double* x, double t)> dg_dt_fn;
  // Root condition r(x, t, p) and its partials (for IFT → dt_root/dp)
  std::function<double(const double* x, double t, int i)> dr_dx_fn;
  std::function<double(const double* x, double t, int k)> dr_dp_fn;
  std::function<double(const double* x, double t)> dr_dt_fn;
}};

static std::vector<RootEvent> build_root_events(const double* params,
                                                const std::vector<const cppode::PchipForcing<double>*>& F) {{
  (void)params; (void)F;
  std::vector<RootEvent> ev;
"""
        re_body = []
        for i, e in enumerate(root_events):
            v_idx = e["var_idx"]
            g_cpp = e["g_cpp"]
            dg_dt_cpp = e["dg_dt_cpp"]
            dr_dt_cpp = e["dr_dt_cpp"]
            direction = e["direction"]
            terminal = "true" if e["terminal"] else "false"
            dgx_cases = "\n".join(
                f"        case {j}: return {dx};"
                for j, dx in enumerate(e["dg_dx_cpp"])
            )
            dgp_cases = "\n".join(
                f"        case {k}: return {dp};"
                for k, dp in enumerate(e["dg_dp_cpp"])
            )
            drx_cases = "\n".join(
                f"        case {j}: return {dx};"
                for j, dx in enumerate(e["dr_dx_cpp"])
            )
            drp_cases = "\n".join(
                f"        case {k}: return {dp};"
                for k, dp in enumerate(e["dr_dp_cpp"])
            )
            re_body.append(f"""  {{
    RootEvent e;
    e.var_idx   = {v_idx};
    e.direction = {direction};
    e.terminal  = {terminal};
    e.g_fn    = [params, &F](const double* x, double t) -> double {{
      return {g_cpp};
    }};
    e.dg_dx_fn = [params, &F](const double* x, double t, int i) -> double {{
      switch (i) {{
{dgx_cases}
        default: return 0.0;
      }}
    }};
    e.dg_dp_fn = [params, &F](const double* x, double t, int k) -> double {{
      switch (k) {{
{dgp_cases}
        default: return 0.0;
      }}
    }};
    e.dg_dt_fn = [params, &F](const double* x, double t) -> double {{
      return {dg_dt_cpp};
    }};
    e.dr_dx_fn = [params, &F](const double* x, double t, int i) -> double {{
      switch (i) {{
{drx_cases}
        default: return 0.0;
      }}
    }};
    e.dr_dp_fn = [params, &F](const double* x, double t, int k) -> double {{
      switch (k) {{
{drp_cases}
        default: return 0.0;
      }}
    }};
    e.dr_dt_fn = [params, &F](const double* x, double t) -> double {{
      return {dr_dt_cpp};
    }};
    ev.push_back(std::move(e));
  }}""")
        event_struct += "\n".join(re_body) + """
  return ev;
}
"""

    # ---- Conditional sens code snippets ----
    # Defined here (rather than alongside the linear-solver setup lower
    # down) because the event and main-loop blocks below reference them.
    if deriv and has_reparam:
        phi_rows = n_states + n_params
        sens_init_block = f"""  if (Ns_active > 0) {{
    yS = N_VCloneVectorArray(Ns_active, y);
    Ns_alloc = Ns_active;
    if (!yS) {{ cleanup(); Rf_error("N_VCloneVectorArray failed"); }}
    // Under reparametrization sens1ini carries the full Phi'(theta) of shape
    // [NEQ + n_params, NTHETA] (column-major); copy to UserData for the
    // sens RHS callback and event-saltation chain rule.
    const int phi_rows_rt = {phi_rows};
    const int expected_len = phi_rows_rt * Ns_active;
    if (Rf_isNull(sens1iniSEXP)) {{
      cleanup();
      Rf_error("sens1ini is required when the model is compiled with ntheta");
    }}
    if (Rf_length(sens1iniSEXP) != expected_len) {{
      cleanup();
      Rf_error("sens1ini length %d != expected %d (phi_rows * ntheta)",
               Rf_length(sens1iniSEXP), expected_len);
    }}
    ud.Phi_prime.assign(REAL(sens1iniSEXP), REAL(sens1iniSEXP) + expected_len);
    // yS0[iS][i] = Phi'[i, iS] for state-row i
    for (int iS = 0; iS < Ns_active; ++iS) {{
      double* v = N_VGetArrayPointer(yS[iS]);
      for (int i = 0; i < NEQ; ++i) v[i] = ud.Phi_prime[i + phi_rows_rt * iS];
    }}
    if (CVodeSensInit1(cvode_mem, Ns_active, CV_STAGGERED, sens_rhs1_fn, yS) < 0) {{
      cleanup(); Rf_error("CVodeSensInit1 failed");
    }}
    if (CVodeSensEEtolerances(cvode_mem) < 0) {{
      cleanup(); Rf_error("CVodeSensEEtolerances failed");
    }}
    if (CVodeSetSensErrCon(cvode_mem, SUNTRUE) < 0) {{
      cleanup(); Rf_error("CVodeSetSensErrCon failed");
    }}
  }}
"""
    elif deriv:
        sens_init_block = """  if (Ns_active > 0) {
    yS = N_VCloneVectorArray(Ns_active, y);
    Ns_alloc = Ns_active;
    if (!yS) { cleanup(); Rf_error("N_VCloneVectorArray failed"); }
    const bool has_s1ini = !Rf_isNull(sens1iniSEXP);
    const double* s1 = has_s1ini ? REAL(sens1iniSEXP) : nullptr;
    if (has_s1ini && Rf_length(sens1iniSEXP) != NEQ * Ns_active) {
      cleanup();
      Rf_error("sens1ini length != NEQ * Ns_active (%d * %d)", NEQ, Ns_active);
    }
    for (int iS = 0; iS < Ns_active; ++iS) {
      double* v = N_VGetArrayPointer(yS[iS]);
      for (int i = 0; i < NEQ; ++i) v[i] = 0.0;
      if (has_s1ini) {
        for (int i = 0; i < NEQ; ++i) v[i] = s1[i + NEQ * iS];
      } else {
        const int c = ud.active_to_compile[iS];
        const int g = kSensToGlobal[c];
        if (g < NEQ) v[g] = 1.0;
      }
    }
    if (CVodeSensInit1(cvode_mem, Ns_active, CV_STAGGERED, sens_rhs1_fn, yS) < 0) {
      cleanup(); Rf_error("CVodeSensInit1 failed");
    }
    if (CVodeSensEEtolerances(cvode_mem) < 0) {
      cleanup(); Rf_error("CVodeSensEEtolerances failed");
    }
    if (CVodeSetSensErrCon(cvode_mem, SUNTRUE) < 0) {
      cleanup(); Rf_error("CVodeSetSensErrCon failed");
    }
  }
"""
    else:
        sens_init_block = ""

    if deriv:
        sens_t0_block = """    for (int iS = 0; iS < Ns_active; ++iS) {
      double* v = N_VGetArrayPointer(yS[iS]);
      for (int i = 0; i < NEQ; ++i) out_s.push_back(v[i]);
    }"""
        sens_get_block = """    {
      int flag_gs = CVodeGetSens(cvode_mem, &tret, yS);
      if (flag_gs < 0) {
        return_code = flag_gs;
        solver_msg = "CVodeGetSens failed";
        break;
      }
    }"""
        sens_store_block = """    for (int iS = 0; iS < Ns_active; ++iS) {
      double* v = N_VGetArrayPointer(yS[iS]);
      for (int i = 0; i < NEQ; ++i) out_s.push_back(v[i]);
    }"""
        sens_fevals_block = """  {
    long nfeS = 0;
    CVodeGetSensNumRhsEvals(cvode_mem, &nfeS);
    n_fe += nfeS;
  }"""
    else:
        sens_t0_block = ""
        sens_get_block = ""
        sens_store_block = ""
        sens_fevals_block = ""

    # Event helpers emitted into the entry point.
    # `apply_event` updates y[var_idx] and (if deriv) yS[*][var_idx]
    # using the precomputed symbolic partials dg/dx, dg/dp.
    # Shared saltation formula body (lives inside a lambda with captures
    # to y, f_buf, yS, Ns_active, ud).  Templated on event type via the
    # `ev` handle; works for both TimeEvent (uses ev.dt_dp_fn directly)
    # and RootEvent (IFT-computed dt_root_dp passed in).
    event_sens_reinit = ""
    time_event_apply_lambda = ""
    root_event_apply_lambda = ""

    if has_events:
        event_builder_block = (
            "  f_buf = N_VClone(y);\n"
            "  if (!f_buf) { cleanup(); Rf_error(\"N_VClone failed for event scratch\"); }\n"
        )
        if has_time_events:
            event_builder_block += (
                "  auto time_events = build_time_events(ud.params.data(), ud.F);\n"
            )
        if has_root_events:
            event_builder_block += (
                "  auto event_roots = build_root_events(ud.params.data(), ud.F);\n"
            )
        if deriv:
            event_sens_reinit = ("      if (CVodeSensReInit(cvode_mem, CV_STAGGERED, yS) < 0) "
                                 "{ cleanup(); Rf_error(\"CVodeSensReInit failed\"); }\n")
    else:
        event_builder_block = ""

    if has_time_events:
        # Full first-order saltation for parameterised time events
        # (Barton & Lee; derived from the flow's dependence on the
        # shifted event-time boundary):
        #   S_new[var]     = Σ dg/dx_i · S_old[i] + dg/dp_k
        #                  + (Σ dg/dx_i · f_old[i] + dg/dt − f_new[var]) · dt_e/dp_k
        #   S_new[j ≠ var] = S_old[j] + (f_old[j] − f_new[j]) · dt_e/dp_k
        # IC sens slots: dt_e/dp_k is zero (t_e depends on parameters only),
        # so the time-shift term vanishes and we fall back to the
        # fixed-time formula.
        #
        # Under reparametrization (HAS_REPARAM), iS indexes theta-slots;
        # we apply chain-rule at the call site: dt_e/dθ_iS = Σ_k dt_e/dp_k · M[NEQ+k, iS]
        # with M taken from ud.Phi_prime. Non-reparam: iS maps to a single
        # param slot (or IC slot), no summation needed.
        if deriv and has_reparam:
            phi_rows_e = n_states + n_params
            time_event_apply_lambda = f"""  auto apply_time_event = [&](const TimeEvent& ev) {{
    double* y_arr = N_VGetArrayPointer(y);
    double t_e = ev.time;
    std::vector<double> x_old(NEQ);
    for (int i = 0; i < NEQ; ++i) x_old[i] = y_arr[i];

    rhs_fn(t_e, y, f_buf, &ud);
    std::vector<double> f_old(NEQ);
    {{ const double* fb = N_VGetArrayPointer(f_buf);
      for (int i = 0; i < NEQ; ++i) f_old[i] = fb[i]; }}

    double new_v = ev.g_fn(x_old.data(), t_e);
    y_arr[ev.var_idx] = new_v;

    rhs_fn(t_e, y, f_buf, &ud);
    std::vector<double> f_new(NEQ);
    {{ const double* fb = N_VGetArrayPointer(f_buf);
      for (int i = 0; i < NEQ; ++i) f_new[i] = fb[i]; }}

    const double dg_dt = ev.dg_dt_fn(x_old.data(), t_e);
    const int phi_rows_rt = {phi_rows_e};

    for (int iS = 0; iS < Ns_active; ++iS) {{
      double* yS_k = N_VGetArrayPointer(yS[iS]);
      std::vector<double> S_old(NEQ);
      for (int i = 0; i < NEQ; ++i) S_old[i] = yS_k[i];

      // Chain-rule over model-param slots via Phi'(theta) param block.
      double dt_dp = 0.0;
      double dg_dp_k = 0.0;
      for (int pk = 0; pk < {n_params}; ++pk) {{
        double M = ud.Phi_prime[(NEQ + pk) + phi_rows_rt * iS];
        if (M == 0.0) continue;
        dt_dp   += ev.dt_dp_fn(x_old.data(), t_e, pk) * M;
        dg_dp_k += ev.dg_dp_fn(x_old.data(), t_e, pk) * M;
      }}

      double sum_gx_S = 0.0, sum_gx_f = 0.0;
      for (int i = 0; i < NEQ; ++i) {{
        double gx_i = ev.dg_dx_fn(x_old.data(), t_e, i);
        sum_gx_S += gx_i * S_old[i];
        sum_gx_f += gx_i * f_old[i];
      }}

      yS_k[ev.var_idx] = sum_gx_S + dg_dp_k
                       + (sum_gx_f + dg_dt - f_new[ev.var_idx]) * dt_dp;

      if (dt_dp != 0.0) {{
        for (int j = 0; j < NEQ; ++j) {{
          if (j == ev.var_idx) continue;
          yS_k[j] = S_old[j] + (f_old[j] - f_new[j]) * dt_dp;
        }}
      }}
    }}
  }};
"""
        elif deriv:
            time_event_apply_lambda = """  auto apply_time_event = [&](const TimeEvent& ev) {
    double* y_arr = N_VGetArrayPointer(y);
    double t_e = ev.time;
    std::vector<double> x_old(NEQ);
    for (int i = 0; i < NEQ; ++i) x_old[i] = y_arr[i];

    rhs_fn(t_e, y, f_buf, &ud);
    std::vector<double> f_old(NEQ);
    { const double* fb = N_VGetArrayPointer(f_buf);
      for (int i = 0; i < NEQ; ++i) f_old[i] = fb[i]; }

    double new_v = ev.g_fn(x_old.data(), t_e);
    y_arr[ev.var_idx] = new_v;

    rhs_fn(t_e, y, f_buf, &ud);
    std::vector<double> f_new(NEQ);
    { const double* fb = N_VGetArrayPointer(f_buf);
      for (int i = 0; i < NEQ; ++i) f_new[i] = fb[i]; }

    const double dg_dt = ev.dg_dt_fn(x_old.data(), t_e);

    for (int iS = 0; iS < Ns_active; ++iS) {
      int compile_idx = ud.active_to_compile[iS];
      int global_idx  = kSensToGlobal[compile_idx];
      double* yS_k = N_VGetArrayPointer(yS[iS]);
      std::vector<double> S_old(NEQ);
      for (int i = 0; i < NEQ; ++i) S_old[i] = yS_k[i];

      double dt_dp = 0.0;
      double dg_dp_k = 0.0;
      if (global_idx >= NEQ) {
        int pk = global_idx - NEQ;
        dt_dp   = ev.dt_dp_fn(x_old.data(), t_e, pk);
        dg_dp_k = ev.dg_dp_fn(x_old.data(), t_e, pk);
      }

      double sum_gx_S = 0.0, sum_gx_f = 0.0;
      for (int i = 0; i < NEQ; ++i) {
        double gx_i = ev.dg_dx_fn(x_old.data(), t_e, i);
        sum_gx_S += gx_i * S_old[i];
        sum_gx_f += gx_i * f_old[i];
      }

      yS_k[ev.var_idx] = sum_gx_S + dg_dp_k
                       + (sum_gx_f + dg_dt - f_new[ev.var_idx]) * dt_dp;

      if (dt_dp != 0.0) {
        for (int j = 0; j < NEQ; ++j) {
          if (j == ev.var_idx) continue;
          yS_k[j] = S_old[j] + (f_old[j] - f_new[j]) * dt_dp;
        }
      }
    }
  };
"""
        else:
            time_event_apply_lambda = """  auto apply_time_event = [&](const TimeEvent& ev) {
    double* y_arr = N_VGetArrayPointer(y);
    double t_e = ev.time;
    std::vector<double> x_old(NEQ);
    for (int i = 0; i < NEQ; ++i) x_old[i] = y_arr[i];
    y_arr[ev.var_idx] = ev.g_fn(x_old.data(), t_e);
  };
"""
        # Pre-t0 time events are applied before we write the t0 row.
        event_pre_t0_block = f"""  {{
    size_t applied_pre = 0;
    while (applied_pre < time_events.size() && time_events[applied_pre].time <= t0) {{
      apply_time_event(time_events[applied_pre]);
      applied_pre++;
    }}
    if (applied_pre > 0) {{
      if (CVodeReInit(cvode_mem, t0, y) < 0)
        {{ cleanup(); Rf_error("CVodeReInit failed (pre-t0 events)"); }}
{event_sens_reinit}    }}
    (void)applied_pre;
  }}
  size_t ev_idx = 0;
  while (ev_idx < time_events.size() && time_events[ev_idx].time <= t0) ev_idx++;
"""
    else:
        event_pre_t0_block = ""

    if has_root_events:
        # Batched root-event application.  Every event triggering at the
        # same t_e is resolved against a single pre-event snapshot (x_old,
        # f_old, S_old) and shares one dt*/dp per sens slot; the flow
        # correction (f_old − f_new)·dt_dp is applied at most once per
        # state.  Sequential apply would double-count the flow term on
        # non-target states whenever two events fire simultaneously (e.g.
        # two events sharing the same root function).  dt_dp is taken
        # from the first non-terminal triggered event; simultaneous roots
        # from different functions cross at the same t by construction,
        # so their dt* agrees to first order.
        if deriv and has_reparam:
            phi_rows_r = n_states + n_params
            root_event_apply_lambda = f"""  auto apply_root_events_batch = [&](const std::vector<int>& triggered_idx,
                                     double t_e) -> bool {{
    double* y_arr = N_VGetArrayPointer(y);
    std::vector<double> x_old(NEQ);
    for (int i = 0; i < NEQ; ++i) x_old[i] = y_arr[i];

    rhs_fn(t_e, y, f_buf, &ud);
    std::vector<double> f_old(NEQ);
    {{ const double* fb = N_VGetArrayPointer(f_buf);
      for (int i = 0; i < NEQ; ++i) f_old[i] = fb[i]; }}

    bool any_terminal = false;
    std::vector<char> is_modified(NEQ, 0);
    for (int j : triggered_idx) {{
      const auto& ev = event_roots[j];
      if (ev.terminal) {{ any_terminal = true; continue; }}
      y_arr[ev.var_idx] = ev.g_fn(x_old.data(), t_e);
      is_modified[ev.var_idx] = 1;
    }}

    rhs_fn(t_e, y, f_buf, &ud);
    std::vector<double> f_new(NEQ);
    {{ const double* fb = N_VGetArrayPointer(f_buf);
      for (int i = 0; i < NEQ; ++i) f_new[i] = fb[i]; }}

    int ref_j = -1;
    for (int j : triggered_idx) {{
      if (!event_roots[j].terminal) {{ ref_j = j; break; }}
    }}
    if (ref_j < 0) return any_terminal;
    const auto& ref = event_roots[ref_j];

    double g_dot = ref.dr_dt_fn(x_old.data(), t_e);
    std::vector<double> rx(NEQ);
    for (int i = 0; i < NEQ; ++i) {{
      rx[i] = ref.dr_dx_fn(x_old.data(), t_e, i);
      g_dot += rx[i] * f_old[i];
    }}
    if (std::fabs(g_dot) < 1e-14) {{
      Rf_warning("Root event: tangential crossing (g_dot ~ 0) — sensitivities may be unreliable at t=%.6e", t_e);
    }}

    const int phi_rows_rt = {phi_rows_r};

    for (int iS = 0; iS < Ns_active; ++iS) {{
      double* yS_k = N_VGetArrayPointer(yS[iS]);
      std::vector<double> S_old(NEQ);
      for (int i = 0; i < NEQ; ++i) S_old[i] = yS_k[i];

      // dr_num = Σ rx_i · S_old_i + Σ_pk dr/dp_pk · M[NEQ+pk, iS]
      double dr_num = 0.0;
      for (int i = 0; i < NEQ; ++i) dr_num += rx[i] * S_old[i];
      for (int pk = 0; pk < {n_params}; ++pk) {{
        double M = ud.Phi_prime[(NEQ + pk) + phi_rows_rt * iS];
        if (M == 0.0) continue;
        dr_num += ref.dr_dp_fn(x_old.data(), t_e, pk) * M;
      }}
      double dt_dp = (g_dot != 0.0) ? -dr_num / g_dot : 0.0;

      if (dt_dp != 0.0) {{
        for (int i = 0; i < NEQ; ++i) {{
          if (is_modified[i]) continue;
          yS_k[i] = S_old[i] + (f_old[i] - f_new[i]) * dt_dp;
        }}
      }}

      for (int j : triggered_idx) {{
        const auto& ev = event_roots[j];
        if (ev.terminal) continue;
        int v = ev.var_idx;

        double sum_gx_S = 0.0, sum_gx_f = 0.0;
        for (int i = 0; i < NEQ; ++i) {{
          double gx_i = ev.dg_dx_fn(x_old.data(), t_e, i);
          sum_gx_S += gx_i * S_old[i];
          sum_gx_f += gx_i * f_old[i];
        }}
        double dg_dt = ev.dg_dt_fn(x_old.data(), t_e);
        double dg_dp_k = 0.0;
        for (int pk = 0; pk < {n_params}; ++pk) {{
          double M = ud.Phi_prime[(NEQ + pk) + phi_rows_rt * iS];
          if (M == 0.0) continue;
          dg_dp_k += ev.dg_dp_fn(x_old.data(), t_e, pk) * M;
        }}

        yS_k[v] = sum_gx_S + dg_dp_k
                + (sum_gx_f + dg_dt - f_new[v]) * dt_dp;
      }}
    }}

    return any_terminal;
  }};
"""
        elif deriv:
            root_event_apply_lambda = """  auto apply_root_events_batch = [&](const std::vector<int>& triggered_idx,
                                     double t_e) -> bool {
    double* y_arr = N_VGetArrayPointer(y);
    std::vector<double> x_old(NEQ);
    for (int i = 0; i < NEQ; ++i) x_old[i] = y_arr[i];

    rhs_fn(t_e, y, f_buf, &ud);
    std::vector<double> f_old(NEQ);
    { const double* fb = N_VGetArrayPointer(f_buf);
      for (int i = 0; i < NEQ; ++i) f_old[i] = fb[i]; }

    // Apply all non-terminal state changes atomically from x_old.
    bool any_terminal = false;
    std::vector<char> is_modified(NEQ, 0);
    for (int j : triggered_idx) {
      const auto& ev = event_roots[j];
      if (ev.terminal) { any_terminal = true; continue; }
      y_arr[ev.var_idx] = ev.g_fn(x_old.data(), t_e);
      is_modified[ev.var_idx] = 1;
    }

    rhs_fn(t_e, y, f_buf, &ud);
    std::vector<double> f_new(NEQ);
    { const double* fb = N_VGetArrayPointer(f_buf);
      for (int i = 0; i < NEQ; ++i) f_new[i] = fb[i]; }

    // Reference event for the shared dt_dp.
    int ref_j = -1;
    for (int j : triggered_idx) {
      if (!event_roots[j].terminal) { ref_j = j; break; }
    }
    if (ref_j < 0) return any_terminal;
    const auto& ref = event_roots[ref_j];

    // ġ = Σ dr/dx_i · f_old_i + dr/dt  (one value, shared across slots)
    double g_dot = ref.dr_dt_fn(x_old.data(), t_e);
    std::vector<double> rx(NEQ);
    for (int i = 0; i < NEQ; ++i) {
      rx[i] = ref.dr_dx_fn(x_old.data(), t_e, i);
      g_dot += rx[i] * f_old[i];
    }
    if (std::fabs(g_dot) < 1e-14) {
      Rf_warning("Root event: tangential crossing (g_dot ~ 0) — sensitivities may be unreliable at t=%.6e", t_e);
    }

    for (int iS = 0; iS < Ns_active; ++iS) {
      int compile_idx = ud.active_to_compile[iS];
      int global_idx  = kSensToGlobal[compile_idx];
      double* yS_k = N_VGetArrayPointer(yS[iS]);
      std::vector<double> S_old(NEQ);
      for (int i = 0; i < NEQ; ++i) S_old[i] = yS_k[i];

      // dt_dp from the reference event's IFT relation on x_old / S_old.
      double dr_num = 0.0;
      for (int i = 0; i < NEQ; ++i) dr_num += rx[i] * S_old[i];
      if (global_idx >= NEQ) {
        int pk = global_idx - NEQ;
        dr_num += ref.dr_dp_fn(x_old.data(), t_e, pk);
      }
      double dt_dp = (g_dot != 0.0) ? -dr_num / g_dot : 0.0;

      // Flow correction on non-modified states: applied exactly once.
      if (dt_dp != 0.0) {
        for (int i = 0; i < NEQ; ++i) {
          if (is_modified[i]) continue;
          yS_k[i] = S_old[i] + (f_old[i] - f_new[i]) * dt_dp;
        }
      }

      // Event-specific saltation on each directly modified state.
      for (int j : triggered_idx) {
        const auto& ev = event_roots[j];
        if (ev.terminal) continue;
        int v = ev.var_idx;

        double sum_gx_S = 0.0, sum_gx_f = 0.0;
        for (int i = 0; i < NEQ; ++i) {
          double gx_i = ev.dg_dx_fn(x_old.data(), t_e, i);
          sum_gx_S += gx_i * S_old[i];
          sum_gx_f += gx_i * f_old[i];
        }
        double dg_dt = ev.dg_dt_fn(x_old.data(), t_e);
        double dg_dp_k = 0.0;
        if (global_idx >= NEQ) {
          int pk = global_idx - NEQ;
          dg_dp_k = ev.dg_dp_fn(x_old.data(), t_e, pk);
        }

        yS_k[v] = sum_gx_S + dg_dp_k
                + (sum_gx_f + dg_dt - f_new[v]) * dt_dp;
      }
    }

    return any_terminal;
  };
"""
        else:
            root_event_apply_lambda = """  auto apply_root_events_batch = [&](const std::vector<int>& triggered_idx,
                                     double t_e) -> bool {
    double* y_arr = N_VGetArrayPointer(y);
    std::vector<double> x_old(NEQ);
    for (int i = 0; i < NEQ; ++i) x_old[i] = y_arr[i];
    bool any_terminal = false;
    for (int j : triggered_idx) {
      const auto& ev = event_roots[j];
      if (ev.terminal) { any_terminal = true; continue; }
      y_arr[ev.var_idx] = ev.g_fn(x_old.data(), t_e);
    }
    return any_terminal;
  };
"""

    event_apply_lambda = time_event_apply_lambda + root_event_apply_lambda

    if rootfunc_mode == "equilibrate":
        # Post-step check uses rhs_fn to compute ydot and tests max |ydot|.
        equilibrate_check_block = """    {
      N_Vector ydot_tmp = N_VClone(y);
      rhs_fn(t_reached, y, ydot_tmp, &ud);
      const double* yd = N_VGetArrayPointer(ydot_tmp);
      double max_abs = 0.0;
      for (int i = 0; i < NEQ; ++i) {
        double a = std::fabs(yd[i]);
        if (a > max_abs) max_abs = a;
      }
      N_VDestroy(ydot_tmp);
      if (max_abs < root_tol) {
        solver_msg = "Terminated: steady state reached (equilibrate)";
        root_terminated = true;
        break;
      }
    }
"""
    else:
        equilibrate_check_block = ""

    # --- do_cvode_step: single step with CV_ROOT_RETURN dispatch ---
    # Returns: 0 = reached target normally
    #          1 = user rootfunc terminated
    #          2 = terminal root event terminated
    #         -1 = error (return_code/solver_msg set; caller branches on rc<0)
    # The lambda's internal `-1` is a local signal; `return_code` itself
    # carries the raw CVODE flag so diagnostics() can map it precisely.
    # On (1)/(2) the caller is responsible for pushing the final output row.
    # The retry-loop body is only emitted when events or a user rootfunc are
    # present — otherwise CVode never returns CV_ROOT_RETURN.
    if has_events:
        time_check = "has_time_events = true; (void)has_time_events;"  # placeholder
        if deriv:
            sens_get_lambda = """      {
        int flag_gs = CVodeGetSens(cvode_mem, &tret, yS);
        if (flag_gs < 0) {
          return_code = flag_gs;
          solver_msg = "CVodeGetSens failed";
          return -1;
        }
      }
"""
        else:
            sens_get_lambda = ""

        # Root-dispatch body: only meaningful when CVodeRootInit was called.
        if need_cvode_root:
            user_check = ""
            if n_user_rootfunc > 0:
                user_check = f"""      for (int i = 0; i < {n_user_rootfunc}; ++i) {{
        if (rinfo[i] != 0) {{
          solver_msg = "Terminated: rootfunc crossed zero";
          root_terminated = true;
          return 1;
        }}
      }}
"""
            event_apply_sec = ""
            if has_root_events:
                event_apply_sec = f"""      std::vector<int> triggered_idx;
      for (int j = 0; j < {n_event_roots}; ++j) {{
        if (rinfo[{n_user_rootfunc} + j] != 0 &&
            ud.root_fired[j] < ud.maxroot) {{
          triggered_idx.push_back(j);
        }}
      }}
      bool any_terminal = false;
      if (!triggered_idx.empty()) {{
        any_terminal = apply_root_events_batch(triggered_idx, (double)tret);
        for (int j : triggered_idx) ud.root_fired[j]++;
        if (CVodeReInit(cvode_mem, tret, y) < 0) {{
          cleanup(); Rf_error("CVodeReInit after root event failed");
        }}
{event_sens_reinit}      }}
      if (any_terminal) {{
        solver_msg = "Terminated: terminal root event fired";
        root_terminated = true;
        return 2;
      }}
"""
            root_dispatch_block = f"""      std::vector<int> rinfo({n_total_roots});
      CVodeGetRootInfo(cvode_mem, rinfo.data());
{user_check}{event_apply_sec}"""
        else:
            root_dispatch_block = ("      solver_msg = \"unexpected CV_ROOT_RETURN\";\n"
                                   "      return_code = CV_UNRECOGNIZED_ERR; return -1;\n")

        do_cvode_step_lambda = f"""  auto do_cvode_step = [&](double target, double& tret_out) -> int {{
    while (true) {{
      sunrealtype tret;
      int flag = CVode(cvode_mem, target, y, &tret, CV_NORMAL);
      if (flag < 0) {{
        return_code = flag;
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "CVode failed at t=%.6e with flag %d", target, flag);
        solver_msg = buf;
        return -1;
      }}
{sens_get_lambda}      tret_out = (double)tret;
      if (flag != CV_ROOT_RETURN) return 0;
{root_dispatch_block}      if (tret_out >= target) return 0;
    }}
  }};
"""
    else:
        do_cvode_step_lambda = ""

    # --- Main integration loop body ---
    if has_events:
        if has_time_events:
            time_interleave_block = f"""    // Time-event interleave: integrate to each event time, apply, reinit.
    while (ev_idx < time_events.size() && time_events[ev_idx].time <= times[k]) {{
      double t_e = time_events[ev_idx].time;
      if (t_e > t_reached) {{
        double tret_loc;
        int rc = do_cvode_step(t_e, tret_loc);
        if (rc < 0) {{ stop = true; break; }}
        t_reached = tret_loc;
        if (rc >= 1) {{
          out_t.push_back(t_reached);
          {{ const double* y_arr = N_VGetArrayPointer(y);
            for (int i = 0; i < NEQ; ++i) out_y.push_back(y_arr[i]); }}
{sens_store_block}
          stop = true; break;
        }}
      }}
      apply_time_event(time_events[ev_idx]);
      if (CVodeReInit(cvode_mem, t_e, y) < 0)
        {{ cleanup(); Rf_error("CVodeReInit failed at t=%.6e", t_e); }}
{event_sens_reinit}      ev_idx++;
    }}
    if (stop) break;
"""
        else:
            time_interleave_block = ""

        main_loop_body = f"""  bool stop = false;
  for (int k = 1; k < n_times && !stop; ++k) {{
{time_interleave_block}
    // Integrate to the next output time.
    if (times[k] > t_reached) {{
      double tret_loc;
      int rc = do_cvode_step(times[k], tret_loc);
      if (rc < 0) break;
      t_reached = tret_loc;
      out_t.push_back(t_reached);
      {{ const double* y_arr = N_VGetArrayPointer(y);
        for (int i = 0; i < NEQ; ++i) out_y.push_back(y_arr[i]); }}
{sens_store_block}
      if (rc >= 1) {{ stop = true; break; }}
{equilibrate_check_block}
    }} else {{
      // Reached times[k] already via an event — record snapshot.
      out_t.push_back(t_reached);
      {{ const double* y_arr = N_VGetArrayPointer(y);
        for (int i = 0; i < NEQ; ++i) out_y.push_back(y_arr[i]); }}
{sens_store_block}
    }}
  }}
"""
    else:
        main_loop_body = f"""
#ifdef CVODE_STEP_TRACE
  {{
    // ---- CV_ONE_STEP trace mode: one CSV row per accepted internal step,
    //      user-time outputs via dense interpolation (CVodeGetDky /
    //      CVodeGetSensDky).  Events and rootfunc are not supported in
    //      this path — the simple `else`-branch here is entered only
    //      when neither is active in the model.
    N_Vector _ele_buf    = N_VClone(y);
    N_Vector _ewt_buf    = N_VClone(y);
    N_Vector _y_interp   = N_VClone(y);
    N_Vector* _eleS_buf  = (Ns_active > 0) ? N_VCloneVectorArray(Ns_active, y) : nullptr;
    N_Vector* _ewtS_buf  = (Ns_active > 0) ? N_VCloneVectorArray(Ns_active, y) : nullptr;
    N_Vector* _yS_interp = (Ns_active > 0) ? N_VCloneVectorArray(Ns_active, y) : nullptr;

    const double t_final = (double)times[n_times - 1];
    CVodeSetStopTime(cvode_mem, t_final);

    int out_idx = 1;  // times[0] = t0 already written
    sunrealtype tret = t0;
    while (out_idx < n_times) {{
      int flag = CVode(cvode_mem, t_final, y, &tret, CV_ONE_STEP);
      if (flag < 0) {{
        return_code = flag;
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "CVode failed at t=%.6e with flag %d", (double)tret, flag);
        solver_msg = buf;
        break;
      }}

      cvode_emit_trace_row(cvode_mem, _ele_buf, _ewt_buf,
                           _eleS_buf, _ewtS_buf, Ns_active, (double)tret);

      while (out_idx < n_times && (double)times[out_idx] <= (double)tret) {{
        {{
          int flag_dky = CVodeGetDky(cvode_mem, times[out_idx], 0, _y_interp);
          if (flag_dky != 0) {{
            return_code = flag_dky;
            solver_msg = "CVodeGetDky failed";
            out_idx = n_times;
            break;
          }}
        }}
        out_t.push_back((double)times[out_idx]);
        {{
          const double* y_arr = N_VGetArrayPointer(_y_interp);
          for (int i = 0; i < NEQ; ++i) out_y.push_back(y_arr[i]);
        }}
        if (Ns_active > 0 && _yS_interp != nullptr) {{
          int flag_sdky = CVodeGetSensDky(cvode_mem, times[out_idx], 0, _yS_interp);
          if (flag_sdky != 0) {{
            return_code = flag_sdky;
            solver_msg = "CVodeGetSensDky failed";
            out_idx = n_times;
            break;
          }}
          for (int j = 0; j < Ns_active; ++j) {{
            const double* yS_arr = N_VGetArrayPointer(_yS_interp[j]);
            for (int i = 0; i < NEQ; ++i) out_s.push_back(yS_arr[i]);
          }}
        }}
        ++out_idx;
      }}
      t_reached = (double)tret;
      if (flag == CV_TSTOP_RETURN) break;
    }}

    N_VDestroy(_ele_buf);
    N_VDestroy(_ewt_buf);
    N_VDestroy(_y_interp);
    if (_eleS_buf)  N_VDestroyVectorArray(_eleS_buf,  Ns_active);
    if (_ewtS_buf)  N_VDestroyVectorArray(_ewtS_buf,  Ns_active);
    if (_yS_interp) N_VDestroyVectorArray(_yS_interp, Ns_active);
  }}
#else
  for (int k = 1; k < n_times; ++k) {{
    sunrealtype tret;
    int flag = CVode(cvode_mem, times[k], y, &tret, CV_NORMAL);
    if (flag < 0) {{
      return_code = flag;
      char buf[160];
      std::snprintf(buf, sizeof(buf),
                    "CVode failed at t=%.6e with flag %d", (double)times[k], flag);
      solver_msg = buf;
      break;
    }}
{sens_get_block}
    t_reached = (double)tret;
    out_t.push_back(t_reached);
    {{
      const double* y_arr = N_VGetArrayPointer(y);
      for (int i = 0; i < NEQ; ++i) out_y.push_back(y_arr[i]);
    }}
{sens_store_block}
    if (flag == CV_ROOT_RETURN) {{
      solver_msg = "Terminated: rootfunc crossed zero";
      root_terminated = true;
      break;
    }}
{equilibrate_check_block}
  }}
#endif
"""

    # Linear-solver setup
    if use_sparse:
        ls_includes = (
            "#include <sunmatrix/sunmatrix_sparse.h>\n"
            "#include <sunlinsol/sunlinsol_klu.h>\n"
        )
        jac_decl = ("static int jac_fn(sunrealtype t, N_Vector y, N_Vector fy, "
                    "SUNMatrix J, void* ud_vp, N_Vector, N_Vector, N_Vector);")
        jac_impl = f"""
static int jac_fn(sunrealtype t, N_Vector y, N_Vector fy,
                  SUNMatrix J, void* ud_vp,
                  N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {{
  (void)t; (void)fy; (void)tmp1; (void)tmp2; (void)tmp3;
  UserData* ud = static_cast<UserData*>(ud_vp);
  const double* params = ud->params.data();
  const double* x = N_VGetArrayPointer(y);
  (void)x; (void)params;
{forcing_local}
{jac_body_sparse}
  return 0;
}}
"""
        ls_setup = f"""  A = SUNSparseMatrix(NEQ, NEQ, {nnz_total}, CSC_MAT, ctx);
  if (!A) {{ cleanup(); Rf_error("SUNSparseMatrix failed"); }}
  LS = SUNLinSol_KLU(y, A, ctx);
  if (!LS) {{ cleanup(); Rf_error("SUNLinSol_KLU failed"); }}
  if (CVodeSetLinearSolver(cvode_mem, LS, A) < 0) {{ cleanup(); Rf_error("CVodeSetLinearSolver failed"); }}
  if (CVodeSetJacFn(cvode_mem, jac_fn) < 0) {{ cleanup(); Rf_error("CVodeSetJacFn failed"); }}
"""
    else:
        ls_includes = ""
        jac_decl = ("static int jac_fn(sunrealtype t, N_Vector y, N_Vector fy, "
                    "SUNMatrix J, void* ud_vp, N_Vector, N_Vector, N_Vector);")
        jac_impl = f"""
static int jac_fn(sunrealtype t, N_Vector y, N_Vector fy,
                  SUNMatrix J, void* ud_vp,
                  N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {{
  (void)t; (void)fy; (void)tmp1; (void)tmp2; (void)tmp3;
  UserData* ud = static_cast<UserData*>(ud_vp);
  const double* params = ud->params.data();
  const double* x = N_VGetArrayPointer(y);
  (void)x; (void)params;
{forcing_local}
  SUNMatZero(J);
{jac_body_dense}
  return 0;
}}
"""
        ls_setup = """  A = SUNDenseMatrix(NEQ, NEQ, ctx);
  if (!A) { cleanup(); Rf_error("SUNDenseMatrix failed"); }
  LS = SUNLinSol_Dense(y, A, ctx);
  if (!LS) { cleanup(); Rf_error("SUNLinSol_Dense failed"); }
  if (CVodeSetLinearSolver(cvode_mem, LS, A) < 0) { cleanup(); Rf_error("CVodeSetLinearSolver failed"); }
  if (CVodeSetJacFn(cvode_mem, jac_fn) < 0) { cleanup(); Rf_error("CVodeSetJacFn failed"); }
"""

    sens_decl = ""
    sens_impl = ""
    if deriv:
        sens_decl = ("static int sens_rhs1_fn(int Ns, sunrealtype t, N_Vector y, "
                     "N_Vector ydot, int iS, N_Vector yS, N_Vector ySdot, "
                     "void* ud_vp, N_Vector tmp1, N_Vector tmp2);")
        if has_reparam:
            # Emit chain-rule accumulation: ySdot += sum_pk (df/dp_pk)(x,p) * M[NEQ+pk, iS]
            # Grouped by pk so we read M[NEQ+pk, iS] once per pk.
            by_pk = {}
            for (i_row, pk, d_expr) in (df_dp_entries or []):
                by_pk.setdefault(pk, []).append((i_row, d_expr))
            phi_rows = n_states + n_params
            reparam_part2_lines = [f"  const int phi_rows = {phi_rows};"]
            for pk in sorted(by_pk.keys()):
                reparam_part2_lines.append(f"  {{  // pk = {pk}")
                reparam_part2_lines.append(
                    f"    const double M_pk = ud->Phi_prime[{n_states + pk} + phi_rows * iS];")
                for (i_row, d_expr) in by_pk[pk]:
                    reparam_part2_lines.append(
                        f"    ySdot_arr[{i_row}] += ({_to_cpp(d_expr, states_list, params_list, n_states, 'double', forcings_list)}) * M_pk;")
                reparam_part2_lines.append("  }")
            reparam_part2_body = "\n".join(reparam_part2_lines)
            sens_impl = f"""
static int sens_rhs1_fn(int Ns, sunrealtype t,
                        N_Vector y, N_Vector ydot,
                        int iS, N_Vector yS, N_Vector ySdot,
                        void* ud_vp,
                        N_Vector tmp1, N_Vector tmp2) {{
  (void)Ns; (void)t; (void)ydot; (void)tmp1; (void)tmp2;
  UserData* ud = static_cast<UserData*>(ud_vp);
  const double* params = ud->params.data();
  const double* x = N_VGetArrayPointer(y);
  const double* yS_arr  = N_VGetArrayPointer(yS);
  double*       ySdot_arr = N_VGetArrayPointer(ySdot);
  (void)x; (void)params;
{forcing_local}

  // --- Part 1: ySdot = J(x,p) * yS ---
{jac_times_yS_body}

  // --- Part 2 (reparam): ySdot += (df/dp_d) * M_param[:, iS] ---
  // where M_param is the parameter block of Phi'(theta).
{reparam_part2_body}
  return 0;
}}
"""
        else:
            sens_impl = f"""
static int sens_rhs1_fn(int Ns, sunrealtype t,
                        N_Vector y, N_Vector ydot,
                        int iS, N_Vector yS, N_Vector ySdot,
                        void* ud_vp,
                        N_Vector tmp1, N_Vector tmp2) {{
  (void)Ns; (void)t; (void)ydot; (void)tmp1; (void)tmp2;
  UserData* ud = static_cast<UserData*>(ud_vp);
  const double* params = ud->params.data();
  const double* x = N_VGetArrayPointer(y);
  const double* yS_arr  = N_VGetArrayPointer(yS);
  double*       ySdot_arr = N_VGetArrayPointer(ySdot);
  (void)x; (void)params;
{forcing_local}

  // --- Part 1: ySdot = J(x,p) * yS ---
{jac_times_yS_body}

  // --- Part 2: add df/dp_k if iS maps to a real parameter ---
  int compile_idx = ud->active_to_compile[iS];
  int global_idx = kSensToGlobal[compile_idx];
  if (global_idx >= NEQ) {{
    int pk = global_idx - NEQ;
    switch (pk) {{
{df_dp_switch_body}
      default: break;
    }}
  }}
  return 0;
}}
"""

    return f"""/** Code auto-generated by CppODE {version} (CVODE backend) **/

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
{ls_includes}#include <sundials/sundials_types.h>
#include <sundials/sundials_context.h>
#include <cppode/cppode_step_trace.hpp>
{forcing_include}{event_block_includes}

namespace {{

constexpr int NEQ    = {n_states};
constexpr int NPARMS = {n_global};           // n_states + n_params (flat layout)
constexpr int NSENS_COMPILE = {n_sens_compile};
constexpr int NTHETA = {ntheta};             // theta dim (== NSENS_COMPILE unless reparam)
constexpr bool HAS_REPARAM = {has_reparam_cpp};

static const int kSensToGlobal[{n_sens_arr}]  = {{ {s2g_str} }};
static const int kSensToParamIdx[{n_sens_arr}] = {{ {s2pk_str} }};

struct UserData {{
  std::vector<double> params;               // length NPARMS
  std::vector<int>    active_to_compile;    // length Ns_active
{reparam_ud_members}{forcing_ud_members}{events_ud_members}}};

static int rhs_fn(sunrealtype t, N_Vector y, N_Vector ydot, void* ud_vp);
{jac_decl}
{sens_decl}
{rootfunc_decl}

// ---- RHS ----
static int rhs_fn(sunrealtype t, N_Vector y, N_Vector ydot, void* ud_vp) {{
  (void)t;
  UserData* ud = static_cast<UserData*>(ud_vp);
  const double* params = ud->params.data();
  const double* x = N_VGetArrayPointer(y);
  double* ydot_arr = N_VGetArrayPointer(ydot);
  (void)x; (void)params;
{forcing_local}
{ode_body}
  return 0;
}}
{jac_impl}
{sens_impl}
{rootfunc_impl}
{event_struct}

// ---- Step trace (CVODE_STEP_TRACE) ----
// When compiled with -DCVODE_STEP_TRACE, the generated main loop drives the
// integrator in CV_ONE_STEP mode and appends one row per accepted internal
// step into the CppODE trace buffer (see cppode_step_trace.hpp).  After
// integration the entry point marshals the buffer into the `trace` element
// of the returned R list; `solveODE()` then decides whether to attach it
// as a data.frame or write a CSV.  Fields not exposed by the CVODES public
// API (`tq[2]`, pre-scale `dsm`, Nordsieck `gamma`) are pushed as NaN.
#ifdef CVODE_STEP_TRACE
static void cvode_emit_trace_row(void* cvode_mem,
                                 N_Vector ele_buf, N_Vector ewt_buf,
                                 N_Vector* eleS_buf, N_Vector* ewtS_buf,
                                 int ns_active, double t_reached) {{
  long n_steps = 0, n_fe = 0, n_je = 0, n_setups = 0;
  int  last_order = 0;
  double last_h = 0.0;
  CVodeGetNumSteps(cvode_mem, &n_steps);
  CVodeGetNumRhsEvals(cvode_mem, &n_fe);
  CVodeGetNumJacEvals(cvode_mem, &n_je);
  CVodeGetNumLinSolvSetups(cvode_mem, &n_setups);
  CVodeGetLastOrder(cvode_mem, &last_order);
  CVodeGetLastStep(cvode_mem, &last_h);

  // Reconstruct dsm from ele + ewt.  IMPORTANT: `CVodeGetEstLocalErrors`
  // returns `ele = acor · tq[2]` — already the *scaled* local-error
  // estimate, not the raw corrector-predictor difference.  Therefore
  // `WRMS(ele · ewt)` yields the controller's state-side `dsm` directly,
  // not a raw `acnrm`.  The CVODES public API exposes neither `acor` nor
  // `tq[2]` separately, so only `dsm_state` can be reconstructed; `acnrm`
  // is emitted as NaN in this trace.
  //
  // STATE-ONLY: This trace shows the STATE contribution only.  The full
  // dsm CVODES uses internally for step control includes per-sensitivity
  // contributions via `cvSensUpdateNorm` (CV_STAGGERED) when sensErrCon
  // is on — but `CVodeGetSensEstLocalErrors` does not exist in SUNDIALS
  // 6.4.x's public API, and `cv_acorS`/`cv_tq[2]` are not exposed via
  // any post-step Get function.  Reading them would require including
  // <cvodes/cvodes_impl.h>, which is not shipped with libsundials-dev.
  //
  // For apples-to-apples comparison against CppODE: compute
  //   CppODE_state_dsm = trace$acnrm_state * trace$tq2
  // and compare against this trace's `dsm` column.  CppODE's `dsm`
  // column is the full sens-aware controller value (max over state and
  // each sensitivity vector); CVODE's effective full dsm is unobservable
  // via the public API.
  (void)eleS_buf; (void)ewtS_buf; (void)ns_active;
  double sumsq = 0.0;
  long   N_eff = 0;
  if (CVodeGetEstLocalErrors(cvode_mem, ele_buf) == 0 &&
      CVodeGetErrWeights(cvode_mem, ewt_buf)    == 0) {{
    const double* e = N_VGetArrayPointer(ele_buf);
    const double* w = N_VGetArrayPointer(ewt_buf);
    for (int i = 0; i < NEQ; ++i) {{
      double r = e[i] * w[i];
      sumsq += r * r;
      ++N_eff;
    }}
  }}
  double dsm_reconstructed =
    (N_eff > 0) ? std::sqrt(sumsq / static_cast<double>(N_eff)) : 0.0;

  double nan_val = std::nan("");
  auto& tb = cppode::ndf_detail::get_trace_buffer();
  tb.nst.push_back(static_cast<int>(n_steps));
  tb.t.push_back(t_reached);
  tb.h.push_back(last_h);
  tb.q.push_back(last_order);
  tb.dsm.push_back(dsm_reconstructed);
  tb.acnrm.push_back(nan_val);        // raw acor not exposed by CVODES API
  tb.acnrm_state.push_back(nan_val);  // see comment above `dsm_reconstructed`
  tb.tq2.push_back(nan_val);
  tb.gamma.push_back(nan_val);
  tb.gamrat.push_back(nan_val);
  tb.newton_conv.push_back(1);
  tb.mode.emplace_back("CVODE");
  tb.nfe.push_back(static_cast<int>(n_fe));
  tb.njev.push_back(static_cast<int>(n_je));
  tb.nsetups.push_back(static_cast<int>(n_setups));
  tb.setup_reason.emplace_back("");
  tb.pece_iters.push_back(0);
  tb.pece_diverged.push_back(0);
}}
#endif  // CVODE_STEP_TRACE

}} // anonymous namespace

// =====================================================================
// R entry point — matches CppODE's 15-SEXP signature
// =====================================================================

extern "C" SEXP solve_{modelname}(
    SEXP timesSEXP, SEXP paramsSEXP, SEXP sens1iniSEXP, SEXP sens2iniSEXP,
    SEXP fixedSEXP, SEXP abstolSEXP, SEXP reltolSEXP, SEXP maxprogressSEXP,
    SEXP maxstepsSEXP, SEXP hiniSEXP, SEXP root_tolSEXP, SEXP maxrootSEXP,
    SEXP forcingTimesSEXP, SEXP forcingValuesSEXP, SEXP pidModeSEXP)
{{
  (void)maxprogressSEXP; (void)pidModeSEXP;

  if (!Rf_isNull(sens2iniSEXP))
    Rf_error("sens2ini not supported by CVODES backend");

  constexpr bool deriv = {deriv_flag};

  const int n_times_in = Rf_length(timesSEXP);
  if (n_times_in < 1) Rf_error("times must be non-empty");
  const double abstol = REAL(abstolSEXP)[0];
  const double reltol = REAL(reltolSEXP)[0];
  const double hini   = REAL(hiniSEXP)[0];
  const double root_tol = REAL(root_tolSEXP)[0]; (void)root_tol;
  const long   maxsteps = static_cast<long>(INTEGER(maxstepsSEXP)[0]);

  UserData ud;
  ud.params.assign(REAL(paramsSEXP), REAL(paramsSEXP) + NPARMS);
  {{
    const int mr = INTEGER(maxrootSEXP)[0];
    ud.maxroot = (mr > 0) ? mr : 1;
    ud.root_fired.assign({n_event_roots}, 0);
  }}

{forcing_init_block}
  // Runtime-fixed → build active_to_compile mapping (non-reparam only).
  // Under reparam, `fixed` is rejected and Ns = NTHETA directly.
  std::vector<int> active_to_compile;
  if (deriv) {{
    if (HAS_REPARAM && !Rf_isNull(fixedSEXP) && Rf_length(fixedSEXP) > 0) {{
      Rf_error("runtime 'fixed' is not supported together with ntheta reparametrization; "
               "express fixedness via zero rows in sens1ini");
    }}
    std::vector<char> fixed_mask(NSENS_COMPILE, 0);
    if (!Rf_isNull(fixedSEXP)) {{
      const int* fx = INTEGER(fixedSEXP);
      const int  nf = Rf_length(fixedSEXP);
      for (int i = 0; i < nf; ++i)
        if (fx[i] >= 0 && fx[i] < NSENS_COMPILE) fixed_mask[fx[i]] = 1;
    }}
    active_to_compile.reserve(NSENS_COMPILE);
    for (int i = 0; i < NSENS_COMPILE; ++i)
      if (!fixed_mask[i]) active_to_compile.push_back(i);
    ud.active_to_compile = active_to_compile;
  }}
  const int Ns_active = deriv
                         ? (HAS_REPARAM ? NTHETA
                                        : static_cast<int>(active_to_compile.size()))
                         : 0;

  // --- times ---
  std::vector<double> times(REAL(timesSEXP), REAL(timesSEXP) + n_times_in);
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());
  const int n_times = static_cast<int>(times.size());
  const double t0 = times[0];

  // --- output buffers ---
  std::vector<double> out_t;  out_t.reserve(n_times);
  std::vector<double> out_y;  out_y.reserve(static_cast<size_t>(n_times) * NEQ);
  std::vector<double> out_s;
  if (deriv) out_s.reserve(static_cast<size_t>(n_times) * NEQ * Ns_active);

  // --- SUNDIALS resources ---
  SUNContext ctx = nullptr;
  void* cvode_mem = nullptr;
  N_Vector y  = nullptr;
  N_Vector* yS = nullptr;
  N_Vector f_buf = nullptr;  // scratch for rhs evaluation at events
  SUNMatrix A = nullptr;
  SUNLinearSolver LS = nullptr;
  int Ns_alloc = 0;

  auto cleanup = [&]() {{
    if (yS) {{ N_VDestroyVectorArray(yS, Ns_alloc); yS = nullptr; }}
    if (f_buf) {{ N_VDestroy(f_buf); f_buf = nullptr; }}
    if (cvode_mem) {{ CVodeFree(&cvode_mem); cvode_mem = nullptr; }}
    if (LS) {{ SUNLinSolFree(LS); LS = nullptr; }}
    if (A)  {{ SUNMatDestroy(A); A = nullptr; }}
    if (y)  {{ N_VDestroy(y); y = nullptr; }}
    if (ctx) {{ SUNContext_Free(&ctx); ctx = nullptr; }}
  }};

  if (SUNContext_Create(nullptr, &ctx) < 0)
    Rf_error("SUNContext_Create failed");

  y = N_VNew_Serial(NEQ, ctx);
  if (!y) {{ cleanup(); Rf_error("N_VNew_Serial failed"); }}
  {{
    double* y0 = N_VGetArrayPointer(y);
    for (int i = 0; i < NEQ; ++i) y0[i] = ud.params[i];
  }}

  cvode_mem = CVodeCreate({cv_method}, ctx);
  if (!cvode_mem) {{ cleanup(); Rf_error("CVodeCreate failed"); }}
  if (CVodeInit(cvode_mem, rhs_fn, t0, y) < 0) {{ cleanup(); Rf_error("CVodeInit failed"); }}
  if (CVodeSStolerances(cvode_mem, reltol, abstol) < 0) {{ cleanup(); Rf_error("CVodeSStolerances failed"); }}
  if (CVodeSetUserData(cvode_mem, &ud) < 0) {{ cleanup(); Rf_error("CVodeSetUserData failed"); }}
  CVodeSetMaxNumSteps(cvode_mem, maxsteps);
  if (hini > 0.0) CVodeSetInitStep(cvode_mem, hini);
{ls_setup}
{rootfunc_init_block}
  // --- sensitivities ---
{sens_init_block}

  // --- write t0 row ---
  out_t.push_back(t0);
  {{
    const double* y0 = N_VGetArrayPointer(y);
    for (int i = 0; i < NEQ; ++i) out_y.push_back(y0[i]);
{sens_t0_block}
  }}

  // --- time-step loop ---
  std::string solver_msg;
  int return_code = 0;
  double t_reached = t0;
  bool root_terminated = false; (void)root_terminated;

{event_builder_block}{event_apply_lambda}{do_cvode_step_lambda}{event_pre_t0_block}{main_loop_body}

  // --- diagnostics ---
  long n_steps = 0, n_fe = 0, n_je = 0, n_etf = 0, n_setups = 0;
  int  last_order = 0;
  double last_h = 0.0;
  CVodeGetNumSteps(cvode_mem, &n_steps);
  CVodeGetNumRhsEvals(cvode_mem, &n_fe);
  CVodeGetNumErrTestFails(cvode_mem, &n_etf);
  CVodeGetNumLinSolvSetups(cvode_mem, &n_setups);
  CVodeGetLastOrder(cvode_mem, &last_order);
  CVodeGetLastStep(cvode_mem, &last_h);
  CVodeGetNumJacEvals(cvode_mem, &n_je);
{sens_fevals_block}

  const int n_out = static_cast<int>(out_t.size());

  // --- allocate R outputs (time-first layout) ---
  // Internal push_back buffers are state-first:
  //   out_y : [NEQ, n_out]                (state varies fastest per time step)
  //   out_s : [NEQ, Ns_active, n_out]     (state fastest, then sens, then time)
  // R-facing arrays are time-first:
  //   variable_mat : [n_out, NEQ]
  //   sens1_arr    : [n_out, NEQ, Ns_active]
  // We transpose on the final copy (single pass, cheap vs. integration).
  SEXP time_vec     = PROTECT(Rf_allocVector(REALSXP, n_out));
  SEXP variable_mat = PROTECT(Rf_allocMatrix(REALSXP, n_out, NEQ));
  std::memcpy(REAL(time_vec), out_t.data(), sizeof(double) * n_out);
  {{
    double* vout = REAL(variable_mat);
    for (int i = 0; i < n_out; ++i)
      for (int j = 0; j < NEQ; ++j)
        vout[i + (size_t)n_out * j] = out_y[(size_t)i * NEQ + j];
  }}

  SEXP sens1_arr = R_NilValue;
  if (deriv) {{
    SEXP dim3 = PROTECT(Rf_allocVector(INTSXP, 3));
    INTEGER(dim3)[0] = n_out;
    INTEGER(dim3)[1] = NEQ;
    INTEGER(dim3)[2] = Ns_active;
    sens1_arr = PROTECT(Rf_allocArray(REALSXP, dim3));
    if (!out_s.empty()) {{
      double* sout = REAL(sens1_arr);
      for (int i = 0; i < n_out; ++i)
        for (int iS = 0; iS < Ns_active; ++iS)
          for (int j = 0; j < NEQ; ++j)
            sout[i + (size_t)n_out * (j + (size_t)NEQ * iS)] =
              out_s[((size_t)i * Ns_active + iS) * NEQ + j];
    }}
  }}

  // --- diagnostics list ---
  SEXP diag = PROTECT(Rf_allocVector(VECSXP, 10));
  SEXP diag_names = PROTECT(Rf_allocVector(STRSXP, 10));
  SET_STRING_ELT(diag_names, 0, Rf_mkChar("return_code"));
  SET_STRING_ELT(diag_names, 1, Rf_mkChar("message"));
  SET_STRING_ELT(diag_names, 2, Rf_mkChar("accepted"));
  SET_STRING_ELT(diag_names, 3, Rf_mkChar("rejected"));
  SET_STRING_ELT(diag_names, 4, Rf_mkChar("fevals"));
  SET_STRING_ELT(diag_names, 5, Rf_mkChar("jevals"));
  SET_STRING_ELT(diag_names, 6, Rf_mkChar("setups"));
  SET_STRING_ELT(diag_names, 7, Rf_mkChar("last_dt"));
  SET_STRING_ELT(diag_names, 8, Rf_mkChar("last_order"));
  SET_STRING_ELT(diag_names, 9, Rf_mkChar("t_reached"));
  Rf_setAttrib(diag, R_NamesSymbol, diag_names);
  SET_VECTOR_ELT(diag, 0, Rf_ScalarInteger(return_code));
  {{
    SEXP m = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(m, 0, Rf_mkChar(
      solver_msg.empty() ? "Integration was successful." : solver_msg.c_str()));
    SET_VECTOR_ELT(diag, 1, m);
    UNPROTECT(1);
  }}
  SET_VECTOR_ELT(diag, 2, Rf_ScalarInteger((int)(n_steps - n_etf)));
  SET_VECTOR_ELT(diag, 3, Rf_ScalarInteger((int)n_etf));
  SET_VECTOR_ELT(diag, 4, Rf_ScalarInteger((int)n_fe));
  SET_VECTOR_ELT(diag, 5, Rf_ScalarInteger((int)n_je));
  SET_VECTOR_ELT(diag, 6, Rf_ScalarInteger((int)n_setups));
  SET_VECTOR_ELT(diag, 7, Rf_ScalarReal(last_h));
  SET_VECTOR_ELT(diag, 8, Rf_ScalarInteger(last_order));
  SET_VECTOR_ELT(diag, 9, Rf_ScalarReal(t_reached));

  // --- marshal step-trace buffer into a named list of vectors ---
  // (empty lists when the model was compiled without -DCVODE_STEP_TRACE).
  SEXP trace_list;
  {{
    auto& tb = cppode::ndf_detail::get_trace_buffer();
    const R_xlen_t n_trace = static_cast<R_xlen_t>(tb.size());
    constexpr int n_cols = 18;
    static const char* col_names[n_cols] = {{
      "nst","t","h","q","dsm","acnrm","acnrm_state","tq2",
      "gamma","gamrat","newton_conv","mode","nfe","njev",
      "nsetups","setup_reason","pece_iters","pece_diverged"
    }};
    trace_list = PROTECT(Rf_allocVector(VECSXP, n_cols));
    SEXP tn = PROTECT(Rf_allocVector(STRSXP, n_cols));
    for (int i = 0; i < n_cols; ++i)
      SET_STRING_ELT(tn, i, Rf_mkChar(col_names[i]));
    Rf_setAttrib(trace_list, R_NamesSymbol, tn);
    UNPROTECT(1);  // tn
    auto put_int = [&](int slot, const std::vector<int>& v) {{
      SEXP s = PROTECT(Rf_allocVector(INTSXP, n_trace));
      int* p = INTEGER(s);
      for (R_xlen_t i = 0; i < n_trace; ++i) p[i] = v[i];
      SET_VECTOR_ELT(trace_list, slot, s);
      UNPROTECT(1);
    }};
    auto put_dbl = [&](int slot, const std::vector<double>& v) {{
      SEXP s = PROTECT(Rf_allocVector(REALSXP, n_trace));
      double* p = REAL(s);
      for (R_xlen_t i = 0; i < n_trace; ++i) p[i] = v[i];
      SET_VECTOR_ELT(trace_list, slot, s);
      UNPROTECT(1);
    }};
    auto put_str = [&](int slot, const std::vector<std::string>& v) {{
      SEXP s = PROTECT(Rf_allocVector(STRSXP, n_trace));
      for (R_xlen_t i = 0; i < n_trace; ++i)
        SET_STRING_ELT(s, i, Rf_mkChar(v[i].c_str()));
      SET_VECTOR_ELT(trace_list, slot, s);
      UNPROTECT(1);
    }};
    put_int(0,  tb.nst);         put_dbl(1,  tb.t);
    put_dbl(2,  tb.h);           put_int(3,  tb.q);
    put_dbl(4,  tb.dsm);         put_dbl(5,  tb.acnrm);
    put_dbl(6,  tb.acnrm_state); put_dbl(7,  tb.tq2);
    put_dbl(8,  tb.gamma);       put_dbl(9,  tb.gamrat);
    put_int(10, tb.newton_conv); put_str(11, tb.mode);
    put_int(12, tb.nfe);         put_int(13, tb.njev);
    put_int(14, tb.nsetups);     put_str(15, tb.setup_reason);
    put_int(16, tb.pece_iters);  put_int(17, tb.pece_diverged);
    tb.clear();
  }}

  // --- result list ---
  SEXP ans, names_ans;
  if (deriv) {{
    ans       = PROTECT(Rf_allocVector(VECSXP, 5));
    names_ans = PROTECT(Rf_allocVector(STRSXP, 5));
    SET_STRING_ELT(names_ans, 0, Rf_mkChar("time"));
    SET_STRING_ELT(names_ans, 1, Rf_mkChar("variable"));
    SET_STRING_ELT(names_ans, 2, Rf_mkChar("sens1"));
    SET_STRING_ELT(names_ans, 3, Rf_mkChar("diagnostics"));
    SET_STRING_ELT(names_ans, 4, Rf_mkChar("trace"));
    Rf_setAttrib(ans, R_NamesSymbol, names_ans);
    SET_VECTOR_ELT(ans, 0, time_vec);
    SET_VECTOR_ELT(ans, 1, variable_mat);
    SET_VECTOR_ELT(ans, 2, sens1_arr);
    SET_VECTOR_ELT(ans, 3, diag);
    SET_VECTOR_ELT(ans, 4, trace_list);
    // free SUNDIALS resources before return
    cleanup();
    UNPROTECT(9);   // trace_list + 8 previously
    return ans;
  }} else {{
    ans       = PROTECT(Rf_allocVector(VECSXP, 4));
    names_ans = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_STRING_ELT(names_ans, 0, Rf_mkChar("time"));
    SET_STRING_ELT(names_ans, 1, Rf_mkChar("variable"));
    SET_STRING_ELT(names_ans, 2, Rf_mkChar("diagnostics"));
    SET_STRING_ELT(names_ans, 3, Rf_mkChar("trace"));
    Rf_setAttrib(ans, R_NamesSymbol, names_ans);
    SET_VECTOR_ELT(ans, 0, time_vec);
    SET_VECTOR_ELT(ans, 1, variable_mat);
    SET_VECTOR_ELT(ans, 2, diag);
    SET_VECTOR_ELT(ans, 3, trace_list);
    cleanup();
    UNPROTECT(7);   // trace_list + 6 previously
    return ans;
  }}
}}
"""
