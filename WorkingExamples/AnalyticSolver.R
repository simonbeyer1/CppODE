# Analytic ODE solver + sensitivites (1st and 2nd order)
library(reticulate)
library(data.table)
solveOdeAnalytic <- function(ode, times, pars, events = NULL) {
  library(reticulate)
  library(data.table)
  sympy <- import("sympy")
  builtins <- import_builtins()

  # Variablenname aus ODE extrahieren
  var_name <- names(ode)

  # Symbole
  t <- sympy$symbols("t", real = TRUE)
  x0 <- sympy$symbols("x0", real = TRUE)

  # Parameter-Symbole (ohne die Variable selbst)
  par_names <- setdiff(names(pars), var_name)
  par_syms <- lapply(par_names, function(p) sympy$symbols(p, real = TRUE))
  names(par_syms) <- par_names

  # Spezielle Funktionen
  local_dict <- c(
    list(t = t,
         exp = sympy$exp, sin = sympy$sin, cos = sympy$cos,
         log = sympy$log, sqrt = sympy$sqrt),
    par_syms
  )

  # ^ → **
  rhs_str <- gsub("\\^", "**", ode[var_name])

  # Alle Parameter-Symbole
  all_par_syms <- c(list(x0 = x0), par_syms)
  all_par_names <- names(all_par_syms)

  # ODE Segment lösen
  solve_ode_segment <- function(rhs_str, t_start_sym, x_init_sym, local_dict) {
    tau <- sympy$symbols("tau", real = TRUE, nonnegative = TRUE)
    y <- sympy$symbols("y", cls = sympy$Function)
    y_tau <- y(tau)

    parse_dict <- local_dict
    parse_dict[[var_name]] <- y_tau
    parse_dict[["t"]] <- tau + t_start_sym

    ode_rhs <- sympy$parse_expr(rhs_str, local_dict = parse_dict)
    ode_eq <- sympy$Eq(sympy$diff(y_tau, tau), ode_rhs)

    sol_general <- sympy$dsolve(ode_eq, y_tau)
    sol_rhs <- sol_general$rhs

    C1 <- sympy$symbols("C1")
    eq_ic <- sympy$Eq(sol_rhs$subs(tau, sympy$Integer(0L)), x_init_sym)
    C1_val <- sympy$solve(eq_ic, C1)

    if (length(C1_val) > 0) {
      sol_rhs <- sol_rhs$subs(C1, C1_val[[1]])
    }

    sol_rhs$subs(tau, t - t_start_sym)
  }

  # Zero crossing finden: Löse root_expr = 0 für t
  find_zero_crossing <- function(sol_expr, root_str, t_start_sym, local_dict, par_vals) {
    root_dict <- local_dict
    root_dict[[var_name]] <- sol_expr

    root_str_py <- gsub("\\^", "**", root_str)
    root_expr <- sympy$parse_expr(root_str_py, local_dict = root_dict)
    root_expr <- sympy$simplify(root_expr)

    cat(sprintf("  Suche Nulldurchgang: %s = 0\n", as.character(root_expr)))

    t_zeros <- tryCatch({
      sympy$solve(root_expr, t)
    }, error = function(e) {
      cat(sprintf("  WARNUNG: sympy$solve fehlgeschlagen: %s\n", e$message))
      list()
    })

    if (inherits(t_zeros, "python.builtin.list")) {
      t_zeros <- py_to_r(t_zeros)
    }
    if (!is.list(t_zeros)) {
      t_zeros <- as.list(t_zeros)
    }

    cat(sprintf("  Gefundene Nullstellen: %d\n", length(t_zeros)))

    if (length(t_zeros) == 0) {
      return(NULL)
    }

    for (i in seq_along(t_zeros)) {
      cat(sprintf("    t[%d] = %s\n", i, as.character(t_zeros[[i]])))
    }

    valid_zeros <- list()
    for (zero in t_zeros) {
      zero_num <- zero
      for (p in names(par_vals)) {
        if (p %in% names(all_par_syms)) {
          zero_num <- zero_num$subs(all_par_syms[[p]], par_vals[[p]])
        }
      }

      tryCatch({
        zero_eval <- zero_num$evalf()
        zero_val <- as.numeric(builtins$float(zero_eval))

        t_start_num <- t_start_sym
        for (p in names(par_vals)) {
          if (p %in% names(all_par_syms)) {
            t_start_num <- t_start_num$subs(all_par_syms[[p]], par_vals[[p]])
          }
        }
        t_start_val <- tryCatch(
          as.numeric(builtins$float(t_start_num$evalf())),
          error = function(e) 0
        )

        cat(sprintf("    Numerisch: t = %f (t_start = %f)\n", zero_val, t_start_val))

        if (is.finite(zero_val) && zero_val > t_start_val + 1e-10) {
          valid_zeros[[length(valid_zeros) + 1]] <- list(sym = zero, val = zero_val)
        }
      }, error = function(e) {
        cat(sprintf("    Konnte nicht evaluieren: %s\n", e$message))
      })
    }

    if (length(valid_zeros) == 0) {
      return(NULL)
    }

    vals <- sapply(valid_zeros, function(z) z$val)
    best_idx <- which.min(vals)
    cat(sprintf("  Erster Nulldurchgang: t* = %s (numerisch: %f)\n",
                as.character(valid_zeros[[best_idx]]$sym), valid_zeros[[best_idx]]$val))

    valid_zeros[[best_idx]]$sym
  }

  cat("=== ODE ===\n")
  x_display <- sympy$Function(var_name)(t)
  display_dict <- c(list(t = t), par_syms)
  display_dict[[var_name]] <- x_display
  ode_display <- sympy$parse_expr(rhs_str, local_dict = display_dict)
  cat(sprintf("d%s/dt = %s\n\n", var_name, as.character(ode_display)))

  # Parameter-Werte
  par_vals <- list(x0 = unname(pars[var_name]))
  for (n in names(pars)) {
    if (n != var_name) par_vals[[n]] <- unname(pars[n])
  }

  # Lösung berechnen
  if (is.null(events) || nrow(events) == 0) {
    sol_rhs <- solve_ode_segment(rhs_str, sympy$Integer(0L), x0, local_dict)
    sol_rhs <- sympy$simplify(sol_rhs)

    cat("=== Analytische Lösung ===\n")
    cat(sprintf("%s(t) = %s\n\n", var_name, as.character(sol_rhs)))

    solutions <- list(list(expr = sol_rhs, t_start = sympy$Integer(0L), t_end = sympy$oo, t_end_is_inf = TRUE))
    full_solution <- sol_rhs

  } else {
    # Event-Symbole hinzufügen
    for (i in seq_len(nrow(events))) {
      ev <- events[i, ]
      if (!is.na(ev$time) && nchar(ev$time) > 0 && !ev$time %in% names(all_par_syms)) {
        all_par_syms[[ev$time]] <- sympy$symbols(ev$time, real = TRUE)
        local_dict[[ev$time]] <- all_par_syms[[ev$time]]
      }
      if (!is.na(ev$value) && !ev$value %in% names(all_par_syms)) {
        all_par_syms[[ev$value]] <- sympy$symbols(ev$value, real = TRUE)
        local_dict[[ev$value]] <- all_par_syms[[ev$value]]
      }
      if (!is.na(ev$root) && nchar(ev$root) > 0) {
        root_vars <- unique(unlist(regmatches(ev$root, gregexpr("[a-zA-Z_][a-zA-Z0-9_]*", ev$root))))
        root_vars <- setdiff(root_vars, c(var_name, "t", "exp", "sin", "cos", "log", "sqrt"))
        for (rv in root_vars) {
          if (!rv %in% names(all_par_syms)) {
            all_par_syms[[rv]] <- sympy$symbols(rv, real = TRUE)
            local_dict[[rv]] <- all_par_syms[[rv]]
          }
        }
      }
    }
    all_par_names <- names(all_par_syms)

    solutions <- list()
    t0_sym <- sympy$Integer(0L)
    x_init_sym <- x0
    event_completed <- FALSE

    for (i in seq_len(nrow(events))) {
      ev <- events[i, ]
      v_sym <- all_par_syms[[ev$value]]

      sol_seg <- solve_ode_segment(rhs_str, t0_sym, x_init_sym, local_dict)
      sol_seg <- sympy$simplify(sol_seg)

      cat(sprintf("=== Segment %d ===\n", i))
      cat(sprintf("%s(t) = %s\n", var_name, as.character(sol_seg)))

      if (!is.na(ev$root) && nchar(ev$root) > 0) {
        cat(sprintf("Zero-Crossing-Bedingung: %s = 0\n", ev$root))
        te_sym <- find_zero_crossing(sol_seg, ev$root, t0_sym,
                                     c(local_dict, all_par_syms), par_vals)

        if (is.null(te_sym)) {
          cat("WARNUNG: Kein Nulldurchgang gefunden, beende Event-Verarbeitung\n\n")
          solutions[[length(solutions) + 1]] <- list(
            expr = sol_seg, t_start = t0_sym, t_end = sympy$oo, t_end_is_inf = TRUE
          )
          event_completed <- TRUE
          break
        }
        te_sym <- sympy$simplify(te_sym)

      } else if (!is.na(ev$time) && nchar(ev$time) > 0) {
        te_sym <- all_par_syms[[ev$time]]
      } else {
        stop("Event muss entweder 'time' oder 'root' haben (nicht NA/leer)")
      }

      solutions[[length(solutions) + 1]] <- list(
        expr = sol_seg, t_start = t0_sym, t_end = te_sym, t_end_is_inf = FALSE
      )

      cat(sprintf("Segment-Bereich: t ∈ [%s, %s)\n", as.character(t0_sym), as.character(te_sym)))

      x_at_te <- sol_seg$subs(t, te_sym)

      x_after_event <- switch(ev$method,
                              add = x_at_te + v_sym,
                              mult = x_at_te * v_sym,
                              replace = v_sym,
                              stop("Unbekannte Event-Methode: ", ev$method)
      )
      x_after_event <- sympy$simplify(x_after_event)

      cat(sprintf("Event: %s → %s %s %s\n", var_name, var_name,
                  switch(ev$method, add = "+", mult = "*", replace = ":="),
                  as.character(v_sym)))
      cat(sprintf("%s(t*⁺) = %s\n\n", var_name, as.character(x_after_event)))

      t0_sym <- te_sym
      x_init_sym <- x_after_event
    }

    # Letztes Segment falls nötig
    if (!event_completed) {
      sol_final <- solve_ode_segment(rhs_str, t0_sym, x_init_sym, local_dict)
      sol_final <- sympy$simplify(sol_final)
      solutions[[length(solutions) + 1]] <- list(
        expr = sol_final, t_start = t0_sym, t_end = sympy$oo, t_end_is_inf = TRUE
      )

      cat(sprintf("=== Finales Segment: t ∈ [%s, ∞) ===\n", as.character(t0_sym)))
      cat(sprintf("%s(t) = %s\n\n", var_name, as.character(sol_final)))
    }

    # Piecewise
    if (length(solutions) == 1) {
      full_solution <- solutions[[1]]$expr
    } else {
      pieces <- lapply(seq_along(solutions), function(i) {
        s <- solutions[[i]]
        if (i == length(solutions)) {
          list(s$expr, sympy$true)
        } else {
          list(s$expr, sympy$And(t >= s$t_start, t < s$t_end))
        }
      })
      full_solution <- do.call(sympy$Piecewise, pieces)
    }

    all_par_names <- names(all_par_syms)
  }

  cat("=== Vollständige Lösung ===\n")
  cat(sprintf("%s(t) = %s\n\n", var_name, as.character(full_solution)))

  # Sensitivitäten (1. Ordnung)
  sens_par_names <- intersect(all_par_names, c("x0", names(pars)))

  # Hilfsfunktion: stückweise Ableitung nach einem Symbol
  piecewise_diff <- function(solutions, sym) {
    if (length(solutions) == 1) {
      sympy$diff(solutions[[1]]$expr, sym)
    } else {
      pieces <- lapply(seq_along(solutions), function(i) {
        s <- solutions[[i]]
        d_seg <- sympy$diff(s$expr, sym)
        if (i == length(solutions)) {
          list(d_seg, sympy$true)
        } else {
          list(d_seg, sympy$And(t >= s$t_start, t < s$t_end))
        }
      })
      do.call(sympy$Piecewise, pieces)
    }
  }

  cat("=== Sensitivitäten (1. Ordnung) ===\n")
  sensitivities <- lapply(sens_par_names, function(p) {
    piecewise_diff(solutions, all_par_syms[[p]])
  })
  names(sensitivities) <- sens_par_names

  for (p in sens_par_names) {
    p_display <- if (p == "x0") var_name else p
    cat(sprintf("∂%s/∂%s = %s\n", var_name, p_display, as.character(sympy$simplify(sensitivities[[p]]))))
  }
  cat("\n")

  # Sensitivitäten (2. Ordnung) — untere Dreiecksmatrix (Schwarz)
  # Paare (i,j) mit i >= j, d.h. ∂²x/∂p_i∂p_j
  n_pars <- length(sens_par_names)
  sens2_pairs <- list()
  for (i in seq_len(n_pars)) {
    for (j in seq_len(i)) {
      sens2_pairs[[length(sens2_pairs) + 1]] <- c(i, j)
    }
  }

  cat("=== Sensitivitäten (2. Ordnung, untere Dreiecksmatrix) ===\n")
  sensitivities2 <- list()
  for (idx in seq_along(sens2_pairs)) {
    ij <- sens2_pairs[[idx]]
    pi_name <- sens_par_names[ij[1]]
    pj_name <- sens_par_names[ij[2]]
    pi_sym <- all_par_syms[[pi_name]]
    pj_sym <- all_par_syms[[pj_name]]

    # ∂²x/∂p_i∂p_j  — erst nach p_j, dann nach p_i (Schwarz: Reihenfolge egal)
    sens2_expr <- piecewise_diff(solutions, pj_sym)
    # Zweite Ableitung: diff des Ergebnis nach p_i
    if (length(solutions) == 1) {
      sens2_expr <- sympy$diff(sens2_expr, pi_sym)
    } else {
      pieces2 <- lapply(seq_along(solutions), function(k) {
        s <- solutions[[k]]
        d2_seg <- sympy$diff(sympy$diff(s$expr, pj_sym), pi_sym)
        if (k == length(solutions)) {
          list(d2_seg, sympy$true)
        } else {
          list(d2_seg, sympy$And(t >= s$t_start, t < s$t_end))
        }
      })
      sens2_expr <- do.call(sympy$Piecewise, pieces2)
    }

    key <- paste0(pi_name, ".", pj_name)
    sensitivities2[[key]] <- sens2_expr

    pi_display <- if (pi_name == "x0") var_name else pi_name
    pj_display <- if (pj_name == "x0") var_name else pj_name
    cat(sprintf("∂²%s/∂%s∂%s = %s\n", var_name, pi_display, pj_display,
                as.character(sympy$simplify(sens2_expr))))
  }
  cat("\n")

  # Numerische Evaluation
  sympy_numeric_eval <- function(expr) {
    expr_sub <- expr
    for (p in names(par_vals)) {
      if (p %in% names(all_par_syms)) {
        expr_sub <- expr_sub$subs(all_par_syms[[p]], par_vals[[p]])
      }
    }
    expr_sub <- sympy$simplify(expr_sub)

    function(t_vals) {
      sapply(t_vals, function(tv) {
        result <- expr_sub$subs(t, tv)
        result_eval <- result$evalf()
        tryCatch({
          as.numeric(builtins$float(result_eval))
        }, error = function(e) {
          tryCatch({
            as.numeric(py_to_r(sympy$re(result_eval)))
          }, error = function(e2) {
            NA_real_
          })
        })
      })
    }
  }

  # Ergebnis
  result <- data.table(time = times)

  f_sol <- sympy_numeric_eval(full_solution)
  result[[var_name]] <- f_sol(times)

  # 1. Ordnung
  for (p in sens_par_names) {
    p_display <- if (p == "x0") var_name else p
    col_name <- paste0("\u2202", var_name, "/\u2202", p_display)
    f_sens <- sympy_numeric_eval(sensitivities[[p]])
    result[[col_name]] <- f_sens(times)
  }

  # 2. Ordnung
  for (key in names(sensitivities2)) {
    parts <- strsplit(key, "\\.")[[1]]
    pi_name <- parts[1]
    pj_name <- parts[2]
    pi_display <- if (pi_name == "x0") var_name else pi_name
    pj_display <- if (pj_name == "x0") var_name else pj_name
    col_name <- paste0("\u2202\u00B2", var_name, "/\u2202", pi_display, "\u2202", pj_display)
    f_sens2 <- sympy_numeric_eval(sensitivities2[[key]])
    result[[col_name]] <- f_sens2(times)
  }

  # Attribute
  attr(result, "solution") <- as.character(full_solution)
  attr(result, "sensitivities") <- lapply(sensitivities, as.character)
  attr(result, "sensitivities2") <- lapply(sensitivities2, as.character)
  attr(result, "parameters") <- sens_par_names
  attr(result, "events") <- events
  attr(result, "segments") <- lapply(solutions, function(s) as.character(s$expr))
  attr(result, "variable") <- var_name

  result
}
