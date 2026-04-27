#!/usr/bin/env julia
# =================================================================
# Companion benchmark to bench_cascade_model.R
#
# Runs the same 14-state cascade signaling model with three
# OrdinaryDiffEq.jl solvers (Tsit5, AutoTsit5(Rodas4), QNDF).
#
# When calcSens=true, also benchmarks forward sensitivities via
# ForwardDiff.jacobian over solve. Uses ForwardDiff's default chunk
# size (= 12 for 42 inputs, i.e. 4 passes with Dual{Float64,12});
# larger chunks segfault on Julia 1.12. This is forward AD over the
# solver itself -- conceptually the same scheme as CppODE's "dual"
# backend, just split across multiple sweeps instead of one.
#
# Per-solver timing stats are written to CSV so the R driver can fold
# them into its summary table.
#
# Usage:
#   julia --project=<this dir> bench_cascade_julia.jl \
#         [calcSens=true|false] [abstol] [reltol] [n_evals] \
#         [out_bench_csv] [out_solution_csv]
#
# The Project.toml/Manifest.toml in this directory pin
# OrdinaryDiffEq, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqBDF,
# BenchmarkTools, ForwardDiff.
# =================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

using OrdinaryDiffEq
using OrdinaryDiffEqRosenbrock: Rodas4
using OrdinaryDiffEqBDF: QNDF
using BenchmarkTools
using ForwardDiff
using Printf

const calcSens  = length(ARGS) >= 1 ? parse(Bool,    ARGS[1]) : false
const abstol    = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1e-6
const reltol    = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-6
const n_evals   = length(ARGS) >= 4 ? parse(Int,     ARGS[4]) : 20
const out_bench = length(ARGS) >= 5 ? ARGS[5] : "julia_bench_results.csv"
const out_sol   = length(ARGS) >= 6 ? ARGS[6] : "julia_solution.csv"

# -----------------------------------------------------------------
# 14-state cascade RHS  (mirrors rhs in bench_cascade_model.R)
# -----------------------------------------------------------------
function cascade!(du, u, p, t)
    R_, LR, KKK, KKKa, KK, KKa, K_, Ka, TF, TFa,
        mRNA_target, P_target, mRNA_inhib, P_inhib = u
    L, kon, koff, kint, ksyn_R,
        k1, Km1, k2, Km2,
        k3, Km3, k4, Km4,
        k5, Km5, k6, Km6, ki,
        k7, k8,
        ktx, kdeg_mRNA, ktl, kdeg_P,
        ktx_i, kdeg_mRNA_i, ktl_i, kdeg_P_i = p

    du[1]  = -kon*L*R_ + koff*LR + ksyn_R
    du[2]  =  kon*L*R_ - koff*LR - kint*LR
    du[3]  = -k1*LR*KKK/(Km1+KKK) + k2*KKKa/(Km2+KKKa)*(1 + ki*P_inhib)
    du[4]  =  k1*LR*KKK/(Km1+KKK) - k2*KKKa/(Km2+KKKa)*(1 + ki*P_inhib)
    du[5]  = -k3*KKKa*KK/(Km3+KK) + k4*KKa/(Km4+KKa)
    du[6]  =  k3*KKKa*KK/(Km3+KK) - k4*KKa/(Km4+KKa)
    du[7]  = -k5*KKa*K_/(Km5+K_) + k6*Ka/(Km6+Ka)
    du[8]  =  k5*KKa*K_/(Km5+K_) - k6*Ka/(Km6+Ka)
    du[9]  = -k7*Ka*TF + k8*TFa
    du[10] =  k7*Ka*TF - k8*TFa
    du[11] =  ktx*TFa - kdeg_mRNA*mRNA_target
    du[12] =  ktl*mRNA_target - kdeg_P*P_target
    du[13] =  ktx_i*TFa - kdeg_mRNA_i*mRNA_inhib
    du[14] =  ktl_i*mRNA_inhib - kdeg_P_i*P_inhib
    return nothing
end

const state_names = ["R","LR","KKK","KKKa","KK","KKa","K","Ka","TF","TFa",
                     "mRNA_target","P_target","mRNA_inhib","P_inhib"]

const param_names = ["L","kon","koff","kint","ksyn_R",
                     "k1","Km1","k2","Km2",
                     "k3","Km3","k4","Km4",
                     "k5","Km5","k6","Km6","ki",
                     "k7","k8",
                     "ktx","kdeg_mRNA","ktl","kdeg_P",
                     "ktx_i","kdeg_mRNA_i","ktl_i","kdeg_P_i"]

# Sensitivity variable names: theta = [u0_init...; params...] (matches cOde
# sensitivitiesSymb convention -- initial-state sens named by state, param sens
# named by parameter).
const sens_var_names = vcat(state_names, param_names)

const u0 = [100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0,
            100.0, 0.0,   0.0, 0.0,   0.0, 0.0]

# Order matches the cascade! signature above and param_names above.
const p_ = [10.0, 0.1, 0.05, 0.01, 0.5,
            0.5, 10.0, 0.2, 10.0,
            0.3, 10.0, 0.15, 10.0,
            0.4, 10.0, 0.1, 10.0, 0.5,
            0.2, 0.05,
            1.0, 0.1, 0.5, 0.05,
            0.3, 0.2, 0.4, 0.1]

const tspan    = (0.0, 100.0)
const times    = collect(range(0.0, 100.0; length = 500))
const N_STATES = length(u0)
const N_PARAMS = length(p_)
const N_THETA  = N_STATES + N_PARAMS                 # = 42, matches CppODE default
const theta0   = Float64[u0; p_]

prob_base = ODEProblem(cascade!, u0, tspan, p_)

# -----------------------------------------------------------------
# Custom internalnorm: WRMS-2 norm over primal value AND every Dual
# partial, recursively unwrapping nested Duals. The default
# ODE_DEFAULT_NORM uses only the primal of each Dual, so the step
# controller would ignore tangent error. Earlier versions of this
# script used an unscaled inf-norm -- effective at making sens
# components controlled, but not tolerance-conformant: a large sens
# magnitude would force much tighter steps than the user asked for.
#
# This version mirrors the WRMS-2 norm OrdinaryDiffEq uses for plain
# Float64 states: each scalar contribution x is divided by
# (abstol + reltol * |x|) before being squared and averaged.
#
# Caveat vs OrdinaryDiffEq's "real" path: that one scales with
# max(|u|, |uprev|), but `internalnorm` doesn't see uprev, so we scale
# against |u| only. In practice this matches the requested tolerance
# closely; the only difference is at sharp transients where |uprev|
# would briefly dominate.
#
# Recursion is required because implicit solvers (Rodas4, QNDF) wrap
# the outer Dual{Tag_out, Float64, N} in their own Dual{Tag_in, ...}
# during Newton iteration -- so we have to descend through ALL layers
# to get a real scalar back. Returning a non-Float64 (e.g. an outer
# Dual) breaks dt assignment.
#
# Each invocation of dual_internalnorm accumulates (sumsq, count) and
# returns sqrt(sumsq / count). For a pure-Float64 array this reduces
# exactly to the standard WRMS-2 norm.
@inline function _accum_wrms(acc::Tuple{Float64,Float64}, x::Real,
                             atol::Float64, rtol::Float64)
    v   = float(x)
    sc  = atol + rtol * abs(v)
    r   = v / sc
    return (acc[1] + r * r, acc[2] + 1.0)
end
@inline function _accum_wrms(acc::Tuple{Float64,Float64}, x::ForwardDiff.Dual,
                             atol::Float64, rtol::Float64)
    acc = _accum_wrms(acc, ForwardDiff.value(x), atol, rtol)
    pp  = ForwardDiff.partials(x)
    @inbounds for i in 1:length(pp)
        acc = _accum_wrms(acc, pp[i], atol, rtol)
    end
    return acc
end

function dual_internalnorm(u::AbstractArray, t)
    acc = (0.0, 0.0)
    @inbounds for ui in u
        acc = _accum_wrms(acc, ui, abstol, reltol)
    end
    return acc[2] == 0.0 ? 0.0 : sqrt(acc[1] / acc[2])
end
function dual_internalnorm(u::Real, t)
    acc = _accum_wrms((0.0, 0.0), u, abstol, reltol)
    return acc[2] == 0.0 ? 0.0 : sqrt(acc[1] / acc[2])
end

# Flat solve: theta = [u0; p] -> stacked state trajectory (length N_STATES * length(times))
function sens_solve_flat(theta::AbstractVector, alg, abstol, reltol)
    u0_loc = theta[1:N_STATES]
    p_loc  = @view theta[N_STATES+1:end]
    prob   = ODEProblem(cascade!, u0_loc, tspan, p_loc)
    sol    = solve(prob, alg; saveat = times,
                   abstol = abstol, reltol = reltol,
                   internalnorm = dual_internalnorm)
    return reduce(vcat, sol.u)
end

solvers = [
    ("Tsit5",                 Tsit5()),
    ("AutoTsit5(Rodas4)",     AutoTsit5(Rodas4())),
    ("QNDF",                  QNDF()),
]

@printf("Julia %s | abstol=%.1e reltol=%.1e samples=%d | calcSens=%s%s\n",
        VERSION, abstol, reltol, n_evals, calcSens,
        calcSens ? " (ForwardDiff over solve + WRMS dual-aware internalnorm)" : "")

io = IOBuffer()
println(io, "solver,min_ms,mean_ms,median_ms,max_ms,neval")

for (name, alg) in solvers
    label = calcSens ? "$(name) +sens" : name
    @printf("  %-26s ... ", label)
    flush(stdout)

    if calcSens
        let f = theta -> sens_solve_flat(theta, alg, abstol, reltol)
            ForwardDiff.jacobian(f, theta0)  # warm-up (compiles solver for Dual numbers)
            b = @benchmark ForwardDiff.jacobian($f, $theta0) samples = n_evals evals = 1 seconds = 600
            min_ms    = minimum(b).time / 1e6
            mean_ms   = mean(b).time    / 1e6
            median_ms = median(b).time  / 1e6
            max_ms    = maximum(b).time / 1e6
            @printf(io, "%s,%.6f,%.6f,%.6f,%.6f,%d\n",
                    label, min_ms, mean_ms, median_ms, max_ms, length(b.times))
            @printf("median = %8.3f ms  (min %8.3f, max %8.3f, n=%d)\n",
                    median_ms, min_ms, max_ms, length(b.times))
        end
    else
        solve(prob_base, alg; saveat = times, abstol = abstol, reltol = reltol)  # warm-up
        b = @benchmark solve($prob_base, $alg; saveat = $times,
                            abstol = $abstol, reltol = $reltol) samples = n_evals evals = 1 seconds = 300
        min_ms    = minimum(b).time / 1e6
        mean_ms   = mean(b).time    / 1e6
        median_ms = median(b).time  / 1e6
        max_ms    = maximum(b).time / 1e6
        @printf(io, "%s,%.6f,%.6f,%.6f,%.6f,%d\n",
                label, min_ms, mean_ms, median_ms, max_ms, length(b.times))
        @printf("median = %8.3f ms  (min %8.3f, max %8.3f, n=%d)\n",
                median_ms, min_ms, max_ms, length(b.times))
    end
end

write(out_bench, take!(io))
println("Wrote ", out_bench)

# Reference solution from Tsit5 at the requested tolerance.
# When calcSens=true, also dump the full Jacobian d state(t_i) / d theta_k,
# laid out in cOde "wide" convention so the R driver can compare it to the
# LSODES augmented-system reference column-by-column.
sol_ref = solve(prob_base, Tsit5(); saveat = times, abstol = abstol, reltol = reltol)

open(out_sol, "w") do io2
    cols = String["time"]
    append!(cols, state_names)
    if calcSens
        # cOde convention: outer(state_names, sens_var_names, sep = ".") -- state
        # varies fastest within each sens-var block. Match that here.
        for k in 1:N_THETA
            for j in 1:N_STATES
                push!(cols, string(state_names[j], ".", sens_var_names[k]))
            end
        end
    end
    println(io2, join(cols, ","))

    if calcSens
        # One ForwardDiff jacobian over Tsit5 at the requested tolerance.
        f_ref = theta -> sens_solve_flat(theta, Tsit5(), 1e-12, 1e-12)
        J_ref = ForwardDiff.jacobian(f_ref, theta0)
        # J_ref shape: (N_STATES * length(times), N_THETA)
        # Row m = (i-1)*N_STATES + j  ->  d state_j(t_i) / d theta_k
        # Reshape to (N_STATES, length(times), N_THETA), then permute to (i, j, k).
        sens_arr = permutedims(reshape(J_ref, N_STATES, length(times), N_THETA), (2, 1, 3))

        for i in eachindex(times)
            row = Float64[times[i]]
            append!(row, sol_ref.u[i])
            for k in 1:N_THETA
                for j in 1:N_STATES
                    push!(row, sens_arr[i, j, k])
                end
            end
            println(io2, join(row, ","))
        end
    else
        for i in eachindex(times)
            row = [times[i]; sol_ref.u[i]...]
            println(io2, join(row, ","))
        end
    end
end
println("Wrote ", out_sol)
