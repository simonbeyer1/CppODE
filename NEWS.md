# CppODE 1.0.0

## Major Features

* **First stable release.**

* **Automated generation, compilation, and execution of C++ code for ODE models** via `CppODE()`.

* **Automated generation, compilation, and execution of C++ code for functions** via `funCpp()`:
  - Including first- and second-order derivatives

* **Symbolic differentiation** via `derivSymb()`:
  - Uses SymPy backend to compute analytical Jacobians and Hessians
  - Used by `funCpp()` for derivative generation

## Sensitivity Analysis and Derivatives

* **Support for first- and second-order sensitivity analysis in `CppODE()`** via automatic differentiation.

* **Derivatives (sensitivities and Hessians) are returned as list of arrays.**

## Event Handling

* **Event handling with parameterized events**:
  - Uses deSolve-style event logic based on the Boost.Odeint library

## Numerical Methods

* **Adaptive step size control and dense output integration** with derivative components.

* **Automatic initial step size estimation** based on the heuristic proposed by Hairer, Nørsett, and Wanner.
