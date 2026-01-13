\dontrun{
# funCpp() makes use of the Python module 'sympy'
eqns <- c(A = "k_p * (k2 + k_d) / (k1*k_d)", B = "k_p/k_d")

f <- funCpp(eqns,
            parameters = c("k_p", "k1", "k2", "k_d"),
            deriv      = TRUE,
            deriv2     = TRUE,
            compile    = TRUE,
            convenient = TRUE)

res <- f(k_p = 0.3, k1 = 0.1, k2 = 0.2, k_d = 0.4, deriv2 = TRUE)

# Output values
res$out

# Jacobian for first (and only) observation
res$jacobian[1, , ]

# Symbolic Jacobian
attributes(f)$jacobian.symb

# Hessian for output "A" at first observation
res$hessian[1, "A", , ]

# Symbolic Hessian
attributes(f)$hessian.symb$A
}


