/*
 BLAS-3 chain rule contraction for the symbolic-mode funCpp() back-end.

 Two inline helpers:

   cppode::chain_jac(J, S, J_theta, n_obs, n_out, n_diff, n_theta)
       Per-obs DGEMM: J_theta[obs, ., .] = J[obs, ., .] %*% S[obs, ., .]
       Layouts (R column-major):
           J        [n_obs, n_out, n_diff]
           S        [n_obs, n_diff, n_theta]
           J_theta  [n_obs, n_out, n_theta]

   cppode::chain_hess(H, J, S, S2, H_theta,
                      n_obs, n_out, n_diff, n_theta)
       Per-obs / per-output: H_theta[obs, o, ., .] = S' H[obs, o, ., .] S
                                               + sum_i J[obs, o, i] S2[obs, i, ., .]
       Layouts:
           H        [n_obs, n_out, n_diff, n_diff]
           J        [n_obs, n_out, n_diff]
           S        [n_obs, n_diff, n_theta]
           S2       [n_obs, n_diff, n_theta, n_theta] or nullptr (skip J*S2 term)
           H_theta  [n_obs, n_out, n_theta, n_theta]

 R-layout slices are not stride-1 in the leading axis, so each obs (and each
 output for the Hessian) is packed into a column-major scratch buffer before
 the DGEMM call and scattered back afterwards. The pack/unpack overhead is
 O(n_diff^2) per (obs, out); the DGEMM proper is O(n_diff * n_diff * n_theta)
 per (obs, out), so the BLAS-3 inner kernel dominates as soon as n_theta is
 a few or n_diff is moderate.

 The J*S2 contribution uses DGEMV: viewing S2[obs, ., ., .] as an
 [n_diff, n_theta * n_theta] matrix, the contribution to H_theta_o is
 S2_obs' * J_slice  -- exactly DGEMV with trans = 'T'.

 Copyright (C) 2026 Simon Beyer
*/

#ifndef CPPODE_CHAIN_BLAS_HPP
#define CPPODE_CHAIN_BLAS_HPP

#ifndef USE_FC_LEN_T
#define USE_FC_LEN_T
#endif

#include <R_ext/BLAS.h>
#include <vector>
#include <cstddef>

#ifndef FCONE
#define FCONE
#endif

namespace cppode {

inline void chain_jac(const double* J,
                      const double* S,
                      double* J_theta,
                      int n_obs, int n_out, int n_diff, int n_theta)
{
    if (n_obs <= 0 || n_out <= 0 || n_theta <= 0) return;

    std::vector<double> J_buf(static_cast<std::size_t>(n_out) * n_diff);
    std::vector<double> S_buf(static_cast<std::size_t>(n_diff) * n_theta);
    std::vector<double> C_buf(static_cast<std::size_t>(n_out) * n_theta);

    const double one = 1.0, zero = 0.0;
    const char N = 'N';

    for (int obs = 0; obs < n_obs; ++obs) {
        // Pack J[obs, ., .] -> J_buf [n_out, n_diff], column-major.
        for (int j = 0; j < n_diff; ++j)
            for (int i = 0; i < n_out; ++i)
                J_buf[static_cast<std::size_t>(i) + static_cast<std::size_t>(n_out) * j]
                    = J[static_cast<std::size_t>(obs) + static_cast<std::size_t>(n_obs) *
                        (static_cast<std::size_t>(i) + static_cast<std::size_t>(n_out) * j)];

        // Pack S[obs, ., .] -> S_buf [n_diff, n_theta], column-major.
        for (int t = 0; t < n_theta; ++t)
            for (int j = 0; j < n_diff; ++j)
                S_buf[static_cast<std::size_t>(j) + static_cast<std::size_t>(n_diff) * t]
                    = S[static_cast<std::size_t>(obs) + static_cast<std::size_t>(n_obs) *
                        (static_cast<std::size_t>(j) + static_cast<std::size_t>(n_diff) * t)];

        // C = J_buf * S_buf, [n_out, n_theta].
        F77_CALL(dgemm)(&N, &N, &n_out, &n_theta, &n_diff, &one,
                        J_buf.data(), &n_out,
                        S_buf.data(), &n_diff,
                        &zero, C_buf.data(), &n_out FCONE FCONE);

        // Scatter C_buf back into J_theta[obs, ., .].
        for (int t = 0; t < n_theta; ++t)
            for (int i = 0; i < n_out; ++i)
                J_theta[static_cast<std::size_t>(obs) + static_cast<std::size_t>(n_obs) *
                        (static_cast<std::size_t>(i) + static_cast<std::size_t>(n_out) * t)]
                    = C_buf[static_cast<std::size_t>(i) + static_cast<std::size_t>(n_out) * t];
    }
}

inline void chain_hess(const double* H,
                       const double* J,
                       const double* S,
                       const double* S2,
                       double* H_theta,
                       int n_obs, int n_out, int n_diff, int n_theta)
{
    if (n_obs <= 0 || n_out <= 0 || n_theta <= 0) return;

    const std::size_t nx_diff   = static_cast<std::size_t>(n_diff);
    const std::size_t nx_theta  = static_cast<std::size_t>(n_theta);
    const std::size_t nx_out    = static_cast<std::size_t>(n_out);
    const std::size_t nx_obs    = static_cast<std::size_t>(n_obs);
    const std::size_t nx_th2    = nx_theta * nx_theta;

    std::vector<double> H_buf(nx_diff * nx_diff);
    std::vector<double> S_buf(nx_diff * nx_theta);
    std::vector<double> Tmp(nx_diff * nx_theta);
    std::vector<double> Result(nx_th2);

    std::vector<double> S2_buf;
    std::vector<double> J_slice;
    if (S2 != nullptr) {
        S2_buf.resize(nx_diff * nx_th2);
        J_slice.resize(nx_diff);
    }

    const double one = 1.0, zero = 0.0;
    const char N = 'N', T = 'T';
    const int  inc1 = 1;
    const int  n_th2 = n_theta * n_theta;

    for (int obs = 0; obs < n_obs; ++obs) {
        // Pack S[obs, ., .] -> S_buf [n_diff, n_theta], column-major.
        for (int t = 0; t < n_theta; ++t)
            for (int j = 0; j < n_diff; ++j)
                S_buf[static_cast<std::size_t>(j) + nx_diff * t]
                    = S[static_cast<std::size_t>(obs) + nx_obs *
                        (static_cast<std::size_t>(j) + nx_diff * t)];

        // Pack S2[obs, ., ., .] -> S2_buf [n_diff, n_theta * n_theta], column-major.
        if (S2 != nullptr) {
            for (int b = 0; b < n_theta; ++b)
                for (int a = 0; a < n_theta; ++a)
                    for (int j = 0; j < n_diff; ++j)
                        S2_buf[static_cast<std::size_t>(j) + nx_diff *
                               (static_cast<std::size_t>(a) + nx_theta * b)]
                            = S2[static_cast<std::size_t>(obs) + nx_obs *
                                 (static_cast<std::size_t>(j) + nx_diff *
                                  (static_cast<std::size_t>(a) + nx_theta * b))];
        }

        for (int o = 0; o < n_out; ++o) {
            // Pack H[obs, o, ., .] -> H_buf [n_diff, n_diff], column-major.
            for (int j = 0; j < n_diff; ++j)
                for (int i = 0; i < n_diff; ++i)
                    H_buf[static_cast<std::size_t>(i) + nx_diff * j]
                        = H[static_cast<std::size_t>(obs) + nx_obs *
                            (static_cast<std::size_t>(o) + nx_out *
                             (static_cast<std::size_t>(i) + nx_diff * j))];

            // Tmp = H_buf * S_buf,  [n_diff, n_theta]
            F77_CALL(dgemm)(&N, &N, &n_diff, &n_theta, &n_diff, &one,
                            H_buf.data(), &n_diff,
                            S_buf.data(), &n_diff,
                            &zero, Tmp.data(), &n_diff FCONE FCONE);

            // Result = S_buf' * Tmp,  [n_theta, n_theta]
            F77_CALL(dgemm)(&T, &N, &n_theta, &n_theta, &n_diff, &one,
                            S_buf.data(), &n_diff,
                            Tmp.data(), &n_diff,
                            &zero, Result.data(), &n_theta FCONE FCONE);

            // Optional J*S2 contribution: Result += S2_buf' * J_slice
            if (S2 != nullptr) {
                for (int j = 0; j < n_diff; ++j)
                    J_slice[static_cast<std::size_t>(j)]
                        = J[static_cast<std::size_t>(obs) + nx_obs *
                            (static_cast<std::size_t>(o) + nx_out * static_cast<std::size_t>(j))];

                F77_CALL(dgemv)(&T, &n_diff, &n_th2, &one,
                                S2_buf.data(), &n_diff,
                                J_slice.data(), &inc1,
                                &one, Result.data(), &inc1 FCONE);
            }

            // Scatter Result back into H_theta[obs, o, ., .].
            for (int b = 0; b < n_theta; ++b)
                for (int a = 0; a < n_theta; ++a)
                    H_theta[static_cast<std::size_t>(obs) + nx_obs *
                            (static_cast<std::size_t>(o) + nx_out *
                             (static_cast<std::size_t>(a) + nx_theta * b))]
                        = Result[static_cast<std::size_t>(a) + nx_theta * b];
        }
    }
}

} // namespace cppode

#endif // CPPODE_CHAIN_BLAS_HPP
