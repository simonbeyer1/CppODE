/*
 CppODE return-code constants.

 Values are kept numerically identical to the SUNDIALS CVODE flag scheme
 (see sundials/cvode.h) so that solveODE()'s `return_code` can be
 interpreted uniformly across the CVODE backend and the native CppODE
 backend.  The CVODE backend simply passes the raw `CVode()` flag through;
 the native backend maps its own exceptions to the closest CVODE flag.

 Copyright (C) 2026 Simon Beyer
 */

#ifndef CPPODE_RETURN_CODES_HPP
#define CPPODE_RETURN_CODES_HPP

namespace cppode {

// --- Success / informational (CVODE positive codes) ---
constexpr int RC_SUCCESS            =   0;  // CV_SUCCESS
constexpr int RC_TSTOP_RETURN       =   1;  // CV_TSTOP_RETURN
constexpr int RC_ROOT_RETURN        =   2;  // CV_ROOT_RETURN

// --- Error codes (CVODE negative codes) ---
constexpr int RC_TOO_MUCH_WORK      =  -1;  // CV_TOO_MUCH_WORK   (maxsteps)
constexpr int RC_TOO_MUCH_ACC       =  -2;  // CV_TOO_MUCH_ACC
constexpr int RC_ERR_FAILURE        =  -3;  // CV_ERR_FAILURE
constexpr int RC_CONV_FAILURE       =  -4;  // CV_CONV_FAILURE    (maxprogress / repeated conv failures)
constexpr int RC_LINIT_FAIL         =  -5;  // CV_LINIT_FAIL
constexpr int RC_LSETUP_FAIL        =  -6;  // CV_LSETUP_FAIL     (KLU factor failure in native backend)
constexpr int RC_LSOLVE_FAIL        =  -7;  // CV_LSOLVE_FAIL
constexpr int RC_RHSFUNC_FAIL       =  -8;  // CV_RHSFUNC_FAIL
constexpr int RC_FIRST_RHSFUNC_ERR  =  -9;  // CV_FIRST_RHSFUNC_ERR
constexpr int RC_REPTD_RHSFUNC_ERR  = -10;  // CV_REPTD_RHSFUNC_ERR
constexpr int RC_UNREC_RHSFUNC_ERR  = -11;  // CV_UNREC_RHSFUNC_ERR
constexpr int RC_RTFUNC_FAIL        = -12;  // CV_RTFUNC_FAIL
constexpr int RC_ILL_INPUT          = -22;  // CV_ILL_INPUT
constexpr int RC_NO_MALLOC          = -23;  // CV_NO_MALLOC
constexpr int RC_TOO_CLOSE          = -27;  // CV_TOO_CLOSE
constexpr int RC_UNRECOGNIZED_ERR   = -99;  // CV_UNRECOGNIZED_ERR (fallback for C++ exceptions)

} // namespace cppode

#endif // CPPODE_RETURN_CODES_HPP
