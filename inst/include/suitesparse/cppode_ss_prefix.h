/*
 * CppODE SuiteSparse symbol prefix header
 * =========================================
 *
 * Must be included BEFORE any SuiteSparse header to rename all public
 * symbols with a cppode_ prefix.  This prevents linker collisions with
 * R's Matrix package (which bundles its own AMD, COLAMD, etc.).
 *
 * Copyright (C) 2026 Simon Beyer — same license as CppODE.
 */

#ifndef CPPODE_SS_PREFIX_H
#define CPPODE_SS_PREFIX_H

/* ---- SuiteSparse_config ---- */
#define SuiteSparse_config_malloc_func_get   cppode_SuiteSparse_config_malloc_func_get
#define SuiteSparse_config_calloc_func_get   cppode_SuiteSparse_config_calloc_func_get
#define SuiteSparse_config_realloc_func_get  cppode_SuiteSparse_config_realloc_func_get
#define SuiteSparse_config_free_func_get     cppode_SuiteSparse_config_free_func_get
#define SuiteSparse_config_printf_func_get   cppode_SuiteSparse_config_printf_func_get
#define SuiteSparse_config_hypot_func_get    cppode_SuiteSparse_config_hypot_func_get
#define SuiteSparse_config_divcomplex_func_get cppode_SuiteSparse_config_divcomplex_func_get
#define SuiteSparse_config_malloc_func_set   cppode_SuiteSparse_config_malloc_func_set
#define SuiteSparse_config_calloc_func_set   cppode_SuiteSparse_config_calloc_func_set
#define SuiteSparse_config_realloc_func_set  cppode_SuiteSparse_config_realloc_func_set
#define SuiteSparse_config_free_func_set     cppode_SuiteSparse_config_free_func_set
#define SuiteSparse_config_printf_func_set   cppode_SuiteSparse_config_printf_func_set
#define SuiteSparse_config_hypot_func_set    cppode_SuiteSparse_config_hypot_func_set
#define SuiteSparse_config_divcomplex_func_set cppode_SuiteSparse_config_divcomplex_func_set
#define SuiteSparse_config_malloc            cppode_SuiteSparse_config_malloc
#define SuiteSparse_config_calloc            cppode_SuiteSparse_config_calloc
#define SuiteSparse_config_realloc           cppode_SuiteSparse_config_realloc
#define SuiteSparse_config_free              cppode_SuiteSparse_config_free
#define SuiteSparse_config_hypot             cppode_SuiteSparse_config_hypot
#define SuiteSparse_config_divcomplex        cppode_SuiteSparse_config_divcomplex
#define SuiteSparse_start                    cppode_SuiteSparse_start
#define SuiteSparse_finish                   cppode_SuiteSparse_finish
#define SuiteSparse_version                  cppode_SuiteSparse_version
#define SuiteSparse_malloc                   cppode_SuiteSparse_malloc
#define SuiteSparse_calloc                   cppode_SuiteSparse_calloc
#define SuiteSparse_realloc                  cppode_SuiteSparse_realloc
#define SuiteSparse_free                     cppode_SuiteSparse_free
#define SuiteSparse_hypot                    cppode_SuiteSparse_hypot
#define SuiteSparse_divcomplex               cppode_SuiteSparse_divcomplex
#define SuiteSparse_BLAS_library             cppode_SuiteSparse_BLAS_library
#define SuiteSparse_BLAS_integer_size        cppode_SuiteSparse_BLAS_integer_size
#define SuiteSparse_tic                      cppode_SuiteSparse_tic
#define SuiteSparse_toc                      cppode_SuiteSparse_toc
#define SuiteSparse_time                     cppode_SuiteSparse_time

/* ---- COLAMD extras (symamd) ---- */
#define symamd                   cppode_symamd
#define symamd_l                 cppode_symamd_l
#define symamd_report            cppode_symamd_report
#define symamd_l_report          cppode_symamd_l_report
#define symamd_set_defaults      cppode_symamd_set_defaults
#define symamd_l_set_defaults    cppode_symamd_l_set_defaults

/* ---- AMD (int32 + int64) ---- */
#define amd_order           cppode_amd_order
#define amd_l_order         cppode_amd_l_order
#define amd_2               cppode_amd_2
#define amd_l2              cppode_amd_l2
#define amd_valid           cppode_amd_valid
#define amd_l_valid         cppode_amd_l_valid
#define amd_defaults        cppode_amd_defaults
#define amd_l_defaults      cppode_amd_l_defaults
#define amd_control         cppode_amd_control
#define amd_l_control       cppode_amd_l_control
#define amd_info            cppode_amd_info
#define amd_l_info          cppode_amd_l_info
#define amd_version         cppode_amd_version
#define amd_aat             cppode_amd_aat
#define amd_l_aat           cppode_amd_l_aat
#define amd_1               cppode_amd_1
#define amd_l1              cppode_amd_l1
#define amd_postorder       cppode_amd_postorder
#define amd_l_postorder     cppode_amd_l_postorder
#define amd_post_tree       cppode_amd_post_tree
#define amd_l_post_tree     cppode_amd_l_post_tree
#define amd_preprocess      cppode_amd_preprocess
#define amd_l_preprocess    cppode_amd_l_preprocess
#define amd_dump            cppode_amd_dump
#define amd_l_dump          cppode_amd_l_dump

/* ---- BTF ---- */
#define btf_maxtrans        cppode_btf_maxtrans
#define btf_l_maxtrans      cppode_btf_l_maxtrans
#define btf_strongcomp      cppode_btf_strongcomp
#define btf_l_strongcomp    cppode_btf_l_strongcomp
#define btf_order           cppode_btf_order
#define btf_l_order         cppode_btf_l_order
#define btf_version         cppode_btf_version

/* ---- COLAMD ---- */
#define colamd_recommended       cppode_colamd_recommended
#define colamd_l_recommended     cppode_colamd_l_recommended
#define colamd_set_defaults      cppode_colamd_set_defaults
#define colamd_l_set_defaults    cppode_colamd_l_set_defaults
#define colamd                   cppode_colamd
#define colamd_l                 cppode_colamd_l
#define colamd_report            cppode_colamd_report
#define colamd_l_report          cppode_colamd_l_report
#define colamd_version           cppode_colamd_version

/* ---- KLU (int32 real — the only variant we use) ---- */
#define klu_defaults        cppode_klu_defaults
#define klu_analyze         cppode_klu_analyze
#define klu_analyze_given   cppode_klu_analyze_given
#define klu_factor          cppode_klu_factor
#define klu_refactor        cppode_klu_refactor
#define klu_solve           cppode_klu_solve
#define klu_tsolve          cppode_klu_tsolve
#define klu_free_symbolic   cppode_klu_free_symbolic
#define klu_free_numeric    cppode_klu_free_numeric
#define klu_sort            cppode_klu_sort
#define klu_scale           cppode_klu_scale
#define klu_extract         cppode_klu_extract
#define klu_diagnostics     cppode_klu_diagnostics
#define klu_rgrowth         cppode_klu_rgrowth
#define klu_condest         cppode_klu_condest
#define klu_rcond           cppode_klu_rcond
#define klu_flops           cppode_klu_flops
#define klu_malloc          cppode_klu_malloc
#define klu_realloc         cppode_klu_realloc
#define klu_free            cppode_klu_free
#define klu_version         cppode_klu_version
#define klu_kernel          cppode_klu_kernel
#define klu_kernel_factor   cppode_klu_kernel_factor
#define klu_valid           cppode_klu_valid
#define klu_valid_LU        cppode_klu_valid_LU
#define klu_alloc_symbolic  cppode_klu_alloc_symbolic
#define klu_add_size_t      cppode_klu_add_size_t
#define klu_mult_size_t     cppode_klu_mult_size_t
#define klu_lsolve          cppode_klu_lsolve
#define klu_ltsolve         cppode_klu_ltsolve
#define klu_usolve          cppode_klu_usolve
#define klu_utsolve         cppode_klu_utsolve

/* ---- KLU struct type names (also need prefixing) ---- */
#define klu_symbolic        cppode_klu_symbolic
#define klu_numeric         cppode_klu_numeric
#define klu_common          cppode_klu_common
#define klu_common_struct   cppode_klu_common_struct

#endif /* CPPODE_SS_PREFIX_H */
