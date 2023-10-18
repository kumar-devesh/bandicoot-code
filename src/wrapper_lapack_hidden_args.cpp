// Copyright 2023 Ryan Curtin (http://ratml.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



#include <complex>

#include "bandicoot_bits/config.hpp"

#undef COOT_USE_WRAPPER
#define COOT_USE_FORTRAN_HIDDEN_ARGS

// At this stage we have prototypes for LAPACK functions; so, now make the wrapper functions;
// in this version of the wrapper, all functions with const char* arguments must
// also have blas_len arguments indicating the length of the string.

#include "bandicoot_bits/compiler_setup.hpp"
#include "bandicoot_bits/include_opencl.hpp"
#include "bandicoot_bits/include_cuda.hpp"
#include "bandicoot_bits/typedef_elem.hpp"

namespace coot
  {
  #include "bandicoot_bits/def_lapack.hpp"

  extern "C"
    {



    //
    // LU factorisation (no hidden arguments)
    //



    void coot_fortran_prefix(coot_sgetrf)(blas_int* m, blas_int* n,  float* a, blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgetrf)(m, n, a, lda, ipiv, info);
      }



    void coot_fortran_prefix(coot_dgetrf)(blas_int* m, blas_int* n, double* a, blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgetrf)(m, n, a, lda, ipiv, info);
      }



    void coot_fortran_prefix(coot_cgetrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgetrf)(m, n, a, lda, ipiv, info);
      }



    void coot_fortran_prefix(coot_zgetrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda, blas_int* ipiv, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgetrf)(m, n, a, lda, ipiv, info);
      }



    //
    // matrix inversion (triangular matrices)
    //



    void coot_fortran_prefix(coot_strtri)(const char* uplo, const char* diag, blas_int* n,  float* a, blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len)
      {
      coot_fortran_noprefix(coot_strtri)(uplo, diag, n, a, lda, info, uplo_len, diag_len);
      }



    void coot_fortran_prefix(coot_dtrtri)(const char* uplo, const char* diag, blas_int* n, double* a, blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len)
      {
      coot_fortran_noprefix(coot_dtrtri)(uplo, diag, n, a, lda, info, uplo_len, diag_len);
      }



    void coot_fortran_prefix(coot_ctrtri)(const char* uplo, const char* diag, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len)
      {
      coot_fortran_noprefix(coot_ctrtri)(uplo, diag, n, a, lda, info, uplo_len, diag_len);
      }



    void coot_fortran_prefix(coot_ztrtri)(const char* uplo, const char* diag, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len)
      {
      coot_fortran_noprefix(coot_ztrtri)(uplo, diag, n, a, lda, info, uplo_len, diag_len);
      }



    //
    // eigen decomposition of symmetric real matrices by divide and conquer
    //



    void coot_fortran_prefix(coot_ssyevd)(const char* jobz, const char* uplo, blas_int* n,  float* a, blas_int* lda,  float* w,  float* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_ssyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, jobz_len, uplo_len);
      }



    void coot_fortran_prefix(coot_dsyevd)(const char* jobz, const char* uplo, blas_int* n, double* a, blas_int* lda, double* w, double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info, blas_len jobz_len, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_dsyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, jobz_len, uplo_len);
      }



    //
    // Cholesky decomposition
    //



    void coot_fortran_prefix(coot_spotrf)(const char* uplo, blas_int* n,  float* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_spotrf)(uplo, n, a, lda, info, uplo_len);
      }



    void coot_fortran_prefix(coot_dpotrf)(const char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_dpotrf)(uplo, n, a, lda, info, uplo_len);
      }



    void coot_fortran_prefix(coot_cpotrf)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_cpotrf)(uplo, n, a, lda, info, uplo_len);
      }



    void coot_fortran_prefix(coot_zpotrf)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_zpotrf)(uplo, n, a, lda, info, uplo_len);
      }



    //
    // QR decomposition (no hidden arguments)
    //



    void coot_fortran_prefix(coot_sgeqrf)(blas_int* m, blas_int* n,  float* a, blas_int* lda,  float* tau,  float* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dgeqrf)(blas_int* m, blas_int* n, double* a, blas_int* lda, double* tau, double* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_cgeqrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_zgeqrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgeqrf)(m, n, a, lda, tau, work, lwork, info);
      }



    //
    // Q matrix calculation from QR decomposition (real matrices) (no hidden arguments)
    //



    void coot_fortran_prefix(coot_sorgqr)(blas_int* m, blas_int* n, blas_int* k,  float* a, blas_int* lda,  float* tau,  float* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sorgqr)(m, n, k, a, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dorgqr)(blas_int* m, blas_int* n, blas_int* k, double* a, blas_int* lda, double* tau, double* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dorgqr)(m, n, k, a, lda, tau, work, lwork, info);
      }



    //
    // 1-norm
    //



    float  coot_fortran_prefix(coot_slange)(const char* norm, blas_int* m, blas_int* n,  float* a, blas_int* lda,  float* work, blas_len norm_len)
      {
      return coot_fortran_noprefix(coot_slange)(norm, m, n, a, lda, work, norm_len);
      }



    double coot_fortran_prefix(coot_dlange)(const char* norm, blas_int* m, blas_int* n, double* a, blas_int* lda, double* work, blas_len norm_len)
      {
      return coot_fortran_noprefix(coot_dlange)(norm, m, n, a, lda, work, norm_len);
      }



    float  coot_fortran_prefix(coot_clange)(const char* norm, blas_int* m, blas_int* n,   void* a, blas_int* lda,  float* work, blas_len norm_len)
      {
      return coot_fortran_noprefix(coot_clange)(norm, m, n, a, lda, work, norm_len);
      }



    double coot_fortran_prefix(coot_zlange)(const char* norm, blas_int* m, blas_int* n,   void* a, blas_int* lda, double* work, blas_len norm_len)
      {
      return coot_fortran_noprefix(coot_zlange)(norm, m, n, a, lda, work, norm_len);
      }



    //
    // triangular factor of block reflector
    //



    void coot_fortran_prefix(coot_slarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, float*  v, blas_int* ldv, float*  tau, float*  t, blas_int* ldt, blas_len direct_len, blas_len storev_len)
      {
      coot_fortran_noprefix(coot_slarft)(direct, storev, n, k, v, ldv, tau, t, ldt, direct_len, storev_len);
      }



    void coot_fortran_prefix(coot_dlarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, double* v, blas_int* ldv, double* tau, double* t, blas_int* ldt, blas_len direct_len, blas_len storev_len)
      {
      coot_fortran_noprefix(coot_dlarft)(direct, storev, n, k, v, ldv, tau, t, ldt, direct_len, storev_len);
      }



    void coot_fortran_prefix(coot_clarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, void*   v, blas_int* ldv, void*   tau, void*   t, blas_int* ldt, blas_len direct_len, blas_len storev_len)
      {
      coot_fortran_noprefix(coot_clarft)(direct, storev, n, k, v, ldv, tau, t, ldt, direct_len, storev_len);
      }



    void coot_fortran_prefix(coot_zlarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, void*   v, blas_int* ldv, void*   tau, void*   t, blas_int* ldt, blas_len direct_len, blas_len storev_len)
      {
      coot_fortran_noprefix(coot_zlarft)(direct, storev, n, k, v, ldv, tau, t, ldt, direct_len, storev_len);
      }



    //
    // generate an elementary reflector (no hidden arguments)
    //



    void coot_fortran_prefix(coot_slarfg)(const blas_int* n, float*  alpha, float*  x, const blas_int* incx, float*  tau)
      {
      coot_fortran_noprefix(coot_slarfg)(n, alpha, x, incx, tau);
      }



    void coot_fortran_prefix(coot_dlarfg)(const blas_int* n, double* alpha, double* x, const blas_int* incx, double* tau)
      {
      coot_fortran_noprefix(coot_dlarfg)(n, alpha, x, incx, tau);
      }



    void coot_fortran_prefix(coot_clarfg)(const blas_int* n, void*   alpha, void*   x, const blas_int* incx, void*   tau)
      {
      coot_fortran_noprefix(coot_clarfg)(n, alpha, x, incx, tau);
      }



    void coot_fortran_prefix(coot_zlarfg)(const blas_int* n, void*   alpha, void*   x, const blas_int* incx, void*   tau)
      {
      coot_fortran_noprefix(coot_zlarfg)(n, alpha, x, incx, tau);
      }



    //
    // reduce a general matrix to bidiagonal form (no hidden arguments)
    //



    void coot_fortran_prefix(coot_sgebrd)(const blas_int* m, const blas_int* n, float*  a, const blas_int* lda, float*  d, float*  e, float*  tauq, float*  taup, float*  work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgebrd)(m, n, a, lda, d, e, tauq, taup, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dgebrd)(const blas_int* m, const blas_int* n, double* a, const blas_int* lda, double* d, double* e, double* tauq, double* taup, double* work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgebrd)(m, n, a, lda, d, e, tauq, taup, work, lwork, info);
      }



    void coot_fortran_prefix(coot_cgebrd)(const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, void*   d, void*   e, void*   tauq, void*   taup, void*   work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgebrd)(m, n, a, lda, d, e, tauq, taup, work, lwork, info);
      }



    void coot_fortran_prefix(coot_zgebrd)(const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, void*   d, void*   e, void*   tauq, void*   taup, void*   work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgebrd)(m, n, a, lda, d, e, tauq, taup, work, lwork, info);
      }



    //
    // overwrite matrix with geqrf-generated orthogonal transformation
    //



    void coot_fortran_prefix(coot_sormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len)
      {
      coot_fortran_noprefix(coot_sormqr)(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info, side_len, trans_len);
      }



    void coot_fortran_prefix(coot_dormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len)
      {
      coot_fortran_noprefix(coot_dormqr)(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info, side_len, trans_len);
      }



    //
    // overwrite matrix with gelqf-generated orthogonal matrix
    //



    void coot_fortran_prefix(coot_sormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len)
      {
      coot_fortran_noprefix(coot_sormlq)(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info, side_len, trans_len);
      }



    void coot_fortran_prefix(coot_dormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len)
      {
      coot_fortran_noprefix(coot_dormlq)(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info, side_len, trans_len);
      }



    //
    // copy all or part of one 2d array to another
    //



    void coot_fortran_prefix(coot_slacpy)(const char* uplo, const blas_int* m, const blas_int* n, const float*  A, const blas_int* lda, float*  B, const blas_int* ldb, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_slacpy)(uplo, m, n, A, lda, B, ldb, uplo_len);
      }



    void coot_fortran_prefix(coot_dlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const double* A, const blas_int* lda, double* B, const blas_int* ldb, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_dlacpy)(uplo, m, n, A, lda, B, ldb, uplo_len);
      }



    void coot_fortran_prefix(coot_clacpy)(const char* uplo, const blas_int* m, const blas_int* n, const void*   A, const blas_int* lda, void*   B, const blas_int* ldb, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_clacpy)(uplo, m, n, A, lda, B, ldb, uplo_len);
      }



    void coot_fortran_prefix(coot_zlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const void*   A, const blas_int* lda, void*   B, const blas_int* ldb, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_zlacpy)(uplo, m, n, A, lda, B, ldb, uplo_len);
      }



    //
    // initialize a matrix with different elements on and off the diagonal
    //



    void coot_fortran_prefix(coot_slaset)(const char* uplo, const blas_int* m, const blas_int* n, const float*  alpha, const float*  beta, float*  A, const blas_int* lda, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_slaset)(uplo, m, n, alpha, beta, A, lda, uplo_len);
      }



    void coot_fortran_prefix(coot_dlaset)(const char* uplo, const blas_int* m, const blas_int* n, const double* alpha, const double* beta, double* A, const blas_int* lda, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_dlaset)(uplo, m, n, alpha, beta, A, lda, uplo_len);
      }



    void coot_fortran_prefix(coot_claset)(const char* uplo, const blas_int* m, const blas_int* n, const void*   alpha, const void*   beta, void*   A, const blas_int* lda, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_claset)(uplo, m, n, alpha, beta, A, lda, uplo_len);
      }



    void coot_fortran_prefix(coot_zlaset)(const char* uplo, const blas_int* m, const blas_int* n, const void*   alpha, const void*   beta, void*   A, const blas_int* lda, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_zlaset)(uplo, m, n, alpha, beta, A, lda, uplo_len);
      }



    //
    // apply block reflector to general rectangular matrix
    //



    void coot_fortran_prefix(coot_slarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const float*  V, const blas_int* ldv, const float*  T, const blas_int* ldt, float*  C, const blas_int* ldc, float*  work, const blas_int* ldwork, blas_len side_len, blas_len trans_len, blas_len direct_len, blas_len storev_len)
      {
      coot_fortran_noprefix(coot_slarfb)(side, trans, direct, storev, M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork, side_len, trans_len, direct_len, storev_len);
      }



    void coot_fortran_prefix(coot_dlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const double* V, const blas_int* ldv, const double* T, const blas_int* ldt, double* C, const blas_int* ldc, double* work, const blas_int* ldwork, blas_len side_len, blas_len trans_len, blas_len direct_len, blas_len storev_len)
      {
      coot_fortran_noprefix(coot_dlarfb)(side, trans, direct, storev, M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork, side_len, trans_len, direct_len, storev_len);
      }



    void coot_fortran_prefix(coot_clarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const void*   V, const blas_int* ldv, const void*   T, const blas_int* ldt, void*   C, const blas_int* ldc, void*   work, const blas_int* ldwork, blas_len side_len, blas_len trans_len, blas_len direct_len, blas_len storev_len)
      {
      coot_fortran_noprefix(coot_clarfb)(side, trans, direct, storev, M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork, side_len, trans_len, direct_len, storev_len);
      }



    void coot_fortran_prefix(coot_zlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const void*   V, const blas_int* ldv, const void*   T, const blas_int* ldt, void*   C, const blas_int* ldc, void*   work, const blas_int* ldwork, blas_len side_len, blas_len trans_len, blas_len direct_len, blas_len storev_len)
      {
      coot_fortran_noprefix(coot_zlarfb)(side, trans, direct, storev, M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork, side_len, trans_len, direct_len, storev_len);
      }



    //
    // get machine parameters
    //



    float  coot_fortran_prefix(coot_slamch)(const char* cmach, blas_len cmach_len)
      {
      return coot_fortran_noprefix(coot_slamch)(cmach, cmach_len);
      }



    double coot_fortran_prefix(coot_dlamch)(const char* cmach, blas_len cmach_len)
      {
      return coot_fortran_noprefix(coot_dlamch)(cmach, cmach_len);
      }



    //
    // scale matrix by a scalar
    //



    void coot_fortran_prefix(coot_slascl)(const char* type, const blas_int* kl, const blas_int* ku, const float*  cfrom, const float*  cto, const blas_int* m, const blas_int* n, float*  a, const blas_int* lda, blas_int* info, blas_len type_len)
      {
      coot_fortran_noprefix(coot_slascl)(type, kl, ku, cfrom, cto, m, n, a, lda, info, type_len);
      }



    void coot_fortran_prefix(coot_dlascl)(const char* type, const blas_int* kl, const blas_int* ku, const double* cfrom, const double* cto, const blas_int* m, const blas_int* n, double* a, const blas_int* lda, blas_int* info, blas_len type_len)
      {
      coot_fortran_noprefix(coot_dlascl)(type, kl, ku, cfrom, cto, m, n, a, lda, info, type_len);
      }



    void coot_fortran_prefix(coot_clascl)(const char* type, const blas_int* kl, const blas_int* ku, const void*   cfrom, const void*   cto, const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, blas_int* info, blas_len type_len)
      {
      coot_fortran_noprefix(coot_clascl)(type, kl, ku, cfrom, cto, m, n, a, lda, info, type_len);
      }



    void coot_fortran_prefix(coot_zlascl)(const char* type, const blas_int* kl, const blas_int* ku, const void*   cfrom, const void*   cto, const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, blas_int* info, blas_len type_len)
      {
      coot_fortran_noprefix(coot_zlascl)(type, kl, ku, cfrom, cto, m, n, a, lda, info, type_len);
      }



    //
    // compute singular values of bidiagonal matrix
    //



    void coot_fortran_prefix(coot_sbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, float*  vt, const blas_int* ldvt, float*  u, const blas_int* ldu, float*  c, const blas_int* ldc, float*  work, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_sbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work, info, uplo_len);
      }



    void coot_fortran_prefix(coot_dbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, double* vt, const blas_int* ldvt, double* u, const blas_int* ldu, double* c, const blas_int* ldc, double* work, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_dbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work, info, uplo_len);
      }



    void coot_fortran_prefix(coot_cbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, void*   vt, const blas_int* ldvt, void*   u, const blas_int* ldu, void*   c, const blas_int* ldc, float* work, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_cbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work, info, uplo_len);
      }



    void coot_fortran_prefix(coot_zbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, void*   vt, const blas_int* ldvt, void*   u, const blas_int* ldu, void*   c, const blas_int* ldc, double*  work, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_zbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work, info, uplo_len);
      }



    //
    // merges two sets of eigenvalues together into a single sorted set (no hidden arguments)
    //



    void coot_fortran_prefix(coot_slaed2)(blas_int* k, const blas_int* n, const blas_int* n1, float*  D, float*  Q, const blas_int* ldq, blas_int* indxq, float*  rho, const float*  Z, float*  dlamda, float*  W, float*  Q2, blas_int* indx, blas_int* indxc, blas_int* indxp, blas_int* coltyp, blas_int* info)
      {
      coot_fortran_noprefix(coot_slaed2)(k, n, n1, D, Q, ldq, indxq, rho, Z, dlamda, W, Q2, indx, indxc, indxp, coltyp, info);
      }



    void coot_fortran_prefix(coot_dlaed2)(blas_int* k, const blas_int* n, const blas_int* n1, double* D, double* Q, const blas_int* ldq, blas_int* indxq, double* rho, const double* Z, double* dlamda, double* W, double* Q2, blas_int* indx, blas_int* indxc, blas_int* indxp, blas_int* coltyp, blas_int* info)
      {
      coot_fortran_noprefix(coot_dlaed2)(k, n, n1, D, Q, ldq, indxq, rho, Z, dlamda, W, Q2, indx, indxc, indxp, coltyp, info);
      }



    //
    // compute all eigenvalues (and optionally eigenvectors) of symmetric tridiagonal matrix
    //



    void coot_fortran_prefix(coot_ssteqr)(const char* compz, const blas_int* n, float*  D, float*  E, float*  Z, const blas_int* ldz, float*  work, blas_int* info, blas_len compz_len)
      {
      coot_fortran_noprefix(coot_ssteqr)(compz, n, D, E, Z, ldz, work, info, compz_len);
      }



    void coot_fortran_prefix(coot_dsteqr)(const char* compz, const blas_int* n, double* D, double* E, double* Z, const blas_int* ldz, double* work, blas_int* info, blas_len compz_len)
      {
      coot_fortran_noprefix(coot_dsteqr)(compz, n, D, E, Z, ldz, work, info, compz_len);
      }



    void coot_fortran_prefix(coot_csteqr)(const char* compz, const blas_int* n, void*   D, void*   E, void*   Z, const blas_int* ldz, void*   work, blas_int* info, blas_len compz_len)
      {
      coot_fortran_noprefix(coot_csteqr)(compz, n, D, E, Z, ldz, work, info, compz_len);
      }



    void coot_fortran_prefix(coot_zsteqr)(const char* compz, const blas_int* n, void*   D, void*   E, void*   Z, const blas_int* ldz, void*   work, blas_int* info, blas_len compz_len)
      {
      coot_fortran_noprefix(coot_zsteqr)(compz, n, D, E, Z, ldz, work, info, compz_len);
      }



    //
    // compute 1-norm/Frobenius norm/inf norm of real symmetric tridiagonal matrix
    //



    float  coot_fortran_prefix(coot_slanst)(const char* norm, const blas_int* n, const float*  D, const float*  E, blas_len norm_len)
      {
      return coot_fortran_noprefix(coot_slanst)(norm, n, D, E, norm_len);
      }



    double coot_fortran_prefix(coot_dlanst)(const char* norm, const blas_int* n, const double* D, const double* E, blas_len norm_len)
      {
      return coot_fortran_noprefix(coot_dlanst)(norm, n, D, E, norm_len);
      }



    //
    // reduce real symmetric matrix to tridiagonal form
    //



    void coot_fortran_prefix(coot_ssytrd)(const char* uplo, const blas_int* n, float*  A, const blas_int* lda, float*  D, float*  E, float*  tau, float*  work, const blas_int* lwork, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_ssytrd)(uplo, n, A, lda, D, E, tau, work, lwork, info, uplo_len);
      }



    void coot_fortran_prefix(coot_dsytrd)(const char* uplo, const blas_int* n, double* A, const blas_int* lda, double* D, double* E, double* tau, double* work, const blas_int* lwork, blas_int* info, blas_len uplo_len)
      {
      coot_fortran_noprefix(coot_dsytrd)(uplo, n, A, lda, D, E, tau, work, lwork, info, uplo_len);
      }



    //
    // force A and B to be stored prior to doing the addition of A and B (no hidden arguments)
    //



    float  coot_fortran_prefix(coot_slamc3)(const float*  A, const float*  B)
      {
      return coot_fortran_noprefix(coot_slamc3)(A, B);
      }



    double coot_fortran_prefix(coot_dlamc3)(const double* A, const double* B)
      {
      return coot_fortran_noprefix(coot_dlamc3)(A, B);
      }



    //
    // compute the i'th updated eigenvalue of a symmetric rank-one modification to the diagonal matrix in d (no hidden arguments)
    //



    void coot_fortran_prefix(coot_slaed4)(const blas_int* n, const blas_int* i, const float*  D, const float*  Z, float*  delta, const float*  rho, float*  dlam, blas_int* info)
      {
      coot_fortran_noprefix(coot_slaed4)(n, i, D, Z, delta, rho, dlam, info);
      }



    void coot_fortran_prefix(coot_dlaed4)(const blas_int* n, const blas_int* i, const double* D, const double* Z, double* delta, const double* rho, double* dlam, blas_int* info)
      {
      coot_fortran_noprefix(coot_dlaed4)(n, i, D, Z, delta, rho, dlam, info);
      }



    //
    // create a permutation list to merge the elements of A into a single set (no hidden arguments)
    //



    void coot_fortran_prefix(coot_slamrg)(const blas_int* n1, const blas_int* n2, const float*  A, const blas_int* dtrd1, const blas_int* dtrd2, blas_int* index)
      {
      coot_fortran_noprefix(coot_slamrg)(n1, n2, A, dtrd1, dtrd2, index);
      }



    void coot_fortran_prefix(coot_dlamrg)(const blas_int* n1, const blas_int* n2, const double* A, const blas_int* dtrd1, const blas_int* dtrd2, blas_int* index)
      {
      coot_fortran_noprefix(coot_dlamrg)(n1, n2, A, dtrd1, dtrd2, index);
      }



    //
    // compute all eigenvalues of symmetric tridiagonal matrix (no hidden arguments)
    //



    void coot_fortran_prefix(coot_ssterf)(const blas_int* n, float*  D, float*  E, blas_int* info)
      {
      coot_fortran_noprefix(coot_ssterf)(n, D, E, info);
      }



    void coot_fortran_prefix(coot_dsterf)(const blas_int* n, double* D, double* E, blas_int* info)
      {
      coot_fortran_noprefix(coot_dsterf)(n, D, E, info);
      }



    //
    // perform a series of row interchanges (no hidden arguments)
    //

    void coot_fortran_prefix(coot_slaswp)(const blas_int* n, float*  A, const blas_int* lda, const blas_int* k1, const blas_int* k2, const blas_int* ipiv, const blas_int* incx)
      {
      coot_fortran_noprefix(coot_slaswp)(n, A, lda, k1, k2, ipiv, incx);
      }



    void coot_fortran_prefix(coot_dlaswp)(const blas_int* n, double* A, const blas_int* lda, const blas_int* k1, const blas_int* k2, const blas_int* ipiv, const blas_int* incx)
      {
      coot_fortran_noprefix(coot_dlaswp)(n, A, lda, k1, k2, ipiv, incx);
      }



    } // extern "C"
  } // namespace coot
