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

#include "bandicoot_bits/compiler_setup.hpp"

#if defined(COOT_USE_OPENCL)
  #if defined(__APPLE__)
    #include <OpenCL/opencl.h>
  #else
    #include <CL/opencl.h>
  #endif


  // TODO: make this conditional on clBLAS being present
  #include <clBLAS.h>
#endif

// Import CUDA headers if needed.
#if defined(COOT_USE_CUDA)
  #include <cuda.h>
  #include <cuda_runtime_api.h>
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <nvrtc.h>
  #include <curand.h>
  #include <cusolverDn.h>
#endif

#include "bandicoot_bits/typedef_elem.hpp"

namespace coot
  {
  #include "bandicoot_bits/def_lapack.hpp"

  // at this stage we have prototypes for LAPACK functions; so, now make the wrapper functions

  extern "C"
    {



    //
    // LU factorisation
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



    void coot_fortran_prefix(coot_strtri)(const char* uplo, const char* diag, blas_int* n,  float* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_strtri)(uplo, diag, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_dtrtri)(const char* uplo, const char* diag, blas_int* n, double* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_dtrtri)(uplo, diag, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_ctrtri)(const char* uplo, const char* diag, blas_int* n,   void* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_ctrtri)(uplo, diag, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_ztrtri)(const char* uplo, const char* diag, blas_int* n,   void* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_ztrtri)(uplo, diag, n, a, lda, info);
      }



    //
    // eigen decomposition of symmetric real matrices
    //



    void coot_fortran_prefix(coot_ssyev)(const char* jobz, const char* uplo, blas_int* n,  float* a, blas_int* lda,  float* w,  float* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_ssyev)(jobz, uplo, n, a, lda, w, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dsyev)(const char* jobz, const char* uplo, blas_int* n, double* a, blas_int* lda, double* w, double* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dsyev)(jobz, uplo, n, a, lda, w, work, lwork, info);
      }



    //
    // eigen decomposition of hermitian matrices (complex)
    //



    void coot_fortran_prefix(coot_cheev)(const char* jobz, const char* uplo, blas_int* n,   void* a, blas_int* lda,  float* w,   void* work, blas_int* lwork,  float* rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cheev)(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
      }



    void coot_fortran_prefix(coot_zheev)(const char* jobz, const char* uplo, blas_int* n,   void* a, blas_int* lda, double* w,   void* work, blas_int* lwork, double* rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zheev)(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
      }



    //
    // eigen decomposition of symmetric real matrices by divide and conquer
    //



    void coot_fortran_prefix(coot_ssyevd)(const char* jobz, const char* uplo, blas_int* n,  float* a, blas_int* lda,  float* w,  float* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_ssyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
      }



    void coot_fortran_prefix(coot_dsyevd)(const char* jobz, const char* uplo, blas_int* n, double* a, blas_int* lda, double* w, double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dsyevd)(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
      }



    //
    // eigen decomposition of hermitian matrices (complex) by divide and conquer
    //



    void coot_fortran_prefix(coot_cheevd)(const char* jobz, const char* uplo, blas_int* n,   void* a, blas_int* lda,  float* w,   void* work, blas_int* lwork,  float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cheevd)(jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info);
      }



    void coot_fortran_prefix(coot_zheevd)(const char* jobz, const char* uplo, blas_int* n,   void* a, blas_int* lda, double* w,   void* work, blas_int* lwork, double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zheevd)(jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info);
      }



    //
    // eigen decomposition of general real matrix pair
    //



    void coot_fortran_prefix(coot_sggev)(const char* jobvl, const char* jobvr, blas_int* n,  float* a, blas_int* lda,  float* b, blas_int* ldb,  float* alphar,  float* alphai,  float* beta,  float* vl, blas_int* ldvl,  float* vr, blas_int* ldvr,  float* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sggev)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dggev)(const char* jobvl, const char* jobvr, blas_int* n, double* a, blas_int* lda, double* b, blas_int* ldb, double* alphar, double* alphai, double* beta, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, double* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dggev)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);
      }



    //
    // eigen decomposition of general complex matrix pair
    //



    void coot_fortran_prefix(coot_cggev)(const char* jobvl, const char* jobvr, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, void* alpha, void* beta, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork,  float* rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cggev)(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
      }



    void coot_fortran_prefix(coot_zggev)(const char* jobvl, const char* jobvr, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, void* alpha, void* beta, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork, double* rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zggev)(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
      }



    //
    // Cholesky decomposition
    //



    void coot_fortran_prefix(coot_spotrf)(const char* uplo, blas_int* n,  float* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_spotrf)(uplo, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_dpotrf)(const char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_dpotrf)(uplo, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_cpotrf)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_cpotrf)(uplo, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_zpotrf)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_zpotrf)(uplo, n, a, lda, info);
      }



    //
    // matrix inversion (using Cholesky decomposition result)
    //



    void coot_fortran_prefix(coot_spotri)(const char* uplo, blas_int* n,  float* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_spotri)(uplo, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_dpotri)(const char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_dpotri)(uplo, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_cpotri)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_cpotri)(uplo, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_zpotri)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_zpotri)(uplo, n, a, lda, info);
      }



    //
    // QR decomposition
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
    // Q matrix calculation from QR decomposition (real matrices)
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
    // Q matrix calculation from QR decomposition (complex matrices)
    //



    void coot_fortran_prefix(coot_cungqr)(blas_int* m, blas_int* n, blas_int* k,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cungqr)(m, n, k, a, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_zungqr)(blas_int* m, blas_int* n, blas_int* k,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zungqr)(m, n, k, a, lda, tau, work, lwork, info);
      }



    //
    // SVD (real matrices)
    //



    void coot_fortran_prefix(coot_sgesvd)(const char* jobu, const char* jobvt, blas_int* m, blas_int* n, float*  a, blas_int* lda, float*  s, float*  u, blas_int* ldu, float*  vt, blas_int* ldvt, float*  work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dgesvd)(const char* jobu, const char* jobvt, blas_int* m, blas_int* n, double* a, blas_int* lda, double* s, double* u, blas_int* ldu, double* vt, blas_int* ldvt, double* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
      }



    //
    // SVD (complex matrices)\
    //



    void coot_fortran_prefix(coot_cgesvd)(const char* jobu, const char* jobvt, blas_int* m, blas_int* n, void*   a, blas_int* lda, float*  s, void*   u, blas_int* ldu, void*   vt, blas_int* ldvt, void*   work, blas_int* lwork, float*  rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
      }



    void coot_fortran_prefix(coot_zgesvd)(const char* jobu, const char* jobvt, blas_int* m, blas_int* n, void*   a, blas_int* lda, double* s, void*   u, blas_int* ldu, void*   vt, blas_int* ldvt, void*   work, blas_int* lwork, double* rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgesvd)(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
      }



    //
    // SVD (real matrices) by divide and conquer
    //



    void coot_fortran_prefix(coot_sgesdd)(const char* jobz, blas_int* m, blas_int* n, float*  a, blas_int* lda, float*  s, float*  u, blas_int* ldu, float*  vt, blas_int* ldvt, float*  work, blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
      }



    void coot_fortran_prefix(coot_dgesdd)(const char* jobz, blas_int* m, blas_int* n, double* a, blas_int* lda, double* s, double* u, blas_int* ldu, double* vt, blas_int* ldvt, double* work, blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
      }



    //
    // SVD (complex matrices) by divide and conquer
    //



    void coot_fortran_prefix(coot_cgesdd)(const char* jobz, blas_int* m, blas_int* n, void* a, blas_int* lda, float*  s, void* u, blas_int* ldu, void* vt, blas_int* ldvt, void* work, blas_int* lwork, float*  rwork, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
      }



    void coot_fortran_prefix(coot_zgesdd)(const char* jobz, blas_int* m, blas_int* n, void* a, blas_int* lda, double* s, void* u, blas_int* ldu, void* vt, blas_int* ldvt, void* work, blas_int* lwork, double* rwork, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
      }



    //
    // solve system of linear equations (general square matrix)
    //



    void coot_fortran_prefix(coot_sgesv)(blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, blas_int* ipiv, float*  b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    void coot_fortran_prefix(coot_dgesv)(blas_int* n, blas_int* nrhs, double* a, blas_int* lda, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    void coot_fortran_prefix(coot_cgesv)(blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    void coot_fortran_prefix(coot_zgesv)(blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    //
    // solve system of linear equations (general square matrix, advanced form, real matrices)
    //



    void coot_fortran_prefix(coot_sgesvx)(const char* fact, const char* trans, blas_int* n, blas_int* nrhs,  float* a, blas_int* lda,  float* af, blas_int* ldaf, blas_int* ipiv, const char* equed,  float* r,  float* c,  float* b, blas_int* ldb,  float* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }



    void coot_fortran_prefix(coot_dgesvx)(const char* fact, const char* trans, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* af, blas_int* ldaf, blas_int* ipiv, const char* equed, double* r, double* c, double* b, blas_int* ldb, double* x, blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
      }



    //
    // solve system of linear equations (general square matrix, advanced form, complex matrices)
    //



    void coot_fortran_prefix(coot_cgesvx)(const char* fact, const char* trans, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* af, blas_int* ldaf, blas_int* ipiv, const char* equed,  float* r,  float* c, void* b, blas_int* ldb, void* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr, void* work,  float* rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }



    void coot_fortran_prefix(coot_zgesvx)(const char* fact, const char* trans, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* af, blas_int* ldaf, blas_int* ipiv, const char* equed, double* r, double* c, void* b, blas_int* ldb, void* x, blas_int* ldx, double* rcond, double* ferr, double* berr, void* work, double* rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgesvx)(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr, berr, work, rwork, info);
      }



    //
    // solve over/under-determined system of linear equations
    //



    void coot_fortran_prefix(coot_sgels)(const char* trans, blas_int* m, blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, float*  b, blas_int* ldb, float*  work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dgels)(const char* trans, blas_int* m, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* b, blas_int* ldb, double* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
      }



    void coot_fortran_prefix(coot_cgels)(const char* trans, blas_int* m, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, void*   b, blas_int* ldb, void*   work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
      }



    void coot_fortran_prefix(coot_zgels)(const char* trans, blas_int* m, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, void*   b, blas_int* ldb, void*   work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgels)(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
      }



    //
    // approximately solve system of linear equations using svd (real)
    //



    void coot_fortran_prefix(coot_sgelsd)(blas_int* m, blas_int* n, blas_int* nrhs,  float* a, blas_int* lda,  float* b, blas_int* ldb,  float* S,  float* rcond, blas_int* rank,  float* work, blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, iwork, info);
      }



    void coot_fortran_prefix(coot_dgelsd)(blas_int* m, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* b, blas_int* ldb, double* S, double* rcond, blas_int* rank, double* work, blas_int* lwork, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, iwork, info);
      }



    //
    // approximately solve system of linear equations using svd (complex)
    //



    void coot_fortran_prefix(coot_cgelsd)(blas_int* m, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* b, blas_int* ldb,  float* S,  float* rcond, blas_int* rank, void* work, blas_int* lwork,  float* rwork, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, rwork, iwork, info);
      }



    void coot_fortran_prefix(coot_zgelsd)(blas_int* m, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* b, blas_int* ldb, double* S, double* rcond, blas_int* rank, void* work, blas_int* lwork, double* rwork, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgelsd)(m, n, nrhs, a, lda, b, ldb, S, rcond, rank, work, lwork, rwork, iwork, info);
      }



    //
    // solve system of linear equations (triangular matrix)
    //



    void coot_fortran_prefix(coot_strtrs)(const char* uplo, const char* trans, const char* diag, blas_int* n, blas_int* nrhs, const float*  a, blas_int* lda, float*  b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_strtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
      }



    void coot_fortran_prefix(coot_dtrtrs)(const char* uplo, const char* trans, const char* diag, blas_int* n, blas_int* nrhs, const double* a, blas_int* lda, double* b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_dtrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
      }



    void coot_fortran_prefix(coot_ctrtrs)(const char* uplo, const char* trans, const char* diag, blas_int* n, blas_int* nrhs, const void*   a, blas_int* lda, void*   b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_ctrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
      }



    void coot_fortran_prefix(coot_ztrtrs)(const char* uplo, const char* trans, const char* diag, blas_int* n, blas_int* nrhs, const void*   a, blas_int* lda, void*   b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_ztrtrs)(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
      }



    //
    // Schur decomposition (real matrices)
    //



    void coot_fortran_prefix(coot_sgees)(const char* jobvs, const char* sort, void* select, blas_int* n, float*  a, blas_int* lda, blas_int* sdim, float*  wr, float*  wi, float*  vs, blas_int* ldvs, float*  work, blas_int* lwork, blas_int* bwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
      }



    void coot_fortran_prefix(coot_dgees)(const char* jobvs, const char* sort, void* select, blas_int* n, double* a, blas_int* lda, blas_int* sdim, double* wr, double* wi, double* vs, blas_int* ldvs, double* work, blas_int* lwork, blas_int* bwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
      }



    //
    // Schur decomposition (complex matrices)
    //



    void coot_fortran_prefix(coot_cgees)(const char* jobvs, const char* sort, void* select, blas_int* n, void* a, blas_int* lda, blas_int* sdim, void* w, void* vs, blas_int* ldvs, void* work, blas_int* lwork, float*  rwork, blas_int* bwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgees)(jobvs, sort, select, n, a, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info);
      }



    void coot_fortran_prefix(coot_zgees)(const char* jobvs, const char* sort, void* select, blas_int* n, void* a, blas_int* lda, blas_int* sdim, void* w, void* vs, blas_int* ldvs, void* work, blas_int* lwork, double* rwork, blas_int* bwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgees)(jobvs, sort, select, n, a, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info);
      }



    //
    // solve a Sylvester equation ax + xb = c, with a and b assumed to be in Schur form
    //



    void coot_fortran_prefix(coot_strsyl)(const char* transa, const char* transb, blas_int* isgn, blas_int* m, blas_int* n, const float*  a, blas_int* lda, const float*  b, blas_int* ldb, float*  c, blas_int* ldc, float*  scale, blas_int* info)
      {
      coot_fortran_noprefix(coot_strsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info);
      }



    void coot_fortran_prefix(coot_dtrsyl)(const char* transa, const char* transb, blas_int* isgn, blas_int* m, blas_int* n, const double* a, blas_int* lda, const double* b, blas_int* ldb, double* c, blas_int* ldc, double* scale, blas_int* info)
      {
      coot_fortran_noprefix(coot_dtrsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info);
      }



    void coot_fortran_prefix(coot_ctrsyl)(const char* transa, const char* transb, blas_int* isgn, blas_int* m, blas_int* n, const void*   a, blas_int* lda, const void*   b, blas_int* ldb, void*   c, blas_int* ldc, float*  scale, blas_int* info)
      {
      coot_fortran_noprefix(coot_ctrsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info);
      }



    void coot_fortran_prefix(coot_ztrsyl)(const char* transa, const char* transb, blas_int* isgn, blas_int* m, blas_int* n, const void*   a, blas_int* lda, const void*   b, blas_int* ldb, void*   c, blas_int* ldc, double* scale, blas_int* info)
      {
      coot_fortran_noprefix(coot_ztrsyl)(transa, transb, isgn, m, n, a, lda, b, ldb, c, ldc, scale, info);
      }




    void coot_fortran_prefix(coot_ssytrf)(const char* uplo, blas_int* n, float*  a, blas_int* lda, blas_int* ipiv, float*  work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_ssytrf)(uplo, n, a, lda, ipiv, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dsytrf)(const char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* ipiv, double* work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dsytrf)(uplo, n, a, lda, ipiv, work, lwork, info);
      }



    void coot_fortran_prefix(coot_csytrf)(const char* uplo, blas_int* n, void*   a, blas_int* lda, blas_int* ipiv, void*   work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_csytrf)(uplo, n, a, lda, ipiv, work, lwork, info);
      }



    void coot_fortran_prefix(coot_zsytrf)(const char* uplo, blas_int* n, void*   a, blas_int* lda, blas_int* ipiv, void*   work, blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zsytrf)(uplo, n, a, lda, ipiv, work, lwork, info);
      }



    void coot_fortran_prefix(coot_ssytri)(const char* uplo, blas_int* n, float*  a, blas_int* lda, blas_int* ipiv, float*  work, blas_int* info)
      {
      coot_fortran_noprefix(coot_ssytri)(uplo, n, a, lda, ipiv, work, info);
      }



    void coot_fortran_prefix(coot_dsytri)(const char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* ipiv, double* work, blas_int* info)
      {
      coot_fortran_noprefix(coot_dsytri)(uplo, n, a, lda, ipiv, work, info);
      }



    void coot_fortran_prefix(coot_csytri)(const char* uplo, blas_int* n, void*   a, blas_int* lda, blas_int* ipiv, void*   work, blas_int* info)
      {
      coot_fortran_noprefix(coot_csytri)(uplo, n, a, lda, ipiv, work, info);
      }



    void coot_fortran_prefix(coot_zsytri)(const char* uplo, blas_int* n, void*   a, blas_int* lda, blas_int* ipiv, void*   work, blas_int* info)
      {
      coot_fortran_noprefix(coot_zsytri)(uplo, n, a, lda, ipiv, work, info);
      }



    //
    // QZ decomposition (real matrices)
    //



    void coot_fortran_prefix(coot_sgges)(const char* jobvsl, const char* jobvsr, const char* sort, void* selctg, blas_int* n,  float* a, blas_int* lda,  float* b, blas_int* ldb, blas_int* sdim,  float* alphar,  float* alphai,  float* beta,  float* vsl, blas_int* ldvsl,  float* vsr, blas_int* ldvsr,  float* work, blas_int* lwork,  float* bwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork, info);
      }



    void coot_fortran_prefix(coot_dgges)(const char* jobvsl, const char* jobvsr, const char* sort, void* selctg, blas_int* n, double* a, blas_int* lda, double* b, blas_int* ldb, blas_int* sdim, double* alphar, double* alphai, double* beta, double* vsl, blas_int* ldvsl, double* vsr, blas_int* ldvsr, double* work, blas_int* lwork, double* bwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, bwork, info);
      }



    //
    // QZ decomposition (complex matrices)
    //



    void coot_fortran_prefix(coot_cgges)(const char* jobvsl, const char* jobvsr, const char* sort, void* selctg, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, blas_int* sdim, void* alpha, void* beta, void* vsl, blas_int* ldvsl, void* vsr, blas_int* ldvsr, void* work, blas_int* lwork,  float* rwork,  float* bwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alpha, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, rwork, bwork, info);
      }



    void coot_fortran_prefix(coot_zgges)(const char* jobvsl, const char* jobvsr, const char* sort, void* selctg, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, blas_int* sdim, void* alpha, void* beta, void* vsl, blas_int* ldvsl, void* vsr, blas_int* ldvsr, void* work, blas_int* lwork, double* rwork, double* bwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgges)(jobvsl, jobvsr, sort, selctg, n, a, lda, b, ldb, sdim, alpha, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, rwork, bwork, info);
      }



    //
    // 1-norm
    //



    float  coot_fortran_prefix(coot_slange)(const char* norm, blas_int* m, blas_int* n,  float* a, blas_int* lda,  float* work)
      {
      return coot_fortran_noprefix(coot_slange)(norm, m, n, a, lda, work);
      }



    double coot_fortran_prefix(coot_dlange)(const char* norm, blas_int* m, blas_int* n, double* a, blas_int* lda, double* work)
      {
      return coot_fortran_noprefix(coot_dlange)(norm, m, n, a, lda, work);
      }



    float  coot_fortran_prefix(coot_clange)(const char* norm, blas_int* m, blas_int* n,   void* a, blas_int* lda,  float* work)
      {
      return coot_fortran_noprefix(coot_clange)(norm, m, n, a, lda, work);
      }



    double coot_fortran_prefix(coot_zlange)(const char* norm, blas_int* m, blas_int* n,   void* a, blas_int* lda, double* work)
      {
      return coot_fortran_noprefix(coot_zlange)(norm, m, n, a, lda, work);
      }



    //
    // symmetric 1-norm
    //



    float  coot_fortran_prefix(coot_slansy)(const char* norm, const char* uplo, const blas_int* N, float*  A, const blas_int* lda, float*  work)
      {
      return coot_fortran_noprefix(coot_slansy)(norm, uplo, N, A, lda, work);
      }



    double coot_fortran_prefix(coot_dlansy)(const char* norm, const char* uplo, const blas_int* N, double* A, const blas_int* lda, double* work)
      {
      return coot_fortran_noprefix(coot_dlansy)(norm, uplo, N, A, lda, work);
      }



    float  coot_fortran_prefix(coot_clansy)(const char* norm, const char* uplo, const blas_int* N, void*   A, const blas_int* lda, float*  work)
      {
      return coot_fortran_noprefix(coot_clansy)(norm, uplo, N, A, lda, work);
      }



    double coot_fortran_prefix(coot_zlansy)(const char* norm, const char* uplo, const blas_int* N, void*   A, const blas_int* lda, double* work)
      {
      return coot_fortran_noprefix(coot_zlansy)(norm, uplo, N, A, lda, work);
      }



    //
    // reciprocal of condition number (real)
    //



    void coot_fortran_prefix(coot_sgecon)(const char* norm, blas_int* n,  float* a, blas_int* lda,  float* anorm,  float* rcond,  float* work, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgecon)(norm, n, a, lda, anorm, rcond, work, iwork, info);
      }



    void coot_fortran_prefix(coot_dgecon)(const char* norm, blas_int* n, double* a, blas_int* lda, double* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgecon)(norm, n, a, lda, anorm, rcond, work, iwork, info);
      }



    //
    // reciprocal of condition number (complex)
    //



    void coot_fortran_prefix(coot_cgecon)(const char* norm, blas_int* n, void* a, blas_int* lda,  float* anorm,  float* rcond, void* work,  float* rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgecon)(norm, n, a, lda, anorm, rcond, work, rwork, info);
      }



    void coot_fortran_prefix(coot_zgecon)(const char* norm, blas_int* n, void* a, blas_int* lda, double* anorm, double* rcond, void* work, double* rwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgecon)(norm, n, a, lda, anorm, rcond, work, rwork, info);
      }



    //
    // obtain parameters according to the local configuration of lapack
    //



    blas_int coot_fortran_prefix(coot_ilaenv)(blas_int* ispec, const char* name, const char* opts, blas_int* n1, blas_int* n2, blas_int* n3, blas_int* n4)
      {
      return coot_fortran_noprefix(coot_ilaenv)(ispec, name, opts, n1, n2, n3, n4);
      }



    //
    // solve linear equations using LDL decomposition
    //



    void coot_fortran_prefix(coot_ssytrs)(const char* uplo, blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, blas_int* ipiv, float*  b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_ssytrs)(uplo, n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    void coot_fortran_prefix(coot_dsytrs)(const char* uplo, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_dsytrs)(uplo, n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    void coot_fortran_prefix(coot_csytrs)(const char* uplo, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_csytrs)(uplo, n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    void coot_fortran_prefix(coot_zsytrs)(const char* uplo, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_zsytrs)(uplo, n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    //
    // solve linear equations using LU decomposition
    //



    void coot_fortran_prefix(coot_sgetrs)(const char* trans, blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, blas_int* ipiv, float*  b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgetrs)(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    void coot_fortran_prefix(coot_dgetrs)(const char* trans, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgetrs)(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    void coot_fortran_prefix(coot_cgetrs)(const char* trans, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgetrs)(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    void coot_fortran_prefix(coot_zgetrs)(const char* trans, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgetrs)(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
      }



    //
    // calculate eigenvalues of an upper Hessenberg matrix
    //



    void coot_fortran_prefix(coot_slahqr)(blas_int* wantt, blas_int* wantz, blas_int* n, blas_int* ilo, blas_int* ihi, float*  h, blas_int* ldh, float*  wr, float*  wi, blas_int* iloz, blas_int* ihiz, float*  z, blas_int* ldz, blas_int* info)
      {
      coot_fortran_noprefix(coot_slahqr)(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info);
      }



    void coot_fortran_prefix(coot_dlahqr)(blas_int* wantt, blas_int* wantz, blas_int* n, blas_int* ilo, blas_int* ihi, double* h, blas_int* ldh, double* wr, double* wi, blas_int* iloz, blas_int* ihiz, double* z, blas_int* ldz, blas_int* info)
      {
      coot_fortran_noprefix(coot_dlahqr)(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info);
      }



    //
    // calculate eigenvalues of a symmetric tridiagonal matrix
    //



    void coot_fortran_prefix(coot_sstedc)(const char* compz, blas_int* n, float*  d, float*  e, float*  z, blas_int* ldz, float*  work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sstedc)(compz, n, d, e, z, ldz, work, lwork, iwork, liwork, info);
      }



    void coot_fortran_prefix(coot_dstedc)(const char* compz, blas_int* n, double* d, double* e, double* z, blas_int* ldz, double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dstedc)(compz, n, d, e, z, ldz, work, lwork, iwork, liwork, info);
      }



    //
    // calculate eigenvectors of a Schur form matrix
    //



    void coot_fortran_prefix(coot_strevc)(const char* side, const char* howmny, blas_int* select, blas_int* n, float*  t, blas_int* ldt, float*  vl, blas_int* ldvl, float*  vr, blas_int* ldvr, blas_int* mm, blas_int* m, float*  work, blas_int* info)
      {
      coot_fortran_noprefix(coot_strevc)(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info);
      }



    void coot_fortran_prefix(coot_dtrevc)(const char* side, const char* howmny, blas_int* select, blas_int* n, double* t, blas_int* ldt, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, blas_int* mm, blas_int* m, double* work, blas_int* info)
      {
      coot_fortran_noprefix(coot_dtrevc)(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, m, work, info);
      }



    //
    // generate a vector of random numbers
    //



    void coot_fortran_prefix(coot_slarnv)(blas_int* idist, blas_int* iseed, blas_int* n, float*  x)
      {
      coot_fortran_noprefix(coot_slarnv)(idist, iseed, n, x);
      }



    void coot_fortran_prefix(coot_dlarnv)(blas_int* idist, blas_int* iseed, blas_int* n, double* x)
      {
      coot_fortran_noprefix(coot_dlarnv)(idist, iseed, n, x);
      }



    //
    // triangular factor of block reflector
    //



    void coot_fortran_prefix(coot_slarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, float*  v, blas_int* ldv, float*  tau, float*  t, blas_int* ldt)
      {
      coot_fortran_noprefix(coot_slarft)(direct, storev, n, k, v, ldv, tau, t, ldt);
      }



    void coot_fortran_prefix(coot_dlarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, double* v, blas_int* ldv, double* tau, double* t, blas_int* ldt)
      {
      coot_fortran_noprefix(coot_dlarft)(direct, storev, n, k, v, ldv, tau, t, ldt);
      }



    void coot_fortran_prefix(coot_clarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, void*   v, blas_int* ldv, void*   tau, void*   t, blas_int* ldt)
      {
      coot_fortran_noprefix(coot_clarft)(direct, storev, n, k, v, ldv, tau, t, ldt);
      }



    void coot_fortran_prefix(coot_zlarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, void*   v, blas_int* ldv, void*   tau, void*   t, blas_int* ldt)
      {
      coot_fortran_noprefix(coot_zlarft)(direct, storev, n, k, v, ldv, tau, t, ldt);
      }



    //
    // generate an elementary reflector
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
    // reduce a general matrix to bidiagonal form
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
    // generate Q or P**T determined by gebrd
    //



    void coot_fortran_prefix(coot_sorgbr)(const char* vect, const blas_int* m, const blas_int* n, const blas_int* k, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sorgbr)(vect, m, n, k, A, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dorgbr)(const char* vect, const blas_int* m, const blas_int* n, const blas_int* k, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dorgbr)(vect, m, n, k, A, lda, tau, work, lwork, info);
      }



    //
    // generate Q with orthonormal rows
    //



    void coot_fortran_prefix(coot_sorglq)(const blas_int* m, const blas_int* n, const blas_int* k, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sorglq)(m, n, k, A, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dorglq)(const blas_int* m, const blas_int* n, const blas_int* k, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dorglq)(m, n, k, A, lda, tau, work, lwork, info);
      }



    //
    // overwrite matrix with geqrf-generated orthogonal transformation
    //



    void coot_fortran_prefix(coot_sormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sormqr)(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dormqr)(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
      }



    //
    // overwrite matrix with gelqf-generated orthogonal matrix
    //



    void coot_fortran_prefix(coot_sormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sormlq)(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dormlq)(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
      }



    //
    // overwrite matrix with gebrd-generated orthogonal matrix products
    //



    void coot_fortran_prefix(coot_sormbr)(const char* vect, const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sormbr)(vect, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dormbr)(const char* vect, const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dormbr)(vect, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
      }



    //
    // copy all or part of one 2d array to another
    //



    void coot_fortran_prefix(coot_slacpy)(const char* uplo, const blas_int* m, const blas_int* n, const float*  A, const blas_int* lda, float*  B, const blas_int* ldb)
      {
      coot_fortran_noprefix(coot_slacpy)(uplo, m, n, A, lda, B, ldb);
      }



    void coot_fortran_prefix(coot_dlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const double* A, const blas_int* lda, double* B, const blas_int* ldb)
      {
      coot_fortran_noprefix(coot_dlacpy)(uplo, m, n, A, lda, B, ldb);
      }



    void coot_fortran_prefix(coot_clacpy)(const char* uplo, const blas_int* m, const blas_int* n, const void*   A, const blas_int* lda, void*   B, const blas_int* ldb)
      {
      coot_fortran_noprefix(coot_clacpy)(uplo, m, n, A, lda, B, ldb);
      }



    void coot_fortran_prefix(coot_zlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const void*   A, const blas_int* lda, void*   B, const blas_int* ldb)
      {
      coot_fortran_noprefix(coot_zlacpy)(uplo, m, n, A, lda, B, ldb);
      }



    //
    // initialize a matrix with different elements on and off the diagonal
    //



    void coot_fortran_prefix(coot_slaset)(const char* uplo, const blas_int* m, const blas_int* n, const float*  alpha, const float*  beta, float*  A, const blas_int* lda)
      {
      coot_fortran_noprefix(coot_slaset)(uplo, m, n, alpha, beta, A, lda);
      }



    void coot_fortran_prefix(coot_dlaset)(const char* uplo, const blas_int* m, const blas_int* n, const double* alpha, const double* beta, double* A, const blas_int* lda)
      {
      coot_fortran_noprefix(coot_dlaset)(uplo, m, n, alpha, beta, A, lda);
      }



    void coot_fortran_prefix(coot_claset)(const char* uplo, const blas_int* m, const blas_int* n, const void*   alpha, const void*   beta, void*   A, const blas_int* lda)
      {
      coot_fortran_noprefix(coot_claset)(uplo, m, n, alpha, beta, A, lda);
      }



    void coot_fortran_prefix(coot_zlaset)(const char* uplo, const blas_int* m, const blas_int* n, const void*   alpha, const void*   beta, void*   A, const blas_int* lda)
      {
      coot_fortran_noprefix(coot_zlaset)(uplo, m, n, alpha, beta, A, lda);
      }



    //
    // apply block reflector to general rectangular matrix
    //



    void coot_fortran_prefix(coot_slarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const float*  V, const blas_int* ldv, const float*  T, const blas_int* ldt, float*  C, const blas_int* ldc, float*  work, const blas_int* ldwork)
      {
      coot_fortran_noprefix(coot_slarfb)(side, trans, direct, storev, M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork);
      }



    void coot_fortran_prefix(coot_dlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const double* V, const blas_int* ldv, const double* T, const blas_int* ldt, double* C, const blas_int* ldc, double* work, const blas_int* ldwork)
      {
      coot_fortran_noprefix(coot_dlarfb)(side, trans, direct, storev, M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork);
      }



    void coot_fortran_prefix(coot_clarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const void*   V, const blas_int* ldv, const void*   T, const blas_int* ldt, void*   C, const blas_int* ldc, void*   work, const blas_int* ldwork)
      {
      coot_fortran_noprefix(coot_clarfb)(side, trans, direct, storev, M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork);
      }



    void coot_fortran_prefix(coot_zlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const void*   V, const blas_int* ldv, const void*   T, const blas_int* ldt, void*   C, const blas_int* ldc, void*   work, const blas_int* ldwork)
      {
      coot_fortran_noprefix(coot_zlarfb)(side, trans, direct, storev, M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork);
      }



    //
    // compute LQ factorization
    //



    void coot_fortran_prefix(coot_sgelqf)(const blas_int* M, const blas_int* N, float*  A, const blas_int* lda, float*  tau, float*  work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sgelqf)(M, N, A, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dgelqf)(const blas_int* M, const blas_int* N, double* A, const blas_int* lda, double* tau, double* work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dgelqf)(M, N, A, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_cgelqf)(const blas_int* M, const blas_int* N, void*   A, const blas_int* lda, void*   tau, void*   work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_cgelqf)(M, N, A, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_zgelqf)(const blas_int* M, const blas_int* N, void*   A, const blas_int* lda, void*   tau, void*   work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_zgelqf)(M, N, A, lda, tau, work, lwork, info);
      }



    //
    // get machine parameters
    //



    float  coot_fortran_prefix(coot_slamch)(const char* cmach)
      {
      return coot_fortran_noprefix(coot_slamch)(cmach);
      }



    double coot_fortran_prefix(coot_dlamch)(const char* cmach)
      {
      return coot_fortran_noprefix(coot_dlamch)(cmach);
      }



    //
    // scale matrix by a scalar
    //



    void coot_fortran_prefix(coot_slascl)(const char* type, const blas_int* kl, const blas_int* ku, const float*  cfrom, const float*  cto, const blas_int* m, const blas_int* n, float*  a, const blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_slascl)(type, kl, ku, cfrom, cto, m, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_dlascl)(const char* type, const blas_int* kl, const blas_int* ku, const double* cfrom, const double* cto, const blas_int* m, const blas_int* n, double* a, const blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_dlascl)(type, kl, ku, cfrom, cto, m, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_clascl)(const char* type, const blas_int* kl, const blas_int* ku, const void*   cfrom, const void*   cto, const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_clascl)(type, kl, ku, cfrom, cto, m, n, a, lda, info);
      }



    void coot_fortran_prefix(coot_zlascl)(const char* type, const blas_int* kl, const blas_int* ku, const void*   cfrom, const void*   cto, const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, blas_int* info)
      {
      coot_fortran_noprefix(coot_zlascl)(type, kl, ku, cfrom, cto, m, n, a, lda, info);
      }



    //
    // compute singular values of bidiagonal matrix
    //



    void coot_fortran_prefix(coot_sbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, float*  vt, const blas_int* ldvt, float*  u, const blas_int* ldu, float*  c, const blas_int* ldc, float*  work, blas_int* info)
      {
      coot_fortran_noprefix(coot_sbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work, info);
      }



    void coot_fortran_prefix(coot_dbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, double* vt, const blas_int* ldvt, double* u, const blas_int* ldu, double* c, const blas_int* ldc, double* work, blas_int* info)
      {
      coot_fortran_noprefix(coot_dbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work, info);
      }



    void coot_fortran_prefix(coot_cbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, void*   vt, const blas_int* ldvt, void*   u, const blas_int* ldu, void*   c, const blas_int* ldc, float* work, blas_int* info)
      {
      coot_fortran_noprefix(coot_cbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work, info);
      }



    void coot_fortran_prefix(coot_zbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, void*   vt, const blas_int* ldvt, void*   u, const blas_int* ldu, void*   c, const blas_int* ldc, double*  work, blas_int* info)
      {
      coot_fortran_noprefix(coot_zbdsqr)(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc, work, info);
      }



    //
    // solve matrix equations op(A)*X = aB or related equations
    //



    void coot_fortran_prefix(coot_strsm)(const char* side, const char* uplo, const char* transA, const char* diag, const blas_int* m, const blas_int* n, const float*  alpha, const float*  A, const blas_int* lda, float*  B, const blas_int* ldb)
      {
      coot_fortran_noprefix(coot_strsm)(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
      }



    void coot_fortran_prefix(coot_dtrsm)(const char* side, const char* uplo, const char* transA, const char* diag, const blas_int* m, const blas_int* n, const double* alpha, const double* A, const blas_int* lda, double* B, const blas_int* ldb)
      {
      coot_fortran_noprefix(coot_dtrsm)(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
      }



    void coot_fortran_prefix(coot_ctrsm)(const char* side, const char* uplo, const char* transA, const char* diag, const blas_int* m, const blas_int* n, const void*   alpha, const void*   A, const blas_int* lda, void*   B, const blas_int* ldb)
      {
      coot_fortran_noprefix(coot_ctrsm)(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
      }



    void coot_fortran_prefix(coot_ztrsm)(const char* side, const char* uplo, const char* transA, const char* diag, const blas_int* m, const blas_int* n, const void*   alpha, const void*   A, const blas_int* lda, void*   B, const blas_int* ldb)
      {
      coot_fortran_noprefix(coot_ztrsm)(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
      }



    //
    // compute A <-- alpha * x * y^T + A
    //



    void coot_fortran_prefix(coot_sger)(const blas_int* m, const blas_int* n, const float*  alpha, const float*  x, const blas_int* incx, const float*  y, const blas_int* incy, float*  A, const blas_int* lda)
      {
      coot_fortran_noprefix(coot_sger)(m, n, alpha, x, incx, y, incy, A, lda);
      }



    void coot_fortran_prefix(coot_dger)(const blas_int* m, const blas_int* n, const double* alpha, const double* x, const blas_int* incx, const double* y, const blas_int* incy, double* A, const blas_int* lda)
      {
      coot_fortran_noprefix(coot_dger)(m, n, alpha, x, incx, y, incy, A, lda);
      }



    //
    // merges two sets of eigenvalues together into a single sorted set
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



    void coot_fortran_prefix(coot_ssteqr)(const char* compz, const blas_int* n, float*  D, float*  E, float*  Z, const blas_int* ldz, float*  work, blas_int* info)
      {
      coot_fortran_noprefix(coot_ssteqr)(compz, n, D, E, Z, ldz, work, info);
      }



    void coot_fortran_prefix(coot_dsteqr)(const char* compz, const blas_int* n, double* D, double* E, double* Z, const blas_int* ldz, double* work, blas_int* info)
      {
      coot_fortran_noprefix(coot_dsteqr)(compz, n, D, E, Z, ldz, work, info);
      }



    void coot_fortran_prefix(coot_csteqr)(const char* compz, const blas_int* n, void*   D, void*   E, void*   Z, const blas_int* ldz, void*   work, blas_int* info)
      {
      coot_fortran_noprefix(coot_csteqr)(compz, n, D, E, Z, ldz, work, info);
      }



    void coot_fortran_prefix(coot_zsteqr)(const char* compz, const blas_int* n, void*   D, void*   E, void*   Z, const blas_int* ldz, void*   work, blas_int* info)
      {
      coot_fortran_noprefix(coot_zsteqr)(compz, n, D, E, Z, ldz, work, info);
      }



    //
    // compute 1-norm/Frobenius norm/inf norm of real symmetric tridiagonal matrix
    //



    float  coot_fortran_prefix(coot_slanst)(const char* norm, const blas_int* n, const float*  D, const float*  E)
      {
      return coot_fortran_noprefix(coot_slanst)(norm, n, D, E);
      }



    double coot_fortran_prefix(coot_dlanst)(const char* norm, const blas_int* n, const double* D, const double* E)
      {
      return coot_fortran_noprefix(coot_dlanst)(norm, n, D, E);
      }



    //
    // reduce real symmetric matrix to tridiagonal form
    //



    void coot_fortran_prefix(coot_ssytrd)(const char* uplo, const blas_int* n, float*  A, const blas_int* lda, float*  D, float*  E, float*  tau, float*  work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_ssytrd)(uplo, n, A, lda, D, E, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dsytrd)(const char* uplo, const blas_int* n, double* A, const blas_int* lda, double* D, double* E, double* tau, double* work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dsytrd)(uplo, n, A, lda, D, E, tau, work, lwork, info);
      }



    //
    // force A and B to be stored prior to doing the addition of A and B
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
    // compute the i'th updated eigenvalue of a symmetric rank-one modification to the diagonal matrix in d
    //



    void coot_fortran_prefix(coot_slaed4)(const blas_int* n, const blas_int* i, const float*  D, const float*  Z, float*  delta, const float*  rho, float*  dlam, const blas_int* info)
      {
      coot_fortran_noprefix(coot_slaed4)(n, i, D, Z, delta, rho, dlam, info);
      }



    void coot_fortran_prefix(coot_dlaed4)(const blas_int* n, const blas_int* i, const double* D, const double* Z, double* delta, const double* rho, double* dlam, const blas_int* info)
      {
      coot_fortran_noprefix(coot_dlaed4)(n, i, D, Z, delta, rho, dlam, info);
      }



    //
    // create a permutation list to merge the element of A into a single set
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
    // generate real orthogonal matrix as the product of dsytrd-generated elementary reflectors
    //



    void coot_fortran_prefix(coot_sorgtr)(const char* uplo, const blas_int* n, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_sorgtr)(uplo, n, A, lda, tau, work, lwork, info);
      }



    void coot_fortran_prefix(coot_dorgtr)(const char* uplo, const blas_int* n, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info)
      {
      coot_fortran_noprefix(coot_dorgtr)(uplo, n, A, lda, tau, work, lwork, info);
      }



    //
    // compute all eigenvalues of symmetric tridiagonal matrix
    //



    void coot_fortran_prefix(coot_ssterf)(const blas_int* n, float*  D, float*  E, blas_int* info)
      {
      coot_fortran_noprefix(coot_ssterf)(n, D, E, info);
      }



    void coot_fortran_prefix(coot_dsterf)(const blas_int* n, double* D, double* E, blas_int* info)
      {
      coot_fortran_noprefix(coot_dsterf)(n, D, E, info);
      }



    } // extern "C"
  } // namespace coot
