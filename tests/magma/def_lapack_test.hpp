// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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

// Definitions of test-specific functions from LAPACK.
// Note that the test program here expects COOT_USE_WRAPPER to be false.

#if defined(COOT_LAPACK_NOEXCEPT)
  #undef  COOT_NOEXCEPT
  #define COOT_NOEXCEPT noexcept
#else
  #undef  COOT_NOEXCEPT
  #define COOT_NOEXCEPT
#endif

namespace coot
  {

#if !defined(COOT_BLAS_CAPITALS)

  #define coot_sbdt01 sbdt01
  #define coot_dbdt01 dbdt01

  #define coot_sort01 sort01
  #define coot_dort01 dort01

  #define coot_slaswp slaswp
  #define coot_dlaswp dlaswp

  #define coot_sgeqlf sgeqlf
  #define coot_dgeqlf dgeqlf

  #define coot_sormql sormql
  #define coot_dormql dormql

  #define coot_ssyt21 ssyt21
  #define coot_dsyt21 dsyt21

  #define coot_ssyt22 ssyt22
  #define coot_dsyt22 dsyt22

  #define coot_sgesvd sgesvd
  #define coot_dgesvd dgesvd
  #define coot_cgesvd cgesvd
  #define coot_zgesvd zgesvd

  #define coot_slansy slansy
  #define coot_dlansy dlansy
  #define coot_clansy clansy
  #define coot_zlansy zlansy

  #define coot_sgetrs sgetrs
  #define coot_dgetrs dgetrs
  #define coot_cgetrs cgetrs
  #define coot_zgetrs zgetrs

  #define coot_slarnv slarnv
  #define coot_dlarnv dlarnv

  #define coot_sorgbr sorgbr
  #define coot_dorgbr dorgbr

  #define coot_sorglq sorglq
  #define coot_dorglq dorglq

  #define coot_sormbr sormbr
  #define coot_dormbr dormbr

  #define coot_sgelqf sgelqf
  #define coot_dgelqf dgelqf
  #define coot_cgelqf cgelqf
  #define coot_zgelqf zgelqf

  #define coot_sorgtr sorgtr
  #define coot_dorgtr dorgtr

#else

  #define coot_sbdt01 SBDT01
  #define coot_dbdt01 DBDT01

  #define coot_sort01 SORT01
  #define coot_dort01 DORT01

  #define coot_slaswp SLASWP
  #define coot_dlaswp DLASWP

  #define coot_sgeqlf SGEQLF
  #define coot_dgeqlf DGEQLF

  #define coot_sormql SORMQL
  #define coot_dormql DORMQL

  #define coot_ssyt21 SSYT21
  #define coot_dsyt21 DSYT21

  #define coot_ssyt22 SSYT22
  #define coot_dsyt22 DSYT22

  #define coot_sgesvd SGESVD
  #define coot_dgesvd DGESVD
  #define coot_cgesvd CGESVD
  #define coot_zgesvd ZGESVD

  #define coot_slansy SLANSY
  #define coot_dlansy DLANSY
  #define coot_clansy CLANSY
  #define coot_zlansy ZLANSY

  #define coot_sgetrs SGETRS
  #define coot_dgetrs DGETRS
  #define coot_cgetrs CGETRS
  #define coot_zgetrs ZGETRS

  #define coot_slarnv SLARNV
  #define coot_dlarnv DLARNV

  #define coot_sorgbr SORGBR
  #define coot_dorgbr DORGBR

  #define coot_sorglq SORGLQ
  #define coot_dorglq DORGLQ

  #define coot_sormbr SORMBR
  #define coot_dormbr DORMBR

  #define coot_sgelqf SGELQF
  #define coot_dgelqf DGELQF
  #define coot_cgelqf CGELQF
  #define coot_zgelqf ZGELQF

  #define coot_sorgtr SORGTR
  #define coot_dorgtr DORGTR

#endif

extern "C"
  {
  // reconstruct from bidiagonal form
  void coot_fortran(coot_sbdt01)(const blas_int* m, const blas_int* n, const blas_int* kd, const float*  A, const blas_int* lda, const float*  Q, const blas_int* ldq, const float*  d, const float*  e, const float*  pt, const blas_int* ldpt, float*  work, float*  resid) COOT_NOEXCEPT;
  void coot_fortran(coot_dbdt01)(const blas_int* m, const blas_int* n, const blas_int* kd, const double* A, const blas_int* lda, const double* Q, const blas_int* ldq, const double* d, const double* e, const double* pt, const blas_int* ldpt, double* work, double* resid) COOT_NOEXCEPT;

  // check that matrix is orthogonal
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sort01)(const char* rowcol, const blas_int* m, const blas_int* n, const float*  u, const blas_int* ldu, float*  work, const blas_int* lwork, float*  resid, blas_len rowcol_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dort01)(const char* rowcol, const blas_int* m, const blas_int* n, const double* u, const blas_int* ldu, double* work, const blas_int* lwork, double* resid, blas_len rowcol_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sort01)(const char* rowcol, const blas_int* m, const blas_int* n, const float*  u, const blas_int* ldu, float*  work, const blas_int* lwork, float*  resid) COOT_NOEXCEPT;
  void coot_fortran(coot_dort01)(const char* rowcol, const blas_int* m, const blas_int* n, const double* u, const blas_int* ldu, double* work, const blas_int* lwork, double* resid) COOT_NOEXCEPT;
  #endif

  // QL factorisation of real matrix
  void coot_fortran(coot_sgeqlf)(const blas_int* m, const blas_int* n, float*  A, const blas_int* lda, float*  tau, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dgeqlf)(const blas_int* m, const blas_int* n, double* A, const blas_int* lda, double* tau, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;

  // multiply matrix C by orthogonal matrix Q, which came from gelqf
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sormql)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dormql)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sormql)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dormql)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  #endif

  // check a decomposition of the form U S U^T
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_ssyt21)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* kband, const float*  A, const blas_int* lda, const float*  D, const float*  E, const float*  U, const blas_int* ldu, const float*  V, const blas_int* ldv, const float*  tau, float*  work, float*  result, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dsyt21)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* kband, const double* A, const blas_int* lda, const double* D, const double* E, const double* U, const blas_int* ldu, const double* V, const blas_int* ldv, const double* tau, double* work, double* result, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_ssyt21)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* kband, const float*  A, const blas_int* lda, const float*  D, const float*  E, const float*  U, const blas_int* ldu, const float*  V, const blas_int* ldv, const float*  tau, float*  work, float*  result) COOT_NOEXCEPT;
  void coot_fortran(coot_dsyt21)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* kband, const double* A, const blas_int* lda, const double* D, const double* E, const double* U, const blas_int* ldu, const double* V, const blas_int* ldv, const double* tau, double* work, double* result) COOT_NOEXCEPT;
  #endif

  // check a decomposition of the form A U = U S
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_ssyt22)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* m, const blas_int* kband, const float*  A, const blas_int* lda, const float*  D, const float*  E, const float*  U, const blas_int* ldu, const float*  V, const blas_int* ldv, const float*  tau, float*  work, float*  result, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dsyt22)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* m, const blas_int* kband, const double* A, const blas_int* lda, const double* D, const double* E, const double* U, const blas_int* ldu, const double* V, const blas_int* ldv, const double* tau, double* work, double* result, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_ssyt22)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* m, const blas_int* kband, const float*  A, const blas_int* lda, const float*  D, const float*  E, const float*  U, const blas_int* ldu, const float*  V, const blas_int* ldv, const float*  tau, float*  work, float*  result) COOT_NOEXCEPT;
  void coot_fortran(coot_dsyt22)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* m, const blas_int* kband, const double* A, const blas_int* lda, const double* D, const double* E, const double* U, const blas_int* ldu, const double* V, const blas_int* ldv, const double* tau, double* work, double* result) COOT_NOEXCEPT;
  #endif

  // SVD (real matrices)
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n,   float*  a, const blas_int* lda, float*  s,   float*  u, const blas_int* ldu,   float*  vt, const blas_int* ldvt,   float*  work, const blas_int* lwork,                blas_int* info, blas_len jobu_len, blas_len jobvt_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n,   double* a, const blas_int* lda, double* s,   double* u, const blas_int* ldu,   double* vt, const blas_int* ldvt,   double* work, const blas_int* lwork,                blas_int* info, blas_len jobu_len, blas_len jobvt_len) COOT_NOEXCEPT;
  void coot_fortran(coot_cgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, float*  s, blas_cxf* u, const blas_int* ldu, blas_cxf* vt, const blas_int* ldvt, blas_cxf* work, const blas_int* lwork, float*  rwork, blas_int* info, blas_len jobu_len, blas_len jobvt_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, double* s, blas_cxd* u, const blas_int* ldu, blas_cxd* vt, const blas_int* ldvt, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* info, blas_len jobu_len, blas_len jobvt_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n,   float*  a, const blas_int* lda, float*  s,   float*  u, const blas_int* ldu,   float*  vt, const blas_int* ldvt,   float*  work, const blas_int* lwork,                blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n,   double* a, const blas_int* lda, double* s,   double* u, const blas_int* ldu,   double* vt, const blas_int* ldvt,   double* work, const blas_int* lwork,                blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_cgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, float*  s, blas_cxf* u, const blas_int* ldu, blas_cxf* vt, const blas_int* ldvt, blas_cxf* work, const blas_int* lwork, float*  rwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zgesvd)(const char* jobu, const char* jobvt, const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, double* s, blas_cxd* u, const blas_int* ldu, blas_cxd* vt, const blas_int* ldvt, blas_cxd* work, const blas_int* lwork, double* rwork, blas_int* info) COOT_NOEXCEPT;
  #endif

  // symmetric 1-norm
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  float  coot_fortran(coot_slansy)(const char* norm, const char* uplo, const blas_int* N,   const float*  A, const blas_int* lda, float*  work, blas_len norm_len, blas_len uplo_len) COOT_NOEXCEPT;
  double coot_fortran(coot_dlansy)(const char* norm, const char* uplo, const blas_int* N,   const double* A, const blas_int* lda, double* work, blas_len norm_len, blas_len uplo_len) COOT_NOEXCEPT;
  float  coot_fortran(coot_clansy)(const char* norm, const char* uplo, const blas_int* N, const blas_cxf* A, const blas_int* lda, float*  work, blas_len norm_len, blas_len uplo_len) COOT_NOEXCEPT;
  double coot_fortran(coot_zlansy)(const char* norm, const char* uplo, const blas_int* N, const blas_cxd* A, const blas_int* lda, double* work, blas_len norm_len, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  float  coot_fortran(coot_slansy)(const char* norm, const char* uplo, const blas_int* N,   const float*  A, const blas_int* lda, float*  work) COOT_NOEXCEPT;
  double coot_fortran(coot_dlansy)(const char* norm, const char* uplo, const blas_int* N,  const double* A, const blas_int* lda, double* work) COOT_NOEXCEPT;
  float  coot_fortran(coot_clansy)(const char* norm, const char* uplo, const blas_int* N, const blas_cxf* A, const blas_int* lda, float*  work) COOT_NOEXCEPT;
  double coot_fortran(coot_zlansy)(const char* norm, const char* uplo, const blas_int* N, const blas_cxd* A, const blas_int* lda, double* work) COOT_NOEXCEPT;
  #endif

  // solve linear equations using LU decomposition
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const float*    a, const blas_int* lda, const blas_int* ipiv, float*    b, const blas_int* ldb, blas_int* info, blas_len trans_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const double*   a, const blas_int* lda, const blas_int* ipiv, double*   b, const blas_int* ldb, blas_int* info, blas_len trans_len) COOT_NOEXCEPT;
  void coot_fortran(coot_cgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const blas_cxf* a, const blas_int* lda, const blas_int* ipiv, blas_cxf* b, const blas_int* ldb, blas_int* info, blas_len trans_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const blas_cxd* a, const blas_int* lda, const blas_int* ipiv, blas_cxd* b, const blas_int* ldb, blas_int* info, blas_len trans_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const float*    a, const blas_int* lda, const blas_int* ipiv, float*    b, const blas_int* ldb, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const double*   a, const blas_int* lda, const blas_int* ipiv, double*   b, const blas_int* ldb, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_cgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const blas_cxf* a, const blas_int* lda, const blas_int* ipiv, blas_cxf* b, const blas_int* ldb, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zgetrs)(const char* trans, const blas_int* n, const blas_int* nrhs, const blas_cxd* a, const blas_int* lda, const blas_int* ipiv, blas_cxd* b, const blas_int* ldb, blas_int* info) COOT_NOEXCEPT;
  #endif

  // generate a vector of random numbers
  void coot_fortran(coot_slarnv)(const blas_int* idist, blas_int* iseed, const blas_int* n, float*  x) COOT_NOEXCEPT;
  void coot_fortran(coot_dlarnv)(const blas_int* idist, blas_int* iseed, const blas_int* n, double* x) COOT_NOEXCEPT;

  // generate Q or P**T determined by gebrd
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sorgbr)(const char* vect, const blas_int* m, const blas_int* n, const blas_int* k, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info, blas_len vect_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dorgbr)(const char* vect, const blas_int* m, const blas_int* n, const blas_int* k, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info, blas_len vect_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sorgbr)(const char* vect, const blas_int* m, const blas_int* n, const blas_int* k, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dorgbr)(const char* vect, const blas_int* m, const blas_int* n, const blas_int* k, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  #endif

  // generate Q with orthonormal rows
  void coot_fortran(coot_sorglq)(const blas_int* m, const blas_int* n, const blas_int* k, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dorglq)(const blas_int* m, const blas_int* n, const blas_int* k, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;

  // overwrite matrix with geqrf-generated orthogonal transformation
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  #endif

  // overwrite matrix with gebrd-generated orthogonal matrix products
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sormbr)(const char* vect, const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info, blas_len vect_len, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dormbr)(const char* vect, const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info, blas_len vect_len, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sormbr)(const char* vect, const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dormbr)(const char* vect, const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  #endif

  // compute LQ factorization
  void coot_fortran(coot_sgelqf)(const blas_int* M, const blas_int* N, float*    A, const blas_int* lda, float*    tau, float*    work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dgelqf)(const blas_int* M, const blas_int* N, double*   A, const blas_int* lda, double*   tau, double*   work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_cgelqf)(const blas_int* M, const blas_int* N, blas_cxf* A, const blas_int* lda, blas_cxf* tau, blas_cxf* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zgelqf)(const blas_int* M, const blas_int* N, blas_cxd* A, const blas_int* lda, blas_cxd* tau, blas_cxd* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;

  // generate real orthogonal matrix as the product of dsytrd-generated elementary reflectors
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sorgtr)(const char* uplo, const blas_int* n, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dorgtr)(const char* uplo, const blas_int* n, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sorgtr)(const char* uplo, const blas_int* n, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dorgtr)(const char* uplo, const blas_int* n, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  #endif
  }

  }; // namespace coot

#undef COOT_NOEXCEPT
