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

namespace coot
  {

#if !defined(COOT_BLAS_CAPITALS)

  #define coot_sbdt01 sbdt01
  #define coot_dbdt01 dbdt01

  #define coot_sort01 sort01
  #define coot_dort01 dort01

  #define coot_sgeqlf sgeqlf
  #define coot_dgeqlf dgeqlf

  #define coot_sormql sormql
  #define coot_dormql dormql

  #define coot_ssyt21 ssyt21
  #define coot_dsyt21 dsyt21

#else

  #define coot_sbdt01 SBDT01
  #define coot_dbdt01 DBDT01

  #define coot_sort01 SORT01
  #define coot_dort01 DORT01

  #define coot_sgeqlf SGEQLF
  #define coot_dgeqlf DGEQLF

  #define coot_sormql SORMQL
  #define coot_dormql DORMQL

  #define coot_ssyt21 SSYT21
  #define coot_dsyt21 DSYT21

#endif

extern "C"
  {
  // reconstruct from bidiagonal form
  void coot_fortran(coot_sbdt01)(const blas_int* m, const blas_int* n, const blas_int* kd, const float*  A, const blas_int* lda, const float*  Q, const blas_int* ldq, const float*  d, const float*  e, const float*  pt, const blas_int* ldpt, float*  work, float*  resid);
  void coot_fortran(coot_dbdt01)(const blas_int* m, const blas_int* n, const blas_int* kd, const double* A, const blas_int* lda, const double* Q, const blas_int* ldq, const double* d, const double* e, const double* pt, const blas_int* ldpt, double* work, double* resid);

  // check that matrix is orthogonal
  void coot_fortran(coot_sort01)(const char* rowcol, const blas_int* m, const blas_int* n, const float*  u, const blas_int* ldu, float*  work, const blas_int* lwork, float*  resid);
  void coot_fortran(coot_dort01)(const char* rowcol, const blas_int* m, const blas_int* n, const double* u, const blas_int* ldu, double* work, const blas_int* lwork, double* resid);

  // QL factorisation of real matrix
  void coot_fortran(coot_sgeqlf)(const blas_int* m, const blas_int* n, const float*  A, const blas_int* lda, const float*  tau, const float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dgeqlf)(const blas_int* m, const blas_int* n, const double* A, const blas_int* lda, const double* tau, const double* work, const blas_int* lwork, blas_int* info);

  // multiply matrix C by orthogonal matrix Q, which came from gelqf
  void coot_fortran(coot_sormql)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dormql)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info);

  // check a decomposition of the form U S U^T
  void coot_fortran(coot_ssyt21)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* kband, const float*  A, const blas_int* lda, const float*  D, const float*  E, const float*  U, const blas_int* ldu, const float*  V, const blas_int* ldv, const float*  tau, float*  work, float*  result);
  void coot_fortran(coot_dsyt21)(const blas_int* itype, const char* uplo, const blas_int* n, const blas_int* kband, const double* A, const blas_int* lda, const double* D, const double* E, const double* U, const blas_int* ldu, const double* V, const blas_int* ldv, const double* tau, double* work, double* result);

  }

  }; // namespace coot
