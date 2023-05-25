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

  #define coot_slaswp slaswp
  #define coot_dlaswp dlaswp

#else

  #define coot_sbdt01 SBDT01
  #define coot_dbdt01 DBDT01

  #define coot_sort01 SORT01
  #define coot_dort01 DORT01

  #define coot_slaswp SLASWP
  #define coot_dlaswp DLASWP

#endif

extern "C"
  {
  // reconstruct from bidiagonal form
  void coot_fortran(coot_sbdt01)(const blas_int* m, const blas_int* n, const blas_int* kd, const float*  A, const blas_int* lda, const float*  Q, const blas_int* ldq, const float*  d, const float*  e, const float*  pt, const blas_int* ldpt, float*  work, float*  resid);
  void coot_fortran(coot_dbdt01)(const blas_int* m, const blas_int* n, const blas_int* kd, const double* A, const blas_int* lda, const double* Q, const blas_int* ldq, const double* d, const double* e, const double* pt, const blas_int* ldpt, double* work, double* resid);

  // check that matrix is orthogonal
  void coot_fortran(coot_sort01)(const char* rowcol, const blas_int* m, const blas_int* n, const float*  u, const blas_int* ldu, float*  work, const blas_int* lwork, float*  resid);
  void coot_fortran(coot_dort01)(const char* rowcol, const blas_int* m, const blas_int* n, const double* u, const blas_int* ldu, double* work, const blas_int* lwork, double* resid);

  // perform a series of row interchanges
  void coot_fortran(coot_slaswp)(const blas_int* n, float*  A, const blas_int* lda, const blas_int* k1, const blas_int* k2, const blas_int* ipiv, const blas_int* incx);
  void coot_fortran(coot_dlaswp)(const blas_int* n, double* A, const blas_int* lda, const blas_int* k1, const blas_int* k2, const blas_int* ipiv, const blas_int* incx);
  }

  }; // namespace coot
