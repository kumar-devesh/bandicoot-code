// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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



#if !defined(COOT_BLAS_CAPITALS)

  #define coot_sgemm sgemm
  #define coot_dgemm dgemm
  #define coot_cgemm cgemm
  #define coot_zgemm zgemm

  #define coot_sgemv sgemv
  #define coot_dgemv dgemv
  #define coot_cgemv cgemv
  #define coot_zgemv zgemv

  #define coot_saxpy saxpy
  #define coot_daxpy daxpy
  #define coot_caxpy caxpy
  #define coot_zaxpy zaxpy

  #define coot_sscal sscal
  #define coot_dscal dscal
  #define coot_cscal cscal
  #define coot_zscal zscal

  #define coot_ssyrk ssyrk
  #define coot_dsyrk dsyrk

  #define coot_scopy scopy
  #define coot_dcopy dcopy
  #define coot_ccopy ccopy
  #define coot_zcopy zcopy

  #define coot_sswap sswap
  #define coot_dswap dswap
  #define coot_cswap cswap
  #define coot_zswap zswap

#else

  #define coot_sgemm SGEMM
  #define coot_dgemm DGEMM
  #define coot_cgemm CGEMM
  #define coot_zgemm ZGEMM

  #define coot_sgemv SGEMV
  #define coot_dgemv DGEMV
  #define coot_cgemv CGEMV
  #define coot_zgemv ZGEMV

  #define coot_saxpy SAXPY
  #define coot_daxpy DAXPY
  #define coot_caxpy CAXPY
  #define coot_zaxpy ZAXPY

  #define coot_sscal SSCAL
  #define coot_dscal DSCAL
  #define coot_cscal CSCAL
  #define coot_zscal ZSCAL

  #define coot_ssyrk SSYRK
  #define coot_dsyrk DSYRK

  #define coot_scopy SCOPY
  #define coot_dcopy DCOPY
  #define coot_ccopy CCOPY
  #define coot_zcopy ZCOPY

  #define coot_sswap SSWAP
  #define coot_dswap DSWAP
  #define coot_cswap CSWAP
  #define coot_zswap ZSWAP

#endif



extern "C"
  {
  // matrix-matrix multiplication
  void coot_fortran(coot_sgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const float*  alpha, const float*  A, const blas_int* ldA, const float*  B, const blas_int* ldB, const float*  beta, float*  C, const blas_int* ldC);
  void coot_fortran(coot_dgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* B, const blas_int* ldB, const double* beta, double* C, const blas_int* ldC);
  void coot_fortran(coot_cgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const void*   alpha, const void*   A, const blas_int* ldA, const void*   B, const blas_int* ldB, const void*   beta, void*   C, const blas_int* ldC);
  void coot_fortran(coot_zgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const void*   alpha, const void*   A, const blas_int* ldA, const void*   B, const blas_int* ldB, const void*   beta, void*   C, const blas_int* ldC);

  // matrix-vector multiplication
  void coot_fortran(coot_sgemv)(const char* transA, const blas_int* m, const blas_int* n, const float*  alpha, const float*  A, const blas_int* ldA, const float*  x, const blas_int* incx, const float*  beta, float*  y, const blas_int* incy);
  void coot_fortran(coot_dgemv)(const char* transA, const blas_int* m, const blas_int* n, const double* alpha, const double* A, const blas_int* ldA, const double* x, const blas_int* incx, const double* beta, double* y, const blas_int* incy);
  void coot_fortran(coot_cgemv)(const char* transA, const blas_int* m, const blas_int* n, const void*   alpha, const void*   A, const blas_int* ldA, const void*   x, const blas_int* incx, const void*   beta, void*   y, const blas_int* incy);
  void coot_fortran(coot_zgemv)(const char* transA, const blas_int* m, const blas_int* n, const void*   alpha, const void*   A, const blas_int* ldA, const void*   x, const blas_int* incx, const void*   beta, void*   y, const blas_int* incy);

  // scalar multiply + add
  void coot_fortran(coot_saxpy)(const blas_int* m, const float*  da, const float*  dx, const blas_int* incx, float*  dy, const blas_int* incy);
  void coot_fortran(coot_daxpy)(const blas_int* m, const double* da, const double* dx, const blas_int* incx, double* dy, const blas_int* incy);
  void coot_fortran(coot_caxpy)(const blas_int* m, const void*   da, const void*   dx, const blas_int* incx, void*   dy, const blas_int* incy);
  void coot_fortran(coot_zaxpy)(const blas_int* m, const void*   da, const void*   dx, const blas_int* incx, void*   dy, const blas_int* incy);

  // scale vector by constant
  void coot_fortran(coot_sscal)(const blas_int* n, const float*  da, float*  dx, const blas_int* incx);
  void coot_fortran(coot_dscal)(const blas_int* n, const double* da, double* dx, const blas_int* incx);
  void coot_fortran(coot_cscal)(const blas_int* n, const void*   da, void*   dx, const blas_int* incx);
  void coot_fortran(coot_zscal)(const blas_int* n, const void*   da, void*   dx, const blas_int* incx);

  // symmetric rank-k a*A*A' + b*C
  void coot_fortran(coot_ssyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const  float* alpha, const  float* A, const blas_int* ldA, const  float* beta,  float* C, const blas_int* ldC);
  void coot_fortran(coot_dsyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* beta, double* C, const blas_int* ldC);

  // copy a vector X to Y
  void coot_fortran(coot_scopy)(const blas_int* n, const float*  X, const blas_int* incx, float*  Y, const blas_int* incy);
  void coot_fortran(coot_dcopy)(const blas_int* n, const double* X, const blas_int* incx, double* Y, const blas_int* incy);
  void coot_fortran(coot_ccopy)(const blas_int* n, const void*   X, const blas_int* incx, void*   Y, const blas_int* incy);
  void coot_fortran(coot_zcopy)(const blas_int* n, const void*   X, const blas_int* incx, void*   Y, const blas_int* incy);

  // interchange two vectors
  void coot_fortran(coot_sswap)(const blas_int* n, float*  dx, const blas_int* incx, float*  dy, const blas_int* incy);
  void coot_fortran(coot_dswap)(const blas_int* n, double* dx, const blas_int* incx, double* dy, const blas_int* incy);
  void coot_fortran(coot_cswap)(const blas_int* n, void*   dx, const blas_int* incx, void*   dy, const blas_int* incy);
  void coot_fortran(coot_zswap)(const blas_int* n, void*   dx, const blas_int* incx, void*   dy, const blas_int* incy);
  }
