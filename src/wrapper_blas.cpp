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
  #include "bandicoot_bits/def_blas.hpp"

  // at this stage we have prototypes for BLAS functions; so, now make the wrapper functions

  extern "C"
    {



    //
    // matrix-matrix multiplication
    //



    void coot_fortran_prefix(coot_sgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const float*  alpha, const float*  A, const blas_int* ldA, const float*  B, const blas_int* ldB, const float*  beta, float*  C, const blas_int* ldC)
      {
      coot_fortran_noprefix(coot_sgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
      }



    void coot_fortran_prefix(coot_dgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* B, const blas_int* ldB, const double* beta, double* C, const blas_int* ldC)
      {
      coot_fortran_noprefix(coot_dgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
      }



    void coot_fortran_prefix(coot_cgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const void*   alpha, const void*   A, const blas_int* ldA, const void*   B, const blas_int* ldB, const void*   beta, void*   C, const blas_int* ldC)
      {
      coot_fortran_noprefix(coot_cgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
      }



    void coot_fortran_prefix(coot_zgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const void*   alpha, const void*   A, const blas_int* ldA, const void*   B, const blas_int* ldB, const void*   beta, void*   C, const blas_int* ldC)
      {
      coot_fortran_noprefix(coot_zgemm)(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
      }



    //
    // matrix-vector multiplication
    //



    void coot_fortran_prefix(coot_sgemv)(const char* transA, const blas_int* m, const blas_int* n, const float*  alpha, const float*  A, const blas_int* ldA, const float*  x, const blas_int* incx, const float*  beta, float*  y, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_sgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy);
      }



    void coot_fortran_prefix(coot_dgemv)(const char* transA, const blas_int* m, const blas_int* n, const double* alpha, const double* A, const blas_int* ldA, const double* x, const blas_int* incx, const double* beta, double* y, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_dgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy);
      }



    void coot_fortran_prefix(coot_cgemv)(const char* transA, const blas_int* m, const blas_int* n, const void*   alpha, const void*   A, const blas_int* ldA, const void*   x, const blas_int* incx, const void*   beta, void*   y, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_cgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy);
      }



    void coot_fortran_prefix(coot_zgemv)(const char* transA, const blas_int* m, const blas_int* n, const void*   alpha, const void*   A, const blas_int* ldA, const void*   x, const blas_int* incx, const void*   beta, void*   y, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_zgemv)(transA, m, n, alpha, A, ldA, x, incx, beta, y, incy);
      }



    //
    // scalar multiply + add
    //



    void coot_fortran_prefix(coot_saxpy)(const blas_int* m, const float*  da, const float*  dx, const blas_int* incx, float*  dy, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_saxpy)(m, da, dx, incx, dy, incy);
      }



    void coot_fortran_prefix(coot_daxpy)(const blas_int* m, const double* da, const double* dx, const blas_int* incx, double* dy, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_daxpy)(m, da, dx, incx, dy, incy);
      }



    void coot_fortran_prefix(coot_caxpy)(const blas_int* m, const void*   da, const void*   dx, const blas_int* incx, void*   dy, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_caxpy)(m, da, dx, incx, dy, incy);
      }



    void coot_fortran_prefix(coot_zaxpy)(const blas_int* m, const void*   da, const void*   dx, const blas_int* incx, void*   dy, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_zaxpy)(m, da, dx, incx, dy, incy);
      }



    //
    // scale vector by constant
    //



    void coot_fortran_prefix(coot_sscal)(const blas_int* n, const float*  da, float*  dx, const blas_int* incx)
      {
      coot_fortran_noprefix(coot_sscal)(n, da, dx, incx);
      }



    void coot_fortran_prefix(coot_dscal)(const blas_int* n, const double* da, double* dx, const blas_int* incx)
      {
      coot_fortran_noprefix(coot_dscal)(n, da, dx, incx);
      }



    void coot_fortran_prefix(coot_cscal)(const blas_int* n, const void*   da, void*   dx, const blas_int* incx)
      {
      coot_fortran_noprefix(coot_cscal)(n, da, dx, incx);
      }



    void coot_fortran_prefix(coot_zscal)(const blas_int* n, const void*   da, void*   dx, const blas_int* incx)
      {
      coot_fortran_noprefix(coot_zscal)(n, da, dx, incx);
      }



    //
    // symmetric rank-k a*A*A' + b*C
    //



    void coot_fortran_prefix(coot_ssyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const  float* alpha, const  float* A, const blas_int* ldA, const  float* beta,  float* C, const blas_int* ldC)
      {
      coot_fortran_noprefix(coot_ssyrk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC);
      }



    void coot_fortran_prefix(coot_dsyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* beta, double* C, const blas_int* ldC)
      {
      coot_fortran_noprefix(coot_dsyrk)(uplo, transA, n, k, alpha, A, ldA, beta, C, ldC);
      }



    //
    // copy a vector X to Y
    //



    void coot_fortran_prefix(coot_scopy)(const blas_int* n, const float*  X, const blas_int* incx, float*  Y, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_scopy)(n, X, incx, Y, incy);
      }



    void coot_fortran_prefix(coot_dcopy)(const blas_int* n, const double* X, const blas_int* incx, double* Y, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_dcopy)(n, X, incx, Y, incy);
      }



    void coot_fortran_prefix(coot_ccopy)(const blas_int* n, const void*   X, const blas_int* incx, void*   Y, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_ccopy)(n, X, incx, Y, incy);
      }



    void coot_fortran_prefix(coot_zcopy)(const blas_int* n, const void*   X, const blas_int* incx, void*   Y, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_zcopy)(n, X, incx, Y, incy);
      }



    //
    // interchange two vectors
    //



    void coot_fortran_prefix(coot_sswap)(const blas_int* n, float*  dx, const blas_int* incx, float*  dy, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_sswap)(n, dx, incx, dy, incy);
      }



    void coot_fortran_prefix(coot_dswap)(const blas_int* n, double* dx, const blas_int* incx, double* dy, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_dswap)(n, dx, incx, dy, incy);
      }



    void coot_fortran_prefix(coot_cswap)(const blas_int* n, void*   dx, const blas_int* incx, void*   dy, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_cswap)(n, dx, incx, dy, incy);
      }



    void coot_fortran_prefix(coot_zswap)(const blas_int* n, void*   dx, const blas_int* incx, void*   dy, const blas_int* incy)
      {
      coot_fortran_noprefix(coot_zswap)(n, dx, incx, dy, incy);
      }


    } // extern C
  } // namespace coot
