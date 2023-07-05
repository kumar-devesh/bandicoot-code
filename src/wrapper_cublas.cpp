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
#include "bandicoot_bits/include_opencl.hpp"
#include "bandicoot_bits/include_cuda.hpp"
#include "bandicoot_bits/typedef_elem.hpp"

#ifdef COOT_USE_CUDA

namespace coot
  {
  #include "bandicoot_bits/cuda/def_cublas.hpp"

  extern "C"
    {



    //
    // setup/teardown functions
    //



    cublasStatus_t wrapper_cublasCreate(cublasHandle_t* handle)
      {
      return cublasCreate(handle);
      }



    cublasStatus_t wrapper_cublasDestroy(cublasHandle_t handle)
      {
      return cublasDestroy(handle);
      }



    //
    // matrix-vector multiplications
    //



    cublasStatus_t wrapper_cublasSgemv(cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int m,
                                       int n,
                                       const float* alpha,
                                       const float* A,
                                       int lda,
                                       const float* x,
                                       int incx,
                                       const float* beta,
                                       float* y,
                                       int incy)
      {
      return cublasSgemv(handle,
                         trans,
                         m,
                         n,
                         alpha,
                         A,
                         lda,
                         x,
                         incx,
                         beta,
                         y,
                         incy);
      }



    cublasStatus_t wrapper_cublasDgemv(cublasHandle_t handle,
                                       cublasOperation_t trans,
                                       int m,
                                       int n,
                                       const double* alpha,
                                       const double* A,
                                       int lda,
                                       const double* x,
                                       int incx,
                                       const double* beta,
                                       double* y,
                                       int incy)
      {
      return cublasDgemv(handle,
                         trans,
                         m,
                         n,
                         alpha,
                         A,
                         lda,
                         x,
                         incx,
                         beta,
                         y,
                         incy);
      }



    //
    // matrix-matrix multiplications
    //



    cublasStatus_t wrapper_cublasSgemm(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m,
                                       int n,
                                       int k,
                                       const float* alpha,
                                       const float* A,
                                       int lda,
                                       const float* B,
                                       int ldb,
                                       const float* beta,
                                       float* C,
                                       int ldc)
      {
      return cublasSgemm(handle,
                         transa,
                         transb,
                         m,
                         n,
                         k,
                         alpha,
                         A,
                         lda,
                         B,
                         ldb,
                         beta,
                         C,
                         ldc);
      }



    cublasStatus_t wrapper_cublasDgemm(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m,
                                       int n,
                                       int k,
                                       const double* alpha,
                                       const double* A,
                                       int lda,
                                       const double* B,
                                       int ldb,
                                       const double* beta,
                                       double* C,
                                       int ldc)
      {
      return cublasDgemm(handle,
                         transa,
                         transb,
                         m,
                         n,
                         k,
                         alpha,
                         A,
                         lda,
                         B,
                         ldb,
                         beta,
                         C,
                         ldc);
      }



    //
    // matrix addition and transposition
    //



    cublasStatus_t wrapper_cublasSgeam(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m,
                                       int n,
                                       const float* alpha,
                                       const float* A,
                                       int lda,
                                       const float* beta,
                                       const float* B,
                                       int ldb,
                                       float* C,
                                       int ldc)
      {
      return cublasSgeam(handle,
                         transa,
                         transb,
                         m,
                         n,
                         alpha,
                         A,
                         lda,
                         beta,
                         B,
                         ldb,
                         C,
                         ldc);
      }



    cublasStatus_t wrapper_cublasDgeam(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m,
                                       int n,
                                       const double* alpha,
                                       const double* A,
                                       int lda,
                                       const double* beta,
                                       const double* B,
                                       int ldb,
                                       double* C,
                                       int ldc)
      {
      return cublasDgeam(handle,
                         transa,
                         transb,
                         m,
                         n,
                         alpha,
                         A,
                         lda,
                         beta,
                         B,
                         ldb,
                         C,
                         ldc);
      }



    //
    // compute Euclidean norm
    //



    cublasStatus_t wrapper_cublasSnrm2(cublasHandle_t handle,
                                       int n,
                                       const float* x,
                                       int incx,
                                       float* result)
      {
      return cublasSnrm2(handle, n, x, incx, result);
      }



    cublasStatus_t wrapper_cublasDnrm2(cublasHandle_t handle,
                                       int n,
                                       const double* x,
                                       int incx,
                                       double* result)
      {
      return cublasDnrm2(handle, n, x, incx, result);
      }



    //
    // dot product
    //



    cublasStatus_t wrapper_cublasSdot(cublasHandle_t handle,
                                      int n,
                                      const float* x,
                                      int incx,
                                      const float* y,
                                      int incy,
                                      float* result)
      {
      return cublasSdot(handle, n, x, incx, y, incy, result);
      }



    cublasStatus_t wrapper_cublasDdot(cublasHandle_t handle,
                                      int n,
                                      const double* x,
                                      int incx,
                                      const double* y,
                                      int incy,
                                      double* result)
      {
      return cublasDdot(handle, n, x, incx, y, incy, result);
      }



    } // extern "C"
  } // namespace coot

#endif
