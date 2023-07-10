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

#ifdef COOT_USE_OPENCL

namespace coot
  {
  #include "bandicoot_bits/opencl/def_clblas.hpp"

  // at this stage we have prototypes for clBLAS functions; so, now make the wrapper functions

  extern "C"
    {
    //
    // setup and teardown
    //



    clblasStatus wrapper_clblasSetup()
      {
      return clblasSetup();
      }



    void wrapper_clblasTeardown()
      {
      clblasTeardown();
      }



    //
    // matrix-vector multiplication
    //



    clblasStatus wrapper_clblasSgemv(clblasOrder order,
                                     clblasTranspose transA,
                                     size_t M,
                                     size_t N,
                                     cl_float alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     const cl_mem x,
                                     size_t offx,
                                     int incx,
                                     cl_float beta,
                                     cl_mem y,
                                     size_t offy,
                                     int incy,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasSgemv(order,
                         transA,
                         M,
                         N,
                         alpha,
                         A,
                         offA,
                         lda,
                         x,
                         offx,
                         incx,
                         beta,
                         y,
                         offy,
                         incy,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



    clblasStatus wrapper_clblasDgemv(clblasOrder order,
                                     clblasTranspose transA,
                                     size_t M,
                                     size_t N,
                                     cl_double alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     const cl_mem x,
                                     size_t offx,
                                     int incx,
                                     cl_double beta,
                                     cl_mem y,
                                     size_t offy,
                                     int incy,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasDgemv(order,
                         transA,
                         M,
                         N,
                         alpha,
                         A,
                         offA,
                         lda,
                         x,
                         offx,
                         incx,
                         beta,
                         y,
                         offy,
                         incy,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



  //
  // matrix-matrix multiplication
  //



  clblasStatus wrapper_clblasSgemm(clblasOrder order,
                                   clblasTranspose transA,
                                   clblasTranspose transB,
                                   size_t M,
                                   size_t N,
                                   size_t K,
                                   cl_float alpha,
                                   const cl_mem A,
                                   size_t offA,
                                   size_t lda,
                                   const cl_mem B,
                                   size_t offB,
                                   size_t ldb,
                                   cl_float beta,
                                   cl_mem C,
                                   size_t offC,
                                   size_t ldc,
                                   cl_uint numCommandQueues,
                                   cl_command_queue* commandQueues,
                                   cl_uint numEventsInWaitList,
                                   const cl_event* eventWaitList,
                                   cl_event* events)
    {
    return clblasSgemm(order,
                       transA,
                       transB,
                       M,
                       N,
                       K,
                       alpha,
                       A,
                       offA,
                       lda,
                       B,
                       offB,
                       ldb,
                       beta,
                       C,
                       offC,
                       ldc,
                       numCommandQueues,
                       commandQueues,
                       numEventsInWaitList,
                       eventWaitList,
                       events);
    }



  clblasStatus wrapper_clblasDgemm(clblasOrder order,
                                   clblasTranspose transA,
                                   clblasTranspose transB,
                                   size_t M,
                                   size_t N,
                                   size_t K,
                                   cl_double alpha,
                                   const cl_mem A,
                                   size_t offA,
                                   size_t lda,
                                   const cl_mem B,
                                   size_t offB,
                                   size_t ldb,
                                   cl_double beta,
                                   cl_mem C,
                                   size_t offC,
                                   size_t ldc,
                                   cl_uint numCommandQueues,
                                   cl_command_queue* commandQueues,
                                   cl_uint numEventsInWaitList,
                                   const cl_event* eventWaitList,
                                   cl_event* events)
    {
    return clblasDgemm(order,
                       transA,
                       transB,
                       M,
                       N,
                       K,
                       alpha,
                       A,
                       offA,
                       lda,
                       B,
                       offB,
                       ldb,
                       beta,
                       C,
                       offC,
                       ldc,
                       numCommandQueues,
                       commandQueues,
                       numEventsInWaitList,
                       eventWaitList,
                       events);
      }



    //
    // symmetric rank-k update
    //



    clblasStatus wrapper_clblasSsyrk(clblasOrder order,
                                     clblasUplo uplo,
                                     clblasTranspose transA,
                                     size_t N,
                                     size_t K,
                                     cl_float alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     cl_float beta,
                                     cl_mem C,
                                     size_t offC,
                                     size_t ldc,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasSsyrk(order,
                         uplo,
                         transA,
                         N,
                         K,
                         alpha,
                         A,
                         offA,
                         lda,
                         beta,
                         C,
                         offC,
                         ldc,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



    clblasStatus wrapper_clblasDsyrk(clblasOrder order,
                                     clblasUplo uplo,
                                     clblasTranspose transA,
                                     size_t N,
                                     size_t K,
                                     cl_double alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     cl_double beta,
                                     cl_mem C,
                                     size_t offC,
                                     size_t ldc,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasDsyrk(order,
                         uplo,
                         transA,
                         N,
                         K,
                         alpha,
                         A,
                         offA,
                         lda,
                         beta,
                         C,
                         offC,
                         ldc,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



    //
    // solve triangular systems of equations
    //



    clblasStatus wrapper_clblasStrsm(clblasOrder order,
                                     clblasSide side,
                                     clblasUplo uplo,
                                     clblasTranspose transA,
                                     clblasDiag diag,
                                     size_t M,
                                     size_t N,
                                     cl_float alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     cl_mem B,
                                     size_t offB,
                                     size_t ldb,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasStrsm(order,
                         side,
                         uplo,
                         transA,
                         diag,
                         M,
                         N,
                         alpha,
                         A,
                         offA,
                         lda,
                         B,
                         offB,
                         ldb,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



    clblasStatus wrapper_clblasDtrsm(clblasOrder order,
                                     clblasSide side,
                                     clblasUplo uplo,
                                     clblasTranspose transA,
                                     clblasDiag diag,
                                     size_t M,
                                     size_t N,
                                     cl_double alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     cl_mem B,
                                     size_t offB,
                                     size_t ldb,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasDtrsm(order,
                         side,
                         uplo,
                         transA,
                         diag,
                         M,
                         N,
                         alpha,
                         A,
                         offA,
                         lda,
                         B,
                         offB,
                         ldb,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



    //
    // triangular matrix-matrix multiplication
    //



    clblasStatus wrapper_clblasStrmm(clblasOrder order,
                                     clblasSide side,
                                     clblasUplo uplo,
                                     clblasTranspose transA,
                                     clblasDiag diag,
                                     size_t M,
                                     size_t N,
                                     cl_float alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     cl_mem B,
                                     size_t offB,
                                     size_t ldb,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasStrmm(order,
                         side,
                         uplo,
                         transA,
                         diag,
                         M,
                         N,
                         alpha,
                         A,
                         offA,
                         lda,
                         B,
                         offB,
                         ldb,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



    clblasStatus wrapper_clblasDtrmm(clblasOrder order,
                                     clblasSide side,
                                     clblasUplo uplo,
                                     clblasTranspose transA,
                                     clblasDiag diag,
                                     size_t M,
                                     size_t N,
                                     cl_double alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     cl_mem B,
                                     size_t offB,
                                     size_t ldb,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasDtrmm(order,
                         side,
                         uplo,
                         transA,
                         diag,
                         M,
                         N,
                         alpha,
                         A,
                         offA,
                         lda,
                         B,
                         offB,
                         ldb,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



    //
    // symmetric matrix-vector multiplication
    //



    clblasStatus wrapper_clblasSsymv(clblasOrder order,
                                     clblasUplo uplo,
                                     size_t N,
                                     cl_float alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     const cl_mem x,
                                     size_t offx,
                                     int incx,
                                     cl_float beta,
                                     cl_mem y,
                                     size_t offy,
                                     int incy,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasSsymv(order,
                         uplo,
                         N,
                         alpha,
                         A,
                         offA,
                         lda,
                         x,
                         offx,
                         incx,
                         beta,
                         y,
                         offy,
                         incy,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



    clblasStatus wrapper_clblasDsymv(clblasOrder order,
                                     clblasUplo uplo,
                                     size_t N,
                                     cl_double alpha,
                                     const cl_mem A,
                                     size_t offA,
                                     size_t lda,
                                     const cl_mem x,
                                     size_t offx,
                                     int incx,
                                     cl_double beta,
                                     cl_mem y,
                                     size_t offy,
                                     int incy,
                                     cl_uint numCommandQueues,
                                     cl_command_queue* commandQueues,
                                     cl_uint numEventsInWaitList,
                                     const cl_event* eventWaitList,
                                     cl_event* events)
      {
      return clblasDsymv(order,
                         uplo,
                         N,
                         alpha,
                         A,
                         offA,
                         lda,
                         x,
                         offx,
                         incx,
                         beta,
                         y,
                         offy,
                         incy,
                         numCommandQueues,
                         commandQueues,
                         numEventsInWaitList,
                         eventWaitList,
                         events);
      }



    //
    // symmetric rank-2k update to a matrix
    //



    clblasStatus wrapper_clblasSsyr2k(clblasOrder order,
                                      clblasUplo uplo,
                                      clblasTranspose transAB,
                                      size_t N,
                                      size_t K,
                                      cl_float alpha,
                                      const cl_mem A,
                                      size_t offA,
                                      size_t lda,
                                      const cl_mem B,
                                      size_t offB,
                                      size_t ldb,
                                      cl_float beta,
                                      cl_mem C,
                                      size_t offC,
                                      size_t ldc,
                                      cl_uint numCommandQueues,
                                      cl_command_queue* commandQueues,
                                      cl_uint numEventsInWaitList,
                                      const cl_event* eventWaitList,
                                      cl_event* events)
      {
      return clblasSsyr2k(order,
                          uplo,
                          transAB,
                          N,
                          K,
                          alpha,
                          A,
                          offA,
                          lda,
                          B,
                          offB,
                          ldb,
                          beta,
                          C,
                          offC,
                          ldc,
                          numCommandQueues,
                          commandQueues,
                          numEventsInWaitList,
                          eventWaitList,
                          events);
      }



    clblasStatus wrapper_clblasDsyr2k(clblasOrder order,
                                      clblasUplo uplo,
                                      clblasTranspose transAB,
                                      size_t N,
                                      size_t K,
                                      cl_double alpha,
                                      const cl_mem A,
                                      size_t offA,
                                      size_t lda,
                                      const cl_mem B,
                                      size_t offB,
                                      size_t ldb,
                                      cl_double beta,
                                      cl_mem C,
                                      size_t offC,
                                      size_t ldc,
                                      cl_uint numCommandQueues,
                                      cl_command_queue* commandQueues,
                                      cl_uint numEventsInWaitList,
                                      const cl_event* eventWaitList,
                                      cl_event* events)
      {
      return clblasDsyr2k(order,
                          uplo,
                          transAB,
                          N,
                          K,
                          alpha,
                          A,
                          offA,
                          lda,
                          B,
                          offB,
                          ldb,
                          beta,
                          C,
                          offC,
                          ldc,
                          numCommandQueues,
                          commandQueues,
                          numEventsInWaitList,
                          eventWaitList,
                          events);
      }



    //
    // index of max absolute value
    //



    clblasStatus wrapper_clblasiSamax(size_t N,
                                      cl_mem iMax,
                                      size_t offiMax,
                                      const cl_mem X,
                                      size_t offx,
                                      int incx,
                                      cl_mem scratchBuff,
                                      cl_uint numCommandQueues,
                                      cl_command_queue* commandQueues,
                                      cl_uint numEventsInWaitList,
                                      const cl_event* eventWaitList,
                                      cl_event* events)
      {
      return clblasiSamax(N,
                          iMax,
                          offiMax,
                          X,
                          offx,
                          incx,
                          scratchBuff,
                          numCommandQueues,
                          commandQueues,
                          numEventsInWaitList,
                          eventWaitList,
                          events);
      }



    clblasStatus wrapper_clblasiDamax(size_t N,
                                      cl_mem iMax,
                                      size_t offiMax,
                                      const cl_mem X,
                                      size_t offx,
                                      int incx,
                                      cl_mem scratchBuff,
                                      cl_uint numCommandQueues,
                                      cl_command_queue* commandQueues,
                                      cl_uint numEventsInWaitList,
                                      const cl_event* eventWaitList,
                                      cl_event* events)
      {
      return clblasiDamax(N,
                          iMax,
                          offiMax,
                          X,
                          offx,
                          incx,
                          scratchBuff,
                          numCommandQueues,
                          commandQueues,
                          numEventsInWaitList,
                          eventWaitList,
                          events);
      }

    } // extern C
  } // namespace coot

#endif
