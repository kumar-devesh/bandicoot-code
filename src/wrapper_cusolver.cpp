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
  #include "bandicoot_bits/cuda/def_cusolver.hpp"

  extern "C"
    {



    //
    // setup/teardown functions
    //



    cusolverStatus_t wrapper_cusolverDnCreate(cusolverDnHandle_t* handle)
      {
      return cusolverDnCreate(handle);
      }



    cusolverStatus_t wrapper_cusolverDnDestroy(cusolverDnHandle_t handle)
      {
      return cusolverDnDestroy(handle);
      }



    //
    // eigendecomposition
    //



    cusolverStatus_t wrapper_cusolverDnXsyevd_bufferSize(cusolverDnHandle_t handle,
                                                         cusolverDnParams_t params,
                                                         cusolverEigMode_t jobz,
                                                         cublasFillMode_t uplo,
                                                         int64_t n,
                                                         cudaDataType dataTypeA,
                                                         const void* A,
                                                         int64_t lda,
                                                         cudaDataType dataTypeW,
                                                         const void* W,
                                                         cudaDataType computeType,
                                                         size_t* workspaceInBytesOnDevice,
                                                         size_t* workspaceInBytesOnHost)
      {
      return cusolverDnXsyevd_bufferSize(handle,
                                         params,
                                         jobz,
                                         uplo,
                                         n,
                                         dataTypeA,
                                         A,
                                         lda,
                                         dataTypeW,
                                         W,
                                         computeType,
                                         workspaceInBytesOnDevice,
                                         workspaceInBytesOnHost);
      }



    cusolverStatus_t wrapper_cusolverDnXsyevd(cusolverDnHandle_t handle,
                                              cusolverDnParams_t params,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo,
                                              int64_t n,
                                              cudaDataType dataTypeA,
                                              void* A,
                                              int64_t lda,
                                              cudaDataType dataTypeW,
                                              void* W,
                                              cudaDataType computeType,
                                              void* bufferOnDevice,
                                              size_t workspaceInBytesOnDevice,
                                              void* bufferOnHost,
                                              size_t workspaceInBytesOnHost,
                                              int* info)
      {
      return cusolverDnXsyevd(handle,
                              params,
                              jobz,
                              uplo,
                              n,
                              dataTypeA,
                              A,
                              lda,
                              dataTypeW,
                              W,
                              computeType,
                              bufferOnDevice,
                              workspaceInBytesOnDevice,
                              bufferOnHost,
                              workspaceInBytesOnHost,
                              info);
      }



    //
    // cholesky decomposition
    //



    cusolverStatus_t wrapper_cusolverDnXpotrf_bufferSize(cusolverDnHandle_t handle,
                                                         cusolverDnParams_t params,
                                                         cublasFillMode_t uplo,
                                                         int64_t n,
                                                         cudaDataType dataTypeA,
                                                         const void* A,
                                                         int64_t lda,
                                                         cudaDataType computeType,
                                                         size_t* workspaceInBytesOnDevice,
                                                         size_t* workspaceInBytesOnHost)
      {
      return cusolverDnXpotrf_bufferSize(handle,
                                         params,
                                         uplo,
                                         n,
                                         dataTypeA,
                                         A,
                                         lda,
                                         computeType,
                                         workspaceInBytesOnDevice,
                                         workspaceInBytesOnHost);
      }



    cusolverStatus_t wrapper_cusolverDnXpotrf(cusolverDnHandle_t handle,
                                              cusolverDnParams_t params,
                                              cublasFillMode_t uplo,
                                              int64_t n,
                                              cudaDataType dataTypeA,
                                              void* A,
                                              int64_t lda,
                                              cudaDataType computeType,
                                              void* bufferOnDevice,
                                              size_t workspaceInBytesOnDevice,
                                              void* bufferOnHost,
                                              size_t workspaceInBytesOnHost,
                                              int* info)
      {
      return cusolverDnXpotrf(handle,
                              params,
                              uplo,
                              n,
                              dataTypeA,
                              A,
                              lda,
                              computeType,
                              bufferOnDevice,
                              workspaceInBytesOnDevice,
                              bufferOnHost,
                              workspaceInBytesOnHost,
                              info);
      }



    //
    // lu decomposition
    //



    cusolverStatus_t wrapper_cusolverDnXgetrf_bufferSize(cusolverDnHandle_t handle,
                                                         cusolverDnParams_t params,
                                                         int64_t m,
                                                         int64_t n,
                                                         cudaDataType dataTypeA,
                                                         const void* A,
                                                         int64_t lda,
                                                         cudaDataType computeType,
                                                         size_t* workspaceInBytesOnDevice,
                                                         size_t* workspaceInBytesOnHost)
      {
      return cusolverDnXgetrf_bufferSize(handle,
                                         params,
                                         m,
                                         n,
                                         dataTypeA,
                                         A,
                                         lda,
                                         computeType,
                                         workspaceInBytesOnDevice,
                                         workspaceInBytesOnHost);
      }



    cusolverStatus_t wrapper_cusolverDnXgetrf(cusolverDnHandle_t handle,
                                              cusolverDnParams_t params,
                                              int64_t m,
                                              int64_t n,
                                              cudaDataType dataTypeA,
                                              void* A,
                                              int64_t lda,
                                              int64_t* ipiv,
                                              cudaDataType computeType,
                                              void* bufferOnDevice,
                                              size_t workspaceInBytesOnDevice,
                                              void* bufferOnHost,
                                              size_t workspaceInBytesOnHost,
                                              int* info)
      {
      return cusolverDnXgetrf(handle,
                              params,
                              m,
                              n,
                              dataTypeA,
                              A,
                              lda,
                              ipiv,
                              computeType,
                              bufferOnDevice,
                              workspaceInBytesOnDevice,
                              bufferOnHost,
                              workspaceInBytesOnHost,
                              info);
      }



    //
    // singular value decomposition
    //



    cusolverStatus_t wrapper_cusolverDnXgesvd_bufferSize(cusolverDnHandle_t handle,
                                                         cusolverDnParams_t params,
                                                         signed char jobu,
                                                         signed char jobvt,
                                                         int64_t m,
                                                         int64_t n,
                                                         cudaDataType dataTypeA,
                                                         const void* A,
                                                         int64_t lda,
                                                         cudaDataType dataTypeS,
                                                         const void* S,
                                                         cudaDataType dataTypeU,
                                                         const void* U,
                                                         int64_t ldu,
                                                         cudaDataType dataTypeVT,
                                                         const void* VT,
                                                         int64_t ldvt,
                                                         cudaDataType computeType,
                                                         size_t* workspaceInBytesOnDevice,
                                                         size_t* workspaceInBytesOnHost)
      {
      return cusolverDnXgesvd_bufferSize(handle,
                                         params,
                                         jobu,
                                         jobvt,
                                         m,
                                         n,
                                         dataTypeA,
                                         A,
                                         lda,
                                         dataTypeS,
                                         S,
                                         dataTypeU,
                                         U,
                                         ldu,
                                         dataTypeVT,
                                         VT,
                                         ldvt,
                                         computeType,
                                         workspaceInBytesOnDevice,
                                         workspaceInBytesOnHost);
      }



    cusolverStatus_t wrapper_cusolverDnXgesvd(cusolverDnHandle_t handle,
                                              cusolverDnParams_t params,
                                              signed char jobu,
                                              signed char jobvt,
                                              int64_t m,
                                              int64_t n,
                                              cudaDataType dataTypeA,
                                              void* A,
                                              int64_t lda,
                                              cudaDataType dataTypeS,
                                              void* S,
                                              cudaDataType dataTypeU,
                                              void* U,
                                              int64_t ldu,
                                              cudaDataType dataTypeVT,
                                              void* VT,
                                              int64_t ldvt,
                                              cudaDataType computeType,
                                              void* bufferOnDevice,
                                              size_t workspaceInBytesOnDevice,
                                              void* bufferOnHost,
                                              size_t workspaceInBytesOnHost,
                                              int* info)
      {
      return cusolverDnXgesvd(handle,
                              params,
                              jobu,
                              jobvt,
                              m,
                              n,
                              dataTypeA,
                              A,
                              lda,
                              dataTypeS,
                              S,
                              dataTypeU,
                              U,
                              ldu,
                              dataTypeVT,
                              VT,
                              ldvt,
                              computeType,
                              bufferOnDevice,
                              workspaceInBytesOnDevice,
                              bufferOnHost,
                              workspaceInBytesOnHost,
                              info);
      }



    } // extern "C"
  } // namespace coot

#endif
