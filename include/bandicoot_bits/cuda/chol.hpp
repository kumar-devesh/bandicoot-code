// Copyright 2019 Ryan Curtin (http://www.ratml.org)
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


//! \addtogroup cuda
//! @{



/**
 * Compute the Cholesky decomposition using CUDA.
 */
template<typename eT>
inline
bool
chol(dev_mem_t<eT> mem, const uword n_rows)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::chol(): cuda runtime not valid");

  // The cuSolverDN library provides a potrf() implementation.  ...and its own entirely separate error code type.

  cusolverDnHandle_t handle = NULL;

  cusolverStatus_t status = cusolverDnCreate(&handle);
  coot_check_cusolver_error(status, "coot::cuda::chol(): cusolverDnCreate() failed");

  cudaError_t status2;

  // This is an additional error code for cusolverDn; but it is an error code on the device...
  // TODO: check it for additional better error output
  int* dev_info = NULL;
  cudaMalloc(&dev_info, sizeof(int));

  if (std::is_same<eT, float>::value)
    {
    int workspace_size = 0;
    status = cusolverDnSpotrf_bufferSize(handle,
                                         CUBLAS_FILL_MODE_UPPER,
                                         (int) n_rows,
                                         (float*) mem.cuda_mem_ptr,
                                         (int) n_rows,
                                         &workspace_size);
    coot_check_cusolver_error(status, "coot::cuda::chol(): couldn't calculate workspace size with cusolverDnSpotrf_bufferSize()");

    // Now allocate workspace for cuSolverDn.
    float* workspace_mem = NULL;
    status2 = cudaMalloc((void**) &workspace_mem, sizeof(float) * workspace_size);
    coot_check_cuda_error(status2, "coot::cuda::chol(): couldn't cudaMalloc() workspace memory");

    status = cusolverDnSpotrf(handle,
                              CUBLAS_FILL_MODE_UPPER,
                              (int) n_rows,
                              (float*) mem.cuda_mem_ptr,
                              (int) n_rows,
                              workspace_mem,
                              workspace_size,
                              dev_info);
    coot_check_cusolver_error(status, "coot::cuda::chol(): couldn't run cusolverDnSpotrf()");

    if (workspace_mem)
      {
      status2 = cudaFree(workspace_mem);
      coot_check_cuda_error(status2, "coot::cuda::chol(): couldn't cudaFree() workspace memory");
      }
    }
  else if (std::is_same<eT, double>::value)
    {
    int workspace_size = 0;
    status = cusolverDnDpotrf_bufferSize(handle,
                                         CUBLAS_FILL_MODE_UPPER,
                                         (int) n_rows,
                                         (double*) mem.cuda_mem_ptr,
                                         (int) n_rows,
                                         &workspace_size);
    coot_check_cusolver_error(status, "coot::cuda::chol(): couldn't calculate workspace size with cusolverDnDpotrf_bufferSize()");

    // Now allocate workspace for cuSolverDN.
    double* workspace_mem = NULL;
    status2 = cudaMalloc((void**) &workspace_mem, sizeof(double) * workspace_size);
    coot_check_cuda_error(status2, "coot::cuda::chol(): couldn't cudaMalloc() workspace memory");

    status = cusolverDnDpotrf(handle,
                              CUBLAS_FILL_MODE_UPPER,
                              (int) n_rows,
                              (double*) mem.cuda_mem_ptr,
                              (int) n_rows,
                              workspace_mem,
                              workspace_size,
                              dev_info);
    coot_check_cusolver_error(status, "coot::cuda::chol(): couldn't run cusolverDnSpotrf()");

    if (workspace_mem)
      {
      status2 = cudaFree(workspace_mem);
      coot_check_cuda_error(status2, "coot::cuda::chol(): couldn't cudaFree() workspace memory");
      }
    }
  else
    {
    // RC-TODO: better error
    throw std::runtime_error("coot::cuda::chol(): type not supported");
    }

  if (dev_info)
    {
    status2 = cudaFree(dev_info);
    coot_check_cuda_error(status2, "coot::cuda::chol(): couldn't cudaFree() auxiliary dev_info variable");
    }

  // Now we need to set the lower triangular part of the matrix to zeros.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::ltri_set_zero);

  const void* args[] = {
      &(mem.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_rows);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::chol(): cuLaunchKernel() failed for kernel ltri_set_zero");

  cusolverDnDestroy(handle);

  return true;
  }



//! @}
