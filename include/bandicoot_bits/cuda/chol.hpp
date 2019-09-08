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

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::chol(): cuda runtime not valid");

  // The cuSolverDN library provides a potrf() implementation.  ...and its own entirely separate error code type.

  cusolverDnHandle_t handle = NULL;

  cusolverStatus_t status = cusolverDnCreate(&handle);
  coot_check_cusolver_error(status, "cuda::chol(): cusolverDnCreate() failed");

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
    coot_check_cusolver_error(status, "cuda::chol(): couldn't calculate workspace size with cusolverDnSpotrf_bufferSize()");

    // Now allocate workspace for cuSolverDn.
    float* workspace_mem = NULL;
    status2 = cudaMalloc((void**) &workspace_mem, sizeof(float) * workspace_size);
    coot_check_cuda_error(status2, "cuda::chol(): couldn't cudaMalloc() workspace memory");

    status = cusolverDnSpotrf(handle,
                              CUBLAS_FILL_MODE_UPPER,
                              (int) n_rows,
                              (float*) mem.cuda_mem_ptr,
                              (int) n_rows,
                              workspace_mem,
                              workspace_size,
                              dev_info);
    coot_check_cusolver_error(status, "cuda::chol(): couldn't run cusolverDnSpotrf()");

    if (workspace_mem)
      {
      status2 = cudaFree(workspace_mem);
      coot_check_cuda_error(status2, "cuda::chol(): couldn't cudaFree() workspace memory");
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
    coot_check_cusolver_error(status, "cuda::chol(): couldn't calculate workspace size with cusolverDnDpotrf_bufferSize()");

    // Now allocate workspace for cuSolverDN.
    double* workspace_mem = NULL;
    status2 = cudaMalloc((void**) &workspace_mem, sizeof(double) * workspace_size);
    coot_check_cuda_error(status2, "cuda::chol(): couldn't cudaMalloc() workspace memory");

    status = cusolverDnDpotrf(handle,
                              CUBLAS_FILL_MODE_UPPER,
                              (int) n_rows,
                              (double*) mem.cuda_mem_ptr,
                              (int) n_rows,
                              workspace_mem,
                              workspace_size,
                              dev_info);
    coot_check_cusolver_error(status, "cuda::chol(): couldn't run cusolverDnSpotrf()");

    if (workspace_mem)
      {
      status2 = cudaFree(workspace_mem);
      coot_check_cuda_error(status2, "cuda::chol(): couldn't cudaFree() workspace memory");
      }
    }
  else
    {
    // RC-TODO: better error
    throw std::runtime_error("cuda::chol(): type not supported");
    }

  if (dev_info)
    {
    status2 = cudaFree(dev_info);
    coot_check_cuda_error(status2, "cuda::chol(): couldn't cudaFree() auxiliary dev_info variable");
    }

  // Now we need to set the lower triangular part of the matrix to zeros.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::ltri_set_zero);

  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_cuda_error(result, "cuda::chol(): couldn't get device properties");

  const void* args[] = {
      &(mem.cuda_mem_ptr),
      (size_t*) &n_rows,
      (size_t*) &n_rows };

  // grid dimensions:
  //   ideally, we want to use [n_rows, n_cols, 1]; but we have limits.  so   //   we might need to block it up a bit.  so, if n_rows * n_cols < maxThreadsPerBlock,
  //   we can use [n_rows, n_cols, 1]; otherwise, if n_rows < maxThreadsPerBlock,
  //      we can use [n_rows, maxThreadsPerBlock / n_rows, 1];
  //      and in this case we'll need a grid size of [1, ceil(n_cols / (mtpb / n_rows)), 1];
  //
  //   and if n_rows > mtpb,
  //      we can use [mtpb, 1, 1]
  //      and a grid size of [ceil(n_rows / mtpb), n_cols, 1].
  //
  // TODO: move this to some auxiliary code because it will surely be useful elsewhere
  const uword n_elem = n_rows * n_rows;
  size_t blockSize[2] = { n_rows, n_rows };
  size_t gridSize[2] = { 1, 1 };

  if (n_rows > dev_prop.maxThreadsPerBlock)
    {
    blockSize[0] = dev_prop.maxThreadsPerBlock;
    blockSize[1] = 1;

    gridSize[0] = std::ceil((double) n_rows / (double) dev_prop.maxThreadsPerBlock);
    gridSize[1] = n_rows;
    }
  else if (n_elem > dev_prop.maxThreadsPerBlock)
    {
    blockSize[0] = n_rows;
    blockSize[1] = std::floor((double) dev_prop.maxThreadsPerBlock / (double) n_rows);

    gridSize[1] = std::ceil((double) n_rows / (double) blockSize[1]);
    }

  CUresult result2 = cuLaunchKernel(
      kernel,
      gridSize[0], gridSize[1], 1, // grid dims
      blockSize[0], blockSize[1], 1, // block dims
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result2, "cuda::chol(): cuLaunchKernel() failed for kernel ltri_set_zero");

  cuCtxSynchronize();

  cusolverDnDestroy(handle);

  return true;
  }


//! @}
