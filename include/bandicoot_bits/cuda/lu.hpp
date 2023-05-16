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



/**
 * Compute the LU factorisation using OpenCL.
 */
template<typename eT>
inline
bool
lu(dev_mem_t<eT> L, dev_mem_t<eT> U, const bool pivoting, dev_mem_t<eT> P, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::lu(): CUDA runtime not valid");

  cusolverStatus_t status;
  cudaError_t status2;

  // This is an additional error code for cusolverDn; but it is an error code on the device...
  int* dev_info = NULL;
  status2 = cudaMalloc((void**) &dev_info, sizeof(int));
  coot_check_cuda_error(status2, "coot::cuda::lu(): couldn't cudaMalloc() device info holder");

  cudaDataType data_type;
  if (is_float<eT>::value)
    {
    data_type = CUDA_R_32F;
    }
  else if (is_double<eT>::value)
    {
    data_type = CUDA_R_64F;
    }
  else
    {
    coot_stop_runtime_error("coot::cuda::lu(): unknown data type, must be float or double");
    }

  size_t host_workspace_size = 0;
  size_t gpu_workspace_size = 0;
  status = cusolverDnXgetrf_bufferSize(get_rt().cuda_rt.cusolver_handle,
                                       NULL,
                                       n_rows,
                                       n_cols,
                                       data_type,
                                       U.cuda_mem_ptr,
                                       n_rows,
                                       data_type,
                                       &gpu_workspace_size,
                                       &host_workspace_size);
  coot_check_cusolver_error(status, "coot::cuda::lu(): couldn't compute workspace size with cusolverDnXgetrf_bufferSize()");

  s64* ipiv = NULL;
  const uword ipiv_size = std::min(n_rows, n_cols);
  if (pivoting)
    {
    // Allocate space for pivots.
    status2 = cudaMalloc((void**) &ipiv, sizeof(s64) * ipiv_size);
    coot_check_cuda_error(status2, "coot::cuda::lu(): couldn't cudaMalloc() pivot array");
    }

  void* gpu_workspace = NULL;
  status2 = cudaMalloc((void**) &gpu_workspace, gpu_workspace_size);
  coot_check_cuda_error(status2, "coot::cuda::lu(): couldn't cudaMalloc() GPU workspace memory");

  char* host_workspace = cpu_memory::acquire<char>(host_workspace_size);

  status = cusolverDnXgetrf(get_rt().cuda_rt.cusolver_handle,
                            NULL,
                            n_rows,
                            n_cols,
                            data_type,
                            U.cuda_mem_ptr,
                            n_rows,
                            ipiv,
                            data_type,
                            gpu_workspace,
                            gpu_workspace_size,
                            (void*) host_workspace,
                            host_workspace_size,
                            dev_info);
  coot_check_cusolver_error(status, "coot::cuda::lu(): cusolverDnXgetrf() failed");

  status2 = cudaFree(gpu_workspace);
  coot_check_cuda_error(status2, "coot::cuda::lu(): couldn't cudaFree() GPU workspace memory");
  cpu_memory::release(host_workspace);

  // Check whether the factorisation was successful.
  int info_result;
  status2 = cudaMemcpy(&info_result, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  coot_check_cuda_error(status2, "coot::cuda::lu(): couldn't copy device info holder to host");
  status2 = cudaFree(dev_info);
  coot_check_cuda_error(status2, "coot::cuda::lu(): couldn't cudaFree() device info holder");
  if (info_result < 0)
    {
    std::ostringstream oss;
    oss << "coot::cuda::lu(): parameter " << -info_result << " was incorrect in call to cusolverDnXgetrf()";
    coot_stop_runtime_error(oss.str());
    }
  else if (info_result > 0 && info_result < (int) std::max(n_rows, n_cols))
    {
    // Technically any positive info_result indicates a failed decomposition, but it looks like it randomly sometimes returns very large (invalid) numbers.
    // So... we ignore those.
    std::ostringstream oss;
    oss << "coot::cuda::lu(): decomposition failed, U(" << (info_result - 1) << ", " << (info_result - 1) << ") was found to be 0";
    coot_stop_runtime_error(oss.str());
    }

  // Now extract the lower triangular part (excluding diagonal).  This is done with a custom kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::lu_extract_l);

  const void* args[] = {
      &(L.cuda_mem_ptr),
      &(U.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols };

  const size_t max_rc = std::max(n_rows, n_cols);
  const kernel_dims dims = two_dimensional_grid_dims(n_rows, max_rc);

  CUresult status3 = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], dims.d[5],
      0, NULL, (void**) args, 0);

  coot_check_cuda_error(status3, "coot::cuda::lu(): cuLaunchKernel() failed for lu_extract_l kernel");

  // If pivoting was allowed, extract the permutation matrix.
  if (pivoting)
    {
    // First the pivoting needs to be "unwound" into a way where we can make P.
    uword* ipiv2 = cpu_memory::acquire<uword>(n_rows);
    for (uword i = 0; i < n_rows; ++i)
      {
      ipiv2[i] = i;
      }

    s64* ipiv_cpu = cpu_memory::acquire<s64>(ipiv_size);
    status2 = cudaMemcpy(ipiv_cpu, ipiv, ipiv_size * sizeof(s64), cudaMemcpyDeviceToHost);
    coot_check_cuda_error(status2, "coot::cuda::lu(): couldn't copy pivot array from GPU");

    for (uword i = 0; i < ipiv_size; ++i)
      {
      const int k = ipiv_cpu[i] - 1; // cusolverDnXgetrf() returns one-indexed results

      if (ipiv2[i] != ipiv2[k])
        {
        std::swap( ipiv2[i], ipiv2[k] );
        }
      }

    dev_mem_t<uword> ipiv_gpu;
    ipiv_gpu.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<uword>(n_rows);
    copy_into_dev_mem(ipiv_gpu, ipiv2, n_rows);
    cpu_memory::release(ipiv_cpu);
    cpu_memory::release(ipiv2);
    status2 = cudaFree(ipiv);
    coot_check_cuda_error(status2, "coot::cuda::lu(): couldn't cudaFree() pivot array");

    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::lu_extract_p);

    const void* args2[] = {
        &(P.cuda_mem_ptr),
        &(ipiv_gpu.cuda_mem_ptr),
        (uword*) &n_rows };

    const kernel_dims dims2 = one_dimensional_grid_dims(n_rows);

    status3 = cuLaunchKernel(
        kernel,
        dims2.d[0], dims2.d[1], dims2.d[2], dims2.d[3], dims2.d[4], dims2.d[5],
        0, NULL, (void**) args2, 0);
    coot_check_cuda_error(status3, "coot::cuda::lu(): cuLaunchKernel() failed for lu_extract_p kernel");
    }

  return true;
  }
