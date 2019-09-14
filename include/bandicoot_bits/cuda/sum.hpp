// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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



template<typename eT>
inline
void
sum_colwise(dev_mem_t<eT> out, const dev_mem_t<eT> A, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::sum_colwise);

  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_cuda_error(result, "cuda::sum_colwise(): couldn't get device properties");

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(A.cuda_mem_ptr),
      (size_t*) &n_rows,
      (size_t*) &n_cols };

  CUresult result2 = cuLaunchKernel(
      kernel,
      std::ceil((double) n_cols / (double) dev_prop.maxThreadsPerBlock), 1, 1, // grid dims
      dev_prop.maxThreadsPerBlock, 1, 1, // block dims
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result2, "cuda::sum_colwise(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }



template<typename eT>
inline
void
sum_rowwise(dev_mem_t<eT> out, const dev_mem_t<eT> A, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::sum_rowwise);

  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_cuda_error(result, "cuda::sum_rowwise(): couldn't get device properties");

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(A.cuda_mem_ptr),
      (size_t*) &n_rows,
      (size_t*) &n_cols };

  CUresult result2 = cuLaunchKernel(
      kernel,
      std::ceil((double) n_rows / (double) dev_prop.maxThreadsPerBlock), 1, 1, // grid dims
      dev_prop.maxThreadsPerBlock, 1, 1, // block dims
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result2, "cuda::sum_rowwise(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }



template<typename eT>
inline
void
sum_colwise_subview(dev_mem_t<eT> out, const dev_mem_t<eT> A, const uword M_n_rows, const uword start_row, const uword start_col, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::submat_sum_colwise);

  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_cuda_error(result, "cuda::sum_colwise_subview(): couldn't get device properties");

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(A.cuda_mem_ptr),
      (size_t*) &M_n_rows,
      (size_t*) &start_row,
      (size_t*) &start_col,
      (size_t*) &n_rows,
      (size_t*) &n_cols };

  CUresult result2 = cuLaunchKernel(
      kernel,
      std::ceil((double) n_cols / (double) dev_prop.maxThreadsPerBlock), 1, 1, // grid dims
      dev_prop.maxThreadsPerBlock, 1, 1, // block dims
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result2, "cuda::sum_colwise_subview(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }



template<typename eT>
inline
void
sum_rowwise_subview(dev_mem_t<eT> out, const dev_mem_t<eT> A, const uword M_n_rows, const uword start_row, const uword start_col, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::submat_sum_rowwise);

  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_cuda_error(result, "cuda::sum_rowwise_subview(): couldn't get device properties");

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(A.cuda_mem_ptr),
      (size_t*) &M_n_rows,
      (size_t*) &start_row,
      (size_t*) &start_col,
      (size_t*) &n_rows,
      (size_t*) &n_cols };

  CUresult result2 = cuLaunchKernel(
      kernel,
      std::ceil((double) n_rows / (double) dev_prop.maxThreadsPerBlock), 1, 1, // grid dims
      dev_prop.maxThreadsPerBlock, 1, 1, // block dims
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result2, "cuda::sum_rowwise_subview(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }
