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
 * Accumulate all the elements in the device memory in a chunked fashion.
 */
template<typename eT>
inline
eT
accu_chunked(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  // work out number of chunks
  // make sure there are at least 4 elements per compunit

  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_cuda_error(result, "cuda::accu_chunked(): couldn't get device properties");

  uword n_chunks = dev_prop.multiProcessorCount;

  while(n_chunks >= 1)
    {
    if( (n_elem / n_chunks) >= uword(4) )  { break; }

    n_chunks /= uword(2);
    }

  n_chunks = (std::max)(uword(1), n_chunks);

  const uword chunk_size = n_elem / n_chunks;

  Mat<eT> tmp(n_chunks, 1);

  CUfunction k1 = get_rt().cuda_rt.get_kernel<eT>(kernel_id::accu_chunked);

  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  const void* args[] = {
      &(tmp_mem.cuda_mem_ptr),
      &(mem.cuda_mem_ptr),
      (size_t*) &chunk_size,
      (size_t*) &n_chunks };

  CUresult result2 = cuLaunchKernel(
      k1,
      std::ceil((double) n_chunks / (double) dev_prop.maxThreadsPerBlock), 1, 1, // grid dims
      dev_prop.maxThreadsPerBlock, 1, 1, // block dims
      0, NULL, // shared mem and stream
      (void**) args,
      0);

  coot_check_cuda_error(result2, "cuda::accu_chunked(): cuLaunchKernel() failed");

  CUfunction k2 = get_rt().cuda_rt.get_kernel<eT>(kernel_id::accu_twostage);

  const size_t A_start = n_chunks * chunk_size;

  const void* args2[] = {
      &(tmp_mem.cuda_mem_ptr),
      (size_t*) &tmp.n_elem,
      &(mem.cuda_mem_ptr),
      (size_t*) &A_start,
      (size_t*) &n_elem };

  result2 = cuLaunchKernel(
      k2,
      1, 1, 1, // grid dims
      1, 1, 1, // block dims
      0, NULL,
      (void**) args2,
      0);

  coot_check_cuda_error(result2, "cuda::accu_chunked(): cuLaunchKernel() failed");

  return tmp(0);

  }



/**
 * Accumulate all the elements in the device memory, but using a simple one-pass strategy.
 */
template<typename eT>
inline
eT
accu_simple(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  Mat<eT> tmp(1, 1);

  CUfunction k1 = get_rt().cuda_rt.get_kernel<eT>(kernel_id::accu_simple);

  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  const void* args[] = {
      &(tmp_mem.cuda_mem_ptr),
      &(mem.cuda_mem_ptr),
      (size_t*) &n_elem };

  CUresult result2 = cuLaunchKernel(
      k1,
      1, 1, 1, // grid dims
      1, 1, 1, // block dims
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result2, "cuda::accu_simple(): cuLaunchKernel() failed");

  return tmp(0);
  }



template<typename eT>
inline
eT
accu_subview(dev_mem_t<eT> mem, const uword m_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  // TODO: implement specialised handling for two cases: (i) n_cols = 1, (ii) n_rows = 1

  Mat<eT> tmp(1, n_cols);

  CUfunction k1 = get_rt().cuda_rt.get_kernel<eT>(kernel_id::submat_sum_colwise);
  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_cuda_error(result, "cuda::accu_chunked(): couldn't get device properties");

  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  const void* args[] = {
      &(tmp_mem.cuda_mem_ptr),
      &(mem.cuda_mem_ptr),
      (size_t*) &m_n_rows,
      (size_t*) &aux_row1,
      (size_t*) &aux_col1,
      (size_t*) &n_rows,
      (size_t*) &n_cols };

  CUresult result2 = cuLaunchKernel(
      k1,
      std::ceil((double) n_cols / (double) dev_prop.maxThreadsPerBlock), 1, 1, // grid dims
      dev_prop.maxThreadsPerBlock, 1, 1, // block dims
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result2, "cuda::accu_subview(): cuLaunchKernel() failed");

  // combine the column sums

  CUfunction k2 = get_rt().cuda_rt.get_kernel<eT>(kernel_id::accu_simple);

  const void* args2[] = {
      &(tmp_mem.cuda_mem_ptr),
      &(tmp_mem.cuda_mem_ptr),
      (size_t*) &n_cols };

  result2 = cuLaunchKernel(
      k2,
      1, 1, 1, // grid dims
      1, 1, 1, // block dims
      0, NULL,
      (void**) args2,
      0);

  coot_check_cuda_error(result2, "cuda::accu_subview(): cuLaunchKernel() failed");

  return tmp(0);
  }



//! @}
