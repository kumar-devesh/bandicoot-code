// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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
 * Sort the data in each column.
 */
template<typename eT>
inline
void
sort_colwise(dev_mem_t<eT> A, const uword n_rows, const uword n_cols, const uword sort_type)
  {
  coot_extra_debug_sigprint();

  // If the matrix is empty, don't do anything.
  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  // First, allocate a temporary matrix we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_rows * n_cols);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(sort_type == 0 ? oneway_kernel_id::radix_sort_colwise_ascending : oneway_kernel_id::radix_sort_colwise_descending);

  const void* args[] = {
      &(A.cuda_mem_ptr),
      &(tmp_mem.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = one_dimensional_grid_dims(n_cols);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::sort_colwise(): cuLaunchKernel() failed");

  get_rt().cuda_rt.synchronise();
  get_rt().cuda_rt.release_memory(tmp_mem.cuda_mem_ptr);
  }



/**
 * Sort the data in each row.
 */
template<typename eT>
inline
void
sort_rowwise(dev_mem_t<eT> A, const uword n_rows, const uword n_cols, const uword sort_type)
  {
  coot_extra_debug_sigprint();

  // If the matrix is empty, don't do anything.
  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  // First, allocate a temporary matrix we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_rows * n_cols);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(sort_type == 0 ? oneway_kernel_id::radix_sort_rowwise_ascending : oneway_kernel_id::radix_sort_rowwise_descending);

  const void* args[] = {
      &(A.cuda_mem_ptr),
      &(tmp_mem.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = one_dimensional_grid_dims(n_rows);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::sort_rowwise(): cuLaunchKernel() failed");

  get_rt().cuda_rt.synchronise();
  get_rt().cuda_rt.release_memory(tmp_mem.cuda_mem_ptr);
  }



/**
 * Sort the data in the block of memory.
 */
template<typename eT>
inline
void
sort_vec(dev_mem_t<eT> A, const uword n_elem, const uword sort_type)
  {
  coot_extra_debug_sigprint();

  // If the vector is empty, don't do anything.
  if (n_elem == 0)
    {
    return;
    }

  // The kernel requires that all threads are in one block.
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
  const size_t num_threads = std::min(mtpb, size_t(std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))))));
  // The number of threads needs to be a power of two.
  const size_t pow2_num_threads = std::min(mtpb, (size_t) std::pow(2.0f, std::ceil(std::log2((float) num_threads))));

  // First, allocate temporary memory we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_elem);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(sort_type == 0 ? oneway_kernel_id::radix_sort_ascending : oneway_kernel_id::radix_sort_descending);

  const void* args[] = {
      &(A.cuda_mem_ptr),
      &(tmp_mem.cuda_mem_ptr),
      (uword*) &n_elem };

  CUresult result = cuLaunchKernel(
      kernel,
      1, 1, 1, pow2_num_threads, 1, 1,
      2 * pow2_num_threads * sizeof(eT), // shared memory should have size equal to the number of threads times 2
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::sort(): cuLaunchKernel() failed");

  get_rt().cuda_rt.synchronise();
  get_rt().cuda_rt.release_memory(tmp_mem.cuda_mem_ptr);
  }
