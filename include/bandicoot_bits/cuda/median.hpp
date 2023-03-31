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
radix_sort_colwise(dev_mem_t<eT> A, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  // First, allocate a temporary matrix we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_rows * n_cols);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::radix_sort_colwise);

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

  coot_check_cuda_error(result, "coot::cuda::radix_sort_colwise(): cuLaunchKernel() failed");

  get_rt().cuda_rt.synchronise();
  get_rt().cuda_rt.release_memory(tmp_mem.cuda_mem_ptr);
  }



/**
 * Sort the data in each row.
 */
template<typename eT>
inline
void
radix_sort_rowwise(dev_mem_t<eT> A, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  // First, allocate a temporary matrix we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_rows * n_cols);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::radix_sort_rowwise);

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

  coot_check_cuda_error(result, "coot::cuda::radix_sort_rowwise(): cuLaunchKernel() failed");

  get_rt().cuda_rt.synchronise();
  get_rt().cuda_rt.release_memory(tmp_mem.cuda_mem_ptr);
  }



/**
 * Compute the row-wise or column-wise mean of the input matrix, storing the result in the output matrix.
 */
template<typename eT2, typename eT1>
inline
void
median(dev_mem_t<eT2> out, dev_mem_t<eT1> in, const uword n_rows, const uword n_cols, const uword dim)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::median(): CUDA runtime not valid" );

  if (dim == 0)
    {
    // Sort the data in each column.
    radix_sort_colwise(in, n_rows, n_cols);
    const uword middle_element = (n_rows / 2);

    if (n_rows % 2 == 0)
      {
      // Even number of elements; we have to do a little extra processing.
      sum_colwise_subview(out, in, n_rows, middle_element - 1, 0, 2, n_cols, true);
      inplace_op_scalar(out, eT2(2), n_cols, oneway_kernel_id::inplace_div_scalar);
      }
    else
      {
      // Odd number of elements: the middle element is the result.
      // Now extract that row into the output.
      copy_subview(out, in, middle_element, 0, n_rows, n_cols, 1, n_cols);
      }
    }
  else
    {
    // Sort the data in each row.
    radix_sort_rowwise(in, n_rows, n_cols);
    const uword middle_element = (n_cols / 2);

    if (n_cols % 2 == 0)
      {
      // Even number of elements; we have to do a little extra processing.
      sum_rowwise_subview(out, in, n_rows, 0, middle_element - 1, n_rows, 2, true);
      inplace_op_scalar(out, eT2(2), n_rows, oneway_kernel_id::inplace_div_scalar);
      }
    else
      {
      // Odd number of elements: the middle element is the result.
      // Now extract that column into the output.
      copy_subview(out, in, 0, middle_element, n_rows, n_cols, n_rows, 1);
      }
    }
  }



/**
 * Sort the data in the block of memory.
 */
template<typename eT>
inline
void
radix_sort(dev_mem_t<eT> A, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  // The kernel requires that all threads are in one block.
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
  const size_t num_threads = std::min(mtpb, size_t(std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))))));
  // The number of threads needs to be a power of two.
  const size_t pow2_num_threads = std::min(mtpb, (size_t) std::pow(2.0f, std::ceil(std::log2((float) num_threads))));

  // First, allocate temporary memory we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_elem);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::radix_sort);

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

  coot_check_cuda_error(result, "coot::cuda::radix_sort(): cuLaunchKernel() failed");

  get_rt().cuda_rt.synchronise();
  get_rt().cuda_rt.release_memory(tmp_mem.cuda_mem_ptr);
  }



template<typename eT>
inline
eT
median_vec(dev_mem_t<eT> in, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::median(): CUDA runtime not valid" );

  // Sort the data.
  radix_sort(in, n_elem);
  // Now get the median element.
  const uword middle_element = n_elem / 2;
  if (n_elem % 2 == 0)
    {
    // Even number of elements: average the two middle elements.
    eT val1 = get_val(in, middle_element - 1);
    eT val2 = get_val(in, middle_element);
    return (val1 + val2) / 2;
    }
  else
    {
    // Odd number of elements: the easy case.
    return get_val(in, middle_element);
    }
  }
