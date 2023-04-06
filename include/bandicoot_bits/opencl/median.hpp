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
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_rows * n_cols);

  runtime_t::cq_guard guard;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_colwise);

  cl_int status = 0;

  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);

  status |= clSetKernelArg(k, 0, sizeof(cl_mem), &(A.cl_mem_ptr));
  status |= clSetKernelArg(k, 1, sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= clSetKernelArg(k, 2, cl_n_rows.size, cl_n_rows.addr);
  status |= clSetKernelArg(k, 3, cl_n_cols.size, cl_n_cols.addr);

  coot_check_cl_error(status, "coot::opencl::radix_sort_colwise(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { n_cols };

  status = clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::radix_sort_colwise(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
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
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_rows * n_cols);

  runtime_t::cq_guard guard;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_rowwise);

  cl_int status = 0;

  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);

  status |= clSetKernelArg(k, 0, sizeof(cl_mem), &(A.cl_mem_ptr));
  status |= clSetKernelArg(k, 1, sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= clSetKernelArg(k, 2, cl_n_rows.size, cl_n_rows.addr);
  status |= clSetKernelArg(k, 3, cl_n_cols.size, cl_n_cols.addr);

  coot_check_cl_error(status, "coot::opencl::radix_sort_rowwise(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { n_rows };

  status = clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::radix_sort_rowwise(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
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

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::median(): OpenCL runtime not valid" );

  if (dim == 0)
    {
    // Sort the data in each column.
    radix_sort_colwise(in, n_rows, n_cols);
    const uword middle_element = (n_rows / 2);

    if (n_rows % 2 == 0)
      {
      // Even number of elements: we have to do a little extra processing.
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
      // Even number of elements: we have to do a little extra processing.
      sum_rowwise_subview(out, in, n_rows, 0, middle_element - 1, n_rows, 2, true);
      inplace_op_scalar(out, eT2(2), n_rows, oneway_kernel_id::inplace_div_scalar);
      }
    else
      {
      // Odd number of elements: the middle element is the result.
      // Now extract the column into the output.
      copy_subview(out, in, 0, middle_element, n_rows, n_cols, n_rows, 1);
      }
    }
  }



template<typename eT>
inline
void
radix_sort(dev_mem_t<eT> A, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort);

  size_t kernel_wg_size;
  cl_int status = clGetKernelWorkGroupInfo(k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "coot::opencl::radix_sort(): clGetKernelWorkGroupInfo() failed");

  const size_t total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  const size_t pow2_num_threads = std::min(kernel_wg_size, (size_t) std::pow(2.0f, std::ceil(std::log2((float) total_num_threads))));

  // First, allocate a temporary matrix we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_elem);

  status = 0;

  runtime_t::adapt_uword cl_n_elem(n_elem);

  status |= clSetKernelArg(k, 0, sizeof(cl_mem),                    &(A.cl_mem_ptr));
  status |= clSetKernelArg(k, 1, sizeof(cl_mem),                    &(tmp_mem.cl_mem_ptr));
  status |= clSetKernelArg(k, 2, cl_n_elem.size,                    cl_n_elem.addr);
  status |= clSetKernelArg(k, 3, 2 * sizeof(eT) * pow2_num_threads, NULL);

  coot_check_cl_error(status, "coot::opencl::radix_sort(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { pow2_num_threads };

  status = clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::radix_sort(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
  }



template<typename eT>
inline
eT
median_vec(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::median_vec(): OpenCL runtime not valid" );

  // Sort the data.
  radix_sort(mem, n_elem);
  // Now get the median element.
  const uword middle_element = n_elem / 2;
  if (n_elem % 2 == 0)
    {
    // Even number of elements: average the two middle elements.
    eT val1 = get_val(mem, middle_element - 1);
    eT val2 = get_val(mem, middle_element);
    return (val1 + val2) / 2;
    }
  else
    {
    // Odd number of elements: the easy case.
    return get_val(mem, middle_element);
    }
  }
