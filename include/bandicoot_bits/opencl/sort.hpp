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
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_rows * n_cols);

  runtime_t::cq_guard guard;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(sort_type == 0 ? oneway_kernel_id::radix_sort_colwise_ascending : oneway_kernel_id::radix_sort_colwise_descending);

  cl_int status = 0;

  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);

  status |= coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem), &(A.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 1, sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 2, cl_n_rows.size, cl_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 3, cl_n_cols.size, cl_n_cols.addr);

  coot_check_cl_error(status, "coot::opencl::sort_colwise(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { n_cols };

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::sort_colwise(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
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
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_rows * n_cols);

  runtime_t::cq_guard guard;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(sort_type == 0 ? oneway_kernel_id::radix_sort_rowwise_ascending : oneway_kernel_id::radix_sort_rowwise_descending);

  cl_int status = 0;

  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);

  status |= coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem), &(A.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 1, sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 2, cl_n_rows.size, cl_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 3, cl_n_cols.size, cl_n_cols.addr);

  coot_check_cl_error(status, "coot::opencl::sort_rowwise(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { n_rows };

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::sort_rowwise(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
  }



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

  runtime_t::cq_guard guard;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(sort_type == 0 ? oneway_kernel_id::radix_sort_ascending : oneway_kernel_id::radix_sort_descending);

  size_t kernel_wg_size;
  cl_int status = coot_wrapper(clGetKernelWorkGroupInfo)(k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "coot::opencl::sort(): clGetKernelWorkGroupInfo() failed");

  const size_t total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  const size_t pow2_num_threads = std::min(kernel_wg_size, (size_t) std::pow(2.0f, std::ceil(std::log2((float) total_num_threads))));

  // First, allocate a temporary matrix we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_elem);

  status = 0;

  runtime_t::adapt_uword cl_n_elem(n_elem);

  status |= coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem),                    &(A.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 1, sizeof(cl_mem),                    &(tmp_mem.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 2, cl_n_elem.size,                    cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 3, 2 * sizeof(eT) * pow2_num_threads, NULL);

  coot_check_cl_error(status, "coot::opencl::sort(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { pow2_num_threads };

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::sort(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
  }
