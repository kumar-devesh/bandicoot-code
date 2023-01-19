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
 * Accumulate all elements in `mem`.
 */
template<typename eT>
inline
eT
accu(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::accu(): OpenCL runtime not valid" );

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  return generic_reduce(k, k_small, "accu", mem, n_elem);
  }



template<typename eT>
inline
eT
accu_subview(dev_mem_t<eT> mem, const uword m_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::accu(): OpenCL runtime not valid" );

  // TODO: implement specialised handling for two cases: (i) n_cols = 1, (ii) n_rows = 1

  Mat<eT> tmp(1, n_cols);

  runtime_t::cq_guard guard;

  cl_kernel k1 = get_rt().cl_rt.get_kernel<eT, eT>(twoway_kernel_id::submat_sum_colwise_conv_pre);

  cl_int status = 0;

  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  runtime_t::adapt_uword S_m_n_rows(m_n_rows);

  runtime_t::adapt_uword start_row(aux_row1);
  runtime_t::adapt_uword start_col(aux_col1);

  runtime_t::adapt_uword S_n_rows(n_rows);
  runtime_t::adapt_uword S_n_cols(n_cols);

  status |= clSetKernelArg(k1, 0,  sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= clSetKernelArg(k1, 1,  sizeof(cl_mem), &(mem.cl_mem_ptr)    );
  status |= clSetKernelArg(k1, 2, S_m_n_rows.size, S_m_n_rows.addr      );
  status |= clSetKernelArg(k1, 3,  start_row.size,  start_row.addr      );
  status |= clSetKernelArg(k1, 4,  start_col.size,  start_col.addr      );
  status |= clSetKernelArg(k1, 5,   S_n_rows.size,   S_n_rows.addr      );
  status |= clSetKernelArg(k1, 6,   S_n_cols.size,   S_n_cols.addr      );

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0                };
  const size_t k1_work_size[1]   = { size_t(n_cols) };

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k1, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "accu()");

  // combine the column sums

  cl_kernel k2 = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu_simple);

  status |= clSetKernelArg(k2, 0, sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= clSetKernelArg(k2, 1, sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= clSetKernelArg(k2, 2,  S_n_cols.size, S_n_cols.addr        );

  const size_t k2_work_dim       = 1;
  const size_t k2_work_offset[1] = { 0 };
  const size_t k2_work_size[1]   = { 1 };

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k2, k2_work_dim, k2_work_offset, k2_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "accu()");

  return tmp(0);
  }
