// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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



template<typename eT1, typename eT2>
inline
void
mean(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const uword n_rows, const uword n_cols, const uword dim, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::mean(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  cl_kernel kernel;
  if (dim == 0)
    {
    kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::mean_colwise_conv_post : twoway_kernel_id::mean_colwise_conv_pre);
    }
  else
    {
    kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::mean_rowwise_conv_post : twoway_kernel_id::mean_rowwise_conv_pre);
    }

  cl_int status = 0;

  runtime_t::adapt_uword A_n_rows(n_rows);
  runtime_t::adapt_uword A_n_cols(n_cols);

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(out.cl_mem_ptr));
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &(A.cl_mem_ptr)  );
  status |= clSetKernelArg(kernel, 2, A_n_rows.size,  A_n_rows.addr    );
  status |= clSetKernelArg(kernel, 3, A_n_cols.size,  A_n_cols.addr    );

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { (dim == 0) ? size_t(n_cols) : size_t(n_rows) };

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::mean(): failed to run kernel");
  }



template<typename eT1, typename eT2>
inline
void
mean_subview(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const uword M_n_rows, const uword start_row, const uword start_col, const uword n_rows, const uword n_cols, const uword dim, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::mean_subview(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  cl_kernel k1;
  if (dim == 0)
    {
    k1 = get_rt().cl_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::submat_mean_colwise_conv_post : twoway_kernel_id::submat_mean_colwise_conv_pre);
    }
  else
    {
    k1 = get_rt().cl_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::submat_mean_rowwise_conv_post : twoway_kernel_id::submat_mean_rowwise_conv_pre);
    }

  cl_int status = 0;

  runtime_t::adapt_uword sv_m_n_rows(M_n_rows);

  runtime_t::adapt_uword cl_start_row(start_row);
  runtime_t::adapt_uword cl_start_col(start_col);

  runtime_t::adapt_uword sub_n_rows(n_rows);
  runtime_t::adapt_uword sub_n_cols(n_cols);

  status |= clSetKernelArg(k1, 0,    sizeof(cl_mem),  &(out.cl_mem_ptr));
  status |= clSetKernelArg(k1, 1,    sizeof(cl_mem),  &(A.cl_mem_ptr)  );
  status |= clSetKernelArg(k1, 2,  sv_m_n_rows.size,  sv_m_n_rows.addr );
  status |= clSetKernelArg(k1, 3, cl_start_row.size, cl_start_row.addr );
  status |= clSetKernelArg(k1, 4, cl_start_col.size, cl_start_col.addr );
  status |= clSetKernelArg(k1, 5,   sub_n_rows.size,   sub_n_rows.addr );
  status |= clSetKernelArg(k1, 6,   sub_n_cols.size,   sub_n_cols.addr );

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { (dim == 0) ? size_t(n_cols) : size_t(n_rows) };

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k1, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::mean_subview(): failed to run kernel");
  }
