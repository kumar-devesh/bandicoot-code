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
 * Run an OpenCL non-inplace elementwise kernel.
 */
template<typename eT1, typename eT2>
inline
void
eop_scalar(const twoway_kernel_id::enum_id num,
           dev_mem_t<eT2> dest,
           const dev_mem_t<eT1> src,
           const eT1 aux_val_pre,
           const eT2 aux_val_post,
           // logical size of source and destination
           const uword n_rows,
           const uword n_cols,
           // submatrix source offsets (set to 0, 0, and n_rows if not a subview)
           const uword src_row_offset,
           const uword src_col_offset,
           const uword src_M_n_rows,
           // submatrix destination offsets (set to 0, 0, and n_rows if not a subview)
           const uword dest_row_offset,
           const uword dest_col_offset,
           const uword dest_M_n_rows)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(num);

  const uword n_elem = n_rows * n_cols;
  runtime_t::cq_guard guard;
  runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &dest.cl_mem_ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem), & src.cl_mem_ptr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(eT1),    &aux_val_pre    );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, sizeof(eT2),    &aux_val_post   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, N.size,         N.addr          );

  size_t work_size = size_t(n_elem);

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::eop_scalar(): couldn't execute kernel");
  }
