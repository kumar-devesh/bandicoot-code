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
 * Run an OpenCL elementwise kernel that uses a scalar.
 */
template<typename eT>
inline
void
inplace_op_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem, oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (n_elem == 0)
    return;

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(num);

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(eT),     &val               );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, N.size,         N.addr             );
  coot_check_cl_error(status, "coot::opencl::inplace_op_scalar(): couldn't set kernel arguments");

  const size_t global_work_size[1] = { size_t(n_elem) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::inplace_op_scalar(): couldn't execute kernel");
  }



/**
 * Run an OpenCL array-wise kernel.
 */
template<typename eT1, typename eT2>
inline
void
inplace_op_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem, twoway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(num);

  opencl::runtime_t::cq_guard guard;

  opencl::runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr)  );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem), &(src.cl_mem_ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, N.size,         N.addr              );
  coot_check_cl_error(status, "coot::opencl::inplace_op_array(): couldn't set kernel arguments");

  const size_t global_work_size[1] = { size_t(n_elem) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::inplace_op_array(): couldn't execute kernel");
  }



/**
 * Run an OpenCL kernel on a subview.
 */
template<typename eT>
inline
void
inplace_op_subview(dev_mem_t<eT> dest, const uword dest_offset, const eT val, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword m_n_rows, oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 && n_cols == 0) { return; }

  runtime_t::cq_guard guard;

  const uword end_row = aux_row1 + n_rows - 1;
  const uword end_col = aux_col1 + n_cols - 1;

  runtime_t::adapt_uword m_dest_offset(dest_offset);
  runtime_t::adapt_uword m_end_row(end_row);
  runtime_t::adapt_uword m_end_col(end_col);
  runtime_t::adapt_uword m_n_rows_a(m_n_rows);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(num);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),     &(dest.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, m_dest_offset.size, m_dest_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(eT),         &val);
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, m_end_row.size,     m_end_row.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, m_end_col.size,     m_end_col.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, m_n_rows_a.size,    m_n_rows_a.addr);
  coot_check_cl_error(status, "coot::opencl::inplace_op_subview(): couldn't set kernel arguments");

  size_t global_work_offset[2] = { size_t(aux_row1), size_t(aux_col1) }; // starting point in parent matrix
  size_t global_work_size[2]   = { size_t(n_rows),   size_t(n_cols)   }; // size of submatrix

  // NOTE: Clover / Mesa 13.0.4 can't handle offsets
  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, global_work_offset, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::inplace_op_subview(): couldn't execute kernel");
  }



/**
 * Run an OpenCL kernel that performs an in-place scalar operation on a diagonal
 * of a matrix.
 */
template<typename eT>
inline
void
inplace_op_diag(dev_mem_t<eT> dest, const uword mem_offset, const eT val, const uword n_rows, const uword len, oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (len == 0) { return; }

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(num);

  cl_int status = 0;

  runtime_t::adapt_uword cl_dest_offset(mem_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_len(len);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),      &(dest.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_dest_offset.size, cl_dest_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(eT),          &val);
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_n_rows.size,      cl_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, cl_len.size,         cl_len.addr);
  coot_check_cl_error(status, "coot::opencl::inplace_op_diag(): couldn't set kernel arguments");

  const size_t global_work_size[1] = { size_t(len) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::inplace_op_diag(): failed to run kernel");
  }



/**
 * Run an OpenCL kernel on a subview where the operation involves another matrix.
 */
template<typename eT1, typename eT2>
inline
void
inplace_op_subview(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, twoway_kernel_id::enum_id num, const char* identifier)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(num);

  runtime_t::adapt_uword start_row(aux_row1);
  runtime_t::adapt_uword start_col(aux_col1);

  runtime_t::adapt_uword m_n_rows(M_n_rows);

  runtime_t::adapt_uword X_n_rows(n_rows);
  runtime_t::adapt_uword X_n_cols(n_cols);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem),  &(src.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, start_row.size,     start_row.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, start_col.size,     start_col.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4,  m_n_rows.size,      m_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5,  X_n_rows.size,      X_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6,  X_n_cols.size,      X_n_cols.addr);

  size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), std::string(identifier) + std::string(": couldn't execute kernel") );
  }
