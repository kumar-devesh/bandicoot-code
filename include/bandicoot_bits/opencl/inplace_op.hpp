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


//! \addtogroup opencl
//! @{

/**
 * Run an OpenCL elementwise kernel that uses a scalar.
 */
template<typename eT>
inline
void
inplace_op_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem, kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(num);

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr) );
  status |= clSetKernelArg(kernel, 1, sizeof(eT),     &val               );
  status |= clSetKernelArg(kernel, 2, N.size,         N.addr             );

  const size_t global_work_size[1] = { size_t(n_elem) };

  coot_extra_debug_print("clEnqueueNDRangeKernel()");

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "opencl::inplace_op_scalar(): couldn't execute kernel" );
  }



/**
 * Run an OpenCL array-wise kernel.
 */
template<typename eT>
inline
void
inplace_op_array(dev_mem_t<eT> dest, dev_mem_t<eT> src, const uword n_elem, kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(num);

  opencl::runtime_t::cq_guard guard;

  opencl::runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr)  );
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &(src.cl_mem_ptr)   );
  status |= clSetKernelArg(kernel, 2, N.size,         N.addr              );

  const size_t global_work_size[1] = { size_t(n_elem) };

  coot_extra_debug_print("clEnqueueNDRangeKernel()");

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "opencl::inplace_op_array(): couldn't execute kernel");
  }



/**
 * Run an OpenCL kernel on a subview.
 */
template<typename eT>
inline
void
inplace_op_subview(dev_mem_t<eT> dest, const eT val, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword m_n_rows, kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 && n_cols == 0) { return; }

  runtime_t::cq_guard guard;

  const uword end_row = aux_row1 + n_rows - 1;
  const uword end_col = aux_col1 + n_cols - 1;

  runtime_t::adapt_uword m_end_row(end_row);
  runtime_t::adapt_uword m_end_col(end_col);
  runtime_t::adapt_uword m_n_rows_a(m_n_rows);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(num);

  cl_int status = 0;

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr));
  status |= clSetKernelArg(kernel, 1, sizeof(eT),     &val);
  status |= clSetKernelArg(kernel, 2,  m_end_row.size, m_end_row.addr);
  status |= clSetKernelArg(kernel, 3,  m_end_col.size, m_end_col.addr);
  status |= clSetKernelArg(kernel, 4, m_n_rows_a.size, m_n_rows_a.addr);

  size_t global_work_offset[2] = { size_t(aux_row1), size_t(aux_col1) }; // starting point in parent matrix
  size_t global_work_size[2]   = { size_t(n_rows),   size_t(n_cols)   }; // size of submatrix

  // NOTE: Clover / Mesa 13.0.4 can't handle offsets
  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 2, global_work_offset, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "opencl::inplace_op_subview(): couldn't execute kernel" );
  }



/**
 * Run an OpenCL kernel on a subview where the operation involves another matrix.
 */
template<typename eT>
inline
void
inplace_op_subview(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, kernel_id::enum_id num, const char* identifier)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(num);

  runtime_t::adapt_uword start_row(aux_row1);
  runtime_t::adapt_uword start_col(aux_col1);

  runtime_t::adapt_uword m_n_rows(M_n_rows);

  runtime_t::adapt_uword X_n_rows(n_rows);
  runtime_t::adapt_uword X_n_cols(n_cols);

  cl_int status = 0;

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr));
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem),  &(src.cl_mem_ptr));
  status |= clSetKernelArg(kernel, 2, start_row.size,     start_row.addr);
  status |= clSetKernelArg(kernel, 3, start_col.size,     start_col.addr);
  status |= clSetKernelArg(kernel, 4,  m_n_rows.size,      m_n_rows.addr);
  status |= clSetKernelArg(kernel, 5,  X_n_rows.size,      X_n_rows.addr);
  status |= clSetKernelArg(kernel, 6,  X_n_cols.size,      X_n_cols.addr);

  size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), std::string(identifier) + std::string(": couldn't execute kernel") );
  }



/**
 * Use OpenCL to extract a subview into the place of a matrix.
 */
template<typename eT>
inline
void
extract_subview(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  // treat the matrix as an image rotated 90 degrees
  // width  of img = number of rows
  // height of img = number of cols

  // whoever designed the API for clEnqueueCopyBufferRect() should be permanently removed from the gene pool;
  // the starting row needs to be multiplied by the element size,
  // because it was too logical to add a separate "size of element" argument

  // TODO: is using clEnqueueCopyBufferRect actually faster than using a dedicated kernel?

  size_t src_origin[3] = { aux_row1 * sizeof(eT), aux_col1, 0 };
  size_t dst_origin[3] = { 0, 0, 0 };

  size_t region[3] = { n_rows * sizeof(eT), n_cols, 1 };

  size_t src_row_pitch   = sizeof(eT) * M_n_rows;
  size_t src_slice_pitch = sizeof(eT) * M_n_cols * M_n_rows;

  size_t dst_row_pitch   = 0;
  size_t dst_slice_pitch = 0;

  cl_int status = clEnqueueCopyBufferRect(get_rt().cl_rt.get_cq(), src.cl_mem_ptr, dest.cl_mem_ptr, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "subview::extract(): couldn't copy buffer");
  }



//! @}
