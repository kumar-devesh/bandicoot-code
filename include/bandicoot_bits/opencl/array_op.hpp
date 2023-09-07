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
 * Run an OpenCL elementwise kernel that performs an operation on two matrices.
 */
template<typename eT1, typename eT2, typename eT3>
inline
void
eop_array(const threeway_kernel_id::enum_id num,
          dev_mem_t<eT3> dest,
          const dev_mem_t<eT1> src_A,
          const dev_mem_t<eT2> src_B,
          // logical size of source and destination
          const uword n_rows,
          const uword n_cols,
          // submatrix destination offsets (set to 0, 0, and n_rows if not a subview)
          const uword dest_row_offset,
          const uword dest_col_offset,
          const uword dest_M_n_rows,
          // submatrix source offsets (set to 0, 0, and n_rows if not a subview)
          const uword src_A_row_offset,
          const uword src_A_col_offset,
          const uword src_A_M_n_rows,
          const uword src_B_row_offset,
          const uword src_B_col_offset,
          const uword src_B_M_n_rows)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT3, eT2, eT1>(num);

  const uword src_A_offset = src_A_row_offset + src_A_col_offset * src_A_M_n_rows;
  const uword src_B_offset = src_B_row_offset + src_B_col_offset * src_B_M_n_rows;
  const uword dest_offset  =  dest_row_offset +  dest_col_offset * dest_M_n_rows;

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword cl_dest_offset(dest_offset);
  runtime_t::adapt_uword cl_src_A_offset(src_A_offset);
  runtime_t::adapt_uword cl_src_B_offset(src_B_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_dest_M_n_rows(dest_M_n_rows);
  runtime_t::adapt_uword cl_src_A_M_n_rows(src_A_M_n_rows);
  runtime_t::adapt_uword cl_src_B_M_n_rows(src_B_M_n_rows);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel,  0,         sizeof(cl_mem), &( dest.cl_mem_ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  1,    cl_dest_offset.size, cl_dest_offset.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  2,         sizeof(cl_mem), &(src_A.cl_mem_ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  3,   cl_src_A_offset.size, cl_src_A_offset.addr  );
  status |= coot_wrapper(clSetKernelArg)(kernel,  4,         sizeof(cl_mem), &(src_B.cl_mem_ptr)   );
  status |= coot_wrapper(clSetKernelArg)(kernel,  5,   cl_src_B_offset.size, cl_src_B_offset.addr  );
  status |= coot_wrapper(clSetKernelArg)(kernel,  6,         cl_n_rows.size, cl_n_rows.addr        );
  status |= coot_wrapper(clSetKernelArg)(kernel,  7,         cl_n_cols.size, cl_n_cols.addr        );
  status |= coot_wrapper(clSetKernelArg)(kernel,  8,  cl_dest_M_n_rows.size, cl_dest_M_n_rows.addr );
  status |= coot_wrapper(clSetKernelArg)(kernel,  9, cl_src_A_M_n_rows.size, cl_src_A_M_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 10, cl_src_B_M_n_rows.size, cl_src_B_M_n_rows.addr);

  const size_t global_work_size[2] = { size_t(n_rows), size_t(n_cols) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::eop_array(): couldn't execute kernel" );

  status = coot_wrapper(clFinish)(get_rt().cl_rt.get_cq());
  coot_check_cl_error(status, "coot::opencl::eop_scalar(): clFinish() failed");
  }



/**
 * Use OpenCL to copy the source memory to the destination.
 */
template<typename eT>
inline
void
copy_array(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  coot_extra_debug_print("clEnqueueCopyBuffer()");

  cl_int status = coot_wrapper(clEnqueueCopyBuffer)(get_rt().cl_rt.get_cq(), src.cl_mem_ptr, dest.cl_mem_ptr, size_t(0), size_t(0), sizeof(eT) * size_t(n_elem), cl_uint(0), NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::copy_array(): couldn't copy buffer" );
  }



/**
 * Copy an array via OpenCL and cast the type of its elements.
 */
template<typename out_eT, typename in_eT>
inline
void
copy_array(dev_mem_t<out_eT> dest, const dev_mem_t<in_eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<out_eT, in_eT>(twoway_kernel_id::convert_type);

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem), &( src.cl_mem_ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, N.size,         N.addr             );

  const size_t global_work_size = size_t(n_elem);

  coot_extra_debug_print("clEnqueueNDRangeKernel()");

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::copy_array(): couldn't copy buffer");
  }



/**
 * Use OpenCL to extract a subview into the place of a matrix.
 */
template<typename eT>
inline
void
copy_subview(dev_mem_t<eT> dest, const uword dest_offset, const dev_mem_t<eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword M_n_cols, const uword n_rows, const uword n_cols)
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

  size_t src_origin[3] = { aux_row1 * sizeof(eT),    aux_col1, 0 };
  size_t dst_origin[3] = { dest_offset * sizeof(eT), 0,        0 };

  size_t region[3] = { n_rows * sizeof(eT), n_cols, 1 };

  size_t src_row_pitch   = sizeof(eT) * M_n_rows;
  size_t src_slice_pitch = sizeof(eT) * M_n_cols * M_n_rows;

  size_t dst_row_pitch   = 0;
  size_t dst_slice_pitch = 0;

  cl_int status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(), src.cl_mem_ptr, dest.cl_mem_ptr, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, 0, NULL, NULL);

  coot_check_cl_error( status, "coot::opencl::copy_subview(): couldn't copy buffer");
  }



template<typename out_eT, typename in_eT>
inline
void
copy_subview(dev_mem_t<out_eT> dest, const uword dest_offset, const dev_mem_t<in_eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword /* M_n_cols */, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<out_eT, in_eT>(twoway_kernel_id::submat_extract);

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword a_aux_row1(aux_row1);
  runtime_t::adapt_uword a_aux_col1(aux_col1);
  runtime_t::adapt_uword a_M_n_rows(M_n_rows);
  runtime_t::adapt_uword a_n_rows(n_rows);
  runtime_t::adapt_uword a_n_cols(n_cols);
  runtime_t::adapt_uword cl_dest_offset(dest_offset);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),      &(dest.cl_mem_ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_dest_offset.size, cl_dest_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),      &( src.cl_mem_ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, a_aux_row1.size,     a_aux_row1.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, a_aux_col1.size,     a_aux_col1.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, a_M_n_rows.size,     a_M_n_rows.addr   );
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, a_n_rows.size,       a_n_rows.addr     );
  status |= coot_wrapper(clSetKernelArg)(kernel, 7, a_n_cols.size,       a_n_cols.addr     );

  const size_t global_work_size[2] = { n_rows, n_cols };

  coot_extra_debug_print("clEnqueueNDRangeKernel()");

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error( status, "coot::opencl::copy_subview(): couldn't copy buffer");
  }



/**
 * Use OpenCL to extract a subview into the place of another subview.
 */
template<typename eT>
inline
void
copy_subview_to_subview(dev_mem_t<eT> dest,
                        const uword dest_aux_row1,
                        const uword dest_aux_col1,
                        const uword dest_M_n_rows,
                        const uword dest_M_n_cols,
                        const dev_mem_t<eT> src,
                        const uword src_aux_row1,
                        const uword src_aux_col1,
                        const uword src_M_n_rows,
                        const uword src_M_n_cols,
                        const uword n_rows,
                        const uword n_cols)
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

  size_t src_origin[3] = { src_aux_row1 * sizeof(eT),  src_aux_col1,  0 };
  size_t dst_origin[3] = { dest_aux_row1 * sizeof(eT), dest_aux_col1, 0 };

  size_t region[3] = { n_rows * sizeof(eT), n_cols, 1 };

  size_t src_row_pitch   = sizeof(eT) * src_M_n_rows;
  size_t src_slice_pitch = sizeof(eT) * src_M_n_cols * src_M_n_rows;

  size_t dst_row_pitch   = sizeof(eT) * dest_M_n_rows;
  size_t dst_slice_pitch = sizeof(eT) * dest_M_n_cols * dest_M_n_rows;

  cl_int status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(), src.cl_mem_ptr, dest.cl_mem_ptr, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, 0, NULL, NULL);

  coot_check_cl_error( status, "coot::opencl::copy_subview_to_subview(): couldn't copy buffer");
  }
