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
array_op(dev_mem_t<eT3> out, const uword n_elem, dev_mem_t<eT1> in_a, dev_mem_t<eT2> in_b, threeway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT1, eT2, eT3>(num);

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &( out.cl_mem_ptr) );
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &(in_a.cl_mem_ptr) );
  status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &(in_b.cl_mem_ptr) );
  status |= clSetKernelArg(kernel, 3, N.size,         N.addr             );

  const size_t global_work_size = size_t(n_elem);

  coot_extra_debug_print("clEnqueueNDRangeKernel()");

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::array_op(): couldn't execute kernel" );
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

  cl_int status = clEnqueueCopyBuffer(get_rt().cl_rt.get_cq(), src.cl_mem_ptr, dest.cl_mem_ptr, size_t(0), size_t(0), sizeof(eT) * size_t(n_elem), cl_uint(0), NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::copy_array(): couldn't copy buffer" );
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

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr) );
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &( src.cl_mem_ptr) );
  status |= clSetKernelArg(kernel, 2, N.size,         N.addr             );

  const size_t global_work_size = size_t(n_elem);

  coot_extra_debug_print("clEnqueueNDRangeKernel()");

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::copy_array(): couldn't copy buffer");
  }



/**
 * Use OpenCL to extract a subview into the place of a matrix.
 */
template<typename eT>
inline
void
copy_subview(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword M_n_cols, const uword n_rows, const uword n_cols)
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

  coot_check_runtime_error( (status != 0), "coot::opencl::copy_subview(): couldn't copy buffer");
  }



template<typename out_eT, typename in_eT>
inline
void
copy_subview(dev_mem_t<out_eT> dest, const dev_mem_t<in_eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword /* M_n_cols */, const uword n_rows, const uword n_cols)
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

  cl_int status = 0;

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr) );
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &( src.cl_mem_ptr) );
  status |= clSetKernelArg(kernel, 2, a_aux_row1.size, a_aux_row1.addr   );
  status |= clSetKernelArg(kernel, 3, a_aux_col1.size, a_aux_col1.addr   );
  status |= clSetKernelArg(kernel, 4, a_M_n_rows.size, a_M_n_rows.addr   );
  status |= clSetKernelArg(kernel, 5, a_n_rows.size,   a_n_rows.addr     );
  status |= clSetKernelArg(kernel, 6, a_n_cols.size,   a_n_cols.addr     );

  const size_t global_work_size = size_t(n_rows * n_cols);

  coot_extra_debug_print("clEnqueueNDRangeKernel()");

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::copy_subview(): couldn't copy buffer");
  }
