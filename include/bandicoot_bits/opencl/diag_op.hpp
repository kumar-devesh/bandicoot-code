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
 * Set the diagonal of a matrix to the given column vector.
 */
template<typename eT2, typename eT1>
inline
void
set_diag(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword mem_offset, const uword n_rows, const uword len)
  {
  coot_extra_debug_sigprint();

  if (len == 0) { return; }

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::set_diag);

  cl_int status = 0;

  runtime_t::adapt_uword cl_out_offset(mem_offset);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_len(len);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),      &(out.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_out_offset.size,  cl_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),      &(in.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_n_rows.size,      cl_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, cl_len.size,         cl_len.addr);
  coot_check_cl_error(status, "coot::opencl::set_diag(): couldn't set kernel arguments");

  const size_t global_work_size[1] = { size_t(len) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::set_diag(): failed to run kernel");
  }



/**
 * Set the diagonal of a matrix to the values of a diagonal of another matrix.
 */
template<typename eT>
inline
void
copy_diag(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword out_mem_offset, const uword in_mem_offset, const uword out_n_rows, const uword in_n_rows, const uword len)
  {
  coot_extra_debug_sigprint();

  if (len == 0) { return; }

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::copy_diag);

  cl_int status = 0;

  runtime_t::adapt_uword cl_out_offset(out_mem_offset);
  runtime_t::adapt_uword cl_in_offset(in_mem_offset);
  runtime_t::adapt_uword cl_out_n_rows(out_n_rows);
  runtime_t::adapt_uword cl_in_n_rows(in_n_rows);
  runtime_t::adapt_uword cl_len(len);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),      &(out.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, cl_out_offset.size,  cl_out_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),      &(in.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, cl_in_offset.size,   cl_in_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, cl_out_n_rows.size,  cl_out_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, cl_in_n_rows.size,   cl_in_n_rows.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, cl_len.size,         cl_len.addr);
  coot_check_cl_error(status, "coot::opencl::copy_diag(): couldn't set kernel arguments");

  const size_t global_work_size[1] = { size_t(len) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::copy_diag(): failed to run kernel");
  }
