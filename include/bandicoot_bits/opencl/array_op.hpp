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

  coot_check_runtime_error( (status != 0), "opencl::array_op(): couldn't execute kernel" );
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

  coot_check_runtime_error( (status != 0), "opencl::copy_array(): couldn't copy buffer" );
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

  coot_check_runtime_error( (status != 0), "opencl::copy_array(): couldn't copy buffer");
  }



//! @}
