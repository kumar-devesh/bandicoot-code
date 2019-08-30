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
template<typename eT>
inline
void
array_op(dev_mem_t<eT> out, const uword n_elem, dev_mem_t<eT> in_a, dev_mem_t<eT> in_b, kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(num);

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



//! @}
