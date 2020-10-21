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
 * Run an OpenCL non-inplace elementwise kernel.
 */
template<typename eT1, typename eT2>
inline
void
eop_scalar(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_elem, const eT1 aux_val_pre, const eT2 aux_val_post, twoway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(num);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &dest.cl_mem_ptr);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), & src.cl_mem_ptr);
  status |= clSetKernelArg(kernel, 2, sizeof(eT1),    &aux_val_pre    );
  status |= clSetKernelArg(kernel, 3, sizeof(eT2),    &aux_val_post   );
  status |= clSetKernelArg(kernel, 4, N.size,         N.addr          );

  size_t work_size = size_t(n_elem);

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "opencl::eop_scalar(): couldn't execute kernel");
  }


//! @}
