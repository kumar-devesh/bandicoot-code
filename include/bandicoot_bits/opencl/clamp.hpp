// Copyright 2023 Ryan Curtin (http://ratml.org)
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
 * Clamp `dest` to have the elements of `src` limited to the range `[min_val, max_val]`.
 */
template<typename eT1, typename eT2>
inline
void
clamp(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const eT1 min_val, const eT1 max_val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::clamp(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword local_n_elem(n_elem);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::clamp);

  cl_int status = 0;

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),    &(dest.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem),    &(src.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(eT1),       &min_val);
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, sizeof(eT1),       &max_val);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, local_n_elem.size, local_n_elem.addr);
  coot_check_runtime_error( (status != 0), "coot::opencl::clamp(): couldn't set input arguments");

  size_t global_work_size = size_t(n_elem);
  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

  coot_check_runtime_error( (status != 0), "coot::opencl::clamp(): couldn't execute kernel");
  }
