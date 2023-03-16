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



template<typename eT1, typename eT2>
inline
void
relational_scalar_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  // Shortcut: if there is nothing to do, return.
  if (n_elem == 0)
    return;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(num);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword N(n_elem);

  cl_int status = 0;

  status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &out_mem.cl_mem_ptr);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &in_mem.cl_mem_ptr );
  status |= clSetKernelArg(kernel, 2, N.size,         N.addr             );
  status |= clSetKernelArg(kernel, 3, sizeof(eT2),    &val               );
  coot_check_cl_error(status, "coot::opencl::relational_scalar_op() (" + name + "): couldn't set kernel arguments");

  size_t work_size = size_t(n_elem);

  status = clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::relational_scalar_op() (" + name + "): couldn't execute kernel");
  }
