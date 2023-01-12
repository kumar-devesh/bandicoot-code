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
 * Compute the Cholesky decomposition using OpenCL.
 */
template<typename eT>
inline
bool
chol(dev_mem_t<eT> mem, const uword n_rows)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::chol(): OpenCL runtime not valid");

  magma_int_t info   = 0;
  magma_int_t status = 0;

  // using MAGMA 2.2

  // OpenCL uses opaque memory pointers which hide the underlying type,
  // so we don't need to do template tricks or casting

  if(is_float<eT>::value)
    {
    status = magma_spotrf_gpu(MagmaUpper, n_rows, mem.cl_mem_ptr, n_rows, &info);
    }
  else if(is_double<eT>::value)
    {
    status = magma_dpotrf_gpu(MagmaUpper, n_rows, mem.cl_mem_ptr, n_rows, &info);
    }
  else
    {
    coot_debug_check( true, "coot::opencl::chol(): not implemented for given type" );
    }

  coot_check_magma_error(status, "coot::opencl::chol(): MAGMA failure in potrf_gpu()");

  // now set the lower triangular part (excluding diagonal) to zero
  cl_int status2 = 0;

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::ltri_set_zero);

  // n_rows == n_cols because the Cholesky decomposition requires square matrices.
  runtime_t::adapt_uword dev_n_rows(n_rows);

  status2 |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(mem.cl_mem_ptr));
  status2 |= clSetKernelArg(kernel, 1, dev_n_rows.size, dev_n_rows.addr);
  status2 |= clSetKernelArg(kernel, 2, dev_n_rows.size, dev_n_rows.addr);

  size_t global_work_offset[2] = { 0, 0 };
  size_t global_work_size[2] = { size_t(n_rows), size_t(n_rows) };

  status2 |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 2, global_work_offset, global_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status2, "coot::opencl::chol(): failed to run kernel ltri_set_zero");

  //// using MAGMA 1.3
  //status = magma_dpotrf_gpu(MagmaUpper, out.n_rows, out.get_dev_mem(), 0, out.n_rows, get_rt().cl_rt.get_cq(), &info);

  return true;

  }
