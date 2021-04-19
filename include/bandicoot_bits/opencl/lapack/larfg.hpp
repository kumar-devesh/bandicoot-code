// Copyright 2021 Ryan Curtin (https://www.ratml.org/)
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
 * Perform the operations involved in larfg().  See the higher-level larfg() documentation for more information.
 *
 * Note that we are assuming that `alpha` is the first element of `x`.
 *
 * Returns tau, and modified x such that x is now [beta, v].
 */
template<typename eT>
inline
double
larfg(dev_mem_t<eT> x, const uword n_elem, const uword rescaling_pass = 1)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot_cl_rt not valid" );

  cl_int status = 0;

  cl_kernel dot_k = get_rt().cl_rt.get_kernel<eT, eT>(twoway_kernel_id::dot);
  cl_kernel dot_small_k = get_rt().cl_rt.get_kernel<eT, eT>(twoway_kernel_id::dot_small);
  cl_kernel larfg_k = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::larfg);

  // Compute workgroup sizes.
  size_t kernel_wg_size;
  status = clGetKernelWorkGroupInfo(dot_k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "larfg()");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset    = 0;
  const uword wavefront_size = get_rt().cl_rt.get_wavefront_size();

  uword total_num_threads = std::ceil(n_elem / (2 * std::ceil(std::log2(n_elem))));
  uword local_group_size = std::min(kernel_wg_size, total_num_threads);

  // We require for size that aux_size is 1.
  const uword aux_size = std::ceil((total_num_threads + (local_group_size - 1)) / local_group_size);
  // TODO: better error message... or maybe just check at a higher level?
  coot_debug_check( (aux_size > 1), "larfg(): input size too large" );

  // This will store `norm`.
  Mat<eT> norm(1, 1);

  runtime_t::cq_guard guard;

  dev_mem_t<eT> norm_mem = norm.get_dev_mem(false);

  runtime_t::adapt_uword dev_n_elem(n_elem);

  const uword pow2_group_size = (uword) std::pow(2.0f, std::ceil(std::log2((float) local_group_size)));
  const uword pow2_total_num_threads = (total_num_threads % pow2_group_size == 0) ? total_num_threads : ((total_num_threads / pow2_group_size) + 1) * pow2_group_size;

  cl_kernel* dot_use = (pow2_group_size < wavefront_size) ? &dot_small_k : &dot_k;

  status |= clSetKernelArg(*dot_use, 0, sizeof(cl_mem),                        &(norm_mem.cl_mem_ptr));
  status |= clSetKernelArg(*dot_use, 1, sizeof(cl_mem),                        &(x.cl_mem_ptr));
  status |= clSetKernelArg(*dot_use, 2, sizeof(cl_mem),                        &(x.cl_mem_ptr));
  status |= clSetKernelArg(*dot_use, 3, dev_n_elem.size,                       dev_n_elem.addr);
  status |= clSetKernelArg(*dot_use, 4, sizeof(eT) * pow2_group_size,          NULL);

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), *dot_use, k1_work_dim, &k1_work_offset, &pow2_total_num_threads, &pow2_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "larfg()");

  // This is an approximation of dlamch('E') / dlamch('S').
  const eT min_norm = std::numeric_limits<eT>::epsilon() / std::numeric_limits<eT>::min();

  // Now immediately enqueue the larfg_work kernel.
  status |= clSetKernelArg(larfg_k, 0, sizeof(cl_mem), &(x.cl_mem_ptr));
  status |= clSetKernelArg(larfg_k, 1, dev_n_elem.size, dev_n_elem.addr);
  status |= clSetKernelArg(larfg_k, 2, sizeof(cl_mem), &(norm_mem.cl_mem_ptr));
  status |= clSetKernelArg(larfg_k, 3, sizeof(eT),     &min_norm);

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), larfg_k, k1_work_dim, &k1_work_offset, &pow2_total_num_threads, &pow2_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "larfg()");

  // Get `alpha` and `beta` to compute `tau`.
  eT beta = eT(0);
  status |= clEnqueueReadBuffer(get_rt().cl_rt.get_cq(), x.cl_mem_ptr, CL_TRUE, 0, sizeof(eT), &beta, 0, NULL, NULL);
  coot_check_cl_error(status, "larfg()");

  // Was `beta` too small?  If so, we have to rescale and try again.
  // This will call larfg() recursively up to 20 times.
  if (std::abs(beta) < min_norm && rescaling_pass < 20)
    {
    // Scale all elements in x.
    inplace_op_scalar(x, min_norm, n_elem, oneway_kernel_id::inplace_div_scalar);

    // Now, try again.
    const eT tau = larfg(x, n_elem, rescaling_pass + 1);

    // Set beta back to its unscaled value---but only if we're at the top of the recursion.
    if (rescaling_pass == 1)
    {
      status |= clEnqueueWriteBuffer(get_rt().cl_rt.get_cq(), x.cl_mem_ptr, CL_TRUE, 0, sizeof(eT), &beta, 0, NULL, NULL);
      coot_check_cl_error(status, "larfg()");
    }

    return tau;
    }
  else
    {
    // No rescaling needed---just compute tau and return it.
    const eT alpha = norm[0];
    const eT tau = (beta / (alpha - beta));

    return tau;
    }
  }

//! @}
