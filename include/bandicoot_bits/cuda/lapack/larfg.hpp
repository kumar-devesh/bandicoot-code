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

//! \addtogroup cuda
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

  // Edge case: if x has one or zero elements, tau is zero and we don't need to do anything.
  if (n_elem <= 1)
    {
    return eT(0);
    }

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  // Get the kernels we may use.
  CUfunction dot_k = get_rt().cuda_rt.get_kernel<eT, eT>(twoway_kernel_id::dot);
  CUfunction dot_small_k = get_rt().cuda_rt.get_kernel<eT, eT>(twoway_kernel_id::dot_small);
  CUfunction larfg_k = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::larfg);

  // Compute grid size; ideally we want to use the maximum possible number of threads per block for the dot() computation.
  kernel_dims dims = one_dimensional_grid_dims(n_elem / (2 * std::ceil(std::log2(n_elem))));

  coot_debug_check( (dims.d[0] > 1), "larfg(): input size too large" );

  // This will store `norm`, and after the larfg kernel, it will store [alpha, beta, status].
  Mat<eT> aux(3, 1);
  dev_mem_t<eT> aux_mem = aux.get_dev_mem(false);

  // Ensure we always use a power of 2 for the number of threads.
  const uword num_threads = (uword) std::pow(2.0f, std::ceil(std::log2((float) dims.d[3])));

  const void* args[] = {
      &(aux_mem.cuda_mem_ptr),
      &(x.cuda_mem_ptr),
      &(x.cuda_mem_ptr),
      (uword*) &n_elem };

  CUresult result = cuLaunchKernel(
      num_threads < 32 ? dot_small_k : dot_k, // if we have fewer threads than a single warp, we can use a more optimized version of the kernel
      dims.d[0], dims.d[1], dims.d[2],
      num_threads, dims.d[4], dims.d[5],
      2 * num_threads * sizeof(eT), // shared mem should have size equal to number of threads times 2
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "cuda::larfg(): cuLaunchKernel() failed");

  // Now set up the larfg kernel and launch it too.

  // This is an approximation of dlamch('S') / dlamch('E').
  const eT min_norm = std::numeric_limits<eT>::min() / std::numeric_limits<eT>::epsilon();

  const void* larfg_args[] = {
      &(x.cuda_mem_ptr),
      (uword*) &n_elem,
      &(aux_mem.cuda_mem_ptr),
      (eT*) &min_norm };

  dims = one_dimensional_grid_dims(n_elem);

  result = cuLaunchKernel(
      larfg_k,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) larfg_args,
      0);

  coot_check_cuda_error(result, "cuda::larfg(): cuLaunchKernel() failed");

  // Get results to compute `tau`.
  eT out_vals[3];
  copy_from_dev_mem(&out_vals[0], aux_mem, 3);

  // Was `beta` too small?  If so, we have to rescale and try again.
  // This will call larfg() recursively up to 20 times.
  // NOTE: this functionality is currently inaccessible, since dot() will just return a zero norm in this case.
  // (Relevant functionality in the kernel is commented out.)
  if (out_vals[2] == eT(-2) && rescaling_pass < 20)
    {
    // Scale all elements in x.
    inplace_op_scalar(x, min_norm, n_elem, oneway_kernel_id::inplace_div_scalar);

    // Now, try again.
    const eT tau = larfg(x, n_elem, rescaling_pass + 1);

    // Set beta back to its unscaled value---but only if we're at the top of the recursion.
    if (rescaling_pass == 1)
    {
      copy_into_dev_mem(x, &out_vals[1], 1);
    }

    return tau;
    }
  else if (out_vals[2] == eT(-1))
    {
    // In this case, x was all zeros, and we return tau = 0.
    return 0.0;
    }
  else
    {
    // No rescaling needed---just compute tau and return it.
    const eT tau = (out_vals[1] - out_vals[0]) / out_vals[1];

    return tau;
    }
  }

//! @}
