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



template<typename eT>
inline
double
norm_2(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  // For non-floating point types, to match Armadillo's behavior, we cast to double for computation.
  // This means that the first pass has to involve a cast, and therefore we do it manually instead of using generic_reduce().
  // (We can use generic_reduce() for follow-up passes, with the accu() kernel.)

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::sqrd_dbl_norm2);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::sqrd_dbl_norm2_small);

  // Compute grid size; ideally we want to use the maximum possible number of threads per block.
  kernel_dims dims = one_dimensional_grid_dims(std::ceil(n_elem / (2 * std::ceil(std::log2(n_elem)))));

  // Create auxiliary memory, with size equal to the number of blocks.
  Mat<double> aux(dims.d[0], 1);
  dev_mem_t<double> aux_mem = aux.get_dev_mem(false);

  const uword num_threads = (uword) std::pow(2.0f, std::ceil(std::log2((float) dims.d[3])));

  const void* args[] = {
      &(mem.cuda_mem_ptr),
      (uword*) &n_elem,
      &(aux_mem.cuda_mem_ptr) };

  CUresult result = cuLaunchKernel(
      num_threads <= 32 ? kernel_small : kernel,
      dims.d[0], dims.d[1], dims.d[2],
      num_threads, dims.d[4], dims.d[5],
      2 * num_threads * sizeof(double), // shared mem should have size equal to number of threads times 2
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error( result, "coot::cuda::norm_2(): cuLaunchKernel() failed" );

  if (dims.d[0] == 1)
    {
    // We're done, so get the result.  Note that the kernel does not take the square root.
    double result = std::sqrt(double(aux[0]));

    // Check for overflow or underflow, and redo a robust computation if so.
    if (result == double(0) || !coot_isfinite(result))
      {
      coot_extra_debug_print("coot::cuda::norm_2(): detected possible underflow or overflow");

      return norm_2_robust(mem, n_elem);
      }
    }
  else
    {
    // Now we need to accumulate all the results in `aux`.  We can just use the accu() kernel.
    CUfunction accu_k = get_rt().cuda_rt.get_kernel<double>(oneway_kernel_id::accu);
    CUfunction accu_k_small = get_rt().cuda_rt.get_kernel<double>(oneway_kernel_id::accu_small);

    const double result = std::sqrt(generic_reduce(k, k_small, "norm_2", aux_mem, dims.d[0]));

    if (result == double(0) || !coot_isfinite(result))
      {
      coot_extra_debug_print("coot::cuda::norm_2(): detected possible underflow or overflow");

      return norm_2_robust(mem, n_elem);
      }
    }
  }



template<typename eT>
inline
double
norm_2_robust(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  // For the robust version, we find the maximum element and normalize to [-1, 1] for norm computation.
  double max_val = double(max_abs(mem, n_elem));

  // For non-floating point types, to match Armadillo's behavior, we cast to double for computation.
  // This means that the first pass has to involve a cast, and therefore we do it manually instead of using generic_reduce().
  // (We can use generic_reduce() for follow-up passes, with the accu() kernel.)

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::sqrd_dbl_norm2_robust);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::sqrd_dbl_norm2_robust_small);

  // Compute grid size; ideally we want to use the maximum possible number of threads per block.
  kernel_dims dims = one_dimensional_grid_dims(std::ceil(n_elem / (2 * std::ceil(std::log2(n_elem)))));

  // Create auxiliary memory, with size equal to the number of blocks.
  Mat<double> aux(dims.d[0], 1);
  dev_mem_t<double> aux_mem = aux.get_dev_mem(false);

  const uword num_threads = (uword) std::pow(2.0f, std::ceil(std::log2((float) dims.d[3])));

  const void* args[] = {
      &(mem.cuda_mem_ptr),
      (uword*) &n_elem,
      &(aux_mem.cuda_mem_ptr),
      (double*) &max_val };

  CUresult result = cuLaunchKernel(
      num_threads <= 32 ? kernel_small : kernel,
      dims.d[0], dims.d[1], dims.d[2],
      num_threads, dims.d[4], dims.d[5],
      2 * num_threads * sizeof(double), // shared mem should have size equal to number of threads times 2
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error( result, "coot::cuda::norm_2_robust(): cuLaunchKernel() failed" );

  if (dims.d[0] == 1)
    {
    const double result = std::sqrt(double(aux[0])) * max_val;
    return result;
    }
  else
    {
    // Now we need to accumulate all the results in `aux`.  We can just use the accu() kernel.
    CUfunction accu_k = get_rt().cuda_rt.get_kernel<double>(oneway_kernel_id::accu);
    CUfunction accu_k_small = get_rt().cuda_rt.get_kernel<double>(oneway_kernel_id::accu_small);

    const double reduce_result = generic_reduce(accu_k, accu_k_small, "norm_2_robust", aux_mem, dims.d[0]);
    const double result = std::sqrt(reduce_result) * max_val;
    return result;
    }
  }



template<typename eT>
inline
eT
norm_k(dev_mem_t<eT> mem, const uword n_elem, const uword k, const typename coot_real_only<eT>::result* junk = 0)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  // For floating-point types, we perform a power-k accumulation.
  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::powk_norm);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::powk_norm_small);

  // Compute grid size; ideally we want to use the maximum possible number of threads per block.
  kernel_dims dims = one_dimensional_grid_dims(std::ceil(n_elem / (2 * std::ceil(std::log2(n_elem)))));

  // Create auxiliary memory, with size equal to the number of blocks.
  Mat<eT> aux(dims.d[0], 1);
  dev_mem_t<eT> aux_mem = aux.get_dev_mem(false);

  const uword num_threads = (uword) std::pow(2.0f, std::ceil(std::log2((float) dims.d[3])));

  const void* args[] = {
      &(mem.cuda_mem_ptr),
      (uword*) &n_elem,
      &(aux_mem.cuda_mem_ptr),
      (uword*) &k };

  CUresult result = cuLaunchKernel(
      num_threads <= 32 ? kernel_small : kernel,
      dims.d[0], dims.d[1], dims.d[2],
      num_threads, dims.d[4], dims.d[5],
      2 * num_threads * sizeof(double), // shared mem should have size equal to number of threads times 2
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error( result, "coot::cuda::norm_k(): cuLaunchKernel() failed" );

  // Armadillo doesn't do overflow/underflow detection for k != 2; so, neither do we.
  return std::pow(generic_reduce(k, k_small, mem, n_elem, k), eT(1.0) / eT(k));
  }



template<typename eT>
inline
double
norm_k(dev_mem_t<eT> mem, const uword n_elem, const uword k, const typename coot_integral_only<eT>::result* junk = 0)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  // For integral types, we imitate Armadillo's behavior, which casts the type to a double and then computes the norm.

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::powk_dbl_norm);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::powk_dbl_norm_small);

  // Compute grid size; ideally we want to use the maximum possible number of threads per block.
  kernel_dims dims = one_dimensional_grid_dims(std::ceil(n_elem / (2 * std::ceil(std::log2(n_elem)))));

  // Create auxiliary memory, with size equal to the number of blocks.
  Mat<double> aux(dims.d[0], 1);
  dev_mem_t<double> aux_mem = aux.get_dev_mem(false);

  const uword num_threads = (uword) std::pow(2.0f, std::ceil(std::log2((float) dims.d[3])));

  const void* args[] = {
      &(mem.cuda_mem_ptr),
      (uword*) &n_elem,
      &(aux_mem.cuda_mem_ptr) };

  CUresult result = cuLaunchKernel(
      num_threads <= 32 ? kernel_small : kernel,
      dims.d[0], dims.d[1], dims.d[2],
      num_threads, dims.d[4], dims.d[5],
      2 * num_threads * sizeof(double), // shared mem should have size equal to number of threads times 2
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error( result, "coot::cuda::norm_k(): cuLaunchKernel() failed" );

  if (dims.d[0] == 1)
    {
    return std::pow(double(aux[0]), 1.0 / double(k));
    }
  else
    {
    CUfunction accu_k = get_rt().cuda_rt.get_kernel<double>(oneway_kernel_id::accu);
    CUfunction accu_k_small = get_rt().cuda_rt.get_kernel<double>(oneway_kernel_id::accu_small);

    return std::pow(generic_reduce(accu_k, accu_k_small, "norm_k", aux_mem, dims.d[0]), 1.0 / double(k));
    }
  }
