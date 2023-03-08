// Copyright 2021 Ryan Curtin (http://www.ratml.org)
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
 * Compute the maximum of all elements in `mem`.
 * This is basically identical to `accu()`.
 */
template<typename eT>
inline
eT
max(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::max(): cuda runtime not valid" );

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::max);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::max_small);

  return generic_reduce(mem, n_elem, "max", k, k_small, std::make_tuple(/* no extra args */));
  }



template<typename eT1, typename eT2>
inline
void
max_colwise(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::max_colwise_conv_post : twoway_kernel_id::max_colwise_conv_pre);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(A.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = one_dimensional_grid_dims(n_cols);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::max_colwise(): cuLaunchKernel() failed");
  }



template<typename eT1, typename eT2>
inline
void
max_rowwise(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::max_rowwise_conv_post : twoway_kernel_id::max_rowwise_conv_pre);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(A.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = one_dimensional_grid_dims(n_rows);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::max_rowwise(): cuLaunchKernel() failed");
  }



template<typename eT1, typename eT2>
inline
void
max_colwise_subview(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const uword M_n_rows, const uword start_row, const uword start_col, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::submat_max_colwise_conv_post : twoway_kernel_id::submat_max_colwise_conv_pre);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(A.cuda_mem_ptr),
      (uword*) &M_n_rows,
      (uword*) &start_row,
      (uword*) &start_col,
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = one_dimensional_grid_dims(n_cols);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::max_colwise_subview(): cuLaunchKernel() failed");
  }



template<typename eT1, typename eT2>
inline
void
max_rowwise_subview(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const uword M_n_rows, const uword start_row, const uword start_col, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::submat_max_rowwise_conv_post : twoway_kernel_id::submat_max_rowwise_conv_pre);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(A.cuda_mem_ptr),
      (uword*) &M_n_rows,
      (uword*) &start_row,
      (uword*) &start_col,
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = one_dimensional_grid_dims(n_rows);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::max_rowwise_subview(): cuLaunchKernel() failed");
  }
