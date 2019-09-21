// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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



template<typename eT>
inline
void
sum_colwise(dev_mem_t<eT> out, const dev_mem_t<eT> A, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::sum_colwise);

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

  coot_check_cuda_error(result, "cuda::sum_colwise(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }



template<typename eT>
inline
void
sum_rowwise(dev_mem_t<eT> out, const dev_mem_t<eT> A, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::sum_rowwise);

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

  coot_check_cuda_error(result, "cuda::sum_rowwise(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }



template<typename eT>
inline
void
sum_colwise_subview(dev_mem_t<eT> out, const dev_mem_t<eT> A, const uword M_n_rows, const uword start_row, const uword start_col, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::submat_sum_colwise);

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

  coot_check_cuda_error(result, "cuda::sum_colwise_subview(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }



template<typename eT>
inline
void
sum_rowwise_subview(dev_mem_t<eT> out, const dev_mem_t<eT> A, const uword M_n_rows, const uword start_row, const uword start_col, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::submat_sum_rowwise);

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

  coot_check_cuda_error(result, "cuda::sum_rowwise_subview(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }



//! @}
