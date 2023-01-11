// Copyright 2021 Marcus Edel (http://kurg.org)
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
 * Generate a vector with num elements; the values of the elements are linearly spaced from start to (and including) end via CUDA.
 */
template<typename eT>
inline
void
clamp(dev_mem_t<eT> mem, const dev_mem_t<eT> A, const eT min_val, const eT max_val, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::clamp(): cuda runtime not valid");

  const uword n_elem = n_rows * n_cols;

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::clamp);

  const void* args[] = {
      &(A.cuda_mem_ptr),
      (eT*) &min_val,
      (eT*) &max_val,
      (uword*) &n_elem,
      &(mem.cuda_mem_ptr) };

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2], // grid dims
      dims.d[3], dims.d[4], dims.d[5], // block dims
      0,
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "cuda::clamp(): cuLaunchKernel() failed");
  }
