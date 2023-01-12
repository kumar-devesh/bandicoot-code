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


//! \addtogroup cuda
//! @{

/**
 * Compute the minimum of all elements in `mem`.
 * This is basically identical to `accu()`.
 */
template<typename eT>
inline
eT
norm(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  /* const kernel_dims dims = one_dimensional_grid_dims(n_elem); */

  /* CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::norm); */
  if (std::is_same<eT, float>::value)
    {
    float result;
    cublasStatus_t status = cublasSnrm2(get_rt().cuda_rt.cublas_handle, n_elem, (float*) mem.cuda_mem_ptr, 1, &result);
    return result;
    }
  else if (std::is_same<eT, double>::value)
    {
    double  result;
    cublasStatus_t status = cublasDnrm2(get_rt().cuda_rt.cublas_handle, n_elem, (double*) mem.cuda_mem_ptr, 1, &result);
    return result;
    }

  return 0;
  }



//! @}
