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
min(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::min(): cuda runtime not valid" );

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::min);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::min_small);

  // Compute grid size; ideally we want to use the maximum possible number of threads per block.
  kernel_dims dims = one_dimensional_grid_dims(std::ceil(n_elem / (2 * std::ceil(std::log2(n_elem)))));

  // Create auxiliary memory, with size equal to the number of blocks.
  Mat<eT> aux(dims.d[0], 1);
  dev_mem_t<eT> aux_mem = aux.get_dev_mem(false);
  // Initialize this to the right size, if we will have a second run.
  Mat<eT> aux2;
  if (dims.d[0] > 1)
    {
    kernel_dims second_dims = one_dimensional_grid_dims(dims.d[0]);
    aux2.zeros(second_dims.d[0], 1);
    }
  dev_mem_t<eT> aux_mem2 = aux2.get_dev_mem(false);
  Mat<eT>* out = &aux;

  dev_mem_t<eT>* in_mem = &mem;
  dev_mem_t<eT>* out_mem = &aux_mem;

  // Each outer iteration will reduce down to the number of blocks.
  // So, we'll simply keep reducing until we only have one block left.
  uword in_n_elem = n_elem;
  do
    {
    // Ensure we always use a power of 2 for the number of threads.
    const uword num_threads = (uword) std::pow(2.0f, std::ceil(std::log2((float) dims.d[3])));

    const void* args[] = {
        &(in_mem->cuda_mem_ptr),
        (uword*) &in_n_elem,
        &(out_mem->cuda_mem_ptr) };

    CUresult result = cuLaunchKernel(
        num_threads <= 32 ? k_small : k, // if we have fewer threads than a single warp, we can use a more optimized version of the kernel
        dims.d[0], dims.d[1], dims.d[2],
        num_threads, dims.d[4], dims.d[5],
        2 * num_threads * sizeof(eT), // shared mem should have size equal to number of threads times 2
        NULL,
        (void**) args,
        0);

    coot_check_cuda_error(result, "coot::cuda::min(): cuLaunchKernel() failed");

    if (dims.d[0] == 1)
      {
      // We are done.  Terminate.
      break;
      }

    in_n_elem = out->n_elem;
    if (in_mem == &mem)
      {
      in_mem = &aux_mem;
      out_mem = &aux_mem2;
      out = &aux2;
      }
    else
      {
      std::swap(in_mem, out_mem);
      out = (out == &aux) ? &aux2 : &aux;
      }

    // Now compute sizes for the next iteration.
    dims = one_dimensional_grid_dims(std::ceil(in_n_elem / (2 * std::ceil(std::log2(in_n_elem)))));

    } while (true);

  return eT((*out)[0]);
  }



//! @}
