// Copyright 2020 Ryan Curtin (http://www.ratml.org)
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
 * Compute a dot product between two vectors.
 */
template<typename eT1, typename eT2>
inline
typename promote_type<eT1, eT2>::result
dot(dev_mem_t<eT1> mem1, dev_mem_t<eT2> mem2, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  typedef typename promote_type<eT1, eT2>::result promoted_eT;

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  // work out the number of chunks, making sure there are at least 4 elements per compunit

  uword n_chunks = get_rt().cuda_rt.dev_prop.multiProcessorCount;

  while(n_chunks >= 1)
    {
    if( (n_elem / n_chunks) >= uword(4) ) { break; }

    n_chunks /= uword(2);
    }

  n_chunks = (std::max)(uword(1), n_chunks);

  const uword chunk_size = n_elem / n_chunks;

  Mat<promoted_eT> tmp(n_chunks, 1);

  CUfunction k1 = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::dot_chunked);

  dev_mem_t<promoted_eT> tmp_mem = tmp.get_dev_mem(false);

  const void* args[] = {
      &(tmp_mem.cuda_mem_ptr),
      &(mem1.cuda_mem_ptr),
      &(mem2.cuda_mem_ptr),
      (uword*) &chunk_size,
      (uword*) &n_chunks };

  const kernel_dims dims = one_dimensional_grid_dims(n_chunks);

  CUresult result = cuLaunchKernel(
      k1,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args,
      0);

  coot_check_cuda_error(result, "cuda::dot(): cuLaunchKernel() failed");

  // Now that we've computed a partial sum, sum it.  This is really not the best approach; it only uses one thread.  It could be improved to repeatedly use, e.g., accu_chunked.

  CUfunction k2 = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::dot_twostage);

  const size_t A_start = n_chunks * chunk_size;

  const void* args2[] = {
      &(tmp_mem.cuda_mem_ptr),
      (uword*) &tmp.n_elem,
      &(mem1.cuda_mem_ptr),
      &(mem2.cuda_mem_ptr),
      (uword*) &A_start,
      (uword*) &n_elem };

  result = cuLaunchKernel(
      k2,
      1, 1, 1, // grid dims
      1, 1, 1, // block dims
      0, NULL,
      (void**) args2,
      0);

  coot_check_cuda_error(result, "cuda::dot(): cuLaunchKernel() failed");

  promoted_eT ret = promoted_eT(tmp(0));

  return ret;
  }

//! @}

