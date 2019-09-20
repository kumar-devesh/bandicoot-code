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


//! \addtogroup cuda
//! @{

/**
 * Run a CUDA non-inplace elementwise kernel.
 */
template<typename eT>
inline
void
eop_scalar(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword n_elem, const eT aux_val, kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      &aux_val,
      (uword*) &n_elem };

  CUresult result = cuLaunchKernel(
      kernel,
      std::ceil((double) n_elem / (double) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock), 1, 1, // grid dims
      get_rt().cuda_rt.dev_prop.maxThreadsPerBlock, 1, 1, // block dims
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "cuda::eop_scalar(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }



//! @}
