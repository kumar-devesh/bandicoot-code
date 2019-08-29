// Copyright 2019 Ryan Curtin (http://ratml.org)
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

//! \addtogroup MatValProxy
//! @{

// Utility functions for MatValProxy with the CUDA backend.

template<typename eT>
inline
eT
get_val(const dev_mem_t<eT> mem, const uword index)
  {
  coot_extra_debug_sigprint();

  // We'll just use cudaMemcpy() to copy back the single value.
  // This is inefficient, but without using Unified Memory, I don't see
  // an alternative.

  eT val = eT(0);

  cudaError_t status = cudaMemcpy((void*) &val,
                                  (void*) (mem.cuda_mem_ptr + index),
                                  sizeof(eT),
                                  cudaMemcpyDeviceToHost);

  coot_check_cuda_error(status, "cuda::get_val(): couldn't access device memory");

  cuCtxSynchronize(); // TODO: is this needed for all CUDA operations?

  return val;
  }

//! @}
