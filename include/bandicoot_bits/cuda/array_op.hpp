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
 * Run a CUDA elementwise kernel that performs an operation on two matrices.
 */
template<typename eT>
inline
void
array_op(dev_mem_t<eT> out, const uword n_elem, dev_mem_t<eT> in_a, dev_mem_t<eT> in_b, kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(num);

  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_cuda_error(result, "cuda::array_op(): couldn't get device properties");

  const void* args[] = { &(out.cuda_mem_ptr), &(in_a.cuda_mem_ptr), &(in_b.cuda_mem_ptr), (size_t*) &n_elem };

  CUresult result2 = cuLaunchKernel(
      kernel,
      dev_prop.maxThreadsPerBlock, 1, 1, // grid dims
      std::ceil((double) n_elem / (double) dev_prop.maxThreadsPerBlock), 1, 1, // block dims
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result2, "cuda::array_op(): cuLaunchKernel() failed" );

  cuCtxSynchronize();
  }



//! @}
