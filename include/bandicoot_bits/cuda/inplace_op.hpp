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
 * Run a CUDA elementwise kernel that uses a scalar.
 */
template<typename eT>
inline
void
inplace_op_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem, kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = coot_rt.cuda_rt.get_kernel<eT>(num);

  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_runtime_error( (result != cudaSuccess), "cuda::inplace_op_scalar(): couldn't get device properties");

  const void* args[] = { &(dest.cuda_mem_ptr), &val, (size_t*) &n_elem };

  CUresult result2 = cuLaunchKernel(
      kernel,
      dev_prop.maxThreadsPerBlock, 1, 1, // grid dims
      std::ceil((double) n_elem / (double) dev_prop.maxThreadsPerBlock), 1, 1, // block dims
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_runtime_error( (result2 != CUDA_SUCCESS), "cuda::inplace_op_scalar(): cuLaunchKernel() failed!");

  cuCtxSynchronize();
  }

//! @}
