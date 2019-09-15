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
 * Assign the given memory to be the identity matrix via CUDA.
 */
template<typename eT>
inline
void
eye(dev_mem_t<eT> dest, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::eye(): cuda runtime not valid");

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(kernel_id::inplace_set_eye);

  cudaDeviceProp dev_prop;
  cudaError_t result = cudaGetDeviceProperties(&dev_prop, 0);
  coot_check_cuda_error(result, "cuda::inplace_op_scalar(): couldn't get device properties");

  const void* args[] = { &(dest.cuda_mem_ptr), (size_t*) &n_rows, (size_t*) &n_cols };

  size_t blockSize[2] = { n_rows, n_cols };
  size_t gridSize[2] = { 1, 1 };

  const uword n_elem = n_rows * n_cols;

  if (int(n_rows) > dev_prop.maxThreadsPerBlock)
    {
    blockSize[0] = dev_prop.maxThreadsPerBlock;
    blockSize[1] = 1;

    gridSize[0] = std::ceil((double) n_rows / (double) dev_prop.maxThreadsPerBlock);
    gridSize[1] = n_cols;
    }
  else if (int(n_elem) > dev_prop.maxThreadsPerBlock)
    {
    blockSize[0] = n_rows;
    blockSize[1] = std::floor((double) dev_prop.maxThreadsPerBlock / (double) n_rows);

    gridSize[1] = std::ceil((double) n_rows / (double) blockSize[1]);
    }

  CUresult result2 = cuLaunchKernel(
      kernel,
      gridSize[0], gridSize[1], 1,
      blockSize[0], blockSize[1], 1,
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result2, "cuda::eye(): cuLaunchKernel() failed");
  }
