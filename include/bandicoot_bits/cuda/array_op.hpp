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
template<typename eT1, typename eT2, typename eT3>
inline
void
array_op(dev_mem_t<eT3> out, const uword n_elem, dev_mem_t<eT1> in_a, dev_mem_t<eT2> in_b, threeway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT1, eT2, eT3>(num);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(in_a.cuda_mem_ptr),
      &(in_b.cuda_mem_ptr),
      (uword*) &n_elem };

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "cuda::array_op(): cuLaunchKernel() failed" );
  }



/**
 * Use CUDA to copy the source memory to the destination.
 */
template<typename eT>
inline
void
copy_array(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  cudaError_t result = cudaMemcpy(dest.cuda_mem_ptr, src.cuda_mem_ptr, sizeof(eT) * size_t(n_elem), cudaMemcpyDeviceToDevice);

  coot_check_cuda_error(result, "cuda::copy_array(): couldn't copy buffer" );
  }



/*
 * Copy source memory to the destination, changing types.
 */
template<typename out_eT, typename in_eT>
inline
void
copy_array(dev_mem_t<out_eT> dest, const dev_mem_t<in_eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<out_eT, in_eT>(twoway_kernel_id::convert_type);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      (uword*) &n_elem };

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "cuda::copy_array(): cuLaunchKernel() failed");
  }


//! @}
