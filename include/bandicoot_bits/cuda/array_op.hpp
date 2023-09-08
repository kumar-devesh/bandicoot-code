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



/**
 * Run a CUDA elementwise kernel that performs an operation on two matrices.
 */
template<typename eT1, typename eT2, typename eT3>
inline
void
eop_array(const threeway_kernel_id::enum_id num,
          dev_mem_t<eT3> dest,
          const dev_mem_t<eT1> src_A,
          const dev_mem_t<eT2> src_B,
          // logical size of source and destination
          const uword n_rows,
          const uword n_cols,
          // submatrix destination offsets (set to 0, 0, and n_rows if not a subview)
          const uword dest_row_offset,
          const uword dest_col_offset,
          const uword dest_M_n_rows,
          // submatrix source offsets (set to 0, 0, and n_rows if not a subview)
          const uword src_A_row_offset,
          const uword src_A_col_offset,
          const uword src_A_M_n_rows,
          const uword src_B_row_offset,
          const uword src_B_col_offset,
          const uword src_B_M_n_rows)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT3, eT2, eT1>(num);

  const uword src_A_offset = src_A_row_offset + src_A_col_offset * src_A_M_n_rows;
  const uword src_B_offset = src_B_row_offset + src_B_col_offset * src_B_M_n_rows;
  const uword dest_offset  =  dest_row_offset +  dest_col_offset * dest_M_n_rows;

  const eT1* src_A_ptr = src_A.cuda_mem_ptr + src_A_offset;
  const eT2* src_B_ptr = src_B.cuda_mem_ptr + src_B_offset;
  const eT3* dest_ptr  =  dest.cuda_mem_ptr + dest_offset;

  const void* args[] = {
      &dest_ptr,
      &src_A_ptr,
      &src_B_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_M_n_rows,
      (uword*) &src_A_M_n_rows,
      (uword*) &src_B_M_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::eop_array(): cuLaunchKernel() failed" );
  }



/**
 * Use CUDA to copy the source memory to the destination.
 */
template<typename eT>
inline
void
copy_array(dev_mem_t<eT> dest,
           const dev_mem_t<eT> src,
           const uword n_rows,
           const uword n_cols,
           const uword dest_row_offset,
           const uword dest_col_offset,
           const uword dest_M_n_rows,
           const uword src_row_offset,
           const uword src_col_offset,
           const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  cudaError_t result = coot_wrapper(cudaMemcpy2D)(dest.cuda_mem_ptr,
                                                  sizeof(eT) * dest_M_n_rows,
                                                  src.cuda_mem_ptr,
                                                  sizeof(eT) * src_M_n_rows,
                                                  sizeof(eT) * n_rows,
                                                  n_cols,
                                                  cudaMemcpyDeviceToDevice);

  coot_check_cuda_error(result, "coot::cuda::copy_array(): couldn't copy buffer" );
  }



/*
 * Copy source memory to the destination, changing types.
 */
template<typename eT2, typename eT1>
inline
void
copy_array(dev_mem_t<eT2> dest,
           const dev_mem_t<eT1> src,
           const uword n_rows,
           const uword n_cols,
           const uword dest_row_offset,
           const uword dest_col_offset,
           const uword dest_M_n_rows,
           const uword src_row_offset,
           const uword src_col_offset,
           const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::convert_type);

  const uword dest_offset = dest_row_offset + dest_col_offset * dest_M_n_rows;
  const uword  src_offset =  src_row_offset +  src_col_offset * src_M_n_rows;

  const eT2* dest_ptr = dest.cuda_mem_ptr + dest_offset;
  const eT1*  src_ptr =  src.cuda_mem_ptr + src_offset;

  const void* args[] = {
      &dest_ptr,
      &src_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_M_n_rows,
      (uword*) &src_M_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "coot::cuda::copy_array(): cuLaunchKernel() failed");
  }
