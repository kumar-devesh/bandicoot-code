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
copy_array(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  cudaError_t result = coot_wrapper(cudaMemcpy)(dest.cuda_mem_ptr, src.cuda_mem_ptr, sizeof(eT) * size_t(n_elem), cudaMemcpyDeviceToDevice);

  coot_check_cuda_error(result, "coot::cuda::copy_array(): couldn't copy buffer" );
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

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "coot::cuda::copy_array(): cuLaunchKernel() failed");
  }



/**
 * Use CUDA to extract a subview into the place of a matrix.
 */
template<typename eT>
inline
void
copy_subview(dev_mem_t<eT> dest, const uword dest_offset, const dev_mem_t<eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword /* M_n_cols */, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  // The width is in bytes, but the height is in number of columns.
  // Note that the terminology here is transposed---"height" refers to columns and "width" refers to rows...
  const size_t height = n_cols;
  const size_t width = n_rows * sizeof(eT);

  // s_pitch and d_pitch refer to the width in bytes of each column of the matrix.
  const size_t s_pitch = M_n_rows * sizeof(eT);
  const size_t d_pitch = n_rows * sizeof(eT);

  // TODO: check that d_pitch or s_pitch isn't too big?
  // TODO: check that memory does not overlap?

  cudaError_t result = coot_wrapper(cudaMemcpy2D)(
      dest.cuda_mem_ptr + dest_offset,
      d_pitch,
      src.cuda_mem_ptr + aux_col1 * M_n_rows + aux_row1, // offset to right place
      s_pitch,
      width,
      height,
      cudaMemcpyDeviceToDevice);

  coot_check_cuda_error(result, "coot::cuda::copy_subview(): couldn't copy buffer");
  }



/**
 * Extract a subview into a destination matrix, optionally changing types.
 */
template<typename out_eT, typename in_eT>
inline
void
copy_subview(dev_mem_t<out_eT> dest, const uword dest_offset, const dev_mem_t<in_eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword /* M_n_cols */, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  // Get the kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<out_eT, in_eT>(twoway_kernel_id::submat_extract);

  out_eT* dest_ptr = dest.cuda_mem_ptr + dest_offset;
  const void* args[] = {
      &(dest_ptr),
      &(src.cuda_mem_ptr),
      (uword*) &aux_row1,
      (uword*) &aux_col1,
      (uword*) &M_n_rows,
      (uword*) &n_rows,
      (uword*) &n_cols
  };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::copy_subview(): cuLaunchKernel() failed");
  }



/**
 * Copy a subview to another subview.
 */
template<typename eT>
inline
void
copy_subview_to_subview(dev_mem_t<eT> dest,
                        const uword dest_aux_row1,
                        const uword dest_aux_col1,
                        const uword dest_M_n_rows,
                        const uword dest_M_n_cols,
                        const dev_mem_t<eT> src,
                        const uword src_aux_row1,
                        const uword src_aux_col1,
                        const uword src_M_n_rows,
                        const uword src_M_n_cols,
                        const uword n_rows,
                        const uword n_cols)
  {
  coot_extra_debug_sigprint();

  // The width is in bytes, but the height is in number of columns.
  // Note that the terminology here is transposed---"height" refers to columns and "width" refers to rows...
  const size_t height = n_cols;
  const size_t width = n_rows * sizeof(eT);

  // s_pitch and d_pitch refer to the width in bytes of each column of the matrix.
  const size_t s_pitch = src_M_n_rows * sizeof(eT);
  const size_t d_pitch = dest_M_n_rows * sizeof(eT);

  // TODO: check that d_pitch or s_pitch isn't too big?
  // TODO: check that memory does not overlap?

  cudaError_t result = coot_wrapper(cudaMemcpy2D)(
      dest.cuda_mem_ptr + dest_aux_col1 * dest_M_n_rows + dest_aux_row1,
      d_pitch,
      src.cuda_mem_ptr + src_aux_col1 * src_M_n_rows + src_aux_row1, // offset to right place
      s_pitch,
      width,
      height,
      cudaMemcpyDeviceToDevice);

  coot_check_cuda_error(result, "coot::cuda::copy_subview_to_subview(): couldn't copy buffer");
  }
