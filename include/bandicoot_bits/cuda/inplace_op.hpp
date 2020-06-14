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
inplace_op_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem, oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &val,
      (uword*) &n_elem };

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "cuda::inplace_op_scalar(): cuLaunchKernel() failed" );
  }



/**
 * Run a CUDA array-wise kernel.
 */
template<typename eT1, typename eT2>
inline
void
inplace_op_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem, twoway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

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

  coot_check_cuda_error( result, "cuda::inplace_op_array(): cuLaunchKernel() failed" );
  }



/**
 * Run a CUDA kernel on a subview.
 */
template<typename eT>
inline
void
inplace_op_subview(dev_mem_t<eT> dest, const eT val, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword m_n_rows, oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 && n_cols == 0) { return; }

  const uword end_row = aux_row1 + n_rows - 1;
  const uword end_col = aux_col1 + n_cols - 1;

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &val,
      (uword*) &end_row,
      (uword*) &end_col,
      (uword*) &m_n_rows,
      (uword*) &aux_row1,
      (uword*) &aux_col1 };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args,
      0);

  coot_check_cuda_error( result, "cuda::inplace_op_subview(): cuLaunchKernel() failed");
  }



/**
 * Run a CUDA kernel on a subview where the operation involves another matrix.
 */
template<typename eT1, typename eT2>
inline
void
inplace_op_subview(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, twoway_kernel_id::enum_id num, const char* identifier)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      (uword*) &aux_row1,
      (uword*) &aux_col1,
      (uword*) &M_n_rows,
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, std::string(identifier) + ": cuLaunchKernel() failed");
  }



/**
 * Use CUDA to extract a subview into the place of a matrix.
 */
template<typename eT>
inline
void
extract_subview(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword M_n_rows, const uword /* M_n_cols */, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
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

  cudaError_t result = cudaMemcpy2D(
      dest.cuda_mem_ptr,
      d_pitch,
      src.cuda_mem_ptr + aux_col1 * M_n_rows + aux_row1, // offset to right place
      s_pitch,
      width,
      height,
      cudaMemcpyDeviceToDevice);

  coot_check_cuda_error(result, "subview::extract(): couldn't copy buffer");
  }



//! @}
