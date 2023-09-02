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
 * Run a CUDA elementwise kernel that uses a scalar.
 */
template<typename eT>
inline
void
fill(dev_mem_t<eT> dest,
     const eT val,
     const uword n_rows,
     const uword n_cols,
     const uword row_offset,
     const uword col_offset,
     const uword M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    return;

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::fill);

  const eT* dest_ptr = dest.cuda_mem_ptr + row_offset + (col_offset * M_n_rows);
  const void* args[] = {
      &dest_ptr,
      &val,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &M_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::fill(): cuLaunchKernel() failed" );
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

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::inplace_op_array(): cuLaunchKernel() failed" );
  }



/**
 * Run a CUDA kernel that performs an in-place scalar operation on a diagonal of
 * a matrix.
 */
template<typename eT>
inline
void
inplace_op_diag(dev_mem_t<eT> dest, const uword mem_offset, const eT val, const uword n_rows, const uword len, oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (len == 0) { return; }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(num);

  eT* start_mem = dest.cuda_mem_ptr + mem_offset;
  const void* args[] = {
      &start_mem,
      &val,
      (uword*) &n_rows,
      (uword*) &len };

  const kernel_dims dims = one_dimensional_grid_dims(len);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error( result, "coot::cuda::inplace_op_diag(): cuLaunchKernel() failed");
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

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, std::string(identifier) + ": cuLaunchKernel() failed");
  }
