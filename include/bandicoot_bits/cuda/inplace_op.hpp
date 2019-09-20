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
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &val,
      (uword*) &n_elem };

  CUresult result = cuLaunchKernel(
      kernel,
      std::ceil((double) n_elem / (double) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock), 1, 1, // grid dims
      get_rt().cuda_rt.dev_prop.maxThreadsPerBlock, 1, 1, // block dims
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "cuda::inplace_op_scalar(): cuLaunchKernel() failed" );

  cuCtxSynchronize();
  }



/**
 * Run a CUDA array-wise kernel.
 */
template<typename eT>
inline
void
inplace_op_array(dev_mem_t<eT> dest, dev_mem_t<eT> src, const uword n_elem, kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      (uword*) &n_elem };

  CUresult result = cuLaunchKernel(
      kernel,
      std::ceil((double) n_elem / (double) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock), 1, 1, // grid dims
      get_rt().cuda_rt.dev_prop.maxThreadsPerBlock, 1, 1, // block dims
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "cuda::inplace_op_array(): cuLaunchKernel() failed" );

  cuCtxSynchronize();
  }



/**
 * Run a CUDA kernel on a subview.
 */
template<typename eT>
inline
void
inplace_op_subview(dev_mem_t<eT> dest, const eT val, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword m_n_rows, kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 && n_cols == 0) { return; }

  const uword end_row = aux_row1 + n_rows - 1;
  const uword end_col = aux_col1 + n_cols - 1;
  const uword n_elem = n_rows * n_cols; // TODO: maybe pass this?

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

  // grid dimensions:
  //   ideally, we want to use [n_rows, n_cols, 1]; but we have limits.  so   //   we might need to block it up a bit.  so, if n_rows * n_cols < maxThreadsPerBlock,
  //   we can use [n_rows, n_cols, 1]; otherwise, if n_rows < maxThreadsPerBlock,
  //      we can use [n_rows, maxThreadsPerBlock / n_rows, 1];
  //      and in this case we'll need a grid size of [1, ceil(n_cols / (mtpb / n_rows)), 1];
  //
  //   and if n_rows > mtpb,
  //      we can use [mtpb, 1, 1]
  //      and a grid size of [ceil(n_rows / mtpb), n_cols, 1].
  //
  // TODO: move this to some auxiliary code because it will surely be useful elsewhere
  size_t blockSize[2] = { n_rows, n_cols };
  size_t gridSize[2] = { 1, 1 };

  if (int(n_rows) > get_rt().cuda_rt.dev_prop.maxThreadsPerBlock)
    {
    blockSize[0] = get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
    blockSize[1] = 1;

    gridSize[0] = std::ceil((double) n_rows / (double) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock);
    gridSize[1] = n_cols;
    }
  else if (int(n_elem) > get_rt().cuda_rt.dev_prop.maxThreadsPerBlock)
    {
    blockSize[0] = n_rows;
    blockSize[1] = std::floor((double) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock / (double) n_rows);

    gridSize[1] = std::ceil((double) n_rows / (double) blockSize[1]);
    }

  CUresult result = cuLaunchKernel(
      kernel,
      gridSize[0], gridSize[1], 1,
      blockSize[0], blockSize[1], 1,
      0, NULL, // shared mem and stream
      (void**) args,
      0);

  coot_check_cuda_error( result, "cuda::inplace_op_subview(): cuLaunchKernel() failed");

  cuCtxSynchronize();
  }



/**
 * Run a CUDA kernel on a subview where the operation involves another matrix.
 */
template<typename eT>
inline
void
inplace_op_subview(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, kernel_id::enum_id num, const char* identifier)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      (uword*) &aux_row1,
      (uword*) &aux_col1,
      (uword*) &M_n_rows,
      (uword*) &n_rows,
      (uword*) &n_cols };

  // Compute grid dimensions like the previous bit of code.
  size_t blockSize[2] = { n_rows, n_cols };
  size_t gridSize[2] = { 1, 1 };

  const uword n_elem = n_rows * n_cols;

  if (int(n_rows) > get_rt().cuda_rt.dev_prop.maxThreadsPerBlock)
    {
    blockSize[0] = get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
    blockSize[1] = 1;

    gridSize[0] = std::ceil((double) n_rows / (double) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock);
    gridSize[1] = n_cols;
    }
  else if (int(n_elem) > get_rt().cuda_rt.dev_prop.maxThreadsPerBlock)
    {
    blockSize[0] = n_rows;
    blockSize[1] = std::floor((double) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock / (double) n_rows);

    gridSize[1] = std::ceil((double) n_rows / (double) blockSize[1]);
    }

  CUresult result = cuLaunchKernel(
      kernel,
      gridSize[0], gridSize[1], 1,
      blockSize[0], blockSize[1], 1,
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, std::string(identifier) + ": cuLaunchKernel() failed");

  cuCtxSynchronize();
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
