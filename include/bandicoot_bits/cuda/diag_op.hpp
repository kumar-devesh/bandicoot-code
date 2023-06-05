// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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
 * Extract a diagonal from a matrix into a column vector.
 */
template<typename eT2, typename eT1>
inline
void
extract_diag(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword mem_offset, const uword n_rows, const uword len)
  {
  coot_extra_debug_sigprint();

  if (len == 0) { return; }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::get_diag);

  eT1* start_mem = in.cuda_mem_ptr + mem_offset;
  const void* args[] = {
      &(out.cuda_mem_ptr),
      &start_mem,
      (uword*) &n_rows,
      (uword*) &len };

  const kernel_dims dims = one_dimensional_grid_dims(len);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error( result, "coot::cuda::extract_diag(): cuLaunchKernel() failed");
  }



/**
 * Set a diagonal in a matrix to the values in a given vector.
 */
template<typename eT>
inline
void
set_diag(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword mem_offset, const uword n_rows, const uword len)
  {
  coot_extra_debug_sigprint();

  if (len == 0) { return; }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::set_diag);

  eT* start_mem = out.cuda_mem_ptr + mem_offset;
  const void* args[] = {
      &start_mem,
      &(in.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &len };

  const kernel_dims dims = one_dimensional_grid_dims(len);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error( result, "coot::cuda::set_diag(): cuLaunchKernel() failed" );
  }



/**
 * Set a diagonal in a matrix to the values in the diagonal of another matrix.
 */
template<typename eT>
inline
void
copy_diag(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword out_mem_offset, const uword in_mem_offset, const uword out_n_rows, const uword in_n_rows, const uword len)
  {
  coot_extra_debug_sigprint();

  if (len == 0) { return; }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::copy_diag);

  eT* out_start_mem = out.cuda_mem_ptr + out_mem_offset;
  eT* in_start_mem  =  in.cuda_mem_ptr +  in_mem_offset;
  const void* args[] = {
      &out_start_mem,
      &in_start_mem,
      (uword*) &out_n_rows,
      (uword*) &in_n_rows,
      (uword*) &len };

  const kernel_dims dims = one_dimensional_grid_dims(len);

  CUresult result = cuLaunchKernel(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error( result, "coot::cuda::copy_diag(): cuLaunchKernel() failed" );
  }
