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



template<typename eT>
inline
void
join_rows(dev_mem_t<eT> out, const dev_mem_t<eT> A, const dev_mem_t<eT> B, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_extra_debug_sigprint();

  // When the types are all the same, we can use cudaMemcpy() directly.

  const size_t A_n_elem = A_n_rows * A_n_cols;
  const size_t B_n_elem = B_n_rows * B_n_cols;

  cudaError_t result;
  result = cudaMemcpy((void*) out.cuda_mem_ptr,
                      (void*) A.cuda_mem_ptr,
                      A_n_elem * sizeof(eT),
                      cudaMemcpyDeviceToDevice);
  coot_check_cuda_error(result, "coot::cuda::join_rows(): could not copy first argument");

  result = cudaMemcpy((void*) (out.cuda_mem_ptr + A_n_elem),
                      (void*) B.cuda_mem_ptr,
                      B_n_elem,
                      cudaMemcpyDeviceToDevice);
  coot_check_cuda_error(result, "coot::cuda::join_rows(): could not copy second argument");
  }



template<typename eT1, typename eT2, typename eT3>
inline
void
join_cols(dev_mem_t<eT3> out, const dev_mem_t<eT1> A, const dev_mem_t<eT2> B, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_extra_debug_sigprint();

  const uword A_n_elem = A_n_rows * A_n_cols;
  const uword B_n_elem = B_n_rows * B_n_cols;

  // If the types are different, we need to perform a cast during the copy.  We can use the submat_inplace_set_mat kernel for this.
  copy_array(out, A, A_n_elem);
  dev_mem_t<eT3> out_offset;
  out_offset.cuda_mem_ptr = out.cuda_mem_ptr + A_n_elem;
  copy_array(out_offset, B, B_n_elem);
  }
