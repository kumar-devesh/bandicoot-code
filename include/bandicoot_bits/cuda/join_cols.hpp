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
join_cols(dev_mem_t<eT> out, const dev_mem_t<eT> A, const dev_mem_t<eT> B, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_extra_debug_sigprint();

  // When the types are all the same, we can use cudaMemcpy2D().

  cudaError_t result;
  result = cudaMemcpy2D((void*) out.cuda_mem_ptr,
                        (A_n_rows + B_n_rows) * sizeof(eT),
                        (void*) A.cuda_mem_ptr,
                        A_n_rows * sizeof(eT),
                        A_n_rows * sizeof(eT),
                        A_n_cols,
                        cudaMemcpyDeviceToDevice);
  coot_check_cuda_error(result, "coot::cuda::join_cols(): could not copy first argument");

  result = cudaMemcpy2D((void*) (out.cuda_mem_ptr + A_n_rows),
                        (A_n_rows + B_n_rows) * sizeof(eT),
                        (void*) B.cuda_mem_ptr,
                        B_n_rows * sizeof(eT),
                        B_n_rows * sizeof(eT),
                        B_n_cols,
                        cudaMemcpyDeviceToDevice);
  coot_check_cuda_error(result, "coot::cuda::join_cols(): could not copy second argument");
  }



template<typename eT1, typename eT2, typename eT3>
inline
void
join_cols(dev_mem_t<eT3> out, const dev_mem_t<eT1> A, const dev_mem_t<eT2> B, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_extra_debug_sigprint();

  const uword out_n_rows = A_n_rows + B_n_rows;

  // If the types are different, we need to perform a cast during the copy.  We can use the submat_inplace_set_mat kernel for this.
  inplace_op_subview(out, A, out_n_rows, 0,        0, A_n_rows, A_n_cols, twoway_kernel_id::submat_inplace_set_mat, "coot::cuda::join_cols()");
  inplace_op_subview(out, B, out_n_rows, A_n_rows, 0, B_n_rows, B_n_cols, twoway_kernel_id::submat_inplace_set_mat, "coot::cuda::join_cols()");
  }
