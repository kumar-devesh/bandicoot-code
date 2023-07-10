// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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



// Assumption: the operation is not diagmat(A) * diagmat(B)!
// Assumption: if the non-diagonal matrix is transposed, it is not also the output!
template<typename eT>
inline
void
mul_diag
  (
  dev_mem_t<eT> C_mem,
  const uword C_n_rows,
  const uword C_n_cols,
  const eT alpha,
  const dev_mem_t<eT> A_mem,
  const bool A_is_diag,
  const bool A_trans,
  const dev_mem_t<eT> B_mem,
  const bool B_is_diag,
  const bool B_trans
  )
  {
  coot_extra_debug_sigprint();

  eT* diag_arg_ptr;
  eT* mat_arg_ptr;

  CUfunction kernel;
  kernel_dims dims;

  if (A_is_diag && !B_is_diag && !B_trans)
    {
    // diagmat(A) * B
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::mul_rowwise);

    diag_arg_ptr = A_mem.cuda_mem_ptr;
    mat_arg_ptr  = B_mem.cuda_mem_ptr;

    dims = one_dimensional_grid_dims(C_n_cols);
    }
  else if (!A_is_diag && !A_trans && B_is_diag)
    {
    // A * diagmat(B)
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::mul_colwise);

    diag_arg_ptr = B_mem.cuda_mem_ptr;
    mat_arg_ptr  = A_mem.cuda_mem_ptr;

    dims = one_dimensional_grid_dims(C_n_rows);
    }
  else if (A_is_diag && !B_is_diag && B_trans)
    {
    // diagmat(A) * B'
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::mul_rowwise_trans);

    coot_debug_check( C_mem.cuda_mem_ptr == B_mem.cuda_mem_ptr, "coot::cuda::mul_diag(): incorrect call, alias and transpose not allowed" );

    diag_arg_ptr = A_mem.cuda_mem_ptr;
    mat_arg_ptr  = B_mem.cuda_mem_ptr;

    dims = one_dimensional_grid_dims(C_n_rows);
    }
  else if (!A_is_diag && A_trans && B_is_diag)
    {
    // A' * diagmat(B)
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::mul_colwise_trans);

    coot_debug_check( C_mem.cuda_mem_ptr == A_mem.cuda_mem_ptr, "coot::cuda::mul_diag(): incorrect call, alias and transpose not allowed" );

    diag_arg_ptr = B_mem.cuda_mem_ptr;
    mat_arg_ptr  = A_mem.cuda_mem_ptr;

    dims = one_dimensional_grid_dims(C_n_cols);
    }
  else
    {
    coot_stop_runtime_error("coot::cuda::mul_diag(): incorrect call, at least one matrix must be non-diagonal");
    }

  const void* args[] = {
      &(C_mem.cuda_mem_ptr),
      &diag_arg_ptr,
      &mat_arg_ptr,
      (eT*) &alpha,
      (uword*) &C_n_rows,
      (uword*) &C_n_cols };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::inplace_op_scalar(): cuLaunchKernel() failed" );
  }
