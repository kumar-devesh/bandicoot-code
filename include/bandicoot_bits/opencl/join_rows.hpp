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

  const uword A_n_elem = A_n_rows * A_n_cols;
  const uword B_n_elem = B_n_rows * B_n_cols;

  cl_int status;
  runtime_t::cq_guard guard;

  status = clEnqueueCopyBuffer(get_rt().cl_rt.get_cq(),
                               A.cl_mem_ptr,
                               out.cl_mem_ptr,
                               0,
                               0,
                               sizeof(eT) * A_n_elem,
                               0,
                               NULL,
                               NULL);
  coot_check_cl_error(status, "coot::opencl::join_rows(): clEnqueueCopyBuffer() failed for first argument");

  status = clEnqueueCopyBuffer(get_rt().cl_rt.get_cq(),
                               B.cl_mem_ptr,
                               out.cl_mem_ptr,
                               0,
                               A_n_elem,
                               sizeof(eT) * B_n_elem,
                               0,
                               NULL,
                               NULL);
  coot_check_cl_error(status, "coot::opencl::join_rows(): clEnqueueCopyBuffer() failed for second argument");
  }



template<typename eT1, typename eT2, typename eT3>
inline
void
join_cols(dev_mem_t<eT3> out, const dev_mem_t<eT1> A, const dev_mem_t<eT2> B, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_extra_debug_sigprint();

  const uword out_n_rows = (std::max)(A_n_rows, B_n_rows);

  // If the types are different, we need to perform a cast during the copy.  We can use the submat_inplace_set_mat kernel for this.
  inplace_op_subview(out, A, out_n_rows, 0,        0, A_n_rows, A_n_cols, twoway_kernel_id::submat_inplace_set_mat, "coot::opencl::join_rows()");
  inplace_op_subview(out, B, out_n_rows, 0, A_n_cols, B_n_rows, B_n_cols, twoway_kernel_id::submat_inplace_set_mat, "coot::opencl::join_rows()");
  }
