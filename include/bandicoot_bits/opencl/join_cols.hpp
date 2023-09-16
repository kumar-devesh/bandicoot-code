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
join_cols(dev_mem_t<eT> out, const dev_mem_t<eT> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> B, const uword B_n_rows, const uword B_n_cols, const dev_mem_t<eT> C, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT> D, const uword D_n_rows, const uword D_n_cols)
  {
  coot_extra_debug_sigprint();

  const uword out_n_rows = A_n_rows + B_n_rows + C_n_rows + D_n_rows;

  const size_t zeros[3]    = { 0,                                             0,        0 };
  const size_t B_dst[3]    = { A_n_rows * sizeof(eT),                         0,        0 };
  const size_t C_dst[3]    = { (A_n_rows + B_n_rows) * sizeof(eT),            0,        0 };
  const size_t D_dst[3]    = { (A_n_rows + B_n_rows + C_n_rows) * sizeof(eT), 0,        0 };
  const size_t A_region[3] = { A_n_rows * sizeof(eT),                         A_n_cols, 1 };
  const size_t B_region[3] = { B_n_rows * sizeof(eT),                         B_n_cols, 1 };
  const size_t C_region[3] = { C_n_rows * sizeof(eT),                         C_n_cols, 1 };
  const size_t D_region[3] = { D_n_rows * sizeof(eT),                         D_n_cols, 1 };

  const uword A_n_elem = A_n_rows * A_n_cols;
  const uword B_n_elem = B_n_rows * B_n_cols;
  const uword C_n_elem = C_n_rows * C_n_cols;
  const uword D_n_elem = D_n_rows * D_n_cols;

  cl_int status;
  runtime_t::cq_guard guard;

  if (A_n_elem > 0)
    {
    status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(),
                                                   A.cl_mem_ptr,
                                                   out.cl_mem_ptr,
                                                   zeros, // origin is 0 for the first copy, for both input and output
                                                   zeros,
                                                   A_region,
                                                   A_n_rows * sizeof(eT),
                                                   0,
                                                   out_n_rows * sizeof(eT),
                                                   0,
                                                   0,
                                                   NULL,
                                                   NULL);
    coot_check_cl_error(status, "coot::opencl::join_cols(): clEnqueueCopyBufferRect() failed for first argument");
    }

  if (B_n_elem > 0)
    {
    status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(),
                                                   B.cl_mem_ptr,
                                                   out.cl_mem_ptr,
                                                   zeros,
                                                   B_dst, // the B matrix is offset in memory
                                                   B_region,
                                                   B_n_rows * sizeof(eT),
                                                   0,
                                                   out_n_rows * sizeof(eT),
                                                   0,
                                                   0,
                                                   NULL,
                                                   NULL);
    coot_check_cl_error(status, "coot::opencl::join_cols(): clEnqueueCopyBufferRect() failed for second argument");
    }

  if (C_n_elem > 0)
    {
    status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(),
                                                   C.cl_mem_ptr,
                                                   out.cl_mem_ptr,
                                                   zeros,
                                                   C_dst, // the C matrix is offset in memory
                                                   C_region,
                                                   C_n_rows * sizeof(eT),
                                                   0,
                                                   out_n_rows * sizeof(eT),
                                                   0,
                                                   0,
                                                   NULL,
                                                   NULL);
    coot_check_cl_error(status, "coot::opencl::join_cols(): clEnqueueCopyBufferRect() failed for third argument");
    }

  if (D_n_elem > 0)
    {
    status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(),
                                                   D.cl_mem_ptr,
                                                   out.cl_mem_ptr,
                                                   zeros,
                                                   D_dst, // the D matrix is offset in memory
                                                   D_region,
                                                   D_n_rows * sizeof(eT),
                                                   0,
                                                   out_n_rows * sizeof(eT),
                                                   0,
                                                   0,
                                                   NULL,
                                                   NULL);
    coot_check_cl_error(status, "coot::opencl::join_cols(): clEnqueueCopyBufferRect() failed for fourth argument");
    }
  }



template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
inline
void
join_cols(dev_mem_t<eT5> out, const dev_mem_t<eT1> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT2> B, const uword B_n_rows, const uword B_n_cols, const dev_mem_t<eT3> C, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT4> D, const uword D_n_rows, const uword D_n_cols)
  {
  coot_extra_debug_sigprint();

  const uword A_n_elem = A_n_rows * A_n_cols;
  const uword B_n_elem = B_n_rows * B_n_cols;
  const uword C_n_elem = C_n_rows * C_n_cols;
  const uword D_n_elem = D_n_rows * D_n_cols;

  const uword out_n_rows = A_n_rows + B_n_rows + C_n_rows + D_n_rows;

  // If the types are different, we need to perform a cast during the copy.
  if (A_n_elem > 0)
    {
    copy_array(out, A,
               A_n_rows, A_n_cols,
               0, 0, out_n_rows,
               0, 0, A_n_rows);
    }

  if (B_n_elem > 0)
    {
    copy_array(out, B,
               B_n_rows, B_n_cols,
               A_n_rows, 0, out_n_rows,
               0, 0, B_n_rows);
    }

  if (C_n_elem > 0)
    {
    copy_array(out, C,
               C_n_rows, C_n_cols,
               A_n_rows + B_n_rows, 0, out_n_rows,
               0, 0, C_n_rows);
    }

  if (D_n_elem > 0)
    {
    copy_array(out, D,
               D_n_rows, D_n_cols,
               A_n_rows + B_n_rows + C_n_rows, 0, out_n_rows,
               0, 0, D_n_rows);
    }
  }
