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
join_cols(dev_mem_t<eT> out,
          const dev_mem_t<eT> A,
          const uword A_n_rows,
          const uword A_n_cols,
          const dev_mem_t<eT> B,
          const uword B_n_rows,
          const uword B_n_cols,
          const dev_mem_t<eT> C,
          const uword C_n_rows,
          const uword C_n_cols,
          const dev_mem_t<eT> D,
          const uword D_n_rows,
          const uword D_n_cols,
          // subview arguments
          const uword out_row_offset,
          const uword out_col_offset,
          const uword out_M_n_rows,
          const uword A_row_offset,
          const uword A_col_offset,
          const uword A_M_n_rows,
          const uword B_row_offset,
          const uword B_col_offset,
          const uword B_M_n_rows,
          const uword C_row_offset,
          const uword C_col_offset,
          const uword C_M_n_rows,
          const uword D_row_offset,
          const uword D_col_offset,
          const uword D_M_n_rows)
  {
  coot_extra_debug_sigprint();

  cl_int status;
  runtime_t::cq_guard guard;

  if (A_n_rows > 0 && A_n_cols > 0)
    {
    const size_t out_origin[3] = { sizeof(eT) * out_row_offset, out_col_offset, 0 };
    const size_t A_origin[3]   = { sizeof(eT) * A_row_offset,   A_col_offset,   0 };
    const size_t region[3]     = { A_n_rows * sizeof(eT),       A_n_cols,       1 };

    status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(),
                                                   A.cl_mem_ptr,
                                                   out.cl_mem_ptr,
                                                   A_origin,
                                                   out_origin,
                                                   region,
                                                   A_M_n_rows * sizeof(eT),
                                                   0,
                                                   out_M_n_rows * sizeof(eT),
                                                   0,
                                                   0,
                                                   NULL,
                                                   NULL);
    coot_check_cl_error(status, "coot::opencl::join_cols(): clEnqueueCopyBufferRect() failed for first argument");
    }

  if (B_n_rows > 0 && B_n_cols > 0)
    {
    const size_t out_origin[3] = { sizeof(eT) * (A_n_rows + out_row_offset), out_col_offset, 0 };
    const size_t B_origin[3]   = { sizeof(eT) * B_row_offset,                B_col_offset,   0 };
    const size_t region[3]     = { B_n_rows * sizeof(eT),                    B_n_cols,       1 };

    status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(),
                                                   B.cl_mem_ptr,
                                                   out.cl_mem_ptr,
                                                   B_origin,
                                                   out_origin,
                                                   region,
                                                   B_M_n_rows * sizeof(eT),
                                                   0,
                                                   out_M_n_rows * sizeof(eT),
                                                   0,
                                                   0,
                                                   NULL,
                                                   NULL);
    coot_check_cl_error(status, "coot::opencl::join_cols(): clEnqueueCopyBufferRect() failed for second argument");
    }

  if (C_n_rows > 0 && C_n_cols > 0)
    {
    const size_t out_origin[3] = { sizeof(eT) * (A_n_rows + B_n_rows + out_row_offset), out_col_offset, 0 };
    const size_t C_origin[3]   = { sizeof(eT) * C_row_offset,                           C_col_offset,   0 };
    const size_t region[3]     = { C_n_rows * sizeof(eT),                               C_n_cols,       1 };

    status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(),
                                                   C.cl_mem_ptr,
                                                   out.cl_mem_ptr,
                                                   C_origin,
                                                   out_origin,
                                                   region,
                                                   C_M_n_rows * sizeof(eT),
                                                   0,
                                                   out_M_n_rows * sizeof(eT),
                                                   0,
                                                   0,
                                                   NULL,
                                                   NULL);
    coot_check_cl_error(status, "coot::opencl::join_cols(): clEnqueueCopyBufferRect() failed for third argument");
    }

  if (D_n_rows > 0 && D_n_cols > 0)
    {
    const size_t out_origin[3] = { sizeof(eT) * (A_n_rows + B_n_rows + C_n_rows + out_row_offset), out_col_offset, 0 };
    const size_t D_origin[3]   = { sizeof(eT) * D_row_offset,                                      D_col_offset,   0 };
    const size_t region[3]     = { D_n_rows * sizeof(eT),                                          D_n_cols,       1 };

    status = coot_wrapper(clEnqueueCopyBufferRect)(get_rt().cl_rt.get_cq(),
                                                   D.cl_mem_ptr,
                                                   out.cl_mem_ptr,
                                                   D_origin,
                                                   out_origin,
                                                   region,
                                                   D_M_n_rows * sizeof(eT),
                                                   0,
                                                   out_M_n_rows * sizeof(eT),
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
join_cols(dev_mem_t<eT5> out,
          const dev_mem_t<eT1> A,
          const uword A_n_rows,
          const uword A_n_cols,
          const dev_mem_t<eT2> B,
          const uword B_n_rows,
          const uword B_n_cols,
          const dev_mem_t<eT3> C,
          const uword C_n_rows,
          const uword C_n_cols,
          const dev_mem_t<eT4> D,
          const uword D_n_rows,
          const uword D_n_cols,
          // subview arguments
          const uword out_row_offset,
          const uword out_col_offset,
          const uword out_M_n_rows,
          const uword A_row_offset,
          const uword A_col_offset,
          const uword A_M_n_rows,
          const uword B_row_offset,
          const uword B_col_offset,
          const uword B_M_n_rows,
          const uword C_row_offset,
          const uword C_col_offset,
          const uword C_M_n_rows,
          const uword D_row_offset,
          const uword D_col_offset,
          const uword D_M_n_rows)
  {
  coot_extra_debug_sigprint();

  // If the types are different, we need to perform a cast during the copy.
  if (A_n_rows > 0 && A_n_cols > 0)
    {
    copy_mat(out, A,
             A_n_rows, A_n_cols,
             out_row_offset, out_col_offset, out_M_n_rows,
             A_row_offset, A_col_offset, A_M_n_rows);
    }

  if (B_n_rows > 0 && B_n_cols > 0)
    {
    copy_mat(out, B,
             B_n_rows, B_n_cols,
             A_n_rows + out_row_offset, out_col_offset, out_M_n_rows,
             B_row_offset, B_col_offset, B_M_n_rows);
    }

  if (C_n_rows > 0 && C_n_cols > 0)
    {
    copy_mat(out, C,
             C_n_rows, C_n_cols,
             A_n_rows + B_n_rows + out_row_offset, out_col_offset, out_M_n_rows,
             C_row_offset, C_col_offset, C_M_n_rows);
    }

  if (D_n_rows > 0 && D_n_cols > 0)
    {
    copy_mat(out, D,
             D_n_rows, D_n_cols,
             A_n_rows + B_n_rows + C_n_rows + out_row_offset, out_col_offset, out_M_n_rows,
             D_row_offset, D_col_offset, D_M_n_rows);
    }
  }
