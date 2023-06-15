// Copyright 2023 Ryan Curtin (https://www.ratml.org)
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



template<typename out_eT, typename T1, typename T2>
inline
void
glue_conv2::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_conv2>& in)
  {
  coot_extra_debug_sigprint();

  // This implementation is a pretty simple im2col-based implementation.
  // TODO: handle transposed inputs and other delayed inputs or optimizations

  unwrap<T1> UA(in.A);
  unwrap<T2> UB(in.B);

  // TODO: handle aliases

  typedef typename T1::elem_type eT;

  const Mat<eT>& A = (UA.M.n_elem >= UB.M.n_elem) ? UA.M : UB.M;
  const Mat<eT>& K = (UA.M.n_elem >= UB.M.n_elem) ? UB.M : UA.M;

  // The kernel "K" needs to be repacked into a vector in a specific way:
  // Specifically, we need to flip K and store it in a column-major form.
  Col<eT> K_mod(K.n_elem);
  coot_rt_t::rotate_180(K_mod.get_dev_mem(false), K.get_dev_mem(false), K.n_rows, K.n_cols);

  // Assuming also that "full" is what we are doing.

  const uword out_n_rows = A.n_rows + K.n_rows - 1;
  const uword out_n_cols = A.n_cols + K.n_cols - 1;

  out.set_size(out_n_rows, out_n_cols);

  // First let's start with the trivial implementation.
  // Here we treat the smaller matrix (the kernel, call it B) as a row vector.
  // Then, we have to create a "buffer" matrix with size A.n_rows x (B.n_cols * A.n_cols).
  // We will extract copies of the windows being used by each output element with a custom kernel.
  // These copies will correspond only to interleaved columns, not interleaved rows.
  // Thus we will need to loop over rows.
  // In any case, this gives a gemv call (not a gemm call because we only have one kernel).
  // The output of the gemv call is a row of the output matrix.
  // Then we will iterate over rows (B.n_rows) and do the same thing to get each row of the output matrix.

  const uword kernel_rows = K.n_rows;
  const uword kernel_cols = K.n_cols;

  // Before trying blocking, let's just duplicate everything.
  const uword buffer_n_rows = kernel_rows * (A.n_cols + 2 * (kernel_cols - 1));
  const uword buffer_n_cols = A.n_rows + kernel_rows - 1;

  Mat<eT> buffer(buffer_n_rows, buffer_n_cols);

  // The top and bottom rows of the buffer correspond to sections where K's columns do not fully overlap with A's columns.
  // We zero these out with operations equivalent to the following:
  //    buffer.rows(0, kernel_rows * kernel_cols - 1) = 0
  //    buffer.rows(kernel_rows * A.n_rows, kernel_rows * (A.n_rows + kernel_cols) - 1) = 0
  coot_rt_t::inplace_op_subview(buffer.get_dev_mem(false),
                                0,
                                (eT) 0,
                                0,
                                0,
                                kernel_rows * (kernel_cols - 1),
                                buffer_n_cols,
                                buffer.n_rows,
                                oneway_kernel_id::submat_inplace_set_scalar);
  coot_rt_t::inplace_op_subview(buffer.get_dev_mem(false),
                                0,
                                (eT) 0,
                                kernel_rows * (A.n_rows + kernel_cols - 1),
                                0,
                                kernel_rows * (kernel_cols - 1),
                                buffer.n_cols,
                                buffer.n_rows,
                                oneway_kernel_id::submat_inplace_set_scalar);

  // First, consider the sections where K's rows do not fully overlap with A's rows.
  for (uword i = 0; i < kernel_rows - 1; ++i)
    {
    // Each individual patch has rows in the range [0, kernel_rows - i - 2] filled with zeros.
    // The way that we do this is a little bit clever, but treat buffer.col(i) as a matrix of size kernel_rows x A.n_cols (call it bufmat_i).
    // Note that for buffer.col(i) we ignore the top and bottom row zero padding.
    // Then, we can say:
    //    bufmat_i.submat(0, 0, kernel_rows - i - 2, A.n_cols - 1) = 0
    //    bufmat_i.submat(kernel_rows - i - 1, 0, kernel_rows, A.n_cols - 1) = A.submat(0, 0, i, A.n_cols - 1)
    coot_rt_t::inplace_op_subview(buffer.get_dev_mem(false),
                                  i * buffer.n_rows + kernel_rows * (kernel_cols - 1),
                                  (eT) 0,
                                  0,
                                  0,
                                  kernel_rows - i - 1,
                                  A.n_cols,
                                  kernel_rows,
                                  oneway_kernel_id::submat_inplace_set_scalar);

    coot_rt_t::copy_subview_to_subview(buffer.get_dev_mem(false),
                                       kernel_rows - i - 1 + (i * buffer.n_rows) + kernel_rows * (kernel_cols - 1),
                                       0,
                                       kernel_rows,
                                       A.n_cols,
                                       A.get_dev_mem(false),
                                       0,
                                       0,
                                       A.n_rows,
                                       A.n_cols,
                                       i + 1,
                                       A.n_cols);
    }

  // Now, consider the regions where A and K overlap fully.
  for (uword i = kernel_rows - 1; i < A.n_rows; ++i)
    {
    const uword A_row = i - (kernel_rows - 1);

    // Copy each individual block.
    // Equivalent to:
    //    buffer.col(i) = vectorise(A.submat(A_row, 0, A_row + kernel_rows - 1, A.n_cols - 1))
    // Note that for buffer.col(i) we ignore the top and bottom row zero padding.
    coot_rt_t::copy_subview(buffer.get_dev_mem(false),
                            i * buffer_n_rows + kernel_rows * (kernel_cols - 1),
                            A.get_dev_mem(false),
                            A_row,
                            0,
                            A.n_rows,
                            A.n_cols,
                            kernel_rows,
                            A.n_cols);
    }

  // Lastly, consider the regions where K has passed over A and no longer overlaps fully.
  for (uword i = A.n_rows; i < buffer_n_cols; ++i)
    {
    // Each individual patch from A has its last (i - (kernel_rows + A.n_rows - 1) + 1) rows filled with zeros.
    // (That's rows [(i - (kernel_rows + A.n_rows - 1) + 1), kernel_rows - 1].)
    const uword num_zero_rows = i - A.n_rows + 1;

    // The way that we do this is a little bit clever, but treat buffer.col(i) as a matrix of size kernel_rows x A.n_cols (call it bufmat_i).
    // Then, we can say:
    //    bufmat_i.submat(0, 0, kernel_rows - num_zero_rows - 1, A.n_cols - 1) = A.submat(i - kernel_rows - 1, 0, A.n_rows - 1, A.n_cols - 1)
    //    bufmat_i.submat(kernel_rows - num_zero_rows, 0, kernel_rows - 1, A.n_cols - 1) = 0
    // Note that for buffer.col(i) (or bufmat_i) we ignore the top and bottom zero padding.
    coot_rt_t::copy_subview_to_subview(buffer.get_dev_mem(false),
                                       i * buffer_n_rows + kernel_rows * (kernel_cols - 1),
                                       0,
                                       kernel_rows,
                                       A.n_cols,
                                       A.get_dev_mem(false),
                                       i - (kernel_rows - 1),
                                       0,
                                       A.n_rows,
                                       A.n_cols,
                                       kernel_rows - num_zero_rows,
                                       A.n_cols);

    coot_rt_t::inplace_op_subview(buffer.get_dev_mem(false),
                                  i * buffer.n_rows + kernel_rows * (kernel_cols - 1),
                                  (eT) 0,
                                  kernel_rows - num_zero_rows,
                                  0,
                                  num_zero_rows,
                                  A.n_cols,
                                  kernel_rows,
                                  oneway_kernel_id::submat_inplace_set_scalar);

    }

  for (uword i = 0; i < A.n_cols + kernel_cols - 1; ++i)
    {
    // Now multiply a batch of patches with the kernel.
    coot_rt_t::gemv<eT, true>(out.get_dev_mem(false),
                              i * out.n_rows,
                              1,
                              buffer.get_dev_mem(false),
                              i * kernel_cols,
                              K_mod.n_elem, // this is the number of rows in each column
                              buffer.n_rows, // this is the actual number of rows in `buffer`
                              buffer.n_cols,
                              K_mod.get_dev_mem(false),
                              0,
                              1,
                              1.0,
                              0.0);
    }

  // It would be nice to also try this without the extraction step.
  }



template<typename T1, typename T2>
inline
uword
glue_conv2::compute_n_rows(const Glue<T1, T2, glue_conv2>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(A_n_cols);
  coot_ignore(B_n_cols);

  if (glue.aux_uword == 0)
    {
    // full
    return A_n_rows + B_n_rows - 1;
    }
  else
    {
    // same
    return A_n_rows;
    }
  }



template<typename T1, typename T2>
inline
uword
glue_conv2::compute_n_cols(const Glue<T1, T2, glue_conv2>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);

  if (glue.aux_uword == 0)
    {
    // full
    return A_n_cols + B_n_cols - 1;
    }
  else
    {
    // same
    return A_n_cols;
    }
  }
