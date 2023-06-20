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

  // We compute with A, the "constant" matrix, and K, the "kernel" that we rotate.
  // Armadillo selects the "kernel" based on maximum number of elements.
  // However, here we use the number of rows; our code that repacks A into a format where we can use gemvs to compute the result requires it.
  const Mat<eT>& A = (UA.M.n_rows >= UB.M.n_rows) ? UA.M : UB.M;
  const Mat<eT>& K = (UA.M.n_rows >= UB.M.n_rows) ? UB.M : UA.M;

  // The kernel "K" needs to be repacked into a vector in a specific way:
  // Specifically, we need to flip K and store it in a column-major form.
  Col<eT> K_mod(K.n_elem);
  coot_rt_t::rotate_180(K_mod.get_dev_mem(false), K.get_dev_mem(false), K.n_rows, K.n_cols);

  // First let's start with the trivial implementation.
  // Here we treat the smaller matrix (the kernel, call it B) as a row vector.
  // Then, we have to create a "buffer" matrix with size A.n_rows x (B.n_cols * A.n_cols).
  // We will extract copies of the windows being used by each output element with a custom kernel.
  // These copies will correspond only to interleaved columns, not interleaved rows.
  // Thus we will need to loop over rows.
  // In any case, this gives a gemv call (not a gemm call because we only have one kernel).
  // The output of the gemv call is a row of the output matrix.
  // Then we will iterate over rows (B.n_rows) and do the same thing to get each row of the output matrix.

  Mat<eT> buffer;
  const uword mode = in.aux_uword;

  if (mode == 0)
    {
    // "full"
    create_gemv_full_buffer(buffer, A, K);

    const uword out_n_rows = A.n_rows + K.n_rows - 1;
    const uword out_n_cols = A.n_cols + K.n_cols - 1;

    out.set_size(out_n_rows, out_n_cols);
    }
  else
    {
    // "same"
    // Note that we have to create the buffer differently depending on what the output size is.
    if (&A == &UA.M)
      {
      create_gemv_same_buffer(buffer, A, K); // output size is A.size()
      out.set_size(A.n_rows, A.n_cols);
      }
    else
      {
      create_gemv_same_buffer_small(buffer, A, K); // output size is K.size()
      out.set_size(K.n_rows, K.n_cols);
      }
    }

  // Multiplying the flattened kernel with each row of the buffer will give us our results.
  for (uword i = 0; i < out.n_cols; ++i)
    {
    // Now multiply a batch of patches with the kernel.
    coot_rt_t::gemv<eT, true>(out.get_dev_mem(false),
                              i * out.n_rows,
                              1,
                              buffer.get_dev_mem(false),
                              i * K.n_rows,
                              K.n_elem, // this is the number of rows in each column
                              buffer.n_rows, // this is the actual number of rows in `buffer`
                              buffer.n_cols,
                              K_mod.get_dev_mem(false),
                              0,
                              1,
                              1.0,
                              0.0);
    }
  }



template<typename eT>
inline
void
glue_conv2::fill_gemv_buffer_top_bottom(Mat<eT>& buffer, const uword buffer_top_padding, const uword buffer_bottom_padding, const uword kernel_rows)
  {
  // The top and bottom rows of the buffer correspond to sections where K's columns do not fully overlap with A's columns.
  // We zero these out with operations equivalent to the following:
  //    buffer.rows(0, kernel_rows * buffer_top_padding - 1) = 0
  //    buffer.rows(buffer.n_rows - kernel_rows * buffer_bottom_padding, buffer.n_rows - 1) = 0
  coot_rt_t::inplace_op_subview(buffer.get_dev_mem(false),
                                0,
                                (eT) 0,
                                0,
                                0,
                                kernel_rows * buffer_top_padding,
                                buffer.n_cols,
                                buffer.n_rows,
                                oneway_kernel_id::submat_inplace_set_scalar);
  coot_rt_t::inplace_op_subview(buffer.get_dev_mem(false),
                                0,
                                (eT) 0,
                                buffer.n_rows - (kernel_rows * buffer_bottom_padding),
                                0,
                                kernel_rows * buffer_bottom_padding,
                                buffer.n_cols,
                                buffer.n_rows,
                                oneway_kernel_id::submat_inplace_set_scalar);
  }



// `i` is the index into the "full" buffer
// `j` is the index of the column in `buffer` that we will use
template<typename eT>
inline
void
glue_conv2::fill_gemv_buffer_col(Mat<eT>& buffer, const uword i, const uword j, const Mat<eT>& A, const Mat<eT>& K, const uword buffer_top_padding, const uword A_col_offset)
  {
  const uword cols_to_copy = (std::min)(A.n_cols - A_col_offset, buffer.n_rows / K.n_rows);

  if (i < K.n_rows - 1)
    {
    // This column corresponds to where K does not yet fully overlap A.
    //
    // Rows in the range [0, K.n_rows - i - 2] are filled with zeros.
    // The way that we do this is a little bit clever, but treat buffer.col(j) as a matrix of size K.n_rows x A.n_cols (call it bufmat_j).
    // Note that for buffer.col(j) we ignore the top and bottom row zero padding.
    // Then, we can say:
    //    bufmat_j.submat(0, 0, K.n_rows - i - 2, A.n_cols - 1) = 0
    //    bufmat_j.submat(K.n_rows - i - 1, 0, K.n_rows, A.n_cols - 1) = A.submat(0, 0, i, A.n_cols - 1)
    coot_rt_t::inplace_op_subview(buffer.get_dev_mem(false),
                                  j * buffer.n_rows + K.n_rows * buffer_top_padding,
                                  (eT) 0,
                                  0,
                                  0,
                                  K.n_rows - i - 1,
                                  cols_to_copy,
                                  K.n_rows,
                                  oneway_kernel_id::submat_inplace_set_scalar);

    coot_rt_t::copy_subview_to_subview(buffer.get_dev_mem(false),
                                       K.n_rows - i - 1 + ((j * buffer.n_rows) + K.n_rows * buffer_top_padding),
                                       0,
                                       K.n_rows,
                                       A.n_cols,
                                       A.get_dev_mem(false),
                                       0,
                                       A_col_offset,
                                       A.n_rows,
                                       A.n_cols,
                                       i + 1,
                                       cols_to_copy);
    }
  else if (i < A.n_rows)
    {
    // This column corresponds to the region where K fully overlaps A.
    const uword A_row = i - (K.n_rows - 1);

    // Copy each individual block.
    // Equivalent to:
    //    buffer.col(j) = vectorise(A.submat(A_row, 0, A_row + K.n_rows - 1, A.n_cols - 1))
    // Note that for buffer.col(j) we ignore the top and bottom row zero padding.
    coot_rt_t::copy_subview(buffer.get_dev_mem(false),
                            j * buffer.n_rows + K.n_rows * buffer_top_padding,
                            A.get_dev_mem(false),
                            A_row,
                            A_col_offset,
                            A.n_rows,
                            A.n_cols,
                            K.n_rows,
                            cols_to_copy);
    }
  else if (i < A.n_rows + 2 * (K.n_rows - 1))
    {
    // Each individual patch from A has its last (i - (kernel_rows + A.n_rows - 1) + 1) rows filled with zeros.
    // (That's rows [(i - (kernel_rows + A.n_rows - 1) + 1), kernel_rows - 1].)
    const uword num_zero_rows = i - A.n_rows + 1;

    // The way that we do this is a little bit clever, but treat buffer.col(j) as a matrix of size kernel_rows x A.n_cols (call it bufmat_j).
    // Then, we can say:
    //    bufmat_j.submat(0, 0, K.n_rows - num_zero_rows - 1, A.n_cols - 1) = A.submat(i - K.n_rows - 1, 0, A.n_rows - 1, A.n_cols - 1)
    //    bufmat_j.submat(K.n_rows - num_zero_rows, 0, K.n_rows - 1, A.n_cols - 1) = 0
    // Note that for buffer.col(j) (or bufmat_j) we ignore the top and bottom zero padding.
    coot_rt_t::copy_subview_to_subview(buffer.get_dev_mem(false),
                                       j * buffer.n_rows + K.n_rows * buffer_top_padding,
                                       0,
                                       K.n_rows,
                                       A.n_cols,
                                       A.get_dev_mem(false),
                                       i - (K.n_rows - 1),
                                       A_col_offset,
                                       A.n_rows,
                                       A.n_cols,
                                       K.n_rows - num_zero_rows,
                                       cols_to_copy);
    coot_rt_t::inplace_op_subview(buffer.get_dev_mem(false),
                                  j * buffer.n_rows + K.n_rows * buffer_top_padding,
                                  (eT) 0,
                                  K.n_rows - num_zero_rows,
                                  0,
                                  num_zero_rows,
                                  cols_to_copy,
                                  K.n_rows,
                                  oneway_kernel_id::submat_inplace_set_scalar);
    }
  }



template<typename eT>
inline
void
glue_conv2::create_gemv_full_buffer(Mat<eT>& buffer, const Mat<eT>& A, const Mat<eT>& K)
  {
  coot_extra_debug_sigprint();

  const uword buffer_n_rows = K.n_rows * (A.n_cols + 2 * (K.n_cols - 1));
  const uword buffer_n_cols = A.n_rows + K.n_rows - 1;

  buffer.set_size(buffer_n_rows, buffer_n_cols);

  // Pad the top and bottom of the buffer with zeros to correspond to regions where K's columns do not fully overlap with A's columns.
  fill_gemv_buffer_top_bottom(buffer, (K.n_cols - 1), (K.n_cols - 1), K.n_rows);

  // Fill each column of the buffer with patches of A.
  for (size_t i = 0; i < buffer_n_cols; ++i)
    {
    fill_gemv_buffer_col(buffer, i, i, A, K, (K.n_cols - 1), 0);
    }
  }



template<typename eT>
inline
void
glue_conv2::create_gemv_same_buffer(Mat<eT>& buffer, const Mat<eT>& A, const Mat<eT>& K)
  {
  coot_extra_debug_sigprint();

  // This is the same as create_gemv_full_buffer()---but the padding strategy is different.
  // Since the output matrix size is the same as the input size, we need to have at least a little bit of padding.
  // (No padding is the "valid" strategy.)
  //
  // We use the same logic as Armadillo/Octave for determining how much zero padding to apply on each side.

  const uword start_row = uword( K.n_rows / 2 );
  const uword start_col = uword( K.n_cols / 2 );

  const uword start_col_padding = K.n_cols - start_col - 1;
  const uword end_col_padding = K.n_cols - start_col_padding - 1;

  // So we only need to consider the region where A and K overlap fully.

  const uword buffer_n_rows = K.n_rows * (A.n_cols + K.n_cols - 1);
  const uword buffer_n_cols = A.n_rows;

  buffer.set_size(buffer_n_rows, buffer_n_cols);

  // Pad the top and bottom of the buffer with zeros to correspond to regions where K's columns do not fully overlap with A's columns.
  fill_gemv_buffer_top_bottom(buffer, start_col_padding, end_col_padding, K.n_rows);

  for (uword i = 0; i < buffer_n_cols; ++i)
    {
    fill_gemv_buffer_col(buffer, i + start_row, i, A, K, start_col_padding, 0);
    }
  }



template<typename eT>
inline
void
glue_conv2::create_gemv_same_buffer_small(Mat<eT>& buffer, const Mat<eT>& A, const Mat<eT>& K)
  {
  coot_extra_debug_sigprint();

  // This is the same as create_gemv_same_buffer()---but here the output matrix is the size of K, not the size of A.
  // Note that K.n_rows < A.n_rows.
  //
  // We use the same logic as Armadillo/Octave for determining how much zero padding to apply on each side.

  // Total row size: A.n_rows + 2 * (K.n_rows - 1) - (K.n_rows + K.n_rows) --> leftover is A.n_rows
  const uword start_row = uword( A.n_rows / 2 );
  const uword start_col = uword( A.n_cols / 2 );

  const uword start_col_padding = (start_col < K.n_cols) ? K.n_cols - start_col - 1 : 0;
  const uword end_col_padding = (start_col < K.n_cols) ? (2 * K.n_cols - 1 - A.n_cols - start_col_padding) : 0;

  // So we only need to consider the region where A and K overlap fully.

  const uword buffer_n_rows = K.n_rows * (K.n_cols + K.n_cols - 1);
  const uword buffer_n_cols = K.n_rows;

  buffer.set_size(buffer_n_rows, buffer_n_cols);

  // Pad the top and bottom of the buffer with zeros to correspond to regions where K's columns do not fully overlap with A's columns.
  fill_gemv_buffer_top_bottom(buffer, start_col_padding, end_col_padding, K.n_rows);

  const uword A_col_offset = (start_col > (K.n_cols - 1)) ? start_col - (K.n_cols - 1) : 0;
  for (uword i = 0; i < buffer_n_cols; ++i)
    {
    fill_gemv_buffer_col(buffer, i + start_row, i, A, K, start_col_padding, A_col_offset);
    }
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
