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
 * Compute the row-wise or column-wise mean of the input matrix, storing the result in the output matrix.
 */
template<typename eT2, typename eT1>
inline
void
median(dev_mem_t<eT2> out, dev_mem_t<eT1> in, const uword n_rows, const uword n_cols, const uword dim)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::median(): CUDA runtime not valid" );

  if (dim == 0)
    {
    // Sort the data in each column.
    sort_colwise(in, n_rows, n_cols, 0);
    const uword middle_element = (n_rows / 2);

    if (n_rows % 2 == 0)
      {
      // Even number of elements; we have to do a little extra processing.
      sum_colwise_subview(out, in, n_rows, middle_element - 1, 0, 2, n_cols, true);
      inplace_op_scalar(out, eT2(2), n_cols, oneway_kernel_id::inplace_div_scalar);
      }
    else
      {
      // Odd number of elements: the middle element is the result.
      // Now extract that row into the output.
      copy_subview(out, in, middle_element, 0, n_rows, n_cols, 1, n_cols);
      }
    }
  else
    {
    // Sort the data in each row.
    sort_rowwise(in, n_rows, n_cols, 0);
    const uword middle_element = (n_cols / 2);

    if (n_cols % 2 == 0)
      {
      // Even number of elements; we have to do a little extra processing.
      sum_rowwise_subview(out, in, n_rows, 0, middle_element - 1, n_rows, 2, true);
      inplace_op_scalar(out, eT2(2), n_rows, oneway_kernel_id::inplace_div_scalar);
      }
    else
      {
      // Odd number of elements: the middle element is the result.
      // Now extract that column into the output.
      copy_subview(out, in, 0, middle_element, n_rows, n_cols, n_rows, 1);
      }
    }
  }



template<typename eT>
inline
eT
median_vec(dev_mem_t<eT> in, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::median(): CUDA runtime not valid" );

  // Sort the data.
  sort_vec(in, n_elem, 0);
  // Now get the median element.
  const uword middle_element = n_elem / 2;
  if (n_elem % 2 == 0)
    {
    // Even number of elements: average the two middle elements.
    eT val1 = get_val(in, middle_element - 1);
    eT val2 = get_val(in, middle_element);
    return (val1 + val2) / 2;
    }
  else
    {
    // Odd number of elements: the easy case.
    return get_val(in, middle_element);
    }
  }
