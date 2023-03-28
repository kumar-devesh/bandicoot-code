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



/**
 * Compute the row-wise or column-wise variance of the input matrix, storing the result in the output matrix.
 */
template<typename eT>
inline
void
var(dev_mem_t<eT> out, const dev_mem_t<eT> in, const dev_mem_t<eT> means, const uword n_rows, const uword n_cols, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  coot_ignore(out);
  coot_ignore(in);
  coot_ignore(means);
  coot_ignore(n_rows);
  coot_ignore(n_cols);
  coot_ignore(dim);
  coot_ignore(norm_type);
  }



template<typename eT>
inline
void
var_subview(dev_mem_t<eT> out, const dev_mem_t<eT> in, const dev_mem_t<eT> means, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  coot_ignore(out);
  coot_ignore(in);
  coot_ignore(means);
  coot_ignore(M_n_rows);
  coot_ignore(M_n_cols);
  coot_ignore(aux_row1);
  coot_ignore(aux_col1);
  coot_ignore(n_rows);
  coot_ignore(n_cols);
  coot_ignore(dim);
  coot_ignore(norm_type);
  }



template<typename eT>
inline
eT
var_vec(const dev_mem_t<eT> mem, const eT mean, const uword n_elem, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  coot_ignore(mem);
  coot_ignore(mean);
  coot_ignore(n_elem);
  coot_ignore(norm_type);
  }



template<typename eT>
inline
eT
var_vec_subview(const dev_mem_t<eT> mem, const eT mean, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  coot_ignore(mem);
  coot_ignore(mean);
  coot_ignore(M_n_rows);
  coot_ignore(M_n_cols);
  coot_ignore(aux_row1);
  coot_ignore(aux_col1);
  coot_ignore(n_rows);
  coot_ignore(n_cols);
  coot_ignore(norm_type);
  }
