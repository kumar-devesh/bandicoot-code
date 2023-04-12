// Copyright 2022 Gopi Tatiraju
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



template<typename out_eT, typename T1, typename T2>
inline
void
glue_join_rows::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_join_rows>& glue)
  {
  coot_extra_debug_sigprint();

  const std::string func_name = (glue.aux_uword == 0) ? "join_rows()" : "join_horiz()";

  const no_conv_unwrap<T1> U1(glue.A);
  const no_conv_unwrap<T2> U2(glue.B);

  const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
  const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

  // check for same number of columns
  const uword A_n_rows = X.n_rows;
  const uword A_n_cols = X.n_cols;

  const uword B_n_rows = Y.n_rows;
  const uword B_n_cols = Y.n_cols;

  coot_debug_check
    (
    ( (A_n_cols != B_n_cols) && ( (A_n_rows > 0) || (A_n_cols > 0) ) && ( (B_n_rows > 0) || (B_n_cols > 0) ) ),
    func_name + ": number of columns must be the same in both objects"
    );

  const uword new_n_rows = A_n_rows + B_n_rows;
  const uword new_n_cols = (std::max)(A_n_cols, B_n_cols);

  // Shortcut: if there is nothing to do, leave early.
  if (new_n_rows == 0 || new_n_cols == 0)
    {
    out.set_size(new_n_rows, new_n_cols);
    return;
    }

  if ((void_ptr(&out) == void_ptr(&E1.M)) || (void_ptr(&out) == void_ptr(&E2.M)))
    {
    Mat<out_eT> tmp(new_n_rows, new_n_cols);
    coot_rt_t::join_rows(tmp.get_dev_mem(false), E1.M.get_dev_mem(false), E2.get_dev_mem(false), A_n_rows, A_n_cols, B_n_rows, B_n_cols, func_name);
    out.steal_mem(tmp);
    }
  else
    {
    out.set_size(new_n_rows, new_n_cols);
    coot_rt_t::join_rows(out.get_dev_mem(false), E1.M.get_dev_mem(false), E2.get_dev_mem(false), A_n_rows, A_n_cols, B_n_rows, B_n_cols, func_name);
    }
  }



template<typename T1, typename T2>
inline
uword
glue_join_rows::compute_n_rows(const Glue<T1, T2, glue_join_cols>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_cols);

  return A_n_rows + B_n_rows;
  }



template<typename T1, typename T2>
inline
uword
glue_join_rows::compute_n_cols(const Glue<T1, T2, glue_join_cols>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);

  return (std::max)(A_n_cols, B_n_cols);
  }
