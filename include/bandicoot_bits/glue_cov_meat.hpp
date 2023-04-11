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
glue_cov::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_cov>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const unwrap<T1> U1(in.A);
  const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
  const unwrap<T2> U2(in.B);
  const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

  // If the input is a row vector, we treat it as a column vector instead.
  const Mat<eT>& AA = (E1.M.n_rows == 1)
      ? Mat<eT>(E1.M.get_dev_mem(false), E1.M.n_cols, E1.M.n_rows)
      : Mat<eT>(E1.M.get_dev_mem(false), E1.M.n_rows, E1.M.n_cols);
  const Mat<eT>& BB = (E2.M.n_rows == 1)
      ? Mat<eT>(E2.M.get_dev_mem(false), E2.M.n_cols, E2.M.n_rows)
      : Mat<eT>(E2.M.get_dev_mem(false), E2.M.n_rows, E2.M.n_cols);

  coot_debug_assert_mul_size(AA, BB, true, false, "cov()");

  if (E1.M.n_elem == 0 || E2.M.n_elem == 0)
    {
    out.reset();
    return;
    }

  const uword N         = AA.n_rows;
  const uword norm_type = in.aux_uword;
  const eT norm_val     = (norm_type == 0) ? ( (N > 1) ? eT(N - 1) : eT(1) ) : eT(N);

  // TODO: a dedicated kernel for this particular operation would be widely useful
  Row<eT> mean_vals_AA, mean_vals_BB;
  op_mean::apply_direct(mean_vals_AA, AA, 0, false); // no conversion
  op_mean::apply_direct(mean_vals_BB, BB, 0, false); // no conversion

  Mat<eT> tmp_AA(AA), tmp_BB(BB);
  for (uword i = 0; i < tmp_AA.n_rows; ++i)
    {
    tmp_AA.row(i) -= mean_vals_AA;
    }
  for (uword i = 0; i < tmp_BB.n_rows; ++i)
    {
    tmp_BB.row(i) -= mean_vals_BB;
    }

  out = conv_to<Mat<out_eT>>::from((tmp_AA.t() * tmp_BB) / norm_val);
  }



template<typename T1, typename T2>
inline
uword
glue_cov::compute_n_rows(const Glue<T1, T2, glue_cov>& op, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(op);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  if (A_n_rows == 1)
    {
    // If the input is a row vector, we treat it as a column vector instead.
    return 1;
    }
  else
    {
    return A_n_cols;
    }
  }



template<typename T1, typename T2>
inline
uword
glue_cov::compute_n_cols(const Glue<T1, T2, glue_cov>& op, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(op);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);

  if (B_n_rows == 1)
    {
    // If the input is a row vector, we treat it as a column vector instead.
    return 1;
    }
  else
    {
    return B_n_cols;
    }
  }
