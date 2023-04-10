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



template<typename out_eT, typename T1>
inline
void
op_cor::apply(Mat<out_eT>& out, const Op<T1, op_cor>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const unwrap<T1> U(in.m);
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  if (E.M.n_elem == 0)
    {
    out.reset();
    return;
    }
  else if (E.M.n_elem == 1)
    {
    out.set_size(1, 1);
    out[0] = eT(1);
    return;
    }

  // If the input is a row vector, we treat it as a column vector instead.
  const Mat<eT>& AA = (E.M.n_rows == 1)
      ? Mat<eT>(E.M.get_dev_mem(false), E.M.n_cols, E.M.n_rows)
      : Mat<eT>(E.M.get_dev_mem(false), E.M.n_rows, E.M.n_cols);

  const uword N         = AA.n_rows;
  const uword norm_type = in.aux_uword_a;
  const eT norm_val     = (norm_type == 0) ? ( (N > 1) ? eT(N - 1) : eT(1) ) : eT(N);

  // TODO: a dedicated kernel for this particular operation would be widely useful
  const Col<out_eT> mean_vals;
  op_mean::apply_direct(mean_vals, AA, 0, true); // convert before computing mean

  Mat<out_eT> tmp(AA.n_rows, AA.n_cols);
  coot_rt_t::copy_array(tmp.get_dev_mem(false), AA.get_dev_mem(false), tmp.n_elem);
  for (uword i = 0; i < tmp.n_cols; ++i)
    {
    tmp.col(i) -= mean_vals;
    }
  tmp = (tmp.t() * tmp / norm_val);
  const Col<out_eT> s = sqrt(tmp.diag());

  out = (tmp / (s * s.t()));
  }



template<typename out_eT, typename T1>
inline
void
op_cor::apply(Mat<out_eT>& out, const Op<mtOp<out_eT, T1, mtop_conv_to> op_cor>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const unwrap<T1> U(in.m.m);
  const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

  if (E.M.n_elem == 0)
    {
    out.reset();
    return;
    }
  else if (E.M.n_elem == 1)
    {
    out.set_size(1, 1);
    out[0] = eT(1);
    return;
    }

  // If the input is a row vector, we treat it as a column vector instead.
  const Mat<eT>& AA = (E.M.n_rows == 1)
      ? Mat<eT>(E.M.get_dev_mem(false), E.M.n_cols, E.M.n_rows)
      : Mat<eT>(E.M.get_dev_mem(false), E.M.n_rows, E.M.n_cols);

  const uword N         = AA.n_rows;
  const uword norm_type = in.aux_uword_a;
  const eT norm_val     = (norm_type == 0) ? ( (N > 1) ? eT(N - 1) : eT(1) ) : eT(N);

  // TODO: a dedicated kernel for this particular operation would be widely useful
  const Col<eT> mean_vals = mean(AA, 0);
  Mat<eT> tmp(AA);
  for (uword i = 0; i < tmp.n_cols; ++i)
    {
    tmp.col(i) -= mean_vals;
    }
  tmp = (tmp.t() * tmp / norm_val);
  const Col<eT> s = sqrt(tmp.diag());

  out = conv_to<Mat<out_eT>>::from(tmp / (s * s.t()));
  }



template<typename T1>
inline
uword
op_cor::compute_n_rows(const Op<T1, op_cor>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);

  if (in_n_rows == 1)
    {
    // If the input is a row vector, we treat it as a column vector instead, giving a 1x1 correlation matrix.
    return 1;
    }
  else
    {
    return in_n_cols;
    }
  }



template<typename T1>
inline
uword
op_cor::compute_n_cols(const Op<T1, op_cor>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);

  if (in_n_rows == 1)
    {
    // If the input is a row vector, we treat it as a column vector instead, giving a 1x1 correlation matrix.
    return 1;
    }
  else
    {
    return in_n_cols;
    }
  }
