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



template<typename out_eT, typename T1>
inline
void
op_var::apply(Mat<out_eT>& out, const Op<T1, op_var>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  unwrap<T1> U(in.m);
  // The kernels we have don't operate on subviews, or aliases.
  extract_subview<typename T1::stored_type> E(U.M);
  copy_alias<eT> C(E.M, out);

  const uword dim = in.aux_uword_a;
  const uword norm_type = in.aux_uword_b;
  apply_direct(out, C.M, dim, norm_type);
  }



template<typename out_eT, typename in_eT>
inline
void
op_var::apply_direct(Mat<out_eT>& out, const Mat<in_eT>& in, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  // We require a conversion of the input matrix because we do not have specialized kernels that combine a type conversion and variance computation.
  Mat<out_eT> tmp(in.n_rows, in.n_cols);
  coot_rt_t::copy_array(tmp.get_dev_mem(false), in.get_dev_mem(false), in.n_rows * in.n_cols);
  apply_direct(out, tmp, dim, norm_type);
  }



template<typename eT>
inline
void
op_var::apply_direct(Mat<eT>& out, const Mat<eT>& in, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  if (dim == 0)
    {
    out.set_size(in.n_rows > 0 ? 1 : 0, in.n_cols);
    }
  else
    {
    out.set_size(in.n_rows, in.n_cols > 0 ? 1 : 0);
    }

  // Shortcut: if we don't need to do anything... don't do anything.
  if (out.n_elem == 0)
    {
    return;
    }

  // First we need to compute the mean.
  Mat<eT> mean;
  op_mean::apply_direct(mean, in, dim, false);

  coot_rt_t::var(out.get_dev_mem(false), in.get_dev_mem(false), mean.get_dev_mem(false), in.n_rows, in.n_cols, dim, norm_type);
  }



template<typename out_eT, typename in_eT>
inline
void
op_var::apply_direct(Mat<out_eT>& out, const subview<in_eT>& in, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  // This will require a conversion, so just extract the subview.
  Mat<out_eT> tmp(in);
  apply_direct(out, tmp, dim, norm_type);
  }



template<typename eT>
inline
void
op_var::apply_direct(Mat<eT>& out, const subview<eT>& in, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  if (dim == 0)
    {
    out.set_size(in.n_rows > 0 ? 1 : 0, in.n_cols);
    }
  else
    {
    out.set_size(in.n_rows, in.n_cols > 0 ? 1 : 0);
    }

  // Shortcut: if we don't need to do anything... don't do anything.
  if (out.n_elem == 0)
    {
    return;
    }

  // First, compute the mean.
  Mat<eT> mean;
  op_mean::apply_direct(out, in, dim, false);

  coot_rt_t::var_subview(out.get_dev_mem(false), in.m.get_dev_mem(false), mean.get_dev_mem(false), in.m.n_rows, in.m.n_cols, in.aux_row1, in.aux_col1, in.n_rows, in.n_cols, dim, norm_type);
  }



template<typename T1>
inline
typename T1::elem_type
op_var::var_vec(const T1& X, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.get_ref());
  if (U.M.n_elem == 0)
    {
    return eT(0);
    }

  const eT mean_val = op_mean::mean_all_direct(U.M);
  var_vec_direct(U.M, mean_val);
  }



template<typename eT>
inline
eT
op_var::var_vec_direct(const Mat<eT>& M, const eT mean_val, const uword norm_type)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::var_vec(M.get_dev_mem(false), mean_val, M.n_elem, norm_type);
  }



template<typename eT>
inline
eT
op_var::var_vec_direct(const subview<eT>& M, const eT mean_val, const uword norm_type)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::var_vec_subview(M.m.get_dev_mem(false), mean_val, M.m.n_rows, M.m.n_cols, M.aux_row1, M.aux_col1, M.n_rows, M.n_cols, norm_type);
  }



template<typename T1>
inline
uword
op_var::compute_n_rows(const Op<T1, op_var>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);

  const uword dim = op.aux_uword_a;
  if (dim == 0)
    {
    return std::min(in_n_rows, uword(1)); // either 0 or 1
    }
  else
    {
    return in_n_rows;
    }
  }



template<typename T1>
inline
uword
op_var::compute_n_cols(const Op<T1, op_var>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);

  const uword dim = op.aux_uword_b;
  if (dim == 0)
    {
    return in_n_cols;
    }
  else
    {
    return std::min(in_n_cols, uword(1)); // either 0 or 1
    }
  }
