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
op_median::apply(Mat<out_eT>& out, const Op<T1, op_median>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  unwrap<T1> U(in.m);
  // The kernels we have don't operate on subviews, or aliases.
  extract_subview<typename T1::stored_type> E(U.M);
  copy_alias<eT> C(E.M);

  const uword dim = op.aux_uword_a;
  apply_direct(out, C.M, dim);
  }



template<typename eT, typename T1>
inline
void
op_median::apply(Mat<eT>& out, const Op<mtOp<eT, T1, mtop_conv_to>, op_median>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type in_eT;

  unwrap<T1> U(in.m.q);
  extract_subview<typename T1::stored_type> E(U.M);
  // Aliases aren't possible for a type change.

  const uword dim = op.aux_uword_a;
  apply_direct(out, E.M, dim);
  }



template<typename out_eT, typename in_eT>
inline
void
op_median::apply_direct(Mat<out_eT>& out, const Mat<in_eT>& in, const uword dim)
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

  coot_rt_t::median(out.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols, dim);
  }



template<typename T1>
inline
typename T1::elem_type
op_median::median_all(const T1& X)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X.get_ref());
  const Mat<eT>& M = U.M;

  if (M.n_elem == 0)
    {
    return eT(0);
    }

  return coot_rt_t::median_vec(M.get_dev_mem(false), M.n_elem);
  }



template<typename T1>
inline
uword
op_median::compute_n_rows(const Op<T1, op_median>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);

  const uword dim = op.aux_uword_a;
  if (dim == 0)
    {
    return std::min(in_n_rows, 1); // either 0 or 1
    }
  else
    {
    return in_n_rows;
    }
  }



template<typename T1>
inline
uword
op_median::compute_n_cols(const Op<T1, op_median>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);

  const uword dim = op.aux_uword_b;
  if (dim == 0)
    {
    return in_n_cols;
    }
  else
    {
    return std::min(in_n_cols, 1); // either 0 or 1
    }
  }
