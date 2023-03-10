// Copyright 2023 Ryan Curtin (http://www.ratml.org)
//
// SPDX-License-Identifier: Apache-2.0
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
op_diagmat::apply(Mat<out_eT>& out, const Op<T1, op_diagmat>& in)
  {
  coot_extra_debug_sigprint();

  // If the types are not the same, we have to force a conversion.
  if (std::is_same<out_eT, typename T1::elem_type>::value)
    {
    unwrap<T1> U(in.m);
    op_diagmat::apply_direct(out, U.M);
    }
  else
    {
    mtOp<out_eT, T1, mtop_conv_to> mtop(in.m);
    unwrap<mtOp<out_eT, T1, mtop_conv_to>> U(mtop);
    op_diagmat::apply_direct(out, U.M);
    }
  }



template<typename eT>
inline
void
op_diagmat::apply_direct(Mat<eT>& out, const Mat<eT>& in)
  {
  coot_extra_debug_sigprint();

  out.zeros(in.n_elem, in.n_elem);
  out.diag() = in;
  }



template<typename eT>
inline
void
op_diagmat::apply_direct(Mat<eT>& out, const subview<eT>& in)
  {
  coot_extra_debug_sigprint();

  // Subviews must be extracted.
  Mat<eT> tmp(in);
  out.zeros(tmp.n_elem, tmp.n_elem);
  out.diag() = tmp;
  }



template<typename T1>
inline
uword
op_diagmat::compute_n_rows(const Op<T1, op_diagmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);

  return (std::max)(in_n_rows, in_n_cols);
  }



template<typename T1>
inline
uword
op_diagmat::compute_n_cols(const Op<T1, op_diagmat>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);

  return (std::max)(in_n_rows, in_n_cols);
  }



template<typename out_eT, typename T1>
inline
void
op_diagmat2::apply(Mat<out_eT>& out, const Op<T1, op_diagmat2>& in)
  {
  coot_extra_debug_sigprint();

  const sword k = (in.aux_uword_b == 0) ? in.aux_uword_a : (-sword(in.aux_uword_a));

  // If the types are not the same, we have to force a conversion.
  if (std::is_same<out_eT, typename T1::elem_type>::value)
    {
    unwrap<T1> U(in.m);
    op_diagmat2::apply_direct(out, U.M, k);
    }
  else
    {
    mtOp<out_eT, T1, mtop_conv_to> mtop(in.m);
    unwrap<mtOp<out_eT, T1, mtop_conv_to>> U(mtop);
    op_diagmat2::apply_direct(out, U.M, k);
    }
  }



template<typename eT>
inline
void
op_diagmat2::apply_direct(Mat<eT>& out, const Mat<eT>& in, const sword k)
  {
  coot_extra_debug_sigprint();

  out.zeros(in.n_elem + std::abs(k), in.n_elem + std::abs(k));
  out.diag(k) = in;
  }



template<typename eT>
inline
void
op_diagmat2::apply_direct(Mat<eT>& out, const subview<eT>& in, const sword k)
  {
  coot_extra_debug_sigprint();

  // Subviews must be extracted.
  Mat<eT> tmp(in);
  out.zeros(tmp.n_elem + std::abs(k), tmp.n_elem + std::abs(k));
  out.diag(k) = tmp;
  }



template<typename T1>
inline
uword
op_diagmat2::compute_n_rows(const Op<T1, op_diagmat2>& op, const uword in_n_rows, const uword in_n_cols)
  {
  return (std::max)(in_n_rows, in_n_cols) + op.aux_uword_a;
  }



template<typename T1>
inline
uword
op_diagmat2::compute_n_cols(const Op<T1, op_diagmat2>& op, const uword in_n_rows, const uword in_n_cols)
  {
  return (std::max)(in_n_rows, in_n_cols) + op.aux_uword_a;
  }
