// Copyright 2020 Ryan Curtin (http://www.ratml.org
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
mtop_conv_to::apply(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  // Unwrap the inner operation fully.
  const unwrap<T1> U(X.m.Q);

  arrayops::copy(out.get_dev_mem(false), U.M.get_dev_mem(false), U.M.n_elem);
  }



template<typename out_eT, typename in_eT>
inline
void
mtop_conv_to::apply(Mat<out_eT>& out, const mtOp<out_eT, subview<in_eT>, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const subview<in_eT>& U(X.m.Q);

  arrayops::copy_subview(out.get_dev_mem(false), U.m.get_dev_mem(false), U.aux_row1, U.aux_col1, U.m.n_rows, U.m.n_cols, U.n_rows, U.n_cols);
  }




template<typename out_eT, typename T1, typename eop_type>
inline
void
mtop_conv_to::apply(Mat<out_eT>& out, const mtOp<out_eT, eOp<T1, eop_type>, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  // Apply the operation specifically into the different output type.
  eop_type::apply(out, X.m.Q);
  }



template<typename out_eT, typename T1, typename T2, typename eglue_type>
inline
void
mtop_conv_to::apply(Mat<out_eT>& out, const mtOp<out_eT, eGlue<T1, T2, eglue_type>, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  // Apply the operation specifically into the different output type.
  eglue_type::apply(out, X.m.Q);
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_plus(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(X.m.Q);

  arrayops::inplace_plus_array(out.get_dev_mem(false), U.M.get_dev_mem(false), U.M.n_elem);
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_minus(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(X.m.Q);

  arrayops::inplace_minus_array(out.get_dev_mem(false), U.M.get_dev_mem(false), U.M.n_elem);
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_times(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  // We have to actually perform the conversion here.
  Mat<out_eT> converted(X.m.Q);
  Mat<out_eT> tmp(out);
  tmp *= converted;

  out.steal_mem(tmp);
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_schur(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(X.m.Q);

  arrayops::inplace_mul_array(out.get_dev_mem(false), U.M.get_dev_mem(false), U.M.n_elem);
  }



template<typename out_eT, typename T1>
inline
void
mtop_conv_to::apply_inplace_div(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_conv_to>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(X.m.Q);

  arrayops::inplace_div_array(out.get_dev_mem(false), U.M.get_dev_mem(false), U.M.n_elem);
  }
