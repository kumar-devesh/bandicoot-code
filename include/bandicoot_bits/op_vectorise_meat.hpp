// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2021-2022 Marcus Edel (http://kurg.org)
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
op_vectorise_col::apply(Mat<out_eT>& out, const Op<T1,op_vectorise_col>& in)
  {
  coot_extra_debug_sigprint();

  op_vectorise_col::apply_direct(out, in.m);
  }



template<typename out_eT, typename T1>
inline
void
op_vectorise_col::apply_direct(Mat<out_eT>& out, const T1& expr)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(expr);

  if (U.M.n_elem == 0)
    {
    out.set_size(0, 1);
    return;
    }

  if(U.is_alias(out))
    {
    // output matrix is the same as the input matrix
    out.set_size(out.n_elem, 1);  // set_size() doesn't destroy data as long as the number of elements in the matrix remains the same
    }
  else
    {
    out.set_size(U.M.n_elem, 1);
    arrayops::copy(out.get_dev_mem(false), U.M.get_dev_mem(false), U.M.n_elem);
    }
  }



template<typename out_eT, typename eT>
inline
void
op_vectorise_col::apply_direct(Mat<out_eT>& out, const subview<eT>& sv)
  {
  coot_extra_debug_sigprint();

  if(&out == &(sv.m))
    {
    Mat<out_eT> tmp(sv.n_elem, 1);
    arrayops::copy_subview(tmp.get_dev_mem(false), sv.m.get_dev_mem(false), sv.aux_row1, sv.aux_col1, sv.m.n_rows, sv.m.n_cols, sv.n_rows, sv.n_cols);
    out.steal_mem(tmp);
    }
  else
    {
    out.set_size(sv.n_elem, 1);
    arrayops::copy_subview(out.get_dev_mem(false), sv.m.get_dev_mem(false), sv.aux_row1, sv.aux_col1, sv.m.n_rows, sv.m.n_cols, sv.n_rows, sv.n_cols);
    }
  }



template<typename T1>
inline
uword
op_vectorise_col::compute_n_rows(const Op<T1, op_vectorise_col>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  return in_n_rows * in_n_cols;
  }



template<typename T1>
inline
uword
op_vectorise_col::compute_n_cols(const Op<T1, op_vectorise_col>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);
  return 1;
  }



template<typename out_eT, typename T1>
inline
void
op_vectorise_all::apply(Mat<out_eT>& out, const Op<T1,op_vectorise_all>& in)
  {
  coot_extra_debug_sigprint();

  if (in.aux_uword_a == 0)
    {
    op_vectorise_col::apply_direct(out, in.m);
    }
  else
    {
    // See if we can use op_vectorise_col anyway, which we can do if the object is already a vector.
    SizeProxy<T1> S(in.m);
    if (S.get_n_rows() == 1 || S.get_n_cols() == 1)
      {
      op_vectorise_col::apply_direct(out, in.m);
      }
    else
      {
      op_vectorise_row::apply_direct(out, in.m);
      }
    }
  }



template<typename T1>
inline
uword
op_vectorise_all::compute_n_rows(const Op<T1, op_vectorise_all>& op, const uword in_n_rows, const uword in_n_cols)
  {
  if (op.aux_uword_a == 0)
    return in_n_rows * in_n_cols;
  else
    return 1;
  }



template<typename T1>
inline
uword
op_vectorise_all::compute_n_cols(const Op<T1, op_vectorise_all>& op, const uword in_n_rows, const uword in_n_cols)
  {
  if (op.aux_uword_a == 0)
    return 1;
  else
    return in_n_rows * in_n_cols;
  }



template<typename out_eT, typename T1>
inline
void
op_vectorise_row::apply(Mat<out_eT>& out, const Op<T1,op_vectorise_row>& in)
  {
  coot_extra_debug_sigprint();

  op_vectorise_row::apply_direct(out, in.m);
  }



template<typename T1>
inline
void
op_vectorise_row::apply_direct(Mat<typename T1::elem_type>& out, const T1& expr)
  {
  coot_extra_debug_sigprint();

  // Row-wise vectorisation is equivalent to a transpose followed by a vectorisation.

  // TODO: select htrans/strans based on complex elements or not
  // Using op_htrans as part of the unwrap may combine the htrans with some earlier operations in the expression.
  unwrap<Op<T1, op_htrans>> U(Op<T1, op_htrans>(expr, 0, 0));

  // If U.M is an object we created during unwrapping, steal the memory and set the size.
  // Otherwise, copy U.M.
  if (is_Mat<T1>::value)
    {
    // If `expr` is some type of matrix, then unwrap<T1> just stores the matrix itself.
    // That's not a temporary, and we can't steal its memory---we have to copy it.
    out.set_size(1, U.M.n_elem);
    arrayops::copy(out.get_dev_mem(false), U.M.get_dev_mem(false), U.M.n_elem);
    }
  else
    {
    // We must have created a temporary matrix to perform the operation, and so we can just steal its memory.
    const uword new_n_rows = U.M.n_elem;
    out.steal_mem(U.M);
    out.set_size(1, new_n_rows);
    }
  }



template<typename eT>
inline
void
op_vectorise_row::apply_direct(Mat<eT>& out, const subview<eT>& sv)
  {
  coot_extra_debug_sigprint();

  // If `expr` is a subview, we have to extract the subview.
  if(&out == &(sv.m))
    {
    Mat<eT> tmp(1, sv.n_elem);
    arrayops::copy_subview(tmp.get_dev_mem(false), sv.m.get_dev_mem(false), sv.aux_row1, sv.aux_col1, sv.m.n_rows, sv.m.n_cols, sv.n_rows, sv.n_cols);
    out.steal_mem(tmp);
    }
  else
    {
    out.set_size(1, sv.n_elem);
    arrayops::copy_subview(out.get_dev_mem(false), sv.m.get_dev_mem(false), sv.aux_row1, sv.aux_col1, sv.m.n_rows, sv.m.n_cols, sv.n_rows, sv.n_cols);
    }
  }



template<typename out_eT, typename T1>
inline
void
op_vectorise_row::apply_direct(Mat<out_eT>& out, const T1& expr, const typename enable_if<!std::is_same<out_eT, typename T1::elem_type>::value>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  // Row-wise vectorisation is equivalent to a transpose followed by a vectorisation.

  // TODO: select htrans/strans based on complex elements or not
  // Using op_htrans as part of the unwrap may combine the htrans with some earlier operations in the expression.
  unwrap<Op<T1, op_htrans>> U(Op<T1, op_htrans>(expr, 0, 0));

  // A conversion operation is always necessary when the type is different.
  out.set_size(1, U.M.n_elem);
  arrayops::copy(out.get_dev_mem(false), U.M.get_dev_mem(false), U.M.n_elem);
  }



template<typename out_eT, typename eT>
inline
void
op_vectorise_row::apply_direct(Mat<out_eT>& out, const subview<eT>& sv, const typename enable_if<!std::is_same<out_eT, eT>::value>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  out.set_size(1, sv.n_elem);
  arrayops::copy_subview(out.get_dev_mem(false), sv.m.get_dev_mem(false), sv.aux_row1, sv.aux_col1, sv.m.n_rows, sv.m.n_cols, sv.n_rows, sv.n_cols);
  }



template<typename T1>
inline
uword
op_vectorise_row::compute_n_rows(const Op<T1, op_vectorise_row>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);
  return 1;
  }



template<typename T1>
inline
uword
op_vectorise_row::compute_n_cols(const Op<T1, op_vectorise_row>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  return in_n_rows * in_n_cols;
  }
