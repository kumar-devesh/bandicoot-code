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



template<typename T1>
inline
void
mtop_all::apply(Mat<uword>& out, const mtOp<uword, T1, mtop_all>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword;

  coot_debug_check( (dim > 1), "all(): parameter 'dim' must be 0 or 1" );

  unwrap<T1> U(in.q);

  // Shortcut if the input is empty.
  if (U.M.n_elem == 0)
    {
    if (dim == 0)
      {
      out.set_size(1, 0);
      }
    else
      {
      out.set_size(0, 1);
      }

    return;
    }

  apply_direct(out, U.M, dim);
  }



inline
void
mtop_all::apply_direct(Mat<uword>& out, const Mat<uword>& in, const uword dim)
  {
  coot_extra_debug_sigprint();

  if (&out == &in)
    {
    // For aliases, we have to output into a temporary matrix.
    Mat<uword> tmp;
    apply_direct(tmp, in, dim);
    out.steal_mem(tmp);
    return;
    }

  if (dim == 0)
    {
    out.set_size(1, in.n_cols);
    coot_rt_t::all(out.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols, uword(0), twoway_kernel_id::rel_all_neq_colwise, true);
    }
  else
    {
    out.set_size(in.n_rows, 1);
    coot_rt_t::all(out.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols, uword(0), twoway_kernel_id::rel_all_neq_rowwise, false);
    }
  }



template<typename eT>
inline
void
mtop_all::apply_direct(Mat<uword>& out, const Mat<eT>& in, const uword dim)
  {
  coot_extra_debug_sigprint();

  if (dim == 0)
    {
    out.set_size(1, in.n_cols);
    coot_rt_t::all(out.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols, eT(0), twoway_kernel_id::rel_all_neq_colwise, true);
    }
  else
    {
    out.set_size(in.n_rows, 1);
    coot_rt_t::all(out.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols, eT(0), twoway_kernel_id::rel_all_neq_rowwise, false);
    }
  }



template<typename eT>
inline
void
mtop_all::apply_direct(Mat<uword>& out, const subview<eT>& in, const uword dim)
  {
  coot_extra_debug_sigprint();

  // Subviews must be extracted beforehand, and then we use the regular Mat implementation.
  Mat<eT> tmp(in);
  apply_direct(out, tmp, dim);
  }



template<typename T1>
inline
bool
mtop_all::all_vec(T1& X)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;
  unwrap<T1> U(X);

  return coot_rt_t::all_vec(U.M.get_dev_mem(false), U.M.n_elem, eT(0), twoway_kernel_id::rel_all_neq, twoway_kernel_id::rel_all_neq_small);
  }



template<typename out_eT, typename T1>
inline
bool
mtop_all::all_vec(const mtOp<out_eT, T1, mtop_all>& X)
  {
  coot_extra_debug_sigprint();

  // Apply to inner operation.
  return all_vec(X.q);
  }



template<typename out_eT, typename T1>
inline
uword
mtop_all::compute_n_rows(const mtOp<out_eT, T1, mtop_all>& op, const uword in_n_rows, const uword in_n_cols)
  {
  return (op.aux_uword == 0) ? 1 : in_n_rows;
  }



template<typename out_eT, typename T1>
inline
uword
mtop_all::compute_n_cols(const mtOp<out_eT, T1, mtop_all>& op, const uword in_n_rows, const uword in_n_cols)
  {
  return (op.aux_uword == 0) ? in_n_cols : 1;
  }
