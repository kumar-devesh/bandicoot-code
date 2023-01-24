// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2021 Marcus Edel (http://kurg.org)
// Copyright 2023 Ryan Curtin (http://ratml.org)
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
op_clamp::apply(Mat<out_eT>& out, const Op<T1, op_clamp>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const eT min_val = in.aux;
  const eT max_val = in.aux_b;

  coot_debug_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );

  const unwrap<T1> U(in.m);

  op_clamp::apply_direct(out, U.M, min_val, max_val);
  }



template<typename eT>
inline
void
op_clamp::apply_direct(Mat<eT>& out, const Mat<eT>& X, const eT min_val, const eT max_val)
  {
  coot_extra_debug_sigprint();

  if (X.n_elem == 0)
    {
    out.set_size(X.n_rows, X.n_cols);
    return;
    }

  if (&out == &X)
    {
    Mat<eT> tmp;
    tmp.set_size(X.n_rows, X.n_cols);
    coot_rt_t::clamp(tmp.get_dev_mem(false), X.get_dev_mem(false), min_val, max_val, X.n_elem);
    out.steal_mem(tmp);
    }
  else
    {
    out.set_size(X.n_rows, X.n_cols);
    coot_rt_t::clamp(out.get_dev_mem(false), X.get_dev_mem(false), min_val, max_val, X.n_elem);
    }
  }



template<typename out_eT, typename in_eT>
inline
void
op_clamp::apply_direct(Mat<out_eT>& out, const Mat<in_eT>& X, const in_eT min_val, const in_eT max_val)
  {
  coot_extra_debug_sigprint();

  out.set_size(X.n_rows, X.n_cols);
  if (X.n_elem == 0)
    {
    return;
    }

  coot_rt_t::clamp(out.get_dev_mem(false), X.get_dev_mem(false), min_val, max_val, X.n_elem);
  }



template<typename out_eT, typename in_eT>
inline
void
op_clamp::apply_direct(Mat<out_eT>& out, const subview<in_eT>& sv, const in_eT min_val, const in_eT max_val)
  {
  coot_extra_debug_sigprint();

  out.set_size(sv.n_rows, sv.n_cols);
  if (sv.n_elem == 0)
    {
    return;
    }

  // TODO: this could be improved: we don't do specialized subview clamping, so instead we extract the subview.
  Mat<in_eT> extracted_sv(sv);
  op_clamp::apply_direct(out, extracted_sv, min_val, max_val);
  }
