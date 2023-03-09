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



template<typename out_eT, typename T1>
inline
void
mtop_all::apply(Mat<out_eT>& out, const mtOp<out_eT, T1, mtop_all>& in)
  {
  coot_extra_debug_sigprint();

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
