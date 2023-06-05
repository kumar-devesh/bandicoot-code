// Copyright 2023 Ryan Curtin (https://www.ratml.org/)
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



template<typename eT2, typename T1>
inline
void
op_pinv::apply(Mat<eT2>& out, const Op<T1, op_pinv>& in)
  {

  }



template<typename eT2, typename eT1>
inline
void
op_pinv::apply_direct(Mat<eT2>& out, const Mat<eT1>& in)
  {

  }



template<typename eT2, typename eT1>
inline
void
op_pinv::apply_direct_diag(Mat<eT2>& out, const Mat<eT1>& in)
  {
  coot_extra_debug_sigprint();

  // To take the psuedoinverse of a diagonal matrix, we just need to invert the elements.
  out.zeros(in.n_rows, in.n_cols);

  coot_rt_t::extract_diag(out.get_dev_mem(false), in.get_dev_mem(false), 0, in.n_rows, (std::min)(in.n_rows, in.n_cols));
  coot_rt_t::inplace_op_diag(out.get_dev_mem(false), 0, (eT) 1, out.n_rows, (std::min)(out.n_rows, out.n_cols), oneway_kernel_id::diag_inplace_div_scalar_pre);

  // TODO: now check for any NaNs to indicate success or failure
  }



template<typename eT2, typename eT1>
inline
void
op_pinv::apply_direct_sym(Mat<eT2>& out, const Mat<eT1>& in)
  {

  }



template<typename eT2, typename eT1>
inline
void
op_pinv::apply_direct_gen(Mat<eT2>& out, const Mat<eT1>& in)
  {

  }



template<typename T1>
inline
uword
op_pinv::compute_n_rows(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_pinv::compute_n_cols(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }

  };
