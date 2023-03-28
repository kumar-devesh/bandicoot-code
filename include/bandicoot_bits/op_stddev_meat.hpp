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
op_stddev::apply(Mat<out_eT>& out, const Op<T1, op_stddev>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  unwrap<T1> U(in.m);
  // The kernels we have don't operate on subviews, or aliases.
  extract_subview<typename T1::stored_type> E(U.M);
  copy_alias<eT> C(E.M);

  const uword dim = in.aux_uword_a;
  const uword norm_type = in.aux_uword_b;
  // First compute the variance.
  op_var::apply_direct(out, C.M, dim, norm_type);
  // Now take the square root.
  coot_rt_t::eop_scalar(out.get_dev_mem(false), out.get_dev_mem(false), out_eT(0), out_eT(0), twoway_kernel_id::equ_array_sqrt_pre);
  }
