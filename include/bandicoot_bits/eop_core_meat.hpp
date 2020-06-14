// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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


//! \addtogroup eop_core
//! @{


//
// matrices



template<typename eop_type>
template<typename eT2, typename T1>
inline
void
eop_core<eop_type>::apply(Mat<eT2>& out, const eOp<eT2, T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT1;

  const unwrap<typename SizeProxy<T1>::stored_type> U(x.m.Q);
  const Mat<eT1>& A = U.M;

  twoway_kernel_id::enum_id kernel_num;

       if(is_same_type<eop_type, eop_scalar_plus      >::yes)  { kernel_num = twoway_kernel_id::equ_array_plus_scalar      ; }
  else if(is_same_type<eop_type, eop_neg              >::yes)  { kernel_num = twoway_kernel_id::equ_array_neg              ; }
  else if(is_same_type<eop_type, eop_scalar_minus_pre >::yes)  { kernel_num = twoway_kernel_id::equ_array_minus_scalar_pre ; }
  else if(is_same_type<eop_type, eop_scalar_minus_post>::yes)  { kernel_num = twoway_kernel_id::equ_array_minus_scalar_post; }
  else if(is_same_type<eop_type, eop_scalar_times     >::yes)  { kernel_num = twoway_kernel_id::equ_array_mul_scalar       ; }
  else if(is_same_type<eop_type, eop_scalar_div_pre   >::yes)  { kernel_num = twoway_kernel_id::equ_array_div_scalar_pre   ; }
  else if(is_same_type<eop_type, eop_scalar_div_post  >::yes)  { kernel_num = twoway_kernel_id::equ_array_div_scalar_post  ; }
  else if(is_same_type<eop_type, eop_square           >::yes)  { kernel_num = twoway_kernel_id::equ_array_square           ; }
  else if(is_same_type<eop_type, eop_sqrt             >::yes)  { kernel_num = twoway_kernel_id::equ_array_sqrt             ; }
  else if(is_same_type<eop_type, eop_exp              >::yes)  { kernel_num = twoway_kernel_id::equ_array_exp              ; }
  else if(is_same_type<eop_type, eop_log              >::yes)  { kernel_num = twoway_kernel_id::equ_array_log              ; }
  else { coot_debug_check(true, "fixme: unhandled eop_type"); }

  dev_mem_t<eT2> out_dev_mem = out.get_dev_mem(false);
  dev_mem_t<typename T1::elem_type>   A_dev_mem =   A.get_dev_mem(false);

  typedef get_default<eop_type> get_default_type;

  const eT1 aux_in  = x.use_aux_in  ? x.aux_in  : get_default_type::template val<eT1>();
  const eT2 aux_out = x.use_aux_out ? x.aux_out : get_default_type::template val<eT2>();

  // There is one singular exception: if the op is eop_scalar_div_post and use_aux_in and use_aux_out are both true,
  // and both aux_in and aux_out are eT(0), we have to apply sequentially.
  if (is_same_type<eop_type, eop_scalar_div_post>::value && x.use_aux_in && x.use_aux_out && x.aux_in == eT1(0) && x.aux_out == eT2(0))
    {
    Mat<eT1> tmp(A.n_rows, A.n_cols);
    coot_rt_t::eop_scalar(tmp.get_dev_mem(), A_dev_mem, tmp.get_n_elem(), aux_in, eT1(0), kernel_num);
    coot_rt_t::eop_scalar(out_dev_mem, tmp.get_dev_mem(), out.get_n_elem(), eT1(0), aux_out, kernel_num);

    return;
    }

  coot_rt_t::eop_scalar(out_dev_mem, A_dev_mem, out.get_n_elem(), aux_in, aux_out, kernel_num);
  }



template<typename eop_type>
template<typename eT2, typename T1>
inline
void
eop_core<eop_type>::apply_inplace_plus(Mat<eT2>& out, const eOp<eT2, T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "addition");

  const Mat<eT2> tmp(x);

  out += tmp;
  }



template<typename eop_type>
template<typename eT2, typename T1>
inline
void
eop_core<eop_type>::apply_inplace_minus(Mat<eT2>& out, const eOp<eT2, T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "subtraction");

  const Mat<eT2> tmp(x);

  out -= tmp;
  }



template<typename eop_type>
template<typename eT2, typename T1>
inline
void
eop_core<eop_type>::apply_inplace_schur(Mat<eT2>& out, const eOp<eT2, T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "element-wise multiplication");

  const Mat<eT2> tmp(x);

  out %= tmp;
  }



template<typename eop_type>
template<typename eT2, typename T1>
inline
void
eop_core<eop_type>::apply_inplace_div(Mat<eT2>& out, const eOp<eT2, T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "element-wise division");

  const Mat<eT2> tmp(x);

  out /= tmp;
  }



//! @}
