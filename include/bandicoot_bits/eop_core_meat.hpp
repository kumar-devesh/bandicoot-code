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
template<typename T1>
inline
void
eop_core<eop_type>::apply(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const unwrap<T1>   U(x.m);
  const Mat<eT>& A = U.M;

  kernel_id::enum_id kernel_num;

       if(is_same_type<eop_type, eop_scalar_plus      >::yes)  { kernel_num = kernel_id::equ_array_plus_scalar      ; }
  else if(is_same_type<eop_type, eop_neg              >::yes)  { kernel_num = kernel_id::equ_array_neg              ; }
  else if(is_same_type<eop_type, eop_scalar_minus_pre >::yes)  { kernel_num = kernel_id::equ_array_minus_scalar_pre ; }
  else if(is_same_type<eop_type, eop_scalar_minus_post>::yes)  { kernel_num = kernel_id::equ_array_minus_scalar_post; }
  else if(is_same_type<eop_type, eop_scalar_times     >::yes)  { kernel_num = kernel_id::equ_array_mul_scalar       ; }
  else if(is_same_type<eop_type, eop_scalar_div_pre   >::yes)  { kernel_num = kernel_id::equ_array_div_scalar_pre   ; }
  else if(is_same_type<eop_type, eop_scalar_div_post  >::yes)  { kernel_num = kernel_id::equ_array_div_scalar_post  ; }
  else if(is_same_type<eop_type, eop_square           >::yes)  { kernel_num = kernel_id::equ_array_square           ; }
  else if(is_same_type<eop_type, eop_sqrt             >::yes)  { kernel_num = kernel_id::equ_array_sqrt             ; }
  else if(is_same_type<eop_type, eop_exp              >::yes)  { kernel_num = kernel_id::equ_array_exp              ; }
  else if(is_same_type<eop_type, eop_log              >::yes)  { kernel_num = kernel_id::equ_array_log              ; }
  else { coot_debug_check(true, "fixme: unhandled eop_type"); }

  dev_mem_t<eT> out_dev_mem = out.get_dev_mem(false);
  dev_mem_t<eT>   A_dev_mem =   A.get_dev_mem(false);

  if (get_rt().backend == CUDA_BACKEND)
    {
    cuda::eop_scalar(out_dev_mem, A_dev_mem, out.get_n_elem(), x.aux, kernel_num);
    }
  else
    {
    opencl::eop_scalar(out_dev_mem, A_dev_mem, out.get_n_elem(), x.aux, kernel_num);
    }
  }



template<typename eop_type>
template<typename T1>
inline
void
eop_core<eop_type>::apply_inplace_plus(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "addition");

  const Mat<eT> tmp(x);

  out += tmp;
  }



template<typename eop_type>
template<typename T1>
inline
void
eop_core<eop_type>::apply_inplace_minus(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "subtraction");

  const Mat<eT> tmp(x);

  out -= tmp;
  }



template<typename eop_type>
template<typename T1>
inline
void
eop_core<eop_type>::apply_inplace_schur(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "element-wise multiplication");

  const Mat<eT> tmp(x);

  out %= tmp;
  }



template<typename eop_type>
template<typename T1>
inline
void
eop_core<eop_type>::apply_inplace_div(Mat<typename T1::elem_type>& out, const eOp<T1, eop_type>& x)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  coot_debug_assert_same_size(out.n_rows, out.n_cols, x.get_n_rows(), x.get_n_cols(), "element-wise division");

  const Mat<eT> tmp(x);

  out /= tmp;
  }



//! @}
