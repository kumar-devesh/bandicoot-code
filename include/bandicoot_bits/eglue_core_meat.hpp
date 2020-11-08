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


//! \addtogroup eglue_core
//! @{



//
// matrices



template<typename eglue_type>
template<typename eT3, typename T1, typename T2>
inline
void
eglue_core<eglue_type>::apply(Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x)
  {
  coot_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT1;
  typedef typename T2::elem_type eT2;
  
  const unwrap<typename SizeProxy<T1>::stored_type> UA(x.A.Q);
  const unwrap<typename SizeProxy<T2>::stored_type> UB(x.B.Q);
  
  const Mat<eT1>& A = UA.M;
  const Mat<eT2>& B = UB.M;
  
  if(is_same_type<eglue_type, eglue_plus >::yes)
    {
    coot_rt_t::array_op(out.get_dev_mem(false), out.n_elem, A.get_dev_mem(false), B.get_dev_mem(false), threeway_kernel_id::equ_array_plus_array);
    }
  else if(is_same_type<eglue_type, eglue_minus>::yes)
    {
    coot_rt_t::array_op(out.get_dev_mem(false), out.n_elem, A.get_dev_mem(false), B.get_dev_mem(false), threeway_kernel_id::equ_array_minus_array);
    }
  else if(is_same_type<eglue_type, eglue_div  >::yes)
    {
    coot_rt_t::array_op(out.get_dev_mem(false), out.n_elem, A.get_dev_mem(false), B.get_dev_mem(false), threeway_kernel_id::equ_array_div_array);
    }
  else if(is_same_type<eglue_type, eglue_schur>::yes)
    {
    coot_rt_t::array_op(out.get_dev_mem(false), out.n_elem, A.get_dev_mem(false), B.get_dev_mem(false), threeway_kernel_id::equ_array_mul_array);
    }
  }



template<typename eglue_type>
template<typename eT3, typename T1, typename T2>
inline
void
eglue_core<eglue_type>::apply_inplace_plus(Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x)
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution
  // TODO: replace with code that uses dedicated kernels
  
  const Mat<eT3> tmp(x);
  
  out += tmp;
  }



template<typename eglue_type>
template<typename eT3, typename T1, typename T2>
inline
void
eglue_core<eglue_type>::apply_inplace_minus(Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x)
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution
  // TODO: replace with code that uses dedicated kernels
  
  const Mat<eT3> tmp(x);
  
  out -= tmp;
  }



template<typename eglue_type>
template<typename eT3, typename T1, typename T2>
inline
void
eglue_core<eglue_type>::apply_inplace_schur(Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x)
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution
  // TODO: replace with code that uses dedicated kernels
  
  const Mat<eT3> tmp(x);
  
  out %= tmp;
  }



template<typename eglue_type>
template<typename eT3, typename T1, typename T2>
inline
void
eglue_core<eglue_type>::apply_inplace_div(Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x)
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution
  // TODO: replace with code that uses dedicated kernels
  
  const Mat<eT3> tmp(x);
  
  out /= tmp;
  }



//! @}
