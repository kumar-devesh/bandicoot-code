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


//! \addtogroup fn_elem
//! @{


//
// square

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_square> >::result
square(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_square>(A);
  }



//
// sqrt

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_sqrt> >::result
sqrt(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_sqrt>(A);
  }


//
// exp

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_exp> >::result
exp(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_exp>(A);
  }


//
// exp2

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_exp2> >::result
exp2(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_exp2>(A);
  }


//
// exp10

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_exp10> >::result
exp10(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_exp10>(A);
  }


//
// trunc_exp

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_trunc_exp> >::result
trunc_exp(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_trunc_exp>(A);
  }


//
// log

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_log> >::result
log(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_log>(A);
  }



//
// log2

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_log2> >::result
log2(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_log2>(A);
  }



//
// log10

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_log10> >::result
log10(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_log10>(A);
  }



//
// trunc_log

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_trunc_log> >::result
trunc_log(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_trunc_log>(A);
  }



//
// cos

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_cos> >::result
cos(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_cos>(A);
  }



//
// sin

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_sin> >::result
sin(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_sin>(A);
  }



//
// tan

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_tan> >::result
tan(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_tan>(A);
  }



//
// acos

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_acos> >::result
acos(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_acos>(A);
  }



//
// asin

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_asin> >::result
asin(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_asin>(A);
  }



//
// atan

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_atan> >::result
atan(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_atan>(A);
  }



//
// cosh

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_cosh> >::result
cosh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_cosh>(A);
  }



//
// sinh

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_sinh> >::result
sinh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_sinh>(A);
  }



//
// tanh

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_tanh> >::result
tanh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_tanh>(A);
  }



//
// acosh

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_acosh> >::result
acosh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_acosh>(A);
  }



//
// asinh

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_asinh> >::result
asinh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_asinh>(A);
  }



//
// atanh

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_atanh> >::result
atanh(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_atanh>(A);
  }



//
// atan2

template<typename T1, typename T2>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value && is_coot_type<T2>::value && is_real<typename T1::elem_type>::value && is_real<typename T2::elem_type>::value, const eGlue<T1, T2, eglue_atan2> >::result
atan2(const T1& X, const T2& Y)
  {
  coot_extra_debug_sigprint();

  return eGlue<T1, T2, eglue_atan2>(X, Y);
  }



//
// hypot

template<typename T1, typename T2>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value && is_coot_type<T2>::value && is_real<typename T1::elem_type>::value && is_real<typename T2::elem_type>::value, const eGlue<T1, T2, eglue_hypot> >::result
hypot(const T1& X, const T2& Y)
  {
  coot_extra_debug_sigprint();

  return eGlue<T1, T2, eglue_hypot>(X, Y);
  }



//
// abs

template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value && std::is_signed<typename T1::elem_type>::value, const eOp<T1, eop_abs> >::result
abs(const T1& A)
  {
  coot_extra_debug_sigprint();

  return eOp<T1, eop_abs>(A);
  }



// abs(unsigned)... nothing to do
template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value && !std::is_signed<typename T1::elem_type>::value, const T1&>::result
abs(const T1& A)
  {
  coot_extra_debug_sigprint();

  return A;
  }



// abs(abs)... nothing to do
template<typename T1>
coot_warn_unused
coot_inline
typename enable_if2< is_coot_type<T1>::value, const eOp<T1, eop_abs>& >::result
abs(const eOp<T1, eop_abs>& A)
  {
  coot_extra_debug_sigprint();

  return A;
  }



// TODO: more element-wise functions



//! @}
