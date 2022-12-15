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
