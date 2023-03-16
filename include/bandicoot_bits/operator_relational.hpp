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



// <  : lt
// >  : gt
// <= : lteq
// >= : gteq
// == : eq
// != : noteq



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_lt_pre>
  >::result
operator<
  (
  const typename T1::elem_type val,
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_lt_pre>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_lt_post>
  >::result
operator<
  (
  const T1& X,
  const typename T1::elem_type val
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_lt_post>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_gt_pre>
  >::result
operator>
  (
  const typename T1::elem_type val,
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_gt_pre>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_gt_post>
  >::result
operator>
  (
  const T1& X,
  const typename T1::elem_type val
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_gt_post>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_lteq_pre>
  >::result
operator<=
  (
  const typename T1::elem_type val,
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_lteq_pre>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_lteq_post>
  >::result
operator<=
  (
  const T1& X,
  const typename T1::elem_type val
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_lteq_post>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_gteq_pre>
  >::result
operator>=
  (
  const typename T1::elem_type val,
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_gteq_pre>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_gteq_post>
  >::result
operator>=
  (
  const T1& X,
  const typename T1::elem_type val
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_gteq_post>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_eq>
  >::result
operator==
  (
  const T1& X,
  const typename T1::elem_type val
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_eq>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_eq>
  >::result
operator==
  (
  const typename T1::elem_type val,
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_eq>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_noteq>
  >::result
operator!=
  (
  const T1& X,
  const typename T1::elem_type val
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_noteq>(X, val);
  }



template<typename T1>
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const mtOp<uword, T1, mtop_rel_noteq>
  >::result
operator!=
  (
  const typename T1::elem_type val,
  const T1& X
  )
  {
  coot_extra_debug_sigprint();

  return mtOp<uword, T1, mtop_rel_noteq>(X, val);
  }
