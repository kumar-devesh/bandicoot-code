// Copyright 2022 Gopi Tatiraju
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



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  (is_coot_type<T1>::value && is_coot_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  const Glue<T1, T2, glue_join_cols>
  >::result
join_cols
  (
  const T1& A,
  const T2& B
  )
  {
  coot_extra_debug_sigprint();

  return Glue<T1, T2, glue_join_cols>(A, B, 0);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  (is_coot_type<T1>::value && is_coot_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  const Glue<T1, T2, glue_join_cols>
  >::result
join_vert
  (
  const T1& A,
  const T1& B
  )
  {
  coot_extra_debug_sigprint();

  return Glue<T1, T2, glue_join_cols>(A, B, 1);
  }



template<typename eT, typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  (is_coot_type<T1>::value && is_coot_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  const Glue<T1, T2, glue_join_rows>
  >::result
join_rows
  (
  const Base<eT, T1>& A,
  const Base<eT, T2>& B
  )
  {
  coot_extra_debug_sigprint();

  return Glue<T1, T2, glue_join_rows>(A, B, 0);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename
enable_if2
  <
  (is_coot_type<T1>::value && is_coot_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value),
  const Glue<T1, T2, glue_join_rows>
  >::result
join_horiz
  {
  const T1& A,
  const T2& B
  )
  {
  coot_extra_debug_sigprint();

  return Glue<T1, T2, glue_join_rows>(A, B, 1);
  }
