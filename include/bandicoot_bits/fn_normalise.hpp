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



template<typename T1>
coot_warn_unused
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value && resolves_to_vector<T1>::value == true,
  const Op<T1, op_normalise_vec>
  >::result
normalise
  (
  const T1& X,
  const uword p = uword(2),
  const coot_empty_class junk1 = coot_empty_class(),
  const typename coot_real_or_cx_only<typename T1::elem_type>::result* junk2 = 0
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk1);
  coot_ignore(junk2);

  return Op<T1, op_normalise_vec>(X, p, 0);
  }



template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value && resolves_to_vector<T1>::value == false,
  const Op<T1, op_normalise_mat>
  >::result
normalise
  (
  const T1& X,
  const uword p = uword(2),
  const uword dim = 0,
  const typename coot_real_or_cx_only<typename T1::elem_type>::result* junk = 0
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  return Op<T1, op_normalise_mat>(X, p, dim);
  }