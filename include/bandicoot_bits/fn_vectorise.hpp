// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2021-2022 Marcus Edel (http://kurg.org)
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
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_vectorise_col>
  >::result
vectorise(const T1& X)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_vectorise_col>(X);
  }



template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_vectorise_all>
  >::result
vectorise(const T1& X, const uword dim)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (dim > 1), "vectorise(): parameter 'dim' must be 0 or 1" );

  return Op<T1, op_vectorise_all>(X, dim, 0);
  }
