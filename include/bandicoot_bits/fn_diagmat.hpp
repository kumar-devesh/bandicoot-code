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



//! interpret a matrix or a vector as a diagonal matrix (ie. off-diagonal entries are zero)
template<typename T1>
coot_warn_unused
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat>
  >::result
diagmat(const T1& X)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_diagmat>(X);
  }



//! create a matrix with the k-th diagonal set to the given vector
template<typename T1>
coot_warn_unused
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat2>
  >::result
diagmat(const T1& X, const sword k)
  {
  coot_extra_debug_sigprint();

  const uword a = (std::abs)(k);
  const uword b = (k < 0) ? 1 : 0;

  return Op<T1, op_diagmat2>(X, a, b);
  }
