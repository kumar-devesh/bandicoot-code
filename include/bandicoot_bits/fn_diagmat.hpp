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



// simplification: wrap transposes into diagmat
template<typename T1>
coot_warn_unused
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat>
  >::result
diagmat(const Op<T1, op_htrans>& X)
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_diagmat>(X.m);
  }



template<typename T1>
coot_warn_unused
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagmat2>
  >::result
diagmat(const Op<T1, op_htrans>& X, const sword k)
  {
  coot_extra_debug_sigprint();

  const uword a = (std::abs)(k);
  const uword b = (k < 0) ? 3 : 2;

  return Op<T1, op_diagmat2>(X.m, a, b);
  }



// simplification: diagmat(scalar * Base.t()) -> diagmat(htrans2(Base))
// this gives a form that partial_unwrap will be able to better handle

template<typename T1>
coot_warn_unused
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<eOp<T1, eop_scalar_times>, op_diagmat>
  >::result
diagmat(const Op<T1, op_htrans2>& X)
  {
  coot_extra_debug_sigprint();

  std::cout << "create new eop_scalar_times and strip htrans\n";
  eOp<T1, eop_scalar_times> inner(X.m, X.aux);
  std::cout << "T1: " << typeid(T1).name() << "\n";
  std::cout << "size of inner object is " << inner.get_n_rows() << " x " << inner.get_n_cols() << "\n";
  return Op<eOp<T1, eop_scalar_times>, op_diagmat>(inner);
  }



template<typename T1>
coot_warn_unused
coot_inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<eOp<T1, eop_scalar_times>, op_diagmat2>
  >::result
diagmat(const Op<T1, op_htrans2>& X, const sword k)
  {
  coot_extra_debug_sigprint();

  const uword a = (std::abs)(k);
  const uword b = (k < 0) ? 3 : 2;

  eOp<T1, eop_scalar_times> inner(X.m, X.aux);
  return Op<eOp<T1, eop_scalar_times>, op_diagmat2>(inner, a, b);
  }
