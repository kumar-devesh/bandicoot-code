// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2022 Marcus Edel (http://kurg.org)
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


//! \addtogroup op_norm
//! @{



template<typename T1>
inline
typename T1::pod_type
op_norm::vec_norm_2(const Proxy<T1>& P)
  {
  coot_extra_debug_sigprint();
  /* coot_ignore(junk); */

  /* const bool use_direct_mem = (is_Mat<typename Proxy<T1>::stored_type>::value) || (is_subview_col<typename Proxy<T1>::stored_type>::value) || (arma_config::openmp && Proxy<T1>::use_mp); */

  /* if(use_direct_mem) */
    /* { */
    /* const quasi_unwrap<typename Proxy<T1>::stored_type> tmp(P.Q); */

  typedef typename T1::pod_type T;
  /* Mat<T> A(P.get_ea().get_ref()); */
  Mat<T> B(P.get_aligned_ea().get_ref());

  return coot_rt_t::vec_norm_2(B.get_dev_mem(false), P.get_n_elem());

    /* return op_norm::vec_norm_2_direct_std(tmp.M); */
    /* } */

  /* typedef typename T1::pod_type T; */

  /* T acc = T(0); */

  /* if(Proxy<T1>::use_at == false) */
  /*   { */
  /*   typename Proxy<T1>::ea_type A = P.get_ea(); */

  /*   const uword N = P.get_n_elem(); */

  /*   T acc1 = T(0); */
  /*   T acc2 = T(0); */

  /*   uword i,j; */

  /*   for(i=0, j=1; j<N; i+=2, j+=2) */
  /*     { */
  /*     const T tmp_i = A[i]; */
  /*     const T tmp_j = A[j]; */

  /*     acc1 += tmp_i * tmp_i; */
  /*     acc2 += tmp_j * tmp_j; */
  /*     } */

  /*   if(i < N) */
  /*     { */
  /*     const T tmp_i = A[i]; */

  /*     acc1 += tmp_i * tmp_i; */
  /*     } */

  /*   acc = acc1 + acc2; */
  /*   } */
  /* else */
  /*   { */
  /*   const uword n_rows = P.get_n_rows(); */
  /*   const uword n_cols = P.get_n_cols(); */

  /*   if(n_rows == 1) */
  /*     { */
  /*     for(uword col=0; col<n_cols; ++col) */
  /*       { */
  /*       const T tmp = P.at(0,col); */

  /*       acc += tmp * tmp; */
  /*       } */
  /*     } */
  /*   else */
  /*     { */
  /*     for(uword col=0; col<n_cols; ++col) */
  /*       { */
  /*       uword i,j; */
  /*       for(i=0, j=1; j<n_rows; i+=2, j+=2) */
  /*         { */
  /*         const T tmp_i = P.at(i,col); */
  /*         const T tmp_j = P.at(j,col); */

  /*         acc += tmp_i * tmp_i; */
  /*         acc += tmp_j * tmp_j; */
  /*         } */

  /*       if(i < n_rows) */
  /*         { */
  /*         const T tmp_i = P.at(i,col); */

  /*         acc += tmp_i * tmp_i; */
  /*         } */
  /*       } */
  /*     } */
  /*   } */


  /* const T sqrt_acc = std::sqrt(acc); */

  /* if( (sqrt_acc != T(0)) && arma_isfinite(sqrt_acc) ) */
  /*   { */
  /*   return sqrt_acc; */
  /*   } */
  /* else */
  /*   { */
  /*   arma_extra_debug_print("op_norm::vec_norm_2(): detected possible underflow or overflow"); */

  /*   const quasi_unwrap<typename Proxy<T1>::stored_type> tmp(P.Q); */

  /*   return op_norm::vec_norm_2_direct_robust(tmp.M); */
  /*   } */
  }



//! @}
