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



template<typename eT>
inline
eT
op_norm::vec_norm_1(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::vec_norm_1(X.get_dev_mem(false), X.n_elem);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_1(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: implement
  }



template<typename eT>
inline
eT
op_norm::vec_norm_2(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::vec_norm_2(X.get_dev_mem(false), X.n_elem);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_2(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: implement
  }



template<typename eT>
inline
eT
op_norm::vec_norm_k(const Mat<eT>& X, const uword k)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::vec_norm_k(X.get_dev_mem(false), X.n_elem, k);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_k(const subview<eT>& X, const uword k)
  {
  coot_extra_debug_sigprint();
  // TODO: implement
  }



template<typename eT>
inline
eT
op_norm::vec_norm_min(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::vec_norm_min(X.get_dev_mem(false), X.n_elem);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_min(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: implement
  }



template<typename eT>
inline
eT
op_norm::vec_norm_max(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();
  return coot_rt_t::max_abs(X.get_dev_mem(false), X.n_elem);
  }



template<typename eT>
inline
eT
op_norm::vec_norm_max(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  // TODO: implement
  }
