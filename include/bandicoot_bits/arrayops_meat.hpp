// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2017      Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename out_eT, typename in_eT>
inline
void
arrayops::copy(dev_mem_t<out_eT> dest, dev_mem_t<in_eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (n_elem > 0)
    coot_rt_t::copy_array(dest, src, n_elem);
  }



template<typename out_eT, typename in_eT>
inline
void
arrayops::copy_subview(dev_mem_t<out_eT> dest, const uword dest_offset, dev_mem_t<in_eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword M_n_cols, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (n_rows != 0 && n_cols != 0)
    coot_rt_t::copy_subview(dest, dest_offset, src, aux_row1, aux_col1, M_n_rows, M_n_cols, n_rows, n_cols);
  }



template<typename eT>
inline
void
arrayops::inplace_set_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::inplace_op_scalar(dest, val, n_elem, oneway_kernel_id::inplace_set_scalar);
  }



template<typename eT>
inline
void
arrayops::inplace_plus_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::inplace_op_scalar(dest, val, n_elem, oneway_kernel_id::inplace_plus_scalar);
  }



template<typename eT>
inline
void
arrayops::inplace_minus_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::inplace_op_scalar(dest, val, n_elem, oneway_kernel_id::inplace_minus_scalar);
  }



template<typename eT>
inline
void
arrayops::inplace_mul_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::inplace_op_scalar(dest, val, n_elem, oneway_kernel_id::inplace_mul_scalar);
  }



template<typename eT>
inline
void
arrayops::inplace_div_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::inplace_op_scalar(dest, val, n_elem, oneway_kernel_id::inplace_div_scalar);
  }



template<typename eT1, typename eT2>
inline
void
arrayops::inplace_plus_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::inplace_op_array(dest, src, n_elem, twoway_kernel_id::inplace_plus_array);
  }



template<typename eT1, typename eT2>
inline
void
arrayops::inplace_minus_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::inplace_op_array(dest, src, n_elem, twoway_kernel_id::inplace_minus_array);
  }



template<typename eT1, typename eT2>
inline
void
arrayops::inplace_mul_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::inplace_op_array(dest, src, n_elem, twoway_kernel_id::inplace_mul_array);
  }



template<typename eT1, typename eT2>
inline
void
arrayops::inplace_div_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::inplace_op_array(dest, src, n_elem, twoway_kernel_id::inplace_div_array);
  }



template<typename eT>
inline
eT
arrayops::accumulate(cl_mem src, const uword n_elem)
  {
  // TODO
  }



template<typename eT>
inline
eT
arrayops::product(cl_mem src, const uword n_elem)
  {
  // TODO
  }
