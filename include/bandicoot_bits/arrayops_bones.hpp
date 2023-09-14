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



class arrayops
  {
  public:

  template<typename out_eT, typename in_eT>
  inline static void
  copy(dev_mem_t<out_eT> dest, const dev_mem_t<in_eT> src, const uword n_elem);

  template<typename out_eT, typename in_eT>
  inline static void
  copy_subview(dev_mem_t<out_eT> dest, const uword dest_offset, const dev_mem_t<in_eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword M_n_cols, const uword n_rows, const uword n_cols);


  //
  // array op= array

  template<typename eT1, typename eT2>
  inline static void
  inplace_plus_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem);

  template<typename eT1, typename eT2>
  inline static void
  inplace_minus_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem);

  template<typename eT1, typename eT2>
  inline static void
  inplace_mul_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem);

  template<typename eT1, typename eT2>
  inline static void
  inplace_div_array(dev_mem_t<eT2> dest, dev_mem_t<eT1> src, const uword n_elem);


  //
  // scalar = op(array)

  template<typename eT>
  inline static
  eT
  accumulate(cl_mem src, const uword n_elem);

  template<typename eT>
  inline static
  eT
  product(cl_mem src, const uword n_elem);
  };
