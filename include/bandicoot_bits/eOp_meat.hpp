// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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


//! \addtogroup eOp
//! @{



template<typename out_eT, typename T1, typename eop_type>
inline
eOp<out_eT, T1, eop_type>::eOp(const T1& in_m)
  : m(in_m)
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename eop_type>
inline
eOp<out_eT, T1, eop_type>::eOp(const T1& in_m, const typename T1::elem_type in_aux_in)
  : m(in_m)
  , aux_in(in_aux_in)
  , use_aux_in(true)
  , aux_out(out_eT(0))
  , use_aux_out(false)
  , aux_uword_a(uword(0))
  , aux_uword_b(uword(0))
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename eop_type>
inline
eOp<out_eT, T1, eop_type>::eOp(const T1& in_m, const typename T1::elem_type in_aux_in, const bool in_use_aux_in, const out_eT in_aux_out, const bool in_use_aux_out)
  : m(in_m)
  , aux_in(in_aux_in)
  , use_aux_in(in_use_aux_in)
  , aux_out(in_aux_out)
  , use_aux_out(in_use_aux_out)
  , aux_uword_a(uword(0))
  , aux_uword_b(uword(0))
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename eop_type>
inline
eOp<out_eT, T1, eop_type>::eOp(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : m(in_m)
  , aux_in(typename T1::elem_type(0))
  , use_aux_in(false)
  , aux_out(out_eT(0))
  , use_aux_out(false)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename eop_type>
inline
eOp<out_eT, T1, eop_type>::eOp(const T1& in_m, const typename T1::elem_type in_aux_in, const bool in_use_aux_in, const out_eT in_aux_out, const bool in_use_aux_out, const uword in_aux_uword_a, const uword in_aux_uword_b)
  : m(in_m)
  , aux_in(in_aux_in)
  , use_aux_in(in_use_aux_in)
  , aux_out(in_aux_out)
  , use_aux_out(in_use_aux_out)
  , aux_uword_a(in_aux_uword_a)
  , aux_uword_b(in_aux_uword_b)
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename eop_type>
inline
eOp<out_eT, T1, eop_type>::~eOp()
  {
  coot_extra_debug_sigprint();
  }



// note that in.aux_out is ignored!
template<typename out_eT, typename T1, typename eop_type>
template<typename in_eT>
inline
eOp<out_eT, T1, eop_type>::eOp(const eOp<in_eT, T1, eop_type>& in)
  : m(in.m)
  , aux_in(in.aux_in)
  , use_aux_in(in.use_aux_in)
  , aux_out(out_eT(0))
  , use_aux_out(false)
  , aux_uword_a(in.aux_uword_a)
  , aux_uword_b(in.aux_uword_b)
  {
  coot_extra_debug_sigprint();
  }



// note that in.aux_out is ignored!
template<typename out_eT, typename T1, typename eop_type>
template<typename in_eT>
inline
eOp<out_eT, T1, eop_type>::eOp(const eOp<in_eT, T1, eop_type>& in, const bool in_use_aux_in, const out_eT in_aux_out, const bool in_use_aux_out)
  : m(in.m)
  , aux_in(in.aux_in)
  , use_aux_in(in_use_aux_in)
  , aux_out(in_aux_out)
  , use_aux_out(in_use_aux_out)
  , aux_uword_a(in.aux_uword_a)
  , aux_uword_b(in.aux_uword_b)
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename eop_type>
coot_inline
uword
eOp<out_eT, T1, eop_type>::get_n_rows() const
  {
  return is_row ? 1 : m.get_n_rows();
  }
  


template<typename out_eT, typename T1, typename eop_type>
coot_inline
uword
eOp<out_eT, T1, eop_type>::get_n_cols() const
  {
  return is_col ? 1 : m.get_n_cols();
  }



template<typename out_eT, typename T1, typename eop_type>
coot_inline
uword
eOp<out_eT, T1, eop_type>::get_n_elem() const
  {
  return m.get_n_elem();
  }



//! @}
