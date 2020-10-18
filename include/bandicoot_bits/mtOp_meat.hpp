// Copyright 2017 Ryan Curtin (https://www.ratml.org/)
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


//! \addtogroup mtOp
//! @{



template<typename out_eT, typename T1, typename mtop_type>
inline
mtOp<out_eT, T1, mtop_type>::mtOp(const T1& in_m)
  : m(in_m)
  , q(in_m)
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename mtop_type>
inline
mtOp<out_eT, T1, mtop_type>::~mtOp()
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename mtop_type>
coot_inline
uword
mtOp<out_eT, T1, mtop_type>::get_n_rows() const
  {
  return is_row ? 1 : m.get_n_rows();
  }
  


template<typename out_eT, typename T1, typename mtop_type>
coot_inline
uword
mtOp<out_eT, T1, mtop_type>::get_n_cols() const
  {
  return is_col ? 1 : m.get_n_cols();
  }



template<typename out_eT, typename T1, typename mtop_type>
coot_inline
uword
mtOp<out_eT, T1, mtop_type>::get_n_elem() const
  {
  return m.get_n_elem();
  }



template<typename out_eT, typename T1, typename mtop_type>
inline uword dispatch_mtop_get_n_rows(const mtOp<out_eT, T1, mtop_type>& Q)
  {
  return Q.get_n_rows();
  }



template<typename out_eT>
inline uword dispatch_mtop_get_n_rows(const Mat<out_eT>& Q)
  {
  return Q.n_rows;
  }



template<typename out_eT, typename T1, typename mtop_type>
inline uword dispatch_mtop_get_n_cols(const mtOp<out_eT, T1, mtop_type>& Q)
  {
  return Q.get_n_cols();
  }



template<typename out_eT>
inline uword dispatch_mtop_get_n_cols(const Mat<out_eT>& Q)
  {
  return Q.n_cols;
  }



template<typename out_eT, typename T1, typename mtop_type>
inline uword dispatch_mtop_get_n_elem(const mtOp<out_eT, T1, mtop_type>& Q)
  {
  return Q.get_n_elem();
  }



template<typename out_eT>
inline uword dispatch_mtop_get_n_elem(const Mat<out_eT>& Q)
  {
  return Q.n_elem;
  }



//! @}
