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
class eOp : public Base<out_eT, eOp<out_eT, T1, eop_type> >
  {
  public:
  
  typedef typename T1::elem_type                   in_elem_type;
  typedef out_eT                                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  static const bool is_row = T1::is_row;
  static const bool is_col = T1::is_col;
  
  coot_aligned const SizeProxy<T1> m;
  coot_aligned       in_elem_type  aux_in;       //!< storage of auxiliary data, user defined format
  coot_aligned       bool          use_aux_in;   //!< whether to use aux_in in the operation
  coot_aligned       elem_type     aux_out;      //!< storage of auxiliary data, user defined format
  coot_aligned       bool          use_aux_out;  //!< whether to use aux_out in the operation
  coot_aligned       uword         aux_uword_a;  //!< storage of auxiliary data, uword format
  coot_aligned       uword         aux_uword_b;  //!< storage of auxiliary data, uword format
  
  inline         ~eOp();
  inline explicit eOp(const T1& in_m);
  inline          eOp(const T1& in_m, const in_elem_type in_aux_in);
  inline          eOp(const T1& in_m, const in_elem_type in_aux_in, const bool use_aux_in, const elem_type in_aux_out, const bool use_aux_out);
  inline          eOp(const T1& in_m, const uword in_aux_uword_a, const uword in_aux_uword_b);
  inline          eOp(const T1& in_m, const in_elem_type in_aux_in, const bool use_aux_in, const elem_type in_aux_out, const bool use_aux_out, const uword in_aux_uword_a, const uword in_aux_uword_b);

  template<typename in_eT>
  inline explicit eOp(const eOp<in_eT, T1, eop_type>& in);

  template<typename in_eT>
  inline explicit eOp(const eOp<in_eT, T1, eop_type>& in, const bool in_use_aux_in, const elem_type in_aux_out, const bool in_use_aux_out);
  
  coot_inline uword get_n_rows() const;
  coot_inline uword get_n_cols() const;
  coot_inline uword get_n_elem() const;
  };



//! @}
