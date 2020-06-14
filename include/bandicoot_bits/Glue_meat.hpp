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


//! \addtogroup Glue
//! @{



template<typename out_eT, typename T1, typename T2, typename glue_type>
inline
Glue<out_eT, T1, T2, glue_type>::Glue(const T1& in_A, const T2& in_B)
  : A(in_A)
  , B(in_B)
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename T2, typename glue_type>
inline
Glue<out_eT, T1, T2, glue_type>::Glue(const T1& in_A, const T2& in_B, const uword in_aux_uword)
  : A(in_A)
  , B(in_B)
  , aux_uword(in_aux_uword)
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename T2, typename glue_type>
inline
Glue<out_eT, T1, T2, glue_type>::~Glue()
  {
  coot_extra_debug_sigprint();
  }



template<typename out_eT, typename T1, typename T2, typename glue_type>
template<typename in_eT>
inline
Glue<out_eT, T1, T2, glue_type>::Glue(const Glue<in_eT, T1, T2, glue_type>& in)
  : A(in.A)
  , B(in.B)
  , aux_uword(in.aux_uword)
  {
  coot_extra_debug_sigprint();
  }



//! @}
