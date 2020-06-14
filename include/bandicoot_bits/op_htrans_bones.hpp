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


//! \addtogroup op_htrans
//! @{


class op_htrans
  {
  public:
  
  template<typename out_eT, typename T1>
  inline static void apply(Mat<out_eT>& out, const Op<out_eT, T1, op_htrans>& in);
  
  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out,
                                   const Mat<in_eT>& A,
                                   const typename coot_not_cx<in_eT>::result* junk1 = 0,
                                   const typename coot_not_cx<out_eT>::result* junk2 = 0);
  
  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out,
                                   const Mat<in_eT>& A,
                                   const typename coot_cx_only<in_eT>::result* junk1 = 0,
                                   const typename coot_cx_only<out_eT>::result* junk2 = 0);
  };



class op_htrans2
  {
  public:
  
  template<typename out_eT, typename T1>
  inline static void apply(Mat<out_eT>& out, const Op<out_eT, T1, op_htrans2>& in);
  
  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out,
                                   const Mat<in_eT>& A,
                                   const typename coot_not_cx<in_eT>::result* junk1 = 0,
                                   const typename coot_not_cx<out_eT>::result* junk2 = 0);
  
  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out,
                                   const Mat<in_eT>& A,
                                   const typename coot_cx_only<in_eT>::result* junk1 = 0,
                                   const typename coot_cx_only<out_eT>::result* junk2 = 0);
  };



//! @}
