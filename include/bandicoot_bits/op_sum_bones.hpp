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


//! \addtogroup op_sum
//! @{


class op_sum
  {
  public:
  
  template<typename eT2, typename T1>
  inline static void apply(Mat<eT2>& out, const Op<eT2, T1, op_sum>& in);

  template<typename out_eT, typename T1>
  inline static void apply_after_conv_to(Mat<out_eT>& out, const Op<out_eT, T1, op_sum>& in);
  
  template<typename out_eT, typename in_eT>
  inline static void apply(Mat<out_eT>& out, const Op<out_eT, subview<in_eT>, op_sum>& in);

  template<typename eT>
  inline static void apply(Mat<eT>& out, const Op<eT, subview<eT>, op_sum>& in);
  
  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out, const Mat<in_eT>& A, const uword dim, const bool post_conv_apply);
  
  template<typename out_eT, typename in_eT>
  inline static void apply_noalias(Mat<out_eT>& out, const subview<in_eT>& sv, const uword dim, const bool post_conv_apply);
  };


//! @}
