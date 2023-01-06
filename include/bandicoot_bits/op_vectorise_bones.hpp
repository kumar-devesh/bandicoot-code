// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2021-2022 Marcus Edel (http://kurg.org)
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



class op_vectorise_col
  {
  public:

  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const Op<T1,op_vectorise_col>& in);

  template<typename out_eT, typename T1> inline static void apply_direct(Mat<out_eT>& out, const T1& in);

  template<typename out_eT, typename eT> inline static void apply_direct(Mat<out_eT>& out, const subview<eT>& sv);
  };



class op_vectorise_all
  {
  public:

  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const Op<T1,op_vectorise_all>& in);
  };



class op_vectorise_row
  {
  public:

  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT>& out, const Op<T1,op_vectorise_row>& in);

  template<typename T1> inline static void apply_direct(Mat<typename T1::elem_type>& out, const T1& in);

  template<typename eT> inline static void apply_direct(Mat<eT>& out, const subview<eT>& sv);

  template<typename out_eT, typename T1> inline static void apply_direct(Mat<out_eT>& out, const T1& in, const typename enable_if<!std::is_same<out_eT, typename T1::elem_type>::value>::result* = 0);

  template<typename out_eT, typename eT> inline static void apply_direct(Mat<out_eT>& out, const subview<eT>& sv, const typename enable_if<!std::is_same<out_eT, eT>::value>::result* = 0);
  };
