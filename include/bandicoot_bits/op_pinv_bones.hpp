// Copyright 2023 Ryan Curtin (https://www.ratml.org/)
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



class op_pinv
  : public traits_op_passthru
  {
  public:

  //
  // for use in delayed operations
  //

  template<typename eT2, typename T1>
  inline static void apply(Mat<eT2>& out, const Op<T1, op_pinv>& in);

  template<typename eT2, typename eT1>
  inline static void apply_direct(Mat<eT2>& out, const Mat<eT1>& in);

  template<typename eT2, typename eT1>
  inline static void apply_direct_diag(Mat<eT2>& out, const Mat<eT1>& in);

  template<typename eT2, typename eT1>
  inline static void apply_direct_sym(Mat<eT2>& out, const Mat<eT1>& in);

  template<typename eT2, typename eT1>
  inline static void apply_direct_gen(Mat<eT2>& out, const Mat<eT1>& in);

  template<typename T1> inline static uword compute_n_rows(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols);

  };
