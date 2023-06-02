// Copyright 2023 Ryan Curtin (http://www.ratml.org)
//
// SPDX-License-Identifier: Apache-2.0
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



// Note that Armadillo has also an op_diagvec2 class that handles diagonals that
// are not the main diagonal; but, that gives us no extra optimization
// opportunities for Bandicoot, so we only use op_diagvec here to capture both
// cases.

class op_diagvec
  {
  public:

  template<                 typename T1> inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1, op_diagvec>& in);
  template<typename out_eT, typename T1> inline static void apply(Mat<out_eT                >& out, const Op<T1, op_diagvec>& in, const typename enable_if<is_same_type<out_eT, typename T1::elem_type>::no>::result* junk = 0);

  template<typename eT> inline static void apply_direct(Mat<eT>& out, const Mat<eT>& in, const sword k);
  template<typename eT> inline static void apply_direct(Mat<eT>& out, const subview<eT>& in, const sword k);

  template<typename T1> inline static uword compute_n_rows(const Op<T1, op_diagvec>& op, const uword in_n_rows, const uword in_n_cols);
  template<typename T1> inline static uword compute_n_cols(const Op<T1, op_diagvec>& op, const uword in_n_rows, const uword in_n_cols);
  };
