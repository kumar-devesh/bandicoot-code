// Copyright 2022 Gopi Tatiraju
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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



struct glue_join_rows
  {
  // Note that it's okay if T1 and T2 contain conversions.
  template<typename out_eT, typename T1, typename T2> static inline void apply(Mat<out_eT>& out, const T1& A, const T2& B, const std::string& func_name = "join_rows()");

  template<typename T1, typename T2> static inline uword compute_n_rows(const Glue<T1, T2, glue_join_cols>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename T1, typename T2> static inline uword compute_n_cols(const Glue<T1, T2, glue_join_cols>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  };
