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



// We don't use any delayed evaluation infrastructure for join_cols() or join_rows().
// This is because we can't benefit from any speedups; in Armadillo, we can have Proxy-based
// access to either underlying matrix and thus avoid forming the full matrix, but here
// in Bandicoot we can't do that, as our kernels all work directly on memory.
// So, we always just form the result matrix directly.



template<typename eT, typename T1, typename T2>
coot_warn_unused
inline
Mat<eT>
join_cols
  (
  const Base<eT, T1>& A,
  const Base<eT, T2>& B
  )
  {
  coot_extra_debug_sigprint();

  Mat<eT> out;
  glue_join_cols::apply(out, A, B, "join_cols");
  return out;
  }



template<typename eT, typename T1, typename T2>
coot_warn_unused
inline
Mat<eT>
join_vert
  (
  const Base<eT, T1>& A,
  const Base<eT, T2>& B
  )
  {
  coot_extra_debug_sigprint();

  Mat<eT> out;
  glue_join_cols::apply(out, A, B, "join_vert");
  return out;
  }



template<typename eT, typename T1, typename T2>
coot_warn_unused
inline
Mat<eT>
join_rows
  (
  const Base<eT, T1>& A,
  const Base<eT, T2>& B
  )
  {
  coot_extra_debug_sigprint();

  Mat<eT> out;
  glue_join_rows::apply(out, A, B, "join_rows");
  return out;
  }



template<typename eT, typename T1, typename T2>
coot_warn_unused
inline
Mat<eT>
join_horiz
  {
  const Base<eT, T1>& A,
  const Base<eT, T2>& B
  )
  {
  coot_extra_debug_sigprint();

  Mat<eT> out;
  glue_join_rows::apply(out, A, B, "join_horiz");
  return out;
  }
