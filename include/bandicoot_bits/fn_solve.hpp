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



template<typename T1, typename T2>
coot_warn_unused
inline
typename enable_if2
  <
  is_real<typename T1::elem_type>::value,
  const Glue<T1, T2, glue_solve>
  >::result
solve
  (
  const Base<typename T1::elem_type, T1>& A,
  const Base<typename T1::elem_type, T2>& B
  )
  {
  coot_extra_debug_sigprint();

  return Glue<T1, T2, glue_solve>(A.get_ref(), B.get_ref());
  }



template<typename T1, typename T2>
inline
typename enable_if2
  <
  is_real<typename T1::elem_type>::value,
  bool
  >::result
solve
  (
         Mat<typename T1::elem_type>&     out,
  const Base<typename T1::elem_type, T1>& A,
  const Base<typename T1::elem_type, T2>& B
  )
  {
  coot_extra_debug_sigprint();

  const std::tuple<bool, std::string> result = glue_solve::apply(out, A.get_ref(), B.get_ref(), 0 /* no flags */);

  if (!std::get<0>(result))
    {
    out.reset();
    coot_stop_runtime_error("solve(): " + std::get<1>(result));
    }

  return std::get<0>(result);
  }



template<typename T1, typename T2>
coot_warn_unused
inline
typename enable_if2
  <
  is_real<typename T1::elem_type>::value,
  const Glue<T1, T2, glue_solve>
  >::result
solve
  (
  const Base<typename T1::elem_type, T1>& A,
  const Base<typename T1::elem_type, T2>& B,
  const solve_opts::opts&                 opts
  )
  {
  coot_extra_debug_sigprint();

  return Glue<T1, T2, glue_solve>(A.get_ref(), B.get_ref(), opts.flags);
  }



template<typename T1, typename T2>
inline
typename enable_if2
  <
  is_real<typename T1::elem_type>::value,
  bool
  >::result
solve
  (
         Mat<typename T1::elem_type>&     out,
  const Base<typename T1::elem_type, T1>& A,
  const Base<typename T1::elem_type, T2>& B,
  const solve_opts::opts&                 opts
  )
  {
  coot_extra_debug_sigprint();

  const std::tuple<bool, std::string> result = glue_solve::apply(out, A.get_ref(), B.get_ref(), opts.flags);

  if (!std::get<0>(result))
    {
    out.reset();
    coot_stop_runtime_error("solve(): " + std::get<1>(result));
    }

  return std::get<0>(result);
  }
