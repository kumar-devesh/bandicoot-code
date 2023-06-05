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



template<typename T1>
coot_warn_unused
inline
typename
enable_if2<
  is_real<typename T1::elem_type>::value,
  const Op<T1, op_pinv>
>::result
pinv
  (
  const Base<typename T1::elem_type, T1>& X
  )
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_pinv>(X.get_ref());
  }


template<typename T1>
coot_warn_unused
inline
typename
enable_if2<
  is_real<typename T1::elem_type>::value,
  const Op<T1, op_pinv>
>::result
pinv
  (
  const Base<typename T1::elem_type, T1>& X,
  const typename T1::elem_type            tol,
  const char*                             method = nullptr
  )
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  uword method_id = 0; // default

  if (method != nullptr)
    {
    const char sig = method[0];

    coot_debug_check( ((sig != 's') && (sig != 'd')), "pinv(): unknown method specified" );

    if (sig == 's') { method_id = 1; }
    if (sig == 'd') { method_id = 2; }
    }

  return Op<T1, op_pinv>(X.get_ref(), eT(tol), method_id, uword(0));
  }



template<typename T1>
inline
typename
enable_if2<
  is_real<typename T1::elem_type>::value,
  bool
>::result
pinv
  (
         Mat<typename T1::elem_type>&     out,
  const Base<typename T1::elem_type, T1>& X,
  const typename T1::elem_type            tol = 0.0,
  const char*                             method = nullptr
  )
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  uword method_id = 0; // default

  if (method != nullptr)
    {
    const char sig = method[0];

    coot_debug_check( ((sig != 's') && (sig != 'd')), "pinv(): unknown method specified" );

    if (sig == 's') { method_id = 1; }
    if (sig == 'd') { method_id = 2; }
    }

  const bool status = op_pinv::apply_direct(out, X.get_ref(), tol, method_id);

  if (status == false)
    {
    out.reset();
    // TODO: better error message?
    coot_debug_warn("pinv(): SVD failed");
    }

  return status;
  }
