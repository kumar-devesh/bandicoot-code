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
inline
bool
lu
  (
         Mat<typename T1::elem_type>&     L,
         Mat<typename T1::elem_type>&     U,
  const Base<typename T1::elem_type, T1>& X,
  const typename coot_real_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  coot_debug_check( (&L) == (&U), "lu(): L and U are the same object" );

  // The LU factorisation will be done in-place, into U.
  U = X.get_ref();
  L.set_size(U.n_rows, U.n_rows);

  if (U.n_elem == 0)
    {
    // Nothing to do---leave early.
    L.set_size(U.n_rows, 0);
    U.set_size(0, U.n_cols);
    return true;
    }

  const bool status = coot_rt_t::lu(L.get_dev_mem(true), U.get_dev_mem(true), false /* no pivoting */, U.get_dev_mem(false) /* ignored */, U.n_rows, U.n_cols);

  return status;
  }



template<typename T1>
inline
bool
lu
  (
         Mat<typename T1::elem_type>&     L,
         Mat<typename T1::elem_type>&     U,
         Mat<typename T1::elem_type>&     P,
  const Base<typename T1::elem_type, T1>& X,
  const typename coot_real_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  coot_debug_check( (&L) == (&U), "lu(): L and U are the same object" );

  // The LU factorisation will be done in-place, into U.
  U = X.get_ref();
  L.set_size(U.n_rows, U.n_rows);
  P.zeros(U.n_rows, U.n_rows);

  if (U.n_elem == 0)
    {
    // Nothing to do---leave early.
    L.set_size(U.n_rows, 0);
    U.set_size(0, U.n_cols);
    P.eye(L.n_rows, L.n_rows);
    return true;
    }

  const bool status = coot_rt_t::lu(L.get_dev_mem(true), U.get_dev_mem(true), true, P.get_dev_mem(true), U.n_rows, U.n_cols);

  return status;
  }
