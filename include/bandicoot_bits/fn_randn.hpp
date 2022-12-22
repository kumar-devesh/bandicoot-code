// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2022 Marcus Edel (http://kurg.org)
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


//! \addtogroup fn_randu
//! @{



template<typename T>
coot_warn_unused
inline
T
randn(const uword n_rows, const uword n_cols, const distr_param& param = distr_param(), const typename coot_Mat_Col_Row_only<T>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  if(is_Col<T>::value)
    {
    coot_debug_check( (n_cols != 1), "randn(): incompatible size" );
    }
  else if(is_Row<T>::value)
    {
    coot_debug_check( (n_rows != 1), "randn(): incompatible size" );
    }

  T out(n_rows, n_cols);

  double a, b;
  if (param.state == 0)
    {
    // defaults
    a = 0.0;
    b = 1.0;
    }
  else if (param.state == 1)
    {
    // ints
    a = (double) param.a_int;
    b = (double) param.b_int;
    }
  else if (param.state == 2)
    {
    // doubles
    a = param.a_double;
    b = param.b_double;
    }
  else
    {
    coot_stop_runtime_error("randn(): incorrect distribution parameter settimgs");
    }

  coot_debug_check( (b < 0.0), "randn(): incorrect distribution parameters: sd must be greater than or equal to 0" );

  coot_rng::fill_randn(out.get_dev_mem(false), out.n_elem, a, b);

  return out;
  }



template<typename T>
coot_warn_unused
inline
T
randn(const uword n_elem, const distr_param& param = distr_param(), const typename coot_Mat_Col_Row_only<T>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  const uword n_rows = (is_Row<T>::value) ? uword(1) : n_elem;
  const uword n_cols = (is_Row<T>::value) ? n_elem   : uword(1);

  return randn<T>(n_rows, n_cols, param);
  }



//! @}
