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



template<typename obj_type>
coot_warn_unused
inline
obj_type
/* coot::mat */
randu(const uword n_rows, const uword n_cols, const typename coot_Mat_Col_Row_only<obj_type>::result* junk = nullptr)
/* randu(const uword n_rows, const uword n_cols) */
  {
  coot_extra_debug_sigprint();
  /* coot_ignore(junk); */

  /* typedef typename obj_type::elem_type eT; */

  /* if(is_Col<obj_type>::value) */
  /*   { */
  /*   coot_debug_check( (n_cols != 1), "randu(): incompatible size" ); */
  /*   } */
  /* else */
  /* if(is_Row<obj_type>::value) */
  /*   { */
  /*   coot_debug_check( (n_rows != 1), "randu(): incompatible size" ); */
  /*   } */

  /* obj_type out(n_rows, n_cols, arma_nozeros_indicator()); */
  obj_type out(n_rows, n_cols);
  /* coot::mat out(n_rows, n_cols); */

/*   int a; */
/*   int b; */

/*   if(param.state == 0) */
/*     { */
/*     a = 0; */
/*     b = coot_rng::randi<eT>::max_val(); */
/*     } */
/*   else */
/*   if(param.state == 1) */
/*     { */
/*     a = param.a_int; */
/*     b = param.b_int; */
/*     } */
/*   else */
/*     { */
/*     a = int(param.a_double); */
/*     b = int(param.b_double); */
/*     } */

/*   coot_debug_check( (a > b), "randi(): incorrect distribution parameters: a must be less than b" ); */
  coot_rng::fill_randu(out.get_dev_mem(false), out.n_elem);
  /* coot_rng::randu<eT>::fill(out.get_dev_mem(false), out.n_elem); */

  return out;
  }



//! @}
