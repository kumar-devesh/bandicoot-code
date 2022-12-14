// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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


//! \addtogroup eop_core
//! @{



template<typename eop_type>
class eop_core
  {
  public:
  
  // matrices
  
  template<typename eT, typename T1> inline static void apply              (Mat<eT>& out, const eOp<T1, eop_type>& x);

  template<typename eT, typename T2> inline static void apply              (Mat<eT>& out, const eOp<mtOp<eT, eOp<T2, eop_type>, mtop_conv_to>, eop_type>& X);

  template<typename eT, typename T1> inline static void apply_inplace_plus (Mat<eT>& out, const eOp<T1, eop_type>& x);
  template<typename eT, typename T1> inline static void apply_inplace_minus(Mat<eT>& out, const eOp<T1, eop_type>& x);
  template<typename eT, typename T1> inline static void apply_inplace_schur(Mat<eT>& out, const eOp<T1, eop_type>& x);
  template<typename eT, typename T1> inline static void apply_inplace_div  (Mat<eT>& out, const eOp<T1, eop_type>& x);
  
  
  // cubes
  
  // TODO
  };



// every eop has the ability to apply a conversion before or after; if 'chainable' is true, then
// it is possible to apply a conversion *between* two eops of the same type



class eop_scalar_plus       : public eop_core<eop_scalar_plus>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_plus_scalar;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_plus_scalar;
  const static bool is_chainable = true;
  };

class eop_neg               : public eop_core<eop_neg>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_neg_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_neg_post;
  const static bool is_chainable = false;
  };

class eop_scalar_minus_pre  : public eop_core<eop_scalar_minus_pre>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_minus_scalar_pre_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_minus_scalar_pre_post;
  const static bool is_chainable = true;
  };

class eop_scalar_minus_post : public eop_core<eop_scalar_minus_post>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_minus_scalar_post;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_minus_scalar_post;
  const static bool is_chainable = true;
  };

class eop_scalar_times      : public eop_core<eop_scalar_times>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_mul_scalar;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_mul_scalar;
  const static bool is_chainable = true;
  };
class eop_scalar_div_pre    : public eop_core<eop_scalar_div_pre>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_div_scalar_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_div_scalar_pre;
  const static bool is_chainable = true;
  };

class eop_scalar_div_post   : public eop_core<eop_scalar_div_post>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_div_scalar_post;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_div_scalar_post;
  const static bool is_chainable = true;
  };

class eop_square            : public eop_core<eop_square>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_square_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_square_post;
  const static bool is_chainable = false;
  };

class eop_sqrt              : public eop_core<eop_sqrt>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_sqrt_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_sqrt_post;
  const static bool is_chainable = false;
  };

class eop_log               : public eop_core<eop_log>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_log_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_log_post;
  const static bool is_chainable = false;
  };

class eop_log2              : public eop_core<eop_log2>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_log2_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_log2_post;
  const static bool is_chainable = false;
  };

class eop_log10             : public eop_core<eop_log10>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_log10_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_log10_post;
  const static bool is_chainable = false;
  };

class eop_trunc_log         : public eop_core<eop_trunc_log>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_trunc_log_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_trunc_log_post;
  const static bool is_chainable = false;
  };

class eop_exp               : public eop_core<eop_exp>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_exp_pre;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_exp_post;
  const static bool is_chainable = false;
  };

// class eop_exp2              : public eop_core<eop_exp2>              {};
// class eop_exp10             : public eop_core<eop_exp10>             {};
// class eop_trunc_exp         : public eop_core<eop_trunc_exp>         {};
// class eop_cos               : public eop_core<eop_cos>               {};
// class eop_sin               : public eop_core<eop_sin>               {};
// class eop_tan               : public eop_core<eop_tan>               {};
// class eop_acos              : public eop_core<eop_acos>              {};
// class eop_asin              : public eop_core<eop_asin>              {};
// class eop_atan              : public eop_core<eop_atan>              {};
// class eop_cosh              : public eop_core<eop_cosh>              {};
// class eop_sinh              : public eop_core<eop_sinh>              {};
// class eop_tanh              : public eop_core<eop_tanh>              {};
// class eop_acosh             : public eop_core<eop_acosh>             {};
// class eop_asinh             : public eop_core<eop_asinh>             {};
// class eop_atanh             : public eop_core<eop_atanh>             {};

class eop_abs               : public eop_core<eop_abs>
  {
  public:

  const static twoway_kernel_id::enum_id kernel_conv_pre  = twoway_kernel_id::equ_array_abs;
  const static twoway_kernel_id::enum_id kernel_conv_post = twoway_kernel_id::equ_array_abs;
  const static bool is_chainable = true; // not clear why anyone would write a chain like this
  };

// class eop_arg               : public eop_core<eop_arg>               {};
// class eop_conj              : public eop_core<eop_conj>              {};
// class eop_pow               : public eop_core<eop_pow>               {};
// class eop_floor             : public eop_core<eop_floor>             {};
// class eop_ceil              : public eop_core<eop_ceil>              {};
// class eop_round             : public eop_core<eop_round>             {};
// class eop_trunc             : public eop_core<eop_trunc>             {};
// class eop_sign              : public eop_core<eop_sign>              {};
// class eop_erf               : public eop_core<eop_erf>               {};
// class eop_erfc              : public eop_core<eop_erfc>              {};
// class eop_lgamma            : public eop_core<eop_lgamma>            {};

template<typename eop_type>
struct get_default { template<typename eT> static inline eT val() { return eT(0); } };

template<>
struct get_default<eop_scalar_times> { template<typename eT> static inline eT val() { return eT(1); } };

template<>
struct get_default<eop_scalar_div_post> { template<typename eT> static inline eT val() { return eT(1); } };


//! @}
