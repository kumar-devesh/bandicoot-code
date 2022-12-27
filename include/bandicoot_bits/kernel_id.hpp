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



struct zeroway_kernel_id
  {
  enum enum_id
    {
    invalid_kernel
    };


  static
  inline
  std::vector<std::string>
  init_names()
    {
    // NOTE: the order and names of kernels in "names" must match the order and names in the kernel_id enum

    std::vector<std::string> names;

    return names;
    };


  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }
  };




struct oneway_kernel_id
  {
  enum enum_id
    {
    inplace_set_scalar = 0,
    inplace_plus_scalar,
    inplace_minus_scalar,
    inplace_mul_scalar,
    inplace_div_scalar,
    //
    submat_inplace_set_scalar,
    submat_inplace_plus_scalar,
    submat_inplace_minus_scalar,
    submat_inplace_mul_scalar,
    submat_inplace_div_scalar,
    //
    inplace_set_eye,
    //
    accu_simple,
    accu,
    accu_small,
    //
    min,
    min_small,
    max,
    max_small,
    max_abs,
    max_abs_small,
    //
    trace,
    //
    ltri_set_zero,
    //
    inplace_xorwow_randu,
    inplace_philox_randn,
    //
    invalid_kernel
    };


  static
  inline
  std::vector<std::string>
  init_names()
    {
    // NOTE: the order and names of kernels in "names" must match the order and names in the kernel_id enum

    std::vector<std::string> names;

    names.push_back("inplace_set_scalar");
    names.push_back("inplace_plus_scalar");
    names.push_back("inplace_minus_scalar");
    names.push_back("inplace_mul_scalar");
    names.push_back("inplace_div_scalar");

    names.push_back("submat_inplace_set_scalar");
    names.push_back("submat_inplace_plus_scalar");
    names.push_back("submat_inplace_minus_scalar");
    names.push_back("submat_inplace_mul_scalar");
    names.push_back("submat_inplace_div_scalar");

    names.push_back("inplace_set_eye");

    names.push_back("accu_simple");
    names.push_back("accu");
    names.push_back("accu_small");

    names.push_back("min");
    names.push_back("min_small");
    names.push_back("max");
    names.push_back("max_small");
    names.push_back("max_abs");
    names.push_back("max_abs_small");

    names.push_back("trace");

    names.push_back("ltri_set_zero");

    names.push_back("inplace_xorwow_randu");
    names.push_back("inplace_philox_randn");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }
  };



// These kernels should only be used with float or double element types.
struct oneway_real_kernel_id
  {
  enum enum_id
    {
    larfg = 0
    };


  static
  inline
  std::vector<std::string>
  init_names()
    {
    // NOTE: the order and names of kernels in "names" must match the order and names in the kernel_id enum

    std::vector<std::string> names;

    names.push_back("larfg");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }
  };



struct twoway_kernel_id
  {
  enum enum_id
    {
    // TODO: I don't think these are in-place...
    submat_inplace_set_mat = 0,
    submat_inplace_plus_mat,
    submat_inplace_minus_mat,
    submat_inplace_schur_mat,
    submat_inplace_div_mat,
    submat_extract,
    //
    inplace_plus_array,
    inplace_minus_array,
    inplace_mul_array,
    inplace_div_array,
    //
    equ_array_plus_scalar,
    equ_array_neg_pre,
    equ_array_neg_post,
    equ_array_minus_scalar_pre_pre,
    equ_array_minus_scalar_pre_post,
    equ_array_minus_scalar_post,
    equ_array_mul_scalar,
    equ_array_div_scalar_pre,
    equ_array_div_scalar_post,
    equ_array_square_pre,
    equ_array_square_post,
    equ_array_sqrt_pre,
    equ_array_sqrt_post,
    equ_array_exp_pre,
    equ_array_exp_post,
    equ_array_log_pre,
    equ_array_log_post,
    equ_array_abs,
    //
    get_diag,
    //
    sum_colwise_conv_pre,
    sum_rowwise_conv_pre,
    sum_colwise_conv_post,
    sum_rowwise_conv_post,
    submat_sum_colwise_conv_pre,
    submat_sum_rowwise_conv_pre,
    submat_sum_colwise_conv_post,
    submat_sum_rowwise_conv_post,
    //
    dot,
    dot_small,
    //
    convert_type,
    //
    invalid_kernel
    };



  static
  inline
  std::vector<std::string>
  init_names()
    {
    std::vector<std::string> names;

    names.push_back("submat_inplace_set_mat");
    names.push_back("submat_inplace_plus_mat");
    names.push_back("submat_inplace_minus_mat");
    names.push_back("submat_inplace_schur_mat");
    names.push_back("submat_inplace_div_mat");
    names.push_back("submat_extract");

    names.push_back("inplace_plus_array");
    names.push_back("inplace_minus_array");
    names.push_back("inplace_mul_array");
    names.push_back("inplace_div_array");

    names.push_back("equ_array_plus_scalar");
    names.push_back("equ_array_neg_pre");
    names.push_back("equ_array_neg_post");
    names.push_back("equ_array_minus_scalar_pre_pre");
    names.push_back("equ_array_minus_scalar_pre_post");
    names.push_back("equ_array_minus_scalar_post");
    names.push_back("equ_array_mul_scalar");
    names.push_back("equ_array_div_scalar_pre");
    names.push_back("equ_array_div_scalar_post");
    names.push_back("equ_array_square_pre");
    names.push_back("equ_array_square_post");
    names.push_back("equ_array_sqrt_pre");
    names.push_back("equ_array_sqrt_post");
    names.push_back("equ_array_exp_pre");
    names.push_back("equ_array_exp_post");
    names.push_back("equ_array_log_pre");
    names.push_back("equ_array_log_post");
    names.push_back("equ_array_abs");

    names.push_back("get_diag");

    names.push_back("sum_colwise_conv_pre");
    names.push_back("sum_rowwise_conv_pre");
    names.push_back("sum_colwise_conv_post");
    names.push_back("sum_rowwise_conv_post");
    names.push_back("submat_sum_colwise_conv_pre");
    names.push_back("submat_sum_rowwise_conv_pre");
    names.push_back("submat_sum_colwise_conv_post");
    names.push_back("submat_sum_rowwise_conv_post");

    names.push_back("dot");
    names.push_back("dot_small");

    names.push_back("convert_type");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }
  };



struct threeway_kernel_id
  {
  enum enum_id
    {
    equ_array_plus_array = 0,
    equ_array_minus_array,
    equ_array_mul_array,
    equ_array_div_array,
    //
    invalid_kernel
    };



  static
  inline
  std::vector<std::string>
  init_names()
    {
    std::vector<std::string> names;

    names.push_back("equ_array_plus_array");
    names.push_back("equ_array_minus_array");
    names.push_back("equ_array_mul_array");
    names.push_back("equ_array_div_array");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }
  };
