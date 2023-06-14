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



enum coot_backend_t
  {
  CL_BACKEND = 0,
  CUDA_BACKEND
  };

// TODO: if this is placed into a run-time library and executed there, what happens when two programs use the run-time library at the same time?
class coot_rt_t
  {
  public:

  coot_backend_t backend;

  #if defined(COOT_USE_OPENCL)
  opencl::runtime_t cl_rt;
  #endif

  #if defined(COOT_USE_CUDA)
  cuda::runtime_t cuda_rt;
  #endif

  inline ~coot_rt_t();
  inline  coot_rt_t();

  inline bool init(const bool print_info = false);
  inline bool init(const char*       filename, const bool print_info = false);
  inline bool init(const std::string filename, const bool print_info = false);
  inline bool init(const uword wanted_platform, const uword wanted_device, const bool print_info = false);

  #if defined(COOT_USE_CXX11)
                   coot_rt_t(const coot_rt_t&) = delete;
  coot_rt_t&       operator=(const coot_rt_t&) = delete;
  #endif

  /**
   * all of the functions below here are redirected to the current backend that is in use
   */

  template<typename eT>
  static inline dev_mem_t<eT> acquire_memory(const uword n_elem);

  template<typename eT>
  static inline void release_memory(dev_mem_t<eT> dev_mem);

  template<typename eT>
  static inline bool is_supported_type();

  static inline void set_rng_seed(const u64 seed);

  template<typename out_eT, typename in_eT>
  static inline void copy_array(dev_mem_t<out_eT> dest, dev_mem_t<in_eT> src, const uword n_elem);

  template<typename out_eT, typename in_eT>
  static inline void copy_subview(dev_mem_t<out_eT> dest, dev_mem_t<in_eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword M_n_cols, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline void reorder_cols(dev_mem_t<eT> out, const dev_mem_t<eT> mem, const uword n_rows, const dev_mem_t<uword> order, const uword out_n_cols);

  template<typename eT>
  static inline void extract_diag(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword in_mem_offset, const uword n_rows, const uword len);

  template<typename eT2, typename eT1>
  static inline void set_diag(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword in_mem_offset, const uword n_rows, const uword len);

  template<typename eT>
  static inline void copy_diag(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword out_mem_offset, const uword in_mem_offset, const uword out_n_rows, const uword in_n_rows, const uword len);

  template<typename eT>
  static inline void inplace_op_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem, const oneway_kernel_id::enum_id num);

  template<typename eT1, typename eT2>
  static inline void inplace_op_array(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_elem, const twoway_kernel_id::enum_id num);

  template<typename eT>
  static inline void inplace_op_subview(dev_mem_t<eT> dest, const eT val, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword M_n_rows, const oneway_kernel_id::enum_id num);

  template<typename eT>
  static inline void inplace_op_diag(dev_mem_t<eT> dest, const uword mem_offset, const eT val, const uword n_rows, const uword len, const oneway_kernel_id::enum_id num);

  template<typename eT1, typename eT2>
  static inline void inplace_op_subview(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const twoway_kernel_id::enum_id num, const char* identifier);

  template<typename eT>
  static inline void replace(dev_mem_t<eT> mem, const uword n_elem, const eT val_find, const eT val_replace);

  template<typename eT1, typename eT2>
  static inline void htrans(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_rows, const uword n_cols);

  template<typename eT1, typename eT2>
  static inline void strans(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline void fill_randu(dev_mem_t<eT> dest, const uword n);

  template<typename eT>
  static inline void fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd);

  template<typename eT>
  static inline void fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi);

  template<typename eT1, typename eT2, typename eT3>
  static inline void array_op(dev_mem_t<eT3> dest, const uword n_elem, const dev_mem_t<eT1> A_mem, const dev_mem_t<eT2> B_mem, const threeway_kernel_id::enum_id num);

  template<typename eT1, typename eT2>
  static inline void eop_scalar(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_elem, const eT1 aux_val_pre, const eT2 aux_val_post, const twoway_kernel_id::enum_id num);

  template<typename eT>
  static inline eT accu(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT accu_subview(const dev_mem_t<eT> mem, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline eT prod(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT min(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT max(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT max_abs(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT1, typename eT2>
  static inline bool all_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small);

  template<typename eT1, typename eT2>
  static inline void all(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise);

  template<typename eT1, typename eT2>
  static inline bool any_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small);

  template<typename eT>
  static inline bool any_vec(const dev_mem_t<eT> mem, const uword n_elem, const eT val, const oneway_real_kernel_id::enum_id num, const oneway_real_kernel_id::enum_id num_small);

  template<typename eT1, typename eT2>
  static inline void any(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise);

  template<typename eT1, typename eT2>
  static inline void relational_scalar_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const std::string& name);

  template<typename eT1>
  static inline void relational_unary_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const oneway_real_kernel_id::enum_id num, const std::string& name);

  template<typename eT1, typename eT2>
  static inline void relational_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> X_mem, const dev_mem_t<eT2> Y_mem, const uword n_elem, const twoway_kernel_id::enum_id num, const std::string& name);

  template<typename eT>
  static inline std::tuple<bool, std::string> chol(dev_mem_t<eT> out, const uword n_rows);

  template<typename eT>
  static inline std::tuple<bool, std::string> lu(dev_mem_t<eT> L, dev_mem_t<eT> U, dev_mem_t<eT> in, const bool pivoting, dev_mem_t<eT> P, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline std::tuple<bool, std::string> det(dev_mem_t<eT> A, const uword n_rows, eT& out_val);

  template<typename eT>
  static inline std::tuple<bool, std::string> svd(dev_mem_t<eT> U, dev_mem_t<eT> S, dev_mem_t<eT> V, dev_mem_t<eT> A, const uword n_rows, const uword n_cols, const bool compute_u_vt);

  template<typename eT>
  static inline std::tuple<bool, std::string> eig_sym(dev_mem_t<eT> mem, const uword n_rows, const bool eigenvectors, dev_mem_t<eT> eigenvalues);

  template<typename eT>
  static inline void copy_from_dev_mem(eT* dest, const dev_mem_t<eT> src, const uword N);

  template<typename eT>
  static inline void copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N);

  template<typename eT1, typename eT2>
  static inline void extract_subview(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline void eye(dev_mem_t<eT> out, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline eT get_val(const dev_mem_t<eT> mem, const uword index);

  template<typename eT>
  static inline void set_val(dev_mem_t<eT> mem, const uword index, const eT val);

  template<typename eT> static inline void   val_add_inplace(dev_mem_t<eT> mem, const uword index, const eT val);
  template<typename eT> static inline void val_minus_inplace(dev_mem_t<eT> mem, const uword index, const eT val);
  template<typename eT> static inline void   val_mul_inplace(dev_mem_t<eT> mem, const uword index, const eT val);
  template<typename eT> static inline void   val_div_inplace(dev_mem_t<eT> mem, const uword index, const eT val);

  template<typename eT, const bool do_trans_A, const bool do_trans_B>
  static inline void gemm(dev_mem_t<eT> C_mem, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT> A_mem, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> B_mem, const eT alpha, const eT beta);

  template<typename eT, const bool do_trans_A>
  static inline void gemv(dev_mem_t<eT> y_mem, const dev_mem_t<eT> A_mem, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> x_mem, const eT alpha, const eT beta);

  template<typename eT>
  static inline void mul_diag(dev_mem_t<eT> C_mem, const uword C_n_rows, const uword C_n_cols, const eT alpha, const dev_mem_t<eT> A_mem, const bool A_is_diag, const bool A_trans, const dev_mem_t<eT> B_mem, const bool B_is_diag, const bool B_trans);

  template<typename eT1, typename eT2>
  static inline void sum_colwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void sum_rowwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void sum_colwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword A_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void sum_rowwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword A_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void min_colwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void min_rowwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void min_colwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword A_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void min_rowwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword A_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void max_colwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void max_rowwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void max_colwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword A_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void max_rowwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword A_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT>
  static inline eT trace(const dev_mem_t<eT> mem, const uword n_rows, const uword n_cols);

  template<typename eT1, typename eT2>
  static inline typename promote_type<eT1, eT2>::result dot(const dev_mem_t<eT1> mem1, const dev_mem_t<eT2> mem2, const uword n_elem);

  template<typename eT1, typename eT2>
  static inline void repmat(const dev_mem_t<eT1> src, dev_mem_t<eT2> dest, const uword n_rows, const uword n_cols, const uword copies_per_row, const uword copies_per_col);

  template<typename eT>
  static inline void linspace(const dev_mem_t<eT> mem, const eT start, const eT end, const uword num);

  template<typename eT1, typename eT2>
  static inline void clamp(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const eT1 min_val, const eT1 max_val, const uword n_elem);

  template<typename eT>
  static inline eT vec_norm_1(dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT vec_norm_2(dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT vec_norm_k(dev_mem_t<eT> mem, const uword n_elem, const uword k);

  template<typename eT>
  static inline eT vec_norm_min(dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT1, typename eT2>
  static inline void mean(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword n_rows, const uword n_cols, const uword dim, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void mean_subview(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword M_n_rows, const uword start_row, const uword start_col, const uword n_rows, const uword n_cols, const uword dim, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void median(dev_mem_t<eT2> out, dev_mem_t<eT1> in, const uword n_rows, const uword n_cols, const uword dim);

  template<typename eT>
  static inline eT median_vec(dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline void var(dev_mem_t<eT> out, const dev_mem_t<eT> in, const dev_mem_t<eT> means, const uword n_rows, const uword n_cols, const uword dim, const uword norm_type);

  template<typename eT>
  static inline void var_subview(dev_mem_t<eT> out, const dev_mem_t<eT> in, const dev_mem_t<eT> means, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword dim, const uword norm_type);

  template<typename eT>
  static inline eT var_vec(const dev_mem_t<eT> mem, const eT mean, const uword n_elem, const uword norm_type);

  template<typename eT>
  static inline eT var_vec_subview(const dev_mem_t<eT> mem, const eT mean, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword norm_type);

  template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
  static inline void join_cols(dev_mem_t<eT5> out, const dev_mem_t<eT1> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT2> B, const uword B_n_rows, const uword B_n_cols, const dev_mem_t<eT3> C, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT4> D, const uword D_n_rows, const uword D_n_cols);

  template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
  static inline void join_rows(dev_mem_t<eT5> out, const dev_mem_t<eT1> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT2> B, const uword B_n_rows, const uword B_n_cols, const dev_mem_t<eT3> C, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT4> D, const uword D_n_rows, const uword D_n_cols);

  template<typename eT>
  static inline void sort_colwise(dev_mem_t<eT> mem, const uword n_rows, const uword n_cols, const uword sort_type);

  template<typename eT>
  static inline void sort_rowwise(dev_mem_t<eT> mem, const uword n_rows, const uword n_cols, const uword sort_type);

  template<typename eT>
  static inline void sort_vec(dev_mem_t<eT> mem, const uword n_elem, const uword sort_type);

  template<typename eT>
  static inline void sort_index_vec(dev_mem_t<uword> out, dev_mem_t<eT> mem, const uword n_elem, const uword sort_type, const uword stable_sort);

  template<typename eT>
  static inline void find(dev_mem_t<uword>& out, uword& out_len, const dev_mem_t<eT> A, const uword n_elem, const uword k, const uword find_type);

  template<typename eT1, typename eT2>
  static inline void symmat(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword size, const uword lower);

  static inline void synchronise();

  // RC-TODO: unified interface for some other operations?
  };

// Store coot_rt_t as a singleton.
inline coot_rt_t& get_rt()
  {
  static coot_rt_t rt;
  return rt;
  }
