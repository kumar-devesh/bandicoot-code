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



// this can hold either CUDA memory or CL memory
template<typename eT>
union dev_mem_t
  {
  cl_mem cl_mem_ptr;
  eT* cuda_mem_ptr;
  };

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

  template<typename out_eT, typename in_eT>
  static inline void copy_array(dev_mem_t<out_eT> dest, dev_mem_t<in_eT> src, const uword n_elem);

  template<typename eT>
  static inline void inplace_op_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem, const oneway_kernel_id::enum_id num);

  template<typename eT1, typename eT2>
  static inline void inplace_op_array(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_elem, const twoway_kernel_id::enum_id num);

  template<typename eT>
  static inline void inplace_op_subview(dev_mem_t<eT> dest, const eT val, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword M_n_rows, const oneway_kernel_id::enum_id num);

  template<typename eT1, typename eT2>
  static inline void inplace_op_subview(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const twoway_kernel_id::enum_id num, const char* identifier);

  template<typename eT>
  static inline void fill_randu(dev_mem_t<eT> dest, const uword n);

  template<typename eT>
  static inline void fill_randn(dev_mem_t<eT> dest, const uword n);

  template<typename eT1, typename eT2, typename eT3>
  static inline void array_op(dev_mem_t<eT3> dest, const uword n_elem, const dev_mem_t<eT1> A_mem, const dev_mem_t<eT2> B_mem, const threeway_kernel_id::enum_id num);

  template<typename eT1, typename eT2>
  static inline void eop_scalar(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_elem, const eT1 aux_val_pre, const eT2 aux_val_post, const twoway_kernel_id::enum_id num);

  template<typename eT>
  static inline eT accu_chunked(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT accu_simple(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT accu_subview(const dev_mem_t<eT> mem, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline bool chol(dev_mem_t<eT> out, const uword n_rows);

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

  template<typename eT1, typename eT2>
  static inline void sum_colwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void sum_rowwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void sum_colwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword A_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT1, typename eT2>
  static inline void sum_rowwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword A_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply);

  template<typename eT>
  static inline eT trace(const dev_mem_t<eT> mem, const uword n_rows, const uword n_cols);

  template<typename eT1, typename eT2>
  static inline typename promote_type<eT1, eT2>::result dot(const dev_mem_t<eT1> mem1, const dev_mem_t<eT2> mem2, const uword n_elem);

  static inline void synchronise();

  // RC-TODO: unified interface for some other operations?
  };

// Store coot_rt_t as a singleton.
inline coot_rt_t& get_rt()
  {
  static coot_rt_t rt;
  return rt;
  }
