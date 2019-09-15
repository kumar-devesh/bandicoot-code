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



inline
coot_rt_t::~coot_rt_t()
  {
  coot_extra_debug_sigprint_this(this);
  }



inline
coot_rt_t::coot_rt_t()
  {
  coot_extra_debug_sigprint_this(this);
  backend = CL_BACKEND;
  }



template<typename eT>
inline
dev_mem_t<eT>
coot_rt_t::acquire_memory(const uword n_elem)
  {
  coot_extra_debug_sigprint();
  
//  coot_check_runtime_error( (valid == false), "coot_rt::acquire_memory(): runtime not valid" );
  
  if(n_elem == 0)  { return dev_mem_t<eT>({ NULL }); }
  
  coot_debug_check
   (
   ( size_t(n_elem) > (std::numeric_limits<size_t>::max() / sizeof(eT)) ),
   "coot_rt::acquire_memory(): requested size is too large"
   );

  // use either OpenCL or CUDA backend
  dev_mem_t<eT> result;

  if (get_rt().backend == CUDA_BACKEND)
    {
    result.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_elem);
    }
  else
    {
    result.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_elem);
    }

  return result;
  }



template<typename eT>
inline
void
coot_rt_t::release_memory(dev_mem_t<eT> dev_mem)
  {
  coot_extra_debug_sigprint();
  
//  coot_debug_check( (valid == false), "coot_rt not valid" );
  
  if (get_rt().backend == CL_BACKEND)
    {
    get_rt().cl_rt.release_memory(dev_mem.cl_mem_ptr);
    }
  else
    {
    get_rt().cuda_rt.release_memory(dev_mem.cuda_mem_ptr);
    }
  }



inline
void
coot_rt_t::synchronise()
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    get_rt().cl_rt.synchronise();
    }
  else
    {
    get_rt().cuda_rt.synchronise();
    }
  }



template<typename eT>
inline
void
coot_rt_t::copy_array(dev_mem_t<eT> dest, dev_mem_t<eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CUDA_BACKEND)
    {
    cuda::copy_array(dest, src, n_elem);
    }
  else
    {
    opencl::copy_array(dest, src, n_elem);
    }
  }



template<typename eT>
inline
void
coot_rt_t::inplace_op_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem, const kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CUDA_BACKEND)
    {
    cuda::inplace_op_scalar(dest, val, n_elem, num);
    }
  else
    {
    opencl::inplace_op_scalar(dest, val, n_elem, num);
    }
  }



template<typename eT>
inline
void
coot_rt_t::inplace_op_array(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword n_elem, const kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CUDA_BACKEND)
    {
    cuda::inplace_op_array(dest, src, n_elem, num);
    }
  else
    {
    opencl::inplace_op_array(dest, src, n_elem, num);
    }
  }



template<typename eT>
inline
void
coot_rt_t::inplace_op_subview(dev_mem_t<eT> dest, const eT val, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword M_n_rows, const kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::inplace_op_subview(dest, val, aux_row1, aux_col1, n_rows, n_cols, M_n_rows, num);
    }
  else
    {
    cuda::inplace_op_subview(dest, val, aux_row1, aux_col1, n_rows, n_cols, M_n_rows, num);
    }
  }



template<typename eT>
inline
void
coot_rt_t::inplace_op_subview(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const kernel_id::enum_id num, const char* identifier)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::inplace_op_subview(dest, src, M_n_rows, aux_row1, aux_col1, n_rows, n_cols, num, identifier);
    }
  else
    {
    cuda::inplace_op_subview(dest, src, M_n_rows, aux_row1, aux_col1, n_rows, n_cols, num, identifier);
    }
  }



template<typename eT>
inline
void
coot_rt_t::fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::fill_randu(dest, n);
    }
  else
    {
    cuda::fill_randu(dest, n);
    }
  }



template<typename eT>
inline
void
coot_rt_t::fill_randn(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::fill_randn(dest, n);
    }
  else
    {
    cuda::fill_randn(dest, n);
    }
  }



template<typename eT>
inline
void
coot_rt_t::array_op(dev_mem_t<eT> dest, const uword n_elem, const dev_mem_t<eT> A_mem, const dev_mem_t<eT> B_mem, const kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::array_op(dest, n_elem, A_mem, B_mem, num);
    }
  else
    {
    cuda::array_op(dest, n_elem, A_mem, B_mem, num);
    }
  }



template<typename eT>
inline
void
coot_rt_t::eop_scalar(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword n_elem, const eT aux_val, const kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::eop_scalar(dest, src, n_elem, aux_val, num);
    }
  else
    {
    cuda::eop_scalar(dest, src, n_elem, aux_val, num);
    }
  }



template<typename eT>
inline
eT
coot_rt_t::accu_chunked(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    return opencl::accu_chunked(mem, n_elem);
    }
  else
    {
    return cuda::accu_chunked(mem, n_elem);
    }
  }



template<typename eT>
inline
eT
coot_rt_t::accu_simple(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    return opencl::accu_simple(mem, n_elem);
    }
  else
    {
    return cuda::accu_simple(mem, n_elem);
    }
  }



template<typename eT>
inline
eT
coot_rt_t::accu_subview(const dev_mem_t<eT> mem, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    return opencl::accu_subview(mem, M_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    }
  else
    {
    return cuda::accu_subview(mem, M_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    }
  }



template<typename eT>
inline
bool
coot_rt_t::chol(dev_mem_t<eT> out, const uword n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    return opencl::chol(out, n_rows);
    }
  else
    {
    return cuda::chol(out, n_rows);
    }
  }



template<typename eT>
inline
void
coot_rt_t::copy_from_dev_mem(eT* dest, const dev_mem_t<eT> src, const uword N)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::copy_from_dev_mem(dest, src, N);
    }
  else
    {
    cuda::copy_from_dev_mem(dest, src, N);
    }
  }



template<typename eT>
inline
void
coot_rt_t::copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::copy_into_dev_mem(dest, src, N);
    }
  else
    {
    cuda::copy_into_dev_mem(dest, src, N);
    }
  }



template<typename eT>
inline
void
coot_rt_t::extract_subview(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::extract_subview(out, in, M_n_rows, M_n_cols, aux_row1, aux_col1, n_rows, n_cols);
    }
  else
    {
    cuda::extract_subview(out, in, M_n_rows, M_n_cols, aux_row1, aux_col1, n_rows, n_cols);
    }
  }



template<typename eT>
inline
void
coot_rt_t::eye(dev_mem_t<eT> mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CUDA_BACKEND)
    {
    cuda::eye(mem, n_rows, n_cols);
    }
  else
    {
    opencl::eye(mem, n_rows, n_cols);
    }
  }



template<typename eT>
inline
eT
coot_rt_t::get_val(const dev_mem_t<eT> mem, const uword index)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    return opencl::get_val(mem, index);
    }
  else
    {
    return cuda::get_val(mem, index);
    }
  }



template<typename eT>
inline
void
coot_rt_t::set_val(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::set_val(mem, index, val);
    }
  else
    {
    cuda::set_val(mem, index, val);
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_add_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::val_add_inplace(mem, index, val);
    }
  else
    {
    cuda::val_add_inplace(mem, index, val);
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_minus_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::val_minus_inplace(mem, index, val);
    }
  else
    {
    cuda::val_minus_inplace(mem, index, val);
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_mul_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::val_mul_inplace(mem, index, val);
    }
  else
    {
    cuda::val_mul_inplace(mem, index, val);
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_div_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::val_div_inplace(mem, index, val);
    }
  else
    {
    cuda::val_div_inplace(mem, index, val);
    }
  }



template<typename eT, const bool do_trans_A, const bool do_trans_B>
inline
void
coot_rt_t::gemm(dev_mem_t<eT> C_mem, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT> A_mem, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> B_mem, const eT alpha, const eT beta)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::gemm<do_trans_A, do_trans_B>::apply(C_mem, C_n_rows, C_n_cols, A_mem, A_n_rows, A_n_cols, B_mem, alpha, beta);
    }
  else
    {
    cuda::gemm<do_trans_A, do_trans_B>::apply(C_mem, C_n_rows, C_n_cols, A_mem, A_n_rows, A_n_cols, B_mem, alpha, beta);
    }
  }



template<typename eT, const bool do_trans_A>
inline
void
coot_rt_t::gemv(dev_mem_t<eT> y_mem, const dev_mem_t<eT> A_mem, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> x_mem, const eT alpha, const eT beta)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::gemv<do_trans_A>::apply(y_mem, A_mem, A_n_rows, A_n_cols, x_mem, alpha, beta);
    }
  else
    {
    cuda::gemv<do_trans_A>::apply(y_mem, A_mem, A_n_rows, A_n_cols, x_mem, alpha, beta);
    }
  }



template<typename eT>
inline
void
coot_rt_t::sum_colwise(dev_mem_t<eT> out_mem, const dev_mem_t<eT> A_mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::sum_colwise(out_mem, A_mem, n_rows, n_cols);
    }
  else
    {
    cuda::sum_colwise(out_mem, A_mem, n_rows, n_cols);
    }
  }



template<typename eT>
inline
void
coot_rt_t::sum_rowwise(dev_mem_t<eT> out_mem, const dev_mem_t<eT> A_mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::sum_rowwise(out_mem, A_mem, n_rows, n_cols);
    }
  else
    {
    cuda::sum_rowwise(out_mem, A_mem, n_rows, n_cols);
    }
  }



template<typename et>
inline
void
coot_rt_t::sum_colwise_subview(dev_mem_t<et> out_mem, const dev_mem_t<et> a_mem, const uword a_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::sum_colwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    }
  else
    {
    cuda::sum_colwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    }
  }



template<typename et>
inline
void
coot_rt_t::sum_rowwise_subview(dev_mem_t<et> out_mem, const dev_mem_t<et> a_mem, const uword a_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::sum_rowwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    }
  else
    {
    cuda::sum_rowwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    }
  }



template<typename eT>
inline
eT
coot_rt_t::trace(const dev_mem_t<eT> mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    return opencl::trace(mem, n_rows, n_cols);
    }
  else
    {
    return cuda::trace(mem, n_rows, n_cols);
    }
  }




