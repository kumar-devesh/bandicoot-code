// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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



/**
 * Compute the row-wise or column-wise variance of the input matrix, storing the result in the output matrix.
 */
template<typename eT>
inline
void
var(dev_mem_t<eT> out, const dev_mem_t<eT> in, const dev_mem_t<eT> means, const uword n_rows, const uword n_cols, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::var(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>((dim == 0) ? oneway_kernel_id::var_colwise : oneway_kernel_id::var_rowwise);
  const uword norm_correction = (norm_type == 0) ? 1 : 0;

  cl_int status = 0;

  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_norm_correction(norm_correction);

  status |= clSetKernelArg(k, 0, sizeof(cl_mem),          &(out.cl_mem_ptr));
  status |= clSetKernelArg(k, 1, sizeof(cl_mem),          &(in.cl_mem_ptr));
  status |= clSetKernelArg(k, 2, sizeof(cl_mem),          &(means.cl_mem_ptr));
  status |= clSetKernelArg(k, 3, cl_n_rows.size,          cl_n_rows.addr);
  status |= clSetKernelArg(k, 4, cl_n_cols.size,          cl_n_cols.addr);
  status |= clSetKernelArg(k, 5, cl_norm_correction.size, cl_norm_correction.addr);

  coot_check_cl_error(status, "coot::opencl::var(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { (dim == 0) ? n_cols : n_rows };

  status = clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::var(): failed to run kernel");
  }



template<typename eT>
inline
void
var_subview(dev_mem_t<eT> out, const dev_mem_t<eT> in, const dev_mem_t<eT> means, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();
  coot_ignore(M_n_cols);

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::var_subview(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>((dim == 0) ? oneway_kernel_id::submat_var_colwise : oneway_kernel_id::submat_var_rowwise);
  const uword norm_correction = (norm_type == 0) ? 1 : 0;

  cl_int status = 0;

  runtime_t::adapt_uword cl_M_n_rows(M_n_rows);
  runtime_t::adapt_uword cl_aux_row1(aux_row1);
  runtime_t::adapt_uword cl_aux_col1(aux_col1);
  runtime_t::adapt_uword cl_n_rows(n_rows);
  runtime_t::adapt_uword cl_n_cols(n_cols);
  runtime_t::adapt_uword cl_norm_correction(norm_correction);

  status |= clSetKernelArg(k, 0, sizeof(cl_mem),          &(out.cl_mem_ptr));
  status |= clSetKernelArg(k, 1, sizeof(cl_mem),          &(in.cl_mem_ptr));
  status |= clSetKernelArg(k, 2, sizeof(cl_mem),          &(means.cl_mem_ptr));
  status |= clSetKernelArg(k, 3, cl_M_n_rows.size,        cl_M_n_rows.addr);
  status |= clSetKernelArg(k, 4, cl_aux_row1.size,        cl_aux_row1.addr);
  status |= clSetKernelArg(k, 5, cl_aux_col1.size,        cl_aux_col1.addr);
  status |= clSetKernelArg(k, 6, cl_n_rows.size,          cl_n_rows.addr);
  status |= clSetKernelArg(k, 7, cl_n_cols.size,          cl_n_cols.addr);
  status |= clSetKernelArg(k, 8, cl_norm_correction.size, cl_norm_correction.addr);

  coot_check_cl_error(status, "coot::opencl::var_subview(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { (dim == 0) ? n_cols : n_rows };

  status = clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::var_subview(): failed to run kernel");
  }



template<typename eT>
inline
eT
var_vec(const dev_mem_t<eT> mem, const eT mean, const uword n_elem, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::var);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::var_small);

  cl_kernel accu_k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu);
  cl_kernel accu_k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  const eT result = generic_reduce<eT, eT>(mem,
                                           n_elem,
                                           "var_vec",
                                           k,
                                           k_small,
                                           std::make_tuple(mean),
                                           accu_k,
                                           accu_k_small,
                                           std::make_tuple(/* no extra args for second and later passes */));
  const uword norm_correction = (norm_type == 0) ? 1 : 0;
  return result / ((eT) (n_elem - norm_correction));
  }



template<typename eT>
inline
eT
var_vec_subview(const dev_mem_t<eT> mem, const eT mean, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword norm_type)
  {
  coot_extra_debug_sigprint();
  coot_ignore(M_n_cols);

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::submat_var);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::submat_var_small);

  cl_kernel accu_k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu);
  cl_kernel accu_k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  const uword submat_n_elem = n_rows * n_cols;

  const eT result = generic_reduce<eT, eT>(mem,
                                           submat_n_elem,
                                           "var_vec_subview",
                                           k,
                                           k_small,
                                           std::make_tuple(mean, M_n_rows, aux_row1, aux_col1, n_rows, n_cols),
                                           accu_k,
                                           accu_k_small,
                                           std::make_tuple(/* no extra args for second and later passes */));
  const uword norm_correction = (norm_type == 0) ? 1 : 0;
  return result / ((eT) (submat_n_elem - norm_correction));
  }
