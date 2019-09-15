// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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

//! \addtogroup gemv
//! @{



template<bool do_trans_A = false>
struct gemv
  {

  template<typename eT>
  static
  inline
  void
  apply(dev_mem_t<eT> y, const dev_mem_t<eT> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> x, const eT alpha, const eT beta)
    {
    coot_extra_debug_sigprint();

    coot_stop_runtime_error("opencl::gemv(): unsupported type");
    }



  static
  inline
  void
  apply(dev_mem_t<float> y, const dev_mem_t<float> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<float> x, const float alpha, const float beta)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A);  // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(A_n_rows);
    const size_t N = size_t(A_n_cols);

    const size_t lda = size_t(A_n_rows);
    const size_t inc = size_t(1);

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= clblasSgemv(clblasColumnMajor, transA, M, N, alpha, A.cl_mem_ptr, 0, lda, x.cl_mem_ptr, 0, inc, beta, y.cl_mem_ptr, 0, inc, 1, &queue, 0, NULL, NULL);
    status |= clFlush(queue);

    coot_check_cl_error(status, "opencl::gemv(): eT = float");
    }



  static
  inline
  void
  apply(dev_mem_t<double> y, const dev_mem_t<double> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<double> x, const double alpha, const double beta)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A); // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(A_n_rows);
    const size_t N = size_t(A_n_cols);

    const size_t lda = size_t(A_n_rows);
    const size_t inc = size_t(1);

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= clblasDgemv(clblasColumnMajor, transA, M, N, alpha, A.cl_mem_ptr, 0, lda, x.cl_mem_ptr, 0, inc, beta, y.cl_mem_ptr, 0, inc, 1, &queue, 0, NULL, NULL);
    status |= clFlush(queue);

    coot_check_cl_error(status, "opencl::gemv(): eT = double");
    }
  };



//! @}
