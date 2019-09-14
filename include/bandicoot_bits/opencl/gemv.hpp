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
  apply(Mat<eT>& y, const Mat<eT>& A, const Mat<eT>& x, const eT alpha = eT(1.0), const eT beta = eT(1.0))
    {
    coot_extra_debug_sigprint();

    coot_stop_runtime_error("opencl::gemv(): unsupported type");
    }



  static
  inline
  void
  apply(Mat<float>& y, const Mat<float>& A, const Mat<float>& x, const float alpha = 1.0f, const float beta = 1.0f)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A);  // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(A.n_rows);
    const size_t N = size_t(A.n_cols);

    const size_t lda = size_t(A.n_rows);
    const size_t inc = size_t(1);

    cl_mem A_mem = A.get_dev_mem(false).cl_mem_ptr;
    cl_mem x_mem = x.get_dev_mem(false).cl_mem_ptr;
    cl_mem y_mem = y.get_dev_mem(false).cl_mem_ptr;

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= clblasSgemv(clblasColumnMajor, transA, M, N, alpha, A_mem, 0, lda, x_mem, 0, inc, beta, y_mem, 0, inc, 1, &queue, 0, NULL, NULL);
    status |= clFlush(queue);

    coot_check_cl_error(status, "opencl::gemv(): eT = float");
    }



  static
  inline
  void
  apply(Mat<double>& y, const Mat<double>& A, const Mat<double>& x, const double alpha = 1.0, const double beta = 1.0)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A); // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(A.n_rows);
    const size_t N = size_t(A.n_cols);

    const size_t lda = size_t(A.n_rows);
    const size_t inc = size_t(1);

    cl_mem A_mem = A.get_dev_mem(false).cl_mem_ptr;
    cl_mem x_mem = x.get_dev_mem(false).cl_mem_ptr;
    cl_mem y_mem = y.get_dev_mem(false).cl_mem_ptr;

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= clblasDgemv(clblasColumnMajor, transA, M, N, alpha, A_mem, 0, lda, x_mem, 0, inc, beta, y_mem, 0, inc, 1, &queue, 0, NULL, NULL);
    status |= clFlush(queue);

    coot_check_cl_error(status, "opencl::gemv(): eT = double");
    }
  };



//! @}
