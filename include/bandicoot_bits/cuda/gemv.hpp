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

    coot_stop_runtime_error("cuda::gemv(): unsupported type");
    }



  static
  inline
  void
  apply(Mat<float>& y, const Mat<float>& A, const Mat<float>& x, const float alpha = 1.0f, const float beta = 1.0f)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A);  // TODO: adapt this assert for size_t

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasOperation_t trans_a = (do_trans_A) ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int M = int(A.n_rows);
    const int N = int(A.n_cols);
    const int K = (do_trans_A) ? int(A.n_rows) : int(A.n_cols);

    const int lda = M;
    const int incx = 1;
    const int incy = 1;

    const float* A_mem = A.get_dev_mem(false).cuda_mem_ptr;
    const float* x_mem = x.get_dev_mem(false).cuda_mem_ptr;
    float* y_mem = y.get_dev_mem(false).cuda_mem_ptr;

    cublasStatus_t result;

    result = cublasSgemv(handle, trans_a, M, N, (float*) &alpha, A_mem, lda, x_mem, incx, (float*) &beta, y_mem, incy);

    // TODO: handle errors

    cublasDestroy(handle);
    }



  static
  inline
  void
  apply(Mat<double>& y, const Mat<double>& A, const Mat<double>& x, const double alpha = 1.0, const double beta = 1.0)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A); // TODO: adapt this assert for size_t

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasOperation_t trans_a = (do_trans_A) ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int M = int(A.n_rows);
    const int N = int(A.n_cols);
    const int K = (do_trans_A) ? int(A.n_rows) : int(A.n_cols);

    const int lda = M;
    const int incx = 1;
    const int incy = 1;

    const double* A_mem = A.get_dev_mem(false).cuda_mem_ptr;
    const double* x_mem = x.get_dev_mem(false).cuda_mem_ptr;
    double* y_mem = y.get_dev_mem(false).cuda_mem_ptr;

    cublasStatus_t result;

    result = cublasDgemv(handle, trans_a, M, N, (double*) &alpha, A_mem, lda, x_mem, incx, (double*) &beta, y_mem, incy);

    // TODO: handle errors

    cublasDestroy(handle);
    }
  };



//! @}
