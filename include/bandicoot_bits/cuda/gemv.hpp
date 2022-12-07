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

    coot_stop_runtime_error("coot::cuda::gemv(): unsupported type");
    }



  static
  inline
  void
  apply(dev_mem_t<float> y, const dev_mem_t<float> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<float> x, const float alpha, const float beta)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A);  // TODO: adapt this assert for size_t

    cublasOperation_t trans_a = (do_trans_A) ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int M = int(A_n_rows);
    const int N = int(A_n_cols);

    const int lda = M;
    const int incx = 1;
    const int incy = 1;

    cublasStatus_t result;

    result = cublasSgemv(get_rt().cuda_rt.cublas_handle, trans_a, M, N, (float*) &alpha, A.cuda_mem_ptr, lda, x.cuda_mem_ptr, incx, (float*) &beta, y.cuda_mem_ptr, incy);

    coot_check_cublas_error( result, "coot::cuda::gemv(): call to cublasSgemv() failed" );
    }



  static
  inline
  void
  apply(dev_mem_t<double> y, const dev_mem_t<double> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<double> x, const double alpha, const double beta)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A); // TODO: adapt this assert for size_t

    cublasOperation_t trans_a = (do_trans_A) ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int M = int(A_n_rows);
    const int N = int(A_n_cols);

    const int lda = M;
    const int incx = 1;
    const int incy = 1;

    cublasStatus_t result;

    result = cublasDgemv(get_rt().cuda_rt.cublas_handle, trans_a, M, N, (double*) &alpha, A.cuda_mem_ptr, lda, x.cuda_mem_ptr, incx, (double*) &beta, y.cuda_mem_ptr, incy);

    coot_check_cublas_error( result, "coot::cuda::gemv(): call to cublasSgemv() failed" );
    }
  };



//! @}
