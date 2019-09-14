// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
//~
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//~
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



template<bool do_trans_A = false, bool do_trans_B = false>
struct gemm
  {

  template<typename eT>
  static
  inline
  void
  apply(Mat<eT>& C, const Mat<eT>& A, const Mat<eT>& B, eT alpha = eT(1.0), eT beta = eT(0.0))
    {
    coot_extra_debug_sigprint();

    coot_stop_runtime_error("opencl::gemm(): unsupported type");
    }



  static
  inline
  void
  apply(Mat<float>& C, const Mat<float>& A, const Mat<float>& B, float alpha = 1.0f, float beta = 1.0f)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A,B);  // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;
    const clblasTranspose transB = (do_trans_B) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(C.n_rows);
    const size_t N = size_t(C.n_cols);
    const size_t K = (do_trans_A) ? size_t(A.n_rows) : size_t(A.n_cols);

    const size_t lda = (do_trans_A) ? K : M;
    const size_t ldb = (do_trans_B) ? N : K;
    const size_t ldc = size_t(C.n_rows);

    cl_mem A_mem = A.get_dev_mem(false).cl_mem_ptr;
    cl_mem B_mem = B.get_dev_mem(false).cl_mem_ptr;
    cl_mem C_mem = C.get_dev_mem(false).cl_mem_ptr;

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= clblasSgemm(clblasColumnMajor, transA, transB, M, N, K, alpha, A_mem, 0, lda, B_mem, 0, ldb, beta, C_mem, 0, ldc, 1, &queue, 0, NULL, NULL);
    status |= clFlush(queue);

    coot_check_cl_error(status, "gemm::apply(): eT = float");
    }



  static
  inline
  void
  apply(Mat<double>& C, const Mat<double>& A, const Mat<double>& B, double alpha = 1.0, double beta = 1.0)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A,B);  // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;
    const clblasTranspose transB = (do_trans_B) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(C.n_rows);
    const size_t N = size_t(C.n_cols);
    const size_t K = (do_trans_A) ? size_t(A.n_rows) : size_t(A.n_cols);

    const size_t lda = (do_trans_A) ? K : M;
    const size_t ldb = (do_trans_B) ? N : K;
    const size_t ldc = size_t(C.n_rows);

    cl_mem A_mem = A.get_dev_mem(false).cl_mem_ptr;
    cl_mem B_mem = B.get_dev_mem(false).cl_mem_ptr;
    cl_mem C_mem = C.get_dev_mem(false).cl_mem_ptr;

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= clblasDgemm(clblasColumnMajor, transA, transB, M, N, K, alpha, A_mem, 0, lda, B_mem, 0, ldb, beta, C_mem, 0, ldc, 1, &queue, 0, NULL, NULL);
    status |= clFlush(queue);

    coot_check_cl_error(status, "gemm::apply(): eT = double");
    }
  };
