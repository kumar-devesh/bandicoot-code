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
  apply(dev_mem_t<eT> C, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> B, eT alpha, eT beta)
    {
    coot_extra_debug_sigprint();

    coot_stop_runtime_error("coot::opencl::gemm(): unsupported type");
    }



  static
  inline
  void
  apply(dev_mem_t<float> C, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<float> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<float> B, float alpha, float beta)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A,B);  // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;
    const clblasTranspose transB = (do_trans_B) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(C_n_rows);
    const size_t N = size_t(C_n_cols);
    const size_t K = (do_trans_A) ? size_t(A_n_rows) : size_t(A_n_cols);

    const size_t lda = (do_trans_A) ? K : M;
    const size_t ldb = (do_trans_B) ? N : K;
    const size_t ldc = size_t(C_n_rows);

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= coot_wrapper(clblasSgemm)(clblasColumnMajor, transA, transB, M, N, K, alpha, A.cl_mem_ptr, 0, lda, B.cl_mem_ptr, 0, ldb, beta, C.cl_mem_ptr, 0, ldc, 1, &queue, 0, NULL, NULL);
    status |= coot_wrapper(clFlush)(queue);

    coot_check_cl_error(status, "coot::opencl::gemm(): eT = float");
    }



  static
  inline
  void
  apply(dev_mem_t<double> C, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<double> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<double> B, double alpha, double beta)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A,B);  // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;
    const clblasTranspose transB = (do_trans_B) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(C_n_rows);
    const size_t N = size_t(C_n_cols);
    const size_t K = (do_trans_A) ? size_t(A_n_rows) : size_t(A_n_cols);

    const size_t lda = (do_trans_A) ? K : M;
    const size_t ldb = (do_trans_B) ? N : K;
    const size_t ldc = size_t(C_n_rows);

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= coot_wrapper(clblasDgemm)(clblasColumnMajor, transA, transB, M, N, K, alpha, A.cl_mem_ptr, 0, lda, B.cl_mem_ptr, 0, ldb, beta, C.cl_mem_ptr, 0, ldc, 1, &queue, 0, NULL, NULL);
    status |= coot_wrapper(clFlush)(queue);

    coot_check_cl_error(status, "coot::opencl::gemm(): eT = double");
    }
  };
