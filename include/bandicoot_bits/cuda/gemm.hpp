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

    #ifdef COOT_USE_CUDA // should we also have a COOT_USE_CUBLAS?  I don't think it's needed
    cublasHandle_t handle;
    cublasCreate(&handle);

    // RC-TODO: handle complex?
    cublasOperation_t trans_a = (do_trans_A) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t trans_b = (do_trans_B) ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int M = int(C.n_rows);
    const int N = int(C.n_cols);
    const int K = (do_trans_A) ? int(A.n_rows) : int(A.n_cols);

    const int lda = (do_trans_A) ? K : M;
    const int ldb = (do_trans_B) ? N : K;
    const int ldc = int(C.n_rows);

    const eT* A_mem = A.get_dev_mem(false).cuda_mem_ptr;
    const eT* B_mem = B.get_dev_mem(false).cuda_mem_ptr;
    eT* C_mem = C.get_dev_mem(false).cuda_mem_ptr;

    cublasStatus_t result;

    if (std::is_same<eT, float>::value)
      {
      result = cublasSgemm(handle,
                           trans_a,
                           trans_b,
                           M,
                           N,
                           K,
                           (const float*) &alpha,
                           (const float*) A_mem,
                           lda,
                           (const float*) B_mem,
                           ldb,
                           (const float*) &beta,
                           (float*) C_mem,
                           ldc);
      }
    else if (std::is_same<eT, double>::value)
      {
      result = cublasDgemm(handle,
                           trans_a,
                           trans_b,
                           M,
                           N,
                           K,
                           (const double*) &alpha,
                           (const double*) A_mem,
                           lda,
                           (const double*) B_mem,
                           ldb,
                           (const double*) &beta,
                           (double*) C_mem,
                           ldc);
      }
    else if (std::is_same<eT, std::complex<float>>::value)
      {
      // RC-TODO: this
      throw std::runtime_error("complex not implemented yet");
      }
    else if (std::is_same<eT, std::complex<double>>::value)
      {
      // RC-TODO: this
      throw std::runtime_error("complex not implemented yet");
      }
    else
      {
      // RC-TODO: what about __half from cuBLAS?
      // RC-TODO: actual error message
      cublasDestroy(handle);
      throw std::invalid_argument("cannot multiply with this eT");
      }

    coot_check_cublas_error( result, "cuda::gemm::apply(): call to cublas?gemm() failed" );

    cublasDestroy(handle);
    #else
    cuda_wrapper::throw_unsupported();
    #endif
    }
  };
