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



template<bool do_trans_A = false, bool do_trans_B = false>
struct gemm
  {

  template<typename eT>
  static
  inline
  void
  apply(dev_mem_t<eT> C_mem, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT> A_mem, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> B_mem, eT alpha, eT beta)
    {
    coot_extra_debug_sigprint();

    #ifdef COOT_USE_CUDA // should we also have a COOT_USE_CUBLAS?  I don't think it's needed

    // RC-TODO: handle complex?
    cublasOperation_t trans_a = (do_trans_A) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t trans_b = (do_trans_B) ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int M = int(C_n_rows);
    const int N = int(C_n_cols);
    const int K = (do_trans_A) ? int(A_n_rows) : int(A_n_cols);

    const int lda = (do_trans_A) ? K : M;
    const int ldb = (do_trans_B) ? N : K;
    const int ldc = int(C_n_rows);

    cublasStatus_t result;

    if (std::is_same<eT, float>::value)
      {
      result = coot_wrapper(cublasSgemm)(get_rt().cuda_rt.cublas_handle,
                                         trans_a,
                                         trans_b,
                                         M,
                                         N,
                                         K,
                                         (const float*) &alpha,
                                         (const float*) A_mem.cuda_mem_ptr,
                                         lda,
                                         (const float*) B_mem.cuda_mem_ptr,
                                         ldb,
                                         (const float*) &beta,
                                         (float*) C_mem.cuda_mem_ptr,
                                         ldc);
      }
    else if (std::is_same<eT, double>::value)
      {
      result = coot_wrapper(cublasDgemm)(get_rt().cuda_rt.cublas_handle,
                                         trans_a,
                                         trans_b,
                                         M,
                                         N,
                                         K,
                                         (const double*) &alpha,
                                         (const double*) A_mem.cuda_mem_ptr,
                                         lda,
                                         (const double*) B_mem.cuda_mem_ptr,
                                         ldb,
                                         (const double*) &beta,
                                         (double*) C_mem.cuda_mem_ptr,
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
      throw std::invalid_argument("cannot multiply with this eT");
      }

    coot_check_cublas_error( result, "coot::cuda::gemm(): call to cublas?gemm() failed" );

    #else
    throw std::invalid_argument("coot::cuda::gemm(): CUDA backend not enabled");
    #endif
    }
  };
