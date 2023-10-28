// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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

// Definitions of test-specific functions from LAPACK.
// Note that the test program here expects COOT_USE_WRAPPER to be false.

namespace coot
  {

namespace lapack_test
  {



  //
  // reconstruct from bidiagonal form
  //
  template<typename eT>
  inline
  void
  bdt01(const blas_int m, const blas_int n, const blas_int kd, const eT* A, const blas_int lda, const eT* Q, const blas_int ldq, const eT* d, const eT* e, const eT* pt, const blas_int ldpt, eT* work, eT* resid)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    // No hidden arguments.
    if     ( is_float<eT>::value) { coot_fortran(coot_sbdt01)(&m, &n, &kd,  (const float*) A, &lda,  (const float*) Q, &ldq,  (const float*) d,  (const float*) e,  (const float*) pt, &ldpt,  (float*) work,  (float*) resid); }
    else if(is_double<eT>::value) { coot_fortran(coot_dbdt01)(&m, &n, &kd, (const double*) A, &lda, (const double*) Q, &ldq, (const double*) d, (const double*) e, (const double*) pt, &ldpt, (double*) work, (double*) resid); }
    }



  //
  // check that matrix is orthogonal
  //
  template<typename eT>
  inline
  void
  ort01(const char rowcol, const blas_int m, const blas_int n, const eT* u, const blas_int ldu, eT* work, const blas_int lwork, eT* resid)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sort01)(&rowcol, &m, &n,  (const float*) u, &ldu,  (float*) work, &lwork,  (float*) resid, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dort01)(&rowcol, &m, &n, (const double*) u, &ldu, (double*) work, &lwork, (double*) resid, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sort01)(&rowcol, &m, &n,  (const float*) u, &ldu,  (float*) work, &lwork,  (float*) resid); }
      else if(is_double<eT>::value) { coot_fortran(coot_dort01)(&rowcol, &m, &n, (const double*) u, &ldu, (double*) work, &lwork, (double*) resid); }
      }
    #endif
    }



  //
  // QL factorisation of real matrix
  //
  template<typename eT>
  inline
  void
  geqlf(const blas_int m, const blas_int n, eT* A, const blas_int lda, eT* tau, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    if     ( is_float<eT>::value) { coot_fortran(coot_sgeqlf)(&m, &n,  (float*) A, &lda,  (float*) tau,  (float*) work, &lwork, info); }
    else if(is_double<eT>::value) { coot_fortran(coot_dgeqlf)(&m, &n, (double*) A, &lda, (double*) tau, (double*) work, &lwork, info); }
    }



  //
  // multiply matrix C by orthogonal matrix Q, which came from gelqf
  //
  template<typename eT>
  inline
  void
  ormql(const char side, const char trans, const blas_int m, const blas_int n, const blas_int k, const eT* A, const blas_int lda, const eT* tau, eT* C, const blas_int ldc, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormql)(&side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info, 1, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormql)(&side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info, 1, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormql)(&side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormql)(&side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info); }
      }
    #endif
    }



  //
  // check a decomposition of the form U S U^T
  //
  template<typename eT>
  inline
  void
  syt21(const blas_int itype, const char uplo, const blas_int n, const blas_int kband, const eT* A, const blas_int lda, const eT* D, const eT* E, const eT* U, const blas_int ldu, const eT* V, const blas_int ldv, const eT* tau, eT* work, eT* result)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_ssyt21)(&itype, &uplo, &n, &kband,  (const float*) A, &lda,  (const float*) D,  (const float*) E,  (const float*) U, &ldu,  (const float*) V, &ldv,  (const float*) tau,  (float*) work,  (float*) result, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dsyt21)(&itype, &uplo, &n, &kband, (const double*) A, &lda, (const double*) D, (const double*) E, (const double*) U, &ldu, (const double*) V, &ldv, (const double*) tau, (double*) work, (double*) result, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_ssyt21)(&itype, &uplo, &n, &kband,  (const float*) A, &lda,  (const float*) D,  (const float*) E,  (const float*) U, &ldu,  (const float*) V, &ldv,  (const float*) tau,  (float*) work,  (float*) result); }
      else if(is_double<eT>::value) { coot_fortran(coot_dsyt21)(&itype, &uplo, &n, &kband, (const double*) A, &lda, (const double*) D, (const double*) E, (const double*) U, &ldu, (const double*) V, &ldv, (const double*) tau, (double*) work, (double*) result); }
      }
    #endif
    }



  //
  // check a decomposition of the form A U = U S
  //
  template<typename eT>
  inline
  void
  syt22(const blas_int itype, const char uplo, const blas_int m, const blas_int n, const blas_int kband, const eT* A, const blas_int lda, const eT* D, const eT* E, const eT* U, const blas_int ldu, const eT* V, const blas_int ldv, const eT* tau, eT* work, eT* result)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_ssyt22)(&itype, &uplo, &m, &n, &kband,  (const float*) A, &lda,  (const float*) D,  (const float*) E,  (const float*) U, &ldu,  (const float*) V, &ldv,  (const float*) tau,  (float*) work,  (float*) result, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dsyt22)(&itype, &uplo, &m, &n, &kband, (const double*) A, &lda, (const double*) D, (const double*) E, (const double*) U, &ldu, (const double*) V, &ldv, (const double*) tau, (double*) work, (double*) result, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_ssyt22)(&itype, &uplo, &m, &n, &kband,  (const float*) A, &lda,  (const float*) D,  (const float*) E,  (const float*) U, &ldu,  (const float*) V, &ldv,  (const float*) tau,  (float*) work,  (float*) result); }
      else if(is_double<eT>::value) { coot_fortran(coot_dsyt22)(&itype, &uplo, &m, &n, &kband, (const double*) A, &lda, (const double*) D, (const double*) E, (const double*) U, &ldu, (const double*) V, &ldv, (const double*) tau, (double*) work, (double*) result); }
      }
    #endif
    }



  //
  // SVD (real matrices)
  //
  template<typename eT>
  inline
  void
  gesvd(const char jobu, const char jobvt, const blas_int m, const blas_int n, eT* a, const blas_int lda, typename get_pod_type<eT>::result* s, eT* u, const blas_int ldu, eT* vt, const blas_int ldvt, eT* work, const blas_int lwork, typename get_pod_type<eT>::result* rwork /* only used for complex variants, pass nullptr if not */, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sgesvd)(&jobu, &jobvt, &m, &n,    (float*) a, &lda,  (float*) s,    (float*) u, &ldu,   (float*) vt, &ldvt,    (float*) work, &lwork, info, 1, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dgesvd)(&jobu, &jobvt, &m, &n,   (double*) a, &lda, (double*) s,   (double*) u, &ldu,  (double*) vt, &ldvt,   (double*) work, &lwork, info, 1, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cgesvd)(&jobu, &jobvt, &m, &n, (blas_cxf*) a, &lda,  (float*) s, (blas_cxf*) u, &ldu,(blas_cxf*) vt, &ldvt, (blas_cxf*) work, &lwork,  (float*) rwork, info, 1, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zgesvd)(&jobu, &jobvt, &m, &n, (blas_cxd*) a, &lda, (double*) s, (blas_cxd*) u, &ldu,(blas_cxd*) vt, &ldvt, (blas_cxd*) work, &lwork, (double*) rwork, info, 1, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sgesvd)(&jobu, &jobvt, &m, &n,    (float*) a, &lda,  (float*) s,    (float*) u, &ldu,   (float*) vt, &ldvt,    (float*) work, &lwork, info); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dgesvd)(&jobu, &jobvt, &m, &n,   (double*) a, &lda, (double*) s,   (double*) u, &ldu,  (double*) vt, &ldvt,   (double*) work, &lwork, info); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cgesvd)(&jobu, &jobvt, &m, &n, (blas_cxf*) a, &lda,  (float*) s, (blas_cxf*) u, &ldu,(blas_cxf*) vt, &ldvt, (blas_cxf*) work, &lwork,  (float*) rwork, info); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zgesvd)(&jobu, &jobvt, &m, &n, (blas_cxd*) a, &lda, (double*) s, (blas_cxd*) u, &ldu,(blas_cxd*) vt, &ldvt, (blas_cxd*) work, &lwork, (double*) rwork, info); }
      }
    #endif
    }



  //
  // symmetric 1-norm
  //
  template<typename eT>
  inline
  typename get_pod_type<eT>::result
  lansy(const char norm, const char uplo, const blas_int N, const eT* A, const blas_int lda, typename get_pod_type<eT>::result* work)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { return coot_fortran(coot_slansy)(&norm, &uplo, &N,    (float*) A, &lda,  (float*) work, 1, 1); }
      else if(   is_double<eT>::value) { return coot_fortran(coot_dlansy)(&norm, &uplo, &N,   (double*) A, &lda, (double*) work, 1, 1); }
      else if( is_cx_float<eT>::value) { return coot_fortran(coot_clansy)(&norm, &uplo, &N, (blas_cxf*) A, &lda,  (float*) work, 1, 1); }
      else if(is_cx_double<eT>::value) { return coot_fortran(coot_zlansy)(&norm, &uplo, &N, (blas_cxd*) A, &lda, (double*) work, 1, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { return coot_fortran(coot_slansy)(&norm, &uplo, &N,    (float*) A, &lda,  (float*) work); }
      else if(   is_double<eT>::value) { return coot_fortran(coot_dlansy)(&norm, &uplo, &N,   (double*) A, &lda, (double*) work); }
      else if( is_cx_float<eT>::value) { return coot_fortran(coot_clansy)(&norm, &uplo, &N, (blas_cxf*) A, &lda,  (float*) work); }
      else if(is_cx_double<eT>::value) { return coot_fortran(coot_zlansy)(&norm, &uplo, &N, (blas_cxd*) A, &lda, (double*) work); }
      }
    #endif
    }



  //
  // solve linear equations using LU decomposition
  //
  template<typename eT>
  inline
  void
  getrs(const char trans, const blas_int n, const blas_int nrhs, const eT* a, const blas_int lda, const blas_int* ipiv, eT* b, const blas_int ldb, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sgetrs)(&trans, &n, &nrhs,    (float*) a, &lda, ipiv,    (float*) b, &ldb, info, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dgetrs)(&trans, &n, &nrhs,   (double*) a, &lda, ipiv,   (double*) b, &ldb, info, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cgetrs)(&trans, &n, &nrhs, (blas_cxf*) a, &lda, ipiv, (blas_cxf*) b, &ldb, info, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zgetrs)(&trans, &n, &nrhs, (blas_cxd*) a, &lda, ipiv, (blas_cxd*) b, &ldb, info, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sgetrs)(&trans, &n, &nrhs,    (float*) a, &lda, ipiv,    (float*) b, &ldb, info); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dgetrs)(&trans, &n, &nrhs,   (double*) a, &lda, ipiv,   (double*) b, &ldb, info); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cgetrs)(&trans, &n, &nrhs, (blas_cxf*) a, &lda, ipiv, (blas_cxf*) b, &ldb, info); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zgetrs)(&trans, &n, &nrhs, (blas_cxd*) a, &lda, ipiv, (blas_cxd*) b, &ldb, info); }
      }
    #endif
    }



  //
  // generate a vector of random numbers
  //
  template<typename eT>
  inline
  void
  larnv(const blas_int idist, blas_int* iseed, const blas_int n, eT* x)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    if     ( is_float<eT>::value) { coot_fortran(coot_slarnv)(&idist, iseed, &n,  (float*) x); }
    else if(is_double<eT>::value) { coot_fortran(coot_dlarnv)(&idist, iseed, &n, (double*) x); }
    }



  //
  // generate Q or P**T determined by gebrd
  //
  template<typename eT>
  inline
  void
  orgbr(const char vect, const blas_int m, const blas_int n, const blas_int k, eT* A, const blas_int lda, const eT* tau, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sorgbr)(&vect, &m, &n, &k,  (float*) A, &lda,  (const float*) tau,  (float*) work, &lwork, info, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dorgbr)(&vect, &m, &n, &k, (double*) A, &lda, (const double*) tau, (double*) work, &lwork, info, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sorgbr)(&vect, &m, &n, &k,  (float*) A, &lda,  (const float*) tau,  (float*) work, &lwork, info); }
      else if(is_double<eT>::value) { coot_fortran(coot_dorgbr)(&vect, &m, &n, &k, (double*) A, &lda, (const double*) tau, (double*) work, &lwork, info); }
      }
    #endif
    }



  //
  // generate Q with orthonormal rows
  //
  template<typename eT>
  inline
  void
  orglq(const blas_int m, const blas_int n, const blas_int k, eT* A, const blas_int lda, const eT* tau, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    if     ( is_float<eT>::value) { coot_fortran(coot_sorglq)(&m, &n, &k,  (float*) A, &lda,  (const float*) tau,  (float*) work, &lwork, info); }
    else if(is_double<eT>::value) { coot_fortran(coot_dorglq)(&m, &n, &k, (double*) A, &lda, (const double*) tau, (double*) work, &lwork, info); }
    }



  //
  // overwrite matrix with geqrf-generated orthogonal transformation
  //
  template<typename eT>
  inline
  void
  ormqr(const char side, const char trans, const blas_int m, const blas_int n, const blas_int k, const eT* A, const blas_int lda, const eT* tau, eT* C, const blas_int ldc, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormqr)(&side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info, 1, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormqr)(&side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info, 1, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormqr)(&side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormqr)(&side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info); }
      }
    #endif
    }



  //
  // overwrite matrix with gebrd-generated orthogonal matrix products
  //
  template<typename eT>
  inline
  void
  ormbr(const char vect, const char side, const char trans, const blas_int m, const blas_int n, const blas_int k, const eT* A, const blas_int lda, const eT* tau, eT* C, const blas_int ldc, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormbr)(&vect, &side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info, 1, 1, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormbr)(&vect, &side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info, 1, 1, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormbr)(&vect, &side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormbr)(&vect, &side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info); }
      }
    #endif
    }



  //
  // compute LQ factorization
  //
  template<typename eT>
  inline
  void
  gelqf(const blas_int M, const blas_int N, eT* A, const blas_int lda, eT* tau, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    if     (    is_float<eT>::value) { coot_fortran(coot_sgelqf)(&M, &N,    (float*) A, &lda,    (float*) tau,    (float*) work, &lwork, info); }
    else if(   is_double<eT>::value) { coot_fortran(coot_dgelqf)(&M, &N,   (double*) A, &lda,   (double*) tau,   (double*) work, &lwork, info); }
    else if( is_cx_float<eT>::value) { coot_fortran(coot_cgelqf)(&M, &N, (blas_cxf*) A, &lda, (blas_cxf*) tau, (blas_cxf*) work, &lwork, info); }
    else if(is_cx_double<eT>::value) { coot_fortran(coot_zgelqf)(&M, &N, (blas_cxd*) A, &lda, (blas_cxd*) tau, (blas_cxd*) work, &lwork, info); }
    }



  //
  // generate real orthogonal matrix as the product of dsytrd-generated elementary reflectors
  //
  template<typename eT>
  inline
  void
  orgtr(const char uplo, const blas_int n, eT* A, const blas_int lda, const eT* tau, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sorgtr)(&uplo, &n,  (float*) A, &lda,  (const float*) tau,  (float*) work, &lwork, info, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dorgtr)(&uplo, &n, (double*) A, &lda, (const double*) tau, (double*) work, &lwork, info, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sorgtr)(&uplo, &n,  (float*) A, &lda,  (const float*) tau,  (float*) work, &lwork, info); }
      else if(is_double<eT>::value) { coot_fortran(coot_dorgtr)(&uplo, &n, (double*) A, &lda, (const double*) tau, (double*) work, &lwork, info); }
      }
    #endif
    }

  } // namespace lapack_test

  } // namespace coot
