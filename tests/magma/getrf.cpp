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

// This file contains source code adapted from
// clMAGMA 1.3 (2014-11-14) and/or MAGMA 2.7 (2022-11-09).
// clMAGMA 1.3 and MAGMA 2.7 are distributed under a
// 3-clause BSD license as follows:
//
//  -- Innovative Computing Laboratory
//  -- Electrical Engineering and Computer Science Department
//  -- University of Tennessee
//  -- (C) Copyright 2009-2015
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of the University of Tennessee, Knoxville nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors
//  ``as is'' and any express or implied warranties, including, but not
//  limited to, the implied warranties of merchantability and fitness for
//  a particular purpose are disclaimed. In no event shall the copyright
//  holders or contributors be liable for any direct, indirect, incidental,
//  special, exemplary, or consequential damages (including, but not
//  limited to, procurement of substitute goods or services; loss of use,
//  data, or profits; or business interruption) however caused and on any
//  theory of liability, whether in contract, strict liability, or tort
//  (including negligence or otherwise) arising in any way out of the use
//  of this software, even if advised of the possibility of such damage.

#include <bandicoot>
#include "../catch.hpp"
#include "def_lapack_test.hpp"

using namespace coot;

// These tests are from clMAGMA, only useful for the OpenCL backend.

#if defined(COOT_USE_OPENCL)

// Utility function to compute residual of LU factorisation.
double get_residual_d
  (
  magma_int_t m, magma_int_t n,
  double *A, double *A_orig, magma_int_t lda,
  magma_int_t *ipiv
  )
  {
  REQUIRE( m == n ); // residual check defined only for square matrices

  const double c_one     = MAGMA_D_ONE;
  const double c_neg_one = MAGMA_D_NEG_ONE;
  const blas_int ione = 1;

  // this seed should be DIFFERENT than used in init_matrix
  // (else x is column of A, so residual can be exactly zero)
  magma_int_t info = 0;
  double *x, *b;

  // initialize RHS
  REQUIRE( magma_dmalloc_cpu( &x, n ) == MAGMA_SUCCESS );
  REQUIRE( magma_dmalloc_cpu( &b, n ) == MAGMA_SUCCESS );
  arma::Col<double> x_alias(x, n, false, true);
  x_alias.randu();
  arma::Col<double> b_alias(b, n, false, true);
  b_alias = x_alias;

  // solve Ax = b
  coot_fortran(coot_dgetrs)( "N", &n, (blas_int*) &ione, A, &lda, ipiv, x, &n, &info );
  if (info != 0)
    {
    std::cerr << "LAPACK dgetrs() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
    }

  // compute r = Ax - b, saved in b
  coot_fortran(coot_dgemv)( "N", &m, &n, &c_one, A_orig, &lda, x, &ione, &c_neg_one, b, &ione );

  // compute residual |Ax - b| / (n*|A|*|x|)
  double norm_x, norm_A, norm_r, work[1];
  norm_A = coot_fortran(coot_dlange)( "F", &m, &n, A_orig, &lda, work );
  norm_r = coot_fortran(coot_dlange)( "F", &n, (blas_int*) &ione, b, &n, work );
  norm_x = coot_fortran(coot_dlange)( "F", &n, (blas_int*) &ione, x, &n, work );

  magma_free_cpu( x );
  magma_free_cpu( b );

  return norm_r / (n * norm_A * norm_x);
  }



float get_residual_s
  (
  magma_int_t m, magma_int_t n,
  float *A, float *A_orig, magma_int_t lda,
  magma_int_t *ipiv
  )
  {
  REQUIRE( m == n ); // residual check defined only for square matrices

  const float c_one     = MAGMA_S_ONE;
  const float c_neg_one = MAGMA_S_NEG_ONE;
  const blas_int ione = 1;

  // this seed should be DIFFERENT than used in init_matrix
  // (else x is column of A, so residual can be exactly zero)
  magma_int_t info = 0;
  float *x, *b;

  // initialize RHS
  REQUIRE( magma_smalloc_cpu( &x, n ) == MAGMA_SUCCESS );
  REQUIRE( magma_smalloc_cpu( &b, n ) == MAGMA_SUCCESS );
  arma::Col<float> x_alias(x, n, false, true);
  x_alias.randu();
  arma::Col<float> b_alias(b, n, false, true);
  b_alias = x_alias;

  // solve Ax = b
  coot_fortran(coot_sgetrs)( "N", &n, (blas_int*) &ione, A, &lda, ipiv, x, &n, &info );
  if (info != 0)
    {
    std::cerr << "LAPACK sgetrs() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
    }

  // compute r = Ax - b, saved in b
  coot_fortran(coot_sgemv)( "N", &m, &n, &c_one, A_orig, &lda, x, &ione, &c_neg_one, b, &ione );

  // compute residual |Ax - b| / (n*|A|*|x|)
  float norm_x, norm_A, norm_r, work[1];
  norm_A = coot_fortran(coot_slange)( "F", &m, &n, A_orig, &lda, work );
  norm_r = coot_fortran(coot_slange)( "F", &n, (blas_int*) &ione, b, &n, work );
  norm_x = coot_fortran(coot_slange)( "F", &n, (blas_int*) &ione, x, &n, work );

  magma_free_cpu( x );
  magma_free_cpu( b );

  return norm_r / (n * norm_A * norm_x);
  }



// On input, LU and ipiv is LU factorization of A. On output, LU is overwritten.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 3 more matrices to store A, L, and U.
double get_LU_error_d
  (
  magma_int_t M, magma_int_t N,
  double *LU, double *A, magma_int_t lda,
  magma_int_t *ipiv
  )
  {
  magma_int_t min_mn = std::min(M,N);
  blas_int ione   = 1;
  magma_int_t i, j;

  double alpha = MAGMA_D_ONE;
  double beta  = MAGMA_D_ZERO;
  double *L, *U;
  double work[1], matnorm, residual;

  REQUIRE( magma_dmalloc_cpu( &L, M*min_mn ) == MAGMA_SUCCESS );
  REQUIRE( magma_dmalloc_cpu( &U, min_mn*N ) == MAGMA_SUCCESS );
  memset( L, 0, M*min_mn*sizeof(double) );
  memset( U, 0, min_mn*N*sizeof(double) );

  // permute original matrix
  coot_fortran(coot_dlaswp)( &N, A, &lda, &ione, &min_mn, ipiv, &ione);

  // copy LU to L and U, and set diagonal to 1
  coot_fortran(coot_dlacpy)( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
  coot_fortran(coot_dlacpy)( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );
  for (j=0; j < min_mn; j++)
    {
    L[j+j*M] = (double) 1.0;
    }

  matnorm = coot_fortran(coot_dlange)("f", &M, &N, A, &lda, work);

  coot_fortran(coot_dgemm)("N", "N", &M, &N, &min_mn,
                           &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

  for( j = 0; j < N; j++ )
    {
    for( i = 0; i < M; i++ )
      {
      LU[i+j*lda] = (LU[i+j*lda] - A[i+j*lda]);
      }
    }
  residual = coot_fortran(coot_dlange)("f", &M, &N, LU, &lda, work);

  magma_free_cpu(L);
  magma_free_cpu(U);

  return residual / (matnorm * N);
  }



float get_LU_error_s
  (
  magma_int_t M, magma_int_t N,
  float *LU, float *A, magma_int_t lda,
  magma_int_t *ipiv
  )
  {
  magma_int_t min_mn = std::min(M,N);
  blas_int ione   = 1;
  magma_int_t i, j;

  float alpha = MAGMA_S_ONE;
  float beta  = MAGMA_S_ZERO;
  float *L, *U;
  float work[1], matnorm, residual;

  REQUIRE( magma_smalloc_cpu( &L, M*min_mn ) == MAGMA_SUCCESS );
  REQUIRE( magma_smalloc_cpu( &U, min_mn*N ) == MAGMA_SUCCESS );
  memset( L, 0, M*min_mn*sizeof(float) );
  memset( U, 0, min_mn*N*sizeof(float) );

  // permute original matrix
  coot_fortran(coot_slaswp)( &N, A, &lda, &ione, &min_mn, ipiv, &ione);

  // copy LU to L and U, and set diagonal to 1
  coot_fortran(coot_slacpy)( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
  coot_fortran(coot_slacpy)( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );
  for (j=0; j < min_mn; j++)
    {
    L[j+j*M] = (float) 1.0;
    }

  matnorm = coot_fortran(coot_slange)("f", &M, &N, A, &lda, work);

  coot_fortran(coot_sgemm)("N", "N", &M, &N, &min_mn,
                           &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

  for( j = 0; j < N; j++ )
    {
    for( i = 0; i < M; i++ )
      {
      LU[i+j*lda] = (LU[i+j*lda] - A[i+j*lda]);
      }
    }
  residual = coot_fortran(coot_slange)("f", &M, &N, LU, &lda, work);

  magma_free_cpu(L);
  magma_free_cpu(U);

  return residual / (matnorm * N);
  }



TEST_CASE("magma_dgetrf", "[getrf]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double error;
  double* h_A;
  double* h_A_orig;
  magmaDouble_ptr d_A;
  magma_int_t     *ipiv;
  magma_int_t M, N, n2, lda, ldda, info, min_mn;

  double tol = 30 * std::numeric_limits<double>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    M = 128 * (itest + 1) + 64;
    N = 128 * (itest + 1) + 64;
    min_mn = std::min(M, N);
    lda    = M;
    n2     = lda*N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default

    REQUIRE( magma_imalloc_cpu( &ipiv,     min_mn   ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_A,      n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_A_orig, n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc(     &d_A,      ldda * N ) == MAGMA_SUCCESS );

    // The default test uses a random matrix, so we'll do the same here via Armadillo.
    arma::Mat<double> h_A_alias(h_A, lda, N, false, true);
    h_A_alias.randu();
    // Save an original copy.
    coot_fortran(coot_dlacpy)( "A", &M, &N, h_A, &lda, h_A_orig, &lda );

    magma_dsetmatrix( M, N, h_A, lda, d_A, 0, ldda, queue );

    magma_dgetrf_gpu( M, N, d_A, 0, ldda, ipiv, &info);
    if (info != 0)
      {
      std::cerr << "magma_dgetrf_gpu() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    magma_dgetmatrix( M, N, d_A, 0, ldda, h_A, lda, queue );
    error = get_residual_d( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    error = get_LU_error_d( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    magma_free_cpu( ipiv );
    magma_free_cpu( h_A );
    magma_free( d_A );

    }

  magma_queue_destroy( queue );
  }



TEST_CASE("magma_dgetrf_nopiv", "[getrf]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double error;
  double* h_A;
  double* h_A_orig;
  magmaDouble_ptr d_A;
  magma_int_t     *ipiv;
  magma_int_t M, N, n2, lda, ldda, info, min_mn;

  // seems to need a much wider tolerance
  double tol = 30000 * std::numeric_limits<double>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    M = 128 * (itest + 1) + 64;
    N = 128 * (itest + 1) + 64;
    min_mn = std::min(M, N);
    lda    = M;
    n2     = lda*N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default

    REQUIRE( magma_imalloc_cpu( &ipiv,     min_mn   ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_A,      n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_A_orig, n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc(     &d_A,      ldda * N ) == MAGMA_SUCCESS );

    // The default test uses a random matrix, so we'll do the same here via Armadillo.
    arma::Mat<double> h_A_alias(h_A, lda, N, false, true);
    h_A_alias.randu();
    // Save an original copy.
    coot_fortran(coot_dlacpy)( "A", &M, &N, h_A, &lda, h_A_orig, &lda );

    magma_dsetmatrix( M, N, h_A, lda, d_A, 0, ldda, queue );

    magma_dgetrf_nopiv_gpu( M, N, d_A, 0, ldda, &info);
    if (info != 0)
      {
      std::cerr << "magma_dgetrf_gpu() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    // Set pivots to identity so we can use the same check functions.
    for (magma_int_t i = 0; i < min_mn; ++i)
      {
      ipiv[i] = i + 1;
      }

    magma_dgetmatrix( M, N, d_A, 0, ldda, h_A, lda, queue );
    error = get_residual_d( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    error = get_LU_error_d( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    magma_free_cpu( ipiv );
    magma_free_cpu( h_A );
    magma_free( d_A );

    }

  magma_queue_destroy( queue );
  }



TEST_CASE("magma_sgetrf", "[getrf]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  float error;
  float* h_A;
  float* h_A_orig;
  magmaFloat_ptr d_A;
  magma_int_t     *ipiv;
  magma_int_t M, N, n2, lda, ldda, info, min_mn;

  float tol = 30 * std::numeric_limits<float>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    M = 128 * (itest + 1) + 64;
    N = 128 * (itest + 1) + 64;
    min_mn = std::min(M, N);
    lda    = M;
    n2     = lda*N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default

    REQUIRE( magma_imalloc_cpu( &ipiv,     min_mn   ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &h_A,      n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &h_A_orig, n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc(     &d_A,      ldda * N ) == MAGMA_SUCCESS );

    // The default test uses a random matrix, so we'll do the same here via Armadillo.
    arma::Mat<float> h_A_alias(h_A, lda, N, false, true);
    h_A_alias.randu();
    // Save an original copy.
    coot_fortran(coot_slacpy)( "A", &M, &N, h_A, &lda, h_A_orig, &lda );

    magma_ssetmatrix( M, N, h_A, lda, d_A, 0, ldda, queue );

    magma_sgetrf_gpu( M, N, d_A, 0, ldda, ipiv, &info);
    if (info != 0)
      {
      std::cerr << "magma_sgetrf_gpu() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    magma_sgetmatrix( M, N, d_A, 0, ldda, h_A, lda, queue );
    error = get_residual_s( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    error = get_LU_error_s( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    magma_free_cpu( ipiv );
    magma_free_cpu( h_A );
    magma_free( d_A );

    }

  magma_queue_destroy( queue );
  }



TEST_CASE("magma_sgetrf_nopiv", "[getrf]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  float error;
  float* h_A;
  float* h_A_orig;
  magmaFloat_ptr d_A;
  magma_int_t     *ipiv;
  magma_int_t M, N, n2, lda, ldda, info, min_mn;

  // seems to need a much wider tolerance
  float tol = 30000 * std::numeric_limits<float>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    M = 128 * (itest + 1) + 64;
    N = 128 * (itest + 1) + 64;
    min_mn = std::min(M, N);
    lda    = M;
    n2     = lda*N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default

    REQUIRE( magma_imalloc_cpu( &ipiv,     min_mn   ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &h_A,      n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &h_A_orig, n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc(     &d_A,      ldda * N ) == MAGMA_SUCCESS );

    // The default test uses a random matrix, so we'll do the same here via Armadillo.
    arma::Mat<float> h_A_alias(h_A, lda, N, false, true);
    h_A_alias.randu();
    // Save an original copy.
    coot_fortran(coot_slacpy)( "A", &M, &N, h_A, &lda, h_A_orig, &lda );

    magma_ssetmatrix( M, N, h_A, lda, d_A, 0, ldda, queue );

    magma_sgetrf_nopiv_gpu( M, N, d_A, 0, ldda, &info);
    if (info != 0)
      {
      std::cerr << "magma_sgetrf_gpu() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    // Set pivots to identity so we can use the same check functions.
    for (magma_int_t i = 0; i < min_mn; ++i)
      {
      ipiv[i] = i + 1;
      }

    magma_sgetmatrix( M, N, d_A, 0, ldda, h_A, lda, queue );
    error = get_residual_s( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    error = get_LU_error_s( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    magma_free_cpu( ipiv );
    magma_free_cpu( h_A );
    magma_free( d_A );

    }

  magma_queue_destroy( queue );
  }



TEST_CASE("magma_dgetrf_small", "[getrf]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double error;
  double* h_A;
  double* h_A_orig;
  magmaDouble_ptr d_A;
  magma_int_t     *ipiv;
  magma_int_t M, N, n2, lda, ldda, info, min_mn;

  double tol = 30 * std::numeric_limits<double>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    M = 4 * (itest + 1) + 4;
    N = 4 * (itest + 1) + 4;
    min_mn = std::min(M, N);
    lda    = M;
    n2     = lda*N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default

    REQUIRE( magma_imalloc_cpu( &ipiv,     min_mn   ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_A,      n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_A_orig, n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc(     &d_A,      ldda * N ) == MAGMA_SUCCESS );

    // The default test uses a random matrix, so we'll do the same here via Armadillo.
    arma::Mat<double> h_A_alias(h_A, lda, N, false, true);
    h_A_alias.randu();
    // Save an original copy.
    coot_fortran(coot_dlacpy)( "A", &M, &N, h_A, &lda, h_A_orig, &lda );

    magma_dsetmatrix( M, N, h_A, lda, d_A, 0, ldda, queue );

    magma_dgetrf_gpu( M, N, d_A, 0, ldda, ipiv, &info);
    if (info != 0)
      {
      std::cerr << "magma_dgetrf_gpu() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    magma_dgetmatrix( M, N, d_A, 0, ldda, h_A, lda, queue );
    error = get_residual_d( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    error = get_LU_error_d( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    magma_free_cpu( ipiv );
    magma_free_cpu( h_A );
    magma_free( d_A );

    }

  magma_queue_destroy( queue );
  }



TEST_CASE("magma_dgetrf_nopiv_small", "[getrf]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double error;
  double* h_A;
  double* h_A_orig;
  magmaDouble_ptr d_A;
  magma_int_t     *ipiv;
  magma_int_t M, N, n2, lda, ldda, info, min_mn;

  // seems to need a much wider tolerance
  double tol = 30000 * std::numeric_limits<double>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    M = 4 * (itest + 1) + 4;
    N = 4 * (itest + 1) + 4;
    min_mn = std::min(M, N);
    lda    = M;
    n2     = lda*N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default

    REQUIRE( magma_imalloc_cpu( &ipiv,     min_mn   ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_A,      n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_A_orig, n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc(     &d_A,      ldda * N ) == MAGMA_SUCCESS );

    // The default test uses a random matrix, so we'll do the same here via Armadillo.
    arma::Mat<double> h_A_alias(h_A, lda, N, false, true);
    h_A_alias.randu();
    // Save an original copy.
    coot_fortran(coot_dlacpy)( "A", &M, &N, h_A, &lda, h_A_orig, &lda );

    magma_dsetmatrix( M, N, h_A, lda, d_A, 0, ldda, queue );

    magma_dgetrf_nopiv_gpu( M, N, d_A, 0, ldda, &info);
    if (info != 0)
      {
      std::cerr << "magma_dgetrf_gpu() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    // Set pivots to identity so we can use the same check functions.
    for (magma_int_t i = 0; i < min_mn; ++i)
      {
      ipiv[i] = i + 1;
      }

    magma_dgetmatrix( M, N, d_A, 0, ldda, h_A, lda, queue );
    error = get_residual_d( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    error = get_LU_error_d( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    magma_free_cpu( ipiv );
    magma_free_cpu( h_A );
    magma_free( d_A );

    }

  magma_queue_destroy( queue );
  }



TEST_CASE("magma_sgetrf_small", "[getrf]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  float error;
  float* h_A;
  float* h_A_orig;
  magmaFloat_ptr d_A;
  magma_int_t     *ipiv;
  magma_int_t M, N, n2, lda, ldda, info, min_mn;

  float tol = 30 * std::numeric_limits<float>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    M = 4 * (itest + 1) + 4;
    N = 4 * (itest + 1) + 4;
    min_mn = std::min(M, N);
    lda    = M;
    n2     = lda*N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default

    REQUIRE( magma_imalloc_cpu( &ipiv,     min_mn   ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &h_A,      n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &h_A_orig, n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc(     &d_A,      ldda * N ) == MAGMA_SUCCESS );

    // The default test uses a random matrix, so we'll do the same here via Armadillo.
    arma::Mat<float> h_A_alias(h_A, lda, N, false, true);
    h_A_alias.randu();
    // Save an original copy.
    coot_fortran(coot_slacpy)( "A", &M, &N, h_A, &lda, h_A_orig, &lda );

    magma_ssetmatrix( M, N, h_A, lda, d_A, 0, ldda, queue );

    magma_sgetrf_gpu( M, N, d_A, 0, ldda, ipiv, &info);
    if (info != 0)
      {
      std::cerr << "magma_sgetrf_gpu() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    magma_sgetmatrix( M, N, d_A, 0, ldda, h_A, lda, queue );
    error = get_residual_s( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    error = get_LU_error_s( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    magma_free_cpu( ipiv );
    magma_free_cpu( h_A );
    magma_free( d_A );

    }

  magma_queue_destroy( queue );
  }



TEST_CASE("magma_sgetrf_nopiv_small", "[getrf]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  float error;
  float* h_A;
  float* h_A_orig;
  magmaFloat_ptr d_A;
  magma_int_t     *ipiv;
  magma_int_t M, N, n2, lda, ldda, info, min_mn;

  // seems to need a much wider tolerance
  float tol = 30000 * std::numeric_limits<float>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    M = 4 * (itest + 1) + 4;
    N = 4 * (itest + 1) + 4;
    min_mn = std::min(M, N);
    lda    = M;
    n2     = lda*N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default

    REQUIRE( magma_imalloc_cpu( &ipiv,     min_mn   ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &h_A,      n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &h_A_orig, n2       ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc(     &d_A,      ldda * N ) == MAGMA_SUCCESS );

    // The default test uses a random matrix, so we'll do the same here via Armadillo.
    arma::Mat<float> h_A_alias(h_A, lda, N, false, true);
    h_A_alias.randu();
    // Save an original copy.
    coot_fortran(coot_slacpy)( "A", &M, &N, h_A, &lda, h_A_orig, &lda );

    magma_ssetmatrix( M, N, h_A, lda, d_A, 0, ldda, queue );

    magma_sgetrf_nopiv_gpu( M, N, d_A, 0, ldda, &info);
    if (info != 0)
      {
      std::cerr << "magma_sgetrf_gpu() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    // Set pivots to identity so we can use the same check functions.
    for (magma_int_t i = 0; i < min_mn; ++i)
      {
      ipiv[i] = i + 1;
      }

    magma_sgetmatrix( M, N, d_A, 0, ldda, h_A, lda, queue );
    error = get_residual_s( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    error = get_LU_error_s( M, N, h_A, h_A_orig, lda, ipiv );
    REQUIRE( error < tol );

    magma_free_cpu( ipiv );
    magma_free_cpu( h_A );
    magma_free( d_A );

    }

  magma_queue_destroy( queue );
  }




#endif
