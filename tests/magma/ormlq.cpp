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

TEST_CASE("magma_dormlq_1", "[ormlq]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double Cnorm, error, work[1];
  double c_neg_one = MAGMA_D_NEG_ONE;
  magma_int_t ione = 1;
  magma_int_t nn, m, n, k, size, info;
  magma_int_t ISEED[4] = {0,0,0,1};
  magma_int_t nb, ldc, lda, lwork, lwork_max, ldda;
  double *C, *R, *A, *W, *tau;
  magmaDouble_ptr dA;

  double tol = 60 * std::numeric_limits<double>::epsilon();

  // test all combinations of input parameters
  magma_side_t  side [] = { MagmaLeft,       MagmaRight   };
  magma_trans_t trans[] = { MagmaTrans, MagmaNoTrans };

  magma_queue_t queue = magma_queue_create();

  for( int itest = 0; itest < 10; ++itest )
    {
    for( int iside = 0; iside < 2; ++iside )
      {
      for( int itran = 0; itran < 2; ++itran )
        {
        m = 128 * (itest + 1) + 64;
        n = 128 * (itest + 1) + 64;
        k = 128 * (itest + 1) + 64;
        nb  = magma_get_dgelqf_nb( m, n );
        ldc = m;
        // A is k x nn == k x m (left) or k x n (right)
        nn = (side[iside] == MagmaLeft ? m : n);
        lda = k;
        ldda = magma_roundup( k, 32 );

        // need at least 2*nb*nb for gelqf
        lwork_max = std::max( std::max( m*nb, n*nb ), 2*nb*nb );
        // this rounds it up slightly if needed to agree with lwork query
        lwork_max = magma_int_t( double( magma_dmake_lwork( lwork_max )));

        REQUIRE( magma_dmalloc_cpu( &C,   ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &R,   ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &A,   lda*nn ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &W,   lwork_max ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &tau, k ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc( &dA, ldda * n) == MAGMA_SUCCESS );

        // C is full, m x n
        size = ldc*n;
        coot_fortran(coot_dlarnv)( &ione, ISEED, &size, C );
        coot_fortran(coot_dlacpy)( "F", &m, &n, C, &ldc, R, &ldc );

        // By default the matrix used is random uniform; we'll use Armadillo
        // to do that.
        arma::Mat<double> A_alias(A, lda, nn, false, true);
        A_alias.randu();

        // compute LQ factorization to get Householder vectors in A, tau
        magma_dsetmatrix( k, nn, A, lda, dA, 0, ldda, queue );
        magma_dgelqf_gpu( k, nn, dA, 0, ldda, tau, W, lwork_max, &info );
        magma_dgetmatrix( k, nn, dA, 0, ldda, A, lda, queue );
        if (info != 0)
          {
          std::cerr << "magma_dgelqf() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        coot_fortran(coot_dormlq)( lapack_side_const( side[iside] ), lapack_trans_const( trans[itran] ),
                          &m, &n, &k,
                          A, &lda, tau, C, &ldc, W, &lwork_max, &info );
        if (info != 0)
          {
          std::cerr << "dormlq returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        // query for workspace size
        lwork = -1;
        magma_dormlq( side[iside], trans[itran],
                      m, n, k,
                      A, lda, tau, R, ldc, W, lwork, &info );
        if (info != 0)
          {
          std::cerr << "magma_dormlq (lwork query) returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );
        lwork = (magma_int_t) W[0];
        if ( lwork < 0 || lwork > lwork_max )
          {
          std::cerr << "warning: optimal lwork " << lwork << " > allocated lwork_max " << lwork_max << std::endl;
          }

        magma_dormlq( side[iside], trans[itran],
                      m, n, k,
                      A, lda, tau, R, ldc, W, lwork, &info );
        if (info != 0)
          {
          std::cerr << "magma_dormlq returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        /* =====================================================================
           compute relative error |QC_magma - QC_lapack| / |QC_lapack|
           =================================================================== */
        size = ldc*n;
        coot_fortran(coot_daxpy)( &size, &c_neg_one, C, &ione, R, &ione );
        Cnorm = coot_fortran(coot_dlange)( "F", &m, &n, C, &ldc, work );
        error = coot_fortran(coot_dlange)( "F", &m, &n, R, &ldc, work ) / (std::sqrt(m*n) * Cnorm);

        REQUIRE( error < tol );

        magma_free_cpu( C );
        magma_free_cpu( R );
        magma_free_cpu( A );
        magma_free_cpu( W );
        magma_free_cpu( tau );
        magma_free( dA );
        }
      }
    }
  }



TEST_CASE("magma_sormlq_1", "[ormlq]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  float Cnorm, error, work[1];
  float c_neg_one = MAGMA_S_NEG_ONE;
  magma_int_t ione = 1;
  magma_int_t nn, m, n, k, size, info;
  magma_int_t ISEED[4] = {0,0,0,1};
  magma_int_t nb, ldc, lda, lwork, lwork_max, ldda;
  float *C, *R, *A, *W, *tau;
  magmaFloat_ptr dA;

  float tol = 60 * std::numeric_limits<float>::epsilon();

  // test all combinations of input parameters
  magma_side_t  side [] = { MagmaLeft,       MagmaRight   };
  magma_trans_t trans[] = { MagmaTrans, MagmaNoTrans };

  magma_queue_t queue = magma_queue_create();

  for( int itest = 0; itest < 10; ++itest )
    {
    for( int iside = 0; iside < 2; ++iside )
      {
      for( int itran = 0; itran < 2; ++itran )
        {
        m = 128 * (itest + 1) + 64;
        n = 128 * (itest + 1) + 64;
        k = 128 * (itest + 1) + 64;
        nb  = magma_get_sgelqf_nb( m, n );
        ldc = m;
        // A is k x nn == k x m (left) or k x n (right)
        nn = (side[iside] == MagmaLeft ? m : n);
        lda = k;
        ldda = magma_roundup( k, 32 );

        // need at least 2*nb*nb for gelqf
        lwork_max = std::max( std::max( m*nb, n*nb ), 2*nb*nb );
        // this rounds it up slightly if needed to agree with lwork query
        lwork_max = magma_int_t( float( magma_smake_lwork( lwork_max )));

        REQUIRE( magma_smalloc_cpu( &C,   ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &R,   ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &A,   lda*nn ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &W,   lwork_max ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &tau, k ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc( &dA, ldda * n) == MAGMA_SUCCESS );

        // C is full, m x n
        size = ldc*n;
        coot_fortran(coot_slarnv)( &ione, ISEED, &size, C );
        coot_fortran(coot_slacpy)( "F", &m, &n, C, &ldc, R, &ldc );

        // By default the matrix used is random uniform; we'll use Armadillo
        // to do that.
        arma::Mat<float> A_alias(A, lda, nn, false, true);
        A_alias.randu();

        // compute LQ factorization to get Householder vectors in A, tau
        magma_ssetmatrix( k, nn, A, lda, dA, 0, ldda, queue );
        magma_sgelqf_gpu( k, nn, dA, 0, ldda, tau, W, lwork_max, &info );
        magma_sgetmatrix( k, nn, dA, 0, ldda, A, lda, queue );
        if (info != 0)
          {
          std::cerr << "magma_sgelqf() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        coot_fortran(coot_sormlq)( lapack_side_const( side[iside] ), lapack_trans_const( trans[itran] ),
                          &m, &n, &k,
                          A, &lda, tau, C, &ldc, W, &lwork_max, &info );
        if (info != 0)
          {
          std::cerr << "sormlq returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        // query for workspace size
        lwork = -1;
        magma_sormlq( side[iside], trans[itran],
                      m, n, k,
                      A, lda, tau, R, ldc, W, lwork, &info );
        if (info != 0)
          {
          std::cerr << "magma_sormlq (lwork query) returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );
        lwork = (magma_int_t) W[0];
        if ( lwork < 0 || lwork > lwork_max )
          {
          std::cerr << "warning: optimal lwork " << lwork << " > allocated lwork_max " << lwork_max << std::endl;
          }

        magma_sormlq( side[iside], trans[itran],
                      m, n, k,
                      A, lda, tau, R, ldc, W, lwork, &info );
        if (info != 0)
          {
          std::cerr << "magma_sormlq returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        /* =====================================================================
           compute relative error |QC_magma - QC_lapack| / |QC_lapack|
           =================================================================== */
        size = ldc*n;
        coot_fortran(coot_saxpy)( &size, &c_neg_one, C, &ione, R, &ione );
        Cnorm = coot_fortran(coot_slange)( "F", &m, &n, C, &ldc, work );
        error = coot_fortran(coot_slange)( "F", &m, &n, R, &ldc, work ) / (std::sqrt(m*n) * Cnorm);

        REQUIRE( error < tol );

        magma_free_cpu( C );
        magma_free_cpu( R );
        magma_free_cpu( A );
        magma_free_cpu( W );
        magma_free_cpu( tau );
        magma_free( dA );
        }
      }
    }
  }

#endif
