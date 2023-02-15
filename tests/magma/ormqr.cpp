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

TEST_CASE("magma_dormqr_1", "[ormqr]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double Cnorm, error, work[1];
  double c_neg_one = MAGMA_D_NEG_ONE;
  magma_int_t ione = 1;
  magma_int_t mm, m, n, k, size, info;
  magma_int_t ISEED[4] = {0,0,0,1};
  magma_int_t nb, ldc, lda, lwork, lwork_max;
  double *C, *R, *A, *W, *tau;

  // TODO: can we reuse this?
  magma_queue_t queue = NULL;
  magma_device_t cdev;
  magma_getdevice( &cdev );
  magma_queue_create( cdev, &queue );

  // need slightly looser bound (60*eps instead of 30*eps) for some tests
  double tol = 60 * std::numeric_limits<double>::epsilon();

  // test all combinations of input parameters
  magma_side_t  side [] = { MagmaLeft,       MagmaRight   };
  magma_trans_t trans[] = { MagmaTrans, MagmaNoTrans };

  for ( int itest = 0; itest < 5; ++itest )
    {
    for ( int iside = 0; iside < 2; ++iside )
      {
      for ( int itran = 0; itran < 2; ++itran )
        {
        m = 128 * (itest + 1) + 64;
        n = 128 * (itest + 1) + 64;
        k = 128 * (itest + 1) + 64;
        nb  = magma_get_dgeqrf_nb( m, n );
        ldc = m;
        // A is mm x k == m x k (left) or n x k (right)
        mm = (side[iside] == MagmaLeft ? m : n);
        lda = mm;

        // need at least 2*nb*nb for geqrf
        lwork_max = std::max( std::max( m*nb, n*nb ), 2*nb*nb );
        // this rounds it up slightly if needed to agree with lwork query below
        lwork_max = magma_int_t( double( magma_dmake_lwork( lwork_max )));

        REQUIRE( magma_dmalloc_cpu( &C,   ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &R,   ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &A,   lda*k ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &W,   lwork_max ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &tau, k ) == MAGMA_SUCCESS );

        // C is full, m x n
        size = ldc*n;
        coot_fortran(coot_dlarnv)( &ione, ISEED, &size, C );
        coot_fortran(coot_dlacpy)( "Full", &m, &n, C, &ldc, R, &ldc );

        // A is mm x k.  Here we use uniform random values.
        arma::Mat<double> A_alias(A, lda, k, false, true);
        A_alias.randu();

        // compute QR factorization to get Householder vectors in A, tau
        // (adapted to use GPU version from original test)
        magmaDouble_ptr dA, dT;
        magma_int_t dt_size = ( 2 * std::min(n, k) + magma_roundup( std::max(m, n), 32) ) * nb;
        magma_int_t ldda = magma_roundup( m, 32 );
        REQUIRE( magma_dmalloc( &dA, ldda * k ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc( &dT, dt_size ) == MAGMA_SUCCESS );
        magma_dsetmatrix( mm, k, A, lda, dA, 0, ldda, queue );
        magma_dgeqrf_gpu( mm, k, dA, 0, lda, tau, dT, 0, &info );
        magma_dgetmatrix( mm, k, dA, 0, ldda, A, lda, queue );
        magma_free( dA );
        magma_free( dT );

        if (info != 0)
          {
          std::cerr << "magma_dgeqrf() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        coot_fortran(coot_dormqr)( lapack_side_const( side[iside] ), lapack_trans_const( trans[itran] ),
                          &m, &n, &k,
                          A, &lda, tau, C, &ldc, W, &lwork_max, &info );
        if (info != 0)
          {
          std::cerr << "dormqr() returned error code " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        // query for workspace size
        lwork = -1;
        magma_dormqr( side[iside], trans[itran],
                      m, n, k,
                      A, lda, tau, R, ldc, W, lwork, &info );
        if (info != 0)
          {
          std::cerr << "magma_dormqr (lwork query) returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );
        lwork = (magma_int_t) W[0];
        if ( lwork < 0 || lwork > lwork_max )
          {
          std::cerr << "magma_dormqr: optimal lwork " << lwork << " > allocated lwork_max " << lwork_max << std::endl;
          lwork = lwork_max;
          }

        magma_dormqr( side[iside], trans[itran],
                      m, n, k,
                      A, lda, tau, R, ldc, W, lwork, &info );
        if (info != 0)
          {
          std::cerr << "magma_dormqr returned error " << info << ": " << magma::error_as_string(info) << std::endl;
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
        }
      }
    }
  }

#endif
