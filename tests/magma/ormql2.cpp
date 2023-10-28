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

#include <armadillo>
#include <bandicoot>
#include "../catch.hpp"
#include "def_lapack_test.hpp"
#include "translate_lapack_test.hpp"
#include "sgeqlf.hpp"
#include "dgeqlf.hpp"

using namespace coot;

// These tests are from clMAGMA, only useful for the OpenCL backend.

#if defined(COOT_USE_OPENCL)

TEST_CASE("magma_dormql2_1", "[ormql2]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  if (!coot_rt_t::is_supported_type<double>())
    {
    return;
    }

  double Cnorm, error, work[1];
  double c_neg_one = MAGMA_D_NEG_ONE;
  magma_int_t ione = 1;
  magma_int_t mm, m, n, k, size, info;
  magma_int_t nb, ldc, lda, /*lwork,*/ lwork_max;
  double *C, *R, *A, *hwork, *tau;
  magmaDouble_ptr dC, dA;

  // test all combinations of input parameters
  magma_side_t  side [] = { MagmaLeft,       MagmaRight   };
  magma_trans_t trans[] = { MagmaTrans, MagmaNoTrans };

  // need slightly looser bound (60 instead of 30)
  double tol = 60 * std::numeric_limits<double>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for( int itest = 0; itest < 6; ++itest )
    {
    for( int iside = 0; iside < 2; ++iside )
      {
      for( int itran = 0; itran < 2; ++itran )
        {
        m = 128 * (itest + 1) + 64;
        n = 128 * (itest + 1) + 64;
        k = 128 * (itest + 1) + 64;
        nb  = magma_get_dgeqrf_nb( m, n );
        ldc = magma_roundup( m, 32 );
        // A is mm x k == m x k (left) or n x k (right)
        mm = (side[iside] == MagmaLeft ? m : n);
        lda = magma_roundup( mm, 32 );

        // need at least 2*nb*nb for geqlf
        lwork_max = std::max( std::max( m*nb, n*nb ), 2*nb*nb );
        // this rounds it up slightly if needed to agree with lwork query below
        lwork_max = magma_int_t( magma_dmake_lwork( lwork_max ));

        REQUIRE( magma_dmalloc_cpu( &C,     ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &R,     ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &A,     lda*k ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &hwork, lwork_max ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &tau,   k ) == MAGMA_SUCCESS );

        REQUIRE( magma_dmalloc( &dC, ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc( &dA, lda*k ) == MAGMA_SUCCESS );

        // C is full, mxn
        size = ldc*n;
        arma::mat C_alias(C, ldc, n, false, true);
        C_alias.randu();
        magma_dsetmatrix( m, n, C, ldc, dC, 0, ldc, queue );

        // A is mm x k
        arma::mat A_alias(A, lda, k, false, true);
        A_alias.randu();

        // compute QL factorization to get Householder vectors in A, tau
        // note that magma_dgeqlf() is specific to test code; we do not have it in bandicoot directly
        magma_dgeqlf( mm, k, A, lda, tau, hwork, lwork_max, &info );
        magma_dsetmatrix( mm, k, A, lda, dA, 0, lda, queue );
        if (info != 0)
          {
          std::cerr << "magma_dgeqlf returned error " << info << ": " << magma::error_as_string( info ) << std::endl;
          }
        REQUIRE( info == 0 );

        magma_dgetmatrix( mm, k, dA, 0, lda, A, lda, queue );

        magma_dormql2_gpu( side[iside], trans[itran],
                           m, n, k,
                           dA, 0, lda, tau, dC, 0, ldc, A, lda, &info );

        magma_dgetmatrix( m, n, dC, 0, ldc, R, ldc, queue );

        lapack_test::ormql(lapack_side_const(side[iside])[0], lapack_trans_const(trans[itran])[0],
                           m, n, k,
                           A, lda, tau, C, ldc, hwork, lwork_max, &info);
        if (info != 0)
          {
          std::cerr << "LAPACK dormql returned error " << info << ": " << magma::error_as_string( info ) << std::endl;
          }
        REQUIRE( info == 0 );

        /* =====================================================================
           compute relative error |QC_magma - QC_lapack| / |QC_lapack|
           =================================================================== */
        size = ldc*n;
        blas::axpy(size, c_neg_one, C, ione, R, ione);
        Cnorm = lapack::lange('F', m, n, C, ldc, work);
        error = lapack::lange('F', m, n, R, ldc, work) / (std::sqrt(m*n) * Cnorm);

        REQUIRE (error < tol);

        magma_free_cpu( C );
        magma_free_cpu( R );
        magma_free_cpu( A );
        magma_free_cpu( hwork );
        magma_free_cpu( tau );

        magma_free( dC );
        magma_free( dA );
        }
      }
    }

  magma_queue_destroy( queue );
  }



TEST_CASE("magma_sormql2_1", "[ormql2]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  float Cnorm, error, work[1];
  float c_neg_one = MAGMA_S_NEG_ONE;
  magma_int_t ione = 1;
  magma_int_t mm, m, n, k, size, info;
  magma_int_t nb, ldc, lda, /*lwork,*/ lwork_max;
  float *C, *R, *A, *hwork, *tau;
  magmaFloat_ptr dC, dA;

  // test all combinations of input parameters
  magma_side_t  side [] = { MagmaLeft,       MagmaRight   };
  magma_trans_t trans[] = { MagmaTrans, MagmaNoTrans };

  // need slightly looser bound (60 instead of 30)
  float tol = 60 * std::numeric_limits<float>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for( int itest = 0; itest < 6; ++itest )
    {
    for( int iside = 0; iside < 2; ++iside )
      {
      for( int itran = 0; itran < 2; ++itran )
        {
        m = 128 * (itest + 1) + 64;
        n = 128 * (itest + 1) + 64;
        k = 128 * (itest + 1) + 64;
        nb  = magma_get_sgeqrf_nb( m, n );
        ldc = magma_roundup( m, 32 );
        // A is mm x k == m x k (left) or n x k (right)
        mm = (side[iside] == MagmaLeft ? m : n);
        lda = magma_roundup( mm, 32 );

        // need at least 2*nb*nb for geqlf
        lwork_max = std::max( std::max( m*nb, n*nb ), 2*nb*nb );
        // this rounds it up slightly if needed to agree with lwork query below
        lwork_max = magma_int_t( magma_smake_lwork( lwork_max ));

        REQUIRE( magma_smalloc_cpu( &C,     ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &R,     ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &A,     lda*k ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &hwork, lwork_max ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &tau,   k ) == MAGMA_SUCCESS );

        REQUIRE( magma_smalloc( &dC, ldc*n ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc( &dA, lda*k ) == MAGMA_SUCCESS );

        // C is full, mxn
        size = ldc*n;
        arma::fmat C_alias(C, ldc, n, false, true);
        C_alias.randu();
        magma_ssetmatrix( m, n, C, ldc, dC, 0, ldc, queue );

        // A is mm x k
        arma::fmat A_alias(A, lda, k, false, true);
        A_alias.randu();

        // compute QL factorization to get Householder vectors in A, tau
        // note that magma_dgeqlf() is specific to test code; we do not have it in bandicoot directly
        magma_sgeqlf( mm, k, A, lda, tau, hwork, lwork_max, &info );
        magma_ssetmatrix( mm, k, A, lda, dA, 0, lda, queue );
        if (info != 0)
          {
          std::cerr << "magma_sgeqlf returned error " << info << ": " << magma::error_as_string( info ) << std::endl;
          }
        REQUIRE( info == 0 );

        magma_sgetmatrix( mm, k, dA, 0, lda, A, lda, queue );

        magma_sormql2_gpu( side[iside], trans[itran],
                           m, n, k,
                           dA, 0, lda, tau, dC, 0, ldc, A, lda, &info );

        magma_sgetmatrix( m, n, dC, 0, ldc, R, ldc, queue );

        lapack_test::ormql(lapack_side_const(side[iside])[0], lapack_trans_const(trans[itran])[0],
                           m, n, k,
                           A, lda, tau, C, ldc, hwork, lwork_max, &info);
        if (info != 0)
          {
          std::cerr << "LAPACK sormql returned error " << info << ": " << magma::error_as_string( info ) << std::endl;
          }
        REQUIRE( info == 0 );

        /* =====================================================================
           compute relative error |QC_magma - QC_lapack| / |QC_lapack|
           =================================================================== */
        size = ldc*n;
        blas::axpy(size, c_neg_one, C, ione, R, ione);
        Cnorm = lapack::lange('F', m, n, C, ldc, work);
        error = lapack::lange('F', m, n, R, ldc, work) / (std::sqrt(m*n) * Cnorm);

        REQUIRE (error < tol);

        magma_free_cpu( C );
        magma_free_cpu( R );
        magma_free_cpu( A );
        magma_free_cpu( hwork );
        magma_free_cpu( tau );

        magma_free( dC );
        magma_free( dA );
        }
      }
    }

  magma_queue_destroy( queue );
  }

#endif
