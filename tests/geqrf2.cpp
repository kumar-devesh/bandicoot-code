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
#include "catch.hpp"
#include "def_lapack_test.hpp"

using namespace coot;

// These tests are from clMAGMA, only useful for the OpenCL backend.

#if defined(COOT_USE_OPENCL)

TEST_CASE("magma_geqrf2_1", "[geqrf2]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  const double             d_neg_one = MAGMA_D_NEG_ONE;
  const double             d_one     = MAGMA_D_ONE;
  const double c_neg_one = MAGMA_D_NEG_ONE;
  const double c_one     = MAGMA_D_ONE;
  const double c_zero    = MAGMA_D_ZERO;

  double           Anorm, error=0, error2=0;
  double *h_A, *h_R, *tau, *h_work, tmp[1], unused[1];
  magmaDouble_ptr d_A;
  magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn;

  double tol = 30 * std::numeric_limits<double>::epsilon();

  // TODO: can we reuse this?
  magma_queue_t queue = NULL;
  magma_device_t cdev;
  magma_getdevice( &cdev );
  magma_queue_create( cdev, &queue );

  for (int itest = 0; itest < 10; ++itest)
    {
    M = 128 * (itest + 1) + 64;
    N = 128 * (itest + 1) + 64;
    min_mn = std::min( M, N );
    lda    = M;
    n2     = lda*N;
    ldda   = magma_roundup( M, 32 );  // multiple of 32 by default

    lwork = -1;
    coot_fortran(coot_dgeqrf)(&M, &N, unused, &M, unused, tmp, &lwork, &info);
    lwork = (magma_int_t) tmp[0];

    REQUIRE( magma_dmalloc_cpu( &tau,    min_mn ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_A,    n2     ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_work, lwork  ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_pinned( &h_R,    n2     ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc( &d_A,    ldda*N ) == MAGMA_SUCCESS );

    // Generate a uniform random matrix for h_A.
    arma::mat h_A_alias(h_A, lda, N, false, true);
    h_A_alias.randu();
    // Copy that matrix to h_R and then d_A.
    coot_fortran(coot_dlacpy)( "A", &M, &N, h_A, &lda, h_R, &lda );
    magma_dsetmatrix( M, N, h_R, lda, d_A, 0, ldda, queue );

    // LAPACK complaint arguments
    magma_dgeqrf2_gpu( M, N, d_A, 0, ldda, tau, &info );

    if (info != 0)
      {
      std::cerr << "magma_dgeqrf2_gpu returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    /* =====================================================================
       Check the result, following zqrt01 except using the reduced Q.
       This works for any M,N (square, tall, wide).
       Only for version 2, which has LAPACK complaint output.
       Or   for version 3, after restoring diagonal blocks of A above.
       =================================================================== */
    magma_dgetmatrix( M, N, d_A, 0, ldda, h_R, lda, queue );

    magma_int_t ldq = M;
    magma_int_t ldr = min_mn;
    double *Q, *R;
    double *work;
    REQUIRE( magma_dmalloc_cpu( &Q,    ldq*min_mn ) == MAGMA_SUCCESS );  // M by K
    REQUIRE( magma_dmalloc_cpu( &R,    ldr*N ) == MAGMA_SUCCESS );       // K by N
    REQUIRE( magma_dmalloc_cpu( &work, min_mn ) == MAGMA_SUCCESS );

    // generate M by K matrix Q, where K = min(M,N)
    coot_fortran(coot_dlacpy)("L", &M, &min_mn, h_R, &lda, Q, &ldq);
    coot_fortran(coot_dorgqr)(&M, &min_mn, &min_mn, Q, &ldq, tau, h_work, &lwork, &info);
    REQUIRE( info == 0 );

    // copy K by N matrix R
    coot_fortran(coot_dlaset)("L", &min_mn, &N, &c_zero, &c_zero, R, &ldr);
    coot_fortran(coot_dlacpy)("U", &min_mn, &N, h_R, &lda, R, &ldr);

    // error = || R - Q^H*A || / (N * ||A||)
    coot_fortran(coot_dgemm)("C", "N", &min_mn, &N, &M,
                   &c_neg_one, Q, &ldq, h_A, &lda, &c_one, R, &ldr );
    arma::mat hA_alias(h_A, M, N, false, true);
    Anorm = coot_fortran(coot_dlange)( "1", &M,      &N, h_A, &lda, work );
    error = coot_fortran(coot_dlange)( "1", &min_mn, &N, R,   &ldr, work );
    if ( N > 0 && Anorm > 0 )
      error /= (N*Anorm);

    // set R = I (K by K identity), then R = I - Q^H*Q
    // error = || I - Q^H*Q || / N
    coot_fortran(coot_dlaset)("U", &min_mn, &min_mn, &c_zero, &c_one, R, &ldr);
    coot_fortran(coot_dsyrk)("U", "C", &min_mn, &M, &d_neg_one, Q, &ldq, &d_one, R, &ldr);
    error2 = coot_fortran(coot_dlansy)("1", "U", &min_mn, R, &ldr, work);
    if ( N > 0 )
      error2 /= N;

    magma_free_cpu( Q    );  Q    = NULL;
    magma_free_cpu( R    );  R    = NULL;
    magma_free_cpu( work );  work = NULL;

    REQUIRE( error < tol );
    REQUIRE( error2 < tol );

    magma_free_cpu( tau    );
    magma_free_cpu( h_A    );
    magma_free_cpu( h_work );
    magma_free_pinned( h_R );
    magma_free( d_A );
    }

  #endif
  }
