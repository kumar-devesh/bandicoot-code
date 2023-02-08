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

TEST_CASE("magma_dgebrd_1", "[gebrd]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double *h_A, *h_Q, *h_PT, *h_work;
  double *taup, *tauq;
  double      *diag, *offdiag;
  double      eps, result[3] = {0., 0., 0.};
  magma_int_t M, N, n2, lda, lhwork, info, minmn, nb;
  magma_int_t ione     = 1;

  eps = std::numeric_limits<double>::epsilon();

  double tol = 30. * eps;

  for( int itest = 0; itest < 4; ++itest )
    {
    // Smaller test sizes are to work around nvidia OpenCL compiler bugs that manifest in -9999 GEMM compilation errors in clBLAS.  Sigh.
    M = 64 * (itest + 1) + 64;
    N = 64 * (itest + 1) + 64;
    minmn  = std::min(M, N);
    nb     = magma_get_dgebrd_nb(N, N);
    lda    = M;
    n2     = lda*N;
    lhwork = (M + N)*nb;

    REQUIRE( magma_malloc_cpu( (void**) &h_A,     (lda*N   ) * sizeof(double) ) == MAGMA_SUCCESS );
    REQUIRE( magma_malloc_cpu( (void**) &tauq,    (minmn   ) * sizeof(double) ) == MAGMA_SUCCESS );
    REQUIRE( magma_malloc_cpu( (void**) &taup,    (minmn   ) * sizeof(double) ) == MAGMA_SUCCESS );
    REQUIRE( magma_malloc_cpu( (void**) &diag,    (minmn   ) * sizeof(double) ) == MAGMA_SUCCESS );
    REQUIRE( magma_malloc_cpu( (void**) &offdiag, (minmn-1 ) * sizeof(double) ) == MAGMA_SUCCESS );

    REQUIRE( magma_malloc_cpu( (void**) &h_Q,     (lda*N   ) * sizeof(double) ) == MAGMA_SUCCESS );
    REQUIRE( magma_malloc_cpu( (void**) &h_work,  (lhwork  ) * sizeof(double) ) == MAGMA_SUCCESS );

    /* Initialize the matrices */
    blas_int blas_ione(ione);
    blas_int blas_ISEED[4] = {0,0,0,1};
    blas_int blas_n2(n2);
    blas_int blas_M(M);
    blas_int blas_N(N);
    blas_int blas_lda(lda);
    char uplo = 'A';
    coot_fortran(coot_dlarnv)( &blas_ione, blas_ISEED, &blas_n2, h_A );
    coot_fortran(coot_dlacpy)( &uplo, &blas_M, &blas_N, h_A, &blas_lda, h_Q, &blas_lda );

    /* ====================================================================
       Performs operation using MAGMA
       =================================================================== */
    magma_dgebrd( M, N, h_Q, lda,
                  diag, offdiag, tauq, taup,
                  h_work, lhwork, &info );
    if (info != 0)
      {
      std::cerr << "magma_dgebrd returned error " << info << ": " << magma::error_as_string( info ) << std::endl;
      }
    REQUIRE( info == 0 );

    // Now check the factorization.

    // dorgbr prefers minmn*NB
    // dbdt01 needs M+N
    // dort01 prefers minmn*(minmn+1) to check Q and P
    blas_int lwork_err;
    double *h_work_err;
    lwork_err = std::max( minmn * nb, M+N );
    lwork_err = std::max( lwork_err, minmn*(minmn+1) );
    REQUIRE( magma_malloc_cpu( (void**) &h_PT,       (lda*N     ) * sizeof(double) ) == MAGMA_SUCCESS );
    REQUIRE( magma_malloc_cpu( (void**) &h_work_err, (lwork_err ) * sizeof(double) ) == MAGMA_SUCCESS );

    coot_fortran(coot_dlacpy)( &uplo, &blas_M, &blas_N, h_Q, &blas_lda, h_PT, &blas_lda );

    blas_int blas_info;
    blas_int blas_minmn(minmn);
    // generate Q & P'
    char blas_q = 'Q';
    coot_fortran(coot_dorgbr)( &blas_q, &blas_M, &blas_minmn, &blas_N, h_Q, &blas_lda, tauq, h_work_err, &lwork_err, &blas_info );
    if (blas_info != 0)
      {
      std::cerr << "lapackf77_dorgbr #1 returned error " << info << ": " << magma::error_as_string( magma_int_t(blas_info) ) << std::endl;
      }
    REQUIRE( blas_info == 0 );
    char blas_p = 'P';
    coot_fortran(coot_dorgbr)( &blas_p, &blas_minmn, &blas_N, &blas_M, h_PT, &blas_lda, taup, h_work_err, &lwork_err, &blas_info );
    if (blas_info != 0)
      {
      std::cerr << "lapackf77_dorgbr #2 returned error " << info << ": " << magma::error_as_string( magma_int_t(blas_info) ) << std::endl;
      }
    REQUIRE( blas_info == 0 );

    // Test 1:  Check the decomposition A := Q * B * PT
    //      2:  Check the orthogonality of Q
    //      3:  Check the orthogonality of PT
    char blas_c = 'C';
    char blas_r = 'R';
    coot_fortran(coot_dbdt01)(&blas_M, &blas_N, &blas_ione, h_A, &blas_lda, h_Q, &blas_lda, diag, offdiag, h_PT, &blas_lda, h_work_err, &result[0]);
    coot_fortran(coot_dort01)(&blas_c, &blas_M, &blas_minmn, h_Q, &blas_lda, h_work_err, &lwork_err, &result[1]);
    coot_fortran(coot_dort01)(&blas_r, &blas_minmn, &blas_N, h_PT, &blas_lda, h_work_err, &lwork_err, &result[2]);

    magma_free_cpu( h_PT );
    magma_free_cpu( h_work_err );

    REQUIRE( result[0] * eps < tol );
    REQUIRE( result[1] * eps < tol );
    REQUIRE( result[2] * eps < tol );

    magma_free_cpu( h_A     );
    magma_free_cpu( tauq    );
    magma_free_cpu( taup    );
    magma_free_cpu( diag    );
    magma_free_cpu( offdiag );

    magma_free_cpu( h_Q    );
    magma_free_cpu( h_work );
    }
  }

#endif
