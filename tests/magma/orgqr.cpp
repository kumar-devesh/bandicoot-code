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

TEST_CASE("magma_dorgqr2_1", "[orgqr2]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double           Anorm, error, work[1];
  double  c_neg_one = MAGMA_D_NEG_ONE;
  double *hA, *hR, *tau, *h_work, *hT;
  magmaDouble_ptr dA, dT;
  magma_int_t m, n, k;
  magma_int_t n2, lda, ldda, lwork, min_mn, nb, info;
  magma_int_t ione     = 1;

  magma_queue_t queue = magma_queue_create();

  double tol = 30 * std::numeric_limits<double>::epsilon();

  for (int itest = 0; itest < 10; ++itest)
    {
    // Strange sizes seem necessary to work around what appear to be nvidia
    // OpenCL driver bugs.
    m = 64 * (itest + 1) + 75;
    n = 64 * (itest + 1) + 75;
    k = 64 * (itest + 1) + 75;

    lda  = m;
    ldda = magma_roundup( m, 32 );  // multiple of 32 by default
    n2 = lda*n;
    min_mn = std::min(m, n);
    nb = magma_get_dgeqrf_nb( m, n );
    lwork  = n*nb;

    REQUIRE( magma_dmalloc_pinned( &hR,     lda*n  ) == MAGMA_SUCCESS );

    REQUIRE( magma_dmalloc_cpu( &hA,     lda*n  ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &tau,    min_mn ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &h_work, lwork  ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &hT,     min_mn*nb ) == MAGMA_SUCCESS );

    REQUIRE( magma_dmalloc( &dA,     ldda*n ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc( &dT,     ( 2*min_mn + magma_roundup( n, 32 ) )*nb ) == MAGMA_SUCCESS );

    // By default this uses a random uniform matrix.  We use Armadillo to create
    // that.
    arma::Mat<double> hA_alias(hA, lda, n, false, true);
    hA_alias.randu();
    coot_fortran(coot_dlacpy)( "F", &m, &n, hA, &lda, hR, &lda );

    Anorm = coot_fortran(coot_dlange)("F", &m, &n, hA, &lda, work );

    /* ====================================================================
       Performs operation using MAGMA
       =================================================================== */
    // first, get QR factors in both hA and hR
    // okay that magma_dgeqrf_gpu has special structure for R; R isn't used here.
    magma_dsetmatrix( m, n, hA, lda, dA, 0, ldda, queue );
    magma_dgeqrf_gpu( m, n, dA, 0, ldda, tau, dT, 0, &info );
    if (info != 0)
      {
      std::cerr << "magma_dgeqrf() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );
    magma_dgetmatrix( m, n, dA, 0, ldda, hA, lda, queue );
    coot_fortran(coot_dlacpy)( "F", &m, &n, hA, &lda, hR, &lda );
    magma_dgetmatrix( nb, min_mn, dT, 0, nb, hT, nb, queue );  // for multi GPU

    magma_dorgqr2( m, n, k, hR, lda, tau, &info );
    if (info != 0)
      {
      std::cerr << "magma_dorgqr2() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    /* =====================================================================
       Performs operation using LAPACK
       =================================================================== */
    coot_fortran(coot_dorgqr)( &m, &n, &k, hA, &lda, tau, h_work, &lwork, &info );
    if (info != 0)
      {
      std::cerr << "coot_dorgqr returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    // compute relative error |R|/|A| := |Q_magma - Q_lapack|/|A|
    coot_fortran(coot_daxpy)( &n2, &c_neg_one, hA, &ione, hR, &ione );
    error = coot_fortran(coot_dlange)("F", &m, &n, hR, &lda, work) / Anorm;

    REQUIRE( error < tol );

    magma_free_pinned( hR     );

    magma_free_cpu( hA  );
    magma_free_cpu( tau );
    magma_free_cpu( h_work );
    magma_free_cpu( hT  );

    magma_free( dA );
    magma_free( dT );
    }
  }



TEST_CASE("magma_sorgqr2_1", "[orgqr2]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  float           Anorm, error, work[1];
  float  c_neg_one = MAGMA_S_NEG_ONE;
  float *hA, *hR, *tau, *h_work, *hT;
  magmaFloat_ptr dA, dT;
  magma_int_t m, n, k;
  magma_int_t n2, lda, ldda, lwork, min_mn, nb, info;
  magma_int_t ione     = 1;

  magma_queue_t queue = magma_queue_create();

  float tol = 30 * std::numeric_limits<float>::epsilon();

  for (int itest = 0; itest < 10; ++itest)
    {
    // Strange sizes seem necessary to work around what appear to be nvidia
    // OpenCL driver bugs.
    m = 64 * (itest + 1) + 75;
    n = 64 * (itest + 1) + 75;
    k = 64 * (itest + 1) + 75;

    lda  = m;
    ldda = magma_roundup( m, 32 );  // multiple of 32 by default
    n2 = lda*n;
    min_mn = std::min(m, n);
    nb = magma_get_dgeqrf_nb( m, n );
    lwork  = n*nb;

    REQUIRE( magma_smalloc_pinned( &hR,     lda*n  ) == MAGMA_SUCCESS );

    REQUIRE( magma_smalloc_cpu( &hA,     lda*n  ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &tau,    min_mn ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &h_work, lwork  ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc_cpu( &hT,     min_mn*nb ) == MAGMA_SUCCESS );

    REQUIRE( magma_smalloc( &dA,     ldda*n ) == MAGMA_SUCCESS );
    REQUIRE( magma_smalloc( &dT,     ( 2*min_mn + magma_roundup( n, 32 ) )*nb ) == MAGMA_SUCCESS );

    // By default this uses a random uniform matrix.  We use Armadillo to create
    // that.
    arma::Mat<float> hA_alias(hA, lda, n, false, true);
    hA_alias.randu();
    coot_fortran(coot_slacpy)( "F", &m, &n, hA, &lda, hR, &lda );

    Anorm = coot_fortran(coot_slange)("F", &m, &n, hA, &lda, work );

    /* ====================================================================
       Performs operation using MAGMA
       =================================================================== */
    // first, get QR factors in both hA and hR
    // okay that magma_sgeqrf_gpu has special structure for R; R isn't used here.
    magma_ssetmatrix( m, n, hA, lda, dA, 0, ldda, queue );
    magma_sgeqrf_gpu( m, n, dA, 0, ldda, tau, dT, 0, &info );
    if (info != 0)
      {
      std::cerr << "magma_sgeqrf() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );
    magma_sgetmatrix( m, n, dA, 0, ldda, hA, lda, queue );
    coot_fortran(coot_slacpy)( "F", &m, &n, hA, &lda, hR, &lda );
    magma_sgetmatrix( nb, min_mn, dT, 0, nb, hT, nb, queue );  // for multi GPU

    magma_sorgqr2( m, n, k, hR, lda, tau, &info );
    if (info != 0)
      {
      std::cerr << "magma_sorgqr2() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    /* =====================================================================
       Performs operation using LAPACK
       =================================================================== */
    coot_fortran(coot_sorgqr)( &m, &n, &k, hA, &lda, tau, h_work, &lwork, &info );
    if (info != 0)
      {
      std::cerr << "coot_sorgqr returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    REQUIRE( info == 0 );

    // compute relative error |R|/|A| := |Q_magma - Q_lapack|/|A|
    coot_fortran(coot_saxpy)( &n2, &c_neg_one, hA, &ione, hR, &ione );
    error = coot_fortran(coot_slange)("F", &m, &n, hR, &lda, work) / Anorm;

    REQUIRE( error < tol );

    magma_free_pinned( hR     );

    magma_free_cpu( hA  );
    magma_free_cpu( tau );
    magma_free_cpu( h_work );
    magma_free_cpu( hT  );

    magma_free( dA );
    magma_free( dT );
    }
  }

#endif
