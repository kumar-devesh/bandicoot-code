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

TEST_CASE("magma_dorglq_1", "[orglq]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double           Anorm, error, work[1];
  double  c_neg_one = MAGMA_D_NEG_ONE;
  double *hA, *hR, *tau, *h_work;
  magmaDouble_ptr dA, dT;
  magma_int_t m, n, k;
  magma_int_t n2, lda, ldda, lwork, min_mn, nb, info;
  magma_int_t ione     = 1;

  // TODO: can we reuse this?
  magma_queue_t queue = NULL;
  magma_device_t cdev;
  magma_getdevice( &cdev );
  magma_queue_create( cdev, &queue );

  double tol = 30 * std::numeric_limits<double>::epsilon();

  for( int itest = 0; itest < 10; ++itest )
    {
    m = 128 * (itest + 1) + 64;
    n = 128 * (itest + 1) + 64;
    k = 128 * (itest + 1) + 64;

    lda  = m;
    ldda = magma_roundup( m, 32 );  // multiple of 32 by default
    n2 = lda*n;
    min_mn = std::min(m, n);
    nb = magma_get_dgelqf_nb( m, n );
    lwork  = m*nb;

    REQUIRE( magma_dmalloc_pinned( &h_work, lwork  ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_pinned( &hR,     lda*n  ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &hA,     lda*n  ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc_cpu( &tau,    min_mn ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc( &dA,     ldda*n ) == MAGMA_SUCCESS );
    REQUIRE( magma_dmalloc( &dT,     ( 2*min_mn + magma_roundup( n, 32 ) )*nb ) == MAGMA_SUCCESS );

    // By default this test uses a random mxn matrix; we'll just use existing functionality for this.
    arma::Mat<double> alias(hA, m, n, false, true);
    alias.randu();

    const char uplo = 'A';
    coot_fortran(coot_dlacpy)(&uplo, &m, &n, hA, &lda, hR, &lda);

    Anorm = coot_fortran(coot_dlange)("f", &m, &n, hA, &lda, work );

    /* ====================================================================
       Performs operation using MAGMA
       =================================================================== */
    // first, get LQ factors in both hA and hR
    magma_dsetmatrix( m, n, hA, lda, dA, 0, ldda, queue );
    magma_dgelqf_gpu( m, n, dA, 0, ldda, tau, h_work, lwork, &info );
    if (info != 0)
      {
      std::cerr << "magma_dgelqf_gpu returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }
    magma_dgetmatrix( m, n, dA, 0, ldda, hA, lda, queue );
    coot_fortran(coot_dlacpy)(&uplo, &m, &n, hA, &lda, hR, &lda);

    magma_dorglq( m, n, k, hR, lda, tau, h_work, lwork, &info );
    if (info != 0)
      {
      std::cerr << "magma_dorglq returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }

    coot_fortran(coot_dorglq)(&m, &n, &k, hA, &lda, tau, h_work, &lwork, &info);
    if (info != 0)
      {
      std::cerr << "dorglq returned error " << info << ": " << magma::error_as_string(info) << std::endl;
      }

    // compute relative error |R|/|A| := |Q_magma - Q_lapack|/|A|
    coot_fortran(coot_daxpy)(&n2, &c_neg_one, hA, &ione, hR, &ione);
    error = coot_fortran(coot_dlange)("f", &m, &n, hR, &lda, work) / Anorm;

    REQUIRE( error < tol );

    magma_free_pinned( h_work );
    magma_free_pinned( hR     );
    magma_free_cpu( hA  );
    magma_free_cpu( tau );
    magma_free( dA );
    magma_free( dT );
    }

  #endif
  }
