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

TEST_CASE("magma_dorgbr_1", "[orgbr]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double           Anorm, error, work[1];
  double  c_neg_one = MAGMA_D_NEG_ONE;
  double *d, *e;
  double *hA, *hR, *tauq, *taup, *h_work;
  magma_int_t m, n, k;
  magma_int_t n2, lda, lwork, min_mn, nb, info;
  magma_int_t ione     = 1;
  magma_vect_t vect;

  double tol = 30 * std::numeric_limits<double>::epsilon();

  magma_vect_t vects[] = { MagmaQ, MagmaP };

  for (int itest = 0; itest < 10; ++itest)
    {
    for (int ivect = 0; ivect < 2; ++ivect)
      {
      m = 128 * itest + 65;
      n = 128 * itest + 65;
      k = 128 * itest + 65;
      vect = vects[ivect];

      lda = m;
      n2 = lda*n;
      min_mn = std::min(m, n);
      nb = std::max( magma_get_dgelqf_nb( m, n ),
                     magma_get_dgebrd_nb( m, n ));
      lwork  = (m + n)*nb;

      REQUIRE( magma_dmalloc_pinned( &h_work, lwork  ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_pinned( &hR,     lda*n  ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_cpu( &hA,     lda*n  ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_cpu( &tauq,   min_mn ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_cpu( &taup,   min_mn ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_cpu( &d,      min_mn   ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_cpu( &e,      min_mn-1 ) == MAGMA_SUCCESS );

      // By default a random uniform matrix is used.
      arma::Mat<double> hA_alias(hA, lda, n, false, true);
      hA_alias.randu();
      coot_fortran(coot_dlacpy)( "A", &m, &n, hA, &lda, hR, &lda );

      Anorm = coot_fortran(coot_dlange)("F", &m, &n, hA, &lda, work );

      /* ====================================================================
         Performs operation using MAGMA
         =================================================================== */
      // first, get GEBRD factors in both hA and hR
      magma_dgebrd( m, n, hA, lda, d, e, tauq, taup, h_work, lwork, &info );
      if (info != 0)
        {
        std::cerr << "magma_dgelqf() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
        }
      REQUIRE( info == 0 );
      coot_fortran(coot_dlacpy)( "A", &m, &n, hA, &lda, hR, &lda );

      if (vect == MagmaQ)
        {
        magma_dorgbr( vect, m, n, k, hR, lda, tauq, h_work, lwork, &info );
        }
      else
        {
        magma_dorgbr( vect, m, n, k, hR, lda, taup, h_work, lwork, &info );
        }

      if (info != 0)
        {
        std::cerr << "magma_dorgbr() returned error " << info << ": " << magma::error_as_string(info) << std::endl;
        }

      /* =====================================================================
         Performs operation using LAPACK
         =================================================================== */
      if (vect == MagmaQ)
        {
        coot_fortran(coot_dorgbr)( "Q", &m, &n, &k, hA, &lda, tauq, h_work, &lwork, &info );
        }
      else
        {
        coot_fortran(coot_dorgbr)( "P", &m, &n, &k, hA, &lda, taup, h_work, &lwork, &info );
        }

      if (info != 0)
        {
        std::cerr << "dorgbr returned error " << info << ": " << magma::error_as_string(info) << std::endl;
        }
      REQUIRE( info == 0 );

      // compute relative error |R|/|A| := |Q_magma - Q_lapack|/|A|
      coot_fortran(coot_daxpy)( &n2, &c_neg_one, hA, &ione, hR, &ione );
      error = coot_fortran(coot_dlange)("F", &m, &n, hR, &lda, work) / Anorm;

      REQUIRE( error < tol );

      magma_free_pinned( h_work );
      magma_free_pinned( hR     );

      magma_free_cpu( hA   );
      magma_free_cpu( tauq );
      magma_free_cpu( taup );
      magma_free_cpu( d );
      magma_free_cpu( e );
      }
    }
  }

#endif
