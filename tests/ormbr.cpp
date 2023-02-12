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

inline
const char*
lapack_vect_const( magma_vect_t magma_const )
  {
  REQUIRE( magma_const >= MagmaQ );
  REQUIRE( magma_const <= MagmaP );
  return get_magma2lapack_constants()[ magma_const ];
  }



inline
const char*
lapack_direct_const( magma_direct_t magma_const )
  {
  REQUIRE( magma_const >= MagmaForward );
  REQUIRE( magma_const <= MagmaBackward );
  return get_magma2lapack_constants()[ magma_const ];
  }



inline
const char*
lapack_storev_const( magma_storev_t magma_const )
  {
  REQUIRE( magma_const >= MagmaColumnwise );
  REQUIRE( magma_const <= MagmaRowwise    );
  return get_magma2lapack_constants()[ magma_const ];
  }



TEST_CASE("magma_dormbr_1", "[ormbr]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double Cnorm, error, dwork[1];
  double c_neg_one = MAGMA_D_NEG_ONE;
  magma_int_t ione = 1;
  magma_int_t m, n, k, mm, nn, nq, size, info;
  magma_int_t ISEED[4] = {0,0,0,1};
  magma_int_t nb, ldc, lda, lwork, lwork_max;
  double *C, *R, *A, *work, *tau, *tauq, *taup;
  double *d, *e;

  double tol = 60 * std::numeric_limits<double>::epsilon();

  // test all combinations of input parameters
  magma_vect_t  vect [] = { MagmaQ,          MagmaP       };
  magma_side_t  side [] = { MagmaLeft,       MagmaRight   };
  magma_trans_t trans[] = { MagmaTrans, MagmaNoTrans };

  for (int itest = 0; itest < 10; ++itest)
    {
    for (int ivect = 0; ivect < 2; ++ivect)
      {
      for (int iside = 0; iside < 2; ++iside)
        {
        for (int itran = 0; itran < 2; ++itran)
          {
          m = 128 * (itest + 1) + 63;
          n = 128 * (itest + 1) + 63;
          k = 128 * (itest + 1) + 63;
          nb  = magma_get_dgebrd_nb( m, n );
          ldc = m;
          // A is mm x nn == nq x k (vect=Q) or k x nq (vect=P)
          // where nq=m (left) or nq=n (right)
          nq  = (side[iside] == MagmaLeft ? m  : n );
          mm  = (vect[ivect] == MagmaQ    ? nq : k );
          nn  = (vect[ivect] == MagmaQ    ? k  : nq);
          lda = mm;

          // workspace for gebrd is (mm + nn)*nb
          // workspace for unmbr is m*nb or n*nb, depending on side
          lwork_max = std::max( (mm + nn)*nb, std::max( m*nb, n*nb ));
          // this rounds it up slightly if needed to agree with lwork query below
          lwork_max = magma_int_t( double( magma_dmake_lwork( lwork_max )));

          REQUIRE( magma_dmalloc_cpu( &C,    ldc*n ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_cpu( &R,    ldc*n ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_cpu( &A,    lda*nn ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_cpu( &work, lwork_max ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_cpu( &d,    std::min(mm,nn) ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_cpu( &e,    std::min(mm,nn) ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_cpu( &tauq, std::min(mm,nn) ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_cpu( &taup, std::min(mm,nn) ) == MAGMA_SUCCESS );

          // C is full, m x n
          size = ldc*n;
          coot_fortran(coot_dlarnv)( &ione, ISEED, &size, C );
          coot_fortran(coot_dlacpy)( "F", &m, &n, C, &ldc, R, &ldc );

          // By default the original code used uniform random matrices.
          arma::Mat<double> A_alias(A, lda, nn, false, true);
          A_alias.randu();

          // compute BRD factorization to get Householder vectors in A, tauq, taup
          magma_dgebrd( mm, nn, A, lda, d, e, tauq, taup, work, lwork_max, &info );
          if (info != 0)
            {
            std::cerr << "magma_dgebrd returned error " << info << ": " << magma::error_as_string(info) << std::endl;
            }
          REQUIRE( info == 0 );

          if ( vect[ivect] == MagmaQ )
            {
            tau = tauq;
            }
          else
            {
            tau = taup;
            }

          /* =====================================================================
             Performs operation using LAPACK
             =================================================================== */
          coot_fortran(coot_dormbr)( lapack_vect_const( vect[ivect] ),
                                     lapack_side_const( side[iside] ),
                                     lapack_trans_const( trans[itran] ),
                                     &m, &n, &k,
                                     A, &lda, tau, C, &ldc, work, &lwork_max, &info );
          if (info != 0)
            {
            std::cerr << "dormbr returned error " << info << ": " << magma::error_as_string(info) << std::endl;
            }
          REQUIRE( info == 0 );

          /* ====================================================================
             Performs operation using MAGMA
             =================================================================== */
          // query for workspace size
          lwork = -1;
          magma_dormbr( vect[ivect], side[iside], trans[itran],
                        m, n, k,
                        A, lda, tau, R, ldc, work, lwork, &info );
          if (info != 0)
            {
            std::cerr << "magma_dormbr (lwork query) returned error " << info << ": " << magma::error_as_string(info) << std::endl;
            }
          REQUIRE( info == 0 );
          lwork = magma_int_t( work[0] );
          if ( lwork < 0 || lwork > lwork_max )
            {
            std::cerr << "warning: optimal lwork " << lwork << " > allocated lwork_max " << lwork_max << std::endl;
            lwork = lwork_max;
            }

          magma_dormbr( vect[ivect], side[iside], trans[itran],
                        m, n, k,
                        A, lda, tau, R, ldc, work, lwork, &info );
          if (info != 0)
            {
            std::cerr << "magma_dormbr returned error " << info << ": " << magma::error_as_string(info) << std::endl;
            }
          REQUIRE( info == 0 );

          /* =====================================================================
             compute relative error |QC_magma - QC_lapack| / |QC_lapack|
             =================================================================== */
          size = ldc*n;
          coot_fortran(coot_daxpy)( &size, &c_neg_one, C, &ione, R, &ione );
          Cnorm = coot_fortran(coot_dlange)( "F", &m, &n, C, &ldc, dwork );
          error = coot_fortran(coot_dlange)( "F", &m, &n, R, &ldc, dwork ) / (std::sqrt(m*n) * Cnorm);

          REQUIRE( error < tol );

          magma_free_cpu( C );
          magma_free_cpu( R );
          magma_free_cpu( A );
          magma_free_cpu( work );
          magma_free_cpu( d );
          magma_free_cpu( e );
          magma_free_cpu( taup );
          magma_free_cpu( tauq );
          }
        }
      }
    }
  }

#endif
