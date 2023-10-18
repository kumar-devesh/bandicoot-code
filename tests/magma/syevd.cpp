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

using namespace coot;

// These tests are from clMAGMA, only useful for the OpenCL backend.

#if defined(COOT_USE_OPENCL)

TEST_CASE("magma_dsyevd_1", "[syevd]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  if (!coot_rt_t::is_supported_type<double>())
    {
    return;
    }

  /* Constants */
  const double d_zero = 0;
  const magma_int_t izero = 0;
  const magma_int_t ione  = 1;

  /* Local variables */
  double *h_A, *h_R, *h_work, aux_work[1], unused[1];
  magmaDouble_ptr d_R;
  double *w1, *w2, result[4]={0, 0, 0, 0}, eps, runused[1];
  magma_int_t *iwork, aux_iwork[1];
  magma_int_t N, Nfound, info, lwork, liwork, lda, ldda;
  eps = lapack::lamch<double>('E');

  double tol    = 30 * lapack::lamch<double>('E');
  double tolulp = 30 * lapack::lamch<double>('P');

  magma_queue_t queue = magma_queue_create();

  magma_vec_t  jobzs[2] = { MagmaNoVec, MagmaVec };
  magma_uplo_t uplos[2] = { MagmaLower, MagmaUpper };

  for( int itest = 0; itest < 7; ++itest )
    {
    for ( int ijob = 0; ijob < 2; ++ijob)
      {
      for ( int iuplo = 0; iuplo < 2; ++iuplo )
        {
        N = 128 * (itest + 1) + 64;
        Nfound = N;
        lda  = N;
        ldda = magma_roundup( N, 32 );  // multiple of 32 by default

        magma_dsyevd_gpu( jobzs[ijob], uplos[iuplo],
                          N, NULL, 0, ldda, NULL,  // A, w
                          NULL, lda,            // host A
                          aux_work,  -1,
                          aux_iwork, -1,
                          &info );

        lwork  = (magma_int_t) aux_work[0];
        liwork = aux_iwork[0];

        /* Allocate host memory for the matrix */
        REQUIRE( magma_dmalloc_cpu( &h_A,    N*lda  ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &w1,     N      ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_cpu( &w2,     N      ) == MAGMA_SUCCESS );
        REQUIRE( magma_imalloc_cpu( &iwork,  liwork ) == MAGMA_SUCCESS );

        REQUIRE( magma_dmalloc_pinned( &h_R,    N*lda  ) == MAGMA_SUCCESS );
        REQUIRE( magma_dmalloc_pinned( &h_work, lwork  ) == MAGMA_SUCCESS );

        REQUIRE( magma_dmalloc( &d_R,    N*ldda ) == MAGMA_SUCCESS );

        /* Clear eigenvalues, for |S-S_magma| check when fraction < 1. */
        lapack::laset('F', N, ione, d_zero, d_zero, w1, N);
        lapack::laset('F', N, ione, d_zero, d_zero, w2, N);

        /* Initialize the matrix */
        // We use a random symmetric matrix.
        arma::mat h_A_alias(h_A, N, lda, false, true);
        h_A_alias.randu();
        h_A_alias *= h_A_alias.t();
        magma_dsetmatrix( N, N, h_A, lda, d_R, 0, ldda, queue );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_dsyevd_gpu( jobzs[ijob], uplos[iuplo],
                          N, d_R, 0, ldda, w1,
                          h_R, lda,
                          h_work, lwork,
                          iwork, liwork,
                          &info );
        if (info != 0)
          {
          std::cerr << "magma_dsyevd_gpu returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        if ( jobzs[ijob] != MagmaNoVec )
          {
          /* =====================================================================
             Check the results following the LAPACK's [zcds]drvst routine.
             A is factored as A = U S U^H and the following 3 tests computed:
             (1)    | A - U S U^H | / ( |A| N ) if all eigenvectors were computed
                    | U^H A U - S | / ( |A| Nfound ) otherwise
             (2)    | I - U^H U   | / ( N )
             (3)    | S(with U) - S(w/o U) | / | S |    // currently disabled, but compares to LAPACK
             =================================================================== */
          magma_dgetmatrix( N, N, d_R, 0, ldda, h_R, lda, queue );

          double *work;
          REQUIRE( magma_dmalloc_cpu( &work, 2*N*N ) == MAGMA_SUCCESS );

          // e is unused since kband=0; tau is unused since itype=1
          if( Nfound == N )
            {
            lapack_test::syt21(ione, lapack_uplo_const(uplos[iuplo])[0], N, izero,
                               h_A, lda,
                               w1, runused,
                               h_R, lda,
                               h_R, lda,
                               unused, work,
                               &result[0]);
            }
          else
            {
            lapack_test::syt22(ione, lapack_uplo_const(uplos[iuplo])[0], N, Nfound, izero,
                               h_A, lda,
                               w1, runused,
                               h_R, lda,
                               h_R, lda,
                               unused, work,
                               &result[0]);
            }
          result[0] *= eps;
          result[1] *= eps;

          magma_free_cpu( work );
          work=NULL;
          }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        lapack::syevd(lapack_vec_const(jobzs[ijob])[0], lapack_uplo_const(uplos[iuplo])[0],
                      N, h_A, lda, w2,
                      h_work, lwork,
                      iwork, liwork,
                      &info);
        if (info != 0)
          {
          std::cerr << "LAPACK dsyevd returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }

        // compare eigenvalues
        double maxw=0, diff=0;
        for( int j=0; j < Nfound; j++ )
          {
          maxw = std::max(maxw, fabs(w1[j]));
          maxw = std::max(maxw, fabs(w2[j]));
          diff = std::max(diff, fabs(w1[j] - w2[j]));
          }
        result[3] = diff / (N*maxw);

        REQUIRE( result[0] < tol );
        REQUIRE( result[1] < tol );
        REQUIRE( result[3] < tolulp );

        magma_free_cpu( h_A    );
        magma_free_cpu( w1     );
        magma_free_cpu( w2     );
        magma_free_cpu( iwork  );

        magma_free_pinned( h_R    );
        magma_free_pinned( h_work );

        magma_free( d_R );
        }
      }
    }

  magma_queue_destroy(queue);
  }



TEST_CASE("magma_ssyevd_1", "[syevd]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  /* Constants */
  const float s_zero = 0;
  const magma_int_t izero = 0;
  const magma_int_t ione  = 1;

  /* Local variables */
  float *h_A, *h_R, *h_work, aux_work[1], unused[1];
  magmaFloat_ptr d_R;
  float *w1, *w2, result[4]={0, 0, 0, 0}, eps, runused[1];
  magma_int_t *iwork, aux_iwork[1];
  magma_int_t N, Nfound, info, lwork, liwork, lda, ldda;
  eps = lapack::lamch<float>('E');

  float tol    = 30 * lapack::lamch<float>('E');
  float tolulp = 30 * lapack::lamch<float>('P');

  magma_queue_t queue = magma_queue_create();

  magma_vec_t  jobzs[2] = { MagmaNoVec, MagmaVec };
  magma_uplo_t uplos[2] = { MagmaLower, MagmaUpper };

  for( int itest = 0; itest < 7; ++itest )
    {
    for ( int ijob = 0; ijob < 2; ++ijob)
      {
      for ( int iuplo = 0; iuplo < 2; ++iuplo )
        {
        N = 128 * (itest + 1) + 64;
        Nfound = N;
        lda  = N;
        ldda = magma_roundup( N, 32 );  // multiple of 32 by default

        magma_ssyevd_gpu( jobzs[ijob], uplos[iuplo],
                          N, NULL, 0, ldda, NULL,  // A, w
                          NULL, lda,            // host A
                          aux_work,  -1,
                          aux_iwork, -1,
                          &info );

        lwork  = (magma_int_t) aux_work[0];
        liwork = aux_iwork[0];

        /* Allocate host memory for the matrix */
        REQUIRE( magma_smalloc_cpu( &h_A,    N*lda  ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &w1,     N      ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_cpu( &w2,     N      ) == MAGMA_SUCCESS );
        REQUIRE( magma_imalloc_cpu( &iwork,  liwork ) == MAGMA_SUCCESS );

        REQUIRE( magma_smalloc_pinned( &h_R,    N*lda  ) == MAGMA_SUCCESS );
        REQUIRE( magma_smalloc_pinned( &h_work, lwork  ) == MAGMA_SUCCESS );

        REQUIRE( magma_smalloc( &d_R,    N*ldda ) == MAGMA_SUCCESS );

        /* Clear eigenvalues, for |S-S_magma| check when fraction < 1. */
        lapack::laset('F', N, ione, s_zero, s_zero, w1, N);
        lapack::laset('F', N, ione, s_zero, s_zero, w2, N);

        /* Initialize the matrix */
        // We use a random symmetric matrix.
        arma::fmat h_A_alias(h_A, N, lda, false, true);
        h_A_alias.randu();
        h_A_alias *= h_A_alias.t();
        magma_ssetmatrix( N, N, h_A, lda, d_R, 0, ldda, queue );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_ssyevd_gpu( jobzs[ijob], uplos[iuplo],
                          N, d_R, 0, ldda, w1,
                          h_R, lda,
                          h_work, lwork,
                          iwork, liwork,
                          &info );
        if (info != 0)
          {
          std::cerr << "magma_ssyevd_gpu returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }
        REQUIRE( info == 0 );

        if ( jobzs[ijob] != MagmaNoVec )
          {
          /* =====================================================================
             Check the results following the LAPACK's [zcds]drvst routine.
             A is factored as A = U S U^H and the following 3 tests computed:
             (1)    | A - U S U^H | / ( |A| N ) if all eigenvectors were computed
                    | U^H A U - S | / ( |A| Nfound ) otherwise
             (2)    | I - U^H U   | / ( N )
             (3)    | S(with U) - S(w/o U) | / | S |    // currently disabled, but compares to LAPACK
             =================================================================== */
          magma_sgetmatrix( N, N, d_R, 0, ldda, h_R, lda, queue );

          float *work;
          REQUIRE( magma_smalloc_cpu( &work, 2*N*N ) == MAGMA_SUCCESS );

          // e is unused since kband=0; tau is unused since itype=1
          if( Nfound == N )
            {
            lapack_test::syt21(ione, lapack_uplo_const(uplos[iuplo])[0], N, izero,
                                h_A, lda,
                                w1, runused,
                                h_R, lda,
                                h_R, lda,
                                unused, work,
                                &result[0]);
            }
          else
            {
            lapack_test::syt22(ione, lapack_uplo_const(uplos[iuplo])[0], N, Nfound, izero,
                               h_A, lda,
                               w1, runused,
                               h_R, lda,
                               h_R, lda,
                               unused, work,
                               &result[0]);
            }
          result[0] *= eps;
          result[1] *= eps;

          magma_free_cpu( work );
          work=NULL;
          }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        lapack::syevd(lapack_vec_const(jobzs[ijob])[0], lapack_uplo_const(uplos[iuplo])[0],
                      N, h_A, lda, w2,
                      h_work, lwork,
                      iwork, liwork,
                      &info);
        if (info != 0)
          {
          std::cerr << "LAPACK ssyevd returned error " << info << ": " << magma::error_as_string(info) << std::endl;
          }

        // compare eigenvalues
        float maxw=0, diff=0;
        for( int j=0; j < Nfound; j++ )
          {
          maxw = std::max(maxw, std::abs(w1[j]));
          maxw = std::max(maxw, std::abs(w2[j]));
          diff = std::max(diff, std::abs(w1[j] - w2[j]));
          }
        result[3] = diff / (N*maxw);

        REQUIRE( result[0] < tol );
        REQUIRE( result[1] < tol );
        REQUIRE( result[3] < tolulp );

        magma_free_cpu( h_A    );
        magma_free_cpu( w1     );
        magma_free_cpu( w2     );
        magma_free_cpu( iwork  );

        magma_free_pinned( h_R    );
        magma_free_pinned( h_work );

        magma_free( d_R );
        }
      }
    }

  magma_queue_destroy(queue);
  }

#endif
