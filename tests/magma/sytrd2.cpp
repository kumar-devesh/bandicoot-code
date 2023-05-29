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

TEST_CASE("magma_dsytrd2_1", "[sytrd2]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  double           eps;
  double *h_A, *h_R, *h_Q, *h_work, *work;
  magmaDouble_ptr d_R, dwork;
  double *tau;
  double          *diag, *offdiag;
  double           result[2] = {0., 0.};
  magma_int_t N, lda, ldda, lwork, info, nb, ldwork;
  magma_int_t ione     = 1;
  magma_int_t itwo     = 2;
  magma_int_t ithree   = 3;
  eps = std::numeric_limits<double>::epsilon();

  double tol = 30 * eps;

  magma_queue_t queue = magma_queue_create();

  const magma_uplo_t uplos[2] = { MagmaUpper, MagmaLower };

  for( int itest = 0; itest < 7; ++itest )
    {
    for (int iuplo = 0; iuplo < 2; ++iuplo )
      {
      N = 128 * (itest + 1) + 64;

      lda    = N;
      ldda   = magma_roundup( N, 32 );  // multiple of 32 by default
      nb     = magma_get_dsytrd_nb(N);
      lwork  = N*nb;  /* We suppose the magma nb is bigger than lapack nb */
      ldwork = ldda*magma_ceildiv(N,64) + 2*ldda*nb;

      REQUIRE( magma_dmalloc_cpu( &h_A,     lda*N ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_cpu( &tau,     N     ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_cpu( &diag,    N   ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_cpu( &offdiag, N-1 ) == MAGMA_SUCCESS );

      REQUIRE( magma_dmalloc_pinned( &h_R,     lda*N ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_pinned( &h_work,  lwork ) == MAGMA_SUCCESS );

      REQUIRE( magma_dmalloc( &d_R,     ldda*N ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc( &dwork,   ldwork ) == MAGMA_SUCCESS );

      /* ====================================================================
         Initialize the matrix
         =================================================================== */
      arma::mat h_A_alias((double*) h_A, N, N, false, true);
      h_A_alias.randu();
      magma_dsetmatrix( N, N, h_A, lda, d_R, 0, ldda, queue );

      /* ====================================================================
         Performs operation using MAGMA
         =================================================================== */
      magma_dsytrd2_gpu( uplos[iuplo], N, d_R, 0, ldda, diag, offdiag,
                         tau, h_R, lda, h_work, lwork, dwork, 0, ldwork, &info );
      if (info != 0)
        {
        std::cerr << "magma_dsytrd2_gpu returned error " << info << ": " << magma::error_as_string(info) << std::endl;
        }
      REQUIRE( info == 0 );

      REQUIRE( magma_dmalloc_cpu( &h_Q,  lda*N ) == MAGMA_SUCCESS );
      REQUIRE( magma_dmalloc_cpu( &work, 2*N*N ) == MAGMA_SUCCESS );

      magma_dgetmatrix( N, N, d_R, 0, ldda, h_R, lda, queue );
      magma_dgetmatrix( N, N, d_R, 0, ldda, h_Q, lda, queue );
      coot_fortran(coot_dorgtr)( lapack_uplo_const(uplos[iuplo]), &N, h_Q, &lda, tau, h_work, &lwork, &info );

      coot_fortran(coot_dsyt21)( &itwo, lapack_uplo_const(uplos[iuplo]), &N, &ione,
                                 h_A, &lda, diag, offdiag,
                                 h_Q, &lda, h_R, &lda,
                                 tau, work,
                                 &result[0] );

      coot_fortran(coot_dsyt21)( &ithree, lapack_uplo_const(uplos[iuplo]), &N, &ione,
                                 h_A, &lda, diag, offdiag,
                                 h_Q, &lda, h_R, &lda,
                                 tau, work,
                                 &result[1] );
      result[0] *= eps;
      result[1] *= eps;

      REQUIRE( result[0] < tol );
      REQUIRE( result[1] < tol );

      magma_free_cpu( h_Q  );
      magma_free_cpu( work );

      magma_free_cpu( h_A     );
      magma_free_cpu( tau     );
      magma_free_cpu( diag    );
      magma_free_cpu( offdiag );

      magma_free_pinned( h_R    );
      magma_free_pinned( h_work );

      magma_free( d_R   );
      magma_free( dwork );
      }
    }

  magma_queue_destroy(queue);
  }



TEST_CASE("magma_ssytrd2_1", "[sytrd2]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  float           eps;
  float *h_A, *h_R, *h_Q, *h_work, *work;
  magmaFloat_ptr d_R, dwork;
  float *tau;
  float          *diag, *offdiag;
  float           result[2] = {0., 0.};
  magma_int_t N, lda, ldda, lwork, info, nb, ldwork;
  magma_int_t ione     = 1;
  magma_int_t itwo     = 2;
  magma_int_t ithree   = 3;
  eps = std::numeric_limits<float>::epsilon();

  float tol = 30 * eps;

  magma_queue_t queue = magma_queue_create();

  const magma_uplo_t uplos[2] = { MagmaUpper, MagmaLower };

  for( int itest = 0; itest < 7; ++itest )
    {
    for (int iuplo = 0; iuplo < 2; ++iuplo )
      {
      N = 128 * (itest + 1) + 64;

      lda    = N;
      ldda   = magma_roundup( N, 32 );  // multiple of 32 by default
      nb     = magma_get_ssytrd_nb(N);
      lwork  = N*nb;  /* We suppose the magma nb is bigger than lapack nb */
      ldwork = ldda*magma_ceildiv(N,64) + 2*ldda*nb;

      REQUIRE( magma_smalloc_cpu( &h_A,     lda*N ) == MAGMA_SUCCESS );
      REQUIRE( magma_smalloc_cpu( &tau,     N     ) == MAGMA_SUCCESS );
      REQUIRE( magma_smalloc_cpu( &diag,    N   ) == MAGMA_SUCCESS );
      REQUIRE( magma_smalloc_cpu( &offdiag, N-1 ) == MAGMA_SUCCESS );

      REQUIRE( magma_smalloc_pinned( &h_R,     lda*N ) == MAGMA_SUCCESS );
      REQUIRE( magma_smalloc_pinned( &h_work,  lwork ) == MAGMA_SUCCESS );

      REQUIRE( magma_smalloc( &d_R,     ldda*N ) == MAGMA_SUCCESS );
      REQUIRE( magma_smalloc( &dwork,   ldwork ) == MAGMA_SUCCESS );

      /* ====================================================================
         Initialize the matrix
         =================================================================== */
      arma::fmat h_A_alias((float*) h_A, N, N, false, true);
      h_A_alias.randu();
      magma_ssetmatrix( N, N, h_A, lda, d_R, 0, ldda, queue );

      /* ====================================================================
         Performs operation using MAGMA
         =================================================================== */
      magma_ssytrd2_gpu( uplos[iuplo], N, d_R, 0, ldda, diag, offdiag,
                         tau, h_R, lda, h_work, lwork, dwork, 0, ldwork, &info );
      if (info != 0)
        {
        std::cerr << "magma_ssytrd2_gpu returned error " << info << ": " << magma::error_as_string(info) << std::endl;
        }
      REQUIRE( info == 0 );

      REQUIRE( magma_smalloc_cpu( &h_Q,  lda*N ) == MAGMA_SUCCESS );
      REQUIRE( magma_smalloc_cpu( &work, 2*N*N ) == MAGMA_SUCCESS );

      magma_sgetmatrix( N, N, d_R, 0, ldda, h_R, lda, queue );
      magma_sgetmatrix( N, N, d_R, 0, ldda, h_Q, lda, queue );
      coot_fortran(coot_sorgtr)( lapack_uplo_const(uplos[iuplo]), &N, h_Q, &lda, tau, h_work, &lwork, &info );

      coot_fortran(coot_ssyt21)( &itwo, lapack_uplo_const(uplos[iuplo]), &N, &ione,
                                 h_A, &lda, diag, offdiag,
                                 h_Q, &lda, h_R, &lda,
                                 tau, work,
                                 &result[0] );

      coot_fortran(coot_ssyt21)( &ithree, lapack_uplo_const(uplos[iuplo]), &N, &ione,
                                 h_A, &lda, diag, offdiag,
                                 h_Q, &lda, h_R, &lda,
                                 tau, work,
                                 &result[1] );
      result[0] *= eps;
      result[1] *= eps;

      REQUIRE( result[0] < tol );
      REQUIRE( result[1] < tol );

      magma_free_cpu( h_Q  );
      magma_free_cpu( work );

      magma_free_cpu( h_A     );
      magma_free_cpu( tau     );
      magma_free_cpu( diag    );
      magma_free_cpu( offdiag );

      magma_free_pinned( h_R    );
      magma_free_pinned( h_work );

      magma_free( d_R   );
      magma_free( dwork );
      }
    }

  magma_queue_destroy(queue);
  }



#endif
