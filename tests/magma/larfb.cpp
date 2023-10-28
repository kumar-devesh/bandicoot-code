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



TEST_CASE("magma_dlarfb_1", "[larfb]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  if (!coot_rt_t::is_supported_type<double>())
    {
    return;
    }

  const double c_zero    = MAGMA_D_ZERO;
  const double c_one     = MAGMA_D_ONE;
  const double c_neg_one = MAGMA_D_NEG_ONE;
  const magma_int_t ione = 1;

  magma_int_t M, N, K, ldc, ldv, ldt, ldw, ldw2, nv;
  double Cnorm, error, work[1];

  // test all combinations of input parameters
  magma_side_t   side  [] = { MagmaLeft,       MagmaRight    };
  magma_trans_t  trans [] = { MagmaTrans, MagmaNoTrans  };
  magma_direct_t direct[] = { MagmaForward,    MagmaBackward };
  magma_storev_t storev[] = { MagmaColumnwise, MagmaRowwise  };

  double tol = 30 * std::numeric_limits<double>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    itest += 1;
    M = 128 * (itest + 1) + 65; // If we use 64, crashes are observed with the nvidia OpenCL drivers.
    N = 128 * (itest + 1) + 65; // That's probably an artifact of the poor nvidia OpenCL support, not
    K = 128 * (itest + 1) + 65; // a bug in clBLAS.

    for (int iside = 0; iside < 2; ++iside)
      {
      for (int itran = 0; itran < 2; ++itran)
        {
        for (int idir  = 0; idir  < 2; ++idir )
          {
          for (int istor = 0; istor < 2; ++istor)
            {
            ldc = magma_roundup( M, 32 );  // multiple of 32 by default
            ldt = magma_roundup( K, 32 );  // multiple of 32 by default
            ldw = (side[iside] == MagmaLeft ? N : M);
            ldw2 = std::min( M, N );
            // (ldv, nv) get swapped later if rowwise
            ldv = (side[iside] == MagmaLeft ? M : N);
            nv  = K;

            // Allocate memory for matrices
            double *C, *R, *V, *T, *W;
            REQUIRE( magma_dmalloc_cpu( &C, ldc*N ) == MAGMA_SUCCESS );
            REQUIRE( magma_dmalloc_cpu( &R, ldc*N ) == MAGMA_SUCCESS );
            REQUIRE( magma_dmalloc_cpu( &V, ldv*K ) == MAGMA_SUCCESS );
            REQUIRE( magma_dmalloc_cpu( &T, ldt*K ) == MAGMA_SUCCESS );
            REQUIRE( magma_dmalloc_cpu( &W, ldw*K ) == MAGMA_SUCCESS );

            magmaDouble_ptr dC, dV, dT, dW, dW2;
            REQUIRE( magma_dmalloc( &dC,  ldc*N  ) == MAGMA_SUCCESS );
            REQUIRE( magma_dmalloc( &dV,  ldv*K  ) == MAGMA_SUCCESS );
            REQUIRE( magma_dmalloc( &dT,  ldt*K  ) == MAGMA_SUCCESS );
            REQUIRE( magma_dmalloc( &dW,  ldw*K  ) == MAGMA_SUCCESS );
            REQUIRE( magma_dmalloc( &dW2, ldw2*K ) == MAGMA_SUCCESS );

            // C is M x N.
            // Fill with random values.
            arma::Mat<double> C_alias(C, ldc, N, false, true);
            C_alias.randu();

            // V is ldv x nv. See larfb docs for description.
            // if column-wise and left,  M x K
            // if column-wise and right, N x K
            // if row-wise and left,     K x M
            // if row-wise and right,    K x N
            arma::Mat<double> V_alias(V, ldv, nv, false, true);
            V_alias.randu();

            if ( storev[istor] == MagmaColumnwise )
              {
              if ( direct[idir] == MagmaForward )
                {
                lapack::laset('U', K, K, c_zero, c_one, V, ldv);
                }
              else
                {
                lapack::laset('L', K, K, c_zero, c_one, &V[(ldv-K)], ldv);
                }
              }
            else
              {
              // rowwise, swap V's dimensions
              std::swap( ldv, nv );
              if ( direct[idir] == MagmaForward )
                {
                lapack::laset('L', K, K, c_zero, c_one, V, ldv);
                }
              else
                {
                lapack::laset('U', K, K, c_zero, c_one, &V[(nv-K)*ldv], ldv);
                }
              }

            // T is K x K, upper triangular for forward, and lower triangular for backward
            magma_int_t k1 = K-1;
            arma::Mat<double> T_alias(T, ldt, K, false, true);
            T_alias.randu();
            if ( direct[idir] == MagmaForward )
              {
              lapack::laset('L', k1, k1, c_zero, c_zero, &T[1], ldt);
              }
            else
              {
              lapack::laset('U', k1, k1, c_zero, c_zero, &T[1*ldt], ldt);
              }

            magma_dsetmatrix( M,   N,  C, ldc, dC, 0, ldc, queue );
            magma_dsetmatrix( ldv, nv, V, ldv, dV, 0, ldv, queue );
            magma_dsetmatrix( K,   K,  T, ldt, dT, 0, ldt, queue );

            lapack::larfb(lapack_side_const(side[iside])[0], lapack_trans_const(trans[itran])[0],
                          lapack_direct_const(direct[idir])[0], lapack_storev_const(storev[istor])[0],
                          M, N, K,
                          V, ldv, T, ldt, C, ldc, W, ldw);

            magma_dlarfb_gpu( side[iside], trans[itran], direct[idir], storev[istor],
                              M, N, K,
                              dV, 0, ldv, dT, 0, ldt, dC, 0, ldc, dW, 0, ldw, queue );
            // dC must be ldc*N at least
            magma_dgetmatrix( M, N, dC, 0, ldc, R, ldc, queue );

            // compute relative error |HC_magma - HC_lapack| / |HC_lapack|
            magma_int_t size = ldc*N;
            blas::axpy(size, c_neg_one, C, ione, R, ione);
            Cnorm = lapack::lange('F', M, N, C, ldc, work);
            error = lapack::lange('F', M, N, R, ldc, work) / Cnorm;

            REQUIRE( error < tol );

            magma_free_cpu( C );
            magma_free_cpu( R );
            magma_free_cpu( V );
            magma_free_cpu( T );
            magma_free_cpu( W );

            magma_free( dC  );
            magma_free( dV  );
            magma_free( dT  );
            magma_free( dW  );
            magma_free( dW2 );
            }
          }
        }
      }
    }

  magma_queue_destroy( queue );
  }



TEST_CASE("magma_slarfb_1", "[larfb]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  const float c_zero    = MAGMA_S_ZERO;
  const float c_one     = MAGMA_S_ONE;
  const float c_neg_one = MAGMA_S_NEG_ONE;
  const magma_int_t ione = 1;

  magma_int_t M, N, K, ldc, ldv, ldt, ldw, ldw2, nv;
  float Cnorm, error, work[1];

  // test all combinations of input parameters
  magma_side_t   side  [] = { MagmaLeft,       MagmaRight    };
  magma_trans_t  trans [] = { MagmaTrans, MagmaNoTrans  };
  magma_direct_t direct[] = { MagmaForward,    MagmaBackward };
  magma_storev_t storev[] = { MagmaColumnwise, MagmaRowwise  };

  float tol = 30 * std::numeric_limits<float>::epsilon();

  magma_queue_t queue = magma_queue_create();

  for (int itest = 0; itest < 10; ++itest)
    {
    itest += 1;
    M = 128 * (itest + 1) + 65; // If we use 64, crashes are observed with the nvidia OpenCL drivers.
    N = 128 * (itest + 1) + 65; // That's probably an artifact of the poor nvidia OpenCL support, not
    K = 128 * (itest + 1) + 65; // a bug in clBLAS.

    for (int iside = 0; iside < 2; ++iside)
      {
      for (int itran = 0; itran < 2; ++itran)
        {
        for (int idir  = 0; idir  < 2; ++idir )
          {
          for (int istor = 0; istor < 2; ++istor)
            {
            ldc = magma_roundup( M, 32 );  // multiple of 32 by default
            ldt = magma_roundup( K, 32 );  // multiple of 32 by default
            ldw = (side[iside] == MagmaLeft ? N : M);
            ldw2 = std::min( M, N );
            // (ldv, nv) get swapped later if rowwise
            ldv = (side[iside] == MagmaLeft ? M : N);
            nv  = K;

            // Allocate memory for matrices
            float *C, *R, *V, *T, *W;
            REQUIRE( magma_smalloc_cpu( &C, ldc*N ) == MAGMA_SUCCESS );
            REQUIRE( magma_smalloc_cpu( &R, ldc*N ) == MAGMA_SUCCESS );
            REQUIRE( magma_smalloc_cpu( &V, ldv*K ) == MAGMA_SUCCESS );
            REQUIRE( magma_smalloc_cpu( &T, ldt*K ) == MAGMA_SUCCESS );
            REQUIRE( magma_smalloc_cpu( &W, ldw*K ) == MAGMA_SUCCESS );

            magmaFloat_ptr dC, dV, dT, dW, dW2;
            REQUIRE( magma_smalloc( &dC,  ldc*N  ) == MAGMA_SUCCESS );
            REQUIRE( magma_smalloc( &dV,  ldv*K  ) == MAGMA_SUCCESS );
            REQUIRE( magma_smalloc( &dT,  ldt*K  ) == MAGMA_SUCCESS );
            REQUIRE( magma_smalloc( &dW,  ldw*K  ) == MAGMA_SUCCESS );
            REQUIRE( magma_smalloc( &dW2, ldw2*K ) == MAGMA_SUCCESS );

            // C is M x N.
            // Fill with random values.
            arma::Mat<float> C_alias(C, ldc, N, false, true);
            C_alias.randu();

            // V is ldv x nv. See larfb docs for description.
            // if column-wise and left,  M x K
            // if column-wise and right, N x K
            // if row-wise and left,     K x M
            // if row-wise and right,    K x N
            arma::Mat<float> V_alias(V, ldv, nv, false, true);
            V_alias.randu();

            if ( storev[istor] == MagmaColumnwise )
              {
              if ( direct[idir] == MagmaForward )
                {
                lapack::laset('U', K, K, c_zero, c_one, V, ldv);
                }
              else
                {
                lapack::laset('L', K, K, c_zero, c_one, &V[(ldv-K)], ldv);
                }
              }
            else
              {
              // rowwise, swap V's dimensions
              std::swap( ldv, nv );
              if ( direct[idir] == MagmaForward )
                {
                lapack::laset('L', K, K, c_zero, c_one, V, ldv);
                }
              else
                {
                lapack::laset('U', K, K, c_zero, c_one, &V[(nv-K)*ldv], ldv);
                }
              }

            // T is K x K, upper triangular for forward, and lower triangular for backward
            magma_int_t k1 = K-1;
            arma::Mat<float> T_alias(T, ldt, K, false, true);
            T_alias.randu();
            if ( direct[idir] == MagmaForward )
              {
              lapack::laset('L', k1, k1, c_zero, c_zero, &T[1], ldt);
              }
            else
              {
              lapack::laset('U', k1, k1, c_zero, c_zero, &T[1*ldt], ldt);
              }

            magma_ssetmatrix( M,   N,  C, ldc, dC, 0, ldc, queue );
            magma_ssetmatrix( ldv, nv, V, ldv, dV, 0, ldv, queue );
            magma_ssetmatrix( K,   K,  T, ldt, dT, 0, ldt, queue );

            lapack::larfb(lapack_side_const(side[iside])[0], lapack_trans_const(trans[itran])[0],
                          lapack_direct_const(direct[idir])[0], lapack_storev_const(storev[istor])[0],
                          M, N, K,
                          V, ldv, T, ldt, C, ldc, W, ldw);

            magma_slarfb_gpu( side[iside], trans[itran], direct[idir], storev[istor],
                              M, N, K,
                              dV, 0, ldv, dT, 0, ldt, dC, 0, ldc, dW, 0, ldw, queue );
            // dC must be ldc*N at least
            magma_sgetmatrix( M, N, dC, 0, ldc, R, ldc, queue );

            // compute relative error |HC_magma - HC_lapack| / |HC_lapack|
            magma_int_t size = ldc*N;
            blas::axpy(size, c_neg_one, C, ione, R, ione);
            Cnorm = lapack::lange('F', M, N, C, ldc, work);
            error = lapack::lange('F', M, N, R, ldc, work) / Cnorm;

            REQUIRE( error < tol );

            magma_free_cpu( C );
            magma_free_cpu( R );
            magma_free_cpu( V );
            magma_free_cpu( T );
            magma_free_cpu( W );

            magma_free( dC  );
            magma_free( dV  );
            magma_free( dT  );
            magma_free( dW  );
            magma_free( dW2 );
            }
          }
        }
      }
    }

  magma_queue_destroy( queue );
  }

#endif
