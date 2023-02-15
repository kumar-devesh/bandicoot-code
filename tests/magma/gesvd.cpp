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



inline
const char*
lapack_vec_const(magma_vec_t magma_const)
  {
  REQUIRE( magma_const >= MagmaNoVec );
  REQUIRE( magma_const <= MagmaOverwriteVec );
  return get_magma2lapack_constants()[magma_const];
  }



typedef enum {
    MagmaSVD_all,
    MagmaSVD_query,
    MagmaSVD_doc,
    MagmaSVD_doc_old,
    MagmaSVD_min,
    MagmaSVD_min_1,
    MagmaSVD_min_old,
    MagmaSVD_min_old_1,
    MagmaSVD_min_fast,
    MagmaSVD_min_fast_1,
    MagmaSVD_opt,
    MagmaSVD_opt_old,
    MagmaSVD_opt_slow,
    MagmaSVD_max
} magma_svd_work_t;



// Utility function to choose size of work vector.
// This helps exercise all the different paths through gesvd().
// Note that lots of the things that get returned end up unused.
inline
void
choose_lwork(
    magma_svd_work_t svd_work,
    magma_vec_t jobu,
    magma_vec_t jobv,
    magma_int_t M,
    magma_int_t N,
    magma_int_t query_magma,
    magma_int_t query_lapack,
    magma_int_t& lwork_magma,
    magma_int_t& lwork_lapack)
  {
  lwork_magma  = -1;
  lwork_lapack = -1;
  magma_int_t lwork_doc = 0,
              lwork_min = 0, lwork_min_fast = 0,
              lwork_opt = 0, lwork_opt_slow = 0, lwork_max = 0;
  magma_int_t nb = magma_get_dgesvd_nb( M, N );
  magma_int_t mx = std::max( M, N );
  magma_int_t mn = std::min( M, N );

  // transposed (M < N) switches roles of jobu and jobv in picking path
  magma_vec_t jobu_ = (M >= N ? jobu : jobv);
  magma_vec_t jobv_ = (M >= N ? jobv : jobu);

  /* =====================================================================
     lwork formulas for dgesvd (Real)
     =================================================================== */
  // minimum per LAPACK's documentation; overridden below for Path 1
  lwork_min = std::max( 3*mn + mx, 5*mn );
  lwork_doc = std::max( 3*mn + mx, 5*mn );
  magma_int_t mnthr = (magma_int_t) (1.6 * mn);
  // int path = 0;
  if (mx >= mnthr)
    {
    // applies to Path 3-9; overridden below for Path 1, 2, 10
    lwork_opt_slow = 3*mn + std::max( 2*mn*nb, mx*nb );

    if ( jobu_ == MagmaNoVec /* jobv_ is any */ )
      {
      // path = 1;
      lwork_opt      = 3*mn + 2*mn*nb;
      lwork_opt_slow = 3*mn + 2*mn*nb;  // no slow path
      lwork_min_fast = 5*mn;            // no slow path
      lwork_min      = 5*mn;
      lwork_doc      = 5*mn;
      }
    else if ( jobu_ == MagmaOverwriteVec &&  jobv_ == MagmaNoVec )
      {
      // path = 2;
      lwork_opt      = mn*mn +          3*mn + 2*mn*nb;
      lwork_max      = mn*mn + std::max(3*mn + 2*mn*nb, mn + mx*mn);
      lwork_opt_slow = 3*mn + (mx + mn)*nb;
      lwork_min_fast = mn*mn + 5*mn;
      }
    else if ( jobu_ == MagmaOverwriteVec && (jobv_ == MagmaAllVec || jobv_ == MagmaSomeVec) )
      {
      // path = 3;
      lwork_opt      = mn*mn +          3*mn + 2*mn*nb;
      lwork_max      = mn*mn + std::max(3*mn + 2*mn*nb, mn + mx*mn);
      lwork_min_fast = mn*mn + 5*mn;
      }
    else if ( jobu_ == MagmaSomeVec      &&  jobv_ == MagmaNoVec )
      {
      // path = 4;
      lwork_opt      = mn*mn + 3*mn + 2*mn*nb;
      lwork_min_fast = mn*mn + 5*mn;
      }
    else if ( jobu_ == MagmaSomeVec      &&  jobv_ == MagmaOverwriteVec   )
      {
      // path = 5;
      lwork_opt      = 2*mn*mn + 3*mn + 2*mn*nb;
      lwork_min_fast = 2*mn*mn + 5*mn;
      }
    else if ( jobu_ == MagmaSomeVec      && (jobv_ == MagmaAllVec || jobv_ == MagmaSomeVec) )
      {
      // path = 6;
      lwork_opt      = mn*mn + 3*mn + 2*mn*nb;
      lwork_min_fast = mn*mn + 5*mn;
      }
    else if ( jobu_ == MagmaAllVec       &&  jobv_ == MagmaNoVec )
      {
      // path = 7;
      lwork_opt      = mn*mn + std::max(3*mn + 2*mn*nb, mn + mx*nb);
      lwork_min_fast = mn*mn + std::max(5*mn, mn + mx);
      }
    else if ( jobu_ == MagmaAllVec       &&  jobv_ == MagmaOverwriteVec )
      {
      // path = 8;
      lwork_opt      = 2*mn*mn + std::max(3*mn + 2*mn*nb, mn + mx*nb);
      lwork_min_fast = 2*mn*mn + std::max(5*mn, mn + mx);
      }
    else if ( jobu_ == MagmaAllVec       && (jobv_ == MagmaAllVec || jobv_ == MagmaSomeVec) )
      {
      // path = 9;
      lwork_opt      = mn*mn + std::max(3*mn + 2*mn*nb, mn + mx*nb);
      lwork_min_fast = mn*mn + std::max(5*mn, mn + mx);
      }
    }
  else
    {
    // mx >= mn
    // path = 10;
    lwork_opt      =      3*mn + (mx + mn)*nb;
    lwork_opt_slow =      3*mn + (mx + mn)*nb;   // no slow path
    lwork_min_fast = std::max(3*mn + mx, 5*mn);  // no slow path
    }


  /* =====================================================================
     Select between min, optimal, etc. lwork size
     =================================================================== */
  lwork_magma = lwork_opt;  // MAGMA requires optimal; overridden below by query, min, min-1, max
  switch( svd_work )
    {
    case MagmaSVD_query:
        lwork_lapack = query_lapack;
        lwork_magma  = query_magma;
        break;

    case MagmaSVD_min:
    case MagmaSVD_min_1:
        lwork_lapack = lwork_min;
        // MAGMA requires optimal; use opt slow path if smaller
        if ( lwork_opt_slow && lwork_opt_slow < lwork_opt )
          {
          lwork_magma = lwork_opt_slow;
          }
        if ( svd_work == MagmaSVD_min_1 )
          {
          lwork_lapack -= 1;
          lwork_magma  -= 1;
          }
        break;

    case MagmaSVD_opt_slow:
        lwork_lapack = (lwork_opt_slow ? lwork_opt_slow : lwork_opt);
        lwork_magma  = (lwork_opt_slow ? lwork_opt_slow : lwork_opt);
        break;

    case MagmaSVD_min_fast:
    case MagmaSVD_min_fast_1:
        lwork_lapack = (lwork_min_fast ? lwork_min_fast : lwork_min);
        if ( svd_work == MagmaSVD_min_fast_1 )
          {
          lwork_lapack -= 1;
          lwork_magma  -= 1;
          }
        break;

    case MagmaSVD_opt:
        lwork_lapack = lwork_opt;
        break;

    case MagmaSVD_max:
        lwork_lapack = (lwork_max ? lwork_max : lwork_opt);
        lwork_magma  = (lwork_max ? lwork_max : lwork_opt);
        break;

    case MagmaSVD_doc:
        lwork_lapack = lwork_doc;
        break;

    default:
        std::ostringstream oss;
        oss << "unsupported svd-work " << svd_work;
        throw std::invalid_argument(oss.str());
    }
  }



// Check the results following the LAPACK's [zcds]drvbd routine.
// A is factored as A = U diag(S) VT and the following 4 tests computed:
// (1)    | A - U diag(S) VT | / ( |A| max(m,n) )
// (2)    | I - U^H U   | / ( m )
// (3)    | I - VT VT^H | / ( n )
// (4)    S contains min(m,n) nonnegative values in decreasing order.
//        (Return 0 if true, 1 if false.)
//
// If check is false, skips (1) - (3), but always does (4).
inline
void
check_dgesvd(magma_vec_t jobu,
             magma_vec_t jobv,
             magma_int_t m, magma_int_t n,
             double *A,  magma_int_t lda,
             double *S,
             double *U,  magma_int_t ldu,
             double *VT, magma_int_t ldv,
             double result[4])
  {
  double unused[1];
  const magma_int_t izero = 0;
  double eps = std::numeric_limits<double>::epsilon();

  if (jobu == MagmaNoVec)
    U = NULL;
  if (jobv == MagmaNoVec)
    VT = NULL;

  // -1 indicates check not done
  result[0] = -1;
  result[1] = -1;
  result[2] = -1;
  result[3] = -1;

  magma_int_t min_mn = std::min(m, n);
  magma_int_t n_u  = (jobu == MagmaAllVec ? m : min_mn);
  magma_int_t m_vt = (jobv == MagmaAllVec ? n : min_mn);

  // dbdt01 needs m+n
  // dort01 prefers n*(n+1) to check U; m*(m+1) to check V
  magma_int_t lwork_err = m+n;
  if (U != NULL)
    lwork_err = std::max( lwork_err, n_u*(n_u+1) );
  if (VT != NULL)
    lwork_err = std::max( lwork_err, m_vt*(m_vt+1) );
  double *work_err;
  REQUIRE( magma_dmalloc_cpu( &work_err, lwork_err ) == MAGMA_SUCCESS );

  // dbdt01 and dort01 need max(m,n), depending
  double *rwork_err;
  REQUIRE( magma_dmalloc_cpu( &rwork_err, std::max(m,n) ) == MAGMA_SUCCESS );

  if (U != NULL && VT != NULL)
    {
    // since KD=0 (3rd arg), E is not referenced so pass unused (9th arg)
    coot_fortran(coot_dbdt01)( &m, &n, &izero, A, &lda,
                               U, &ldu, S, unused, VT, &ldv,
                               work_err,
                               &result[0] );
    }
  if ( U != NULL )
    {
    coot_fortran(coot_dort01)( "C", &m,  &n_u, U,  &ldu, work_err, &lwork_err, &result[1] );
    }
  if ( VT != NULL )
    {
    coot_fortran(coot_dort01)( "R", &m_vt, &n, VT, &ldv, work_err, &lwork_err, &result[2] );
    }

  result[0] *= eps;
  result[1] *= eps;
  result[2] *= eps;

  magma_free_cpu( work_err );
  magma_free_cpu( rwork_err );

  // check S is sorted
  result[3] = 0.;
  for (int j = 0; j < min_mn-1; j++)
    {
    if ( S[j] < S[j+1] )
      result[3] = 1.;
    if ( S[j] < 0. )
      result[3] = 1.;
    }

  if (min_mn > 1 && S[min_mn-1] < 0.)
    {
    result[3] = 1.;
    }
  }



TEST_CASE("magma_dgesvd_1", "[gesvd]")
  {
  if (get_rt().backend != CL_BACKEND)
    {
    return;
    }

  // Constants
  magma_int_t ione     = 1;
  magma_int_t ineg_one = -1;
  const double d_neg_one = -1;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Local variables
  double *hA, *hR, *U, *Umalloc, *VT, *VTmalloc, *hwork;
  double dummy[1], unused[1];
  double *S, *Sref, work[1], runused[1];
  magma_int_t M, N, N_U, M_VT, lda, ldu, ldv, min_mn, info;

  double tol = 30 * std::numeric_limits<double>::epsilon();

  std::vector< magma_svd_work_t > svd_works;
  svd_works.push_back(MagmaSVD_min     );
  svd_works.push_back(MagmaSVD_doc     );
  svd_works.push_back(MagmaSVD_opt_slow);
  svd_works.push_back(MagmaSVD_min_fast);
  svd_works.push_back(MagmaSVD_opt     );
  svd_works.push_back(MagmaSVD_max     );
  svd_works.push_back(MagmaSVD_query   );

  std::vector< magma_vec_t > jobs;
  jobs.push_back(MagmaNoVec       );
  jobs.push_back(MagmaSomeVec     );
  jobs.push_back(MagmaOverwriteVec);
  jobs.push_back(MagmaAllVec      );

  for (int itest = 0; itest < 5; ++itest)
    {
    for (size_t ijobu = 0; ijobu < jobs.size(); ++ijobu)
      {
      for (size_t ijobv = 0; ijobv < jobs.size(); ++ijobv)
        {
        magma_vec_t jobu = jobs[ijobu];
        magma_vec_t jobv = jobs[ijobv];

        // Skip invalid combination.
        if (jobu == MagmaOverwriteVec && jobv == MagmaOverwriteVec )
          {
          continue;
          }

        for (size_t isvd_work = 0; isvd_work < svd_works.size(); ++isvd_work)
          {
          magma_svd_work_t svd_work = svd_works[isvd_work];

          M = 256 * (itest + 1) + 65; // 65 is to work around strange clBLAS error with nvidia driver
          N = 256 * (itest + 1) + 65;
          min_mn = std::min(M, N);
          N_U  = (jobu == MagmaAllVec ? M : min_mn);
          M_VT = (jobv == MagmaAllVec ? N : min_mn);
          lda = M;
          ldu = M;
          ldv = M_VT;

          /* =====================================================================
             query for workspace size
             =================================================================== */
          magma_int_t query_magma, query_lapack;
          magma_dgesvd( jobu, jobv, M, N,
                        NULL, lda, NULL, NULL, ldu, NULL, ldv, dummy, ineg_one,
                        &info );
          REQUIRE( info == 0 );
          query_magma = (magma_int_t) dummy[0];

          coot_fortran(coot_dgesvd)( lapack_vec_const(jobu), lapack_vec_const(jobv), &M, &N,
                                     unused, &lda, runused,
                                     unused, &ldu,
                                     unused, &ldv,
                                     dummy, &ineg_one,
                                     &info );
          REQUIRE( info == 0 );
          query_lapack = (magma_int_t) dummy[0];

          // Choose lwork size based on --svd-work option.
          // We recommend using the above query for lwork rather than
          // the formulas; we use formulas to verify the code in all cases.
          // lwork_formula_t is a special class, just for the tester, that
          // saves the lwork value together with a string describing its formula.
          magma_int_t lwork_magma, lwork_lapack;
          choose_lwork( svd_work, jobu, jobv, M, N, query_magma, query_lapack,
                        lwork_magma, lwork_lapack );

          // LAPACK and MAGMA may return different sizes;
          // since we call both, allocate max.
          magma_int_t lwork = std::max( lwork_magma, lwork_lapack );

          /* =====================================================================
             Allocate memory
             =================================================================== */
          REQUIRE( magma_dmalloc_cpu( &hA,    lda*N  ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_cpu( &S,     min_mn ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_cpu( &Sref,  min_mn ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_pinned( &hR,    lda*N ) == MAGMA_SUCCESS );
          REQUIRE( magma_dmalloc_pinned( &hwork, lwork ) == MAGMA_SUCCESS );

          // U and VT either overwrite hR, or are allocated as Umalloc, VTmalloc
          if (jobu == MagmaOverwriteVec)
            {
            U   = hR;
            ldu = lda;
            Umalloc = NULL;
            }
          else
            {
            REQUIRE( magma_dmalloc_cpu( &Umalloc, ldu*N_U ) == MAGMA_SUCCESS ); // M x M (jobz=A) or M x min(M,N)
            U = Umalloc;
            }

          if (jobv == MagmaOverwriteVec)
            {
            VT  = hR;
            ldv = lda;
            VTmalloc = NULL;
            }
          else
            {
            REQUIRE( magma_dmalloc_cpu( &VTmalloc, ldv*N ) == MAGMA_SUCCESS ); // N x N (jobz=A) or min(M,N) x N
            VT = VTmalloc;
            }

          // force check to fail if gesdd returns info error
          double result[5]        = { nan, nan, nan, nan, nan };
          double result_lapack[5] = { nan, nan, nan, nan, nan };

          /* Initialize the matrix (random uniform) */
          arma::Mat<double> hA_alias(hA, lda, N, false, true);
          hA_alias.randu();
          coot_fortran(coot_dlacpy)( MagmaFullStr, &M, &N, hA, &lda, hR, &lda );

          magma_dgesvd( jobu, jobv, M, N,
                        hR, lda, S, U, ldu, VT, ldv, hwork, lwork_magma,
                        &info );

          if ( svd_work == MagmaSVD_min_1 )
            {
            if (info != -13)
              {
              std::cerr << "magma_dgesvd returned error code " << info << " (expected -13): " << magma::error_as_string(info) << std::endl;
              }
            REQUIRE( info == -13 );
            }
          else
            {
            if (info != 0)
              {
              std::cerr << "magma_dgesvd returned error " << info << ": " << magma::error_as_string(info) << std::endl;
              }
            REQUIRE( info == 0 );
            }

          check_dgesvd( jobu, jobv, M, N, hA, lda, S, U, ldu, VT, ldv, result );

          coot_fortran(coot_dlacpy)( MagmaFullStr, &M, &N, hA, &lda, hR, &lda );
          coot_fortran(coot_dgesvd)( lapack_vec_const(jobu), lapack_vec_const(jobv), &M, &N,
                                     hR, &lda, Sref, U, &ldu, VT, &ldv, hwork, &lwork_lapack,
                                     &info);
          if ( svd_work == MagmaSVD_min_1 )
            {
            if (info != -13)
              {
              std::cerr << "dgesvd returned error code " << info << " (expected -13): " << magma::error_as_string(info) << std::endl;
              }
            REQUIRE( info == -13 );
            }
          else
            {
            if (info != 0)
              {
              std::cerr << "dgesvd returned error " << info << ": " << magma::error_as_string(info) << std::endl;
              }
            REQUIRE( info == 0 );
            }

          check_dgesvd( jobu, jobv, M, N, hA, lda, Sref, U, ldu, VT, ldv, result_lapack );

          coot_fortran(coot_daxpy)( &min_mn, &d_neg_one, S, &ione, Sref, &ione );
          result[4]  = coot_fortran(coot_dlange)( "F", &min_mn, &ione, Sref, &min_mn, work );
          result[4] /= coot_fortran(coot_dlange)( "F", &min_mn, &ione, S,    &min_mn, work );

          // Some of the tests may not have been done depending on jobu and
          // jobv; that is indicated with `-1`.
          REQUIRE( result[0] < tol );
          REQUIRE( result[1] < tol );
          REQUIRE( result[2] < tol );
          REQUIRE( result[3] < tol );
          REQUIRE( result[4] < tol );

          REQUIRE( result_lapack[0] < tol );
          REQUIRE( result_lapack[1] < tol );
          REQUIRE( result_lapack[2] < tol );
          REQUIRE( result_lapack[3] < tol );

          magma_free_cpu( hA );
          magma_free_cpu( S  );
          magma_free_cpu( Sref );
          magma_free_pinned( hR    );
          magma_free_pinned( hwork );
          magma_free_cpu( VTmalloc );
          magma_free_cpu( Umalloc  );
          }
        }
      }
    }
  }

#endif
