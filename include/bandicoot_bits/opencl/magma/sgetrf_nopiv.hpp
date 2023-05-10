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



// Purpose
// -------
// SGETRF_NOPIV computes an LU factorization of a general M-by-N
// matrix A without pivoting.
//
// The factorization has the form
//    A = L * U
// where L is lower triangular with unit diagonal elements (lower
// trapezoidal if m > n), and U is upper triangular (upper
// trapezoidal if m < n).
//
// This is the right-looking Level 3 BLAS version of the algorithm.
//
// This is a CPU-only (not accelerated) version.

inline
magma_int_t
magma_sgetrf_nopiv
  (
  magma_int_t m, magma_int_t n,
  float *A, magma_int_t lda,
  magma_int_t *info
  )
  {
  float c_one = MAGMA_S_ONE;
  float c_neg_one = MAGMA_S_NEG_ONE;

  magma_int_t min_mn, m_j_jb, n_j_jb;
  magma_int_t j, jb, nb, iinfo;

  A -= 1 + lda;

  /* Function Body */
  *info = 0;
  if (m < 0)
    {
    *info = -1;
    }
  else if (n < 0)
    {
    *info = -2;
    }
  else if (lda < std::max(1,m))
    {
    *info = -4;
    }

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }

  /* Quick return if possible */
  if (m == 0 || n == 0)
    {
    return *info;
    }

  /* Determine the block size for this environment. */
  nb = 128;
  min_mn = std::min(m,n);
  if (nb <= 1 || nb >= min_mn)
    {
    /* Use unblocked code. */
    magma_sgetf2_nopiv( m, n, A + 1 + 1 * lda, lda, info );
    }
  else
    {
    /* Use blocked code. */
    for (j = 1; j <= min_mn; j += nb)
      {
      jb = std::min( min_mn - j + 1, nb );

      /* Factor diagonal and subdiagonal blocks and test for exact
         singularity. */
      m_j_jb = m - j - jb + 1;
      magma_sgetf2_nopiv( jb, jb, A + j + j * lda, lda, &iinfo );
      coot_fortran(coot_strsm)( "R", "U", "N", "N", &m_j_jb, &jb, &c_one,
                                A + (j   ) + j * lda, &lda,
                                A + (j+jb) + j * lda, &lda );

      /* Adjust INFO */
      if (*info == 0 && iinfo > 0)
        {
        *info = iinfo + j - 1;
        }

      if (j + jb <= n)
        {
        /* Compute block row of U. */
        n_j_jb = n - j - jb + 1;
        coot_fortran(coot_strsm)( "L", "L", "N", "U",
                                  &jb, &n_j_jb, &c_one,
                                  A + j + (j   ) * lda, &lda,
                                  A + j + (j+jb) * lda, &lda );
        if (j + jb <= m)
          {
          /* Update trailing submatrix. */
          m_j_jb = m - j - jb + 1;
          n_j_jb = n - j - jb + 1;
          coot_fortran(coot_sgemm)( "N", "N",
                                    &m_j_jb, &n_j_jb, &jb, &c_neg_one,
                                    A + (j+jb) + (j   ) * lda, &lda,
                                    A + (j   ) + (j+jb) * lda, &lda, &c_one,
                                    A + (j+jb) + (j+jb) * lda, &lda );
          }
        }
      }
    }

  return *info;
  } /* magma_sgetrf_nopiv */



// Purpose
// -------
// SGETRF_NOPIV_GPU computes an LU factorization of a general M-by-N
// matrix A without any pivoting.
//
// The factorization has the form
//     A = L * U
// where L is lower triangular with unit
// diagonal elements (lower trapezoidal if m > n), and U is upper
// triangular (upper trapezoidal if m < n).
//
// This is the right-looking Level 3 BLAS version of the algorithm.

inline
magma_int_t
magma_sgetrf_nopiv_gpu
  (
  magma_int_t m, magma_int_t n,
  magmaFloat_ptr dA, const size_t dA_offset, magma_int_t ldda,
  magma_int_t *info
  )
  {
  float c_one     = MAGMA_S_ONE;
  float c_neg_one = MAGMA_S_NEG_ONE;

  magma_int_t iinfo, nb;
  magma_int_t maxm, mindim; 
  magma_int_t j, rows, s, ldwork;
  float *work;

  /* Check arguments */
  *info = 0;
  if (m < 0)
    {
    *info = -1;
    }
  else if (n < 0)
    {
    *info = -2;
    }
  else if (ldda < std::max(1,m))
    {
    *info = -4;
    }

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }

  /* Quick return if possible */
  if (m == 0 || n == 0)
    {
    return *info;
    }

  /* Function Body */
  mindim = std::min( m, n );
  nb     = magma_get_sgetrf_nb( m );
  s      = mindim / nb;

  magma_queue_t queues[2];

  queues[0] = magma_queue_create();
  queues[1] = magma_queue_create();

  if (nb <= 1 || nb >= std::min(m,n))
    {
    /* Use CPU code. */
    if ( MAGMA_SUCCESS != magma_smalloc_cpu( &work, m*n ))
      {
      *info = MAGMA_ERR_HOST_ALLOC;
      goto cleanup;
      }
    magma_sgetmatrix( m, n, dA, dA_offset, ldda, work, m, queues[0] );
    magma_sgetrf_nopiv( m, n, work, m, info );
    magma_ssetmatrix( m, n, work, m, dA, dA_offset, ldda, queues[0] );
    magma_free_cpu( work );
    }
  else
    {
    /* Use hybrid blocked code. */
    maxm = magma_roundup( m, 32 );

    ldwork = maxm;
    if (MAGMA_SUCCESS != magma_smalloc_pinned( &work, ldwork*nb ))
      {
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
      }

    for( j=0; j < s; j++ )
      {
      // get j-th panel from device
      magma_queue_sync( queues[1] );
      magma_sgetmatrix_async( m-j*nb, nb, dA, dA_offset + j * nb + j * nb * ldda, ldda, work, ldwork, queues[0] );

      if ( j > 0 )
        {
        magma_strsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                     nb, n - (j+1)*nb,
                     c_one, dA, dA_offset + (j-1) * nb + (j-1) * nb * ldda, ldda,
                            dA, dA_offset + (j-1) * nb + (j+1) * nb * ldda, ldda, queues[1] );
        magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                     m-j*nb, n-(j+1)*nb, nb,
                     c_neg_one, dA, dA_offset + (j  ) * nb + (j-1) * nb * ldda, ldda,
                                dA, dA_offset + (j-1) * nb + (j+1) * nb * ldda, ldda,
                     c_one,     dA, dA_offset + (j  ) * nb + (j+1) * nb * ldda, ldda, queues[1] );
        }

      // do the cpu part
      rows = m - j*nb;
      magma_queue_sync( queues[0] );
      magma_sgetrf_nopiv( rows, nb, work, ldwork, &iinfo );
      if ( *info == 0 && iinfo > 0 )
        {
        *info = iinfo + j*nb;
        }

      // send j-th panel to device
      magma_ssetmatrix_async( m-j*nb, nb, work, ldwork, dA, dA_offset + j * nb + j * nb * ldda, ldda, queues[0] );
      magma_queue_sync( queues[0] );

      // do the small non-parallel computations (next panel update)
      if ( s > j+1 )
        {
        magma_strsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                     nb, nb,
                     c_one, dA, dA_offset + (j  ) * nb + (j  ) * nb * ldda, ldda,
                            dA, dA_offset + (j  ) * nb + (j+1) * nb * ldda, ldda, queues[1] );
        magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                     m-(j+1)*nb, nb, nb,
                     c_neg_one, dA, dA_offset + (j+1) * nb + (j  ) * nb * ldda, ldda,
                                dA, dA_offset + (j  ) * nb + (j+1) * nb * ldda, ldda,
                     c_one,     dA, dA_offset + (j+1) * nb + (j+1) * nb * ldda, ldda, queues[1] );
        }
      else
        {
        magma_strsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                     nb, n-s*nb,
                     c_one, dA, dA_offset + (j  ) * nb + (j  ) * nb * ldda, ldda,
                            dA, dA_offset + (j  ) * nb + (j+1) * nb * ldda, ldda, queues[1] );
        magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                     m-(j+1)*nb, n-(j+1)*nb, nb,
                     c_neg_one, dA, dA_offset + (j+1) * nb + (j  ) * nb * ldda, ldda,
                                dA, dA_offset + (j  ) * nb + (j+1) * nb * ldda, ldda,
                     c_one,     dA, dA_offset + (j+1) * nb + (j+1) * nb * ldda, ldda, queues[1] );
        }
      }

    magma_int_t nb0 = std::min( m - s*nb, n - s*nb );
    if ( nb0 > 0 )
      {
      rows = m - s*nb;

      magma_sgetmatrix( rows, nb0, dA, dA_offset + s * nb + s * nb * ldda, ldda, work, ldwork, queues[1] );

      // do the cpu part
      magma_sgetrf_nopiv( rows, nb0, work, ldwork, &iinfo );
      if ( *info == 0 && iinfo > 0 )
        {
        *info = iinfo + s*nb;
        }

      // send j-th panel to device
      magma_ssetmatrix( rows, nb0, work, ldwork, dA, dA_offset + s * nb + s * nb * ldda, ldda, queues[1] );

      magma_strsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                   nb0, n-s*nb-nb0,
                   c_one, dA, dA_offset + s * nb + s * nb * ldda,       ldda,
                          dA, dA_offset + s * nb + s * nb * ldda + nb0, ldda, queues[1] );
      }

    magma_free_pinned( work );
    }

cleanup:
  magma_queue_destroy( queues[0] );
  magma_queue_destroy( queues[1] );

  return *info;
  } /* magma_sgetrf_nopiv_gpu */
