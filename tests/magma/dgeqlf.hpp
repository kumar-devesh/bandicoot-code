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



namespace coot {

// Purpose
// -------
// DGEQLF computes a QL factorization of a DOUBLE PRECISION M-by-N matrix A:
// A = Q * L.



inline
magma_int_t
magma_dgeqlf
  (
  magma_int_t m, magma_int_t n,
  double *A,    magma_int_t lda, double *tau,
  double *work, magma_int_t lwork,
  magma_int_t *info
  )
  {
  /* Constants */
  const double c_one = MAGMA_D_ONE;

  /* Local variables */
  magmaDouble_ptr dA, dwork;
  magma_int_t i, minmn, lddwork, old_i, old_ib, nb;
  magma_int_t rows, cols;
  magma_int_t ib, ki, kk, mu, nu, iinfo, ldda;

  nb = magma_get_dgeqrf_nb( m, n );
  *info = 0;
  bool lquery = (lwork == -1);

  // silence "uninitialized" warnings
  old_ib = nb;
  old_i  = 0;

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

  minmn = std::min(m,n);
  if (*info == 0)
    {
    if (minmn == 0)
      {
      work[0] = c_one;
      }
    else
      {
      work[0] = magma_dmake_lwork( std::max(n*nb, 2*nb*nb) );
      }

    if (lwork < std::max(std::max(1,n), 2*nb*nb) && ! lquery)
      {
      *info = -7;
      }
    }

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }
  else if (lquery)
    {
    return *info;
    }

  /* Quick return if possible */
  if (minmn == 0)
    {
    return *info;
    }

  lddwork = magma_roundup( n, 32 );
  ldda    = magma_roundup( m, 32 );

  if (MAGMA_SUCCESS != magma_dmalloc( &dA, n*ldda + nb*lddwork ))
    {
    *info = MAGMA_ERR_DEVICE_ALLOC;
    return *info;
    }
  dwork = dA;
  size_t dwork_offset = ldda * n;

  magma_queue_t queues[2];
  queues[0] = magma_queue_create();
  queues[1] = magma_queue_create();

  if ( (nb > 1) && (nb < minmn) )
    {
    /*  Use blocked code initially.
        The last kk columns are handled by the block method.
        First, copy the matrix on the GPU except the last kk columns */
    magma_dsetmatrix_async( m, n-nb,
                            A,  lda,
                            dA, 0, ldda, queues[0] );

    ki = ((minmn - nb - 1) / nb) * nb;
    kk = std::min( minmn, ki + nb );
    for (i = minmn - kk + ki; i >= minmn - kk; i -= nb)
      {
      ib = std::min( minmn-i, nb );

      if (i < minmn - kk + ki)
        {
        // 1. Copy asynchronously the current panel to the CPU.
        // 2. Copy asynchronously the submatrix below the panel to the CPU
        rows = m - minmn + i + ib;
        magma_dgetmatrix_async( rows, ib,
                                dA, (n-minmn+i) * ldda, ldda,
                                A + (n-minmn+i) * lda,  lda, queues[1] );

        magma_dgetmatrix_async( m-rows, ib,
                                dA, rows + (n-minmn+i) * ldda, ldda,
                                A + rows + (n-minmn+i) * lda,  lda, queues[0] );

        /* Apply H^H to A(1:m-minmn+i+ib-1,1:n-minmn+i-1) from the left in
           two steps - implementing the lookahead techniques.
           This is the main update from the lookahead techniques. */
        rows = m - minmn + old_i + old_ib;
        cols = n - minmn + old_i - old_ib;
        magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaBackward, MagmaColumnwise,
                          rows, cols, old_ib,
                          dA, (cols+old_ib) * ldda, ldda, dwork, dwork_offset,          lddwork,
                          dA, 0,                    ldda, dwork, dwork_offset + old_ib, lddwork, queues[0] );
        }

      magma_queue_sync( queues[1] );  // wait for panel
      /* Compute the QL factorization of the current block
         A(1:m-minmn+i+ib-1,n-minmn+i:n-minmn+i+ib-1) */
      rows = m - minmn + i + ib;
      cols = n - minmn + i;
      coot_fortran(coot_dgeqlf)( &rows, &ib, A + cols * lda, &lda, tau+i, work, &lwork, &iinfo );

      if (cols > 0)
        {
        /* Form the triangular factor of the block reflector
           H = H(i+ib-1) . . . H(i+1) H(i) */
        coot_fortran(coot_dlarft)( MagmaBackwardStr, MagmaColumnwiseStr,
                                   &rows, &ib,
                                   A + cols * lda, &lda, tau + i, work, &ib );

        magma_dpanel_to_q( MagmaLower, ib, A + (rows-ib) + cols * lda, lda, work+ib*ib );
        magma_dsetmatrix( rows, ib,
                          A + cols * lda,  lda,
                          dA, cols * ldda, ldda, queues[1] );
        magma_dq_to_panel( MagmaLower, ib, A + (rows-ib) + cols * lda, lda, work+ib*ib );

        // wait for main update (above) to finish with dwork
        magma_queue_sync( queues[0] );

        // Send the triangular part to the GPU
        magma_dsetmatrix( ib, ib, work, ib, dwork, dwork_offset, lddwork, queues[1] );

        /* Apply H^H to A(1:m-minmn+i+ib-1,1:n-minmn+i-1) from the left in
           two steps - implementing the lookahead techniques.
           This is the update of first ib columns.                 */
        if (i-ib >= minmn - kk)
          {
          magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaBackward, MagmaColumnwise,
                            rows, ib, ib,
                            dA, cols * ldda,      ldda, dwork, dwork_offset,      lddwork,
                            dA, (cols-ib) * ldda, ldda, dwork, dwork_offset + ib, lddwork, queues[1] );
          // wait for larfb to finish with dwork before larfb in next iteration starts
          magma_queue_sync( queues[1] );
          }
        else
          {
          magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaBackward, MagmaColumnwise,
                            rows, cols, ib,
                            dA, cols * ldda, ldda, dwork, dwork_offset,      lddwork,
                            dA, 0,           ldda, dwork, dwork_offset + ib, lddwork, queues[1] );
          }

        old_i  = i;
        old_ib = ib;
      }
    }
    mu = m - minmn + i + nb;
    nu = n - minmn + i + nb;

    magma_dgetmatrix( m, nu, dA, 0, ldda, A, lda, queues[1] );
    }
  else
    {
    mu = m;
    nu = n;
    }

  /* Use unblocked code to factor the last or only block */
  if (mu > 0 && nu > 0)
    {
    coot_fortran(coot_dgeqlf)( &mu, &nu, A, &lda, tau, work, &lwork, &iinfo );
    }

  magma_queue_destroy( queues[0] );
  magma_queue_destroy( queues[1] );
  magma_free( dA );

  return *info;
  } /* magma_dgeqlf */



  } // namespace coot
