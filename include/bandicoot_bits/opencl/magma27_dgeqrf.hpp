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



// DGEQRF computes a QR factorization of a DOUBLE PRECISION M-by-N matrix A:
// A = Q * R. This version does not require work space on the GPU
// passed as input. GPU memory is allocated in the routine.



inline
magma_int_t
magma_dgeqrf
  (
  magma_int_t m, magma_int_t n,
  double *A,    magma_int_t lda,
  double *tau,
  double *work, magma_int_t lwork,
  magma_int_t *info
  )
  {
  /* Constants */
  const double c_one = MAGMA_D_ONE;

  /* Local variables */
  double* work_local = NULL;
  magmaDouble_ptr dA, dT, dwork;
  size_t dT_offset, dwork_offset;
  magma_int_t i, ib, min_mn, ldda, lddwork, old_i, old_ib;

  /* Function Body */
  *info = 0;
  magma_int_t nb = magma_get_dgeqrf_nb( m, n );

  magma_int_t lwkopt = n*nb;
  work[0] = magma_dmake_lwork( lwkopt );
  bool lquery = (lwork == -1);
  if (m < 0)
    *info = -1;
  else if (n < 0)
    *info = -2;
  else if (lda < std::max(1, m))
    *info = -4;
  else if (lwork < std::max(1, lwkopt) && !lquery)
    *info = -7;

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }
  else if (lquery)
    {
    return *info;
    }

  min_mn = std::min( m, n );
  if (min_mn == 0)
    {
    work[0] = c_one;
    return *info;
    }

  if (nb <= 1 || 4*nb >= std::min(m,n) )
    {
    /* Use CPU code. */
    coot_fortran(coot_dgeqrf)( &m, &n, A, &lda, tau, work, &lwork, info );
    return *info;
    }

  // largest N for larfb is n-nb (trailing matrix lacks 1st panel)
  lddwork = magma_roundup( n, 32 ) - nb;
  ldda    = magma_roundup( m, 32 );

  // allocate space for dA, dwork, and dT
  if (MAGMA_SUCCESS != magma_dmalloc( &dA, n*ldda + nb*lddwork + nb*nb ))
    {
    /* alloc failed so call non-GPU-resident version */
    // TODO: port the function below
    //return magma_dgeqrf_ooc( m, n, A, lda, tau, work, lwork, info );
    *info = MAGMA_ERR_DEVICE_ALLOC;
    return *info;
    }

  // Need at least 2*nb*nb to store T and upper triangle of V simultaneously.
  // For better LAPACK compatability, which needs N*NB,
  // allow lwork < 2*NB*NB and allocate here if needed.
  if (lwork < 2*nb*nb)
    {
    if (MAGMA_SUCCESS != magma_dmalloc_cpu( &work_local, 2*nb*nb ))
      {
      magma_free( dA );
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
      }
    work = work_local;
    }

  dwork = dA;
  dwork_offset = n*ldda;
  dT    = dA;
  dT_offset = n*ldda + nb*lddwork;

  magma_queue_t queues[2];
  magma_device_t cdev;
  magma_getdevice( &cdev );
  magma_queue_create( cdev, &queues[0] );
  magma_queue_create( cdev, &queues[1] );

  if ( (nb > 1) && (nb < min_mn) )
    {
    /* Use blocked code initially.
       Asynchronously send the matrix to the GPU except the first panel. */
    magma_dsetmatrix_async( m, n-nb,
                             &A[nb * lda], lda,
                            dA, nb * ldda, ldda, queues[0] );

    old_i = 0;
    old_ib = nb;
    for (i = 0; i < min_mn-nb; i += nb)
      {
      ib = std::min( min_mn-i, nb );
      if (i > 0)
        {
        /* get i-th panel from device */
        magma_queue_sync( queues[1] );
        magma_dgetmatrix_async( m-i, ib,
                                dA, i + i * ldda, ldda,
                                 &A[i + i * lda], lda, queues[0] );

        /* Apply H' to A(i:m,i+2*ib:n) from the left */
        magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                          m-old_i, n-old_i-2*old_ib, old_ib,
                          dA,    old_i + old_i * ldda,            ldda,
                          dT,    dT_offset,                       nb,
                          dA,    old_i + (old_i+2*old_ib) * ldda, ldda,
                          dwork, dwork_offset,                    lddwork, queues[1] );

        magma_dgetmatrix_async( i, ib,
                                dA, i * ldda, ldda,
                                 &A[i * lda], lda, queues[1] );
        magma_queue_sync( queues[0] );
        }

      magma_int_t rows = m-i;
      coot_fortran(coot_dgeqrf)( &rows, &ib, &A[i + i * lda], &lda, tau+i, work, &lwork, info );

      /* Form the triangular factor of the block reflector
         H = H(i) H(i+1) . . . H(i+ib-1) */
      coot_fortran(coot_dlarft)( MagmaForwardStr, MagmaColumnwiseStr,
                                 &rows, &ib, &A[i + i * lda], &lda, tau+i, work, &ib );

      magma_dpanel_to_q( MagmaUpper, ib, &A[i + i * lda], lda, work+ib*ib );

      /* put i-th V matrix onto device */
      magma_dsetmatrix_async( rows, ib, &A[i + i * lda], lda, dA, i + i * ldda, ldda, queues[0] );

      /* put T matrix onto device */
      magma_queue_sync( queues[1] );
      magma_dsetmatrix_async( ib, ib, work, ib, dT, dT_offset, nb, queues[0] );
      magma_queue_sync( queues[0] );

      if (i + ib < n)
        {
        if (i+ib < min_mn-nb)
          {
          /* Apply H' to A(i:m,i+ib:i+2*ib) from the left (look-ahead) */
          magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, ib, ib,
                            dA,    i + i * ldda,      ldda,
                            dT,    dT_offset,         nb,
                            dA,    i + (i+ib) * ldda, ldda,
                            dwork, dwork_offset,      lddwork, queues[1] );
          magma_dq_to_panel( MagmaUpper, ib, &A[i + i * lda], lda, work+ib*ib );
          }
        else
          {
          /* After last panel, update whole trailing matrix. */
          /* Apply H' to A(i:m,i+ib:n) from the left */
          magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, n-i-ib, ib,
                            dA,    i + i * ldda,      ldda,
                            dT,    dT_offset,         nb,
                            dA,    i + (i+ib) * ldda, ldda,
                            dwork, dwork_offset,      lddwork, queues[1] );
          magma_dq_to_panel( MagmaUpper, ib, &A[i + i * lda], lda, work+ib*ib );
          }

        old_i  = i;
        old_ib = ib;
        }
      }
    }
  else
    {
    i = 0;
    }

  /* Use unblocked code to factor the last or only block. */
  if (i < min_mn)
    {
    ib = n-i;
    if (i != 0)
      {
      magma_dgetmatrix( m, ib, dA, i * ldda, ldda, &A[i * lda], lda, queues[1] );
      }
    magma_int_t rows = m-i;
    coot_fortran(coot_dgeqrf)( &rows, &ib, &A[i + i * lda], &lda, tau+i, work, &lwork, info );
    }

  magma_queue_sync( queues[0] );
  magma_queue_sync( queues[1] );
  magma_queue_destroy( queues[0] );
  magma_queue_destroy( queues[1] );

  work[0] = magma_dmake_lwork( lwkopt );  // before free( work_local )

  magma_free( dA );
  magma_free_cpu( work_local );  // if allocated

  return *info;
  } /* magma_dgeqrf */
