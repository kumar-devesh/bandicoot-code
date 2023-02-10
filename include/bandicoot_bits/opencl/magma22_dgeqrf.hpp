// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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
// clMAGMA 1.3 (2014-11-14) and/or MAGMA 2.2 (2016-11-20).
// clMAGMA 1.3 and MAGMA 2.2 are distributed under a
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


inline
magma_int_t
magma_get_dgeqrf_nb( magma_int_t m, magma_int_t n )
  {
  // Note: this is tuned for AMD Tahiti cards (taken from clBLAS).
  magma_int_t minmn = std::min(m, n);

  if   (minmn <= 2048) return  64;
  else                 return 128;
  }


inline
void dsplit_diag_block_invert(
    magma_int_t ib, double *A, magma_int_t lda,
    double *work )
{
    const double c_zero = MAGMA_D_ZERO;
    const double c_one  = MAGMA_D_ONE;

    magma_int_t i, j, info;
    double *cola, *colw;

    for (i=0; i < ib; i++) {
        cola = A    + i*lda;
        colw = work + i*ib;
        for (j=0; j < i; j++) {
            colw[j] = cola[j];
            cola[j] = c_zero;
        }
        colw[i] = cola[i];
        cola[i] = c_one;
    }
           //lapackf77_dtrtri( MagmaUpperStr, MagmaNonUnitStr, &ib, work, &ib, &info );
    coot_fortran(coot_dtrtri)( MagmaUpperStr, MagmaNonUnitStr, &ib, work, &ib, &info );
}


/***************************************************************************//**
    Purpose
    -------
    DGEQRF computes a QR factorization of a real M-by-N matrix A:
    A = Q * R.

    This version stores the triangular dT matrices used in
    the block QR factorization so that they can be applied directly (i.e.,
    without being recomputed) later. As a result, the application
    of Q is much faster. Also, the upper triangular matrices for V have 0s
    in them. The corresponding parts of the upper triangular R are inverted and
    stored separately in dT.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    tau     DOUBLE PRECISION array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    dT      (workspace) DOUBLE PRECISION array on the GPU,
            dimension (2*MIN(M, N) + ceil(N/32)*32 )*NB,
            where NB can be obtained through magma_get_dgeqrf_nb( M, N ).
            It starts with a MIN(M,N)*NB block that stores the triangular T
            matrices, followed by a MIN(M,N)*NB block that stores inverses of
            the diagonal blocks of the R matrix.
            The rest of the array is used as workspace.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v^H

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqrf
*******************************************************************************/
inline
magma_int_t
magma_dgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT, size_t dT_offset,
    magma_int_t *info )
{
    // #ifdef HAVE_clBLAS
    // #define dA(i_, j_)  dA, (dA_offset + (i_) + (j_)*(ldda))
    // #define dT(i_)      dT, (dT_offset + (i_)*nb)
    // #define dR(i_)      dT, (dT_offset + (  minmn + (i_))*nb)
    // #define dwork(i_)   dT, (dT_offset + (2*minmn + (i_))*nb)
    // #else
    // #define dA(i_, j_) (dA + (i_) + (j_)*(ldda))
    // #define dT(i_)     (dT + (i_)*nb)
    // #define dR(i_)     (dT + (  minmn + (i_))*nb)
    // #define dwork(i_)  (dT + (2*minmn + (i_))*nb)
    // #endif

    // CS: dA_offset is not used

    #define dA(i_, j_)  dA, (dA_offset + (i_) + (j_)*(ldda))
    #define dT(i_)      dT, (dT_offset + (i_)*nb)
    #define dR(i_)      dT, ((  minmn + (i_))*nb)
    #define dwork(i_)   dT, ((2*minmn + (i_))*nb)

    double *work, *hwork, *R;
    magma_int_t cols, i, ib, ldwork, lddwork, lhwork, lwork, minmn, nb, old_i, old_ib, rows;

    // check arguments
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < std::max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        // magma_xerbla( __func__, -(*info) );
        return *info;
    }

    minmn = std::min( m, n );
    if (minmn == 0)
        return *info;

    // TODO: use min(m,n), but that affects dT
    nb = magma_get_dgeqrf_nb( m, n );

    // dT contains 3 blocks:
    // dT    is minmn*nb
    // dR    is minmn*nb
    // dwork is n*nb
    lddwork = n;

    // work  is m*nb for panel
    // hwork is n*nb, and at least nb*nb for T in larft
    // R     is nb*nb
    ldwork = m;
    lhwork = std::max( n*nb, nb*nb );
    lwork  = ldwork*nb + lhwork + nb*nb;
    // last block needs rows*cols for matrix and prefers cols*nb for work
    // worst case is n > m*nb, m a small multiple of nb:
    // needs n*nb + n > (m+n)*nb
    // prefers 2*n*nb, about twice above (m+n)*nb.
    i = ((minmn-1)/nb)*nb;
    lwork = std::max( lwork, (m-i)*(n-i) + (n-i)*nb );

    if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, lwork )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    hwork = work + ldwork*nb;
    R     = work + ldwork*nb + lhwork;
    std::memset( R, 0, nb*nb*sizeof(double) );

    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );

    if ( nb > 1 && nb < minmn ) {
        // need nb*nb for T in larft
        assert( lhwork >= nb*nb );

        // Use blocked code initially
        old_i = 0; old_ib = nb;
        for (i = 0; i < minmn-nb; i += nb) {
            ib = std::min( minmn-i, nb );
            rows = m - i;

            // get i-th panel from device
            magma_dgetmatrix_async( rows, ib,
                                    dA(i,i), ldda,
                                    work,    ldwork, queues[1] );
            if (i > 0) {
                // Apply H^H to A(i:m,i+2*ib:n) from the left
                cols = n - old_i - 2*old_ib;
                magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, cols, old_ib,
                                  dA(old_i, old_i         ), ldda, dT(old_i), nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork(0),  lddwork, queues[0] );

                // Fix the diagonal block
                magma_dsetmatrix_async( old_ib, old_ib,
                                        R,         old_ib,
                                        dR(old_i), old_ib, queues[0] );
            }

            magma_queue_sync( queues[1] );  // wait to get work(i)
                  // lapackf77_dgeqrf( &rows, &ib, work, &ldwork, &tau[i], hwork, &lhwork, info );
            coot_fortran(coot_dgeqrf)( &rows, &ib, work, &ldwork, &tau[i], hwork, &lhwork, info );

            // Form the triangular factor of the block reflector in hwork
            // H = H(i) H(i+1) . . . H(i+ib-1)
            // TODO: wrapper for dlarft
            coot_fortran(coot_dlarft)( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              work, &ldwork, &tau[i], hwork, &ib );

            // wait for previous trailing matrix update (above) to finish with R
            magma_queue_sync( queues[0] );

            // copy the upper triangle of panel to R and invert it, and
            // set  the upper triangle of panel (V) to identity
            dsplit_diag_block_invert( ib, work, ldwork, R );

            // send i-th V matrix to device
            magma_dsetmatrix( rows, ib,
                              work, ldwork,
                              dA(i,i), ldda, queues[1] );

            if (i + ib < n) {
                // send T matrix to device
                magma_dsetmatrix( ib, ib,
                                  hwork, ib,
                                  dT(i), nb, queues[1] );

                if (i+nb < minmn-nb) {
                    // Apply H^H to A(i:m,i+ib:i+2*ib) from the left
                    magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(i, i   ), ldda, dT(i),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
                    // wait for larfb to finish with dwork before larfb in next iteration starts
                    magma_queue_sync( queues[1] );
                }
                else {
                    // Apply H^H to A(i:m,i+ib:n) from the left
                    magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, n-i-ib, ib,
                                      dA(i, i   ), ldda, dT(i),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
                    // Fix the diagonal block
                    magma_dsetmatrix( ib, ib,
                                      R,     ib,
                                      dR(i), ib, queues[1] );
                }
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }

    // Use unblocked code to factor the last or only block.
    if (i < minmn) {
        rows = m-i;
        cols = n-i;
        magma_dgetmatrix( rows, cols, dA(i, i), ldda, work, rows, queues[1] );
        // see comments for lwork above
        lhwork = lwork - rows*cols;
              // lapackf77_dgeqrf( &rows, &cols, work, &rows, &tau[i], &work[rows*cols], &lhwork, info );
        coot_fortran(coot_dgeqrf)( &rows, &cols, work, &rows, &tau[i], &work[rows*cols], &lhwork, info );

        magma_dsetmatrix( rows, cols, work, rows, dA(i, i), ldda, queues[1] );
    }

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    magma_free_pinned( work );

    return *info;

    #undef dA
    #undef dT
    #undef dR
    #undef dwork

} // magma_dgeqrf_gpu
