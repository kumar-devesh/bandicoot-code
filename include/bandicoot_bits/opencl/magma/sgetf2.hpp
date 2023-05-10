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
// SGETF2_NOPIV computes an LU factorization of a general m-by-n
// matrix A without pivoting.
//
// The factorization has the form
//    A = L * U
// where L is lower triangular with unit diagonal elements (lower
// trapezoidal if m > n), and U is upper triangular (upper
// trapezoidal if m < n).
//
// This is the right-looking Level 2 BLAS version of the algorithm.
//
// This is a CPU-only (not accelerated) version.

inline
magma_int_t
magma_sgetf2_nopiv
  (
  magma_int_t m, magma_int_t n,
  float *A, magma_int_t lda,
  magma_int_t *info
  )
  {
  float c_one     = MAGMA_S_ONE;
  float c_zero    = MAGMA_S_ZERO;
  float c_neg_one = MAGMA_S_NEG_ONE;
  magma_int_t ione = 1;

  magma_int_t min_mn, m_j, n_j;
  float inv_Ajj;
  magma_int_t i, j;
  float sfmin;

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

  /* Compute machine safe minimum */
  sfmin = coot_fortran(coot_slamch)("S");

  min_mn = std::min(m,n);
  for (j = 1; j <= min_mn; ++j)
    {
    /* Test for singularity. */
    if ( ! (*(A + j + j * lda) == c_zero))
      {
      /* Compute elements J+1:M of J-th column. */
      if (j < m)
        {
        if (std::abs( *(A + j + j * lda) ) >= sfmin)
          {
          m_j = m - j;
          inv_Ajj = c_one / *(A + j + j * lda);
          coot_fortran(coot_sscal)( &m_j, &inv_Ajj, A + (j+1) + j * lda, &ione );
          }
        else
          {
          m_j = m - j;
          for (i = 1; i <= m_j; ++i)
            {
            *(A + (j+i) + j * lda) = (*(A + (j+i) + j * lda)) / (*(A + j + j * lda));
            }
          }
        }
      }
    else if (*info == 0)
      {
      *info = j;
      }

    if (j < min_mn)
      {
      /* Update trailing submatrix. */
      m_j = m - j;
      n_j = n - j;
      coot_fortran(coot_sger)( &m_j, &n_j, &c_neg_one,
                               A + (j+1) + (j  ) * lda, &ione,
                               A + (j  ) + (j+1) * lda, &lda,
                               A + (j+1) + (j+1) * lda, &lda );
      }
    }

  return *info;
  } /* magma_sgetf2_nopiv */
