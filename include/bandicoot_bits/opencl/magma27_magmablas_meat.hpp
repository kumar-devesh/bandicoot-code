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



// Adaptations of magmablas_* functions to use existing bandicoot backend functionality.

inline
void
magmablas_dlaset
  (
  magma_uplo_t uplo,
  magma_int_t m,
  magma_int_t n,
  double offdiag,
  double diag,
  magmaDouble_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue
  )
  {
  magma_int_t info = 0;
  if (uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull)
    info = -1;
  else if ( m < 0 )
    info = -2;
  else if ( n < 0 )
    info = -3;
  else if ( ldda < std::max(1,m) )
    info = -7;

  if (info != 0)
    {
    // magma_xerbla( __func__, -(info) );
    return;  //info;
    }

  if (m == 0 || n == 0)
    {
    return;
    }

  opencl::magma_real_kernel_id::enum_id num;
  if (uplo == MagmaLower)
    {
    num = opencl::magma_real_kernel_id::laset_lower;
    }
  else if (uplo == MagmaUpper)
    {
    num = opencl::magma_real_kernel_id::laset_upper;
    }
  else
    {
    num = opencl::magma_real_kernel_id::laset_full;
    }

  magmablas_run_laset_kernel(num, uplo, m, n, offdiag, diag, dA, dA_offset, ldda, queue);
  }



template<typename eT>
inline
void
magmablas_run_laset_kernel
  (
  const opencl::magma_real_kernel_id::enum_id num,
  magma_uplo_t uplo,
  magma_int_t m,
  magma_int_t n,
  eT offdiag,
  eT diag,
  cl_mem dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue
  )
  {
  cl_int status;

  opencl::runtime_t::adapt_uword local_m(m);
  opencl::runtime_t::adapt_uword local_n(n);
  opencl::runtime_t::adapt_uword local_dA_offset(dA_offset);
  opencl::runtime_t::adapt_uword local_ldda(ldda);

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(num);

  status  = clSetKernelArg(k, 0, local_m.size,         local_m.addr);
  status |= clSetKernelArg(k, 1, local_n.size,         local_n.addr);
  status |= clSetKernelArg(k, 2, sizeof(eT),           &offdiag);
  status |= clSetKernelArg(k, 3, sizeof(eT),           &diag);
  status |= clSetKernelArg(k, 4, sizeof(cl_mem),       &dA);
  status |= clSetKernelArg(k, 5, local_dA_offset.size, local_dA_offset.addr);
  status |= clSetKernelArg(k, 6, local_ldda.size,      local_ldda.addr);
  coot_check_runtime_error(status, "coot::opencl::magmablas_run_laset_kernel(): couldn't set kernel arguments");

  size_t threads[2] = { MAGMABLAS_BLK_X,                     1                                   };
  size_t grid[2]    = { size_t(m - 1) / MAGMABLAS_BLK_X + 1, size_t(n - 1) / MAGMABLAS_BLK_Y + 1 };
  grid[0] *= threads[0];
  grid[1] *= threads[1];

  status |= clEnqueueNDRangeKernel(queue, k, 2, NULL, grid, threads, 0, NULL, NULL);
  coot_check_runtime_error(status, "coot::opencl::magmablas_run_laset_kernel(): couldn't execute kernel");
  }



inline
void
magmablas_dtranspose
  (
  magma_int_t m,
  magma_int_t n,
  magmaDouble_const_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magmaDouble_ptr dAT,
  size_t dAT_offset,
  magma_int_t lddat,
  magma_queue_t queue
  )
  {
  // TODO
  }
