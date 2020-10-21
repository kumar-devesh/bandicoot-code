// Copyright 2020 Ryan Curtin (http://www.ratml.org)
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


//! \addtogroup opencl
//! @{

/**
 * Compute the dot product of two vectors.
 */
template<typename eT1, typename eT2>
inline
typename promote_type<eT1, eT2>::result
dot(dev_mem_t<eT1> mem1, dev_mem_t<eT2> mem2, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot_cl_rt not valid" );

  typedef typename promote_type<eT1, eT2>::result promoted_eT;

  // work out the number of chunks, ensuring that there are at least 4 elements per compunit

  uword n_chunks = get_rt().cl_rt.get_n_units();

  while(n_chunks >= 1)
    {
    if( (n_elem / n_chunks) >= uword(4) ) { break; }

    n_chunks /= uword(2);
    }

  n_chunks = (std::max)(uword(1), n_chunks);

  const uword chunk_size = n_elem / n_chunks;

  Mat<promoted_eT> tmp(n_chunks, 1);

  runtime_t::cq_guard guard;

  cl_int status = 0;

  cl_kernel k1 = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::dot_chunked);

  dev_mem_t<promoted_eT> tmp_mem = tmp.get_dev_mem(false);

  runtime_t::adapt_uword dev_chunk_size(chunk_size);
  runtime_t::adapt_uword dev_n_chunks  (n_chunks  );

  status |= clSetKernelArg(k1, 0, sizeof(cl_mem),      &(tmp_mem.cl_mem_ptr)           );
  status |= clSetKernelArg(k1, 1, sizeof(cl_mem),      &(mem1.cl_mem_ptr)              );
  status |= clSetKernelArg(k1, 2, sizeof(cl_mem),      &(mem2.cl_mem_ptr)              );
  status |= clSetKernelArg(k1, 3, dev_chunk_size.size, dev_chunk_size.addr             );
  status |= clSetKernelArg(k1, 4, dev_n_chunks.size,   dev_n_chunks.addr               );

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0        };
  const size_t k1_work_size[1]   = { n_chunks };

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k1, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "dot()");

  clFlush(get_rt().cl_rt.get_cq());

  cl_kernel k2 = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::dot_twostage);

  runtime_t::adapt_uword dev_out_len(tmp.n_elem);
  runtime_t::adapt_uword dev_A_start(n_chunks * chunk_size);
  runtime_t::adapt_uword dev_A_len(n_elem);

  status |= clSetKernelArg(k2, 0, sizeof(cl_mem),   &(tmp_mem.cl_mem_ptr)       );
  status |= clSetKernelArg(k2, 1, dev_out_len.size, dev_out_len.addr            );
  status |= clSetKernelArg(k2, 2, sizeof(cl_mem),   &(mem1.cl_mem_ptr)          );
  status |= clSetKernelArg(k2, 3, sizeof(cl_mem),   &(mem2.cl_mem_ptr)          );
  status |= clSetKernelArg(k2, 4, dev_A_start.size, dev_A_start.addr            );
  status |= clSetKernelArg(k2, 5, dev_A_len.size,   dev_A_len.addr              );

  const size_t k2_work_dim       = 1;
  const size_t k2_work_offset[1] = { 0 };
  const size_t k2_work_size[1]   = { 1 };

  status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), k2, k2_work_dim, k2_work_offset, k2_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "dot()");

  clFlush(get_rt().cl_rt.get_cq());

  return promoted_eT(tmp(0));
  }
