// Copyright 2021 Ryan Curtin (http://www.ratml.org)
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



/**
 * Compute the maximum of all elements in `mem`.
 * This is basically the same as accu(), which is also a reduction.
 */
template<typename eT>
inline
eT
max(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::max(): OpenCL runtime not valid" );

  cl_int status = 0;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::max);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::max_small);

  // Compute workgroup sizes.  We use CL_KERNEL_WORK_GROUP_SIZE as an upper bound, which
  // depends on the compiled kernel.  I assume that the results for k will be identical to k_small.
  size_t kernel_wg_size;
  status = clGetKernelWorkGroupInfo(k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "accu()");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset    = 0;
  const uword wavefront_size = get_rt().cl_rt.get_wavefront_size();

  uword total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  uword local_group_size = std::min(kernel_wg_size, total_num_threads);

  // Create auxiliary memory.
  const uword aux_size = std::ceil((total_num_threads + (local_group_size - 1)) / local_group_size);
  Mat<eT> aux(aux_size, 1);
  aux.zeros();
  Mat<eT> aux2;
  if (aux_size > 1)
    {
    const uword aux2_size = std::ceil((aux_size + (local_group_size - 1)) / local_group_size);
    aux2.zeros(aux2_size, 1);
    }

  runtime_t::cq_guard guard;

  dev_mem_t<eT> aux_mem = aux.get_dev_mem(false);
  dev_mem_t<eT> aux_mem2 = aux2.get_dev_mem(false);

  uword in_n_elem = n_elem;
  Mat<eT>* out = &aux;

  dev_mem_t<eT>* in_mem = &mem;
  dev_mem_t<eT>* out_mem = &aux_mem;

  do
    {
    runtime_t::adapt_uword dev_n_elem(in_n_elem);

    // We need to round total_num_threads up to the next power of 2.  (The kernel assumes this.)
    const uword pow2_group_size = (uword) std::pow(2.0f, std::ceil(std::log2((float) local_group_size)));
    const uword pow2_total_num_threads = (total_num_threads % pow2_group_size == 0) ? total_num_threads : ((total_num_threads / pow2_group_size) + 1) * pow2_group_size;

    // If the number of threads is less than the wavefront size, we need to use the small kernel.
    cl_kernel* k_use = (pow2_group_size <= wavefront_size) ? &k_small : &k;

    status |= clSetKernelArg(*k_use, 0, sizeof(cl_mem),               &(in_mem->cl_mem_ptr));
    status |= clSetKernelArg(*k_use, 1, dev_n_elem.size,              dev_n_elem.addr);
    status |= clSetKernelArg(*k_use, 2, sizeof(cl_mem),               &(out_mem->cl_mem_ptr));
    status |= clSetKernelArg(*k_use, 3, sizeof(eT) * pow2_group_size, NULL);

    status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), *k_use, k1_work_dim, &k1_work_offset, &pow2_total_num_threads, &pow2_group_size, 0, NULL, NULL);

    coot_check_cl_error(status, "accu()");

    if (total_num_threads <= local_group_size)
      {
      break;
      }

    // Set the input, number of elements, and auxiliary memory correctly for subsequent runs.
    in_n_elem = out->n_elem;
    if (in_mem == &mem)
      {
      in_mem = &aux_mem;
      out_mem = &aux_mem2;
      out = &aux2;
      }
    else
      {
      std::swap(in_mem, out_mem);
      out = (out == &aux) ? &aux2 : &aux;
      }

    // Now, compute sizes for the next iteration.
    total_num_threads = std::ceil(in_n_elem / std::max(1.0, (2 * std::ceil(std::log2(in_n_elem)))));
    local_group_size = std::min(kernel_wg_size, total_num_threads);

    } while (true); // The loop terminates in the middle.

  return eT((*out)[0]);
  }
