// Copyright 2021-2023 Ryan Curtin (http://www.ratml.org)
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



// Utility to run a full reduce in an efficient way.
//
// Note that all kernels that are used with generic_reduce() are expected to return an eT,
// and the first four arguments should be:
//  - const __global eT1* mem
//  - const UWORD n_elem
//  - __global eT1* out_mem
//  - __local volatile eT1* aux_mem
// Additional arguments are fine, so long as those first four are the same.



template<typename T, typename... Args>
inline
uword
constexpr
count_uwords(const T& arg,
             const Args... args,
             typename enable_if<!std::is_same<T, uword>::value>::type* junk)
  {
  coot_ignore(arg);
  coot_ignore(junk);
  return count_uwords(args...);
  }



template<typename... Args>
inline
uword
constexpr
count_uwords(const uword arg,
             const Args... args)
  {
  coot_ignore(arg);
  return 1 + count_uwords(args...);
  }



inline
uword
constexpr
count_uwords()
  {
  return 0;
  }



template<typename T, typename... Args>
inline
cl_int
set_extra_args(cl_kernel& kernel,
               const uword index,
               runtime_t::adapt_uword* adapt_uwords,
               const uword adapt_uword_index,
               T& arg,
               Args... args,
               typename enable_if<!std::is_same<T, uword>::value>::type* junk = 0)
  {
  coot_ignore(junk);

  cl_int status = clSetKernelArg(kernel, index, sizeof(T), &arg);
  return status | set_extra_args(kernel, index + 1, adapt_uwords, adapt_uword_index, args...);
  }



template<typename... Args>
inline
cl_int
set_extra_args(cl_kernel& kernel,
               const uword index,
               runtime_t::adapt_uword* adapt_uwords,
               const uword adapt_uword_index,
               uword arg,
               Args... args)
  {
  adapt_uwords[adapt_uword_index] = runtime_t::adapt_uword(arg);
  cl_int status = clSetKernelArg(kernel, index, adapt_uwords[adapt_uword_index].size, adapt_uwords[adapt_uword_index].addr);
  return status | set_extra_args(kernel, index + 1, adapt_uwords, adapt_uword_index + 1, args...);
  }



inline
cl_int
set_extra_args(cl_kernel& kernel,
               const uword index,
               runtime_t::adapt_uword* adapt_uwords,
               const uword adapt_uword_index)
  {
  coot_ignore(kernel);
  coot_ignore(index);
  coot_ignore(adapt_uwords);
  coot_ignore(adapt_uword_index);
  return CL_SUCCESS;
  }



template<typename eT, typename... Args>
inline
eT
generic_reduce(cl_kernel& kernel,         // kernel for full-scale reduction passes
               cl_kernel& kernel_small,   // kernel for small reductions
               const char* kernel_name,   // for error reporting
               const dev_mem_t<eT> mem,   // initial memory we are reducing
               const uword n_elem,        // size of memory we are reducing
               Args... extra_args)        // any additional arguments to the reduce kernels; should be references, not pointers
  {
  // Compute workgroup sizes.  We use CL_KERNEL_WORK_GROUP_SIZE as an upper bound, which
  // depends on the compiled kernel.  I assume that the results for k will be identical to k_small.
  size_t kernel_wg_size;
  cl_int status = clGetKernelWorkGroupInfo(kernel, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  // TODO: should we multiply by CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE?
  coot_check_cl_error(status, kernel_name);

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset    = 0;
  const uword wavefront_size = get_rt().cl_rt.get_wavefront_size();

  uword total_num_threads = std::ceil(n_elem / (2 * std::ceil(std::log2(n_elem))));
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

  dev_mem_t<eT>* in_mem = (dev_mem_t<eT>*) &mem;
  dev_mem_t<eT>* out_mem = &aux_mem;

  do
    {
    runtime_t::adapt_uword dev_n_elem(in_n_elem);

    // We need to round total_num_threads up to the next power of 2.  (The kernel assumes this.)
    const uword pow2_group_size = (uword) std::pow(2.0f, std::ceil(std::log2((float) local_group_size)));
    const uword pow2_total_num_threads = (total_num_threads % pow2_group_size == 0) ? total_num_threads : ((total_num_threads / pow2_group_size) + 1) * pow2_group_size;

    // If the number of threads is less than the wavefront size, we need to use the small kernel.
    cl_kernel* k_use = (pow2_group_size <= wavefront_size) ? &kernel_small : &kernel;

    status |= clSetKernelArg(*k_use, 0, sizeof(cl_mem),               &(in_mem->cl_mem_ptr));
    status |= clSetKernelArg(*k_use, 1, dev_n_elem.size,              dev_n_elem.addr);
    status |= clSetKernelArg(*k_use, 2, sizeof(cl_mem),               &(out_mem->cl_mem_ptr));
    status |= clSetKernelArg(*k_use, 3, sizeof(eT) * pow2_group_size, NULL);

    // If we have any uwords in extra_args, we need to allocate adapt_uwords for them, which will be filled in set_extra_args().
    constexpr const uword num_uwords = count_uwords(extra_args...);
    runtime_t::adapt_uword adapt_uwords[num_uwords == 0 ? 1 : num_uwords];
    status |= set_extra_args(*k_use, 4, adapt_uwords, 0, extra_args...);

    status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), *k_use, k1_work_dim, &k1_work_offset, &pow2_total_num_threads, &pow2_group_size, 0, NULL, NULL);

    coot_check_cl_error(status, kernel_name);

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
    total_num_threads = std::ceil(in_n_elem / (2 * std::ceil(std::log2(in_n_elem))));
    local_group_size = std::min(kernel_wg_size, total_num_threads);

    } while (true); // The loop terminates in the middle.

  return eT((*out)[0]);
  }
