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



// Determine whether all elements in the memory satisfy the conditions imposed by the kernel `num` (and its small version `num_small`).
template<typename eT1, typename eT2>
inline
bool
all_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::all_vec(): OpenCL runtime not valid" );

  cl_kernel k = get_rt().cl_rt.get_kernel<eT1, eT2>(num);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT1, eT2>(num_small);
  // Second (and later) passes use the "and" reduction.
  cl_kernel second_k = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce);
  cl_kernel second_k_small = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::and_reduce_small);

  u32 result = generic_reduce<eT1, u32>(mem,
                                        n_elem,
                                        "all",
                                        k,
                                        k_small,
                                        std::make_tuple(val),
                                        second_k,
                                        second_k_small,
                                        std::make_tuple(/* no extra args for second pass */));

  return (result == 0) ? false : true;
  }
