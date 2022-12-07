// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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

// Utility functions for generating random numbers via CUDA (cuRAND).

template<typename eT>
inline
void
fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // TODO: replace awful implementation

  // Generate n random numbers.
  eT* cpu_rand = new eT[n];

  std::mt19937 gen;
  std::uniform_real_distribution<eT> u_distr;
  for (uword i = 0; i < n; ++i)
    {
    cpu_rand[i] = u_distr(gen);
    }

  // Now push it to the device.
  opencl::runtime_t::cq_guard guard;

  cl_int status = clEnqueueWriteBuffer(get_rt().cl_rt.get_cq(), dest.cl_mem_ptr, CL_TRUE, 0, sizeof(eT) * n, cpu_rand, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::fill_randu(): couldn't access device memory");

  delete cpu_rand;
  }



template<typename eT>
inline
void
fill_randn(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // TODO: replace awful implementation

  // Generate n random numbers.
  eT* cpu_rand = new eT[n];

  std::mt19937 gen;
  std::normal_distribution<eT> n_distr;
  for (uword i = 0; i < n; ++i)
    {
    cpu_rand[i] = n_distr(gen);
    }

  // Now push it to the device.
  opencl::runtime_t::cq_guard guard;

  cl_int status = clEnqueueWriteBuffer(get_rt().cl_rt.get_cq(), dest.cl_mem_ptr, CL_TRUE, 0, sizeof(eT) * n, cpu_rand, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::fill_randn(): couldn't access device memory");

  delete cpu_rand;
  }



// TODO: fill_randi, etc...
