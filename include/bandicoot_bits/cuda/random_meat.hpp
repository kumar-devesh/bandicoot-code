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

  // This generic implementation for any type is a joke!
  // TODO: replace it with something that doesn't suck

  // Generate n random numbers.
  eT* cpu_rand = new eT[n];

  std::mt19937 gen;
  std::uniform_real_distribution<eT> u_distr(eT(0), eT(1));
  for (uword i = 0; i < n; ++i)
    {
    cpu_rand[i] = u_distr(gen);
    }

  // Now push it to the device.
  cudaError_t result = cudaMemcpy(dest.cuda_mem_ptr, cpu_rand, n * sizeof(eT), cudaMemcpyHostToDevice);

  coot_check_cuda_error(result, "cuda::fill_randu(): couldn't access device memory");

  delete[] cpu_rand;
  }



template<>
inline
void
fill_randu(dev_mem_t<float> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  curandGenerateUniform(get_rt().cuda_rt.randGen, dest.cuda_mem_ptr, n);
  }



template<typename eT>
inline
void
fill_randn(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // This generic implementation for any type is a joke!
  // TODO: replace it with something that doesn't suck

  // Generate n random numbers.
  eT* cpu_rand = new eT[n];

  std::mt19937 gen;
  std::normal_distribution<eT> n_distr;
  for (uword i = 0; i < n; ++i)
    {
    cpu_rand[i] = n_distr(gen);
    }

  // Now push it to the device.
  cudaError_t result = cudaMemcpy(dest.cuda_mem_ptr, cpu_rand, n * sizeof(eT), cudaMemcpyHostToDevice);

  coot_check_cuda_error(result, "cuda::fill_randu(): couldn't access device memory");

  delete[] cpu_rand;
  }



template<>
inline
void
fill_randn(dev_mem_t<float> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  curandGenerateNormal(get_rt().cuda_rt.randGen, dest.cuda_mem_ptr, n, 0.0, 1.0);
  }



// TODO: fill_randi, etc...
