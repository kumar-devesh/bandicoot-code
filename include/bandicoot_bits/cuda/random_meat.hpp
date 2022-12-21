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

// TODO: allow setting the seed!


template<typename eT>
inline
void
fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // For integral types, truncate to [0, 1] just like Armadillo.
  // We'll generate numbers using a floating-point type of the same width, then pass over it to truncate and cast back to the right type.
  if (std::is_same<eT, u32>::value || std::is_same<eT, s32>::value)
    {
    dev_mem_t<float> reinterpreted_mem;
    reinterpreted_mem.cuda_mem_ptr = (float*) dest.cuda_mem_ptr;
    fill_randu(reinterpreted_mem, n);
    copy_array(dest, reinterpreted_mem, n);
    }
  else if (std::is_same<eT, u64>::value || std::is_same<eT, s64>::value)
    {
    dev_mem_t<double> reinterpreted_mem;
    reinterpreted_mem.cuda_mem_ptr = (double*) dest.cuda_mem_ptr;
    fill_randu(reinterpreted_mem, n);
    copy_array(dest, reinterpreted_mem, n);
    }
  else
    {
    std::ostringstream oss;
    oss << "coot::cuda::fill_randu(): not implemented for type " << typeid(eT).name();
    coot_stop_runtime_error(oss.str());
    }
  }



template<>
inline
void
fill_randu(dev_mem_t<float> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = curandGenerateUniform(get_rt().cuda_rt.xorwow_rand, dest.cuda_mem_ptr, n);
  coot_check_curand_error(result, "coot::cuda::fill_randu(): curandGenerateUniform() failed");
  }



template<>
inline
void
fill_randu(dev_mem_t<double> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = curandGenerateUniformDouble(get_rt().cuda_rt.xorwow_rand, dest.cuda_mem_ptr, n);
  coot_check_curand_error(result, "coot::cuda::fill_randu(): curandGenerateUniform() failed");
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

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<eT> n_distr;
  for (uword i = 0; i < n; ++i)
    {
    cpu_rand[i] = n_distr(gen);
    }

  // Now push it to the device.
  cudaError_t result = cudaMemcpy(dest.cuda_mem_ptr, cpu_rand, n * sizeof(eT), cudaMemcpyHostToDevice);

  coot_check_cuda_error(result, "coot::cuda::fill_randu(): couldn't access device memory");

  delete[] cpu_rand;
  }



template<>
inline
void
fill_randn(dev_mem_t<float> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  curandGenerateNormal(get_rt().cuda_rt.philox_rand, dest.cuda_mem_ptr, n, 0.0, 1.0);
  }



template<typename eT>
inline
void
fill_randi(dev_mem_t<eT> dest, const uword n, const int a, const int b)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // This generic implementation for any type is a joke!
  // TODO: replace it with something that doesn't suck

  // Generate n random numbers.
  eT* cpu_rand = new eT[n];

  std::mt19937 gen;
  std::uniform_real_distribution<float> n_distr;
  for (uword i = 0; i < n; ++i)
    {
    cpu_rand[i] = abs(ceilf(n_distr(gen) * (b - a) + a));
    }

  // Now push it to the device.
  cudaError_t result = cudaMemcpy(dest.cuda_mem_ptr, cpu_rand, n * sizeof(eT), cudaMemcpyHostToDevice);

  coot_check_cuda_error(result, "cuda::fill_randi(): couldn't access device memory");

  delete cpu_rand;
  }

// TODO: fill_randi, etc...
