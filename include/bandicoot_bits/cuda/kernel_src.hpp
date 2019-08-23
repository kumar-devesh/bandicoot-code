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

// utility functions for compiled-on-the-fly CUDA kernels

// shitty single kernel for fill()

inline
std::vector<std::string>
get_cuda_kernel_names()
  {
  std::vector<std::string> names;

  names.push_back("inplace_set_scalar");
  names.push_back("inplace_plus_scalar");
  names.push_back("inplace_minus_scalar");
  names.push_back("inplace_mul_scalar");
  names.push_back("inplace_div_scalar");

  return names;
  }

inline
std::string
get_cuda_kernel_src()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::string source = \

  "#define uint unsigned int\n"
  "#define COOT_FN2(ARG1, ARG2)  ARG1 ## ARG2 \n"
  "#define COOT_FN(ARG1,ARG2) COOT_FN2(ARG1,ARG2)\n"
  "\n"
  "extern \"C\" {\n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_set_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const size_t i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] = val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_plus_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const size_t i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] += val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_minus_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const size_t i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] -= val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_mul_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const size_t i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] *= val; } \n"
  "  } \n"
  "\n"
  "__global__ void COOT_FN(PREFIX,inplace_div_scalar)(eT* out, const eT val, const UWORD N) \n"
  "  { \n"
  "  const size_t i = blockIdx.x * blockDim.x + threadIdx.x; \n"
  "  if(i < N)  { out[i] /= val; } \n"
  "  } \n"
  "\n"
  "}\n";

  return source;
  }
