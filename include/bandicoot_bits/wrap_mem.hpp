// Copyright 2023 Ryan Curtin (http://ratml.org)
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



//
// convenience wrappers for the advanced constructors
//



struct cl_mem_wrapper
  {
  cl_mem m;
  };



inline
cl_mem_wrapper
wrap_mem_cl(cl_mem c)
  {
  coot_extra_debug_sigprint();

  cl_mem_wrapper result;
  result.m = c;
  return result;
  }



template<typename eT>
struct cuda_mem_wrapper
  {
  eT* m;
  };



template<typename eT>
inline
cuda_mem_wrapper<eT>
wrap_mem_cuda(eT* m)
  {
  coot_extra_debug_sigprint();

  cuda_mem_wrapper<eT> result;
  result.m = m;
  return result;
  }
