// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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



// this can hold either CUDA memory or CL memory
template<typename eT>
union dev_mem_t
  {
  cl_mem cl_mem_ptr;
  eT* cuda_mem_ptr;
  };

enum coot_backend_t
  {
  CL_BACKEND = 0,
  CUDA_BACKEND 
  };

// TODO: if this is placed into a run-time library and executed there, what happens when two programs use the run-time library at the same time?
class coot_rt_t
  {
  public:

  coot_backend_t backend;

  // RC-TODO: what if the CL headers are not available?
  opencl::runtime_t cl_rt;
  cuda::runtime_t cuda_rt;

  inline ~coot_rt_t();
  inline  coot_rt_t();

  // RC-TODO: unified constructors?
  /*
  inline bool init(const bool print_info = false);
  inline bool init(const char*       filename, const bool print_info = false);
  inline bool init(const std::string filename, const bool print_info = false);
  inline bool init(const uword wanted_platform, const uword wanted_device, const bool print_info = false);
  */

  #if defined(COOT_USE_CXX11)
                   coot_rt_t(const coot_rt_t&) = delete;
  coot_rt_t&       operator=(const coot_rt_t&) = delete;
  #endif

  template<typename eT>
  inline dev_mem_t<eT> acquire_memory(const uword n_elem);
  
  template<typename eT>
  inline void release_memory(dev_mem_t<eT> dev_mem);

  inline void synchronize();

  // RC-TODO: unified interface for some other operations?
  };

// Store coot_rt_t as a singleton.
inline coot_rt_t& get_rt()
  {
  static coot_rt_t rt;
  return rt;
  }
