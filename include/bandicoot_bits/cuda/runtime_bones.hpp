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

struct runtime_t
  {
  public:

  inline void init()
    {
    internal_init();
    }

  inline bool internal_init();

  template<typename eT>
  inline bool init_kernels(std::vector<CUfunction>& kernels, const std::string& source, const std::vector<std::string>& names);

  template<typename eT>
  inline CUfunction& get_kernel(const kernel_id::enum_id num);

  template<typename eT>
  inline eT* acquire_memory(const uword n_elem);

  template<typename eT>
  inline void release_memory(eT* cuda_mem);

  curandGenerator_t randGen;

  private:

  coot_aligned bool                     valid;

  coot_aligned std::vector<CUfunction>  u32_kernels;
  coot_aligned std::vector<CUfunction>  s32_kernels;
  coot_aligned std::vector<CUfunction>  u64_kernels;
  coot_aligned std::vector<CUfunction>  s64_kernels;
  coot_aligned std::vector<CUfunction>    f_kernels;
  coot_aligned std::vector<CUfunction>    d_kernels;

  CUdevice cuDevice;
  CUcontext context;
  };
