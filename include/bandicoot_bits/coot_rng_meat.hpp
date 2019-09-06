// Copyright 2019 Ryan Curtin <ryan@ratml.org>
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


template<typename eT>
inline void coot_rng::fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::fill_randu(dest, n);
    }
  else
    {
    cuda::fill_randu(dest, n);
    }
  }


template<typename eT>
inline void coot_rng::fill_randn(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::fill_randu(dest, n);
    }
  else
    {
    cuda::fill_randn(dest, n);
    }
  }
