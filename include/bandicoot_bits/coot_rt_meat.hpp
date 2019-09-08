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



inline
coot_rt_t::~coot_rt_t()
  {
  coot_extra_debug_sigprint_this(this);
  }



inline
coot_rt_t::coot_rt_t()
  {
  coot_extra_debug_sigprint_this(this);
  backend = CL_BACKEND;
  }



template<typename eT>
inline
dev_mem_t<eT>
coot_rt_t::acquire_memory(const uword n_elem)
  {
  coot_extra_debug_sigprint();
  
//  coot_check_runtime_error( (valid == false), "coot_rt::acquire_memory(): runtime not valid" );
  
  if(n_elem == 0)  { return dev_mem_t<eT>({ .cl_mem_ptr = NULL }); }
  
  coot_debug_check
   (
   ( size_t(n_elem) > (std::numeric_limits<size_t>::max() / sizeof(eT)) ),
   "coot_rt::acquire_memory(): requested size is too large"
   );

  // use either OpenCL or CUDA backend
  dev_mem_t<eT> result;

  if (backend == CUDA_BACKEND)
    {
    result.cuda_mem_ptr = cuda_rt.acquire_memory<eT>(n_elem);
    }
  else
    {
    result.cl_mem_ptr = cl_rt.acquire_memory<eT>(n_elem);
    }

  return result;
  }



template<typename eT>
inline
void
coot_rt_t::release_memory(dev_mem_t<eT> dev_mem)
  {
  coot_extra_debug_sigprint();
  
//  coot_debug_check( (valid == false), "coot_rt not valid" );
  
  if (backend == CL_BACKEND)
    {
    cl_rt.release_memory(dev_mem.cl_mem_ptr);
    }
  else
    {
    cuda_rt.release_memory(dev_mem.cuda_mem_ptr);
    }
  }



inline
void
coot_rt_t::synchronize()
  {
  if (backend == CL_BACKEND)
    {
    cl_rt.synchronize();
    }
  else
    {
    cuda_rt.synchronize();
    }
  }
