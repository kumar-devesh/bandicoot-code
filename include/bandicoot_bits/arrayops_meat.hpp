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


//! \addtogroup arrayops
//! @{



template<typename eT>
inline
void
arrayops::copy(cl_mem dest, cl_mem src, const uword n_elem)
  {
  // TODO: make work for CUDA
  coot_extra_debug_sigprint();
  
  opencl::runtime_t::cq_guard guard;
  
  coot_extra_debug_print("clEnqueueCopyBuffer()");
  
  cl_int status = clEnqueueCopyBuffer(coot_rt.cl_rt.get_cq(), src, dest, size_t(0), size_t(0), sizeof(eT)*size_t(n_elem), cl_uint(0), NULL, NULL);
  
  coot_check_runtime_error( (status != 0), "arrayops::copy(): couldn't copy buffer" );
  }



template<typename eT>
inline
void
arrayops::inplace_set_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (coot_rt.backend == CUDA_BACKEND)
    {
    cuda::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_set_scalar);
    }
  else
    {
    opencl::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_set_scalar);
    }
  }



template<typename eT>
inline
void
arrayops::inplace_plus_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (coot_rt.backend == CUDA_BACKEND)
    {
    cuda::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_plus_scalar);
    }
  else
    {
    opencl::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_plus_scalar);
    }
  }



template<typename eT>

inline
void
arrayops::inplace_minus_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (coot_rt.backend == CUDA_BACKEND)
    {
    cuda::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_minus_scalar);
    }
  else
    {
    opencl::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_minus_scalar);
    }
  }



template<typename eT>

inline
void
arrayops::inplace_mul_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (coot_rt.backend == CUDA_BACKEND)
    {
    cuda::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_mul_scalar);
    }
  else
    {
    opencl::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_mul_scalar);
    }
  }



template<typename eT>

inline
void
arrayops::inplace_div_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (coot_rt.backend == CUDA_BACKEND)
    {
    cuda::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_div_scalar);
    }
  else
    {
    opencl::inplace_op_scalar(dest, val, n_elem, kernel_id::inplace_div_scalar);
    }
  }



template<typename eT>
inline
void
arrayops::inplace_plus_array(dev_mem_t<eT> dest, dev_mem_t<eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (coot_rt.backend == CUDA_BACKEND)
    {
    cuda::inplace_op_array(dest, src, n_elem, kernel_id::inplace_plus_array);
    }
  else
    {
    opencl::inplace_op_array(dest, src, n_elem, kernel_id::inplace_plus_array);
    }
  }



template<typename eT>
inline
void
arrayops::inplace_minus_array(dev_mem_t<eT> dest, dev_mem_t<eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (coot_rt.backend == CUDA_BACKEND)
    {
    cuda::inplace_op_array(dest, src, n_elem, kernel_id::inplace_minus_array);
    }
  else
    {
    opencl::inplace_op_array(dest, src, n_elem, kernel_id::inplace_minus_array);
    }
  }



template<typename eT>
inline
void
arrayops::inplace_mul_array(dev_mem_t<eT> dest, dev_mem_t<eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (coot_rt.backend == CUDA_BACKEND)
    {
    cuda::inplace_op_array(dest, src, n_elem, kernel_id::inplace_mul_array);
    }
  else
    {
    opencl::inplace_op_array(dest, src, n_elem, kernel_id::inplace_mul_array);
    }
  }



template<typename eT>
inline
void
arrayops::inplace_div_array(dev_mem_t<eT> dest, dev_mem_t<eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (coot_rt.backend == CUDA_BACKEND)
    {
    cuda::inplace_op_array(dest, src, n_elem, kernel_id::inplace_div_array);
    }
  else
    {
    opencl::inplace_op_array(dest, src, n_elem, kernel_id::inplace_div_array);
    }
  }



template<typename eT>
inline
eT
arrayops::accumulate(cl_mem src, const uword n_elem)
  {
  // TODO
  }



template<typename eT>
inline
eT
arrayops::product(cl_mem src, const uword n_elem)
  {
  // TODO
  }



//! @}
