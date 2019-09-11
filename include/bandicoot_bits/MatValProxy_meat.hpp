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


//! \addtogroup MatValProxy
//! @{



template<typename eT>
coot_inline
MatValProxy<eT>::MatValProxy(Mat<eT>& in_M, const uword in_index)
  : M    (in_M    )
  , index(in_index)
  {
  coot_extra_debug_sigprint();
  }



template<typename eT>
coot_inline
MatValProxy<eT>::operator eT()
  {
  return MatValProxy<eT>::get_val(M, index);
  }



template<typename eT>
inline
eT
MatValProxy<eT>::get_val(const Mat<eT>& M, const uword index)
  {
  if (get_rt().backend == CL_BACKEND)
    {
    return opencl::get_val(M.dev_mem, index);
    }
  else
    {
    return cuda::get_val(M.dev_mem, index);
    }
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator=(const eT in_val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::set_val(M.dev_mem, index, in_val);
    }
  else
    {
    cuda::set_val(M.dev_mem, index, in_val);
    }
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator+=(const eT in_val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::val_add_inplace(M.dev_mem, index, in_val);
    }
  else
    {
    cuda::val_add_inplace(M.dev_mem, index, in_val);
    }
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator-=(const eT in_val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::val_minus_inplace(M.dev_mem, index, in_val);
    }
  else
    {
    cuda::val_minus_inplace(M.dev_mem, index, in_val);
    }
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator*=(const eT in_val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::val_mul_inplace(M.dev_mem, index, in_val);
    }
  else
    {
    cuda::val_mul_inplace(M.dev_mem, index, in_val);
    }
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator/=(const eT in_val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    opencl::val_div_inplace(M.dev_mem, index, in_val);
    }
  else
    {
    cuda::val_div_inplace(M.dev_mem, index, in_val);
    }
  }



//! @}
