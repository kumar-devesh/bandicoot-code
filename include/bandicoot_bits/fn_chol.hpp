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


//! \addtogroup fn_chol
//! @{



// TODO: add optional 'layout' argument
template<typename T1>
inline
bool
chol(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type, T1>& X)
  {
  coot_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot_cl_rt not valid" );
  
  out = X.get_ref();
  
  // TODO: check whether given matrix is square
  
  magma_int_t info   = 0;
  magma_int_t status = 0;
  
  // using MAGMA 2.2
  
  // OpenCL uses opaque memory pointers which hide the underlying type,
  // so we don't need to do template tricks or casting
  
  if(is_float<eT>::value)
    {
    //std::cout << "using float" << std::endl;
    status = magma_spotrf_gpu(MagmaUpper, out.n_rows, out.get_dev_mem(), out.n_rows, &info);
    }
  else if(is_double<eT>::value)
    {
    //std::cout << "using double" << std::endl;
    status = magma_dpotrf_gpu(MagmaUpper, out.n_rows, out.get_dev_mem(), out.n_rows, &info);
    }
  else
    {
    coot_debug_check( true, "chol(): not implemented for given type" );
    }
  
  
  
  
  //// using MAGMA 1.3
  //status = magma_dpotrf_gpu(MagmaUpper, out.n_rows, out.get_dev_mem(), 0, out.n_rows, get_rt().cl_rt.get_cq(), &info);
  
  

  // TODO: check status
  
  // TODO: need to set the lower/upper triangular part (excluding the diagonal) to zero
  
  return true;
  }



//! @}
