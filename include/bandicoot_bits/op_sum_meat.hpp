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


//! \addtogroup op_sum
//! @{



template<typename T1>
inline
void
op_sum::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_sum>& in)
  {
  coot_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  const uword dim = in.aux_uword_a;
  
  coot_debug_check( (dim > 1), "sum(): parameter 'dim' must be 0 or 1" );
  
  const unwrap<T1> U(in.m);
  
  if(U.is_alias(out) == false)
    {
    op_sum::apply_noalias(out, U.M, dim);
    }
  else
    {
    Mat<eT> tmp;
    
    op_sum::apply_noalias(tmp, U.M, dim);
    
    out.steal_mem(tmp);
    }
  }



template<typename eT>
inline
void
op_sum::apply(Mat<eT>& out, const Op<subview<eT>,op_sum>& in)
  {
  coot_extra_debug_sigprint();
  
  const uword dim = in.aux_uword_a;
  
  coot_debug_check( (dim > 1), "sum(): parameter 'dim' must be 0 or 1" );
  
  if(&out != &(in.m.m))
    {
    op_sum::apply_noalias(out, in.m, dim);
    }
  else
    {
    Mat<eT> tmp;
    
    op_sum::apply_noalias(tmp, in.m, dim);
    
    out.steal_mem(tmp);
    }
  }



template<typename eT>
inline
void
op_sum::apply_noalias(Mat<eT>& out, const Mat<eT>& A, const uword dim)
  {
  coot_extra_debug_sigprint();
  
  if(dim == 0)
    {
    out.set_size(1, A.n_cols);
    }
  else
  if(dim == 1)
    {
    out.set_size(A.n_rows, 1);
    }
  
  if(A.n_elem == 0)
    {
    out.zeros();
    return;
    }
  
  
  if(dim == 0)
    {
    if (get_rt().backend == CL_BACKEND)
      {
      opencl::sum_colwise(out.get_dev_mem(false), A.get_dev_mem(false), A.n_rows, A.n_cols);
      }
    else
      {
      cuda::sum_colwise(out.get_dev_mem(false), A.get_dev_mem(false), A.n_rows, A.n_cols);
      }
    }
  else
  if(dim == 1)
    {
    if (get_rt().backend == CL_BACKEND)
      {
      opencl::sum_rowwise(out.get_dev_mem(false), A.get_dev_mem(false), A.n_rows, A.n_cols);
      }
    else
      {
      cuda::sum_rowwise(out.get_dev_mem(false), A.get_dev_mem(false), A.n_rows, A.n_cols);
      }
    }
  }



template<typename eT>
inline
void
op_sum::apply_noalias(Mat<eT>& out, const subview<eT>& sv, const uword dim)
  {
  coot_extra_debug_sigprint();
  
  if(dim == 0)
    {
    out.set_size(1, sv.n_cols);
    }
  else
  if(dim == 1)
    {
    out.set_size(sv.n_rows, 1);
    }
  
  if(sv.n_elem == 0)
    {
    out.zeros();
    return;
    }
  
  
  if(dim == 0)
    {
    if (get_rt().backend == CL_BACKEND)
      {
      opencl::sum_colwise_subview(out.get_dev_mem(false), sv.m.get_dev_mem(false), sv.m.n_rows, sv.aux_row1, sv.aux_col1, sv.n_rows, sv.n_cols);
      }
    else
      {
      cuda::sum_colwise_subview(out.get_dev_mem(false), sv.m.get_dev_mem(false), sv.m.n_rows, sv.aux_row1, sv.aux_col1, sv.n_rows, sv.n_cols);
      }
    }
  else
  if(dim == 1)
    {
    if (get_rt().backend == CL_BACKEND)
      {
      opencl::sum_rowwise_subview(out.get_dev_mem(false), sv.m.get_dev_mem(false), sv.m.n_rows, sv.aux_row1, sv.aux_col1, sv.n_rows, sv.n_cols);
      }
    else
      {
      cuda::sum_rowwise_subview(out.get_dev_mem(false), sv.m.get_dev_mem(false), sv.m.n_rows, sv.aux_row1, sv.aux_col1, sv.n_rows, sv.n_cols);
      }
    }
  }



//! @}
