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



template<typename out_eT, typename T1>
inline
void
op_sum::apply(Mat<out_eT>& out, const Op<out_eT, T1, op_sum>& in)
  {
  coot_extra_debug_sigprint();

  // Attempt to apply any conv_to simplifications.
  op_sum::apply_after_conv_to(out, conv_to_preapply(in));
  }



template<typename out_eT, typename T1>
inline
void
op_sum::apply_after_conv_to(Mat<out_eT>& out, const Op<out_eT, T1, op_sum>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword_a;
  
  coot_debug_check( (dim > 1), "sum(): parameter 'dim' must be 0 or 1" );
  
  const unwrap<T1> U(in.m);
  
  if(U.is_alias(out) == false)
    {
    op_sum::apply_noalias(out, U.M, dim, (in.aux_uword_b == 0));
    }
  else
    {
    Mat<out_eT> tmp;
    
    op_sum::apply_noalias(tmp, U.M, dim, (in.aux_uword_b == 0));
    
    out.steal_mem(tmp);
    }
  }



template<typename out_eT, typename in_eT>
inline
void
op_sum::apply(Mat<out_eT>& out, const Op<out_eT, subview<in_eT>, op_sum>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword_a;
  
  coot_debug_check( (dim > 1), "sum(): parameter 'dim' must be 0 or 1" );
  
  Mat<out_eT> tmp;
    
  op_sum::apply_noalias(tmp, in.m, dim, (in.aux_uword_b == 0));
    
  out.steal_mem(tmp);
  }



template<typename eT>
inline
void
op_sum::apply(Mat<eT>& out, const Op<eT, subview<eT>, op_sum>& in)
  {
  coot_extra_debug_sigprint();

  const uword dim = in.aux_uword_a;
  
  coot_debug_check( (dim > 1), "sum(): parameter 'dim' must be 0 or 1" );
  
  if(&out != &(in.m.m))
    {
    op_sum::apply_noalias(out, in.m, dim, (in.aux_uword_b == 0));
    }
  else
    {
    Mat<eT> tmp;
    
    op_sum::apply_noalias(tmp, in.m, dim, (in.aux_uword_b == 0));
    
    out.steal_mem(tmp);
    }
  }



template<typename out_eT, typename in_eT>
inline
void
op_sum::apply_noalias(Mat<out_eT>& out, const Mat<in_eT>& A, const uword dim, const bool post_conv_apply)
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
    coot_rt_t::sum_colwise(out.get_dev_mem(false), A.get_dev_mem(false), A.n_rows, A.n_cols, post_conv_apply);
    }
  else
  if(dim == 1)
    {
    coot_rt_t::sum_rowwise(out.get_dev_mem(false), A.get_dev_mem(false), A.n_rows, A.n_cols, post_conv_apply);
    }
  }



template<typename out_eT, typename in_eT>
inline
void
op_sum::apply_noalias(Mat<out_eT>& out, const subview<in_eT>& sv, const uword dim, const bool post_conv_apply)
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
    coot_rt_t::sum_colwise_subview(out.get_dev_mem(false), sv.m.get_dev_mem(false), sv.m.n_rows, sv.aux_row1, sv.aux_col1, sv.n_rows, sv.n_cols, post_conv_apply);
    }
  else
  if(dim == 1)
    {
    coot_rt_t::sum_rowwise_subview(out.get_dev_mem(false), sv.m.get_dev_mem(false), sv.m.n_rows, sv.aux_row1, sv.aux_col1, sv.n_rows, sv.n_cols, post_conv_apply);
    }
  }



//! @}
