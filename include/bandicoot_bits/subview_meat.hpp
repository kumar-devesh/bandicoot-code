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


//! \addtogroup subview
//! @{



template<typename eT>
coot_inline
uword
subview<eT>::get_n_rows() const
  {
  return n_rows;
  }



template<typename eT>
coot_inline
uword
subview<eT>::get_n_cols() const
  {
  return n_cols;
  }



template<typename eT>
coot_inline
uword
subview<eT>::get_n_elem() const
  {
  return n_elem;
  }



template<typename eT>
inline
subview<eT>::~subview()
  {
  coot_extra_debug_sigprint();
  }



template<typename eT>
inline
subview<eT>::subview(const Mat<eT>& in_m, const uword in_row1, const uword in_col1, const uword in_n_rows, const uword in_n_cols)
  : m(in_m)
  , aux_row1(in_row1)
  , aux_col1(in_col1)
  , n_rows(in_n_rows)
  , n_cols(in_n_cols)
  , n_elem(in_n_rows*in_n_cols)
  {
  coot_extra_debug_sigprint();
  }



template<typename eT>
inline
void
subview<eT>::operator= (const subview<eT>& x)
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution; replace with code using a dedicated kernel
  
  const Mat<eT> tmp(x);
  
  (*this).operator=(tmp);
  
  
  // if(check_overlap(x))
  //   {
  //   const Mat<eT> tmp(x);
  //   
  //   (*this).operator=(tmp);
  //   }
  // else
  //   {
  //   // TODO: implement kernel to copy from submatrix to submatrix
  //   }
  }



template<typename eT>
inline
void
subview<eT>::inplace_op(const eT val, oneway_kernel_id::enum_id kernel)
  {
  coot_extra_debug_sigprint();

  if(n_elem == 0)  { return; }

  coot_rt_t::inplace_op_subview(m.dev_mem, val, aux_row1, aux_col1, n_rows, n_cols, m.n_rows, kernel);
  }



template<typename eT>
inline
void
subview<eT>::operator= (const eT val)
  {
  coot_extra_debug_sigprint();
  
  if(n_elem == 1)
    {
    Mat<eT>& X = const_cast< Mat<eT>& >(m);
    
    X.at(aux_row1, aux_col1) = val;
    }
  else
    {
    coot_debug_assert_same_size(n_rows, n_cols, 1, 1, "subview::operator=");
    }
  }



template<typename eT>
inline
void
subview<eT>::operator+= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(val, oneway_kernel_id::submat_inplace_plus_scalar);
  }



template<typename eT>
inline
void
subview<eT>::operator-= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(val, oneway_kernel_id::submat_inplace_minus_scalar);
  }



template<typename eT>
inline
void
subview<eT>::operator*= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(val, oneway_kernel_id::submat_inplace_mul_scalar);
  }



template<typename eT>
inline
void
subview<eT>::operator/= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(val, oneway_kernel_id::submat_inplace_div_scalar);
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::inplace_op(const Base<eT, T1>& in, twoway_kernel_id::enum_id num, const char* identifier)
  {
  coot_extra_debug_sigprint();

  const no_conv_unwrap<T1> U(in.get_ref());
  const typename no_conv_unwrap<T1>::stored_type X = U.M;

  coot_assert_same_size(n_rows, n_cols, X.n_rows, X.n_cols, identifier);

  if(n_elem == 0)  { return; }

  coot_rt_t::inplace_op_subview(m.get_dev_mem(false), X.get_dev_mem(false), m.n_rows, aux_row1, aux_col1, X.n_rows, X.n_cols, num, identifier);
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator= (const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();

  // TODO: the code below uses the submat_inplace_set_mat kernel, but it may be faster to use the commented-out code with clEnqueueCopyBufferRect() with the OpenCL backend

  inplace_op(in, twoway_kernel_id::submat_inplace_set_mat, "subview::operator=()");
    
  /*
  const unwrap<T1>   U(in.get_ref());
  const Mat<eT>& X = U.M;
    
  coot_assert_same_size(n_rows, n_cols, X.n_rows, X.n_cols, "subview::operator=");
    
  // if the entire range is selected, use simple copy
  // (beignet 1.3 crashes if clEnqueueCopyBufferRect() is used on entire range)
  if( (n_rows == m.n_rows) && (n_cols == m.n_cols) )
    {
    Mat<eT>& mm = const_cast< Mat<eT>& >(m);
    m = in.get_ref();
    return;
    }
    
  size_t src_origin[3] = { 0, 0, 0 };
  size_t dst_origin[3] = { aux_row1*sizeof(eT), aux_col1, 0 };
    
  size_t region[3] = { n_rows*sizeof(eT), n_cols, 1 };
    
  size_t src_row_pitch   = 0;
  size_t src_slice_pitch = 0;
    
  size_t dst_row_pitch   = sizeof(eT) * m.n_rows;
  size_t dst_slice_pitch = sizeof(eT) * m.n_cols * m.n_rows;
    
  cl_int status = clEnqueueCopyBufferRect(get_rt().cl_rt.get_cq(), X.dev_mem, m.dev_mem, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, 0, NULL, NULL);
    
  coot_check_runtime_error( (status != 0), "subview::extract: couldn't copy buffer" );
  */
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator+= (const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();
  
  inplace_op(in, twoway_kernel_id::submat_inplace_plus_mat, "subview::operator+=()");
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator-= (const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();
  
  inplace_op(in, twoway_kernel_id::submat_inplace_minus_mat, "subview::operator-=()");
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator%= (const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();
  
  inplace_op(in, twoway_kernel_id::submat_inplace_schur_mat, "subview::operator%=()");
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator/= (const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();
  
  inplace_op(in, twoway_kernel_id::submat_inplace_div_mat, "subview::operator/=()");
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::inplace_op(const mtOp<eT, T1, mtop_conv_to>& x, twoway_kernel_id::enum_id num, const char* identifier)
  {
  coot_extra_debug_sigprint();

  // Avoid explicitly performing the conv_to so we can incorporate it into our operation here.
  const no_conv_unwrap<T1>   U(x.m.Q);
  const Mat<typename T1::elem_type>& X = U.M;

  coot_assert_same_size(n_rows, n_cols, X.n_rows, X.n_cols, identifier);

  if(n_elem == 0)  { return; }

  coot_rt_t::inplace_op_subview(m.get_dev_mem(false), X.get_dev_mem(false), m.n_rows, aux_row1, aux_col1, X.n_rows, X.n_cols, num, identifier);
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator= (const mtOp<eT, T1, mtop_conv_to>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(x, twoway_kernel_id::submat_inplace_set_mat, "subview::operator=()");
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator+=(const mtOp<eT, T1, mtop_conv_to>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(x, twoway_kernel_id::submat_inplace_plus_mat, "subview::operator+=()");
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator-=(const mtOp<eT, T1, mtop_conv_to>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(x, twoway_kernel_id::submat_inplace_minus_mat, "subview::operator-=()");
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator%=(const mtOp<eT, T1, mtop_conv_to>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(x, twoway_kernel_id::submat_inplace_schur_mat, "subview::operator%=()");
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator/=(const mtOp<eT, T1, mtop_conv_to>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(x, twoway_kernel_id::submat_inplace_div_mat, "subview::operator/=()");
  }



template<typename eT>
inline
void
subview<eT>::clamp(const eT min_val, const eT max_val)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );

  // TODO: this implementation could be improved!
  Mat<eT> tmp;
  op_clamp::apply_direct(tmp, *this, min_val, max_val);
  *this = tmp;
  }



template<typename eT>
inline
void
subview<eT>::fill(const eT val)
  {
  coot_extra_debug_sigprint();
  
  (*this).inplace_op(val, oneway_kernel_id::submat_inplace_set_scalar);
  }



template<typename eT>
inline
void
subview<eT>::zeros()
  {
  coot_extra_debug_sigprint();
  
  (*this).fill(eT(0));
  }



template<typename eT>
inline
void
subview<eT>::ones()
  {
  coot_extra_debug_sigprint();
  
  (*this).fill(eT(1));
  }



template<typename eT>
inline
void
subview<eT>::eye()
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution; replace with code using a dedicated kernel
  
  Mat<eT> tmp(n_rows, n_cols);
  tmp.eye();
  
  (*this).operator=(tmp);
  }



template<typename eT>
inline
bool
subview<eT>::check_overlap(const subview<eT>& x) const
  {
  const subview<eT>& s = *this;
  
  if(&s.m != &x.m)
    {
    return false;
    }
  else
    {
    if( (s.n_elem == 0) || (x.n_elem == 0) )
      {
      return false;
      }
    else
      {
      const uword s_row_start  = s.aux_row1;
      const uword s_row_end_p1 = s_row_start + s.n_rows;
      
      const uword s_col_start  = s.aux_col1;
      const uword s_col_end_p1 = s_col_start + s.n_cols;
      
      
      const uword x_row_start  = x.aux_row1;
      const uword x_row_end_p1 = x_row_start + x.n_rows;
      
      const uword x_col_start  = x.aux_col1;
      const uword x_col_end_p1 = x_col_start + x.n_cols;
      
      
      const bool outside_rows = ( (x_row_start >= s_row_end_p1) || (s_row_start >= x_row_end_p1) );
      const bool outside_cols = ( (x_col_start >= s_col_end_p1) || (s_col_start >= x_col_end_p1) );
      
      return ( (outside_rows == false) && (outside_cols == false) );
      }
    }
  }



template<typename eT>
coot_warn_unused
inline
bool
subview<eT>::is_vec() const
  {
  return ( (n_rows == 1) || (n_cols == 1) );
  }



template<typename eT>
coot_warn_unused
inline
bool
subview<eT>::is_colvec() const
  {
  return (n_cols == 1);
  }



template<typename eT>
coot_warn_unused
inline
bool
subview<eT>::is_rowvec() const
  {
  return (n_rows == 1);
  }



template<typename eT>
coot_warn_unused
inline
bool
subview<eT>::is_square() const
  {
  return (n_rows == n_cols);
  }



template<typename eT>
coot_warn_unused
inline
bool
subview<eT>::is_empty() const
  {
  return (n_elem == 0);
  }



//! X = Y.submat(...)
template<typename eT>
template<typename eT1>
inline
void
subview<eT>::extract(Mat<eT1>& out, const subview<eT>& in)
  {
  coot_extra_debug_sigprint();
  
  // NOTE: we're assuming that the matrix has already been set to the correct size and there is no aliasing;
  // size setting and alias checking is done by either the Mat contructor or operator=()
  
  coot_extra_debug_print(coot_str::format("out.n_rows = %d   out.n_cols = %d    in.m.n_rows = %d  in.m.n_cols = %d") % out.n_rows % out.n_cols % in.m.n_rows % in.m.n_cols );
  
  if(in.n_elem == 0)  { return; }
  
  // if the entire range is selected, use simple copy
  // (beignet 1.3 crashes if clEnqueueCopyBufferRect() is used on entire range)
  if( (in.n_rows == in.m.n_rows) && (in.n_cols == in.m.n_cols) )
    {
    out = in.m;
    return;
    }

  coot_rt_t::copy_subview(out.get_dev_mem(false), in.m.get_dev_mem(false), in.aux_row1, in.aux_col1, in.m.n_rows, in.m.n_cols, in.n_rows, in.n_cols);
  
//  size_t src_origin[3] = { in.aux_row1*sizeof(eT), in.aux_col1, 0 };
//  size_t dst_origin[3] = { 0, 0, 0 };
  
//  size_t region[3] = { in.n_rows*sizeof(eT), in.n_cols, 1 };
  
//  size_t src_row_pitch   = sizeof(eT) * in.m.n_rows;
//  size_t src_slice_pitch = sizeof(eT) * in.m.n_cols * in.m.n_rows;
  
//  size_t dst_row_pitch   = 0;
//  size_t dst_slice_pitch = 0;
  
//  cl_int status = clEnqueueCopyBufferRect(get_rt().cl_rt.get_cq(), in.m.dev_mem, out.dev_mem, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, 0, NULL, NULL);
  
//  coot_check_runtime_error( (status != 0), "subview::extract: couldn't copy buffer" );
  }



//! X += Y.submat(...)
template<typename eT>
template<typename eT1>
inline
void
subview<eT>::plus_inplace(Mat<eT1>& out, const subview<eT>& in)
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution; replace with code using a dedicated kernel
  
  const Mat<eT> tmp(in);
  
  out += tmp;
  }



//! X -= Y.submat(...)
template<typename eT>
template<typename eT1>
inline
void
subview<eT>::minus_inplace(Mat<eT1>& out, const subview<eT>& in)
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution; replace with code using a dedicated kernel
  
  const Mat<eT> tmp(in);
  
  out -= tmp;
  }



//! X %= Y.submat(...)
template<typename eT>
template<typename eT1>
inline
void
subview<eT>::schur_inplace(Mat<eT1>& out, const subview<eT>& in)
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution; replace with code using a dedicated kernel
  
  const Mat<eT> tmp(in);
  
  out %= tmp;
  }



//! X /= Y.submat(...)
template<typename eT>
template<typename eT1>
inline
void
subview<eT>::div_inplace(Mat<eT1>& out, const subview<eT>& in)
  {
  coot_extra_debug_sigprint();
  
  // TODO: this is currently a "better-than-nothing" solution; replace with code using a dedicated kernel
  
  const Mat<eT> tmp(in);
  
  out /= tmp;
  }



//
// subview_col


template<typename eT>
coot_inline
uword
subview_col<eT>::get_n_cols() const
  {
  return uword(1);
  }



template<typename eT>
inline
subview_col<eT>::subview_col(const Mat<eT>& in_m, const uword in_col)
  : subview<eT>(in_m, 0, in_col, in_m.n_rows, 1)
  {
  coot_extra_debug_sigprint();
  }



template<typename eT>
inline
subview_col<eT>::subview_col(const Mat<eT>& in_m, const uword in_col, const uword in_row1, const uword in_n_rows)
  : subview<eT>(in_m, in_row1, in_col, in_n_rows, 1)
  {
  coot_extra_debug_sigprint();
  }



template<typename eT>
inline
void
subview_col<eT>::operator=(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  
  subview<eT>::operator=(X);
  }



template<typename eT>
inline
void
subview_col<eT>::operator=(const subview_col<eT>& X)
  {
  coot_extra_debug_sigprint();
  
  subview<eT>::operator=(X); // interprets 'subview_col' as 'subview'
  }



template<typename eT>
inline
void
subview_col<eT>::operator=(const eT val)
  {
  coot_extra_debug_sigprint();
  
  subview<eT>::operator=(val); // interprets 'subview_col' as 'subview'
  }



template<typename eT>
template<typename T1>
inline
void
subview_col<eT>::operator=(const Base<eT,T1>& X)
  {
  coot_extra_debug_sigprint();
  
  subview<eT>::operator=(X); // interprets 'subview_col' as 'subview'
  }



//
// subview_row


template<typename eT>
coot_inline
uword
subview_row<eT>::get_n_rows() const
  {
  return uword(1);
  }



template<typename eT>
inline
subview_row<eT>::subview_row(const Mat<eT>& in_m, const uword in_row)
  : subview<eT>(in_m, in_row, 0, 1, in_m.n_cols)
  {
  coot_extra_debug_sigprint();
  }



template<typename eT>
inline
subview_row<eT>::subview_row(const Mat<eT>& in_m, const uword in_row, const uword in_col1, const uword in_n_cols)
  : subview<eT>(in_m, in_row, in_col1, 1, in_n_cols)
  {
  coot_extra_debug_sigprint();
  }



template<typename eT>
inline
void
subview_row<eT>::operator=(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();
  
  subview<eT>::operator=(X);
  }



template<typename eT>
inline
void
subview_row<eT>::operator=(const subview_row<eT>& X)
  {
  coot_extra_debug_sigprint();
  
  subview<eT>::operator=(X); // interprets 'subview_row' as 'subview'
  }



template<typename eT>
inline
void
subview_row<eT>::operator=(const eT val)
  {
  coot_extra_debug_sigprint();
  
  subview<eT>::operator=(val); // interprets 'subview_row' as 'subview'
  }



template<typename eT>
template<typename T1>
inline
void
subview_row<eT>::operator=(const Base<eT,T1>& X)
  {
  coot_extra_debug_sigprint();
  
  subview<eT>::operator=(X);
  }



//! @}
