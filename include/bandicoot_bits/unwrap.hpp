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


//! \addtogroup unwrap
//! @{



template<typename T1>
struct unwrap
  {
  typedef typename T1::elem_type eT;
  typedef Mat<eT>                stored_type;
  
  inline
  unwrap(const T1& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }
  
  const Mat<eT> M;
  
  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>&) const { return false; }
  };



template<typename eT>
struct unwrap< Mat<eT> >
  {
  typedef Mat<eT> stored_type;
  
  inline
  unwrap(const Mat<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }
  
  const Mat<eT>& M;
  
  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return (void_ptr(&M) == void_ptr(&X)); }
  };



template<typename eT>
struct unwrap< Row<eT> >
  {
  typedef Row<eT> stored_type;
  
  inline
  unwrap(const Row<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }
  
  const Row<eT>& M;
  
  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return (void_ptr(&M) == void_ptr(&X)); }
  };



template<typename eT>
struct unwrap< Col<eT> >
  {
  typedef Col<eT> stored_type;

  inline
  unwrap(const Col<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }
  
  const Col<eT>& M;
  
  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return (void_ptr(&M) == void_ptr(&X)); }
  };



template<typename out_eT, typename T1>
struct unwrap< Op<out_eT, T1, op_conv_to> >
  {
  typedef Mat<out_eT> stored_type;
  
  inline
  unwrap(const Op<out_eT, T1, op_conv_to>& A)
    // The call to conv_to_preapply() will simplify any conv_to operations by merging them (if possible) with any other operations.
    : M(conv_to_preapply(A))
    {
    coot_extra_debug_sigprint();
    }
  
  const Mat<out_eT> M;
  
  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>&) const { return false; }
  };



//
//
//



template<typename T1>
struct partial_unwrap
  {
  typedef typename T1::elem_type eT;
  typedef Mat<eT>                stored_type;
  
  inline
  partial_unwrap(const T1& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }
  
  coot_inline eT get_val() const { return eT(1); }
  
  coot_inline bool is_alias(const Mat<eT>&) const { return false; }
  
  static const bool do_trans = false;
  static const bool do_times = false;
  
  const Mat<eT> M;
  };



template<typename eT>
struct partial_unwrap< Mat<eT> >
  {
  typedef Mat<eT> stored_type;
  
  inline
  partial_unwrap(const Mat<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }
  
  coot_inline eT get_val() const { return eT(1); }
  
  coot_inline bool is_alias(const Mat<eT>& X) const { return ((&X) == (&M)); }
  
  static const bool do_trans = false;
  static const bool do_times = false;
  
  const Mat<eT>& M;
  };



// Different behavior is needed depending on whether a type conversion is needed.
template<typename out_eT, typename T1>
struct partial_unwrap< Op<out_eT, T1, op_htrans> >
  {
  typedef out_eT                 eT;
  typedef typename T1::elem_type in_eT;
  typedef Mat<eT>                stored_type;
  
  inline
  partial_unwrap(const Op<out_eT, T1, op_htrans>& A)
    // The conv_to will be a no-op if in_eT == out_eT.
    : M(conv_to<Mat<eT>>::from(A.m))
    {
    coot_extra_debug_sigprint();
    }
  
  coot_inline out_eT get_val() const { return eT(1); }
  
  coot_inline bool is_alias(const Mat<out_eT>&) const { return false; }
  
  static const bool do_trans = true;
  static const bool do_times = false;
  
  const Mat<eT> M;
  };



template<typename eT>
struct partial_unwrap< Op< eT, Mat<eT>, op_htrans> >
  {
  typedef Mat<eT> stored_type;
  
  inline
  partial_unwrap(const Op< eT, Mat<eT>, op_htrans>& A)
    : M(A.m)
    {
    coot_extra_debug_sigprint();
    }
  
  coot_inline eT get_val() const { return eT(1); }
  
  coot_inline bool is_alias(const Mat<eT>& X) const { return (void_ptr(&X) == void_ptr(&M)); }
  
  static const bool do_trans = true;
  static const bool do_times = false;
  
  const Mat<eT>& M;
  };



template<typename out_eT, typename T1>
struct partial_unwrap< Op<out_eT, T1, op_htrans2> >
  {
  typedef out_eT  eT;
  typedef Mat<eT> stored_type;
  
  inline
  partial_unwrap(const Op<out_eT, T1, op_htrans2>& A)
    : val(A.aux)
    // if in_eT == out_eT, the conv_to<> will be a no-op
    , M  (conv_to<Mat<out_eT>>::from(A.m))
    {
    coot_extra_debug_sigprint();
    }
  
  coot_inline out_eT get_val() const { return val; }
  
  coot_inline bool is_alias(const Mat<out_eT>&) const { return false; }
  
  static const bool do_trans = true;
  static const bool do_times = true;
  
  const out_eT  val;
  const Mat<eT> M;
  };



template<typename out_eT, typename in_eT>
struct partial_unwrap< Op< out_eT, Mat<in_eT>, op_htrans2> >
  {
  typedef Mat<out_eT> stored_type;
  
  inline
  partial_unwrap(const Op<out_eT, Mat<in_eT>, op_htrans2>& A)
    : val(A.aux)
    // if in_eT == out_eT, the conv_to<> will be a no-op
    , M  (conv_to<Mat<out_eT>>::from(A.m))
    {
    coot_extra_debug_sigprint();
    }
  
  inline out_eT get_val() const { return val; }
  
  coot_hot coot_inline bool is_alias(const Mat<out_eT>& X) const { return (void_ptr(&X) == void_ptr(&M)); }
  
  static const bool do_trans = true;
  static const bool do_times = true;
  
  const out_eT      val;
  const Mat<in_eT>& M;
  };



// The behavior has to be different here depending on whether there is a type conversion.
// If there *is* a type conversion, we can't pull the scalar out.
template<typename out_eT, typename T1>
struct partial_unwrap< eOp<out_eT, T1, eop_scalar_times> >
  {
  typedef out_eT                 eT;
  typedef Mat<eT>                stored_type;
  typedef typename T1::elem_type in_eT;
  
  inline
  partial_unwrap(const eOp<out_eT, T1, eop_scalar_times>& A)
    : val(is_same_type<in_eT, out_eT>::value ? A.aux_in : out_eT(0))
    , M  (is_same_type<in_eT, out_eT>::value ? A.m.Q : conv_to<Mat<eT>>::from(A))
    {
    coot_extra_debug_sigprint();
    }
  
  coot_inline out_eT get_val() const { return val; }
  
  coot_inline bool is_alias(const Mat<out_eT>&) const { return false; }
  
  static const bool do_trans = false;
  static const bool do_times = is_same_type<in_eT, out_eT>::value;
  
  const out_eT  val;
  const Mat<eT> M;
  };



// This only applies if there is no type conversion.
template<typename eT>
struct partial_unwrap< eOp<eT, Mat<eT>, eop_scalar_times> >
  {
  typedef Mat<eT> stored_type;
  
  inline
  partial_unwrap(const eOp<eT, Mat<eT>, eop_scalar_times>& A)
    : val(A.aux_in)
    , M  (A.m.Q)
    {
    coot_extra_debug_sigprint();
    }
  
  inline eT get_val() const { return val; }
  
  coot_hot coot_inline bool is_alias(const Mat<eT>& X) const { return (void_ptr(&X) == void_ptr(&M)); }
  
  static const bool do_trans = false;
  static const bool do_times = true;
  
  const eT       val;
  const Mat<eT>& M;
  };



// Different behavior is needed if there is a type conversion.
template<typename out_eT, typename T1>
struct partial_unwrap< eOp<out_eT, T1, eop_neg> >
  {
  typedef out_eT                 eT;
  typedef typename T1::elem_type in_eT;
  typedef Mat<eT>                stored_type;
  
  inline
  partial_unwrap(const eOp<out_eT, T1, eop_neg>& A)
    : M(is_same_type<in_eT, out_eT>::value ? A.m.Q : conv_to<Mat<eT>>::from(A))
    {
    coot_extra_debug_sigprint();
    }
  
  coot_inline out_eT get_val() const { return is_same_type<in_eT, out_eT>::value ? out_eT(-1) : out_eT(0); }
  
  coot_inline bool is_alias(const Mat<out_eT>&) const { return false; }
  
  static const bool do_trans = false;
  static const bool do_times = is_same_type<in_eT, out_eT>::value;
  
  const Mat<eT> M;
  };



template<typename eT>
struct partial_unwrap< eOp<eT, Mat<eT>, eop_neg> >
  {
  typedef Mat<eT> stored_type;
  
  inline
  partial_unwrap(const eOp<eT, Mat<eT>, eop_neg>& A)
    : M(A.m.Q)
    {
    coot_extra_debug_sigprint();
    }
  
  coot_inline eT get_val() const { return eT(-1); }
  
  coot_inline bool is_alias(const Mat<eT>& X) const { return (void_ptr(&X) == void_ptr(&M)); }
  
  static const bool do_trans = false;
  static const bool do_times = true;
  
  const Mat<eT>& M;
  };



//! @}
