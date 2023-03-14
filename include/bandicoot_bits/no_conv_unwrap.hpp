// Copyright 2020 Ryan Curtin (https://www.ratml.org/)
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



// A version of unwrap<> that avoids a final type conversion if possible.
// It does do type conversions if they are needed for an intermediate operation.
// This is useful for operations that can accept a different input type.

template<typename T1>
struct no_conv_unwrap
  {
  typedef typename T1::elem_type eT;
  typedef Mat<eT>                stored_type;

  // This is the type that unwrap<> should use if it's internally using a no_conv_unwrap.
  typedef Mat<eT>& full_unwrap_type;

  inline
  no_conv_unwrap(const T1& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Mat<eT> M;

  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>&) const { return false; }
  };



template<typename eT>
struct no_conv_unwrap< Mat<eT> >
  {
  typedef Mat<eT> stored_type;

  // This is the type that unwrap<> should use if it's internally using a no_conv_unwrap.
  typedef Mat<eT>& full_unwrap_type;

  inline
  no_conv_unwrap(const Mat<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Mat<eT>& M;

  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return (void_ptr(&M) == void_ptr(&X)); }
  };



template<typename eT>
struct no_conv_unwrap< Row<eT> >
  {
  typedef Row<eT> stored_type;

  // This is the type that unwrap<> should use if it's internally using a no_conv_unwrap.
  typedef Row<eT>& full_unwrap_type;

  inline
  no_conv_unwrap(const Row<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Row<eT>& M;

  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return (void_ptr(&M) == void_ptr(&X)); }
  };



template<typename eT>
struct no_conv_unwrap< Col<eT> >
  {
  typedef Col<eT> stored_type;

  // This is the type that unwrap<> should use if it's internally using a no_conv_unwrap.
  typedef Col<eT>& full_unwrap_type;

  inline
  no_conv_unwrap(const Col<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Col<eT>& M;

  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return (void_ptr(&M) == void_ptr(&X)); }
  };



// We can do special overloads for mtop_conv_to, since there is actually no operation there.
//
// If we get a non-operation as input, *delay* the conversion.
// Unfortunately, this is a bit verbose, as we have one specialization for each underlying type.



template<typename out_eT, typename in_eT>
struct no_conv_unwrap< mtOp<out_eT, Mat< in_eT >, mtop_conv_to> >
  {
  typedef Mat<in_eT> stored_type;

  // This is the type that unwrap<> should use if it's internally using a no_conv_unwrap.
  typedef Mat<out_eT> full_unwrap_type;

  inline
  no_conv_unwrap(const mtOp<out_eT, Mat<in_eT>, mtop_conv_to>& A)
    : M(A.q)
    {
    coot_extra_debug_sigprint();
    }

  const Mat<in_eT>& M;

  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return (void_ptr(&M) == void_ptr(&X)); }
  };



template<typename out_eT, typename in_eT>
struct no_conv_unwrap< mtOp<out_eT, Col< in_eT >, mtop_conv_to> >
  {
  typedef Col<in_eT> stored_type;

  // This is the type that unwrap<> should use if it's internally using a no_conv_unwrap.
  typedef Col<out_eT> full_unwrap_type;

  inline
  no_conv_unwrap(const mtOp<out_eT, Col<in_eT>, mtop_conv_to>& A)
    : M(A.q)
    {
    coot_extra_debug_sigprint();
    }

  const Col<in_eT>& M;

  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return (void_ptr(&M) == void_ptr(&X)); }
  };



template<typename out_eT, typename in_eT>
struct no_conv_unwrap< mtOp<out_eT, Row< in_eT >, mtop_conv_to> >
  {
  typedef Row<in_eT> stored_type;

  // This is the type that unwrap<> should use if it's internally using a no_conv_unwrap.
  typedef Row<out_eT> full_unwrap_type;

  inline
  no_conv_unwrap(const mtOp<out_eT, Row<in_eT>, mtop_conv_to>& A)
    : M(A.q)
    {
    coot_extra_debug_sigprint();
    }

  const Row<in_eT>& M;

  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return (void_ptr(&M) == void_ptr(&X)); }
  };



// Any T1 will be an expression (due to the specializations above), so we ask the T1 to unwrap itself into
// the appropriate output type.
template<typename out_eT, typename T1>
struct no_conv_unwrap< mtOp<out_eT, T1, mtop_conv_to> >
  {
  typedef Mat<out_eT> stored_type;

  // This is the type that unwrap<> should use if it's internally using a no_conv_unwrap.
  typedef Mat<out_eT>& full_unwrap_type;

  inline
  no_conv_unwrap(const mtOp<out_eT, T1, mtop_conv_to>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Mat<out_eT> M;

  template<typename eT2>
  coot_inline bool is_alias(const Mat<eT2>& X) const { return false; }
  };
