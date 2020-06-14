// Copyright 2020 Ryan Curtin (http://www.ratml.org
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



// Empty struct for use with Op.
class op_conv_to { };



template<typename T2>
struct conv_to
  {
  // TODO: block this class for anything other than Mat, Row, and Col (and similar)
  typedef typename T2::elem_type out_eT;

  // Perform a conversion from one type to another.  This returns an mtOp,
  // because other operations in the codebase might be able to use a multi-type
  // kernel to reduce the number of kernels needed.
  template<typename T1>
  inline
  static
  typename enable_if2<
      !is_same_type<typename T1::elem_type, out_eT>::value,
      Op<out_eT, T1, op_conv_to>
  >::result
  from(const T1& in)
    {
    coot_extra_debug_sigprint();

    return Op<out_eT, T1, op_conv_to>(in);
    }



  // Dummy overload for when in_eT == out_eT---in this case, just return the
  // input; it's already the right type.
  template<typename T1>
  inline
  static
  typename enable_if2<
      is_same_type<out_eT, typename T1::elem_type>::value,
      T1
  >::result
  from(const T1& in)
    {
    coot_extra_debug_sigprint();

    return in;
    }



  // Overloads for existing Op, Glue, eOp, and eGlue



  template<typename in_eT, typename T1, typename op_type>
  inline
  static
  typename enable_if2<
      !is_same_type<out_eT, typename T1::elem_type>::value &&
      !is_same_type<op_type, op_conv_to>::value,
      Op<out_eT, T1, op_type>
  >::result
  from(const Op<in_eT, T1, op_type>& in)
    {
    coot_extra_debug_sigprint();

    return Op<out_eT, T1, op_type>(in);
    }




  template<typename in_eT, typename T1, typename eop_type>
  inline
  static
  typename enable_if2<
      !is_same_type<out_eT, typename T1::elem_type>::value,
      eOp<out_eT, T1, eop_type>
  >::result
  from(const eOp<out_eT, T1, eop_type>& in)
    {
    coot_extra_debug_sigprint();

    return eOp<out_eT, T1, eop_type>(in);
    }
  };



// Tools for simplifying chained operations with a conv_to inside of them.
//
// Note that because of the rules above, it's not possible that we can get
// a conv_to no-op (where in_eT == out_eT), so we don't need to consider that
// case.



// base case: no conv_to or no simplification possible
template<typename T1>
inline
static
const T1&
conv_to_preapply(const T1& in)
  {
  coot_extra_debug_sigprint();

  return in;
  }



template<typename out_eT, typename in_eT, typename T1, typename op_type>
inline
static
typename enable_if2<
    !is_same_type<op_type, op_conv_to>::value &&
    // make sure that no conversion is already happening
    is_same_type<in_eT, typename T1::elem_type>::value,
    Op<out_eT, T1, op_type>
>::result
conv_to_preapply(const Op<out_eT, Op<in_eT, T1, op_type>, op_conv_to>& in)
  {
  coot_extra_debug_sigprint();

  return Op<out_eT, T1, op_type>(in);
  }



// sums can be applied pre- or post-conversion
template<typename out_eT, typename T1>
inline
static
Op<out_eT, T1, op_sum>
conv_to_preapply(const Op<out_eT, Op<out_eT, T1, op_conv_to>, op_sum>& in)
  {
  coot_extra_debug_sigprint();

  return Op<out_eT, T1, op_sum>(in.m.m, in.aux_uword_a, 1); // 1 => apply sum pre-conversion
  }



template<typename out_eT, typename in_eT, typename T1, typename eop_type>
inline
static
typename enable_if2<
    // make sure that no conversion is already happening
    is_same_type<in_eT, typename T1::elem_type>::value,
    eOp<out_eT, T1, eop_type>
>::result
conv_to_preapply(const Op<out_eT, eOp<in_eT, T1, eop_type>, op_conv_to>& in)
  {
  coot_extra_debug_sigprint();

  return eOp<out_eT, T1, eop_type>(in.m);
  }



// An eOp that has a conv_to in it wrapped around the same op can also be combined.
template<typename out_eT, typename in_eT, typename T1, typename eop_type>
inline
static
typename enable_if2<
    // make sure that no conversion is already happening
    is_same_type<in_eT, typename T1::elem_type>::value,
    eOp<out_eT, T1, eop_type>
>::result
conv_to_preapply(const eOp<out_eT, Op<out_eT, eOp<in_eT, T1, eop_type>, op_conv_to>, eop_type>& in)
  {
  coot_extra_debug_sigprint();

  return eOp<out_eT, T1, eop_type>(in.m.m, in.aux_in, true);
  }



template<typename out_eT, typename in_eT, typename T1, typename T2, typename glue_type>
inline
static
typename enable_if2<
    // make sure that no conversion is already happening
    is_same_type<in_eT, typename T1::elem_type>::value,
    Glue<out_eT, T1, T2, glue_type>
>::result
conv_to_preapply(const Op<out_eT, Glue<in_eT, T1, T2, glue_type>, op_conv_to>& in)
  {
  coot_extra_debug_sigprint();

  return Glue<out_eT, T1, T2, glue_type>(in);
  }



template<typename out_eT, typename in_eT, typename T1, typename T2, typename glue_type>
inline
static
typename enable_if2<
    // make sure that no conversion is already happening
    is_same_type<in_eT, typename T1::elem_type>::value,
    eGlue<out_eT, T1, T2, glue_type>
>::result
conv_to_preapply(const Op<out_eT, eGlue<in_eT, T1, T2, glue_type>, op_conv_to>& in)
  {
  coot_extra_debug_sigprint();

  return eGlue<out_eT, T1, T2, glue_type>(in);
  }
