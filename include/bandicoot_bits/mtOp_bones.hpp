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



template<typename out_eT, typename T1, typename mtop_type>
class mtOp : public Base< out_eT, mtOp<out_eT, T1, mtop_type> >
  {
  public:
  
  typedef out_eT                                   elem_type;
  typedef typename T1::elem_type                   in_elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  
  static const bool is_row = T1::is_row;
  static const bool is_col = T1::is_col;
  
  // Note that instantiation of an object will never happen here---mtOps aren't formed
  // unless it is possible to do any delayed evaluation.
  coot_aligned const SizeProxy<T1> m;
  const T1& q;
  
  inline         ~mtOp();
  inline explicit mtOp(const T1& in_m);

  coot_inline uword get_n_rows() const;
  coot_inline uword get_n_cols() const;
  coot_inline uword get_n_elem() const;
  };



// Utilities to determine what should be held in a SizeProxy.
// Specifically this controls whether (and how) something is unwrapped in order to compute its size.
template<typename T>
struct mtop_holder_type
  {
  // This should never happen.
  };



template<typename out_eT, typename T1, typename mtop_type>
struct mtop_holder_type<mtOp<out_eT, T1, mtop_type>>
  {
  // By default, don't do any unwrapping.
  typedef const mtOp<out_eT, T1, mtop_type>& result;
  };



template<typename out_eT, typename T1, typename op_type, typename mtop_type>
struct mtop_holder_type<mtOp<out_eT, Op<T1, op_type>, mtop_type>>
  {
  // For an Op, we can't know its size without unwrapping it.
  typedef Mat<out_eT> result;
  };



template<typename out_eT, typename T1, typename T2, typename glue_type, typename mtop_type>
struct mtop_holder_type<mtOp<out_eT, Glue<T1, T2, glue_type>, mtop_type>>
  {
  // For a Glue, we can't know its size without unwrapping it.
  typedef Mat<out_eT> result;
  };



// For op_htrans, we actually *can* know the size.
template<typename out_eT, typename T1, typename mtop_type>
struct mtop_holder_type<mtOp<out_eT, Op<T1, op_htrans>, mtop_type>>
  {
  typedef const mtOp<out_eT, Op<T1, op_htrans>, mtop_type>& result;
  };



template<typename out_eT, typename T1, typename mtop_type>
struct mtop_holder_type<mtOp<out_eT, Op<T1, op_htrans2>, mtop_type>>
  {
  typedef const mtOp<out_eT, Op<T1, op_htrans2>, mtop_type>& result;
  };



template<typename out_eT, typename T1, typename mtop_type> inline uword dispatch_mtop_get_n_rows(const mtOp<out_eT, T1, mtop_type>& Q);
template<typename out_eT>                                  inline uword dispatch_mtop_get_n_rows(const Mat<out_eT>& Q);

template<typename out_eT, typename T1, typename mtop_type> inline uword dispatch_mtop_get_n_cols(const mtOp<out_eT, T1, mtop_type>& Q);
template<typename out_eT>                                  inline uword dispatch_mtop_get_n_cols(const Mat<out_eT>& Q);

template<typename out_eT, typename T1, typename mtop_type> inline uword dispatch_mtop_get_n_elem(const mtOp<out_eT, T1, mtop_type>& Q);
template<typename out_eT>                                  inline uword dispatch_mtop_get_n_elem(const Mat<out_eT>& Q);
