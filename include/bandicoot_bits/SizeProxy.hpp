// Copyright 2020 Ryan Curtin (http://www.ratml.org/)
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


// The SizeProxy class is meant, for now, to provide an interface to partially unwrap types for eOp and eGlue,
// so that the sizes of the operand can be known when the eOp or eGlue is created.  Any time a SizeProxy is
// used, evaluations may be performed.  The underlying object Q should be used for any actual operations.
//
// The SizeProxy class defines the following types and methods:
//
// elem_type        = the type of the elements obtained from object Q
// pod_type         = the underlying type of elements if elem_type is std::complex
// stored_type      = the type of the Q object
//
// is_row           = boolean indicating whether the Q object can be treated a row vector
// is_col           = boolean indicating whether the Q object can be treated a column vector
// is_xvec          = boolean indicating whether the Q object is a vector with unknown orientation
//
// Q                = object that can be unwrapped via the unwrap family of classes (ie. Q must be convertible to Mat)
//
// get_n_rows()     = return the number of rows in Q
// get_n_cols()     = return the number of columns in Q
// get_n_elem()     = return the number of elements in Q

template<typename eT>
class SizeProxy< Mat<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Mat<eT>                                  stored_type;

  static const bool is_row = false;
  static const bool is_col = false;
  static const bool is_xvec = false;

  coot_aligned const Mat<eT>& Q;

  inline explicit SizeProxy(const Mat<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.n_rows; }
  coot_aligned uword get_n_cols() const { return Q.n_cols; }
  coot_aligned uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< Col<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Col<eT>                                  stored_type;

  static const bool is_row = false;
  static const bool is_col = true;
  static const bool is_xvec = false;

  coot_aligned const Col<eT>& Q;

  inline explicit SizeProxy(const Col<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.n_rows; }
  coot_aligned uword get_n_cols() const { return Q.n_cols; }
  coot_aligned uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< Row<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Row<eT>                                  stored_type;

  static const bool is_row = true;
  static const bool is_col = false;
  static const bool is_xvec = false;

  coot_aligned const Row<eT>& Q;

  inline explicit SizeProxy(const Row<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.n_rows; }
  coot_aligned uword get_n_cols() const { return Q.n_cols; }
  coot_aligned uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< subview<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef subview<eT>                              stored_type;

  static const bool is_row = false;
  static const bool is_col = false;
  static const bool is_xvec = false;

  coot_aligned const subview<eT>& Q;

  inline explicit SizeProxy(const subview<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.n_rows; }
  coot_aligned uword get_n_cols() const { return Q.n_cols; }
  coot_aligned uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< subview_col<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef subview_col<eT>                          stored_type;

  static const bool is_row = false;
  static const bool is_col = true;
  static const bool is_xvec = false;

  coot_aligned const subview_col<eT>& Q;

  inline explicit SizeProxy(const subview_col<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.n_rows; }
  coot_aligned uword get_n_cols() const { return Q.n_cols; }
  coot_aligned uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< subview_row<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef subview_row<eT>                          stored_type;

  static const bool is_row = true;
  static const bool is_col = false;
  static const bool is_xvec = false;

  coot_aligned const subview_row<eT>& Q;

  inline explicit SizeProxy(const subview_row<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.n_rows; }
  coot_aligned uword get_n_cols() const { return Q.n_cols; }
  coot_aligned uword get_n_elem() const { return Q.n_elem; }
  };



// eOp
template<typename out_eT, typename T1, typename eop_type>
class SizeProxy< eOp<out_eT, T1, eop_type> >
  {
  public:

  typedef out_eT                                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef eOp<out_eT, T1, eop_type>                stored_type;

  static const bool is_row = eOp<out_eT, T1, eop_type>::is_row;
  static const bool is_col = eOp<out_eT, T1, eop_type>::is_col;
  static const bool is_xvec = eOp<out_eT, T1, eop_type>::is_xvec;

  coot_aligned const eOp<out_eT, T1, eop_type>& Q;

  inline explicit SizeProxy(const eOp<out_eT, T1, eop_type>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.get_n_rows(); }
  coot_aligned uword get_n_cols() const { return Q.get_n_cols(); }
  coot_aligned uword get_n_elem() const { return Q.get_n_elem(); }
  };



// eGlue
template<typename out_eT, typename T1, typename T2, typename eglue_type>
class SizeProxy< eGlue<out_eT, T1, T2, eglue_type> >
  {
  public:

  typedef out_eT                                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef eGlue<out_eT, T1, T2, eglue_type>            stored_type;

  static const bool is_row = eGlue<out_eT, T1, T2, eglue_type>::is_row;
  static const bool is_col = eGlue<out_eT, T1, T2, eglue_type>::is_col;
  static const bool is_xvec = eGlue<out_eT, T1, T2, eglue_type>::is_xvec;

  coot_aligned const eGlue<out_eT, T1, T2, eglue_type>& Q;

  inline explicit SizeProxy(const eGlue<out_eT, T1, T2, eglue_type>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.get_n_rows(); }
  coot_aligned uword get_n_cols() const { return Q.get_n_cols(); }
  coot_aligned uword get_n_elem() const { return Q.get_n_elem(); }
  };



// Op: in order to get its size, we need to unwrap it.
// TODO: maybe this can be done better?
template<typename out_eT, typename T1, typename op_type>
class SizeProxy< Op<out_eT, T1, op_type> >
  {
  public:

  typedef out_eT                                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Mat<out_eT>                              stored_type;

  static const bool is_row = Op<out_eT, T1, op_type>::is_row;
  static const bool is_col = Op<out_eT, T1, op_type>::is_col;
  static const bool is_xvec = Op<out_eT, T1, op_type>::is_xvec;

  coot_aligned const unwrap<Op<out_eT, T1, op_type>> U;
  coot_aligned const Mat<out_eT>& Q;

  inline explicit SizeProxy(const Op<out_eT, T1, op_type>& A)
    : U(A)
    , Q(U.M)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.n_rows; }
  coot_aligned uword get_n_cols() const { return Q.n_cols; }
  coot_aligned uword get_n_elem() const { return Q.n_elem; }
  };



// Glue: in order to get its size, we need to unwrap it.
// TODO: maybe this can be done better?
template<typename out_eT, typename T1, typename T2, typename glue_type>
class SizeProxy< Glue<out_eT, T1, T2, glue_type> >
  {
  public:

  typedef out_eT                                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Mat<out_eT>                              stored_type;

  static const bool is_row = Glue<out_eT, T1, T2, glue_type>::is_row;
  static const bool is_col = Glue<out_eT, T1, T2, glue_type>::is_col;
  static const bool is_xvec = Glue<out_eT, T1, T2, glue_type>::is_xvec;

  coot_aligned const unwrap<Glue<out_eT, T1, T2, glue_type>> U;
  coot_aligned const Mat<out_eT>& Q;

  inline explicit SizeProxy(const Glue<out_eT, T1, T2, glue_type>& A)
    : U(A)
    , Q(U.M)
    {
    coot_extra_debug_sigprint();
    }

  coot_aligned uword get_n_rows() const { return Q.n_rows; }
  coot_aligned uword get_n_cols() const { return Q.n_cols; }
  coot_aligned uword get_n_elem() const { return Q.n_elem; }
  };



//! @}
