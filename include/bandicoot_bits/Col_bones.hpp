// Copyright 2020 Ryan Curtin (http://www.ratml.org)
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



// Class for row vectors (matrices with only one row)

template<typename eT>
class Col : public Mat<eT>
  {
  public:

  typedef eT                                elem_type;  // the type of elements stored in the matrix
  typedef typename get_pod_type<eT>::result  pod_type;  // if eT is std::complex<T>, pod_type is T; otherwise pod_type is eT

  static const bool is_col = true;
  static const bool is_row = false;
  static const bool is_xvec = true;

  inline          Col();
  inline explicit Col(const uword N);
  inline explicit Col(const uword in_rows, const uword in_cols);
  inline explicit Col(const SizeMat& s);

  template<typename fill_type> inline Col(const uword N,                            const fill::fill_class<fill_type>& f);
  template<typename fill_type> inline Col(const uword in_rows, const uword in_cols, const fill::fill_class<fill_type>& f);
  template<typename fill_type> inline Col(const SizeMat& s,                         const fill::fill_class<fill_type>& f);

  inline Col(dev_mem_t<eT> aux_dev_mem, const uword N);
  inline Col(cl_mem        aux_dev_mem, const uword N);
  inline Col(eT*           aux_dev_mem, const uword N);

  inline                  Col(const Col& X);
  inline const Col& operator=(const Col& X);

  inline                  Col(Col&& X);
  inline const Col& operator=(Col&& X);

  // TODO: inline            Col(Mat<eT>&& m);
  // TODO: inline Col& operator=(Mat<eT>&& m);

  template<typename T1> inline            Col(const Base<eT, T1>& X);
  template<typename T1> inline Col& operator=(const Base<eT, T1>& X);

  inline                  Col(const arma::Col<eT>& X);
  inline const Col& operator=(const arma::Col<eT>& X);

  inline explicit operator arma::Col<eT> () const;

  coot_warn_unused inline const Op<Col<eT>, op_htrans>  t() const;
  coot_warn_unused inline const Op<Col<eT>, op_htrans> ht() const;
  coot_warn_unused inline const Op<Col<eT>, op_strans> st() const;

  using Mat<eT>::rows;
  using Mat<eT>::operator();

  coot_inline       subview_col<eT> rows(const uword in_row1, const uword in_row2);
  coot_inline const subview_col<eT> rows(const uword in_row1, const uword in_row2) const;

  coot_inline       subview_col<eT> subvec(const uword in_row1, const uword in_row2);
  coot_inline const subview_col<eT> subvec(const uword in_row1, const uword in_row2) const;

  coot_inline       subview_col<eT> subvec(const uword start_row, const SizeMat& s);
  coot_inline const subview_col<eT> subvec(const uword start_row, const SizeMat& s) const;

  #ifdef COOT_EXTRA_COL_BONES
    #include COOT_INCFILE_WRAP(COOT_EXTRA_COL_BONES)
  #endif
  };
