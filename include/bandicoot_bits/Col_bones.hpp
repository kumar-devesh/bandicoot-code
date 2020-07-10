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


//! \addtogroup Col
//! @{

//! Class for row vectors (matrices with only one row)

template<typename eT>
class Col : public Mat<eT>
  {
  public:

  typedef eT                                elem_type;  //!< the type of elements stored in the matrix
  typedef typename get_pod_type<eT>::result  pod_type;  //!< if eT is std::complex<T>, pod_type is T; otherwise pod_type is eT

  static const bool is_col = true;
  static const bool is_row = false;

  inline          Col();
  inline explicit Col(const uword N);
  inline explicit Col(const uword in_rows, const uword in_cols);

  inline                  Col(const Col& X);
  inline const Col& operator=(const Col& X);

  #if defined(COOT_USE_CXX11)
  inline                  Col(Col&& X);
  inline const Col& operator=(Col&& X);
  #endif

  template<typename T1> inline            Col(const Base<eT, T1>& X);
  template<typename T1> inline Col& operator=(const Base<eT, T1>& X);

  inline                  Row(const arma::Row<eT>& X);
  inline const Row& operator=(const arma::Row<eT>& X);

  inline explicit operator arma::Col<eT> () const;

  coot_inline const Op<Col<eT>, op_htrans>  t() const;
  coot_inline const Op<Col<eT>, op_htrans> ht() const;
  coot_inline const Op<Col<eT>, op_strans> st() const;

  using Mat<eT>::rows;
  using Mat<eT>::operator();

  coot_inline       subview_col<eT> rows(const uword in_row1, const uword in_row2);
  coot_inline const subview_col<eT> rows(const uword in_row1, const uword in_row2) const;

  coot_inline       subview_col<eT> subvec(const uword in_row1, const uword in_row2);
  coot_inline const subview_col<eT> subvec(const uword in_row1, const uword in_row2) const;

  #ifdef COOT_EXTRA_COL_BONES
    #include COOT_INCFILE_WRAP(COOT_EXTRA_COL_BONES)
  #endif
  };

//! @}
