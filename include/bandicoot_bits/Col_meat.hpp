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



template<typename eT>
inline
Col<eT>::Col()
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  }



template<typename eT>
inline
Col<eT>::Col(const uword N)
  : Mat<eT>(N, 1)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  }



template<typename eT>
inline
Col<eT>::Col(const uword in_rows, const uword in_cols)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  Mat<eT>::init(in_rows, in_cols);
  }



template<typename eT>
inline
Col<eT>::Col(const Col<eT>& X)
  : Mat<eT>(X.n_rows, 1)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  arrayops::copy(this->get_dev_mem(), X.get_dev_mem(), X.n_elem);
  }



template<typename eT>
inline
const Col<eT>&
Col<eT>::operator=(const Col<eT>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT>::init(X.n_rows, 1);
  arrayops::copy(this->get_dev_mem(), X.get_dev_mem(), X.n_elem);

  return *this;
  }



template<typename eT>
inline
Col<eT>::Col(Col<eT>&& X)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  Mat<eT>::steal_mem(X);
  // Make sure to restore the other Col's vec_state.
  access::rw(X.vec_state) = 1;
  }



template<typename eT>
inline
const Col<eT>&
Col<eT>::operator=(Col<eT>&& X)
  {
  coot_extra_debug_sigprint();

  // Clean up old memory, if required.
  coot_rt_t::synchronise();
  Mat<eT>::cleanup();

  Mat<eT>::steal_mem(X);
  // Make sure to restore the other Col's vec_state.
  access::rw(X.vec_state) = 1;

  return *this;
  }



template<typename eT>
template<typename T1>
inline
Col<eT>::Col(const Base<eT, T1>& X)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;

  Mat<eT>::operator=(X.get_ref());
  }



template<typename eT>
template<typename T1>
inline
Col<eT>&
Col<eT>::operator=(const Base<eT, T1>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT>::operator=(X.get_ref());

  return *this;
  }



template<typename eT>
inline
Col<eT>::Col(const arma::Col<eT>& X)
  : Mat<eT>((const arma::Mat<eT>&) X)
  {
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
const Col<eT>&
Col<eT>::operator=(const arma::Col<eT>& X)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(X.n_rows, X.n_cols);

  (*this).copy_into_dev_mem(X.memptr(), (*this).n_elem);

  return *this;
  }



template<typename eT>
inline
Col<eT>::operator arma::Col<eT>() const
  {
  coot_extra_debug_sigprint();

  arma::Col<eT> out(Mat<eT>::n_rows, 1);

  (*this).copy_from_dev_mem(out.memptr(), (*this).n_elem);

  return out;
  }



template<typename eT>
coot_inline
const Op<Col<eT>, op_htrans>
Col<eT>::t() const
  {
  return Op<Col<eT>, op_htrans>(*this);
  }



template<typename eT>
coot_inline
const Op<Col<eT>, op_htrans>
Col<eT>::ht() const
  {
  return Op<Col<eT>, op_htrans>(*this);
  }



template<typename eT>
coot_inline
const Op<Col<eT>, op_strans>
Col<eT>::st() const
  {
  return Op<Col<eT>, op_strans>(*this);
  }



template<typename eT>
coot_inline
subview_col<eT>
Col<eT>::rows(const uword in_row1, const uword in_row2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_row1 > in_row2) || (in_row2 >= Mat<eT>::n_rows) ), "Col::rows(): indices out of bounds or incorrectly used");

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview_col<eT>(*this, 0, in_row1, subview_n_rows);
  }



template<typename eT>
coot_inline
const subview_col<eT>
Col<eT>::rows(const uword in_row1, const uword in_row2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_row1 > in_row2) || (in_row2 >= Mat<eT>::n_rows) ), "Col::rows(): indices out of bounds or incorrectly used");

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview_col<eT>(*this, 0, in_row1, subview_n_rows);
  }



template<typename eT>
coot_inline
subview_col<eT>
Col<eT>::subvec(const uword in_row1, const uword in_row2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_row1 > in_row2) || (in_row2 >= Mat<eT>::n_rows) ), "Col::rows(): indices out of bounds or incorrectly used");

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview_col<eT>(*this, 0, in_row1, subview_n_rows);
  }



template<typename eT>
coot_inline
const subview_col<eT>
Col<eT>::subvec(const uword in_row1, const uword in_row2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_row1 > in_row2) || (in_row2 >= Mat<eT>::n_rows) ), "Col::rows(): indices out of bounds or incorrectly used");

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview_col<eT>(*this, 0, in_row1, subview_n_rows);
  }



#ifdef COOT_EXTRA_COL_MEAT
  #include COOT_INCFILE_WRAP(COOT_EXTRA_COL_MEAT)
#endif

//! @}