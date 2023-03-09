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



template<typename eT>
inline
Row<eT>::Row()
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  }



template<typename eT>
inline
Row<eT>::Row(const uword N)
  : Mat<eT>(1, N)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  }



template<typename eT>
inline
Row<eT>::Row(const uword in_rows, const uword in_cols)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  Mat<eT>::init(in_rows, in_cols);
  }



template<typename eT>
inline
Row<eT>::Row(const SizeMat& s)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  Mat<eT>::init(s.n_rows, s.n_cols);
  }



template<typename eT>
inline
Row<eT>::Row(const Row<eT>& X)
  : Mat<eT>(1, X.n_cols)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  arrayops::copy(this->get_dev_mem(), X.get_dev_mem(), X.n_elem);
  }



template<typename eT>
inline
const Row<eT>&
Row<eT>::operator=(const Row<eT>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT>::init(1, X.n_cols);
  arrayops::copy(this->get_dev_mem(), X.get_dev_mem(), X.n_elem);

  return *this;
  }



template<typename eT>
inline
Row<eT>::Row(Row<eT>&& X)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  Mat<eT>::steal_mem(X);
  // Make sure to restore the other Row's vec_state.
  access::rw(X.vec_state) = 2;
  }



template<typename eT>
inline
const Row<eT>&
Row<eT>::operator=(Row<eT>&& X)
  {
  coot_extra_debug_sigprint();

  // Clean up old memory, if required.
  coot_rt_t::synchronise();
  Mat<eT>::cleanup();

  Mat<eT>::steal_mem(X);
  // Make sure to restore the other Row's vec_state.
  access::rw(X.vec_state) = 2;

  return *this;
  }



template<typename eT>
template<typename T1>
inline
Row<eT>::Row(const Base<eT, T1>& X)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;

  Mat<eT>::operator=(X.get_ref());
  }



template<typename eT>
template<typename T1>
inline
Row<eT>&
Row<eT>::operator=(const Base<eT, T1>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT>::operator=(X.get_ref());

  return *this;
  }



template<typename eT>
inline
Row<eT>::Row(const arma::Row<eT>& X)
  : Mat<eT>((const arma::Mat<eT>&) X)
  {
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
const Row<eT>&
Row<eT>::operator=(const arma::Row<eT>& X)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(X.n_rows, X.n_cols);

  (*this).copy_into_dev_mem(X.memptr(), (*this).n_elem);

  return *this;
  }



template<typename eT>
inline
Row<eT>::operator arma::Row<eT>() const
  {
  coot_extra_debug_sigprint();

  arma::Row<eT> out(1, Mat<eT>::n_cols);

  (*this).copy_from_dev_mem(out.memptr(), (*this).n_elem);

  return out;
  }



template<typename eT>
coot_inline
const Op<Row<eT>, op_htrans>
Row<eT>::t() const
  {
  return Op<Row<eT>, op_htrans>(*this);
  }



template<typename eT>
coot_inline
const Op<Row<eT>, op_htrans>
Row<eT>::ht() const
  {
  return Op<Row<eT>, op_htrans>(*this);
  }



template<typename eT>
coot_inline
const Op<Row<eT>, op_strans>
Row<eT>::st() const
  {
  return Op<Row<eT>, op_strans>(*this);
  }



template<typename eT>
coot_inline
subview_row<eT>
Row<eT>::cols(const uword in_col1, const uword in_col2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "Row::cols(): indices out of bounds or incorrectly used");

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
coot_inline
const subview_row<eT>
Row<eT>::cols(const uword in_col1, const uword in_col2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "Row::cols(): indices out of bounds or incorrectly used");

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
coot_inline
subview_row<eT>
Row<eT>::subvec(const uword in_col1, const uword in_col2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "Row::cols(): indices out of bounds or incorrectly used");

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
coot_inline
const subview_row<eT>
Row<eT>::subvec(const uword in_col1, const uword in_col2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "Row::cols(): indices out of bounds or incorrectly used");

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
coot_inline
subview_row<eT>
Row<eT>::subvec(const uword start_col, const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (s.n_rows != 1), "Row::subvec(): given size does not specify a row vector" );

  coot_debug_check_bounds( ( (start_col >= Mat<eT>::n_cols) || ((start_col + s.n_cols) > Mat<eT>::n_cols) ), "Row::subvec(): size out of bounds" );

  return subview_row<eT>(*this, 0, start_col, s.n_cols);
  }



template<typename eT>
coot_inline
const subview_row<eT>
Row<eT>::subvec(const uword start_col, const SizeMat& s) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (s.n_rows != 1), "Row::subvec(): given size does not specify a row vector" );

  coot_debug_check_bounds( ( (start_col >= Mat<eT>::n_cols) || ((start_col + s.n_cols) > Mat<eT>::n_cols) ), "Row::subvec(): size out of bounds" );

  return subview_row<eT>(*this, 0, start_col, s.n_cols);
  }



#ifdef COOT_EXTRA_ROW_MEAT
  #include COOT_INCFILE_WRAP(COOT_EXTRA_ROW_MEAT)
#endif
