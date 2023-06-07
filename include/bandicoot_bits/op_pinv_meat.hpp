// Copyright 2023 Ryan Curtin (https://www.ratml.org/)
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



template<typename eT2, typename T1>
inline
void
op_pinv::apply(Mat<eT2>& out, const Op<T1, op_pinv>& in)
  {
  coot_extra_debug_sigprint();

  const typename T1::elem_type tol = in.aux;

  const std::tuple<bool, std::string> result = apply_direct(out, in.m, tol);
  if (std::get<0>(result) == false)
    {
    coot_stop_runtime_error("pinv(): " + std::get<1>(result));
    }
  }



template<typename eT2, typename T1>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct(Mat<eT2>& out, const T1& in, const typename T1::elem_type tol)
  {
  coot_extra_debug_sigprint();

  // If `in` is a diagmat():
  //    stored in a vector: apply_direct_diag_vec()
  //    stored in a matrix: apply_direct_diag()
  // if `in` is symmetric:
  //    apply_direct_sym()
  // else:
  //    apply_direct_gen()
  if (resolves_to_diagmat<T1>::value)
    {
    // Now detect whether it is stored in a vector or matrix.
    strip_diagmat<T1> s1(in);
    if (s1.M.n_rows > 1 && s1.M.n_cols > 1)
      {
      return apply_direct_diag(out, s1.M, tol);
      }
    else
      {
      // Extract the diagonal into a standalone vector for easier processing.
      const uword N = (std::min)(s1.M.n_rows, s1.M.n_cols);
      Col<typename T1::elem_type> diag(N);
      coot_rt_t::extract_diag(diag.get_dev_mem(false), s1.M.get_dev_mem(false), 0, s1.M.n_rows, N);

      return apply_direct_diag(out, diag, tol);
      }
    }
  else if (resolves_to_symmat<T1>::value)
    {
    // TODO: would be great to avoid actually materializing everything here
    // TODO: a `strip_symmat` struct would be useful for this

    unwrap<T1> U(in);
    extract_subview<typename unwrap<T1>::stored_type> E(U.M);
    // apply_direct_sym() is destructive to the input matrix, so, we may need to make a copy.
    if (is_Mat<T1>::value)
      {
      Mat<typename T1::elem_type> tmp(E.M);
      return apply_direct_sym(out, tmp, tol);
      }
    else
      {
      // We have already created a temporary for unwrapping, so we can destructively use that.
      return apply_direct_sym(out, const_cast<Mat<typename T1::elem_type>&>(E.M), tol);
      }
    }
  else
    {
    unwrap<T1> U(in);
    extract_subview<typename unwrap<T1>::stored_type> E(U.M);
    // apply_direct_gen() is destructive to the input matrix, so, we may need to make a copy.
    if (is_Mat<T1>::value)
      {
      Mat<typename T1::elem_type> tmp(E.M);
      return apply_direct_gen(out, tmp, tol);
      }
    else
      {
      // We have already created a temporary for unwrapping, so we can destructively use that.
      return apply_direct_gen(out, const_cast<Mat<typename T1::elem_type>&>(E.M), tol);
      }
    }
  }



template<typename eT>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_diag(Mat<eT>& out, const Mat<eT>& in, const eT tol)
  {
  coot_extra_debug_sigprint();

  coot_debug_check(in.n_rows != 1 && in.n_cols != 1, "op_pinv::apply_direct_diag_vec(): given input is not a vector (internal error)");

  const uword N = (std::max)(in.n_rows, in.n_cols);

  out.zeros(N, N);

  // Check for any NaNs in the input.
  const bool has_nans = coot_rt_t::any_vec(in.get_dev_mem(false), in.n_elem, (eT) 0, oneway_real_kernel_id::rel_any_nonfinite, oneway_real_kernel_id::rel_any_nonfinite_small);

  if (has_nans == true)
    {
    out.reset();
    return std::make_tuple(false, "NaNs detected in input matrix");
    }

  // Find the values that are below tolerance.
  Mat<eT> abs_in(in.n_rows, in.n_cols);
  coot_rt_t::eop_scalar(abs_in.get_dev_mem(false), in.get_dev_mem(false), in.n_elem, (eT) 0, (eT) 0, twoway_kernel_id::equ_array_abs);

  // Compute tolerance if not given.
  eT tol_use = tol;
  if (tol == (eT) 0)
    {
    const eT max_val = coot_rt_t::max(abs_in.get_dev_mem(false), abs_in.n_elem);
    tol_use = abs_in.n_elem * max_val * std::numeric_limits<eT>::epsilon();
    }

  Mat<uword> tol_indicator(in.n_rows, in.n_cols);
  coot_rt_t::relational_scalar_op(tol_indicator.get_dev_mem(false), abs_in.get_dev_mem(false), abs_in.n_elem, (eT) tol_use, twoway_kernel_id::rel_gt_scalar, "pinv()");

  // Now invert the diagonal.  Any zero values need to changed to 1, so as to not produce infs or nans.
  Mat<eT> out_vec(abs_in.n_rows, abs_in.n_cols);
  coot_rt_t::copy_array(out_vec.get_dev_mem(false), in.get_dev_mem(false), in.n_elem);
  coot_rt_t::replace(out_vec.get_dev_mem(false), out_vec.n_elem, (eT) 0.0, (eT) 1.0);
  coot_rt_t::eop_scalar(out_vec.get_dev_mem(false), out_vec.get_dev_mem(false), in.n_elem, (eT) 0, (eT) 1, twoway_kernel_id::equ_array_div_scalar_pre);

  // Zero out any values that are below the tolerance.
  coot_rt_t::inplace_op_array(out_vec.get_dev_mem(false), tol_indicator.get_dev_mem(false), out_vec.n_elem, twoway_kernel_id::inplace_mul_array);

  // Now set the diagonal of the other matrix.
  coot_rt_t::set_diag(out.get_dev_mem(false), out_vec.get_dev_mem(false), 0, N, N);

  return std::make_tuple(true, "");
  }



template<typename eT2, typename eT1>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_diag(Mat<eT2>& out, const Mat<eT1>& in, const eT1 tol, const typename enable_if<is_same_type<eT1, eT2>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  coot_debug_check(in.n_rows != 1 && in.n_cols != 1, "op_pinv::apply_direct_diag_vec(): given input is not a vector (internal error)");

  const uword N = (std::max)(in.n_rows, in.n_cols);

  // We have to manually convert to eT2, but only after the operation is done.
  // Thus, we have to operate on a temporary.
  Mat<eT1> tmp(N, 1);
  coot_rt_t::eop_scalar(tmp.get_dev_mem(false), in.get_dev_mem(false), N, (eT1) 0, (eT1) 1, twoway_kernel_id::equ_array_div_scalar_pre);

  // Check for any NaNs to indicate success or failure.
  const bool status = coot_rt_t::any_vec(tmp.get_dev_mem(false), N, (eT1) 0, oneway_real_kernel_id::rel_any_nonfinite, oneway_real_kernel_id::rel_any_nonfinite_small);

  if (status == false)
    {
    out.clear();
    return std::make_tuple(false, "NaNs detected in inverted diagonal");
    }
  else
    {
    out.zeros(N, N);
    coot_rt_t::set_diag(out.get_dev_mem(false), tmp.get_dev_mem(false), 0, N, N);
    }

  return std::make_tuple(true, "");
  }



template<typename eT>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_sym(Mat<eT>& out, Mat<eT>& in, const eT tol)
  {
  coot_extra_debug_sigprint();

  Col<eT> eigvals(in.n_rows);

  //
  // Step 1. compute eigendecomposition, sorting eigenvalues descending by absolute value.
  //

  // `in` will store the eigenvectors after this call (destructive).
  const std::tuple<bool, std::string> result = coot_rt_t::eig_sym(in.get_dev_mem(true), in.n_rows, true, eigvals.get_dev_mem(true));
  if (std::get<0>(result) == false)
    {
    out.reset();
    return std::make_tuple(false, "eigendecomposition failed");
    }

  Col<eT> abs_eigvals(in.n_rows);
  coot_rt_t::eop_scalar(abs_eigvals.get_dev_mem(false), eigvals.get_dev_mem(false), eigvals.n_elem, (eT) 0, (eT) 0, twoway_kernel_id::equ_array_abs);

  Col<uword> eigval_order(in.n_rows);
  // This also sorts `abs_eigvals`.
  coot_rt_t::sort_index_vec(eigval_order.get_dev_mem(false), abs_eigvals.get_dev_mem(false), abs_eigvals.n_elem, 1 /* descending */, 0);

  //
  // Step 2. keep all eigenvalues greater than the tolerance.
  //
  const eT tol_use = (tol == eT(0)) ? in.n_rows * abs_eigvals[0] * std::numeric_limits<eT>::epsilon() : tol;

  Col<uword> tol_indicators(eigval_order.n_elem);
  coot_rt_t::relational_scalar_op(tol_indicators.get_dev_mem(false), abs_eigvals.get_dev_mem(false), eigval_order.n_elem, tol_use, twoway_kernel_id::rel_gteq_scalar, "pinv()");
  const uword num_eigvals = coot_rt_t::accu(tol_indicators.get_dev_mem(false), eigval_order.n_elem);
  if (num_eigvals == 0)
    {
    out.zeros(in.n_rows, in.n_cols);
    return std::make_tuple(true, "");
    }

  // Filter the top eigenvalues and eigenvectors.
  Col<eT> filtered_eigvals(num_eigvals);
  Mat<eT> filtered_eigvecs(in.n_rows, num_eigvals);
  coot_rt_t::reorder_cols(filtered_eigvals.get_dev_mem(false), eigvals.get_dev_mem(false), 1, eigval_order.get_dev_mem(false), num_eigvals);
  coot_rt_t::reorder_cols(filtered_eigvecs.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, eigval_order.get_dev_mem(false), num_eigvals);

  //
  // 3. Invert the eigenvalues we kept.
  //
  coot_rt_t::replace(filtered_eigvals.get_dev_mem(false), num_eigvals, (eT) 0, (eT) 1); // avoid divergence
  coot_rt_t::eop_scalar(filtered_eigvals.get_dev_mem(false), filtered_eigvals.get_dev_mem(false), num_eigvals, (eT) 0, (eT) 1, twoway_kernel_id::equ_array_div_scalar_pre);

  //
  // 4. Construct output.
  //
  out.set_size(filtered_eigvecs.n_rows, filtered_eigvecs.n_rows);
  Mat<eT> tmp(filtered_eigvecs.n_rows, filtered_eigvecs.n_cols);
  // tmp = filtered_eigvecs * diagmat(inverted eigvals)
  coot_rt_t::mul_diag(tmp.get_dev_mem(false), tmp.n_rows, tmp.n_cols,
                      (eT) 1, filtered_eigvecs.get_dev_mem(false), false, false,
                      filtered_eigvals.get_dev_mem(false), true /* diag */, false);
  // out = tmp * filtered_eigvecs.t()
  coot_rt_t::gemm<eT, false, true>(out.get_dev_mem(true), out.n_rows, out.n_cols,
                                   tmp.get_dev_mem(true), tmp.n_rows, tmp.n_cols,
                                   filtered_eigvecs.get_dev_mem(true), (eT) 1.0, (eT) 0.0);

  return std::make_tuple(true, "");
  }



template<typename eT2, typename eT1>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_sym(Mat<eT2>& out, Mat<eT1>& in, const eT1 tol, const typename enable_if<is_same_type<eT1, eT2>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  // We need to perform this into a temporary, and then convert.
  Mat<eT1> tmp;
  const std::tuple<bool, std::string> status = apply_direct_sym(tmp, in, tol);
  if (std::get<0>(status) == false)
    {
    return status;
    }

  out.set_size(tmp.n_rows, tmp.n_cols);
  coot_rt_t::copy_array(out.get_dev_mem(false), tmp.get_dev_mem(false), tmp.n_elem);
  return status; // (true, "")
  }




template<typename eT>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_gen(Mat<eT>& out, Mat<eT>& in, const eT tol)
  {
  coot_extra_debug_sigprint();

  //
  // 1. Transpose input if needed so that n_rows >= n_cols.
  //
  Mat<eT> tmp_in;
  Mat<eT>& in_use = (in.n_rows < in.n_cols) ? tmp_in : in;
  if (in.n_rows < in.n_cols)
    {
    tmp_in.set_size(in.n_cols, in.n_rows);
    coot_rt_t::htrans(tmp_in.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols);
    }

  //
  // 2. Compute the SVD.
  //
  Mat<eT> U(in_use.n_rows, in_use.n_rows);
  Col<eT> S((std::min)(in_use.n_rows, in_use.n_cols));
  Mat<eT> V(in_use.n_cols, in_use.n_cols);

  const std::tuple<bool, std::string> status = coot_rt_t::svd(U.get_dev_mem(true),
                                                              S.get_dev_mem(true),
                                                              V.get_dev_mem(true),
                                                              in_use.get_dev_mem(true),
                                                              in_use.n_rows,
                                                              in_use.n_cols,
                                                              true);

  if (std::get<0>(status) == false)
    {
    return std::make_tuple(false, "SVD failed");
    }

  //
  // 2. Compute tolerance.  Note that the singular values are returned in descending order already.
  //
  const eT largest_sv = S[0];
  const eT tol_use = (tol == eT(0)) ? in_use.n_rows * largest_sv * std::numeric_limits<eT>::epsilon() : tol;

  //
  // 3. Keep singular values that are greater than the tolerance.
  //
  Col<uword> S_above_tol(S.n_elem);
  coot_rt_t::relational_scalar_op(S_above_tol.get_dev_mem(false), S.get_dev_mem(false), S_above_tol.n_elem, tol_use, twoway_kernel_id::rel_gteq_scalar, "pinv()");
  const uword num_svs = coot_rt_t::accu(S_above_tol.get_dev_mem(false), S_above_tol.n_elem);
  if (num_svs == 0)
    {
    out.zeros(in.n_rows, in.n_cols);
    return std::make_tuple(true, "");
    }

  // Create aliases for the filtered left/right singular vectors and filtered singular values.
  Mat<eT> filtered_U(U.get_dev_mem(false), U.n_rows, num_svs);
  Mat<eT> filtered_S(S.get_dev_mem(false), num_svs, 1);
  Mat<eT> filtered_V(V.get_dev_mem(false), V.n_rows, num_svs);

  //
  // 4. Invert singular values.
  //
  coot_rt_t::replace(filtered_S.get_dev_mem(false), num_svs, (eT) 0, (eT) 1); // avoid divergence
  coot_rt_t::eop_scalar(filtered_S.get_dev_mem(false), filtered_S.get_dev_mem(false), num_svs, (eT) 0, (eT) 1, twoway_kernel_id::equ_array_div_scalar_pre);

  //
  // 5. Reconstruct as subset of V * diagmat(inv_s) * subset of U
  //    (transposed if n_rows < n_cols).
  //
  if (in.n_rows < in.n_cols)
    {
    // U' = U * diagmat(s)   (in-place into U)
    coot_rt_t::mul_diag(filtered_U.get_dev_mem(false), filtered_U.n_rows, filtered_U.n_cols,
                        (eT) 1.0, filtered_U.get_dev_mem(false), false, false,
                        filtered_S.get_dev_mem(false), true, false);

    // out = U' * V^T (remember V is already transposed)
    out.set_size(filtered_U.n_rows, filtered_V.n_rows);
    coot_rt_t::gemm<eT, false, false>(out.get_dev_mem(true), out.n_rows, out.n_cols,
                                      filtered_U.get_dev_mem(true), filtered_U.n_rows, filtered_U.n_cols,
                                      filtered_V.get_dev_mem(true), (eT) 1.0, (eT) 0.0);
    }
  else
    {
    // tmp = V * diagmat(s)   (in-place into V)
    Mat<eT> tmp(filtered_V.n_cols, filtered_V.n_rows);
    coot_rt_t::mul_diag(tmp.get_dev_mem(false), tmp.n_rows, tmp.n_cols,
                        (eT) 1.0, filtered_V.get_dev_mem(false), false, true,
                        filtered_S.get_dev_mem(false), true, false);

    // out = tmp * U^T
    out.set_size(tmp.n_rows, filtered_U.n_rows);
    coot_rt_t::gemm<eT, false, true>(out.get_dev_mem(true), out.n_rows, out.n_cols,
                                     tmp.get_dev_mem(true), tmp.n_rows, tmp.n_cols,
                                     filtered_U.get_dev_mem(true), (eT) 1.0, (eT) 0.0);
    }

  return std::make_tuple(true, "");
  }



template<typename eT2, typename eT1>
inline
std::tuple<bool, std::string>
op_pinv::apply_direct_gen(Mat<eT2>& out, Mat<eT1>& in, const eT1 tol, const typename enable_if<is_same_type<eT1, eT2>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  // We need to perform this into a temporary, and then convert.
  Mat<eT1> tmp;
  const std::tuple<bool, std::string> status = apply_direct_gen(tmp, in, tol);
  if (std::get<0>(status) == false)
    {
    return status;
    }

  out.set_size(tmp.n_rows, tmp.n_cols);
  coot_rt_t::copy_array(out.get_dev_mem(false), tmp.get_dev_mem(false), tmp.n_elem);
  return status; // (true, "")
  }



template<typename T1>
inline
uword
op_pinv::compute_n_rows(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_pinv::compute_n_cols(const Op<T1, op_pinv>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }
