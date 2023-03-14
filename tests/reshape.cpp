// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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

#include <bandicoot>
#include "catch.hpp"

using namespace coot;

TEMPLATE_TEST_CASE("reshape_same_size", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> y = reshape(x, 15, 60);

  REQUIRE( y.n_rows == 15 );
  REQUIRE( y.n_cols == 60 );
  REQUIRE( y.n_elem == 900 );

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y_cpu(y);

  REQUIRE( arma::approx_equal( arma::vectorise(x_cpu), arma::vectorise(y_cpu), "absdiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("reshape_smaller", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> y = reshape(x, 15, 30);

  REQUIRE( y.n_rows == 15 );
  REQUIRE( y.n_cols == 30 );
  REQUIRE( y.n_elem == 450 );

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y_cpu(y);

  arma::Col<eT> x_vec = arma::vectorise(x_cpu);
  arma::Col<eT> y_vec = arma::vectorise(y_cpu);
  REQUIRE( arma::approx_equal( x_vec.subvec(0, 449), y_vec, "absdiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("reshape_larger", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> y = reshape(x, 15, 70);

  REQUIRE( y.n_rows == 15 );
  REQUIRE( y.n_cols == 70 );
  REQUIRE( y.n_elem == 1050 );

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y_cpu(y);

  arma::Col<eT> x_vec = arma::vectorise(x_cpu);
  arma::Col<eT> y_vec = arma::vectorise(y_cpu);
  REQUIRE( arma::approx_equal( x_vec, y_vec.subvec(0, 899), "absdiff", 1e-5) );
  REQUIRE( arma::max( arma::max( arma::abs( y_vec.subvec(900, 1049) ) ) ) == Approx(eT(0)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("reshape_alias_same_size", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x1 = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> x2 = x1;
  Mat<eT> x_orig = x1;

  x1 = reshape(x1, 15, 60);
  x2.reshape(15, 60);

  REQUIRE( x1.n_rows == 15 );
  REQUIRE( x1.n_cols == 60 );
  REQUIRE( x1.n_elem == 900 );
  REQUIRE( x2.n_rows == 15 );
  REQUIRE( x2.n_cols == 60 );
  REQUIRE( x2.n_elem == 900 );

  arma::Mat<eT> x1_cpu(x1);
  arma::Mat<eT> x2_cpu(x2);
  arma::Mat<eT> x_orig_cpu(x_orig);

  arma::Col<eT> x1_vec = arma::vectorise(x1_cpu);
  arma::Col<eT> x2_vec = arma::vectorise(x2_cpu);
  arma::Col<eT> x_orig_vec = arma::vectorise(x_orig_cpu);
  REQUIRE( arma::approx_equal( x1_vec, x_orig_vec, "absdiff", 1e-5) );
  REQUIRE( arma::approx_equal( x2_vec, x_orig_vec, "absdiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("reshape_alias_smaller", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x1 = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> x2 = x1;
  Mat<eT> x_orig = x1;

  x1 = reshape(x1, 15, 30);
  x2.reshape(15, 30);

  REQUIRE( x1.n_rows == 15 );
  REQUIRE( x1.n_cols == 30 );
  REQUIRE( x1.n_elem == 450 );
  REQUIRE( x2.n_rows == 15 );
  REQUIRE( x2.n_cols == 30 );
  REQUIRE( x2.n_elem == 450 );

  arma::Mat<eT> x1_cpu(x1);
  arma::Mat<eT> x2_cpu(x2);
  arma::Mat<eT> x_orig_cpu(x_orig);

  arma::Col<eT> x1_vec = arma::vectorise(x1_cpu);
  arma::Col<eT> x2_vec = arma::vectorise(x2_cpu);
  arma::Col<eT> x_orig_vec = arma::vectorise(x_orig_cpu);
  REQUIRE( arma::approx_equal( x1_vec, x_orig_vec.subvec(0, 449), "absdiff", 1e-5) );
  REQUIRE( arma::approx_equal( x2_vec, x_orig_vec.subvec(0, 449), "absdiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("reshape_alias_larger", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x1 = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> x2 = x1;
  Mat<eT> x_orig = x1;

  x1 = reshape(x1, 15, 70);
  x2.reshape(15, 70);

  REQUIRE( x1.n_rows == 15 );
  REQUIRE( x1.n_cols == 70 );
  REQUIRE( x1.n_elem == 1050 );
  REQUIRE( x2.n_rows == 15 );
  REQUIRE( x2.n_cols == 70 );
  REQUIRE( x2.n_elem == 1050 );

  arma::Mat<eT> x1_cpu(x1);
  arma::Mat<eT> x2_cpu(x2);
  arma::Mat<eT> x_orig_cpu(x_orig);

  arma::Col<eT> x1_vec = arma::vectorise(x1_cpu);
  arma::Col<eT> x2_vec = arma::vectorise(x2_cpu);
  arma::Col<eT> x_orig_vec = arma::vectorise(x_orig_cpu);
  REQUIRE( arma::approx_equal( x1_vec.subvec(0, 899), x_orig_vec, "absdiff", 1e-5) );
  REQUIRE( arma::max( arma::max( arma::abs( x1_vec.subvec(900, 1049) ) ) ) == Approx(eT(0)).margin(1e-5) );
  REQUIRE( arma::approx_equal( x2_vec.subvec(0, 899), x_orig_vec, "absdiff", 1e-5) );
  REQUIRE( arma::max( arma::max( arma::abs( x2_vec.subvec(900, 1049) ) ) ) == Approx(eT(0)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("reshape_subview_same_size", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> y = reshape(x.submat(5, 5, 24, 24), 10, 40);

  REQUIRE( y.n_rows == 10 );
  REQUIRE( y.n_cols == 40 );
  REQUIRE( y.n_elem == 400 );

  Mat<eT> xs = x.submat(5, 5, 24, 24);
  arma::Mat<eT> xs_cpu(xs);
  arma::Mat<eT> y_cpu(y);

  arma::Col<eT> xs_vec = arma::vectorise(xs_cpu);
  arma::Col<eT> y_vec = arma::vectorise(y_cpu);
  REQUIRE( arma::approx_equal( xs_vec, y_vec, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("reshape_subview_smaller", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> y = reshape(x.submat(5, 5, 24, 24), 10, 20);

  REQUIRE( y.n_rows == 10 );
  REQUIRE( y.n_cols == 20 );
  REQUIRE( y.n_elem == 200 );

  Mat<eT> xs = x.submat(5, 5, 24, 24);
  arma::Mat<eT> xs_cpu(xs);
  arma::Mat<eT> y_cpu(y);

  arma::Col<eT> xs_vec = arma::vectorise(xs_cpu);
  arma::Col<eT> y_vec = arma::vectorise(y_cpu);
  REQUIRE( arma::approx_equal( xs_vec.subvec(0, 199), y_vec, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("reshape_subview_larger", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> y = reshape(x.submat(5, 5, 24, 24), 10, 50);

  REQUIRE( y.n_rows == 10 );
  REQUIRE( y.n_cols == 50 );
  REQUIRE( y.n_elem == 500 );

  Mat<eT> xs = x.submat(5, 5, 24, 24);
  arma::Mat<eT> xs_cpu(xs);
  arma::Mat<eT> y_cpu(y);

  arma::Col<eT> xs_vec = arma::vectorise(xs_cpu);
  arma::Col<eT> y_vec = arma::vectorise(y_cpu);
  REQUIRE( arma::approx_equal( xs_vec, y_vec.subvec(0, 399), "absdiff", 1e-5 ) );
  REQUIRE( arma::max( arma::max( arma::abs( y_vec.subvec(400, 499) ) ) ) == Approx(eT(0)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("reshape_alias_subview_same_size", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> x_orig = x;
  x = reshape(x.submat(5, 5, 24, 24), 10, 40);

  REQUIRE( x.n_rows == 10 );
  REQUIRE( x.n_cols == 40 );
  REQUIRE( x.n_elem == 400 );

  Mat<eT> xs = x_orig.submat(5, 5, 24, 24);
  arma::Mat<eT> xs_cpu(xs);
  arma::Mat<eT> x_cpu(x);

  arma::Col<eT> xs_vec = arma::vectorise(xs_cpu);
  arma::Col<eT> x_vec = arma::vectorise(x_cpu);
  REQUIRE( arma::approx_equal( xs_vec, x_vec, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("reshape_alias_subview_smaller", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> x_orig = x;
  x = reshape(x.submat(5, 5, 24, 24), 10, 20);

  REQUIRE( x.n_rows == 10 );
  REQUIRE( x.n_cols == 20 );
  REQUIRE( x.n_elem == 200 );

  Mat<eT> xs = x_orig.submat(5, 5, 24, 24);
  arma::Mat<eT> xs_cpu(xs);
  arma::Mat<eT> x_cpu(x);

  arma::Col<eT> xs_vec = arma::vectorise(xs_cpu);
  arma::Col<eT> x_vec = arma::vectorise(x_cpu);
  REQUIRE( arma::approx_equal( xs_vec.subvec(0, 199), x_vec, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("reshape_alias_subview_larger", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(10, 20));
  Mat<eT> x_orig = x;
  x = reshape(x.submat(5, 5, 24, 24), 10, 50);

  REQUIRE( x.n_rows == 10 );
  REQUIRE( x.n_cols == 50 );
  REQUIRE( x.n_elem == 500 );

  Mat<eT> xs = x_orig.submat(5, 5, 24, 24);
  arma::Mat<eT> xs_cpu(xs);
  arma::Mat<eT> x_cpu(x);

  arma::Col<eT> xs_vec = arma::vectorise(xs_cpu);
  arma::Col<eT> x_vec = arma::vectorise(x_cpu);
  REQUIRE( arma::approx_equal( xs_vec, x_vec.subvec(0, 399), "absdiff", 1e-5 ) );
  REQUIRE( arma::max( arma::max( arma::abs( x_vec.subvec(400, 499) ) ) ) == Approx(eT(0)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("reshape_from_empty", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x;
  Mat<eT> y = reshape(x, 10, 10);

  REQUIRE( y.n_rows == 10 );
  REQUIRE( y.n_cols == 10 );
  REQUIRE( y.n_elem == 100 );

  REQUIRE( max(max(abs(y))) == Approx(eT(0)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("reshape_to_empty", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(10, 20));
  Mat<eT> y = reshape(x, 0, 0);
  x.reshape(0, 2);

  REQUIRE( x.n_rows == 0 );
  REQUIRE( x.n_cols == 2 );
  REQUIRE( x.n_elem == 0 );
  REQUIRE( y.n_rows == 0 );
  REQUIRE( y.n_cols == 0 );
  REQUIRE( y.n_elem == 0 );
  }



TEMPLATE_TEST_CASE("reshape_op", "[reshape]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 10, distr_param(10, 20));
  Mat<eT> op_res = (trans(x) * 3 + 4);

  Mat<eT> y = reshape(trans(x) * 3 + 4, 8, 9);

  REQUIRE( y.n_rows == 8 );
  REQUIRE( y.n_cols == 9 );
  REQUIRE( y.n_elem == 72 );

  arma::Mat<eT> op_res_cpu(op_res);
  arma::Col<eT> op_res_vec = arma::vectorise(op_res_cpu);

  arma::Mat<eT> y_cpu(y);
  arma::Col<eT> y_vec = arma::vectorise(y_cpu);

  REQUIRE( arma::approx_equal( y_vec, op_res_vec.subvec(0, 71), "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "reshape_conv_to",
  "[reshape]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>)
  )
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  Mat<eT1> x = randi<Mat<eT1>>(5, 5, distr_param(10, 20));

  Mat<eT2> y1 = reshape(conv_to<Mat<eT2>>::from(x), 6, 7);
  Mat<eT2> y2 = conv_to<Mat<eT2>>::from(reshape(x, 6, 7));

  REQUIRE( y1.n_rows == 6 );
  REQUIRE( y1.n_cols == 7 );
  REQUIRE( y1.n_elem == 42 );
  REQUIRE( y2.n_rows == 6 );
  REQUIRE( y2.n_cols == 7 );
  REQUIRE( y2.n_elem == 42 );

  Mat<eT2> x_conv = conv_to<Mat<eT2>>::from(x);
  arma::Mat<eT2> x_conv_cpu(x_conv);
  arma::Mat<eT2> y1_cpu(y1);
  arma::Mat<eT2> y2_cpu(y2);

  arma::Col<eT2> x_conv_vec = arma::vectorise(x_conv_cpu);
  arma::Col<eT2> y1_vec = arma::vectorise(y1_cpu);
  arma::Col<eT2> y2_vec = arma::vectorise(y2_cpu);

  REQUIRE( arma::approx_equal( y1_vec.subvec(0, 24), x_conv_vec, "absdiff", 1e-5 ) );
  REQUIRE( arma::max( arma::max( arma::abs( y1_vec.subvec(25, 41) ) ) ) == eT2(0) );
  REQUIRE( arma::approx_equal( y2_vec.subvec(0, 24), x_conv_vec, "absdiff", 1e-5 ) );
  REQUIRE( arma::max( arma::max( arma::abs( y2_vec.subvec(25, 41) ) ) ) == eT2(0) );
  }
