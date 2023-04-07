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

// We don't need to test quite as much as mean(), median(), var(), or range(), because stddev() is just a wrapper for var(), pretty much.

TEMPLATE_TEST_CASE("stddev_mat_test", "[stddev]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(100, 50, distr_param(1, 10));

  Row<eT> col_vars = var(x);
  Row<eT> col_stddevs1 = stddev(x);
  Row<eT> col_stddevs2 = stddev(x, 0, 0);
  Col<eT> row_vars = var(x, 0, 1);
  Col<eT> row_stddevs = stddev(x, 0, 1);

  Row<eT> col_sqrt_vars = sqrt(col_vars);
  Col<eT> row_sqrt_vars = sqrt(row_vars);

  arma::Row<eT> col_sqrt_vars_cpu(col_sqrt_vars);
  arma::Col<eT> row_sqrt_vars_cpu(row_sqrt_vars);

  arma::Row<eT> col_stddevs1_cpu(col_stddevs1);
  arma::Row<eT> col_stddevs2_cpu(col_stddevs2);
  arma::Col<eT> row_stddevs_cpu(row_stddevs);

  REQUIRE( arma::approx_equal(col_stddevs1_cpu, col_sqrt_vars_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_stddevs2_cpu, col_sqrt_vars_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_stddevs_cpu, row_sqrt_vars_cpu, "reldiff", 1e-5) );

  // Now try without the bias correction.
  col_vars = var(x, 1);
  col_stddevs1 = stddev(x, 1);
  col_stddevs2 = stddev(x, 1, 0);
  row_vars = var(x, 1, 1);
  row_stddevs = stddev(x, 1, 1);

  col_sqrt_vars = sqrt(col_vars);
  row_sqrt_vars = sqrt(row_vars);

  col_sqrt_vars_cpu = arma::Row<eT>(col_sqrt_vars);
  row_sqrt_vars_cpu = arma::Col<eT>(row_sqrt_vars);

  col_stddevs1_cpu = arma::Row<eT>(col_stddevs1);
  col_stddevs2_cpu = arma::Row<eT>(col_stddevs2);
  row_stddevs_cpu = arma::Col<eT>(row_stddevs);

  REQUIRE( arma::approx_equal(col_stddevs1_cpu, col_sqrt_vars_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_stddevs2_cpu, col_sqrt_vars_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_stddevs_cpu, row_sqrt_vars_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("stddev_vec_test", "[stddev]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x1 = randi<Col<eT>>(10000, distr_param(1, 20));
  Row<eT> x2 = randi<Row<eT>>(50000, distr_param(1, 5));

  const eT x1_var1 = var(x1, 0);
  const eT x1_var2 = var(x1, 1);
  const eT x2_var1 = var(x2, 0);
  const eT x2_var2 = var(x2, 1);

  const eT x1_stddev1 = stddev(x1, 0);
  const eT x1_stddev2 = stddev(x1, 1);
  const eT x2_stddev1 = stddev(x2, 0);
  const eT x2_stddev2 = stddev(x2, 1);

  REQUIRE( x1_stddev1 == Approx(eT(std::sqrt(x1_var1))) );
  REQUIRE( x1_stddev2 == Approx(eT(std::sqrt(x1_var2))) );
  REQUIRE( x2_stddev1 == Approx(eT(std::sqrt(x2_var1))) );
  REQUIRE( x2_stddev2 == Approx(eT(std::sqrt(x2_var2))) );
  }



TEST_CASE("stddev_empty_test", "[stddev]")
  {
  mat x;

  vec y1 = stddev(x);
  vec y2 = stddev(x, 0, 0);
  rowvec y3 = stddev(x, 0, 0);

  REQUIRE( y1.n_elem == 0 );
  REQUIRE( y2.n_elem == 0 );
  REQUIRE( y3.n_elem == 0 );

  y1 = stddev(x, 1);
  y2 = stddev(x, 1, 0);
  y3 = stddev(x, 1, 1);

  REQUIRE( y1.n_elem == 0 );
  REQUIRE( y2.n_elem == 0 );
  REQUIRE( y3.n_elem == 0 );
  }



TEST_CASE("stddev_empty_vec_test", "[stddev]")
  {
  vec x;

  REQUIRE( stddev(x) == double(0) );
  REQUIRE( stddev(x, 1) == double(0) );
  }



TEST_CASE("stddev_one_elem_test", "[stddev]")
  {
  vec x(1);
  x(0) = 0.0;

  REQUIRE( (isnan(stddev(x)) || isinf(stddev(x))) );
  REQUIRE( stddev(x, 1) == double(0) );
  }



TEMPLATE_TEST_CASE("stddev_submat_test", "[stddev]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(100, 50, distr_param(1, 10));

  Row<eT> col_vars = var(x.submat(5, 5, 25, 30));
  Row<eT> col_stddevs1 = stddev(x.submat(5, 5, 25, 30));
  Row<eT> col_stddevs2 = stddev(x.submat(5, 5, 25, 30), 0, 0);
  Col<eT> row_vars = var(x.submat(5, 5, 25, 30), 0, 1);
  Col<eT> row_stddevs = stddev(x.submat(5, 5, 25, 30), 0, 1);

  Row<eT> col_sqrt_vars = sqrt(col_vars);
  Col<eT> row_sqrt_vars = sqrt(row_vars);

  arma::Row<eT> col_sqrt_vars_cpu(col_sqrt_vars);
  arma::Col<eT> row_sqrt_vars_cpu(row_sqrt_vars);

  arma::Row<eT> col_stddevs1_cpu(col_stddevs1);
  arma::Row<eT> col_stddevs2_cpu(col_stddevs2);
  arma::Col<eT> row_stddevs_cpu(row_stddevs);

  REQUIRE( arma::approx_equal(col_stddevs1_cpu, col_sqrt_vars_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_stddevs2_cpu, col_sqrt_vars_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_stddevs_cpu, row_sqrt_vars_cpu, "reldiff", 1e-5) );

  // Now try without the bias correction.
  col_vars = var(x.submat(5, 5, 25, 30), 1);
  col_stddevs1 = stddev(x.submat(5, 5, 25, 30), 1);
  col_stddevs2 = stddev(x.submat(5, 5, 25, 30), 1, 0);
  row_vars = var(x.submat(5, 5, 25, 30), 1, 1);
  row_stddevs = stddev(x.submat(5, 5, 25, 30), 1, 1);

  col_sqrt_vars = sqrt(col_vars);
  row_sqrt_vars = sqrt(row_vars);

  col_sqrt_vars_cpu = arma::Row<eT>(col_sqrt_vars);
  row_sqrt_vars_cpu = arma::Col<eT>(row_sqrt_vars);

  col_stddevs1_cpu = arma::Row<eT>(col_stddevs1);
  col_stddevs2_cpu = arma::Row<eT>(col_stddevs2);
  row_stddevs_cpu = arma::Col<eT>(row_stddevs);

  REQUIRE( arma::approx_equal(col_stddevs1_cpu, col_sqrt_vars_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_stddevs2_cpu, col_sqrt_vars_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_stddevs_cpu, row_sqrt_vars_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("stddev_subvec_test", "[stddev]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x1 = randi<Col<eT>>(10000, distr_param(1, 20));
  Row<eT> x2 = randi<Row<eT>>(50000, distr_param(1, 5));

  const eT x1_var1 = var(x1.subvec(515, 1103), 0);
  const eT x1_var2 = var(x1.subvec(515, 1103), 1);
  const eT x2_var1 = var(x2.subvec(515, 1103), 0);
  const eT x2_var2 = var(x2.subvec(515, 1103), 1);

  const eT x1_stddev1 = stddev(x1.subvec(515, 1103), 0);
  const eT x1_stddev2 = stddev(x1.subvec(515, 1103), 1);
  const eT x2_stddev1 = stddev(x2.subvec(515, 1103), 0);
  const eT x2_stddev2 = stddev(x2.subvec(515, 1103), 1);

  REQUIRE( x1_stddev1 == Approx(eT(std::sqrt(x1_var1))) );
  REQUIRE( x1_stddev2 == Approx(eT(std::sqrt(x1_var2))) );
  REQUIRE( x2_stddev1 == Approx(eT(std::sqrt(x2_var1))) );
  REQUIRE( x2_stddev2 == Approx(eT(std::sqrt(x2_var2))) );
  }



TEMPLATE_TEST_CASE
  (
  "stddev_conv_to",
  "[stddev]",
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

  Mat<eT1> x = randi<Mat<eT1>>(100, 50, distr_param(1, 1000));
  Mat<eT2> x_conv = conv_to<Mat<eT2>>::from(x);

  Row<eT2> col_stddevs1 = stddev(conv_to<Mat<eT2>>::from(x));
  Row<eT2> col_stddevs2 = stddev(conv_to<Mat<eT2>>::from(x), 0, 0);
  Col<eT2> row_stddevs1 = stddev(conv_to<Mat<eT2>>::from(x), 0, 1);

  REQUIRE( col_stddevs1.n_elem == 50 );
  REQUIRE( col_stddevs2.n_elem == 50 );
  REQUIRE( row_stddevs1.n_elem == 100 );

  Row<eT2> col_stddevs_ref = stddev(x_conv, 0, 0);
  Col<eT2> row_stddevs_ref = stddev(x_conv, 0, 1);

  arma::Row<eT2> col_stddevs_ref_cpu(col_stddevs_ref);
  arma::Col<eT2> row_stddevs_ref_cpu(row_stddevs_ref);

  arma::Row<eT2> col_stddevs1_cpu(col_stddevs1);
  arma::Row<eT2> col_stddevs2_cpu(col_stddevs2);
  arma::Col<eT2> row_stddevs1_cpu(row_stddevs1);

  REQUIRE( arma::approx_equal( col_stddevs1_cpu, col_stddevs_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_stddevs2_cpu, col_stddevs_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_stddevs1_cpu, row_stddevs_ref_cpu, "reldiff", 1e-5 ) );

  Row<eT2> col_stddevs3 = conv_to<Row<eT2>>::from(stddev(x));
  Row<eT2> col_stddevs4 = conv_to<Row<eT2>>::from(stddev(x, 0, 0));
  Col<eT2> row_stddevs2 = conv_to<Col<eT2>>::from(stddev(x, 0, 1));

  Row<eT1> col_stddevs_preconv = stddev(x, 0, 0);
  Col<eT1> row_stddevs_preconv = stddev(x, 0, 1);

  col_stddevs_ref = conv_to<Row<eT2>>::from(col_stddevs_preconv);
  row_stddevs_ref = conv_to<Col<eT2>>::from(row_stddevs_preconv);

  col_stddevs_ref_cpu = arma::Row<eT2>(col_stddevs_ref);
  row_stddevs_ref_cpu = arma::Col<eT2>(row_stddevs_ref);

  arma::Row<eT2> col_stddevs3_cpu(col_stddevs3);
  arma::Row<eT2> col_stddevs4_cpu(col_stddevs4);
  arma::Col<eT2> row_stddevs2_cpu(row_stddevs2);

  REQUIRE( arma::approx_equal( col_stddevs3_cpu, col_stddevs_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_stddevs4_cpu, col_stddevs_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_stddevs2_cpu, row_stddevs_ref_cpu, "reldiff", 1e-5 ) );

  // Now try without correction.

  col_stddevs1 = stddev(conv_to<Mat<eT2>>::from(x), 1);
  col_stddevs2 = stddev(conv_to<Mat<eT2>>::from(x), 1, 0);
  row_stddevs1 = stddev(conv_to<Mat<eT2>>::from(x), 1, 1);

  REQUIRE( col_stddevs1.n_elem == 50 );
  REQUIRE( col_stddevs2.n_elem == 50 );
  REQUIRE( row_stddevs1.n_elem == 100 );

  col_stddevs_ref = stddev(x_conv, 1, 0);
  row_stddevs_ref = stddev(x_conv, 1, 1);

  col_stddevs_ref_cpu = arma::Row<eT2>(col_stddevs_ref);
  row_stddevs_ref_cpu = arma::Col<eT2>(row_stddevs_ref);

  col_stddevs1_cpu = arma::Row<eT2>(col_stddevs1);
  col_stddevs2_cpu = arma::Row<eT2>(col_stddevs2);
  row_stddevs1_cpu = arma::Col<eT2>(row_stddevs1);

  REQUIRE( arma::approx_equal( col_stddevs1_cpu, col_stddevs_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_stddevs2_cpu, col_stddevs_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_stddevs1_cpu, row_stddevs_ref_cpu, "reldiff", 1e-5 ) );

  col_stddevs3 = conv_to<Row<eT2>>::from(stddev(x, 1));
  col_stddevs4 = conv_to<Row<eT2>>::from(stddev(x, 1, 0));
  row_stddevs2 = conv_to<Col<eT2>>::from(stddev(x, 1, 1));

  col_stddevs_preconv = stddev(x, 1, 0);
  row_stddevs_preconv = stddev(x, 1, 1);

  col_stddevs_ref = conv_to<Row<eT2>>::from(col_stddevs_preconv);
  row_stddevs_ref = conv_to<Col<eT2>>::from(row_stddevs_preconv);

  col_stddevs_ref_cpu = arma::Row<eT2>(col_stddevs_ref);
  row_stddevs_ref_cpu = arma::Col<eT2>(row_stddevs_ref);

  col_stddevs3_cpu = arma::Row<eT2>(col_stddevs3);
  col_stddevs4_cpu = arma::Row<eT2>(col_stddevs4);
  row_stddevs2_cpu = arma::Col<eT2>(row_stddevs2);

  REQUIRE( arma::approx_equal( col_stddevs3_cpu, col_stddevs_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_stddevs4_cpu, col_stddevs_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_stddevs2_cpu, row_stddevs_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "stddev_vec_conv_to",
  "[stddev]",
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

  Col<eT1> x = randi<Col<eT1>>(10000, distr_param(1, 50));
  Col<eT2> x_conv = conv_to<Col<eT2>>::from(x);

  const eT2 stddev_val = stddev(conv_to<Col<eT2>>::from(x));
  const eT2 stddev_ref_val = stddev(x_conv);

  REQUIRE( stddev_val == Approx(stddev_ref_val) );

  // Try without correction.
  const eT2 stddev_val2 = stddev(conv_to<Col<eT2>>::from(x), 1);
  const eT2 stddev_ref_val2 = stddev(x_conv, 1);

  REQUIRE( stddev_val2 == Approx(stddev_ref_val2) );
  }
