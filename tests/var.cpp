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

TEMPLATE_TEST_CASE("simple_var_test", "[var]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(15, 11);
  x.ones();
  for (uword c = 1; c < 11; ++c)
    {
    x.col(c) *= (c + 1);
    }

  Row<eT> col_vars = var(x);
  Row<eT> col_vars2 = var(x, 0, 0);
  Col<eT> row_vars = var(x, 0, 1);

  REQUIRE( col_vars.n_elem == 11 );
  REQUIRE( col_vars2.n_elem == 11 );
  REQUIRE( row_vars.n_elem == 15 );

  for (uword i = 0; i < 11; ++i)
    {
    // Since all the elements are the same in every column, the variance is 0.
    REQUIRE( eT(col_vars[i]) == Approx(eT(0)).margin(1e-5) );
    REQUIRE( eT(col_vars2[i]) == Approx(eT(0)).margin(1e-5) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    // Values are evenly spaced between 1 and 11; the variance is 11.
    REQUIRE( eT(row_vars[i]) == Approx(eT(11)) );
    }

  // Now test adjusted variants.
  col_vars2 = var(x, 1, 0);
  row_vars = var(x, 1, 1);

  for (uword i = 0; i < 11; ++i)
    {
    REQUIRE( eT(col_vars2[i]) == Approx(eT(0)).margin(1e-5) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    REQUIRE( eT(row_vars[i]) == Approx(eT(10)) );
    }
  }



TEMPLATE_TEST_CASE("random_var_test", "[var]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(500, 700);
  x.randu();

  Row<eT> col_vars = var(x);
  Row<eT> col_vars2 = var(x, 0, 0);
  Col<eT> row_vars = var(x, 0, 1);

  REQUIRE( col_vars.n_elem == 700 );
  REQUIRE( col_vars2.n_elem == 700 );
  REQUIRE( row_vars.n_elem == 500 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_vars_ref_cpu = arma::var(x_cpu, 0, 0);
  arma::Col<eT> row_vars_ref_cpu = arma::var(x_cpu, 0, 1);

  arma::Row<eT> col_vars_cpu(col_vars);
  arma::Row<eT> col_vars2_cpu(col_vars2);
  arma::Col<eT> row_vars_cpu(row_vars);

  REQUIRE( arma::approx_equal(col_vars_cpu, col_vars_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_vars2_cpu, col_vars_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_vars_cpu, row_vars_ref_cpu, "reldiff", 1e-5) );

  // Now test adjusted variants.
  col_vars2 = var(x, 1, 0);
  row_vars = var(x, 1, 1);

  col_vars_ref_cpu = arma::var(x_cpu, 1, 0);
  row_vars_ref_cpu = arma::var(x_cpu, 1, 1);

  col_vars2_cpu = arma::Row<eT>(col_vars2);
  row_vars_cpu = arma::Col<eT>(row_vars);

  REQUIRE( arma::approx_equal(col_vars2_cpu, col_vars_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_vars_cpu, row_vars_ref_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("randi_var_test", "[var]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(500, 700, distr_param(1, 100));

  Row<eT> col_vars = var(x);
  Row<eT> col_vars2 = var(x, 0, 0);
  Col<eT> row_vars = var(x, 0, 1);

  REQUIRE( col_vars.n_elem == 700 );
  REQUIRE( col_vars2.n_elem == 700 );
  REQUIRE( row_vars.n_elem == 500 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_vars_ref_cpu = arma::var(x_cpu, 0, 0);
  arma::Col<eT> row_vars_ref_cpu = arma::var(x_cpu, 0, 1);

  arma::Row<eT> col_vars_cpu(col_vars);
  arma::Row<eT> col_vars2_cpu(col_vars2);
  arma::Col<eT> row_vars_cpu(row_vars);

  // Within a margin of 1 (for integer issues).
  REQUIRE( arma::approx_equal(col_vars_cpu, col_vars_ref_cpu, "absdiff", 1) );
  REQUIRE( arma::approx_equal(col_vars2_cpu, col_vars_ref_cpu, "absdiff", 1) );
  REQUIRE( arma::approx_equal(row_vars_cpu, row_vars_ref_cpu, "absdiff", 1) );

  // Now test adjusted variants.
  col_vars2 = var(x, 1, 0);
  row_vars = var(x, 1, 1);

  col_vars_ref_cpu = arma::var(x_cpu, 1, 0);
  row_vars_ref_cpu = arma::var(x_cpu, 1, 1);

  col_vars2_cpu = arma::Row<eT>(col_vars2);
  row_vars_cpu = arma::Col<eT>(row_vars);

  // Within a margin of 1 (for integer issues).
  REQUIRE( arma::approx_equal(col_vars2_cpu, col_vars_ref_cpu, "absdiff", 1) );
  REQUIRE( arma::approx_equal(row_vars_cpu, row_vars_ref_cpu, "absdiff", 1) );
  }



TEMPLATE_TEST_CASE("simple_subview_var_test", "[var]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(15, 11);
  x.ones();
  for(uword c = 1; c < 11; ++c)
    {
    x.col(c) *= (c + 1);
    }

  Mat<eT> y(20, 25);
  y.zeros();
  y.submat(5, 5, 19, 15) = x;

  Row<eT> col_vars = var(y.submat(5, 5, 19, 15));
  Row<eT> col_vars2 = var(y.submat(5, 5, 19, 15), 0);
  Col<eT> row_vars = var(y.submat(5, 5, 19, 15), 0, 1);

  REQUIRE( col_vars.n_elem == 11 );
  REQUIRE( col_vars2.n_elem == 11 );
  REQUIRE( row_vars.n_elem == 15 );

  for (uword i = 0; i < 11; ++i)
    {
    REQUIRE( eT(col_vars[i]) == Approx(eT(0)).margin(1e-5) );
    REQUIRE( eT(col_vars2[i]) == Approx(eT(0)).margin(1e-5) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    // Values are between 1 and 11; the variance is 11.
    REQUIRE( eT(row_vars[i]) == Approx(eT(11)) );
    }

  // Now test unadjusted variance.
  col_vars2 = var(y.submat(5, 5, 19, 15), 1, 0);
  row_vars = var(y.submat(5, 5, 19, 15), 1, 1);

  REQUIRE( col_vars2.n_elem == 11 );
  REQUIRE( row_vars.n_elem == 15 );

  for (uword i = 0; i < 11; ++i)
    {
    REQUIRE( eT(col_vars2[i]) == Approx(eT(0)).margin(1e-5) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    REQUIRE( eT(row_vars[i]) == Approx(eT(10)) );
    }
  }



TEMPLATE_TEST_CASE("random_subview_var_test", "[var]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(500, 700);
  x.randu();

  Row<eT> col_vars = var(x.submat(10, 10, 490, 690));
  Row<eT> col_vars2 = var(x.submat(10, 10, 490, 690), 0);
  Col<eT> row_vars = var(x.submat(10, 10, 490, 690), 0, 1);

  REQUIRE( col_vars.n_elem == 681 );
  REQUIRE( col_vars2.n_elem == 681 );
  REQUIRE( row_vars.n_elem == 481 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_vars_ref_cpu = arma::var(x_cpu.submat(10, 10, 490, 690), 0, 0);
  arma::Col<eT> row_vars_ref_cpu = arma::var(x_cpu.submat(10, 10, 490, 690), 0, 1);

  arma::Row<eT> col_vars_cpu(col_vars);
  arma::Row<eT> col_vars2_cpu(col_vars2);
  arma::Col<eT> row_vars_cpu(row_vars);

  REQUIRE( arma::approx_equal(col_vars_cpu, col_vars_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_vars2_cpu, col_vars_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_vars_cpu, row_vars_ref_cpu, "reldiff", 1e-5) );

  col_vars2 = var(x.submat(10, 10, 490, 690), 1, 0);
  row_vars = var(x.submat(10, 10, 490, 690), 1, 1);

  col_vars_ref_cpu = arma::var(x_cpu.submat(10, 10, 490, 690), 1, 0);
  row_vars_ref_cpu = arma::var(x_cpu.submat(10, 10, 490, 690), 1, 1);

  col_vars2_cpu = arma::Row<eT>(col_vars2);
  row_vars_cpu = arma::Col<eT>(row_vars);

  REQUIRE( arma::approx_equal(col_vars2_cpu, col_vars_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_vars_cpu, row_vars_ref_cpu, "reldiff", 1e-5) );
  }



TEST_CASE("empty_var_test", "[var]")
  {
  mat x;
  rowvec m1 = var(x);
  rowvec m2 = var(x, 0, 0);
  vec m3 = var(x, 0, 1);

  REQUIRE( m1.n_elem == 0 );
  REQUIRE( m2.n_elem == 0 );
  REQUIRE( m3.n_elem == 0 );

  m2 = var(x, 1, 0);
  m3 = var(x, 1, 1);

  REQUIRE( m2.n_elem == 0 );
  REQUIRE( m3.n_elem == 0 );
  }



TEMPLATE_TEST_CASE("simple_var_vec_test", "[var]", float, double)
  {
  typedef TestType eT;

  Col<eT> x(1000);
  x.ones();

  REQUIRE( eT(var(x)) == Approx(eT(0)).margin(1e-5) );
  REQUIRE( eT(var(x, 1)) == Approx(eT(0)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("simple_var_vec_test_2", "[var]", float, double)
  {
  typedef TestType eT;

  Col<eT> x = linspace<Col<eT>>(1, 11, 11);

  REQUIRE( var(x) == Approx(eT(11)) );
  REQUIRE( var(x, 1) == Approx(eT(10)) );
  }



TEMPLATE_TEST_CASE("random_var_vec_test", "[var]", float, double)
  {
  typedef TestType eT;

  Col<eT> x(100000);
  x.randu();

  arma::Col<eT> x_cpu(x);

  const eT var_val = var(x);
  const eT cpu_var_val = arma::var(x_cpu);

  REQUIRE( var_val == Approx(cpu_var_val) );

  const eT var_val2 = var(x, 1);
  const eT cpu_var_val2 = arma::var(x_cpu, 1);

  REQUIRE( var_val2 == Approx(cpu_var_val2) );
  }



TEMPLATE_TEST_CASE("randi_var_vec_test", "[var]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x = randi<Col<eT>>(100000, distr_param(1, 100));

  arma::Col<eT> x_cpu(x);

  const eT var_val = var(x);
  const eT cpu_var_val = arma::var(x_cpu);

  REQUIRE( var_val == Approx(cpu_var_val).margin(1) );

  const eT var_val2 = var(x, 1);
  const eT cpu_var_val2 = arma::var(x_cpu, 1);

  REQUIRE( var_val2 == Approx(cpu_var_val2).margin(1) );
  }



TEST_CASE("empty_var_vec_test", "[var]")
  {
  vec x;
  const double var_val = var(x);
  const double var_val2 = var(x, 1);

  REQUIRE( var_val == 0.0 );
  REQUIRE( var_val2 == 0.0 );
  }



TEST_CASE("var_op_test", "[var]")
  {
  mat x(50, 50);
  x.randu();

  rowvec m1 = var(2 * x + 3);
  mat y = 2 * x + 3;
  rowvec m2 = var(y);

  REQUIRE( m1.n_elem == m2.n_elem );
  for (size_t i = 0; i < m1.n_elem; ++i)
    {
    REQUIRE( double(m1[i]) == Approx(double(m2[i])) );
    }

  m1 = var(2 * x + 3, 1);
  m2 = var(y, 1);

  REQUIRE( m1.n_elem == m2.n_elem );
  for (size_t i = 0; i < m1.n_elem; ++i)
    {
    REQUIRE( double(m1[i]) == Approx(double(m2[i])) );
    }
  }



// single element var
TEST_CASE("single_element_var", "[var]")
  {
  vec x(1);
  x(0) = 3.0;

  REQUIRE( (isinf(var(x)) || isnan(var(x))) ); // adjusted will divide by 0
  REQUIRE( var(x, 1) == 0.0 );
  }



TEMPLATE_TEST_CASE("var_subvec", "[var]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(15, 11);
  x.ones();
  for (uword c = 1; c < 11; ++c)
    {
    x.col(c) *= (c + 1);
    }

  static_assert( is_coot_type<subview_col<double>>::value == true, "subview row is a coot type");
  static_assert( resolves_to_vector<subview_col<double>>::value == true, "subview row resolves to vector");

  for (uword i = 0; i < 11; ++i)
    {
    // Since all the elements are the same in every column, the variance is 0.
    REQUIRE( eT(var(x.col(i), 0)) == Approx(eT(0)).margin(1e-5) );
    REQUIRE( eT(var(x.col(i), 1)) == Approx(eT(0)).margin(1e-5) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    // Values are evenly spaced between 1 and 11; the variance is 11 or 10 depending on whether adjustment is used.
    REQUIRE( eT(var(x.row(i), 0)) == Approx(eT(11)) );
    REQUIRE( eT(var(x.row(i), 1)) == Approx(eT(10)) );
    }
  }



TEMPLATE_TEST_CASE
  (
  "var_conv_to",
  "[var]",
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

  Row<eT2> col_vars1 = var(conv_to<Mat<eT2>>::from(x));
  Row<eT2> col_vars2 = var(conv_to<Mat<eT2>>::from(x), 0, 0);
  Col<eT2> row_vars1 = var(conv_to<Mat<eT2>>::from(x), 0, 1);

  REQUIRE( col_vars1.n_elem == 50 );
  REQUIRE( col_vars2.n_elem == 50 );
  REQUIRE( row_vars1.n_elem == 100 );

  Row<eT2> col_vars_ref = var(x_conv, 0, 0);
  Col<eT2> row_vars_ref = var(x_conv, 0, 1);

  arma::Row<eT2> col_vars_ref_cpu(col_vars_ref);
  arma::Col<eT2> row_vars_ref_cpu(row_vars_ref);

  arma::Row<eT2> col_vars1_cpu(col_vars1);
  arma::Row<eT2> col_vars2_cpu(col_vars2);
  arma::Col<eT2> row_vars1_cpu(row_vars1);

  REQUIRE( arma::approx_equal( col_vars1_cpu, col_vars_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_vars2_cpu, col_vars_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_vars1_cpu, row_vars_ref_cpu, "reldiff", 1e-5 ) );

  Row<eT2> col_vars3 = conv_to<Row<eT2>>::from(var(x));
  Row<eT2> col_vars4 = conv_to<Row<eT2>>::from(var(x, 0, 0));
  Col<eT2> row_vars2 = conv_to<Col<eT2>>::from(var(x, 0, 1));

  Row<eT1> col_vars_preconv = var(x, 0, 0);
  Col<eT1> row_vars_preconv = var(x, 0, 1);

  col_vars_ref = conv_to<Row<eT2>>::from(col_vars_preconv);
  row_vars_ref = conv_to<Col<eT2>>::from(row_vars_preconv);

  col_vars_ref_cpu = arma::Row<eT2>(col_vars_ref);
  row_vars_ref_cpu = arma::Col<eT2>(row_vars_ref);

  arma::Row<eT2> col_vars3_cpu(col_vars3);
  arma::Row<eT2> col_vars4_cpu(col_vars4);
  arma::Col<eT2> row_vars2_cpu(row_vars2);

  REQUIRE( arma::approx_equal( col_vars3_cpu, col_vars_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_vars4_cpu, col_vars_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_vars2_cpu, row_vars_ref_cpu, "reldiff", 1e-5 ) );

  // Now try without correction.

  col_vars1 = var(conv_to<Mat<eT2>>::from(x), 1);
  col_vars2 = var(conv_to<Mat<eT2>>::from(x), 1, 0);
  row_vars1 = var(conv_to<Mat<eT2>>::from(x), 1, 1);

  REQUIRE( col_vars1.n_elem == 50 );
  REQUIRE( col_vars2.n_elem == 50 );
  REQUIRE( row_vars1.n_elem == 100 );

  col_vars_ref = var(x_conv, 1, 0);
  row_vars_ref = var(x_conv, 1, 1);

  col_vars_ref_cpu = arma::Row<eT2>(col_vars_ref);
  row_vars_ref_cpu = arma::Col<eT2>(row_vars_ref);

  col_vars1_cpu = arma::Row<eT2>(col_vars1);
  col_vars2_cpu = arma::Row<eT2>(col_vars2);
  row_vars1_cpu = arma::Col<eT2>(row_vars1);

  REQUIRE( arma::approx_equal( col_vars1_cpu, col_vars_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_vars2_cpu, col_vars_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_vars1_cpu, row_vars_ref_cpu, "reldiff", 1e-5 ) );

  col_vars3 = conv_to<Row<eT2>>::from(var(x, 1));
  col_vars4 = conv_to<Row<eT2>>::from(var(x, 1, 0));
  row_vars2 = conv_to<Col<eT2>>::from(var(x, 1, 1));

  col_vars_preconv = var(x, 1, 0);
  row_vars_preconv = var(x, 1, 1);

  col_vars_ref = conv_to<Row<eT2>>::from(col_vars_preconv);
  row_vars_ref = conv_to<Col<eT2>>::from(row_vars_preconv);

  col_vars_ref_cpu = arma::Row<eT2>(col_vars_ref);
  row_vars_ref_cpu = arma::Col<eT2>(row_vars_ref);

  col_vars3_cpu = arma::Row<eT2>(col_vars3);
  col_vars4_cpu = arma::Row<eT2>(col_vars4);
  row_vars2_cpu = arma::Col<eT2>(row_vars2);

  REQUIRE( arma::approx_equal( col_vars3_cpu, col_vars_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_vars4_cpu, col_vars_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_vars2_cpu, row_vars_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "var_vec_conv_to",
  "[var]",
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

  const eT2 var_val = var(conv_to<Col<eT2>>::from(x));
  const eT2 var_ref_val = var(x_conv);

  REQUIRE( var_val == Approx(var_ref_val) );

  // Try without correction.
  const eT2 var_val2 = var(conv_to<Col<eT2>>::from(x), 1);
  const eT2 var_ref_val2 = var(x_conv, 1);

  REQUIRE( var_val2 == Approx(var_ref_val2) );
  }
