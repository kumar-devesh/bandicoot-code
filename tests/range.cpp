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

TEMPLATE_TEST_CASE("simple_range_test", "[range]", u32, s32, u64, s64, float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(15, 11);
  x.ones();
  for (uword c = 1; c < 11; ++c)
    {
    x.col(c) *= (c + 1);
    }

  Row<eT> col_ranges = range(x);
  Row<eT> col_ranges2 = range(x, 0);
  Col<eT> row_ranges = range(x, 1);

  REQUIRE( col_ranges.n_elem == 11 );
  REQUIRE( col_ranges2.n_elem == 11 );
  REQUIRE( row_ranges.n_elem == 15 );

  for (uword i = 0; i < 11; ++i)
    {
    // Since all the elements are the same in every column, the range is 0.
    REQUIRE( eT(col_ranges[i]) == Approx(eT(0)) );
    REQUIRE( eT(col_ranges2[i]) == Approx(eT(0)) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    // Values are between 1 and 11; the range is therefore 10.
    REQUIRE( eT(row_ranges[i]) == Approx(eT(10)) );
    }
  }



TEMPLATE_TEST_CASE("random_range_test", "[range]", u32, s32, u64, s64, float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(500, 700);
  x.randu();

  Row<eT> col_ranges = range(x);
  Row<eT> col_ranges2 = range(x, 0);
  Col<eT> row_ranges = range(x, 1);

  REQUIRE( col_ranges.n_elem == 700 );
  REQUIRE( col_ranges2.n_elem == 700 );
  REQUIRE( row_ranges.n_elem == 500 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_ranges_ref_cpu = arma::range(x_cpu, 0);
  arma::Col<eT> row_ranges_ref_cpu = arma::range(x_cpu, 1);

  arma::Row<eT> col_ranges_cpu(col_ranges);
  arma::Row<eT> col_ranges2_cpu(col_ranges2);
  arma::Col<eT> row_ranges_cpu(row_ranges);

  REQUIRE( arma::approx_equal(col_ranges_cpu, col_ranges_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_ranges2_cpu, col_ranges_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_ranges_cpu, row_ranges_ref_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("simple_subview_range_test", "[range]", u32, s32, u64, s64, float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(15, 10);
  x.ones();
  for(uword c = 1; c < 10; ++c)
    {
    x.col(c) *= (c + 2);
    }

  Mat<eT> y(20, 25);
  y.zeros();
  y.submat(5, 5, 19, 14) = x;

  Row<eT> col_ranges = range(y.submat(5, 5, 19, 14));
  Row<eT> col_ranges2 = range(y.submat(5, 5, 19, 14), 0);
  Col<eT> row_ranges = range(y.submat(5, 5, 19, 14), 1);

  REQUIRE( col_ranges.n_elem == 10 );
  REQUIRE( col_ranges2.n_elem == 10 );
  REQUIRE( row_ranges.n_elem == 15 );

  for (uword i = 0; i < 10; ++i)
    {
    // Since the columns all have the same value, the range is 0.
    REQUIRE( eT(col_ranges[i]) == Approx(eT(0)) );
    REQUIRE( eT(col_ranges2[i]) == Approx(eT(0)) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    // Values are between 2 and 11; the range is therefore 10.
    REQUIRE( eT(row_ranges[i]) == Approx(eT(10)) );
    }
  }



TEMPLATE_TEST_CASE("random_subview_range_test", "[range]", u32, s32, u64, s64, float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(500, 700);
  x.randu();

  Row<eT> col_ranges = range(x.submat(10, 10, 490, 690));
  Row<eT> col_ranges2 = range(x.submat(10, 10, 490, 690), 0);
  Col<eT> row_ranges = range(x.submat(10, 10, 490, 690), 1);

  REQUIRE( col_ranges.n_elem == 681 );
  REQUIRE( col_ranges2.n_elem == 681 );
  REQUIRE( row_ranges.n_elem == 481 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_ranges_ref_cpu = arma::range(x_cpu.submat(10, 10, 490, 690), 0);
  arma::Col<eT> row_ranges_ref_cpu = arma::range(x_cpu.submat(10, 10, 490, 690), 1);

  arma::Row<eT> col_ranges_cpu(col_ranges);
  arma::Row<eT> col_ranges2_cpu(col_ranges2);
  arma::Col<eT> row_ranges_cpu(row_ranges);

  REQUIRE( arma::approx_equal(col_ranges_cpu, col_ranges_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_ranges2_cpu, col_ranges_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_ranges_cpu, row_ranges_ref_cpu, "reldiff", 1e-5) );
  }



TEST_CASE("empty_range_test", "[range]")
  {
  fmat x;
  frowvec m1 = range(x);
  frowvec m2 = range(x, 0);
  fvec m3 = range(x, 1);

  REQUIRE( m1.n_elem == 0 );
  REQUIRE( m2.n_elem == 0 );
  REQUIRE( m3.n_elem == 0 );
  }



TEMPLATE_TEST_CASE("simple_range_vec_test", "[range]", u32, s32, u64, s64, float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(1000);
  x.ones();

  REQUIRE( eT(range(x)) == Approx(eT(0)) );
  }



TEMPLATE_TEST_CASE("random_range_vec_test", "[range]", u32, s32, u64, s64, float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(100000);
  x.randu();

  arma::Col<eT> x_cpu(x);

  const eT range_val = range(x);
  const eT cpu_range_val = arma::range(x_cpu);

  REQUIRE( range_val == Approx(cpu_range_val) );
  }



TEST_CASE("empty_range_vec_test", "[range]")
  {
  fvec x;
  const float range_val = range(x);

  REQUIRE( range_val == 0.0 );
  }



TEST_CASE("range_op_test", "[range]")
  {
  fmat x(50, 50);
  x.randu();

  frowvec m1 = range(2 * x + 3);
  fmat y = 2 * x + 3;
  frowvec m2 = range(y);

  REQUIRE( m1.n_elem == m2.n_elem );
  for (size_t i = 0; i < m1.n_elem; ++i)
    {
    REQUIRE( float(m1[i]) == Approx(float(m2[i])) );
    }
  }



TEMPLATE_TEST_CASE
  (
  "range_conv_to",
  "[range]",
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

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(100, 50, distr_param(1, 1000));
  Mat<eT2> x_conv = conv_to<Mat<eT2>>::from(x);

  Row<eT2> col_ranges1 = range(conv_to<Mat<eT2>>::from(x));
  Row<eT2> col_ranges2 = range(conv_to<Mat<eT2>>::from(x), 0);
  Col<eT2> row_ranges1 = range(conv_to<Mat<eT2>>::from(x), 1);

  REQUIRE( col_ranges1.n_elem == 50 );
  REQUIRE( col_ranges2.n_elem == 50 );
  REQUIRE( row_ranges1.n_elem == 100 );

  Row<eT2> col_ranges_ref = range(x_conv, 0);
  Col<eT2> row_ranges_ref = range(x_conv, 1);

  arma::Row<eT2> col_ranges_ref_cpu(col_ranges_ref);
  arma::Col<eT2> row_ranges_ref_cpu(row_ranges_ref);

  arma::Row<eT2> col_ranges1_cpu(col_ranges1);
  arma::Row<eT2> col_ranges2_cpu(col_ranges2);
  arma::Col<eT2> row_ranges1_cpu(row_ranges1);

  REQUIRE( arma::approx_equal( col_ranges1_cpu, col_ranges_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_ranges2_cpu, col_ranges_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_ranges1_cpu, row_ranges_ref_cpu, "reldiff", 1e-5 ) );

  Row<eT2> col_ranges3 = conv_to<Row<eT2>>::from(range(x));
  Row<eT2> col_ranges4 = conv_to<Row<eT2>>::from(range(x, 0));
  Col<eT2> row_ranges2 = conv_to<Col<eT2>>::from(range(x, 1));

  Row<eT1> col_ranges_preconv = range(x, 0);
  Col<eT1> row_ranges_preconv = range(x, 1);

  col_ranges_ref = conv_to<Row<eT2>>::from(col_ranges_preconv);
  row_ranges_ref = conv_to<Col<eT2>>::from(row_ranges_preconv);

  col_ranges_ref_cpu = arma::Row<eT2>(col_ranges_ref);
  row_ranges_ref_cpu = arma::Col<eT2>(row_ranges_ref);

  arma::Row<eT2> col_ranges3_cpu(col_ranges3);
  arma::Row<eT2> col_ranges4_cpu(col_ranges4);
  arma::Col<eT2> row_ranges2_cpu(row_ranges2);

  REQUIRE( arma::approx_equal( col_ranges3_cpu, col_ranges_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( col_ranges4_cpu, col_ranges_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( row_ranges2_cpu, row_ranges_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "range_vec_conv_to",
  "[range]",
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

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Col<eT1> x = randi<Col<eT1>>(10000, distr_param(1, 500000));
  Col<eT2> x_conv = conv_to<Col<eT2>>::from(x);

  const eT2 range_val = range(conv_to<Col<eT1>>::from(x));
  const eT2 range_ref_val = range(x_conv);

  REQUIRE( range_val == Approx(range_ref_val) );
  }
