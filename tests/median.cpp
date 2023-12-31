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

#include <armadillo>
#include <bandicoot>
#include "catch.hpp"

using namespace coot;

TEMPLATE_TEST_CASE("simple_median_test", "[median]", float, double, u32, s32, u64, s64)
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

  Row<eT> col_medians = median(x);
  Row<eT> col_medians2 = median(x, 0);
  Col<eT> row_medians = median(x, 1);

  REQUIRE( col_medians.n_elem == 11 );
  REQUIRE( col_medians2.n_elem == 11 );
  REQUIRE( row_medians.n_elem == 15 );

  for (uword i = 0; i < 11; ++i)
    {
    // Since all the elements are the same in every column, the median is the element.
    REQUIRE( eT(col_medians[i]) == Approx(eT(i + 1)) );
    REQUIRE( eT(col_medians2[i]) == Approx(eT(i + 1)) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    // Values are between 1 and 12; the median is therefore 6.
    REQUIRE( eT(row_medians[i]) == Approx(eT(6)) );
    }
  }



TEMPLATE_TEST_CASE("random_median_test", "[median]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(500, 700);
  x.randu();

  Row<eT> col_medians = median(x);
  Row<eT> col_medians2 = median(x, 0);
  Col<eT> row_medians = median(x, 1);

  REQUIRE( col_medians.n_elem == 700 );
  REQUIRE( col_medians2.n_elem == 700 );
  REQUIRE( row_medians.n_elem == 500 );

  arma::Mat<eT> x_cpu(x);
  arma::Col<eT> col1 = arma::sort(x_cpu.col(0));

  arma::Row<eT> col_medians_ref_cpu = arma::median(x_cpu, 0);
  arma::Col<eT> row_medians_ref_cpu = arma::median(x_cpu, 1);

  arma::Row<eT> col_medians_cpu(col_medians);
  arma::Row<eT> col_medians2_cpu(col_medians2);
  arma::Col<eT> row_medians_cpu(row_medians);

  REQUIRE( arma::approx_equal(col_medians_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_medians2_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_medians_cpu, row_medians_ref_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("random_median_randi_test", "[median]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(507, 701, distr_param(0, 1000000));

  Row<eT> col_medians = median(x);
  Row<eT> col_medians2 = median(x, 0);
  Col<eT> row_medians = median(x, 1);

  REQUIRE( col_medians.n_elem == 701 );
  REQUIRE( col_medians2.n_elem == 701 );
  REQUIRE( row_medians.n_elem == 507 );

  arma::Mat<eT> x_cpu(x);
  arma::Col<eT> col1 = arma::sort(x_cpu.col(0));

  arma::Row<eT> col_medians_ref_cpu = arma::median(x_cpu, 0);
  arma::Col<eT> row_medians_ref_cpu = arma::median(x_cpu, 1);

  arma::Row<eT> col_medians_cpu(col_medians);
  arma::Row<eT> col_medians2_cpu(col_medians2);
  arma::Col<eT> row_medians_cpu(row_medians);

  REQUIRE( arma::approx_equal(col_medians_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_medians2_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_medians_cpu, row_medians_ref_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("simple_subview_median_test", "[median]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(15, 11);
  x.ones();
  for(uword c = 1; c < 11; ++c)
    {
    x.col(c) *= (c + 1);
    }

  Mat<eT> y(20, 25);
  y.zeros();
  y.submat(5, 5, 19, 15) = x;

  Row<eT> col_medians = median(y.submat(5, 5, 19, 15));
  Row<eT> col_medians2 = median(y.submat(5, 5, 19, 15), 0);
  Col<eT> row_medians = median(y.submat(5, 5, 19, 15), 1);

  REQUIRE( col_medians.n_elem == 11 );
  REQUIRE( col_medians2.n_elem == 11 );
  REQUIRE( row_medians.n_elem == 15 );

  for (uword i = 0; i < 11; ++i)
    {
    REQUIRE( eT(col_medians[i]) == Approx(eT(i + 1)) );
    REQUIRE( eT(col_medians2[i]) == Approx(eT(i + 1)) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    // Values are between 1 and 11; the median is 6.
    REQUIRE( eT(row_medians[i]) == Approx(eT(6)) );
    }
  }



TEMPLATE_TEST_CASE("random_subview_median_test", "[median]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(500, 700);
  x.randu();

  Row<eT> col_medians = median(x.submat(10, 10, 490, 690));
  Row<eT> col_medians2 = median(x.submat(10, 10, 490, 690), 0);
  Col<eT> row_medians = median(x.submat(10, 10, 490, 690), 1);

  REQUIRE( col_medians.n_elem == 681 );
  REQUIRE( col_medians2.n_elem == 681 );
  REQUIRE( row_medians.n_elem == 481 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_medians_ref_cpu = arma::median(x_cpu.submat(10, 10, 490, 690), 0);
  arma::Col<eT> row_medians_ref_cpu = arma::median(x_cpu.submat(10, 10, 490, 690), 1);

  arma::Row<eT> col_medians_cpu(col_medians);
  arma::Row<eT> col_medians2_cpu(col_medians2);
  arma::Col<eT> row_medians_cpu(row_medians);

  REQUIRE( arma::approx_equal(col_medians_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_medians2_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_medians_cpu, row_medians_ref_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("random_subview_median_randi_test", "[median]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(500, 700, distr_param(0, 1000000));

  Row<eT> col_medians = median(x.submat(10, 10, 490, 690));
  Row<eT> col_medians2 = median(x.submat(10, 10, 490, 690), 0);
  Col<eT> row_medians = median(x.submat(10, 10, 490, 690), 1);

  REQUIRE( col_medians.n_elem == 681 );
  REQUIRE( col_medians2.n_elem == 681 );
  REQUIRE( row_medians.n_elem == 481 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_medians_ref_cpu = arma::median(x_cpu.submat(10, 10, 490, 690), 0);
  arma::Col<eT> row_medians_ref_cpu = arma::median(x_cpu.submat(10, 10, 490, 690), 1);

  arma::Row<eT> col_medians_cpu(col_medians);
  arma::Row<eT> col_medians2_cpu(col_medians2);
  arma::Col<eT> row_medians_cpu(row_medians);

  REQUIRE( arma::approx_equal(col_medians_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_medians2_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_medians_cpu, row_medians_ref_cpu, "reldiff", 1e-5) );
  }



TEST_CASE("empty_median_test", "[median]")
  {
  fmat x;
  frowvec m1 = median(x);
  frowvec m2 = median(x, 0);
  fvec m3 = median(x, 1);

  REQUIRE( m1.n_elem == 0 );
  REQUIRE( m2.n_elem == 0 );
  REQUIRE( m3.n_elem == 0 );
  }



TEMPLATE_TEST_CASE("simple_median_vec_test", "[median]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(500);
  x.ones();

  REQUIRE( eT(median(x)) == Approx(eT(1)) );
  }



TEMPLATE_TEST_CASE("random_median_vec_test", "[median]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(8521);
  x.randu();
  x -= 0.5;

  arma::Col<eT> x_cpu(x);

  const eT median_val = median(x);
  const eT cpu_median_val = arma::median(x_cpu);

  REQUIRE( median_val == Approx(cpu_median_val) );
  }



TEMPLATE_TEST_CASE("random_median_vec_randi_neg_test", "[median]", float, double, s32, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(15, distr_param(-1000, 1000));

  arma::Col<eT> x_cpu(x);

  const eT median_val = median(x);
  const eT cpu_median_val = arma::median(x_cpu);

  REQUIRE( median_val == Approx(cpu_median_val) );
  }



TEMPLATE_TEST_CASE("random_median_vec_randi_test", "[median]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(150001, distr_param(0, 1000000));

  arma::Col<eT> x_cpu(x);

  const eT median_val = median(x);
  const eT cpu_median_val = arma::median(x_cpu);

  REQUIRE( median_val == Approx(cpu_median_val) );
  }



TEST_CASE("empty_median_vec_test", "[median]")
  {
  fvec x;
  const float median_val = median(x);

  REQUIRE( median_val == 0.0 );
  }



TEST_CASE("median_op_test", "[median]")
  {
  fmat x(50, 50);
  x.randu();

  frowvec m1 = median(2 * x + 3);
  fmat y = 2 * x + 3;
  frowvec m2 = median(y);

  REQUIRE( m1.n_elem == m2.n_elem );
  for (size_t i = 0; i < m1.n_elem; ++i)
    {
    REQUIRE( float(m1[i]) == Approx(float(m2[i])) );
    }
  }



TEMPLATE_TEST_CASE("signed_median_integer_test", "[median]", s32, s64)
  {
  typedef TestType eT;

  Row<eT> x = randi<Row<eT>>(10000, distr_param(-100, 100));

  const eT median_val = median(x);

  arma::Row<eT> x_cpu(x);
  const eT cpu_median_val = arma::median(x_cpu);

  REQUIRE( median_val == Approx(cpu_median_val).margin(1) );
  }



TEMPLATE_TEST_CASE("signed_median_rowwise_colwise_integer_test", "[median]", s32, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(10, 4, distr_param(-100, 100));

  Mat<eT> m1 = median(x, 0);
  Mat<eT> m2 = median(x, 1);

  arma::Mat<eT> x_cpu(x);

  arma::Mat<eT> m1_ref_cpu = arma::median(x_cpu, 0);
  arma::Mat<eT> m2_ref_cpu = arma::median(x_cpu, 1);

  arma::Mat<eT> m1_cpu(m1);
  arma::Mat<eT> m2_cpu(m2);

  REQUIRE( arma::approx_equal( m1_cpu, m1_ref_cpu, "absdiff", 1 ) );
  REQUIRE( arma::approx_equal( m2_cpu, m2_ref_cpu, "absdiff", 1 ) );
  }
