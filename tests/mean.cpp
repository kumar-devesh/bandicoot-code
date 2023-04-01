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

TEMPLATE_TEST_CASE("simple_mean_test", "[mean]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x(15, 10);
  x.ones();
  for (uword c = 1; c < 10; ++c)
    {
    x.col(c) *= (c + 1);
    }

  Row<eT> col_means = mean(x);
  Row<eT> col_means2 = mean(x, 0);
  Col<eT> row_means = mean(x, 1);

  REQUIRE( col_means.n_elem == 10 );
  REQUIRE( col_means2.n_elem == 10 );
  REQUIRE( row_means.n_elem == 15 );

  for (uword i = 0; i < 10; ++i)
    {
    REQUIRE( eT(col_means[i]) == Approx(eT(i + 1)) );
    REQUIRE( eT(col_means2[i]) == Approx(eT(i + 1)) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    // Values are between 1 and 11; the mean is 5.5.
    REQUIRE( eT(row_means[i]) == Approx(eT(5.5)) );
    }
  }



TEMPLATE_TEST_CASE("random_mean_test", "[mean]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(500, 700);
  x.randu();

  Row<eT> col_means = mean(x);
  Row<eT> col_means2 = mean(x, 0);
  Col<eT> row_means = mean(x, 1);

  REQUIRE( col_means.n_elem == 700 );
  REQUIRE( col_means2.n_elem == 700 );
  REQUIRE( row_means.n_elem == 500 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_means_ref_cpu = arma::mean(x_cpu, 0);
  arma::Col<eT> row_means_ref_cpu = arma::mean(x_cpu, 1);

  arma::Row<eT> col_means_cpu(col_means);
  arma::Row<eT> col_means2_cpu(col_means2);
  arma::Col<eT> row_means_cpu(row_means);

  REQUIRE( arma::approx_equal(col_means_cpu, col_means_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_means2_cpu, col_means_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_means_cpu, row_means_ref_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("random_mean_randi_test", "[mean]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(500, 700, distr_param(0, 50));

  Row<eT> col_means = mean(x);
  Row<eT> col_means2 = mean(x, 0);
  Col<eT> row_means = mean(x, 1);

  REQUIRE( col_means.n_elem == 700 );
  REQUIRE( col_means2.n_elem == 700 );
  REQUIRE( row_means.n_elem == 500 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_means_ref_cpu = arma::mean(x_cpu, 0);
  arma::Col<eT> row_means_ref_cpu = arma::mean(x_cpu, 1);

  arma::Row<eT> col_means_cpu(col_means);
  arma::Row<eT> col_means2_cpu(col_means2);
  arma::Col<eT> row_means_cpu(row_means);

  REQUIRE( arma::approx_equal(col_means_cpu, col_means_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_means2_cpu, col_means_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_means_cpu, row_means_ref_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("simple_subview_mean_test", "[mean]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x(15, 10);
  x.ones();
  for(uword c = 1; c < 10; ++c)
    {
    x.col(c) *= (c + 1);
    }

  Mat<eT> y(20, 25);
  y.zeros();
  y.submat(5, 5, 19, 14) = x;

  Row<eT> col_means = mean(y.submat(5, 5, 19, 14));
  Row<eT> col_means2 = mean(y.submat(5, 5, 19, 14), 0);
  Col<eT> row_means = mean(y.submat(5, 5, 19, 14), 1);

  REQUIRE( col_means.n_elem == 10 );
  REQUIRE( col_means2.n_elem == 10 );
  REQUIRE( row_means.n_elem == 15 );

  for (uword i = 0; i < 10; ++i)
    {
    REQUIRE( eT(col_means[i]) == Approx(eT(i + 1)) );
    REQUIRE( eT(col_means2[i]) == Approx(eT(i + 1)) );
    }

  for (uword i = 0; i < 15; ++i)
    {
    // Values are between 1 and 11; the mean is 5.5.
    REQUIRE( eT(row_means[i]) == Approx(eT(5.5)) );
    }
  }



TEMPLATE_TEST_CASE("random_subview_mean_test", "[mean]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(500, 700);
  x.randu();

  Row<eT> col_means = mean(x.submat(10, 10, 490, 690));
  Row<eT> col_means2 = mean(x.submat(10, 10, 490, 690), 0);
  Col<eT> row_means = mean(x.submat(10, 10, 490, 690), 1);

  REQUIRE( col_means.n_elem == 681 );
  REQUIRE( col_means2.n_elem == 681 );
  REQUIRE( row_means.n_elem == 481 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_means_ref_cpu = arma::mean(x_cpu.submat(10, 10, 490, 690), 0);
  arma::Col<eT> row_means_ref_cpu = arma::mean(x_cpu.submat(10, 10, 490, 690), 1);

  arma::Row<eT> col_means_cpu(col_means);
  arma::Row<eT> col_means2_cpu(col_means2);
  arma::Col<eT> row_means_cpu(row_means);

  REQUIRE( arma::approx_equal(col_means_cpu, col_means_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_means2_cpu, col_means_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_means_cpu, row_means_ref_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("random_subview_mean_randi_test", "[mean]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(500, 700, distr_param(0, 10));

  Row<eT> col_means = mean(x.submat(10, 10, 490, 690));
  Row<eT> col_means2 = mean(x.submat(10, 10, 490, 690), 0);
  Col<eT> row_means = mean(x.submat(10, 10, 490, 690), 1);

  REQUIRE( col_means.n_elem == 681 );
  REQUIRE( col_means2.n_elem == 681 );
  REQUIRE( row_means.n_elem == 481 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_means_ref_cpu = arma::mean(x_cpu.submat(10, 10, 490, 690), 0);
  arma::Col<eT> row_means_ref_cpu = arma::mean(x_cpu.submat(10, 10, 490, 690), 1);

  arma::Row<eT> col_means_cpu(col_means);
  arma::Row<eT> col_means2_cpu(col_means2);
  arma::Col<eT> row_means_cpu(row_means);

  REQUIRE( arma::approx_equal(col_means_cpu, col_means_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_means2_cpu, col_means_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_means_cpu, row_means_ref_cpu, "reldiff", 1e-5) );
  }



TEST_CASE("empty_mean_test", "[mean]")
  {
  mat x;
  rowvec m1 = mean(x);
  rowvec m2 = mean(x, 0);
  vec m3 = mean(x, 1);

  REQUIRE( m1.n_elem == 0 );
  REQUIRE( m2.n_elem == 0 );
  REQUIRE( m3.n_elem == 0 );
  }



TEMPLATE_TEST_CASE("simple_mean_vec_test", "[mean]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x(1000);
  x.ones();

  REQUIRE( eT(mean(x)) == Approx(eT(1)) );
  }



TEMPLATE_TEST_CASE("random_mean_vec_test", "[mean]", float, double)
  {
  typedef TestType eT;

  Col<eT> x(100000);
  x.randu();

  arma::Col<eT> x_cpu(x);

  const eT mean_val = mean(x);
  const eT cpu_mean_val = arma::mean(x_cpu);

  REQUIRE( mean_val == Approx(cpu_mean_val) );
  }



TEMPLATE_TEST_CASE("random_mean_vec_randi_test", "[mean]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x = randi<Col<eT>>(100000, distr_param(0, 100));

  arma::Col<eT> x_cpu(x);

  const eT mean_val = mean(x);
  const eT cpu_mean_val = arma::mean(x_cpu);

  REQUIRE( mean_val == Approx(cpu_mean_val) );
  }



TEST_CASE("empty_mean_vec_test", "[mean]")
  {
  vec x;
  const double mean_val = mean(x);

  REQUIRE( mean_val == 0.0 );
  }



TEST_CASE("mean_op_test", "[mean]")
  {
  mat x(50, 50);
  x.randu();

  rowvec m1 = mean(2 * x + 3);
  mat y = 2 * x + 3;
  rowvec m2 = mean(y);

  REQUIRE( m1.n_elem == m2.n_elem );
  for (size_t i = 0; i < m1.n_elem; ++i)
    {
    REQUIRE( double(m1[i]) == Approx(double(m2[i])) );
    }
  }
