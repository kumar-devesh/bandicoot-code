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

TEMPLATE_TEST_CASE("simple_median_test", "[median]", float, double)
  {
  typedef TestType eT;

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

  Mat<eT> x(500, 700);
  x.randu();

  Row<eT> col_medians = median(x);
  Row<eT> col_medians2 = median(x, 0);
  Col<eT> row_medians = median(x, 1);

  REQUIRE( col_medians.n_elem == 700 );
  REQUIRE( col_medians2.n_elem == 700 );
  REQUIRE( row_medians.n_elem == 500 );

  arma::Mat<eT> x_cpu(x);

  arma::Row<eT> col_medians_ref_cpu = arma::median(x_cpu, 0);
  arma::Col<eT> row_medians_ref_cpu = arma::median(x_cpu, 1);

  arma::Row<eT> col_medians_cpu(col_medians);
  arma::Row<eT> col_medians2_cpu(col_medians2);
  arma::Col<eT> row_medians_cpu(row_medians);

  REQUIRE( arma::approx_equal(col_medians_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(col_medians2_cpu, col_medians_ref_cpu, "reldiff", 1e-5) );
  REQUIRE( arma::approx_equal(row_medians_cpu, row_medians_ref_cpu, "reldiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("simple_subview_median_test", "[median]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(15, 11);
  x.ones();
  for(uword c = 1; c < 11; ++c)
    {
    x.col(c) *= (c + 2);
    }

  Mat<eT> y(20, 25);
  y.zeros();
  y.submat(5, 5, 19, 14) = x;

  Row<eT> col_medians = median(y.submat(5, 5, 19, 14));
  Row<eT> col_medians2 = median(y.submat(5, 5, 19, 14), 0);
  Col<eT> row_medians = median(y.submat(5, 5, 19, 14), 1);

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



TEST_CASE("empty_median_test", "[median]")
  {
  mat x;
  rowvec m1 = median(x);
  rowvec m2 = median(x, 0);
  vec m3 = median(x, 1);

  REQUIRE( m1.n_elem == 0 );
  REQUIRE( m2.n_elem == 0 );
  REQUIRE( m3.n_elem == 0 );
  }



TEMPLATE_TEST_CASE("simple_median_vec_test", "[median]", float, double)
  {
  typedef TestType eT;

  Col<eT> x(1000);
  x.ones();

  REQUIRE( eT(median(x)) == Approx(eT(1)) );
  }



TEMPLATE_TEST_CASE("random_median_vec_test", "[median]", float, double)
  {
  typedef TestType eT;

  Col<eT> x(100000);
  x.randu();

  arma::Col<eT> x_cpu(x);

  const eT median_val = median(x);
  const eT cpu_median_val = arma::median(x_cpu);

  REQUIRE( median_val == Approx(cpu_median_val) );
  }



TEST_CASE("empty_median_vec_test", "[median]")
  {
  vec x;
  const double median_val = median(x);

  REQUIRE( median_val == 0.0 );
  }



TEST_CASE("median_op_test", "[median]")
  {
  mat x(50, 50);
  x.randu();

  rowvec m1 = median(2 * x + 3);
  mat y = 2 * x + 3;
  rowvec m2 = median(y);

  REQUIRE( m1.n_elem == m2.n_elem );
  for (size_t i = 0; i < m1.n_elem; ++i)
    {
    REQUIRE( double(m1[i]) == Approx(double(m2[i])) );
    }
  }
