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

TEMPLATE_TEST_CASE("all_vec_simple", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x = randi<Col<eT>>(50, distr_param(3, 5));

  REQUIRE( all(x) == true );

  for (uword i = 0; i < 50; ++i)
    {
    Col<eT> y(x);
    y[i] = eT(0);

    REQUIRE( all(y) == false );
    }
  }



TEMPLATE_TEST_CASE("all_vec_large", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x = randi<Col<eT>>(100000, distr_param(3, 5));

  REQUIRE( all(x) == true );

  for (uword i = 0; i < 100; ++i)
    {
    Col<eT> y(x);
    Col<eT> rand_index = randi<Col<eT>>(1, distr_param(0, 99999));

    y[rand_index[0]] = (eT) 0;

    REQUIRE( all(y) == false );
    }
  }



TEMPLATE_TEST_CASE("all_vec_mat", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(3, 5));

  REQUIRE( all(all(x)) == true );

  for (uword i = 0; i < 50; ++i)
    {
    Mat<eT> y(x);
    Col<eT> rand_index = randi<Col<eT>>(1, distr_param(0, 2499));
    y[rand_index[0]] = (eT) 0;

    REQUIRE( all(all(y)) == false );
    }
  }



TEMPLATE_TEST_CASE("all_mat", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(10000, 500, distr_param(10, 30));
  x.col(10).zeros();
  x.col(50).zeros();
  x.col(477).zeros();

  urowvec y = all(x);
  urowvec y2 = all(x, 0);

  REQUIRE( y.n_rows == 1 );
  REQUIRE( y.n_cols == 500 );
  REQUIRE( y.n_elem == 500 );

  REQUIRE( y2.n_rows == 1 );
  REQUIRE( y2.n_cols == 500 );
  REQUIRE( y2.n_elem == 500 );

  arma::Row<uword> y_cpu(y);
  arma::Row<uword> y2_cpu(y2);

  for (size_t i = 0; i < 500; ++i)
    {
    if (i == 10 || i == 50 || i == 477)
      {
      REQUIRE( y_cpu[i] == 0 );
      REQUIRE( y2_cpu[i] == 0 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 1 );
      REQUIRE( y2_cpu[i] == 1 );
      }
    }
  }



TEMPLATE_TEST_CASE("all_mat_large", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(100, 50000, distr_param(10, 30));
  x.col(10).zeros();
  x.col(50).zeros();
  x.col(477).zeros();
  x.col(1132).zeros();
  x.col(49999).zeros();

  urowvec y = all(x);
  urowvec y2 = all(x, 0);

  REQUIRE( y.n_rows == 1 );
  REQUIRE( y.n_cols == 50000 );
  REQUIRE( y.n_elem == 50000 );

  REQUIRE( y2.n_rows == 1 );
  REQUIRE( y2.n_cols == 50000 );
  REQUIRE( y2.n_elem == 50000 );

  arma::Row<uword> y_cpu(y);
  arma::Row<uword> y2_cpu(y2);

  for (size_t i = 0; i < 50000; ++i)
    {
    if (i == 10 || i == 50 || i == 477 || i == 1132 || i == 49999)
      {
      REQUIRE( y_cpu[i] == 0 );
      REQUIRE( y2_cpu[i] == 0 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 1 );
      REQUIRE( y2_cpu[i] == 1 );
      }
    }
  }



TEST_CASE("all_empty", "[all]")
  {
  mat x;

  urowvec y = all(x);
  urowvec y2 = all(x, 0);

  REQUIRE( y.n_rows == 1 );
  REQUIRE( y.n_cols == 0 );
  REQUIRE( y.n_elem == 0 );

  REQUIRE( y2.n_rows == 1 );
  REQUIRE( y2.n_cols == 0 );
  REQUIRE( y2.n_elem == 0 );
  }



TEST_CASE("all_alias", "[all]")
  {
  umat x = randi<umat>(50, 50, distr_param(3, 5));
  x.col(0).zeros();
  x.col(11).zeros();
  x.col(42).zeros();
  umat x2 = x;

  x = all(x);
  x2 = all(x2, 0);

  REQUIRE( x.n_rows == 1 );
  REQUIRE( x.n_cols == 50 );
  REQUIRE( x.n_elem == 50 );

  REQUIRE( x2.n_rows == 1 );
  REQUIRE( x2.n_cols == 50 );
  REQUIRE( x2.n_elem == 50 );

  arma::Mat<uword> x_cpu(x);
  arma::Mat<uword> x2_cpu(x2);

  for (size_t i = 0; i < 50; ++i)
    {
    if (i == 0 || i == 11 || i == 42)
      {
      REQUIRE( x_cpu[i] == 0 );
      REQUIRE( x2_cpu[i] == 0 );
      }
    else
      {
      REQUIRE( x_cpu[i] == 1 );
      REQUIRE( x2_cpu[i] == 1 );
      }
    }
  }



TEMPLATE_TEST_CASE("all_mat_rowwise", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(500, 10000, distr_param(10, 30));
  x.row(10).zeros();
  x.row(50).zeros();
  x.row(477).zeros();

  uvec y = all(x, 1);

  REQUIRE( y.n_rows == 500 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 500 );

  arma::Col<uword> y_cpu(y);

  for (size_t i = 0; i < 500; ++i)
    {
    if (i == 10 || i == 50 || i == 477)
      {
      REQUIRE( y_cpu[i] == 0 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 1 );
      }
    }
  }



TEMPLATE_TEST_CASE("all_mat_rowwise_large", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(50000, 100, distr_param(10, 30));
  x.row(10).zeros();
  x.row(50).zeros();
  x.row(477).zeros();
  x.row(1132).zeros();
  x.row(49999).zeros();

  uvec y = all(x, 1);

  REQUIRE( y.n_rows == 50000 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 50000 );

  arma::Col<uword> y_cpu(y);

  for (size_t i = 0; i < 50000; ++i)
    {
    if (i == 10 || i == 50 || i == 477 || i == 1132 || i == 49999)
      {
      REQUIRE( y_cpu[i] == 0 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 1 );
      }
    }
  }



TEST_CASE("all_rowwise_empty", "[all]")
  {
  mat x;
  uvec y = all(x, 1);

  REQUIRE( y.n_rows == 0 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 0 );
  }



TEST_CASE("all_rowwise_alias", "[all]")
  {
  umat x = randi<umat>(50, 50, distr_param(3, 5));
  x.row(0).zeros();
  x.row(11).zeros();
  x.row(42).zeros();

  x = all(x, 1);

  REQUIRE( x.n_rows == 50 );
  REQUIRE( x.n_cols == 1 );
  REQUIRE( x.n_elem == 50 );

  arma::Mat<uword> x_cpu(x);

  for (size_t i = 0; i < 50; ++i)
    {
    if (i == 0 || i == 11 || i == 42)
      {
      REQUIRE( x_cpu[i] == 0 );
      }
    else
      {
      REQUIRE( x_cpu[i] == 1 );
      }
    }

  }



TEMPLATE_TEST_CASE("all_mat_subview", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(100, 100, distr_param(10, 20));
  x.col(5).zeros();
  x.col(15).zeros();

  urowvec y = all(x.submat(4, 4, 20, 20));
  urowvec y2 = all(x.submat(4, 4, 20, 20), 0);

  REQUIRE( y.n_rows == 1 );
  REQUIRE( y.n_cols == 17 );
  REQUIRE( y.n_elem == 17 );

  REQUIRE( y2.n_rows == 1 );
  REQUIRE( y2.n_cols == 17 );
  REQUIRE( y2.n_elem == 17 );

  arma::Row<uword> y_cpu(y);
  arma::Row<uword> y2_cpu(y2);

  for (size_t i = 0; i < 17; ++i)
    {
    if (i == 1 || i == 11)
      {
      REQUIRE( y_cpu[i] == 0 );
      REQUIRE( y2_cpu[i] == 0 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 1 );
      REQUIRE( y2_cpu[i] == 1 );
      }
    }
  }



TEMPLATE_TEST_CASE("all_mat_rowwise_subview", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(100, 100, distr_param(10, 20));
  x.row(5).zeros();
  x.row(15).zeros();

  uvec y = all(x.submat(4, 4, 20, 20), 1);

  REQUIRE( y.n_rows == 17 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 17 );

  arma::Col<uword> y_cpu(y);

  for (size_t i = 0; i < 17; ++i)
    {
    if (i == 1 || i == 11)
      {
      REQUIRE( y_cpu[i] == 0 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 1 );
      }
    }
  }



TEMPLATE_TEST_CASE("all_mat_expr", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(100, 100, distr_param(10, 20));
  Mat<eT> y = x;
  y.col(1).zeros();
  y.col(10).zeros();
  y.col(50).zeros();

  urowvec z = all(2 * (x - y));
  urowvec z2 = all(2 * (x - y), 0);

  REQUIRE( z.n_rows == 1 );
  REQUIRE( z.n_cols == 100 );
  REQUIRE( z.n_elem == 100 );

  REQUIRE( z2.n_rows == 1 );
  REQUIRE( z2.n_cols == 100 );
  REQUIRE( z2.n_elem == 100 );

  arma::Row<uword> z_cpu(z);
  arma::Row<uword> z2_cpu(z2);

  for (size_t i = 0; i < 100; ++i)
    {
    if (i == 1 || i == 10 || i == 50)
      {
      REQUIRE( z_cpu[i] == 1 );
      REQUIRE( z2_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( z_cpu[i] == 0 );
      REQUIRE( z2_cpu[i] == 0 );
      }
    }
  }



TEMPLATE_TEST_CASE("all_mat_expr_rowwise", "[all]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(100, 100, distr_param(10, 20));
  Mat<eT> y = x;
  y.row(1).zeros();
  y.row(10).zeros();
  y.row(50).zeros();

  uvec z = all(2 * (x - y), 1);

  REQUIRE( z.n_rows == 100 );
  REQUIRE( z.n_cols == 1 );
  REQUIRE( z.n_elem == 100 );

  arma::Col<uword> z_cpu(z);

  for (size_t i = 0; i < 100; ++i)
    {
    if (i == 1 || i == 10 || i == 50)
      {
      REQUIRE( z_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( z_cpu[i] == 0 );
      }
    }
  }
