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

TEMPLATE_TEST_CASE("any_vec_simple", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(50);
  x.zeros();

  REQUIRE( any(x) == false );

  for (uword i = 0; i < 50; ++i)
    {
    Col<eT> y(x);
    y[i] = eT(1);

    REQUIRE( any(y) == true );
    }
  }



TEMPLATE_TEST_CASE("any_vec_large", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(100000);
  x.zeros();

  REQUIRE( any(x) == false );

  for (uword i = 0; i < 100; ++i)
    {
    Col<eT> y(x);
    Col<eT> rand_index = randi<Col<eT>>(1, distr_param(0, 99999));

    y[rand_index[0]] = (eT) 1;

    REQUIRE( any(y) == true );
    }
  }



TEMPLATE_TEST_CASE("any_vec_mat", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(50, 50);
  x.zeros();

  REQUIRE( any(any(x)) == false );

  for (uword i = 0; i < 50; ++i)
    {
    Mat<eT> y(x);
    Col<eT> rand_index = randi<Col<eT>>(1, distr_param(0, 2499));
    y[rand_index[0]] = (eT) 1;

    REQUIRE( any(any(y)) == true );
    }
  }



TEMPLATE_TEST_CASE("any_mat", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10000, 500);
  x.zeros();
  x(3, 10) = eT(1);
  x(44, 50) = eT(1);
  x(1419, 477) = eT(1);

  urowvec y = any(x);
  urowvec y2 = any(x, 0);

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
      REQUIRE( y_cpu[i] == 1 );
      REQUIRE( y2_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 0 );
      REQUIRE( y2_cpu[i] == 0 );
      }
    }
  }



TEMPLATE_TEST_CASE("any_mat_large", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 50000);
  x.zeros();
  x(11, 10) = eT(1);
  x(44, 50) = eT(1);
  x(3, 477) = eT(1);
  x(0, 1132) = eT(1);
  x(99, 49999) = eT(1);

  urowvec y = any(x);
  urowvec y2 = any(x, 0);

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
      REQUIRE( y_cpu[i] == 1 );
      REQUIRE( y2_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 0 );
      REQUIRE( y2_cpu[i] == 0 );
      }
    }
  }



TEST_CASE("any_empty", "[any]")
  {
  fmat x;

  urowvec y = any(x);
  urowvec y2 = any(x, 0);

  REQUIRE( y.n_rows == 1 );
  REQUIRE( y.n_cols == 0 );
  REQUIRE( y.n_elem == 0 );

  REQUIRE( y2.n_rows == 1 );
  REQUIRE( y2.n_cols == 0 );
  REQUIRE( y2.n_elem == 0 );
  }



TEST_CASE("any_alias", "[any]")
  {
  umat x(50, 50);
  x.zeros();
  x.col(0).ones();
  x.col(11).ones();
  x.col(42).ones();
  umat x2 = x;

  x = any(x);
  x2 = any(x2, 0);

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
      REQUIRE( x_cpu[i] == 1 );
      REQUIRE( x2_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( x_cpu[i] == 0 );
      REQUIRE( x2_cpu[i] == 0 );
      }
    }
  }



TEMPLATE_TEST_CASE("any_mat_rowwise", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(500, 10000);
  x.zeros();
  x.row(10).ones();
  x.row(50).ones();
  x.row(477).ones();

  uvec y = any(x, 1);

  REQUIRE( y.n_rows == 500 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 500 );

  arma::Col<uword> y_cpu(y);

  for (size_t i = 0; i < 500; ++i)
    {
    if (i == 10 || i == 50 || i == 477)
      {
      REQUIRE( y_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 0 );
      }
    }
  }



TEMPLATE_TEST_CASE("any_mat_rowwise_large", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(50000, 100);
  x.zeros();
  x(10, 3) = eT(1);
  x(50, 1) = eT(1);
  x(477, 56) = eT(1);
  x(1132, 99) = eT(1);
  x(49999, 0) = eT(1);

  uvec y = any(x, 1);

  REQUIRE( y.n_rows == 50000 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 50000 );

  arma::Col<uword> y_cpu(y);

  for (size_t i = 0; i < 50000; ++i)
    {
    if (i == 10 || i == 50 || i == 477 || i == 1132 || i == 49999)
      {
      REQUIRE( y_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 0 );
      }
    }
  }



TEST_CASE("any_rowwise_empty", "[any]")
  {
  fmat x;
  uvec y = any(x, 1);

  REQUIRE( y.n_rows == 0 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 0 );
  }



TEST_CASE("any_rowwise_alias", "[any]")
  {
  umat x(50, 50);
  x.zeros();
  x.row(0).ones();
  x.row(11).ones();
  x.row(42).ones();

  x = any(x, 1);

  REQUIRE( x.n_rows == 50 );
  REQUIRE( x.n_cols == 1 );
  REQUIRE( x.n_elem == 50 );

  arma::Mat<uword> x_cpu(x);

  for (size_t i = 0; i < 50; ++i)
    {
    if (i == 0 || i == 11 || i == 42)
      {
      REQUIRE( x_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( x_cpu[i] == 0 );
      }
    }

  }



TEMPLATE_TEST_CASE("any_mat_subview", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 100);
  x.zeros();
  x(0, 0) = eT(1);
  x(99, 99) = eT(1);
  x.col(5).ones();
  x.col(15).ones();

  urowvec y = any(x.submat(4, 4, 20, 20));
  urowvec y2 = any(x.submat(4, 4, 20, 20), 0);

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
      REQUIRE( y_cpu[i] == 1 );
      REQUIRE( y2_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 0 );
      REQUIRE( y2_cpu[i] == 0 );
      }
    }
  }



TEMPLATE_TEST_CASE("any_mat_rowwise_subview", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 100);
  x.zeros();
  x(0, 0) = eT(1);
  x(99, 99) = eT(1);
  x.row(5).ones();
  x.row(15).ones();

  uvec y = any(x.submat(4, 4, 20, 20), 1);

  REQUIRE( y.n_rows == 17 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 17 );

  arma::Col<uword> y_cpu(y);

  for (size_t i = 0; i < 17; ++i)
    {
    if (i == 1 || i == 11)
      {
      REQUIRE( y_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 0 );
      }
    }
  }



TEMPLATE_TEST_CASE("any_mat_expr", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 100);
  x.zeros();
  Mat<eT> y = x;
  y.col(1).ones();
  y.col(10).ones();
  y.col(50).ones();

  urowvec z = any(2 * (x - y));
  urowvec z2 = any(2 * (x - y), 0);

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



TEMPLATE_TEST_CASE("any_mat_expr_rowwise", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 100);
  x.zeros();
  Mat<eT> y = x;
  y.row(1).ones();
  y.row(10).ones();
  y.row(50).ones();

  uvec z = any(2 * (x - y), 1);

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



TEMPLATE_TEST_CASE(
  "any_vec_conv_to",
  "[any]",
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

  Col<eT1> x = randi<Col<eT1>>(1000, distr_param(10, 20));

  REQUIRE ( any(conv_to<Col<eT2>>::from(x)) == true );

  x.zeros();

  REQUIRE ( any(conv_to<Col<eT2>>::from(x)) == false );
  }



TEMPLATE_TEST_CASE(
  "any_vec_conv_to_matrix",
  "[any]",
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

  Mat<eT1> x = randi<Mat<eT1>>(500, 500, distr_param(10, 20));

  REQUIRE( any(any(conv_to<Mat<eT2>>::from(x))) == true );

  x.zeros();

  REQUIRE( any(any(conv_to<Mat<eT2>>::from(x))) == false );
  }



TEMPLATE_TEST_CASE(
  "any_conv_to",
  "[any]",
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

  Mat<eT1> x(500, 500);
  x.zeros();
  x.col(10).ones();
  x.col(50).ones();
  x.col(477).ones();

  Row<uword> y = any(conv_to<Mat<eT2>>::from(x));
  Row<uword> y2 = any(conv_to<Mat<eT2>>::from(x));

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
      REQUIRE( y_cpu[i] == 1 );
      REQUIRE( y2_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 0 );
      REQUIRE( y2_cpu[i] == 0 );
      }
    }
  }



TEMPLATE_TEST_CASE(
  "any_conv_to_rowwise",
  "[any]",
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

  Mat<eT1> x(500, 10000);
  x.zeros();
  x.row(10).ones();
  x.row(50).ones();
  x.row(477).ones();

  Col<uword> y = any(conv_to<Mat<eT2>>::from(x), 1);

  REQUIRE( y.n_rows == 500 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 500 );

  arma::Col<uword> y_cpu(y);

  for (size_t i = 0; i < 500; ++i)
    {
    if (i == 10 || i == 50 || i == 477)
      {
      REQUIRE( y_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( y_cpu[i] == 0 );
      }
    }
  }



TEMPLATE_TEST_CASE(
  "any_conv_to_eop",
  "[any]",
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

  Mat<eT1> x(10000, 500);
  x.zeros();
  x.col(10).ones();
  x.col(50).ones();
  x.col(477).ones();

  Row<uword> y = any(conv_to<Mat<eT2>>::from(2 - 2 * x));
  Row<uword> y2 = any(conv_to<Mat<eT2>>::from(2 - 2 * x), 0);

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



// Test special optimizations for some relational expressions.
// (We also test the unoptimized cases just to make sure nothing is wrong.)

TEMPLATE_TEST_CASE("any_relational_expressions", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X = randi<Mat<eT>>(10, 10, distr_param(0, 2));

  umat y11 = any(X < 0);
  umat y12 = any(X < 0, 0);
  umat y13 = any(X < 0, 1);

  umat z = X < 0;
  umat y21 = any(z);
  umat y22 = any(z, 0);
  umat y23 = any(z, 1);

  REQUIRE( any(any(y11 == y21)) );
  REQUIRE( any(any(y12 == y22)) );
  REQUIRE( any(any(y13 == y23)) );

  y11 = any(0 < X);
  y12 = any(0 < X, 0);
  y13 = any(0 < X, 1);

  z = 0 < X;
  y21 = any(z);
  y22 = any(z, 0);
  y23 = any(z, 1);

  REQUIRE( any(any(y11 == y21)) );
  REQUIRE( any(any(y12 == y22)) );
  REQUIRE( any(any(y13 == y23)) );

  y11 = any(X > 0);
  y12 = any(X > 0, 0);
  y13 = any(X > 0, 1);

  z = X > 0;
  y21 = any(z);
  y22 = any(z, 0);
  y23 = any(z, 1);

  REQUIRE( any(any(y11 == y21)) );
  REQUIRE( any(any(y12 == y22)) );
  REQUIRE( any(any(y13 == y23)) );

  y11 = any(0 > X);
  y12 = any(0 > X, 0);
  y13 = any(0 > X, 1);

  z = 0 > X;
  y21 = any(z);
  y22 = any(z, 0);
  y23 = any(z, 1);

  REQUIRE( any(any(y11 == y21)) );
  REQUIRE( any(any(y12 == y22)) );
  REQUIRE( any(any(y13 == y23)) );

  y11 = any(X <= 0);
  y12 = any(X <= 0, 0);
  y13 = any(X <= 0, 1);

  z = X <= 0;
  y21 = any(z);
  y22 = any(z, 0);
  y23 = any(z, 1);

  REQUIRE( any(any(y11 == y21)) );
  REQUIRE( any(any(y12 == y22)) );
  REQUIRE( any(any(y13 == y23)) );

  y11 = any(0 <= X);
  y12 = any(0 <= X, 0);
  y13 = any(0 <= X, 1);

  z = 0 <= X;
  y21 = any(z);
  y22 = any(z, 0);
  y23 = any(z, 1);

  REQUIRE( any(any(y11 == y21)) );
  REQUIRE( any(any(y12 == y22)) );
  REQUIRE( any(any(y13 == y23)) );

  y11 = any(X >= 0);
  y12 = any(X >= 0, 0);
  y13 = any(X >= 0, 1);

  z = X >= 0;
  y21 = any(z);
  y22 = any(z, 0);
  y23 = any(z, 1);

  REQUIRE( any(any(y11 == y21)) );
  REQUIRE( any(any(y12 == y22)) );
  REQUIRE( any(any(y13 == y23)) );

  y11 = any(0 >= X);
  y12 = any(0 >= X, 0);
  y13 = any(0 >= X, 1);

  z = 0 >= X;
  y21 = any(z);
  y22 = any(z, 0);
  y23 = any(z, 1);

  REQUIRE( any(any(y11 == y21)) );
  REQUIRE( any(any(y12 == y22)) );
  REQUIRE( any(any(y13 == y23)) );

  y11 = any(X == 0);
  y12 = any(X == 0, 0);
  y13 = any(X == 0, 1);

  z = X == 0;
  y21 = any(z);
  y22 = any(z, 0);
  y23 = any(z, 1);

  REQUIRE( any(any(y11 == y21)) );
  REQUIRE( any(any(y12 == y22)) );
  REQUIRE( any(any(y13 == y23)) );

  y11 = any(X != 0);
  y12 = any(X != 0, 0);
  y13 = any(X != 0, 1);

  z = X != 0;
  y21 = any(z);
  y22 = any(z, 0);
  y23 = any(z, 1);
  }



// Test special optimizations for some relational expressions.
// (We also test the unoptimized cases just to make sure nothing is wrong.)

TEMPLATE_TEST_CASE("any_vec_relational_op", "[any]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(500, distr_param(0, 2));

  uvec z = (x < 0);
  REQUIRE( any(x < 0) == any(z) );
  z = (0 < x);
  REQUIRE( any(0 < x) == any(z) );
  z = (x > 0);
  REQUIRE( any(x > 0) == any(z) );
  z = (0 > x);
  REQUIRE( any(0 > x) == any(z) );
  z = (x <= 0);
  REQUIRE( any(x <= 0) == any(z) );
  z = (0 <= x);
  REQUIRE( any(0 <= x) == any(z) );
  z = (x >= 0);
  REQUIRE( any(x >= 0) == any(z) );
  z = (0 >= x);
  REQUIRE( any(0 >= x) == any(z) );
  z = (x == 0);
  REQUIRE( any(x == 0) == any(z) );
  z = (x != 0);
  REQUIRE( any(x != 0) == any(z) );
  }
