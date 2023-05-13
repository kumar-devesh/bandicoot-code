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

template<typename eT>
struct tolerance { };

template<>
struct tolerance<float>
  {
  constexpr static float value = 1e-3;
  };

template<>
struct tolerance<double>
  {
  constexpr static double value = 1e-7;
  };

// simple normalisation

TEMPLATE_TEST_CASE("simple_normalise_vec", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> c(10);
  for (uword i = 0; i < 10; ++i)
    c[i] = eT(i);

  Col<eT> out_1 = normalise(c, 1);
  Col<eT> out_2 = normalise(c, 2);
  Col<eT> out_3 = normalise(c, 3);

  REQUIRE( out_1.n_elem == 10 );
  REQUIRE( out_2.n_elem == 10 );
  REQUIRE( out_3.n_elem == 10 );

  for (uword i = 0; i < 10; ++i)
    {
    REQUIRE( eT(out_1[i]) == Approx( eT(i) / eT(45) ) );
    REQUIRE( eT(out_2[i]) == Approx( eT(i) / eT(16.88194301613) ) );
    REQUIRE( eT(out_3[i]) == Approx( eT(i) / eT(12.65148997952) ) );
    }
  }



TEMPLATE_TEST_CASE("simple_normalise_colwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = eT(r + c);
      }
    }

  Mat<eT> out_1a = normalise(x, 1);
  Mat<eT> out_1b = normalise(x, 1, 0);
  Mat<eT> out_2a = normalise(x, 2);
  Mat<eT> out_2b = normalise(x, 2, 0);
  Mat<eT> out_3a = normalise(x, 3);
  Mat<eT> out_3b = normalise(x, 3, 0);

  REQUIRE( out_1a.n_rows == 10 );
  REQUIRE( out_1a.n_cols == 10 );
  REQUIRE( out_1b.n_rows == 10 );
  REQUIRE( out_1b.n_cols == 10 );
  REQUIRE( out_2a.n_rows == 10 );
  REQUIRE( out_2a.n_cols == 10 );
  REQUIRE( out_2b.n_rows == 10 );
  REQUIRE( out_2b.n_cols == 10 );
  REQUIRE( out_3a.n_rows == 10 );
  REQUIRE( out_3a.n_cols == 10 );
  REQUIRE( out_3b.n_rows == 10 );
  REQUIRE( out_3b.n_cols == 10 );

  for (uword c = 0; c < 10; ++c)
    {
    const eT norm_1 = 10 * c + eT(45);
    const eT norm_2 = std::sqrt((10 * c * c) + (2 * c * eT(45)) + eT(285));
    const eT norm_3 = std::pow((10 * c * c * c) + (3 * c * c * eT(45)) + (3 * c * eT(285)) + eT(2025), eT(1.0) / eT(3.0));

    for (uword r = 0; r < 10; ++r)
      {
      REQUIRE( eT(out_1a(r, c)) == Approx( eT(r + c) / norm_1 ) );
      REQUIRE( eT(out_1b(r, c)) == Approx( eT(r + c) / norm_1 ) );
      REQUIRE( eT(out_2a(r, c)) == Approx( eT(r + c) / norm_2 ) );
      REQUIRE( eT(out_2b(r, c)) == Approx( eT(r + c) / norm_2 ) );
      REQUIRE( eT(out_3a(r, c)) == Approx( eT(r + c) / norm_3 ) );
      REQUIRE( eT(out_3b(r, c)) == Approx( eT(r + c) / norm_3 ) );
      }
    }
  }



TEMPLATE_TEST_CASE("simple_normalise_rowwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = eT(r + c);
      }
    }
  x = x.t();

  Mat<eT> out_1 = normalise(x, 1, 1);
  Mat<eT> out_2 = normalise(x, 2, 1);
  Mat<eT> out_3 = normalise(x, 3, 1);

  REQUIRE( out_1.n_rows == 10 );
  REQUIRE( out_1.n_cols == 10 );
  REQUIRE( out_2.n_rows == 10 );
  REQUIRE( out_2.n_cols == 10 );
  REQUIRE( out_3.n_rows == 10 );
  REQUIRE( out_3.n_cols == 10 );

  for (uword r = 0; r < 10; ++r)
    {
    const eT norm_1 = 10 * r + eT(45);
    const eT norm_2 = std::sqrt((10 * r * r) + (2 * r * eT(45)) + eT(285));
    const eT norm_3 = std::pow((10 * r * r * r) + (3 * r * r * eT(45)) + (3 * r * eT(285)) + eT(2025), eT(1.0) / eT(3.0));

    for (uword c = 0; c < 10; ++c)
      {
      REQUIRE( eT(out_1(r, c)) == Approx( eT(r + c) / norm_1 ) );
      REQUIRE( eT(out_2(r, c)) == Approx( eT(r + c) / norm_2 ) );
      REQUIRE( eT(out_3(r, c)) == Approx( eT(r + c) / norm_3 ) );
      }
    }
  }



// large

TEMPLATE_TEST_CASE("large_normalise_vec", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(100000);
  arma::Col<eT> x_cpu(x);

  Col<eT> y1 = normalise(x, 1);
  Col<eT> y2 = normalise(x, 2);
  Col<eT> y3 = normalise(x, 3);

  REQUIRE( y1.n_elem == 100000 );
  REQUIRE( y2.n_elem == 100000 );
  REQUIRE( y3.n_elem == 100000 );

  arma::Col<eT> y1_cpu_ref = arma::normalise(x_cpu, 1);
  arma::Col<eT> y2_cpu_ref = arma::normalise(x_cpu, 2);
  arma::Col<eT> y3_cpu_ref = arma::normalise(x_cpu, 3);

  arma::Col<eT> y1_cpu(y1);
  arma::Col<eT> y2_cpu(y2);
  arma::Col<eT> y3_cpu(y3);

  REQUIRE( arma::approx_equal( y1_cpu, y1_cpu_ref, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y2_cpu, y2_cpu_ref, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y3_cpu, y3_cpu_ref, "reldiff", tolerance<eT>::value ) );
  }



TEMPLATE_TEST_CASE("large_normalise_colwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(10000, 500);
  arma::Mat<eT> x_cpu(x);

  Mat<eT> y1a = normalise(x, 1);
  Mat<eT> y1b = normalise(x, 1, 0);
  Mat<eT> y2a = normalise(x, 2);
  Mat<eT> y2b = normalise(x, 2, 0);
  Mat<eT> y3a = normalise(x, 3);
  Mat<eT> y3b = normalise(x, 3, 0);

  REQUIRE( y1a.n_rows == 10000 );
  REQUIRE( y1a.n_cols == 500 );
  REQUIRE( y1b.n_rows == 10000 );
  REQUIRE( y1b.n_cols == 500 );
  REQUIRE( y2a.n_rows == 10000 );
  REQUIRE( y2a.n_cols == 500 );
  REQUIRE( y2b.n_rows == 10000 );
  REQUIRE( y2b.n_cols == 500 );
  REQUIRE( y3a.n_rows == 10000 );
  REQUIRE( y3a.n_cols == 500 );
  REQUIRE( y3b.n_rows == 10000 );
  REQUIRE( y3b.n_cols == 500 );

  arma::Mat<eT> y1_cpu_ref = arma::normalise(x_cpu, 1);
  arma::Mat<eT> y2_cpu_ref = arma::normalise(x_cpu, 2);
  arma::Mat<eT> y3_cpu_ref = arma::normalise(x_cpu, 3);

  arma::Mat<eT> y1a_cpu(y1a);
  arma::Mat<eT> y1b_cpu(y1b);
  arma::Mat<eT> y2a_cpu(y2a);
  arma::Mat<eT> y2b_cpu(y2b);
  arma::Mat<eT> y3a_cpu(y3a);
  arma::Mat<eT> y3b_cpu(y3b);

  REQUIRE( arma::approx_equal( y1a_cpu, y1_cpu_ref, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y1b_cpu, y1_cpu_ref, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y2a_cpu, y2_cpu_ref, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y2b_cpu, y2_cpu_ref, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y3a_cpu, y3_cpu_ref, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y3b_cpu, y3_cpu_ref, "reldiff", tolerance<eT>::value ) );
  }



TEMPLATE_TEST_CASE("large_normalise_rowwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(1250, 1500);
  arma::Mat<eT> x_cpu(x);

  Mat<eT> y1 = normalise(x, 1, 1);
  Mat<eT> y2 = normalise(x, 2, 1);
  Mat<eT> y3 = normalise(x, 3, 1);

  REQUIRE( y1.n_rows == 1250 );
  REQUIRE( y1.n_cols == 1500 );
  REQUIRE( y2.n_rows == 1250 );
  REQUIRE( y2.n_cols == 1500 );
  REQUIRE( y3.n_rows == 1250 );
  REQUIRE( y3.n_cols == 1500 );

  arma::Mat<eT> y1_cpu_ref = arma::normalise(x_cpu, 1, 1);
  arma::Mat<eT> y2_cpu_ref = arma::normalise(x_cpu, 2, 1);
  arma::Mat<eT> y3_cpu_ref = arma::normalise(x_cpu, 3, 1);

  arma::Mat<eT> y1_cpu(y1);
  arma::Mat<eT> y2_cpu(y2);
  arma::Mat<eT> y3_cpu(y3);

  REQUIRE( arma::approx_equal( y1_cpu, y1_cpu_ref, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y2_cpu, y2_cpu_ref, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y3_cpu, y3_cpu_ref, "reldiff", tolerance<eT>::value ) );
  }



// norm already 1

TEMPLATE_TEST_CASE("already_normalised_vec", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(10000, distr_param(1, 10));

  Col<eT> y = normalise(x);
  Col<eT> z = normalise(y);

  REQUIRE( y.n_elem == z.n_elem );

  arma::Col<eT> y_cpu(y);
  arma::Col<eT> z_cpu(z);

  for (uword i = 0; i < 10000; ++i)
    {
    REQUIRE( eT(y_cpu[i]) == Approx( eT(z_cpu[i]) ) );
    }
  }



TEMPLATE_TEST_CASE("already_normalised_colwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(500, 1000, distr_param(1, 10));

  Mat<eT> y = normalise(x);
  Mat<eT> z1 = normalise(y, 2);
  Mat<eT> z2 = normalise(y, 2, 0);

  REQUIRE( z1.n_rows == y.n_rows );
  REQUIRE( z1.n_cols == y.n_cols );
  REQUIRE( z2.n_rows == y.n_rows );
  REQUIRE( z2.n_cols == y.n_cols );

  arma::Mat<eT> y_cpu(y);
  arma::Mat<eT> z1_cpu(z1);
  arma::Mat<eT> z2_cpu(z2);

  REQUIRE( arma::approx_equal( z1_cpu, y_cpu, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( z2_cpu, y_cpu, "reldiff", tolerance<eT>::value ) );
  }



TEMPLATE_TEST_CASE("already_normalised_rowwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(500, 1000, distr_param(1, 10));

  Mat<eT> y = normalise(x, 2, 1);
  Mat<eT> z = normalise(y, 2, 1);

  REQUIRE( z.n_rows == y.n_rows );
  REQUIRE( z.n_cols == y.n_cols );

  arma::Mat<eT> y_cpu(y);
  arma::Mat<eT> z_cpu(z);

  REQUIRE( arma::approx_equal( z_cpu, y_cpu, "reldiff", tolerance<eT>::value ) );
  }



// zeros

TEMPLATE_TEST_CASE("zeros_normalise_vec", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(100);
  x.zeros();

  Col<eT> y1 = normalise(x, 1);
  Col<eT> y2 = normalise(x, 2);
  Col<eT> y3 = normalise(x, 3);

  REQUIRE( y1.n_elem == 100 );
  REQUIRE( y2.n_elem == 100 );
  REQUIRE( y3.n_elem == 100 );

  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( eT(y1[i]) == eT(0) );
    REQUIRE( eT(y2[i]) == eT(0) );
    REQUIRE( eT(y3[i]) == eT(0) );
    }
  }



TEMPLATE_TEST_CASE("zeros_normalise_colwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 25);
  x.zeros();
  x.col(3) = randu<Col<eT>>(10);
  x.submat(0, 3, 4, 3) += eT(2);
  x.submat(5, 3, 9, 3) -= eT(2);

  Mat<eT> y1a = normalise(x, 1);
  Mat<eT> y1b = normalise(x, 1, 0);
  Mat<eT> y2a = normalise(x, 2);
  Mat<eT> y2b = normalise(x, 2, 0);
  Mat<eT> y3a = normalise(x, 3);
  Mat<eT> y3b = normalise(x, 3, 0);

  REQUIRE( y1a.n_rows == 10 );
  REQUIRE( y1a.n_cols == 25 );
  REQUIRE( y1b.n_rows == 10 );
  REQUIRE( y1b.n_cols == 25 );
  REQUIRE( y2a.n_rows == 10 );
  REQUIRE( y2a.n_cols == 25 );
  REQUIRE( y2b.n_rows == 10 );
  REQUIRE( y2b.n_cols == 25 );
  REQUIRE( y3a.n_rows == 10 );
  REQUIRE( y3a.n_cols == 25 );
  REQUIRE( y3b.n_rows == 10 );
  REQUIRE( y3b.n_cols == 25 );

  for (uword c = 0; c < 25; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      if (c == 3)
        {
        REQUIRE( eT(y1a(r, c)) != eT(0) );
        REQUIRE( eT(y1b(r, c)) != eT(0) );
        REQUIRE( eT(y2a(r, c)) != eT(0) );
        REQUIRE( eT(y2b(r, c)) != eT(0) );
        REQUIRE( eT(y3a(r, c)) != eT(0) );
        REQUIRE( eT(y3b(r, c)) != eT(0) );
        }
      else
        {
        REQUIRE( eT(y1a(r, c)) == eT(0) );
        REQUIRE( eT(y1b(r, c)) == eT(0) );
        REQUIRE( eT(y2a(r, c)) == eT(0) );
        REQUIRE( eT(y2b(r, c)) == eT(0) );
        REQUIRE( eT(y3a(r, c)) == eT(0) );
        REQUIRE( eT(y3b(r, c)) == eT(0) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("zeros_normalise_rowwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(25, 10);
  x.zeros();
  x.row(3) = randu<Row<eT>>(10);
  x.submat(3, 0, 3, 4) += eT(2);
  x.submat(3, 5, 3, 9) -= eT(2);

  Mat<eT> y1 = normalise(x, 1, 1);
  Mat<eT> y2 = normalise(x, 2, 1);
  Mat<eT> y3 = normalise(x, 3, 1);

  REQUIRE( y1.n_rows == 25 );
  REQUIRE( y1.n_cols == 10 );
  REQUIRE( y2.n_rows == 25 );
  REQUIRE( y2.n_cols == 10 );
  REQUIRE( y3.n_rows == 25 );
  REQUIRE( y3.n_cols == 10 );

  for (uword r = 0; r < 25; ++r)
    {
    for (uword c = 0; c < 10; ++c)
      {
      if (r == 3)
        {
        REQUIRE( eT(y1(r, c)) != eT(0) );
        REQUIRE( eT(y2(r, c)) != eT(0) );
        REQUIRE( eT(y3(r, c)) != eT(0) );
        }
      else
        {
        REQUIRE( eT(y1(r, c)) == eT(0) );
        REQUIRE( eT(y2(r, c)) == eT(0) );
        REQUIRE( eT(y3(r, c)) == eT(0) );
        }
      }
    }
  }


// empty

TEST_CASE("empty_normalise_vec", "[normalise]")
  {
  fvec x;
  fvec y = normalise(x);
  REQUIRE( y.n_elem == 0 );
  }



TEST_CASE("empty_normalise_vec_mat", "[normalise]")
  {
  fmat x;
  fmat y1 = normalise(x, 2);
  fmat y2 = normalise(x, 2, 0);
  fmat y3 = normalise(x, 2, 1);

  REQUIRE( y1.n_rows == 0 );
  REQUIRE( y1.n_cols == 0 );
  REQUIRE( y2.n_rows == 0 );
  REQUIRE( y2.n_cols == 0 );
  REQUIRE( y3.n_rows == 0 );
  REQUIRE( y3.n_cols == 0 );
  }


// alias

TEMPLATE_TEST_CASE("alias_normalise_vec", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(1000);
  x = normalise(x);

  REQUIRE( x.n_elem == 1000 );
  const eT norm_x = norm(x);
  REQUIRE( norm_x == Approx( eT(1) ) );
  }



TEMPLATE_TEST_CASE("alias_normalise_colwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(150, 10);
  x = normalise(x, 2, 0);

  REQUIRE( x.n_rows == 150 );
  REQUIRE( x.n_cols == 10 );

  for (uword c = 0; c < 10; ++c)
    {
    const eT norm_col = norm(x.col(c), 2);
    REQUIRE( norm_col == Approx( eT(1) ) );
    }
  }



TEMPLATE_TEST_CASE("alias_normalise_rowwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(10, 150);
  x = normalise(x, 2, 1);

  REQUIRE( x.n_rows == 10 );
  REQUIRE( x.n_cols == 150 );

  for (uword r = 0; r < 10; ++r)
    {
    const eT norm_col = norm(x.row(r), 2);
    REQUIRE( norm_col == Approx( eT(1) ) );
    }

  }


// subview

TEMPLATE_TEST_CASE("subview_normalise_vec", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(10);
  for (uword i = 0; i < 10; ++i)
    {
    x[i] = eT(i);
    }

  Col<eT> y1 = normalise(x.subvec(0, 5), 1);
  Col<eT> y2 = normalise(x.subvec(0, 5), 2);
  Col<eT> y3 = normalise(x.subvec(0, 5), 3);

  REQUIRE( y1.n_elem == 6 );
  REQUIRE( y2.n_elem == 6 );
  REQUIRE( y3.n_elem == 6 );

  for (uword i = 0; i < 6; ++i)
    {
    REQUIRE( eT(y1[i]) == Approx( eT(i) / eT(15) ) );
    REQUIRE( eT(y2[i]) == Approx( eT(i) / eT(7.41619848709) ) );
    REQUIRE( eT(y3[i]) == Approx( eT(i) / eT(6.08220199557) ) );
    }
  }



TEMPLATE_TEST_CASE("subview_normalise_colwise", "[normalise}", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = eT(r + c);
      }
    }

  Mat<eT> out_1a = normalise(x.submat(0, 0, 5, 5), 1);
  Mat<eT> out_1b = normalise(x.submat(0, 0, 5, 5), 1, 0);
  Mat<eT> out_2a = normalise(x.submat(0, 0, 5, 5), 2);
  Mat<eT> out_2b = normalise(x.submat(0, 0, 5, 5), 2, 0);
  Mat<eT> out_3a = normalise(x.submat(0, 0, 5, 5), 3);
  Mat<eT> out_3b = normalise(x.submat(0, 0, 5, 5), 3, 0);

  REQUIRE( out_1a.n_rows == 6 );
  REQUIRE( out_1a.n_cols == 6 );
  REQUIRE( out_1b.n_rows == 6 );
  REQUIRE( out_1b.n_cols == 6 );
  REQUIRE( out_2a.n_rows == 6 );
  REQUIRE( out_2a.n_cols == 6 );
  REQUIRE( out_2b.n_rows == 6 );
  REQUIRE( out_2b.n_cols == 6 );
  REQUIRE( out_3a.n_rows == 6 );
  REQUIRE( out_3a.n_cols == 6 );
  REQUIRE( out_3b.n_rows == 6 );
  REQUIRE( out_3b.n_cols == 6 );

  for (uword c = 0; c < 6; ++c)
    {
    const eT norm_1 = 6 * c + eT(15);
    const eT norm_2 = std::sqrt((6 * c * c) + (2 * c * eT(15)) + eT(55));
    const eT norm_3 = std::pow((6 * c * c * c) + (3 * c * c * eT(15)) + (3 * c * eT(55)) + eT(225), eT(1.0) / eT(3.0));

    for (uword r = 0; r < 6; ++r)
      {
      REQUIRE( eT(out_1a(r, c)) == Approx( eT(r + c) / norm_1 ) );
      REQUIRE( eT(out_1b(r, c)) == Approx( eT(r + c) / norm_1 ) );
      REQUIRE( eT(out_2a(r, c)) == Approx( eT(r + c) / norm_2 ) );
      REQUIRE( eT(out_2b(r, c)) == Approx( eT(r + c) / norm_2 ) );
      REQUIRE( eT(out_3a(r, c)) == Approx( eT(r + c) / norm_3 ) );
      REQUIRE( eT(out_3b(r, c)) == Approx( eT(r + c) / norm_3 ) );
      }
    }
  }



TEMPLATE_TEST_CASE("subview_normalise_rowwise", "[normalise]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = eT(r + c);
      }
    }
  x = x.t();

  Mat<eT> out_1 = normalise(x.submat(0, 0, 5, 5), 1, 1);
  Mat<eT> out_2 = normalise(x.submat(0, 0, 5, 5), 2, 1);
  Mat<eT> out_3 = normalise(x.submat(0, 0, 5, 5), 3, 1);

  REQUIRE( out_1.n_rows == 6 );
  REQUIRE( out_1.n_cols == 6 );
  REQUIRE( out_2.n_rows == 6 );
  REQUIRE( out_2.n_cols == 6 );
  REQUIRE( out_3.n_rows == 6 );
  REQUIRE( out_3.n_cols == 6 );

  for (uword r = 0; r < 6; ++r)
    {
    const eT norm_1 = 6 * r + eT(15);
    const eT norm_2 = std::sqrt((6 * r * r) + (2 * r * eT(15)) + eT(55));
    const eT norm_3 = std::pow((6 * r * r * r) + (3 * r * r * eT(15)) + (3 * r * eT(55)) + eT(225), eT(1.0) / eT(3.0));

    for (uword c = 0; c < 6; ++c)
      {
      REQUIRE( eT(out_1(r, c)) == Approx( eT(r + c) / norm_1 ) );
      REQUIRE( eT(out_2(r, c)) == Approx( eT(r + c) / norm_2 ) );
      REQUIRE( eT(out_3(r, c)) == Approx( eT(r + c) / norm_3 ) );
      }
    }
  }



// conv_to

TEMPLATE_TEST_CASE
  (
  "conv_to_normalise_vec",
  "[normalise]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>)
  )
  {
  typedef typename TestType::first_type eT2;
  typedef typename TestType::second_type eT1;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Col<eT1> x = randi<Col<eT1>>(100, distr_param(10, 15));

  Col<eT2> x_conv = conv_to<Col<eT2>>::from(x);

  Col<eT2> y1 = normalise(conv_to<Col<eT2>>::from(x), 1);
  Col<eT2> y2 = normalise(conv_to<Col<eT2>>::from(x), 2);
  Col<eT2> y3 = normalise(conv_to<Col<eT2>>::from(x), 3);

  Col<eT2> y1_ref = normalise(x_conv, 1);
  Col<eT2> y2_ref = normalise(x_conv, 2);
  Col<eT2> y3_ref = normalise(x_conv, 3);

  REQUIRE( y1.n_elem == 100 );
  REQUIRE( y2.n_elem == 100 );
  REQUIRE( y3.n_elem == 100 );

  arma::Col<eT2> y1_cpu(y1);
  arma::Col<eT2> y2_cpu(y2);
  arma::Col<eT2> y3_cpu(y3);
  arma::Col<eT2> y1_ref_cpu(y1_ref);
  arma::Col<eT2> y2_ref_cpu(y2_ref);
  arma::Col<eT2> y3_ref_cpu(y3_ref);

  REQUIRE( arma::approx_equal( y1_cpu, y1_ref_cpu, "reldiff", tolerance<eT2>::value ) );
  REQUIRE( arma::approx_equal( y2_cpu, y2_ref_cpu, "reldiff", tolerance<eT2>::value ) );
  REQUIRE( arma::approx_equal( y3_cpu, y3_ref_cpu, "reldiff", tolerance<eT2>::value ) );
  }
