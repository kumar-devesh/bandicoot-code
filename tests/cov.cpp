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

template<typename eT>
struct tolerance { };

template<>
struct tolerance<float>
  {
  // Note that sometimes GEMM accuracy for floats is low, which leads to a higher tolerance for the tests.
  constexpr static float value = 1e-3;
  };

template<>
struct tolerance<double>
  {
  constexpr static double value = 1e-7;
  };

// TODO: when gemm support is available for integral types, those can be added to this test suite.

// The covariance of an empty matrix should be empty.
TEST_CASE("empty_cov", "[cov]")
  {
  fmat x;
  fmat y = cov(x);

  REQUIRE( y.n_elem == 0 );

  fmat z = cov(x, y);

  REQUIRE( z.n_elem == 0 );

  y = cov(x, 1);

  REQUIRE( y.n_elem == 0 );

  z = cov(x, y, 1);

  REQUIRE( z.n_elem == 0 );
  }



// The covariance of a 1x1 matrix should be [[1]].
TEST_CASE("single_elem_cov", "[cov]")
  {
  fmat x(1, 1);
  x(0, 0) = 5.0;

  fmat y = cov(x);

  REQUIRE( y.n_elem == 1 );
  REQUIRE( float(y[0]) == Approx(0.0).margin(1e-10) );

  fmat z = cov(x, y);

  REQUIRE( z.n_elem == 1 );
  REQUIRE( float(z[0]) == Approx(0.0).margin(1e-10) );

  y = cov(x, 1);

  REQUIRE( y.n_elem == 1 );
  REQUIRE( float(y[0]) == Approx(0.0).margin(1e-10) );

  z = cov(x, y, 1);

  REQUIRE( z.n_elem == 1 );
  REQUIRE( float(z[0]) == Approx(0.0).margin(1e-10) );
  }



// Test a random 2x2 matrix and compare with Armadillo.
TEMPLATE_TEST_CASE("random_basic_cov", "[cov]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(2, 2, distr_param(1, 10));

  Mat<eT> y1 = cov(x);
  Mat<eT> y2 = cov(x, 1);

  Mat<eT> z1 = cov(x, y1);
  Mat<eT> z2 = cov(x, y2, 1);

  arma::Mat<eT> x_ref_cpu(x);

  arma::Mat<eT> y1_ref_cpu = arma::cov(x_ref_cpu);
  arma::Mat<eT> y2_ref_cpu = arma::cov(x_ref_cpu, 1);

  arma::Mat<eT> z1_ref_cpu = arma::cov(x_ref_cpu, y1_ref_cpu);
  arma::Mat<eT> z2_ref_cpu = arma::cov(x_ref_cpu, y2_ref_cpu, 1);

  REQUIRE( y1.n_rows == y1_ref_cpu.n_rows );
  REQUIRE( y1.n_cols == y1_ref_cpu.n_cols );
  REQUIRE( y2.n_rows == y2_ref_cpu.n_rows );
  REQUIRE( y2.n_cols == y2_ref_cpu.n_cols );
  REQUIRE( z1.n_rows == z1_ref_cpu.n_rows );
  REQUIRE( z1.n_cols == z1_ref_cpu.n_cols );
  REQUIRE( z2.n_rows == z2_ref_cpu.n_rows );
  REQUIRE( z2.n_cols == z2_ref_cpu.n_cols );

  arma::Mat<eT> y1_cpu(y1);
  arma::Mat<eT> y2_cpu(y2);
  arma::Mat<eT> z1_cpu(z1);
  arma::Mat<eT> z2_cpu(z2);

  REQUIRE( arma::approx_equal( y1_cpu, y1_ref_cpu, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( y2_cpu, y2_ref_cpu, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( z1_cpu, z1_ref_cpu, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( z2_cpu, z2_ref_cpu, "reldiff", tolerance<eT>::value ) );
  }



// Test cov() on vectors and compare with Armadillo.
TEMPLATE_TEST_CASE("random_vec_cov", "[cov]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> c = randi<Col<eT>>(100, distr_param(1, 10));
  Row<eT> r = randi<Row<eT>>(100, distr_param(1, 10));

  Mat<eT> cov_c = cov(c);
  Mat<eT> cov_r = cov(r);

  arma::Col<eT> c_cpu(c);
  arma::Row<eT> r_cpu(r);

  arma::Mat<eT> cov_c_ref_cpu = arma::cov(c_cpu);
  arma::Mat<eT> cov_r_ref_cpu = arma::cov(r_cpu);

  REQUIRE( cov_c.n_rows == cov_c_ref_cpu.n_rows );
  REQUIRE( cov_c.n_cols == cov_c_ref_cpu.n_cols );
  REQUIRE( cov_r.n_rows == cov_r_ref_cpu.n_rows );
  REQUIRE( cov_r.n_cols == cov_r_ref_cpu.n_cols );

  arma::Mat<eT> cov_c_cpu(cov_c);
  arma::Mat<eT> cov_r_cpu(cov_r);

  REQUIRE( arma::approx_equal( cov_c_cpu, cov_c_ref_cpu, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( cov_r_cpu, cov_r_ref_cpu, "reldiff", tolerance<eT>::value ) );
  }



// Test cov() on subvectors and compare with Armadillo.
TEMPLATE_TEST_CASE("random_subvec_cov", "[cov]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> c = randi<Col<eT>>(200, distr_param(1, 10));
  Row<eT> r = randi<Row<eT>>(200, distr_param(1, 10));

  Mat<eT> cov_c = cov(c.subvec(10, 109));
  Mat<eT> cov_r = cov(r.subvec(10, 109));

  Col<eT> c_sub = c.subvec(10, 109);
  arma::Col<eT> c_cpu(c_sub);
  Row<eT> r_sub = r.subvec(10, 109);
  arma::Row<eT> r_cpu(r_sub);

  arma::Mat<eT> cov_c_ref_cpu = arma::cov(c_cpu);
  arma::Mat<eT> cov_r_ref_cpu = arma::cov(r_cpu);

  REQUIRE( cov_c.n_rows == cov_c_ref_cpu.n_rows );
  REQUIRE( cov_c.n_cols == cov_c_ref_cpu.n_cols );
  REQUIRE( cov_r.n_rows == cov_r_ref_cpu.n_rows );
  REQUIRE( cov_r.n_cols == cov_r_ref_cpu.n_cols );

  arma::Mat<eT> cov_c_cpu(cov_c);
  arma::Mat<eT> cov_r_cpu(cov_r);

  REQUIRE( arma::approx_equal( cov_c_cpu, cov_c_ref_cpu, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( cov_r_cpu, cov_r_ref_cpu, "reldiff", tolerance<eT>::value ) );
  }



// Same as the test above, but with negative floating-point values.
TEMPLATE_TEST_CASE("random_neg_vec_cov", "[cov]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> c = 10 * randu<Col<eT>>(100) - 5;
  Row<eT> r = 10 * randu<Row<eT>>(100) - 5;

  Mat<eT> cov_c = cov(c);
  Mat<eT> cov_r = cov(r);

  arma::Col<eT> c_cpu(c);
  arma::Row<eT> r_cpu(r);

  arma::Mat<eT> cov_c_ref_cpu = arma::cov(c_cpu);
  arma::Mat<eT> cov_r_ref_cpu = arma::cov(r_cpu);

  REQUIRE( cov_c.n_rows == cov_c_ref_cpu.n_rows );
  REQUIRE( cov_c.n_cols == cov_c_ref_cpu.n_cols );
  REQUIRE( cov_r.n_rows == cov_r_ref_cpu.n_rows );
  REQUIRE( cov_r.n_cols == cov_r_ref_cpu.n_cols );

  arma::Mat<eT> cov_c_cpu(cov_c);
  arma::Mat<eT> cov_r_cpu(cov_r);

  REQUIRE( arma::approx_equal( cov_c_cpu, cov_c_ref_cpu, "reldiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( cov_r_cpu, cov_r_ref_cpu, "reldiff", tolerance<eT>::value ) );
  }



// Test cov() on random matrices of different sizes up to 1k x 1k (compare with Armadillo)
TEMPLATE_TEST_CASE("random_mat_cov_size_sweep", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (size_t i = 0; i < 2; ++i)
    {
    const size_t dim = std::pow(2.0, (double) i);

    Mat<eT> x = randi<Mat<eT>>(dim, dim, distr_param(1, 10));

    arma::Mat<eT> x_cpu(x);

    Mat<eT> y1 = cov(x);
    Mat<eT> y2 = cov(x, 1);

    arma::Mat<eT> y1_ref_cpu = arma::cov(x_cpu);
    arma::Mat<eT> y2_ref_cpu = arma::cov(x_cpu, 1);

    REQUIRE( y1.n_rows == y1_ref_cpu.n_rows );
    REQUIRE( y1.n_cols == y1_ref_cpu.n_cols );
    REQUIRE( y2.n_rows == y2_ref_cpu.n_rows );
    REQUIRE( y2.n_cols == y2_ref_cpu.n_cols );

    arma::Mat<eT> y1_cpu(y1);
    arma::Mat<eT> y2_cpu(y2);

    REQUIRE( arma::approx_equal( y1_cpu, y1_ref_cpu, "reldiff", tolerance<eT>::value ) );
    REQUIRE( arma::approx_equal( y2_cpu, y2_ref_cpu, "reldiff", tolerance<eT>::value ) );
    }
  }



// Same as the test above, but with negative floating-point values.
TEMPLATE_TEST_CASE("randu_neg_mat_cov_size_sweep", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (size_t i = 3; i < 10; ++i)
    {
    const size_t dim = std::pow(2.0, (double) i);

    Mat<eT> x = 10 * randu<Mat<eT>>(dim, dim) - 5;

    arma::Mat<eT> x_cpu(x);

    Mat<eT> y1 = cov(x);
    Mat<eT> y2 = cov(x, 1);

    arma::Mat<eT> y1_ref_cpu = arma::cov(x_cpu);
    arma::Mat<eT> y2_ref_cpu = arma::cov(x_cpu, 1);

    REQUIRE( y1.n_rows == y1_ref_cpu.n_rows );
    REQUIRE( y1.n_cols == y1_ref_cpu.n_cols );
    REQUIRE( y2.n_rows == y2_ref_cpu.n_rows );
    REQUIRE( y2.n_cols == y2_ref_cpu.n_cols );

    arma::Mat<eT> y1_cpu(y1);
    arma::Mat<eT> y2_cpu(y2);

    REQUIRE( arma::approx_equal( y1_cpu, y1_ref_cpu, "absdiff", tolerance<eT>::value ) );
    REQUIRE( arma::approx_equal( y2_cpu, y2_ref_cpu, "absdiff", tolerance<eT>::value ) );
    }
  }



// Take covariance of vectors x vectors.
TEMPLATE_TEST_CASE("vector_x_vector_cov", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x1 = randi<Col<eT>>(1000, distr_param(1, 100));
  Col<eT> x2 = randi<Col<eT>>(1000, distr_param(1, 100));
  Row<eT> x3 = randi<Row<eT>>(1000, distr_param(1, 100));
  Row<eT> x4 = randi<Row<eT>>(1000, distr_param(1, 100));

  Mat<eT> c1 = cov(x1, x2);
  Mat<eT> c2 = cov(x1, x3);
  Mat<eT> c3 = cov(x3, x1);
  Mat<eT> c4 = cov(x3, x4);
  Mat<eT> c5 = cov(x1, x2, 1);
  Mat<eT> c6 = cov(x1, x3, 1);
  Mat<eT> c7 = cov(x3, x1, 1);
  Mat<eT> c8 = cov(x3, x4, 1);

  // Now compute the same with Armadillo.
  arma::Col<eT> x1_cpu(x1);
  arma::Col<eT> x2_cpu(x2);
  arma::Row<eT> x3_cpu(x3);
  arma::Row<eT> x4_cpu(x4);

  arma::Mat<eT> c1_ref_cpu = arma::cov(x1_cpu, x2_cpu);
  arma::Mat<eT> c2_ref_cpu = arma::cov(x1_cpu, x3_cpu);
  arma::Mat<eT> c3_ref_cpu = arma::cov(x3_cpu, x1_cpu);
  arma::Mat<eT> c4_ref_cpu = arma::cov(x3_cpu, x4_cpu);
  arma::Mat<eT> c5_ref_cpu = arma::cov(x1_cpu, x2_cpu, 1);
  arma::Mat<eT> c6_ref_cpu = arma::cov(x1_cpu, x3_cpu, 1);
  arma::Mat<eT> c7_ref_cpu = arma::cov(x3_cpu, x1_cpu, 1);
  arma::Mat<eT> c8_ref_cpu = arma::cov(x3_cpu, x4_cpu, 1);

  REQUIRE( c1.n_rows == c1_ref_cpu.n_rows );
  REQUIRE( c1.n_cols == c1_ref_cpu.n_cols );
  REQUIRE( c2.n_rows == c2_ref_cpu.n_rows );
  REQUIRE( c2.n_cols == c2_ref_cpu.n_cols );
  REQUIRE( c3.n_rows == c3_ref_cpu.n_rows );
  REQUIRE( c3.n_cols == c3_ref_cpu.n_cols );
  REQUIRE( c4.n_rows == c4_ref_cpu.n_rows );
  REQUIRE( c4.n_cols == c4_ref_cpu.n_cols );
  REQUIRE( c5.n_rows == c5_ref_cpu.n_rows );
  REQUIRE( c5.n_cols == c5_ref_cpu.n_cols );
  REQUIRE( c6.n_rows == c6_ref_cpu.n_rows );
  REQUIRE( c6.n_cols == c6_ref_cpu.n_cols );
  REQUIRE( c7.n_rows == c7_ref_cpu.n_rows );
  REQUIRE( c7.n_cols == c7_ref_cpu.n_cols );
  REQUIRE( c8.n_rows == c8_ref_cpu.n_rows );
  REQUIRE( c8.n_cols == c8_ref_cpu.n_cols );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c4_cpu(c4);
  arma::Mat<eT> c5_cpu(c5);
  arma::Mat<eT> c6_cpu(c6);
  arma::Mat<eT> c7_cpu(c7);
  arma::Mat<eT> c8_cpu(c8);

  REQUIRE( arma::approx_equal( c1_cpu, c1_ref_cpu, "absdiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( c2_cpu, c2_ref_cpu, "absdiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( c3_cpu, c3_ref_cpu, "absdiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( c4_cpu, c4_ref_cpu, "absdiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( c5_cpu, c5_ref_cpu, "absdiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( c6_cpu, c6_ref_cpu, "absdiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( c7_cpu, c7_ref_cpu, "absdiff", tolerance<eT>::value ) );
  REQUIRE( arma::approx_equal( c8_cpu, c8_ref_cpu, "absdiff", tolerance<eT>::value ) );
  }



// Compute covariances of vectors x matrices and vice versa; compare with Armadillo.
TEMPLATE_TEST_CASE("vec_x_matrix_cov", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x1 = randi<Col<eT>>(200, distr_param(1, 10));
  Row<eT> x2 = randi<Row<eT>>(200, distr_param(1, 10));
  Mat<eT> x3 = randi<Mat<eT>>(200, 200, distr_param(1, 10));

  Mat<eT> c1 = cov(x1, x3);
  Mat<eT> c2 = cov(x2, x3);
  Mat<eT> c3 = cov(x3, x1);
  Mat<eT> c4 = cov(x3, x2);
  Mat<eT> c5 = cov(x1, x3, 1);
  Mat<eT> c6 = cov(x2, x3, 1);
  Mat<eT> c7 = cov(x3, x1, 1);
  Mat<eT> c8 = cov(x3, x2, 1);

  // Now do the same computation with Armadillo.
  arma::Col<eT> x1_cpu(x1);
  arma::Row<eT> x2_cpu(x2);
  arma::Mat<eT> x3_cpu(x3);

  arma::Mat<eT> c1_ref_cpu = arma::cov(x1_cpu, x3_cpu);
  arma::Mat<eT> c2_ref_cpu = arma::cov(x2_cpu, x3_cpu);
  arma::Mat<eT> c3_ref_cpu = arma::cov(x3_cpu, x1_cpu);
  arma::Mat<eT> c4_ref_cpu = arma::cov(x3_cpu, x2_cpu);
  arma::Mat<eT> c5_ref_cpu = arma::cov(x1_cpu, x3_cpu, 1);
  arma::Mat<eT> c6_ref_cpu = arma::cov(x2_cpu, x3_cpu, 1);
  arma::Mat<eT> c7_ref_cpu = arma::cov(x3_cpu, x1_cpu, 1);
  arma::Mat<eT> c8_ref_cpu = arma::cov(x3_cpu, x2_cpu, 1);

  REQUIRE( c1.n_rows == c1_ref_cpu.n_rows );
  REQUIRE( c1.n_cols == c1_ref_cpu.n_cols );
  REQUIRE( c2.n_rows == c2_ref_cpu.n_rows );
  REQUIRE( c2.n_cols == c2_ref_cpu.n_cols );
  REQUIRE( c3.n_rows == c3_ref_cpu.n_rows );
  REQUIRE( c3.n_cols == c3_ref_cpu.n_cols );
  REQUIRE( c4.n_rows == c4_ref_cpu.n_rows );
  REQUIRE( c4.n_cols == c4_ref_cpu.n_cols );
  REQUIRE( c5.n_rows == c5_ref_cpu.n_rows );
  REQUIRE( c5.n_cols == c5_ref_cpu.n_cols );
  REQUIRE( c6.n_rows == c6_ref_cpu.n_rows );
  REQUIRE( c6.n_cols == c6_ref_cpu.n_cols );
  REQUIRE( c7.n_rows == c7_ref_cpu.n_rows );
  REQUIRE( c7.n_cols == c7_ref_cpu.n_cols );
  REQUIRE( c8.n_rows == c8_ref_cpu.n_rows );
  REQUIRE( c8.n_cols == c8_ref_cpu.n_cols );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c4_cpu(c4);
  arma::Mat<eT> c5_cpu(c5);
  arma::Mat<eT> c6_cpu(c6);
  arma::Mat<eT> c7_cpu(c7);
  arma::Mat<eT> c8_cpu(c8);

  REQUIRE( arma::norm( c1_cpu - c1_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c2_cpu - c2_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c3_cpu - c3_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c4_cpu - c4_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c5_cpu - c5_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c6_cpu - c6_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c7_cpu - c7_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c8_cpu - c8_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  }



// Compute covariances of subvectors x submatrices and vice versa; compare with Armadillo.
TEMPLATE_TEST_CASE("subvec_x_submatrix_cov", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x1 = randi<Col<eT>>(200, distr_param(1, 10));
  Row<eT> x2 = randi<Row<eT>>(200, distr_param(1, 10));
  Mat<eT> x3 = randi<Mat<eT>>(200, 200, distr_param(1, 10));

  Mat<eT> c1 = cov(x1.subvec(10, 109), x3.submat(10, 10, 109, 109));
  Mat<eT> c2 = cov(x2.subvec(10, 109), x3.submat(10, 10, 109, 109));
  Mat<eT> c3 = cov(x3.submat(10, 10, 109, 109), x1.subvec(10, 109));
  Mat<eT> c4 = cov(x3.submat(10, 10, 109, 109), x2.subvec(10, 109));
  Mat<eT> c5 = cov(x1.subvec(10, 109), x3.submat(10, 10, 109, 109), 1);
  Mat<eT> c6 = cov(x2.subvec(10, 109), x3.submat(10, 10, 109, 109), 1);
  Mat<eT> c7 = cov(x3.submat(10, 10, 109, 109), x1.subvec(10, 109), 1);
  Mat<eT> c8 = cov(x3.submat(10, 10, 109, 109), x2.subvec(10, 109), 1);

  // Now do the same computation with Armadillo.
  Col<eT> x1_sub = x1.subvec(10, 109);
  arma::Col<eT> x1_cpu(x1_sub);
  Row<eT> x2_sub = x2.subvec(10, 109);
  arma::Row<eT> x2_cpu(x2_sub);
  Mat<eT> x3_sub = x3.submat(10, 10, 109, 109);
  arma::Mat<eT> x3_cpu(x3_sub);

  arma::Mat<eT> c1_ref_cpu = arma::cov(x1_cpu, x3_cpu);
  arma::Mat<eT> c2_ref_cpu = arma::cov(x2_cpu, x3_cpu);
  arma::Mat<eT> c3_ref_cpu = arma::cov(x3_cpu, x1_cpu);
  arma::Mat<eT> c4_ref_cpu = arma::cov(x3_cpu, x2_cpu);
  arma::Mat<eT> c5_ref_cpu = arma::cov(x1_cpu, x3_cpu, 1);
  arma::Mat<eT> c6_ref_cpu = arma::cov(x2_cpu, x3_cpu, 1);
  arma::Mat<eT> c7_ref_cpu = arma::cov(x3_cpu, x1_cpu, 1);
  arma::Mat<eT> c8_ref_cpu = arma::cov(x3_cpu, x2_cpu, 1);

  REQUIRE( c1.n_rows == c1_ref_cpu.n_rows );
  REQUIRE( c1.n_cols == c1_ref_cpu.n_cols );
  REQUIRE( c2.n_rows == c2_ref_cpu.n_rows );
  REQUIRE( c2.n_cols == c2_ref_cpu.n_cols );
  REQUIRE( c3.n_rows == c3_ref_cpu.n_rows );
  REQUIRE( c3.n_cols == c3_ref_cpu.n_cols );
  REQUIRE( c4.n_rows == c4_ref_cpu.n_rows );
  REQUIRE( c4.n_cols == c4_ref_cpu.n_cols );
  REQUIRE( c5.n_rows == c5_ref_cpu.n_rows );
  REQUIRE( c5.n_cols == c5_ref_cpu.n_cols );
  REQUIRE( c6.n_rows == c6_ref_cpu.n_rows );
  REQUIRE( c6.n_cols == c6_ref_cpu.n_cols );
  REQUIRE( c7.n_rows == c7_ref_cpu.n_rows );
  REQUIRE( c7.n_cols == c7_ref_cpu.n_cols );
  REQUIRE( c8.n_rows == c8_ref_cpu.n_rows );
  REQUIRE( c8.n_cols == c8_ref_cpu.n_cols );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c4_cpu(c4);
  arma::Mat<eT> c5_cpu(c5);
  arma::Mat<eT> c6_cpu(c6);
  arma::Mat<eT> c7_cpu(c7);
  arma::Mat<eT> c8_cpu(c8);

  REQUIRE( arma::norm( c1_cpu - c1_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c2_cpu - c2_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c3_cpu - c3_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c4_cpu - c4_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c5_cpu - c5_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c6_cpu - c6_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c7_cpu - c7_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  REQUIRE( arma::norm( c8_cpu - c8_ref_cpu, 2 ) <= (eT) tolerance<eT>::value );
  }



// Compute covariance between matrices of different sizes, up to 1k x 1k.
TEMPLATE_TEST_CASE("random_mat_x_mat_cov_size_sweep", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (size_t i = 3; i < 10; ++i)
    {
    const size_t dim = std::pow(2.0, (double) i);

    Mat<eT> x1 = randi<Mat<eT>>(dim, dim, distr_param(1, 10));
    Mat<eT> x2 = randi<Mat<eT>>(dim, dim, distr_param(1, 10));

    arma::Mat<eT> x1_cpu(x1);
    arma::Mat<eT> x2_cpu(x2);

    Mat<eT> y1 = cov(x1, x2);
    Mat<eT> y2 = cov(x1, x2, 1);

    arma::Mat<eT> y1_ref_cpu = arma::cov(x1_cpu, x2_cpu);
    arma::Mat<eT> y2_ref_cpu = arma::cov(x1_cpu, x2_cpu, 1);

    REQUIRE( y1.n_rows == y1_ref_cpu.n_rows );
    REQUIRE( y1.n_cols == y1_ref_cpu.n_cols );
    REQUIRE( y2.n_rows == y2_ref_cpu.n_rows );
    REQUIRE( y2.n_cols == y2_ref_cpu.n_cols );

    arma::Mat<eT> y1_cpu(y1);
    arma::Mat<eT> y2_cpu(y2);

    REQUIRE( arma::approx_equal( y1_cpu, y1_ref_cpu, "absdiff", tolerance<eT>::value ) );
    REQUIRE( arma::approx_equal( y2_cpu, y2_ref_cpu, "absdiff", tolerance<eT>::value ) );
    }
  }



// Same as the test above, but with floating point types and negative elements.
TEMPLATE_TEST_CASE("random_neg_mat_x_mat_cov_size_sweep", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (size_t i = 3; i < 10; ++i)
    {
    const size_t dim = std::pow(2.0, (double) i);

    Mat<eT> x1 = 10 * randu<Mat<eT>>(dim, dim) - 5;
    Mat<eT> x2 = 10 * randu<Mat<eT>>(dim, dim) - 5;

    arma::Mat<eT> x1_cpu(x1);
    arma::Mat<eT> x2_cpu(x2);

    Mat<eT> y1 = cov(x1, x2);
    Mat<eT> y2 = cov(x1, x2, 1);

    arma::Mat<eT> y1_ref_cpu = arma::cov(x1_cpu, x2_cpu);
    arma::Mat<eT> y2_ref_cpu = arma::cov(x1_cpu, x2_cpu, 1);

    REQUIRE( y1.n_rows == y1_ref_cpu.n_rows );
    REQUIRE( y1.n_cols == y1_ref_cpu.n_cols );
    REQUIRE( y2.n_rows == y2_ref_cpu.n_rows );
    REQUIRE( y2.n_cols == y2_ref_cpu.n_cols );

    arma::Mat<eT> y1_cpu(y1);
    arma::Mat<eT> y2_cpu(y2);

    REQUIRE( arma::approx_equal( y1_cpu, y1_ref_cpu, "absdiff", tolerance<eT>::value ) );
    REQUIRE( arma::approx_equal( y2_cpu, y2_ref_cpu, "absdiff", tolerance<eT>::value ) );
    }
  }



// Same as the test above, but with floating point types and negative elements, and non-square matrices.
TEMPLATE_TEST_CASE("random_neg_non_square_mat_x_mat_cov_size_sweep", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (size_t i = 3; i < 10; ++i)
    {
    const size_t dim1 = (size_t) std::max((int) 3, (int) std::pow(2.0, (double) i - 1) - 7);
    const size_t dim2 = std::pow(2.0, (double) i) + 3;

    Mat<eT> x1 = 10 * randu<Mat<eT>>(dim1, dim2) - 5;
    Mat<eT> x2 = 10 * randu<Mat<eT>>(dim1, dim2) - 5;

    arma::Mat<eT> x1_cpu(x1);
    arma::Mat<eT> x2_cpu(x2);

    Mat<eT> y1 = cov(x1, x2);
    Mat<eT> y2 = cov(x1, x2, 1);

    arma::Mat<eT> y1_ref_cpu = arma::cov(x1_cpu, x2_cpu);
    arma::Mat<eT> y2_ref_cpu = arma::cov(x1_cpu, x2_cpu, 1);

    REQUIRE( y1.n_rows == y1_ref_cpu.n_rows );
    REQUIRE( y1.n_cols == y1_ref_cpu.n_cols );
    REQUIRE( y2.n_rows == y2_ref_cpu.n_rows );
    REQUIRE( y2.n_cols == y2_ref_cpu.n_cols );

    arma::Mat<eT> y1_cpu(y1);
    arma::Mat<eT> y2_cpu(y2);

    REQUIRE( arma::approx_equal( y1_cpu, y1_ref_cpu, "absdiff", tolerance<eT>::value ) );
    REQUIRE( arma::approx_equal( y2_cpu, y2_ref_cpu, "absdiff", tolerance<eT>::value ) );
    }
  }



// Compute the covariance of an expression (just a sanity check).
TEMPLATE_TEST_CASE("expr_cov", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(100, 100, distr_param(0, 100));
  Col<eT> d = randi<Col<eT>>(100, distr_param(10, 20));

  Mat<eT> expr_result = (2 * x + 3) * diagmat(d).t();

  Mat<eT> c1 = cov((2 * x + 3) * diagmat(d).t());
  Mat<eT> c2 = cov(expr_result);

  REQUIRE( c1.n_rows == c2.n_rows );
  REQUIRE( c1.n_cols == c2.n_cols );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);

  REQUIRE( arma::approx_equal( c1_cpu, c2_cpu, "reldiff", tolerance<eT>::value ) );
  }



// Compute the covariance of two expressions (just a sanity check).
TEMPLATE_TEST_CASE("expr_x_expr_cov", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(100, 100, distr_param(0, 100));
  Col<eT> d = randi<Col<eT>>(100, distr_param(10, 20));

  Mat<eT> expr_result1 = (2 * x + 3) * diagmat(d).t();
  Mat<eT> expr_result2 = (4 * x.t() - 6);

  Mat<eT> c1 = cov((2 * x + 3) * diagmat(d).t(), (4 * x.t() - 6));
  Mat<eT> c2 = cov(expr_result1, expr_result2);

  REQUIRE( c1.n_rows == c2.n_rows );
  REQUIRE( c1.n_cols == c2.n_cols );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);

  REQUIRE( arma::approx_equal( c1_cpu, c2_cpu, "reldiff", tolerance<eT>::value ) );
  }



// Covariance inside expression.
TEMPLATE_TEST_CASE("cov_in_expr", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(250, 500, distr_param(1, 10));
  Col<eT> c = randi<Col<eT>>(500, distr_param(10, 20));

  Mat<eT> y = 2 * cov(x, 1) + 3 * diagmat(c);

  Mat<eT> cov_x = cov(x, 1);
  Mat<eT> y_ref = 2 * cov_x + 3 * diagmat(c);

  REQUIRE( y.n_rows == y_ref.n_rows );
  REQUIRE( y.n_cols == y_ref.n_cols );

  arma::Mat<eT> y_cpu(y);
  arma::Mat<eT> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", tolerance<eT>::value ) );
  }



// Test chained covariance (why not?).
TEMPLATE_TEST_CASE("chained_cov", "[cov]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x1 = randi<Mat<eT>>(173, 174, distr_param(50, 60));
  Mat<eT> x2 = randi<Mat<eT>>(173, 174, distr_param(50, 60));

  Mat<eT> y = cov(cov(x1, x2), 1);

  Mat<eT> c1 = cov(x1, x2);
  Mat<eT> y_ref = cov(c1, 1);

  REQUIRE( y.n_rows == y_ref.n_rows );
  REQUIRE( y.n_cols == y_ref.n_cols );

  arma::Mat<eT> y_cpu(y);
  arma::Mat<eT> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", tolerance<eT>::value ) );
  }



// Test single-argument covariance with a conv_to; we have a special overload for this case.
TEMPLATE_TEST_CASE
  (
  "conv_to_cov",
  "[cov]",
  (std::pair<double, float>), (std::pair<float, double>)
  )
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(512, 162, distr_param(1, 10));

  // First check if we apply cov() before the conversion.
  Mat<eT2> postconv_y = conv_to<Mat<eT2>>::from(cov(x));

  Mat<eT1> cov_x = cov(x);
  Mat<eT2> postconv_y_ref = conv_to<Mat<eT2>>::from(cov_x);

  REQUIRE( postconv_y.n_rows == postconv_y_ref.n_rows );
  REQUIRE( postconv_y.n_cols == postconv_y_ref.n_cols );

  arma::Mat<eT2> postconv_y_cpu(postconv_y);
  arma::Mat<eT2> postconv_y_ref_cpu(postconv_y_ref);

  REQUIRE( arma::approx_equal( postconv_y_cpu, postconv_y_ref_cpu, "reldiff", 1e-5 ) );

  // Now try again when we apply cov() after the conversion (this is where we have a special overload).
  Mat<eT2> preconv_y  = cov(conv_to<Mat<eT2>>::from(x));

  Mat<eT2> x_conv = conv_to<Mat<eT2>>::from(x);
  Mat<eT2> preconv_y_ref = cov(x_conv);

  REQUIRE( preconv_y.n_rows == preconv_y_ref.n_rows );
  REQUIRE( preconv_y.n_cols == preconv_y_ref.n_cols );

  arma::Mat<eT2> preconv_y_cpu(preconv_y);
  arma::Mat<eT2> preconv_y_ref_cpu(preconv_y_ref);

  REQUIRE( arma::approx_equal( preconv_y_cpu, preconv_y_ref_cpu, "reldiff", tolerance<eT2>::value ) );
  }
