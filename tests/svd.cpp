// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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

TEMPLATE_TEST_CASE("trivial_svd", "[svd]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  // The SVD of the identity matrix should produce singular values that are all 1.
  // (We can't say as much about the singular vectors.)
  Mat<eT> x(50, 50);
  x.zeros();
  for (uword i = 0; i < 50; ++i)
    {
    x(i, i) = eT(1);
    }

  Col<eT> s = svd(x);

  REQUIRE( s.n_elem == 50 );
  for (uword i = 0; i < 50; ++i)
    {
    REQUIRE ( eT(s[i]) == Approx(1.0) );
    }
  }



TEMPLATE_TEST_CASE("empty_svd", "[svd]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x;

  // All three SVD variants should return empty matrices.
  Col<eT> s;
  Mat<eT> u, v;

  bool result = svd(s, x);
  REQUIRE( result == true );
  REQUIRE( s.n_elem == 0 );

  s = svd(x);
  REQUIRE( s.n_elem == 0 );

  result = svd(u, s, v, x);
  REQUIRE( result == true );
  REQUIRE( u.n_elem == 0 );
  REQUIRE( s.n_elem == 0 );
  REQUIRE( v.n_elem == 0 );
  }



TEMPLATE_TEST_CASE("single_element_svd", "[svd]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(1, 1);
  x(0, 0) = eT(5);

  Col<eT> s;
  Mat<eT> u, v;

  bool result = svd(s, x);
  REQUIRE( result == true );
  REQUIRE( s.n_elem == 1 );
  REQUIRE( eT(s[0]) == Approx(eT(5)) );

  s = svd(x);
  REQUIRE( s.n_elem == 1 );
  REQUIRE( eT(s[0]) == Approx(eT(5)) );

  result = svd(u, s, v, x);
  REQUIRE( result == true );
  REQUIRE( u.n_elem == 1 );
  REQUIRE( eT(u[0]) == Approx(eT(1)) );
  REQUIRE( s.n_elem == 1 );
  REQUIRE( eT(s[0]) == Approx(eT(5)) );
  REQUIRE( v.n_elem == 1 );
  REQUIRE( eT(v[0]) == Approx(eT(1)) );
  }



template<typename eT>
void test_svd_reconstruction(const uword n_rows, const uword n_cols)
  {
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(n_rows, n_cols);
  x.randu();
  // Add a bit down the diagonal.
  for (uword i = 0; i < std::min(n_rows, n_cols); ++i)
    {
    x(i, i) += eT(0.5);
    }

  Col<eT> s;
  Mat<eT> u, v;

  bool result = svd(s, x);
  REQUIRE( result == true );
  REQUIRE( s.n_elem == std::min(n_rows, n_cols) );

  s = svd(x);
  REQUIRE( s.n_elem == std::min(n_rows, n_cols) );

  result = svd(u, s, v, x);
  REQUIRE( result == true );
  REQUIRE( u.n_rows == n_rows );
  REQUIRE( u.n_cols == n_rows );
  REQUIRE( s.n_elem == std::min(n_rows, n_cols) );
  REQUIRE( v.n_rows == n_cols );
  REQUIRE( v.n_cols == n_cols );

  // Check the residual of the reconstructed matrix.
  Mat<eT> ds(n_rows, n_cols);
  ds.zeros();
  // This could be much faster once we have .diag() and similar.
  for (uword i = 0; i < std::min(n_rows, n_cols); ++i)
    {
    ds(i, i) = eT(s[i]);
    }

  Mat<eT> y = u * ds * v.t();

  REQUIRE( accu(abs(y - x)) <= 0.1 );

  }



TEMPLATE_TEST_CASE("square_svd", "[svd]", float, double)
  {
  test_svd_reconstruction<TestType>(25, 25);
  test_svd_reconstruction<TestType>(50, 50);
  test_svd_reconstruction<TestType>(100, 100);
  }



TEMPLATE_TEST_CASE("tall_svd", "[svd]", float, double)
  {
  test_svd_reconstruction<TestType>(25, 50);
  test_svd_reconstruction<TestType>(50, 100);
  test_svd_reconstruction<TestType>(100, 200);
  }



TEMPLATE_TEST_CASE("wide_svd", "[svd]", float, double)
  {
  test_svd_reconstruction<TestType>(50, 25);
  test_svd_reconstruction<TestType>(100, 50);
  test_svd_reconstruction<TestType>(200, 100);
  }



TEMPLATE_TEST_CASE("arma_svd_comparison", "[svd]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(50, 50);
  x.randu();
  for (uword i = 0; i < 50; ++i)
    {
    x(i, i) += eT(0.5);
    }
  arma::Mat<eT> x_cpu(x);

  Col<eT> s = svd(x);
  arma::Col<eT> s_cpu = svd(x_cpu);

  REQUIRE( s.n_elem == s_cpu.n_elem );
  arma::Col<eT> s2_cpu(s);
  for (uword i = 0; i < s.n_elem; ++i)
    {
    if (std::is_same<eT, float>::value)
      REQUIRE( eT(s2_cpu[i]) == Approx(eT(s_cpu[i])).epsilon(0.001).margin(0.0001) );
    else
      REQUIRE( eT(s2_cpu[i]) == Approx(eT(s_cpu[i])).margin(0.00001) );
    }

  // Make sure that U and V come out similarly too.
  Mat<eT> u, v;
  bool result = svd(u, s, v, x);
  REQUIRE( result == true );

  arma::Mat<eT> u_cpu, v_cpu;
  result = svd(u_cpu, s_cpu, v_cpu, x_cpu);
  REQUIRE( result == true );

  // Check sizes.
  REQUIRE( u.n_rows == u_cpu.n_rows );
  REQUIRE( u.n_cols == u_cpu.n_cols );
  REQUIRE( s.n_elem == s_cpu.n_elem );
  REQUIRE( v.n_rows == v_cpu.n_rows );
  REQUIRE( v.n_cols == v_cpu.n_cols );

  arma::Mat<eT> v2_cpu(v);
  arma::Mat<eT> u2_cpu(u);
  s2_cpu = arma::Col<eT>(s);

  // The singular vectors that are returned may point opposite directions, so we check with abs().
  for (uword i = 0; i < u.n_elem; ++i)
    {
    if (std::is_same<eT, float>::value)
      REQUIRE( std::abs(eT(u2_cpu[i])) == Approx(std::abs(eT(u_cpu[i]))).epsilon(0.05).margin(0.0001) );
    else
      REQUIRE( std::abs(eT(u2_cpu[i])) == Approx(std::abs(eT(u_cpu[i]))).epsilon(0.001).margin(0.00001) );
    }

  for (uword i = 0; i < s.n_elem; ++i)
    {
    if (std::is_same<eT, float>::value)
      REQUIRE( eT(s2_cpu[i]) == Approx(eT(s_cpu[i])).epsilon(0.05).margin(0.0001) );
    else
      REQUIRE( eT(s2_cpu[i]) == Approx(eT(s_cpu[i])).epsilon(0.001).margin(0.00001) );
    }

  for (uword i = 0; i < v.n_elem; ++i)
    {
    if (std::is_same<eT, float>::value)
      REQUIRE( std::abs(eT(v2_cpu[i])) == Approx(std::abs(eT(v_cpu[i]))).epsilon(0.05).margin(0.0001) );
    else
      REQUIRE( std::abs(eT(v2_cpu[i])) == Approx(std::abs(eT(v_cpu[i]))).epsilon(0.001).margin(0.00001) );
    }
  }
