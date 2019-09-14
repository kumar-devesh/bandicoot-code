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

#include <bandicoot>
#include <armadillo>
#include "catch.hpp"

using namespace coot;

template<typename eT>
void test_mat_mul_square()
  {
  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y;

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEST_CASE("mat_mul_square")
  {
  test_mat_mul_square<float>();
  test_mat_mul_square<double>();
  }



template<typename eT>
void test_mat_mul_square_trans_a()
  {
  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t() * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y;

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEST_CASE("mat_mul_square_trans_a")
  {
  test_mat_mul_square_trans_a<float>();
  test_mat_mul_square_trans_a<double>();
  }



template<typename eT>
void test_mat_mul_square_trans_b()
  {
  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y.t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEST_CASE("mat_mul_square_trans_b")
  {
  test_mat_mul_square_trans_b<float>();
  test_mat_mul_square_trans_b<double>();
  }



template<typename eT>
void test_mat_mul_square_trans_both()
  {
  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t()  * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y.t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEST_CASE("mat_mul_square_trans_both")
  {
  test_mat_mul_square_trans_both<float>();
  test_mat_mul_square_trans_both<double>();
  }



template<typename eT>
void test_mat_mul_nonsquare()
  {
  arma::Mat<eT> cpu_x(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_y(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y;

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEST_CASE("mat_mul_nonsquare")
  {
  test_mat_mul_nonsquare<float>();
  test_mat_mul_nonsquare<double>();
  }



template<typename eT>
void test_mat_mul_nonsquare_trans()
  {
  arma::Mat<eT> cpu_x(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y;

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEST_CASE("mat_mul_nonsquare_trans")
  {
  test_mat_mul_nonsquare_trans<float>();
  test_mat_mul_nonsquare_trans<double>();
  }



template<typename eT>
void test_mat_vec_mul_square()
  {
  arma::Mat<eT> cpu_A(50, 50, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A * cpu_x;

  Mat<eT> A(cpu_A);
  Mat<eT> x(cpu_x);
  Mat<eT> y = A * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEST_CASE("mat_vec_mul_square")
  {
  test_mat_vec_mul_square<float>();
  test_mat_vec_mul_square<double>();
  }



template<typename eT>
void test_mat_vec_mul_square_trans()
  {
  arma::Mat<eT> cpu_A(50, 50, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.t() * cpu_x;

  Mat<eT> A(cpu_A);
  Mat<eT> x(cpu_x);
  Mat<eT> y = A.t() * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEST_CASE("mat_vec_mul_square_trans")
  {
  test_mat_vec_mul_square_trans<float>();
  test_mat_vec_mul_square_trans<double>();
  }



template<typename eT>
void test_mat_vec_mul_nonsquare()
  {
  arma::Mat<eT> cpu_A(10, 50, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A * cpu_x;

  Mat<eT> A(cpu_A);
  Mat<eT> x(cpu_x);
  Mat<eT> y = A * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEST_CASE("mat_vec_mul_nonsquare")
  {
  test_mat_vec_mul_nonsquare<float>();
  test_mat_vec_mul_nonsquare<double>();
  }



template<typename eT>
void test_mat_vec_mul_nonsquare_trans()
  {
  arma::Mat<eT> cpu_A(50, 10, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.t() * cpu_x;

  Mat<eT> A(cpu_A);
  Mat<eT> x(cpu_x);
  Mat<eT> y = A.t() * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEST_CASE("mat_vec_mul_nonsquare_trans")
  {
  test_mat_vec_mul_nonsquare_trans<float>();
  test_mat_vec_mul_nonsquare_trans<double>();
  }
