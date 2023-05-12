// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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

// Simple hardcoded test
TEMPLATE_TEST_CASE("simple_hardcoded_join_cols", "[join_cols]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x(2, 2); // [[1, 2; 3 4]]
  x(0, 0) = eT(1);
  x(1, 0) = eT(2);
  x(0, 1) = eT(3);
  x(1, 1) = eT(4);
  Mat<eT> y(2, 2); // [[5, 6; 7, 8]]
  y(0, 0) = eT(5);
  y(1, 0) = eT(6);
  y(0, 1) = eT(7);
  y(1, 1) = eT(8);

  Mat<eT> z1 = join_cols(x, y);
  Mat<eT> z2 = join_vert(x, y);

  REQUIRE( z1.n_rows == 4 );
  REQUIRE( z1.n_cols == 2 );
  REQUIRE( z2.n_rows == 4 );
  REQUIRE( z2.n_cols == 2 );

  REQUIRE( eT(z1(0, 0)) == Approx(eT(1)).margin(1e-5) );
  REQUIRE( eT(z1(1, 0)) == Approx(eT(2)).margin(1e-5) );
  REQUIRE( eT(z1(2, 0)) == Approx(eT(5)).margin(1e-5) );
  REQUIRE( eT(z1(3, 0)) == Approx(eT(6)).margin(1e-5) );
  REQUIRE( eT(z1(0, 1)) == Approx(eT(3)).margin(1e-5) );
  REQUIRE( eT(z1(1, 1)) == Approx(eT(4)).margin(1e-5) );
  REQUIRE( eT(z1(2, 1)) == Approx(eT(7)).margin(1e-5) );
  REQUIRE( eT(z1(3, 1)) == Approx(eT(8)).margin(1e-5) );

  REQUIRE( eT(z2(0, 0)) == Approx(eT(1)).margin(1e-5) );
  REQUIRE( eT(z2(1, 0)) == Approx(eT(2)).margin(1e-5) );
  REQUIRE( eT(z2(2, 0)) == Approx(eT(5)).margin(1e-5) );
  REQUIRE( eT(z2(3, 0)) == Approx(eT(6)).margin(1e-5) );
  REQUIRE( eT(z2(0, 1)) == Approx(eT(3)).margin(1e-5) );
  REQUIRE( eT(z2(1, 1)) == Approx(eT(4)).margin(1e-5) );
  REQUIRE( eT(z2(2, 1)) == Approx(eT(7)).margin(1e-5) );
  REQUIRE( eT(z2(3, 1)) == Approx(eT(8)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("simple_hardcoded_three_join_cols", "[join_cols]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x(2, 2); // [[1, 2; 3 4]]
  x(0, 0) = eT(1);
  x(1, 0) = eT(2);
  x(0, 1) = eT(3);
  x(1, 1) = eT(4);
  Mat<eT> y(2, 2); // [[5, 6; 7, 8]]
  y(0, 0) = eT(5);
  y(1, 0) = eT(6);
  y(0, 1) = eT(7);
  y(1, 1) = eT(8);
  Mat<eT> z(2, 2); // [[9, 10; 11, 12]]
  z(0, 0) = eT(9);
  z(1, 0) = eT(10);
  z(0, 1) = eT(11);
  z(1, 1) = eT(12);

  Mat<eT> z1 = join_cols(x, y, z);
  Mat<eT> z2 = join_vert(x, y, z);

  REQUIRE( z1.n_rows == 6 );
  REQUIRE( z1.n_cols == 2 );
  REQUIRE( z2.n_rows == 6 );
  REQUIRE( z2.n_cols == 2 );

  REQUIRE( eT(z1(0, 0)) == Approx(eT(1)).margin(1e-5) );
  REQUIRE( eT(z1(1, 0)) == Approx(eT(2)).margin(1e-5) );
  REQUIRE( eT(z1(2, 0)) == Approx(eT(5)).margin(1e-5) );
  REQUIRE( eT(z1(3, 0)) == Approx(eT(6)).margin(1e-5) );
  REQUIRE( eT(z1(4, 0)) == Approx(eT(9)).margin(1e-5) );
  REQUIRE( eT(z1(5, 0)) == Approx(eT(10)).margin(1e-5) );
  REQUIRE( eT(z1(0, 1)) == Approx(eT(3)).margin(1e-5) );
  REQUIRE( eT(z1(1, 1)) == Approx(eT(4)).margin(1e-5) );
  REQUIRE( eT(z1(2, 1)) == Approx(eT(7)).margin(1e-5) );
  REQUIRE( eT(z1(3, 1)) == Approx(eT(8)).margin(1e-5) );
  REQUIRE( eT(z1(4, 1)) == Approx(eT(11)).margin(1e-5) );
  REQUIRE( eT(z1(5, 1)) == Approx(eT(12)).margin(1e-5) );

  REQUIRE( eT(z2(0, 0)) == Approx(eT(1)).margin(1e-5) );
  REQUIRE( eT(z2(1, 0)) == Approx(eT(2)).margin(1e-5) );
  REQUIRE( eT(z2(2, 0)) == Approx(eT(5)).margin(1e-5) );
  REQUIRE( eT(z2(3, 0)) == Approx(eT(6)).margin(1e-5) );
  REQUIRE( eT(z2(4, 0)) == Approx(eT(9)).margin(1e-5) );
  REQUIRE( eT(z2(5, 0)) == Approx(eT(10)).margin(1e-5) );
  REQUIRE( eT(z2(0, 1)) == Approx(eT(3)).margin(1e-5) );
  REQUIRE( eT(z2(1, 1)) == Approx(eT(4)).margin(1e-5) );
  REQUIRE( eT(z2(2, 1)) == Approx(eT(7)).margin(1e-5) );
  REQUIRE( eT(z2(3, 1)) == Approx(eT(8)).margin(1e-5) );
  REQUIRE( eT(z2(4, 1)) == Approx(eT(11)).margin(1e-5) );
  REQUIRE( eT(z2(5, 1)) == Approx(eT(12)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("simple_hardcoded_four_join_cols", "[join_cols]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x(2, 2); // [[1, 2; 3 4]]
  x(0, 0) = eT(1);
  x(1, 0) = eT(2);
  x(0, 1) = eT(3);
  x(1, 1) = eT(4);
  Mat<eT> y(2, 2); // [[5, 6; 7, 8]]
  y(0, 0) = eT(5);
  y(1, 0) = eT(6);
  y(0, 1) = eT(7);
  y(1, 1) = eT(8);
  Mat<eT> z(2, 2); // [[9, 10; 11, 12]]
  z(0, 0) = eT(9);
  z(1, 0) = eT(10);
  z(0, 1) = eT(11);
  z(1, 1) = eT(12);
  Mat<eT> w(2, 2); // [[13, 14; 15, 16]]
  w(0, 0) = eT(13);
  w(1, 0) = eT(14);
  w(0, 1) = eT(15);
  w(1, 1) = eT(16);

  Mat<eT> z1 = join_cols(x, y, z, w);
  Mat<eT> z2 = join_vert(x, y, z, w);

  REQUIRE( z1.n_rows == 8 );
  REQUIRE( z1.n_cols == 2 );
  REQUIRE( z2.n_rows == 8 );
  REQUIRE( z2.n_cols == 2 );

  REQUIRE( eT(z1(0, 0)) == Approx(eT(1)).margin(1e-5) );
  REQUIRE( eT(z1(1, 0)) == Approx(eT(2)).margin(1e-5) );
  REQUIRE( eT(z1(2, 0)) == Approx(eT(5)).margin(1e-5) );
  REQUIRE( eT(z1(3, 0)) == Approx(eT(6)).margin(1e-5) );
  REQUIRE( eT(z1(4, 0)) == Approx(eT(9)).margin(1e-5) );
  REQUIRE( eT(z1(5, 0)) == Approx(eT(10)).margin(1e-5) );
  REQUIRE( eT(z1(6, 0)) == Approx(eT(13)).margin(1e-5) );
  REQUIRE( eT(z1(7, 0)) == Approx(eT(14)).margin(1e-5) );
  REQUIRE( eT(z1(0, 1)) == Approx(eT(3)).margin(1e-5) );
  REQUIRE( eT(z1(1, 1)) == Approx(eT(4)).margin(1e-5) );
  REQUIRE( eT(z1(2, 1)) == Approx(eT(7)).margin(1e-5) );
  REQUIRE( eT(z1(3, 1)) == Approx(eT(8)).margin(1e-5) );
  REQUIRE( eT(z1(4, 1)) == Approx(eT(11)).margin(1e-5) );
  REQUIRE( eT(z1(5, 1)) == Approx(eT(12)).margin(1e-5) );
  REQUIRE( eT(z1(6, 1)) == Approx(eT(15)).margin(1e-5) );
  REQUIRE( eT(z1(7, 1)) == Approx(eT(16)).margin(1e-5) );

  REQUIRE( eT(z2(0, 0)) == Approx(eT(1)).margin(1e-5) );
  REQUIRE( eT(z2(1, 0)) == Approx(eT(2)).margin(1e-5) );
  REQUIRE( eT(z2(2, 0)) == Approx(eT(5)).margin(1e-5) );
  REQUIRE( eT(z2(3, 0)) == Approx(eT(6)).margin(1e-5) );
  REQUIRE( eT(z2(4, 0)) == Approx(eT(9)).margin(1e-5) );
  REQUIRE( eT(z2(5, 0)) == Approx(eT(10)).margin(1e-5) );
  REQUIRE( eT(z2(6, 0)) == Approx(eT(13)).margin(1e-5) );
  REQUIRE( eT(z2(7, 0)) == Approx(eT(14)).margin(1e-5) );
  REQUIRE( eT(z2(0, 1)) == Approx(eT(3)).margin(1e-5) );
  REQUIRE( eT(z2(1, 1)) == Approx(eT(4)).margin(1e-5) );
  REQUIRE( eT(z2(2, 1)) == Approx(eT(7)).margin(1e-5) );
  REQUIRE( eT(z2(3, 1)) == Approx(eT(8)).margin(1e-5) );
  REQUIRE( eT(z2(4, 1)) == Approx(eT(11)).margin(1e-5) );
  REQUIRE( eT(z2(5, 1)) == Approx(eT(12)).margin(1e-5) );
  REQUIRE( eT(z2(6, 1)) == Approx(eT(15)).margin(1e-5) );
  REQUIRE( eT(z2(7, 1)) == Approx(eT(16)).margin(1e-5) );
  }



// Large random matrices test
TEMPLATE_TEST_CASE("random_join_cols", "[join_cols]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  for (size_t t = 6; t < 12; ++t)
    {
    const uword n_cols = (uword) std::pow(2.0, (double) t) + 3;
    const uword n_rows_1 = (uword) std::pow(2.0, (double) (t - 1)) - 1;
    const uword n_rows_2 = (uword) std::pow(2.0, (double) (t - 1)) + 2;

    Mat<eT> x = randi<Mat<eT>>(n_rows_1, n_cols, distr_param(0, 100));
    Mat<eT> y = randi<Mat<eT>>(n_rows_2, n_cols, distr_param(100, 200));

    Mat<eT> z11 = join_cols(x, y);
    Mat<eT> z12 = join_vert(x, y);

    REQUIRE( z11.n_rows == n_rows_1 + n_rows_2 );
    REQUIRE( z11.n_cols == n_cols );
    REQUIRE( z12.n_rows == n_rows_1 + n_rows_2 );
    REQUIRE( z12.n_cols == n_cols );

    arma::Mat<eT> x_cpu(x);
    arma::Mat<eT> y_cpu(y);
    arma::Mat<eT> z11_cpu(z11);
    arma::Mat<eT> z12_cpu(z12);

    arma::Mat<eT> z11_cpu_a = z11_cpu.rows(0, n_rows_1 - 1);
    arma::Mat<eT> z11_cpu_b = z11_cpu.rows(n_rows_1, n_rows_1 + n_rows_2 - 1);
    arma::Mat<eT> z12_cpu_a = z12_cpu.rows(0, n_rows_1 - 1);
    arma::Mat<eT> z12_cpu_b = z12_cpu.rows(n_rows_1, n_rows_1 + n_rows_2 - 1);

    REQUIRE( arma::approx_equal( z11_cpu_a, x_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z11_cpu_b, y_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z12_cpu_a, x_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z12_cpu_b, y_cpu, "absdiff", 1e-5 ) );

    Mat<eT> z21 = join_cols(y, x);
    Mat<eT> z22 = join_vert(y, x);

    REQUIRE( z21.n_rows == n_rows_2 + n_rows_1 );
    REQUIRE( z21.n_cols == n_cols );
    REQUIRE( z22.n_rows == n_rows_2 + n_rows_1 );
    REQUIRE( z22.n_cols == n_cols );

    arma::Mat<eT> z21_cpu(z21);
    arma::Mat<eT> z22_cpu(z22);

    arma::Mat<eT> z21_cpu_a = z21_cpu.rows(0, n_rows_2 - 1);
    arma::Mat<eT> z21_cpu_b = z21_cpu.rows(n_rows_2, n_rows_2 + n_rows_1 - 1);
    arma::Mat<eT> z22_cpu_a = z22_cpu.rows(0, n_rows_2 - 1);
    arma::Mat<eT> z22_cpu_b = z22_cpu.rows(n_rows_2, n_rows_2 + n_rows_1 - 1);

    REQUIRE( arma::approx_equal( z21_cpu_a, y_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z21_cpu_b, x_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z22_cpu_a, y_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z22_cpu_b, x_cpu, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("random_three_join_cols", "[join_cols]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  for (size_t t = 6; t < 12; ++t)
    {
    const uword n_rows_1 = (uword) std::pow(2.0, (double) (t - 1)) - 1;
    const uword n_rows_2 = (uword) std::pow(2.0, (double) (t - 1)) + 2;
    const uword n_rows_3 = (uword) std::pow(2.0, (double) (t - 1)) - 2;
    const uword n_cols = (uword) std::pow(2.0, (double) t) + 3;

    Mat<eT> x = randi<Mat<eT>>(n_rows_1, n_cols, distr_param(0, 100));
    Mat<eT> y = randi<Mat<eT>>(n_rows_2, n_cols, distr_param(100, 200));
    Mat<eT> z = randi<Mat<eT>>(n_rows_3, n_cols, distr_param(200, 300));

    Mat<eT> z11 = join_cols(x, y, z);
    Mat<eT> z12 = join_vert(x, y, z);

    REQUIRE( z11.n_rows == n_rows_1 + n_rows_2 + n_rows_3 );
    REQUIRE( z11.n_cols == n_cols );
    REQUIRE( z12.n_rows == n_rows_1 + n_rows_2 + n_rows_3 );
    REQUIRE( z12.n_cols == n_cols );

    arma::Mat<eT> x_cpu(x);
    arma::Mat<eT> y_cpu(y);
    arma::Mat<eT> z_cpu(z);
    arma::Mat<eT> z11_cpu(z11);
    arma::Mat<eT> z12_cpu(z12);

    arma::Mat<eT> z11_cpu_a = z11_cpu.rows(0, n_rows_1 - 1);
    arma::Mat<eT> z11_cpu_b = z11_cpu.rows(n_rows_1, n_rows_1 + n_rows_2 - 1);
    arma::Mat<eT> z11_cpu_c = z11_cpu.rows(n_rows_1 + n_rows_2, n_rows_1 + n_rows_2 + n_rows_3 - 1);
    arma::Mat<eT> z12_cpu_a = z12_cpu.rows(0, n_rows_1 - 1);
    arma::Mat<eT> z12_cpu_b = z12_cpu.rows(n_rows_1, n_rows_1 + n_rows_2 - 1);
    arma::Mat<eT> z12_cpu_c = z12_cpu.rows(n_rows_1 + n_rows_2, n_rows_1 + n_rows_2 + n_rows_3 - 1);

    REQUIRE( arma::approx_equal( z11_cpu_a, x_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z11_cpu_b, y_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z11_cpu_c, z_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z12_cpu_a, x_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z12_cpu_b, y_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z12_cpu_c, z_cpu, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("random_four_join_cols", "[join_cols]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  for (size_t t = 6; t < 10; ++t)
    {
    const uword n_rows_1 = (uword) std::pow(2.0, (double) (t - 1)) - 1;
    const uword n_rows_2 = (uword) std::pow(2.0, (double) (t - 1)) + 2;
    const uword n_rows_3 = (uword) std::pow(2.0, (double) (t - 1)) - 2;
    const uword n_rows_4 = (uword) std::pow(2.0, (double) (t - 1)) + 3;
    const uword n_cols = (uword) std::pow(2.0, (double) t) + 3;

    Mat<eT> x = randi<Mat<eT>>(n_rows_1, n_cols, distr_param(0, 100));
    Mat<eT> y = randi<Mat<eT>>(n_rows_2, n_cols, distr_param(100, 200));
    Mat<eT> z = randi<Mat<eT>>(n_rows_3, n_cols, distr_param(200, 300));
    Mat<eT> w = randi<Mat<eT>>(n_rows_4, n_cols, distr_param(300, 400));

    Mat<eT> z11 = join_cols(x, y, z, w);
    Mat<eT> z12 = join_vert(x, y, z, w);

    REQUIRE( z11.n_rows == n_rows_1 + n_rows_2 + n_rows_3 + n_rows_4 );
    REQUIRE( z11.n_cols == n_cols );
    REQUIRE( z12.n_rows == n_rows_1 + n_rows_2 + n_rows_3 + n_rows_4 );
    REQUIRE( z12.n_cols == n_cols );

    arma::Mat<eT> x_cpu(x);
    arma::Mat<eT> y_cpu(y);
    arma::Mat<eT> z_cpu(z);
    arma::Mat<eT> w_cpu(w);
    arma::Mat<eT> z11_cpu(z11);
    arma::Mat<eT> z12_cpu(z12);

    arma::Mat<eT> z11_cpu_a = z11_cpu.rows(0, n_rows_1 - 1);
    arma::Mat<eT> z11_cpu_b = z11_cpu.rows(n_rows_1, n_rows_1 + n_rows_2 - 1);
    arma::Mat<eT> z11_cpu_c = z11_cpu.rows(n_rows_1 + n_rows_2, n_rows_1 + n_rows_2 + n_rows_3 - 1);
    arma::Mat<eT> z11_cpu_d = z11_cpu.rows(n_rows_1 + n_rows_2 + n_rows_3, n_rows_1 + n_rows_2 + n_rows_3 + n_rows_4 - 1);
    arma::Mat<eT> z12_cpu_a = z12_cpu.rows(0, n_rows_1 - 1);
    arma::Mat<eT> z12_cpu_b = z12_cpu.rows(n_rows_1, n_rows_1 + n_rows_2 - 1);
    arma::Mat<eT> z12_cpu_c = z12_cpu.rows(n_rows_1 + n_rows_2, n_rows_1 + n_rows_2 + n_rows_3 - 1);
    arma::Mat<eT> z12_cpu_d = z12_cpu.rows(n_rows_1 + n_rows_2 + n_rows_3, n_rows_1 + n_rows_2 + n_rows_3 + n_rows_4 - 1);

    REQUIRE( arma::approx_equal( z11_cpu_a, x_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z11_cpu_b, y_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z11_cpu_c, z_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z11_cpu_d, w_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z12_cpu_a, x_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z12_cpu_b, y_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z12_cpu_c, z_cpu, "absdiff", 1e-5 ) );
    REQUIRE( arma::approx_equal( z12_cpu_d, w_cpu, "absdiff", 1e-5 ) );
    }
  }



// One matrix is empty test
TEST_CASE("one_empty_matrix_join_cols", "[join_cols]")
  {
  mat x = randu<mat>(2, 5);
  mat y(0, 5);

  mat z1 = join_cols(x, y);
  mat z2 = join_cols(y, x);
  mat z3 = join_vert(x, y);
  mat z4 = join_vert(y, x);

  REQUIRE( z1.n_rows == 2 );
  REQUIRE( z1.n_cols == 5 );
  REQUIRE( z2.n_rows == 2 );
  REQUIRE( z2.n_cols == 5 );
  REQUIRE( z3.n_rows == 2 );
  REQUIRE( z3.n_cols == 5 );
  REQUIRE( z4.n_rows == 2 );
  REQUIRE( z4.n_cols == 5 );

  arma::mat x_cpu(x);
  arma::mat z1_cpu(z1);
  arma::mat z2_cpu(z2);
  arma::mat z3_cpu(z3);
  arma::mat z4_cpu(z4);

  REQUIRE( arma::approx_equal( z1_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z3_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z4_cpu, x_cpu, "absdiff", 1e-5 ) );
  }



// One matrix is empty and has the wrong number of cols
TEST_CASE("one_wrong_empty_matrix_join_cols", "[join_cols]")
  {
  mat x = randu<mat>(2, 5);
  mat y(0, 3);

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  mat z;
  REQUIRE_THROWS( z = join_cols(x, y) );
  REQUIRE_THROWS( z = join_cols(y, x) );
  REQUIRE_THROWS( z = join_vert(x, y) );
  REQUIRE_THROWS( z = join_vert(y, x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// Matrices do not match in number of cols
TEST_CASE("mismatched_n_cols_join_cols", "[join_cols]")
  {
  mat x = randu<mat>(2, 5);
  mat y = randu<mat>(2, 6);

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  mat z;
  REQUIRE_THROWS( z = join_cols(x, y) );
  REQUIRE_THROWS( z = join_cols(y, x) );
  REQUIRE_THROWS( z = join_vert(x, y) );
  REQUIRE_THROWS( z = join_vert(y, x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



TEST_CASE("mismatched_n_cols_three_join_cols", "[join_cols]")
  {
  mat x = randu<mat>(5, 2);
  mat y(3, 7);
  mat z(6, 1);

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  mat out;
  REQUIRE_THROWS( out = join_cols(x, y, z) );
  REQUIRE_THROWS( out = join_vert(x, y, z) );

  y = randu<mat>(5, 2);
  REQUIRE_THROWS( out = join_cols(x, y, z) );
  REQUIRE_THROWS( out = join_vert(x, y, z) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



TEST_CASE("mismatched_n_cols_four_join_cols", "[join_cols]")
  {
  mat x = randu<mat>(5, 2);
  mat y(3, 7);
  mat z(6, 1);
  mat w(12, 3);

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  mat out;
  REQUIRE_THROWS( out = join_cols(x, y, z, w) );
  REQUIRE_THROWS( out = join_vert(x, y, z, w) );

  y = randu<mat>(5, 2);
  REQUIRE_THROWS( out = join_cols(x, y, z, w) );
  REQUIRE_THROWS( out = join_vert(x, y, z, w) );

  z = randu<mat>(5, 2);
  REQUIRE_THROWS( out = join_cols(x, y, z, w) );
  REQUIRE_THROWS( out = join_vert(x, y, z, w) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// Both matrices are empty but have a nonzero number of cols
TEST_CASE("empty_matrices_nonzero_cols", "[join_cols]")
  {
  mat x(0, 1);
  mat y(0, 123);

  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  mat z;
  REQUIRE_THROWS( z = join_cols(x, y) );
  REQUIRE_THROWS( z = join_cols(y, x) );
  REQUIRE_THROWS( z = join_vert(x, y) );
  REQUIRE_THROWS( z = join_vert(y, x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// Both matrices are totally empty
TEST_CASE("totally_empty_matrices_join_cols", "[join_cols]")
  {
  mat x, y;

  mat z1 = join_cols(x, y);
  mat z2 = join_cols(y, x);
  mat z3 = join_vert(x, y);
  mat z4 = join_vert(y, x);

  REQUIRE( z1.n_rows == 0 );
  REQUIRE( z1.n_cols == 0 );
  REQUIRE( z2.n_rows == 0 );
  REQUIRE( z2.n_cols == 0 );
  REQUIRE( z3.n_rows == 0 );
  REQUIRE( z3.n_cols == 0 );
  REQUIRE( z4.n_rows == 0 );
  REQUIRE( z4.n_cols == 0 );
  }



// One input is an expr
TEMPLATE_TEST_CASE("one_input_expr_join_cols", "[join_cols]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(150, 250, distr_param(0, 100));
  Mat<eT> y = randi<Mat<eT>>(213, 250, distr_param(100, 200));
  Mat<eT> x_mod = 3 * x + 4;
  Mat<eT> y_mod = 2 * y + 3;

  Mat<eT> z1 = join_cols(x, 2 * y + 3);
  Mat<eT> z2 = join_vert(x, 2 * y + 3);
  Mat<eT> z3 = join_cols(3 * x + 4, y);
  Mat<eT> z4 = join_vert(3 * x + 4, y);

  REQUIRE( z1.n_rows == 363 );
  REQUIRE( z1.n_cols == 250 );
  REQUIRE( z2.n_rows == 363 );
  REQUIRE( z2.n_cols == 250 );
  REQUIRE( z3.n_rows == 363 );
  REQUIRE( z3.n_cols == 250 );
  REQUIRE( z4.n_rows == 363 );
  REQUIRE( z4.n_cols == 250 );

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y_cpu(y);
  arma::Mat<eT> x_mod_cpu(x_mod);
  arma::Mat<eT> y_mod_cpu(y_mod);

  arma::Mat<eT> z1_cpu(z1);
  arma::Mat<eT> z1_cpu_a = z1_cpu.rows(0, 149);
  arma::Mat<eT> z1_cpu_b = z1_cpu.rows(150, 362);
  arma::Mat<eT> z2_cpu(z2);
  arma::Mat<eT> z2_cpu_a = z2_cpu.rows(0, 149);
  arma::Mat<eT> z2_cpu_b = z2_cpu.rows(150, 362);
  arma::Mat<eT> z3_cpu(z3);
  arma::Mat<eT> z3_cpu_a = z3_cpu.rows(0, 149);
  arma::Mat<eT> z3_cpu_b = z3_cpu.rows(150, 362);
  arma::Mat<eT> z4_cpu(z4);
  arma::Mat<eT> z4_cpu_a = z4_cpu.rows(0, 149);
  arma::Mat<eT> z4_cpu_b = z4_cpu.rows(150, 362);

  REQUIRE( arma::approx_equal( z1_cpu_a, x_cpu,     "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z1_cpu_b, y_mod_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_a, x_cpu,     "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_b, y_mod_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z3_cpu_a, x_mod_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z3_cpu_b, y_cpu,     "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z4_cpu_a, x_mod_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z4_cpu_b, y_cpu,     "absdiff", 1e-5 ) );
  }



// Both inputs are an expr
TEMPLATE_TEST_CASE("two_input_exprs_join_cols", "[join_cols]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(50, 100, distr_param(0, 100));
  Mat<eT> y = randi<Mat<eT>>(75, 50, distr_param(100, 200));
  Mat<eT> x_mod = repmat(x, 1, 3);
  Mat<eT> y_mod = repmat(y, 2, 6);

  Mat<eT> z1 = join_cols(repmat(x, 1, 3), repmat(y, 2, 6));
  Mat<eT> z2 = join_vert(repmat(x, 1, 3), repmat(y, 2, 6));

  REQUIRE( z1.n_rows == 200 );
  REQUIRE( z1.n_cols == 300 );
  REQUIRE( z2.n_rows == 200 );
  REQUIRE( z2.n_cols == 300 );

  arma::Mat<eT> z1_cpu(z1);
  arma::Mat<eT> z1_cpu_a = z1_cpu.rows(0, 49);
  arma::Mat<eT> z1_cpu_b = z1_cpu.rows(50, 199);
  arma::Mat<eT> z2_cpu(z2);
  arma::Mat<eT> z2_cpu_a = z2_cpu.rows(0, 49);
  arma::Mat<eT> z2_cpu_b = z2_cpu.rows(50, 199);

  arma::Mat<eT> x_mod_cpu(x_mod);
  arma::Mat<eT> y_mod_cpu(y_mod);

  REQUIRE( arma::approx_equal( z1_cpu_a, x_mod_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z1_cpu_b, y_mod_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_a, x_mod_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_b, y_mod_cpu, "absdiff", 1e-5 ) );
  }



// The result is used in an expr
TEMPLATE_TEST_CASE("join_cols_in_expr", "[join_cols]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(10, 15, distr_param(0, 10));
  Mat<eT> y = randi<Mat<eT>>(7, 15, distr_param(10, 20));

  Mat<eT> z1 = vectorise(join_cols(x, y)) + 3;
  Mat<eT> z2 = vectorise(join_vert(x, y)) + 3;

  Mat<eT> tmp = join_cols(x, y);
  Mat<eT> z_ref = vectorise(tmp) + 3;

  arma::Mat<eT> z_ref_cpu(z_ref);
  arma::Mat<eT> z1_cpu(z1);
  arma::Mat<eT> z2_cpu(z2);

  REQUIRE( arma::approx_equal( z1_cpu, z_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu, z_ref_cpu, "absdiff", 1e-5 ) );
  }



// One arg has a conv_to
TEMPLATE_TEST_CASE
  (
  "one_arg_conv_to_join_cols",
  "[join_cols]",
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

  Mat<eT1> x = randi<Mat<eT1>>(143, 10, distr_param(30, 40));
  Mat<eT2> y = randi<Mat<eT2>>(110, 10, distr_param(10, 30));

  Mat<eT1> z1 = join_cols(x, conv_to<Mat<eT1>>::from(y));
  Mat<eT1> z2 = join_vert(x, conv_to<Mat<eT1>>::from(y));
  Mat<eT1> z3 = join_cols(conv_to<Mat<eT1>>::from(y), x);
  Mat<eT1> z4 = join_vert(conv_to<Mat<eT1>>::from(y), x);

  REQUIRE( z1.n_rows == 253 );
  REQUIRE( z1.n_cols == 10 );
  REQUIRE( z2.n_rows == 253 );
  REQUIRE( z2.n_cols == 10 );
  REQUIRE( z3.n_rows == 253 );
  REQUIRE( z3.n_cols == 10 );
  REQUIRE( z4.n_rows == 253 );
  REQUIRE( z4.n_cols == 10 );

  Mat<eT1> y_ref = conv_to<Mat<eT1>>::from(y);

  arma::Mat<eT1> x_cpu(x);
  arma::Mat<eT1> y_ref_cpu(y_ref);
  arma::Mat<eT1> z1_cpu(z1);
  arma::Mat<eT1> z1_cpu_a = z1_cpu.rows(0, 142);
  arma::Mat<eT1> z1_cpu_b = z1_cpu.rows(143, 252);
  arma::Mat<eT1> z2_cpu(z2);
  arma::Mat<eT1> z2_cpu_a = z2_cpu.rows(0, 142);
  arma::Mat<eT1> z2_cpu_b = z2_cpu.rows(143, 252);
  arma::Mat<eT1> z3_cpu(z3);
  arma::Mat<eT1> z3_cpu_a = z3_cpu.rows(0, 109);
  arma::Mat<eT1> z3_cpu_b = z3_cpu.rows(110, 252);
  arma::Mat<eT1> z4_cpu(z4);
  arma::Mat<eT1> z4_cpu_a = z3_cpu.rows(0, 109);
  arma::Mat<eT1> z4_cpu_b = z3_cpu.rows(110, 252);

  REQUIRE( arma::approx_equal( z1_cpu_a, x_cpu,     "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z1_cpu_b, y_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_a, x_cpu,     "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_b, y_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z3_cpu_a, y_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z3_cpu_b, x_cpu,     "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z4_cpu_a, y_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z4_cpu_b, x_cpu,     "absdiff", 1e-5 ) );
  }



// Both args have a conv_to
TEST_CASE("double_conv_to_join_cols", "[join_cols]")
  {
  Mat<u32> x = randi<Mat<u32>>(15, 10, distr_param(0, 10));
  Mat<s32> y = randi<Mat<s32>>(20, 10, distr_param(10, 20));

  Mat<u64> z1 = join_cols(conv_to<Mat<u64>>::from(x), conv_to<Mat<u64>>::from(y));
  Mat<u64> z2 = join_vert(conv_to<Mat<u64>>::from(x), conv_to<Mat<u64>>::from(y));

  REQUIRE( z1.n_rows == 35 );
  REQUIRE( z1.n_cols == 10 );
  REQUIRE( z2.n_rows == 35 );
  REQUIRE( z2.n_cols == 10 );

  Mat<u64> x_ref = conv_to<Mat<u64>>::from(x);
  Mat<u64> y_ref = conv_to<Mat<u64>>::from(y);

  arma::Mat<u64> x_ref_cpu(x_ref);
  arma::Mat<u64> y_ref_cpu(y_ref);
  arma::Mat<u64> z1_cpu(z1);
  arma::Mat<u64> z1_cpu_a = z1_cpu.rows(0, 14);
  arma::Mat<u64> z1_cpu_b = z1_cpu.rows(15, 34);
  arma::Mat<u64> z2_cpu(z2);
  arma::Mat<u64> z2_cpu_a = z2_cpu.rows(0, 14);
  arma::Mat<u64> z2_cpu_b = z2_cpu.rows(15, 34);

  REQUIRE( arma::approx_equal( z1_cpu_a, x_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z1_cpu_b, y_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_a, x_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_b, y_ref_cpu, "absdiff", 1e-5 ) );
  }



// conv_to after join_cols()
TEMPLATE_TEST_CASE
  (
  "conv_to_after_join_cols",
  "[join_cols]",
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

  Mat<eT1> x = randi<Mat<eT1>>(143, 10, distr_param(30, 40));
  Mat<eT1> y = randi<Mat<eT1>>(110, 10, distr_param(10, 30));

  Mat<eT2> z1 = conv_to<Mat<eT2>>::from(join_cols(x, y));
  Mat<eT2> z2 = conv_to<Mat<eT2>>::from(join_vert(x, y));

  REQUIRE( z1.n_rows == 253 );
  REQUIRE( z1.n_cols == 10 );
  REQUIRE( z2.n_rows == 253 );
  REQUIRE( z2.n_cols == 10 );

  Mat<eT2> x_ref = conv_to<Mat<eT2>>::from(x);
  Mat<eT2> y_ref = conv_to<Mat<eT2>>::from(y);

  arma::Mat<eT2> x_ref_cpu(x_ref);
  arma::Mat<eT2> y_ref_cpu(y_ref);
  arma::Mat<eT2> z1_cpu(z1);
  arma::Mat<eT2> z1_cpu_a = z1_cpu.rows(0, 142);
  arma::Mat<eT2> z1_cpu_b = z1_cpu.rows(143, 252);
  arma::Mat<eT2> z2_cpu(z2);
  arma::Mat<eT2> z2_cpu_a = z2_cpu.rows(0, 142);
  arma::Mat<eT2> z2_cpu_b = z2_cpu.rows(143, 252);

  REQUIRE( arma::approx_equal( z1_cpu_a, x_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z1_cpu_b, y_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_a, x_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( z2_cpu_b, y_ref_cpu, "absdiff", 1e-5 ) );
  }



TEST_CASE("alias_join_cols", "[join_cols]")
  {
  mat x = randu<mat>(10, 20);
  mat y = randu<mat>(15, 20);
  mat x_ref(x);

  x = join_cols(x, y);

  REQUIRE( x.n_rows == 25 );
  REQUIRE( x.n_cols == 20 );

  arma::mat x_ref_cpu(x_ref);
  arma::mat y_cpu(y);
  arma::mat x_cpu(x);

  arma::mat x_cpu_1 = x_cpu.rows(0, 9);
  arma::mat x_cpu_2 = x_cpu.rows(10, 24);

  REQUIRE( arma::approx_equal( x_cpu_1, x_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x_cpu_2, y_cpu,     "absdiff", 1e-5 ) );
  }



TEST_CASE("alias_three_join_cols", "[join_cols]")
  {
  mat x = randu<mat>(10, 20);
  mat y = randu<mat>(15, 20);
  mat z = randu<mat>(20, 20);
  mat x_ref(x);

  x = join_cols(x, y, z);

  REQUIRE( x.n_rows == 45 );
  REQUIRE( x.n_cols == 20 );

  arma::mat x_ref_cpu(x_ref);
  arma::mat y_cpu(y);
  arma::mat x_cpu(x);
  arma::mat z_cpu(z);

  arma::mat x_cpu_1 = x_cpu.rows(0, 9);
  arma::mat x_cpu_2 = x_cpu.rows(10, 24);
  arma::mat x_cpu_3 = x_cpu.rows(25, 44);

  REQUIRE( arma::approx_equal( x_cpu_1, x_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x_cpu_2, y_cpu,     "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x_cpu_3, z_cpu,     "absdiff", 1e-5 ) );
  }



TEST_CASE("alias_four_join_cols", "[join_cols]")
  {
  mat x = randu<mat>(10, 20);
  mat y = randu<mat>(15, 20);
  mat z = randu<mat>(20, 20);
  mat w = randu<mat>(25, 20);
  mat x_ref(x);

  x = join_cols(x, y, z, w);

  REQUIRE( x.n_rows == 70 );
  REQUIRE( x.n_cols == 20 );

  arma::mat x_ref_cpu(x_ref);
  arma::mat y_cpu(y);
  arma::mat x_cpu(x);
  arma::mat z_cpu(z);
  arma::mat w_cpu(w);

  arma::mat x_cpu_1 = x_cpu.rows(0, 9);
  arma::mat x_cpu_2 = x_cpu.rows(10, 24);
  arma::mat x_cpu_3 = x_cpu.rows(25, 44);
  arma::mat x_cpu_4 = x_cpu.rows(45, 69);

  REQUIRE( arma::approx_equal( x_cpu_1, x_ref_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x_cpu_2, y_cpu,     "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x_cpu_3, z_cpu,     "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x_cpu_4, w_cpu,     "absdiff", 1e-5 ) );
  }
