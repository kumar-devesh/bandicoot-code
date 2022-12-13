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
void test_plus()
  {
  Mat<eT> x(5, 5);
  x.fill(eT(0));
  Mat<eT> y(5, 5);
  y.fill(eT(3));

  Mat<eT> z1 = x + y;
  Mat<eT> z2 = y + x;
  x += y;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(3)) );
      REQUIRE( eT(z1(r, c)) == Approx(eT(3)) );
      REQUIRE( eT(z2(r, c)) == Approx(eT(3)) );
      }
    }
  }



TEST_CASE("two_mat_plus")
  {
  test_plus<float>();
  test_plus<double>();
  test_plus<u32>();
  test_plus<s32>();
  test_plus<u64>();
  test_plus<s64>();
  }



template<typename eT>
void test_minus()
  {
  Mat<eT> x(5, 5);
  x.fill(eT(10));
  Mat<eT> y(5, 5);
  y.fill(eT(5));

  Mat<eT> z = x - y;
  x -= y;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(5)) );
      REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
      }
    }
  }



TEST_CASE("two_mat_minus")
  {
  test_minus<float>();
  test_minus<double>();
  test_minus<u32>();
  test_minus<s32>();
  test_minus<u64>();
  test_minus<s64>();
  }



template<typename eT>
void test_mul()
  {
  Mat<eT> x(5, 5);
  x.fill(eT(1));
  Mat<eT> y(5, 5);
  y.fill(eT(10));

  Mat<eT> z1 = x % y;
  Mat<eT> z2 = y % x;
  x %= y;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
      REQUIRE( eT(z1(r, c)) == Approx(eT(10)) );
      REQUIRE( eT(z2(r, c)) == Approx(eT(10)) );
      }
    }
  }



TEST_CASE("two_mat_mul")
  {
  test_mul<float>();
  test_mul<double>();
  test_mul<u32>();
  test_mul<s32>();
  test_mul<u64>();
  test_mul<s64>();
  }



template<typename eT>
void test_div()
  {
  Mat<eT> x(5, 5);
  x.fill(eT(10));
  Mat<eT> y(5, 5);
  y.fill(eT(2));

  Mat<eT> z = x / y;
  x /= y;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
      REQUIRE( eT(z(r, c)) == Approx(eT(5)) );
      }
    }
  }



TEST_CASE("two_mat_div")
  {
  test_div<float>();
  test_div<double>();
  test_div<u32>();
  test_div<s32>();
  test_div<u64>();
  test_div<s64>();
  }



template<typename eT>
void test_simple_mul()
  {
  Mat<eT> x(3, 3);
  Mat<eT> y(3, 3);

  x(0, 0) = eT(1);
  x(1, 0) = eT(2);
  x(2, 0) = eT(3);
  x(0, 1) = eT(5);
; x(1, 1) = eT(6);
  x(2, 1) = eT(7);
  x(0, 2) = eT(9);
  x(1, 2) = eT(11);
  x(2, 2) = eT(13);

  y(0, 0) = eT(10);
  y(1, 0) = eT(11);
  y(2, 0) = eT(12);
  y(0, 1) = eT(13);
  y(1, 1) = eT(14);
  y(2, 1) = eT(15);
  y(0, 2) = eT(16);
  y(1, 2) = eT(17);
  y(2, 2) = eT(18);

  Mat<eT> z1 = x * y;
  Mat<eT> z2 = x * y.t();
  Mat<eT> z3 = x.t() * y;
  Mat<eT> z4 = x.t() * y.t();

  y *= x;

  // Check against hand-computed results.
  REQUIRE( eT(z1(0, 0)) == Approx(eT(173)) );
  REQUIRE( eT(z1(1, 0)) == Approx(eT(218)) );
  REQUIRE( eT(z1(2, 0)) == Approx(eT(263)) );
  REQUIRE( eT(z1(0, 1)) == Approx(eT(218)) );
  REQUIRE( eT(z1(1, 1)) == Approx(eT(275)) );
  REQUIRE( eT(z1(2, 1)) == Approx(eT(332)) );
  REQUIRE( eT(z1(0, 2)) == Approx(eT(263)) );
  REQUIRE( eT(z1(1, 2)) == Approx(eT(332)) );
  REQUIRE( eT(z1(2, 2)) == Approx(eT(401)) );

  REQUIRE( eT(z2(0, 0)) == Approx(eT(219)) );
  REQUIRE( eT(z2(1, 0)) == Approx(eT(274)) );
  REQUIRE( eT(z2(2, 0)) == Approx(eT(329)) );
  REQUIRE( eT(z2(0, 1)) == Approx(eT(234)) );
  REQUIRE( eT(z2(1, 1)) == Approx(eT(293)) );
  REQUIRE( eT(z2(2, 1)) == Approx(eT(352)) );
  REQUIRE( eT(z2(0, 2)) == Approx(eT(249)) );
  REQUIRE( eT(z2(1, 2)) == Approx(eT(312)) );
  REQUIRE( eT(z2(2, 2)) == Approx(eT(375)) );

  REQUIRE( eT(z3(0, 0)) == Approx(eT(68)) );
  REQUIRE( eT(z3(1, 0)) == Approx(eT(200)) );
  REQUIRE( eT(z3(2, 0)) == Approx(eT(367)) );
  REQUIRE( eT(z3(0, 1)) == Approx(eT(86)) );
  REQUIRE( eT(z3(1, 1)) == Approx(eT(254)) );
  REQUIRE( eT(z3(2, 1)) == Approx(eT(466)) );
  REQUIRE( eT(z3(0, 2)) == Approx(eT(104)) );
  REQUIRE( eT(z3(1, 2)) == Approx(eT(308)) );
  REQUIRE( eT(z3(2, 2)) == Approx(eT(565)) );

  REQUIRE( eT(z4(0, 0)) == Approx(eT(84)) );
  REQUIRE( eT(z4(1, 0)) == Approx(eT(240)) );
  REQUIRE( eT(z4(2, 0)) == Approx(eT(441)) );
  REQUIRE( eT(z4(0, 1)) == Approx(eT(90)) );
  REQUIRE( eT(z4(1, 1)) == Approx(eT(258)) );
  REQUIRE( eT(z4(2, 1)) == Approx(eT(474)) );
  REQUIRE( eT(z4(0, 2)) == Approx(eT(96)) );
  REQUIRE( eT(z4(1, 2)) == Approx(eT(276)) );
  REQUIRE( eT(z4(2, 2)) == Approx(eT(507)) );

  REQUIRE( eT(y(0, 0)) == Approx(eT(84)) );
  REQUIRE( eT(y(1, 0)) == Approx(eT(90)) );
  REQUIRE( eT(y(2, 0)) == Approx(eT(96)) );
  REQUIRE( eT(y(0, 1)) == Approx(eT(240)) );
  REQUIRE( eT(y(1, 1)) == Approx(eT(258)) );
  REQUIRE( eT(y(2, 1)) == Approx(eT(276)) );
  REQUIRE( eT(y(0, 2)) == Approx(eT(441)) );
  REQUIRE( eT(y(1, 2)) == Approx(eT(474)) );
  REQUIRE( eT(y(2, 2)) == Approx(eT(507)) );
  }



TEST_CASE("simple_mat_mul")
  {
  test_simple_mul<float>();
  test_simple_mul<double>();
//  test_simple_mul<u32>();
//  test_simple_mul<s32>();
//  test_simple_mul<u64>();
//  test_simple_mul<s64>();
  }



template<typename eT>
void test_copy()
{
  Mat<eT> x(10, 10);
  x.randu();

  Mat<eT> y(10, 10);
  y.randu();

  Mat<eT> z;

  y = x;
  z = x;

  REQUIRE( x.n_rows == y.n_rows );
  REQUIRE( x.n_cols == y.n_cols );
  REQUIRE( y.n_rows == z.n_rows );
  REQUIRE( y.n_cols == z.n_cols );

  for (size_t c = 0; c < 10; ++c)
    {
    for (size_t r = 0; r < 10; ++r)
      {
      REQUIRE( eT(x(r, c)) == eT(y(r, c)) );
      REQUIRE( eT(x(r, c)) == eT(z(r, c)) );
      }
    }
  }



TEST_CASE("mat_copy")
  {
  test_copy<float>();
  test_copy<double>();
//  test_copy<u32>();
//  test_copy<s32>();
//  test_copy<u64>();
//  test_copy<s64>();
  }


template<typename eT>
void test_copy_from_dev_mem()
  {
  Mat<eT> x(5, 5);
  for (uword i = 0; i < 25; ++i)
    {
    x(i) = i;
    }

  eT* cpu_mem = new eT[25];

  x.copy_from_dev_mem(cpu_mem, 25);

  for (uword i = 0; i < 25; ++i)
    {
    REQUIRE( cpu_mem[i] == Approx(eT(i)) );
    }

  delete[] cpu_mem;
  }



TEST_CASE("mat_copy_from_dev_mem")
  {
  test_copy_from_dev_mem<float>();
  test_copy_from_dev_mem<double>();
  test_copy_from_dev_mem<u32>();
  test_copy_from_dev_mem<s32>();
  test_copy_from_dev_mem<u64>();
  test_copy_from_dev_mem<s64>();
  }



template<typename eT>
void test_copy_into_dev_mem()
  {
  eT* cpu_mem = new eT[25];

  for (uword i = 0; i < 25; ++i)
    {
    cpu_mem[i] = eT(i);
    }

  Mat<eT> x(5, 5);
  x.copy_into_dev_mem(cpu_mem, 25);

  for (uword i = 0; i < 25; ++i)
    {
    REQUIRE( eT(x(i)) == Approx(eT(i)) );
    }

  delete[] cpu_mem;
  }



TEST_CASE("mat_copy_to_dev_mem")
  {
  test_copy_into_dev_mem<float>();
  test_copy_into_dev_mem<double>();
  test_copy_into_dev_mem<u32>();
  test_copy_into_dev_mem<s32>();
  test_copy_into_dev_mem<u64>();
  test_copy_into_dev_mem<s64>();
  }



template<typename eT>
void test_val_proxy_ops()
  {
  Mat<eT> x(8, 8);
  x.zeros();

  x(1, 1) = eT(3);
  x(1) = eT(2);

  x(2, 2) += eT(5);
  x(2) += eT(3);

  x(3, 3) += eT(10);
  x(3, 3) -= eT(5);
  x(3) += eT(10);
  x(3) -= eT(5);

  x(4, 4) = eT(3);
  x(4, 4) *= eT(2);
  x(4) = eT(3);
  x(4) *= eT(2);

  x(5, 5) = eT(10);
  x(5, 5) /= eT(2);
  x(5) = eT(10);
  x(5) /= eT(2);

  REQUIRE( eT(x(1, 1)) == Approx(eT(3)) );
  REQUIRE( eT(x(1)   ) == Approx(eT(2)) );
  REQUIRE( eT(x(2, 2)) == Approx(eT(5)) );
  REQUIRE( eT(x(2)   ) == Approx(eT(3)) );
  REQUIRE( eT(x(3, 3)) == Approx(eT(5)) );
  REQUIRE( eT(x(3)   ) == Approx(eT(5)) );
  REQUIRE( eT(x(4, 4)) == Approx(eT(6)) );
  REQUIRE( eT(x(4)   ) == Approx(eT(6)) );
  REQUIRE( eT(x(5, 5)) == Approx(eT(5)) );
  REQUIRE( eT(x(5)   ) == Approx(eT(5)) );

  REQUIRE( eT(x(0)   ) == eT(0) );
  REQUIRE( eT(x(6)   ) == eT(0) );
  REQUIRE( eT(x(7)   ) == eT(0) );

  for (uword c = 1; c < 8; ++c)
    {
    for (uword r = 0; r < 8; ++r)
      {
      if (r != c)
        {
        REQUIRE( eT(x(r, c)) == eT(0) );
        }
      }
    }
  }



TEST_CASE("mat_val_proxy_ops_1")
  {
  test_val_proxy_ops<double>();
  test_val_proxy_ops<float>();
  test_val_proxy_ops<u32>();
  test_val_proxy_ops<s32>();
  test_val_proxy_ops<u64>();
  test_val_proxy_ops<s64>();
  }



template<typename eT>
void test_submat_insertion()
  {
  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) = cpu_y;
  x.submat(5, 5, 14, 19) = y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEST_CASE("submat_insertion_1")
  {
  test_submat_insertion<float>();
  test_submat_insertion<double>();
  test_submat_insertion<u32>();
  test_submat_insertion<s32>();
  test_submat_insertion<u64>();
  test_submat_insertion<s64>();
  }



template<typename eT>
void test_submat_add()
  {
  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) += cpu_y;
  x.submat(5, 5, 14, 19) += y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEST_CASE("submat_add_1")
  {
  test_submat_add<float>();
  test_submat_add<double>();
  test_submat_add<u32>();
  test_submat_add<s32>();
  test_submat_add<u64>();
  test_submat_add<s64>();
  }



template<typename eT>
void test_submat_minus()
  {
  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) -= cpu_y;
  x.submat(5, 5, 14, 19) -= y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEST_CASE("submat_minus_1")
  {
  test_submat_minus<float>();
  test_submat_minus<double>();
  test_submat_minus<u32>();
  test_submat_minus<s32>();
  test_submat_minus<u64>();
  test_submat_minus<s64>();
  }



template<typename eT>
void test_submat_schur()
  {
  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) %= cpu_y;
  x.submat(5, 5, 14, 19) %= y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEST_CASE("submat_schur_1")
  {
  test_submat_schur<float>();
  test_submat_schur<double>();
  test_submat_schur<u32>();
  test_submat_schur<s32>();
  test_submat_schur<u64>();
  test_submat_schur<s64>();
  }



template<typename eT>
void test_submat_div()
  {
  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1000, 2000));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(10, 20));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) /= cpu_y;
  x.submat(5, 5, 14, 19) /= y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEST_CASE("submat_div_1")
  {
  test_submat_div<float>();
  test_submat_div<double>();
  test_submat_div<u32>();
  test_submat_div<s32>();
  test_submat_div<u64>();
  test_submat_div<s64>();
  }



template<typename eT>
void test_submat_extract()
  {
  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_y = cpu_x.submat(5, 5, 14, 19);
  y = x.submat(5, 5, 14, 19);

  for (uword c = 0; c < 15; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(cpu_y(r, c))) );
      REQUIRE( eT(y(r, c)) == Approx(eT(x(r + 5, c + 5))) );
      }
    }
  }



TEST_CASE("submat_extract_1")
  {
  test_submat_extract<float>();
  test_submat_extract<double>();
  test_submat_extract<u32>();
  test_submat_extract<s32>();
  test_submat_extract<u64>();
  test_submat_extract<s64>();
  }



template<typename eT>
void test_submat_fill()
  {
  arma::Mat<eT> x(20, 20);
  x.fill(eT(10));

  x.submat(5, 6, 14, 15).fill(eT(2));

  for (uword r = 0; r < 20; ++r)
    {
    for (uword c = 0; c < 20; ++c)
      {
      if (r >= 5 && r <= 14 && c >= 6 && c <= 15)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(2)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
        }
      }
    }
  }



TEST_CASE("submat_fill_1")
  {
  test_submat_fill<float>();
  test_submat_fill<double>();
  test_submat_fill<u32>();
  test_submat_fill<s32>();
  test_submat_fill<u64>();
  test_submat_fill<s64>();
  }
