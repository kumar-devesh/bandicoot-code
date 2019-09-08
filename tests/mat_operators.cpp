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

  delete cpu_mem;
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

  delete cpu_mem;
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
