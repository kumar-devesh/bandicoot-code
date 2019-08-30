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
