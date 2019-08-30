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
void test_fill()
  {
  Mat<eT> x(5, 5);

  x.fill(eT(0));

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(0)) );
      }
    }
  }



TEST_CASE("fill_1")
  {
  printf("double\n");
  test_fill<double>();
  printf("float\n");
  test_fill<float>();
  printf("u32\n");
  test_fill<u32>();
  printf("s32\n");
  test_fill<s32>();
  printf("u64\n");
  test_fill<u64>();
  printf("s64\n");
  test_fill<s64>();
  }



template<typename eT>
void test_fill_2()
  {
  Mat<eT> x(5, 5);

  x.fill(eT(50));;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(50)) );
      }
    }
  }



TEST_CASE("fill_2")
  {
  test_fill_2<float>();
  test_fill_2<double>();
  test_fill_2<u32>();
  test_fill_2<s32>();
  test_fill_2<u64>();
  test_fill_2<s64>();
  }



template<typename eT>
void test_scalar_plus()
  {
  Mat<eT> x(5, 5);
  x.fill(eT(0));

  x += eT(1);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(1)) );
      }
    }
  }



TEST_CASE("scalar_plus")
  {
  test_scalar_plus<float>();
  test_scalar_plus<double>();
  test_scalar_plus<u32>();
  test_scalar_plus<s32>();
  test_scalar_plus<u64>();
  test_scalar_plus<s64>();
  }



template<typename eT>
void test_scalar_minus()
  {
  Mat<eT> x(5, 5);
  x.fill(eT(10));

  x -= eT(5);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
      }
    }
  }



TEST_CASE("scalar_minus")
  {
  test_scalar_minus<float>();
  test_scalar_minus<double>();
  test_scalar_minus<u32>();
  test_scalar_minus<s32>();
  test_scalar_minus<u64>();
  test_scalar_minus<s64>();
  }



template<typename eT>
void test_scalar_mul()
  {
  Mat<eT> x(5, 5);
  x.fill(eT(1));

  x *= eT(10);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
      }
    }
  }



TEST_CASE("scalar_mul")
  {
  test_scalar_mul<float>();
  test_scalar_mul<double>();
  test_scalar_mul<u32>();
  test_scalar_mul<s32>();
  test_scalar_mul<u64>();
  test_scalar_mul<s64>();
  }



template<typename eT>
void test_scalar_div()
  {
  Mat<eT> x(5, 5);
  x.fill(eT(10));

  x /= eT(2);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
      }
    }
  }



TEST_CASE("scalar_div")
  {
  test_scalar_div<float>();
  test_scalar_div<double>();
  test_scalar_div<u32>();
  test_scalar_div<s32>();
  test_scalar_div<u64>();
  test_scalar_div<s64>();
  }
