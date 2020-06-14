// Copyright 2020 Ryan Curtin (http://www.ratml.org/)
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
void test_dot_1()
  {
  Row<eT> x(10);
  Row<eT> y(10);
  for (uword i = 0; i < 10; ++i)
    {
    x[i] = i + 1;
    y[i] = 10 - i;
    }

  eT d = dot(x, y);

  REQUIRE(d == Approx(eT(220)) );
  }



TEST_CASE("dot_1")
  {
  test_dot_1<double>();
  test_dot_1<float>();
  test_dot_1<u32>();
  test_dot_1<s32>();
  test_dot_1<u64>();
  test_dot_1<s64>();
  }



template<typename eT>
void test_dot_2()
  {
  Col<eT> x(1000);
  Row<eT> y(1000);
  x.randu();
  y.randu();

  eT d = dot(x, y);

  eT manual_dot = eT(0);
  for (uword i = 0; i < 1000; ++i)
    {
    manual_dot += eT(x[i]) * eT(y[i]);
    }

  REQUIRE( d == Approx(manual_dot) );
  }



TEST_CASE("dot_2")
  {
  test_dot_2<double>();
  test_dot_2<float>();
  }



template<typename eT>
void test_mat_dot()
  {
  Mat<eT> x(10, 10);
  Mat<eT> y(10, 10);

  x.randu();
  y.randu();

  eT d = dot(x, y);

  eT manual_dot = eT(0);
  for (uword i = 0; i < 100; ++i)
    {
    manual_dot += eT(x[i]) * eT(y[i]);
    }

  REQUIRE( d == Approx(manual_dot) );
  }



TEST_CASE("mat_dot")
  {
  test_mat_dot<double>();
  test_mat_dot<float>();
  }



template<typename eT>
void test_expr_dot()
  {
  Col<eT> x(10);
  Col<eT> y(10);

  for (uword i = 0; i < 10; ++i)
    {
    x[i] = i + 1;
    y[i] = 10 - i;
    }

  eT d = dot(x % y, (y + eT(2)));

  eT manual_dot = eT(0);
  for (uword i = 0; i < 10; ++i)
    {
    manual_dot += eT(x[i]) * eT(y[i]) * eT(y[i] + eT(2));
    }

  REQUIRE( d == Approx(manual_dot) );
  }



TEST_CASE("expr_dot")
  {
  test_expr_dot<double>();
  test_expr_dot<float>();
  test_expr_dot<u32>();
  test_expr_dot<s32>();
  test_expr_dot<u64>();
  test_expr_dot<s64>();
  }



template<typename eT1, typename eT2>
void test_different_eT_dot()
  {
  Col<eT1> x(10);
  Col<eT2> y(10);

  for (uword i = 0; i < 10; ++i)
    {
    x[i] = i + 1;
    y[i] = 10 - i;
    }

  typedef typename promote_type<eT1, eT2>::result promoted_eT;
  promoted_eT result = dot(x, y);

  REQUIRE(result == Approx(promoted_eT(220)));
  }



TEST_CASE("different_eT_dot")
  {
  test_different_eT_dot<u32, u32>();
  test_different_eT_dot<u32, s32>();
  test_different_eT_dot<u32, u64>();
  test_different_eT_dot<u32, s64>();
  test_different_eT_dot<u32, float>();
  test_different_eT_dot<u32, double>();
  test_different_eT_dot<s32, u32>();
  test_different_eT_dot<s32, s32>();
  test_different_eT_dot<s32, u64>();
  test_different_eT_dot<s32, s64>();
  test_different_eT_dot<s32, float>();
  test_different_eT_dot<s32, double>();
  test_different_eT_dot<u64, u32>();
  test_different_eT_dot<u64, s32>();
  test_different_eT_dot<u64, u64>();
  test_different_eT_dot<u64, s64>();
  test_different_eT_dot<u64, float>();
  test_different_eT_dot<u64, double>();
  test_different_eT_dot<s64, u32>();
  test_different_eT_dot<s64, s32>();
  test_different_eT_dot<s64, u64>();
  test_different_eT_dot<s64, s64>();
  test_different_eT_dot<s64, float>();
  test_different_eT_dot<s64, double>();
  test_different_eT_dot<float, u32>();
  test_different_eT_dot<float, s32>();
  test_different_eT_dot<float, u64>();
  test_different_eT_dot<float, s64>();
  test_different_eT_dot<float, float>();
  test_different_eT_dot<float, double>();
  test_different_eT_dot<double, u32>();
  test_different_eT_dot<double, s32>();
  test_different_eT_dot<double, u64>();
  test_different_eT_dot<double, s64>();
  test_different_eT_dot<double, float>();
  test_different_eT_dot<double, double>();
  test_different_eT_dot<u32, u32>();
  test_different_eT_dot<u32, s32>();
  test_different_eT_dot<u32, u64>();
  test_different_eT_dot<u32, s64>();
  test_different_eT_dot<u32, float>();
  test_different_eT_dot<u32, double>();
  }



// Make sure that dot() returns the expected results when one type is signed
// and the other is unsigned.
template<typename ueT1, typename seT2>
void test_signed_unsigned_dot()
  {
  Col<ueT1> x(10);
  Col<seT2> y(10);

  for (uword i = 0; i < 10; ++i)
    {
    x[i] = i + 1;
    y[i] = -(seT2(i) + 1);
    }

  // This should type-promote to seT1 or similar.
  typedef typename promote_type<ueT1, seT2>::result out_eT;
  out_eT result = dot(x, y);

  REQUIRE(result == Approx(out_eT(-385)));
  }


TEST_CASE("signed_unsigned_dot")
  {
  test_signed_unsigned_dot<u32, s32>();
  test_signed_unsigned_dot<u32, s64>();
  test_signed_unsigned_dot<u32, float>();
  test_signed_unsigned_dot<u32, double>();
  test_signed_unsigned_dot<u64, s32>();
  test_signed_unsigned_dot<u64, s64>();
  test_signed_unsigned_dot<u64, float>();
  test_signed_unsigned_dot<u64, double>();
  }
