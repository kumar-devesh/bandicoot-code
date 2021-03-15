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
void test_accu_small()
  {
  Mat<eT> x(4, 4);
  for (uword i = 0; i < 16; ++i)
    x[i] = i + 1;

  eT sum = accu(x);

  REQUIRE(sum == Approx(eT(136)) );
  }



TEST_CASE("accu_small")
  {
  test_accu_small<double>();
  test_accu_small<float>();
  test_accu_small<u32>();
  test_accu_small<s32>();
  test_accu_small<u64>();
  test_accu_small<s64>();
  }



template<typename eT>
void test_accu_1()
  {
  Mat<eT> x(80, 80);
  for (uword i = 0; i < 6400; ++i)
    x[i] = i + 1;

  eT sum = accu(x);

  REQUIRE(sum == Approx(eT(20483200)) );
  }



TEST_CASE("accu_1")
  {
  test_accu_1<double>();
  test_accu_1<float>();
  test_accu_1<u32>();
  test_accu_1<s32>();
  test_accu_1<u64>();
  test_accu_1<s64>();
  }



template<typename eT>
void test_accu_strange_size()
  {
  Col<eT> x(608);

  for(uword i = 0; i < 608; ++i)
    x[i] = i + 1;

  eT sum = accu(x);

  REQUIRE(sum == Approx(eT(185136)));
  }



TEST_CASE("accu_strange_size")
  {
  test_accu_strange_size<double>();
  test_accu_strange_size<float>();
  test_accu_strange_size<u32>();
  test_accu_strange_size<s32>();
  test_accu_strange_size<u64>();
  test_accu_strange_size<s64>();
  }



template<typename eT>
void test_accu_large()
  {
  arma::Col<eT> cpu_x = arma::conv_to<arma::Col<eT>>::from(arma::randu<arma::Col<double>>(100000) * 10.0);
  cpu_x.randu();
  Col<eT> x(cpu_x);

  eT cpu_sum = accu(cpu_x);
  eT sum = accu(x);

  REQUIRE(sum == Approx(cpu_sum));
  }



TEST_CASE("accu_large")
  {
  test_accu_large<double>();
  test_accu_large<float>();
  test_accu_large<u32>();
  test_accu_large<s32>();
  test_accu_large<u64>();
  test_accu_large<s64>();
  }


template<typename eT>
void test_accu_2()
  {
  Mat<eT> x(10, 5);
  x.randu();
  x += eT(1);

  eT sum = accu(x);

  REQUIRE( sum >= eT(50) );
  REQUIRE( sum <= eT(100) );
  }



TEST_CASE("accu_2")
  {
  test_accu_2<double>();
  test_accu_2<float>();
  }


template<typename eT>
void test_accu_subview_1()
  {
  Mat<eT> x(5, 5);
  for (uword i = 0; i < 25; ++i)
    x[i] = i + 1;

  eT sum1 = accu(x.cols(1, 2));
  eT sum2 = accu(x.rows(3, 4));
  eT sum3 = accu(x.submat(1, 1, 3, 3));
  eT sum4 = accu(x.submat(1, 1, 1, 1));

  REQUIRE( sum1 == Approx(eT(105)) );
  REQUIRE( sum2 == Approx(eT(145)) );
  REQUIRE( sum3 == Approx(eT(117)) );
  REQUIRE( sum4 == Approx(eT(7)) );
  }



TEST_CASE("accu_subview_1")
  {
  test_accu_subview_1<double>();
  test_accu_subview_1<float>();
  test_accu_subview_1<u32>();
  test_accu_subview_1<s32>();
  test_accu_subview_1<u64>();
  test_accu_subview_1<s64>();
  }
