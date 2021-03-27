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
void test_min_small()
  {
  Col<eT> x(16);
  for (uword i = 0; i < 16; ++i)
    x[i] = i + 1;

  eT min_val = min(x);

  REQUIRE(min_val == Approx(eT(1)) );
  }



TEST_CASE("min_small")
  {
  test_min_small<double>();
  test_min_small<float>();
  test_min_small<u32>();
  test_min_small<s32>();
  test_min_small<u64>();
  test_min_small<s64>();
  }



template<typename eT>
void test_min_1()
  {
  Col<eT> x(6400);
  for (uword i = 0; i < 6400; ++i)
    x[i] = (6400 - i);

  eT min_val = min(x);

  REQUIRE(min_val == Approx(eT(1)) );
  }



TEST_CASE("min_1")
  {
  test_min_1<double>();
  test_min_1<float>();
  test_min_1<u32>();
  test_min_1<s32>();
  test_min_1<u64>();
  test_min_1<s64>();
  }



template<typename eT>
void test_min_strange_size()
  {
  Col<eT> x(608);

  for(uword i = 0; i < 608; ++i)
    x[i] = i + 1;

  eT min_val = min(x);

  REQUIRE(min_val == Approx(eT(1)));
  }



TEST_CASE("min_strange_size")
  {
  test_min_strange_size<double>();
  test_min_strange_size<float>();
  test_min_strange_size<u32>();
  test_min_strange_size<s32>();
  test_min_strange_size<u64>();
  test_min_strange_size<s64>();
  }



template<typename eT>
void test_min_large()
  {
  arma::Col<eT> cpu_x = arma::conv_to<arma::Col<eT>>::from(arma::randu<arma::Col<double>>(100000) * 10.0);
  cpu_x.randu();
  Col<eT> x(cpu_x);

  eT cpu_min = min(cpu_x);
  eT min_val = min(x);

  REQUIRE(min_val == Approx(cpu_min));
  }



TEST_CASE("min_large")
  {
  test_min_large<double>();
  test_min_large<float>();
  test_min_large<u32>();
  test_min_large<s32>();
  test_min_large<u64>();
  test_min_large<s64>();
  }


template<typename eT>
void test_min_2()
  {
  Col<eT> x(50);
  x.randu();
  x += eT(1);

  eT min_val = min(x);

  REQUIRE( min_val >= eT(1) );
  REQUIRE( min_val <= eT(2) );
  }



TEST_CASE("min_2")
  {
  test_min_2<double>();
  test_min_2<float>();
  }
