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
void test_trace_1()
  {
  Mat<eT> x(5, 5);
  x.zeros();
  for (uword i = 0; i < 5; ++i)
    x(i, i) = i + 1;

  eT sum = trace(x);

  REQUIRE(sum == Approx(eT(15)) );
  }



TEST_CASE("trace_1")
  {
  test_trace_1<double>();
  test_trace_1<float>();
  test_trace_1<u32>();
  test_trace_1<s32>();
  test_trace_1<u64>();
  test_trace_1<s64>();
  }



template<typename eT>
void test_trace_2()
  {
  Mat<eT> x(10, 5);
  x.randu();
  x += eT(1);

  eT sum = trace(x);

  eT manual_sum = eT(0);
  for (uword i = 0; i < 5; ++i)
    {
    manual_sum += eT(x(i, i));
    }

  REQUIRE( sum == Approx(manual_sum) );
  }



TEST_CASE("trace_2")
  {
  test_trace_2<double>();
  test_trace_2<float>();
  }



template<typename eT>
void test_trace_3()
  {
  Mat<eT> x(5, 10);
  x.randu();
  x += eT(1);

  eT sum = trace(x);

  eT manual_sum = eT(0);
  for (uword i = 0; i < 5; ++i)
    {
    manual_sum += eT(x(i, i));
    }

  REQUIRE( sum == Approx(manual_sum) );
  }



TEST_CASE("trace_3")
  {
  test_trace_3<double>();
  test_trace_3<float>();
  }
