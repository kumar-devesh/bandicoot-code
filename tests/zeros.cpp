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
void test_zeros(const uword n_rows, const uword n_cols)
  {
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(n_rows, n_cols);
  x.zeros();

  for (uword c = 0; c < n_cols; ++c)
    {
    for (uword r = 0; r < n_rows; ++r)
      {
      if (r == c)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(0)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == eT(0) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("zeros_1", "[zeros]", double, float, u32, s32, u64, s64)
  {
  test_zeros<TestType>(5, 5);
  }



TEMPLATE_TEST_CASE("zeros_2", "[zeros]", double, float, u32, s32, u64, s64)
  {
  test_zeros<TestType>(10, 50);
  }



TEMPLATE_TEST_CASE("zeros_3", "[zeros]", double, float, u32, s32, u64, s64)
  {
  test_zeros<TestType>(50, 10);
  }



TEMPLATE_TEST_CASE("zeros_4", "[zeros]", double, float, u32, s32, u64, s64)
  {
  test_zeros<TestType>(100, 1);
  }
