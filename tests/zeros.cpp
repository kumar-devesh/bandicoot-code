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


TEST_CASE("zeros_1")
  {
  test_zeros<double>(5, 5);
  test_zeros<float>(5, 5);
  test_zeros<u32>(5, 5);
  test_zeros<s32>(5, 5);
  test_zeros<u64>(5, 5);
  test_zeros<s64>(5, 5);
  }



TEST_CASE("zeros_2")
  {
  test_zeros<double>(10, 50);
  test_zeros<float>(10, 50);
  test_zeros<u32>(10, 50);
  test_zeros<s32>(10, 50);
  test_zeros<u64>(10, 50);
  test_zeros<s64>(10, 50);
  }



TEST_CASE("zeros_3")
  {
  test_zeros<double>(50, 10);
  test_zeros<float>(50, 10);
  test_zeros<u32>(50, 10);
  test_zeros<s32>(50, 10);
  test_zeros<u64>(50, 10);
  test_zeros<s64>(50, 10);
  }

TEST_CASE("zeros_4")
  {
  test_zeros<double>(100, 1);
  test_zeros<float>(100, 1);
  test_zeros<u32>(100, 1);
  test_zeros<s32>(100, 1);
  test_zeros<u64>(100, 1);
  test_zeros<s64>(100, 1);
  }