// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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
void test_ones(const uword n_rows, const uword n_cols)
  {
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(n_rows, n_cols);
  x.ones();

  for (uword c = 0; c < n_cols; ++c)
    {
    for (uword r = 0; r < n_rows; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(1)) );
      }
    }
  }



TEMPLATE_TEST_CASE("ones_1", "[ones]", float, double, u32, s32, u64, s64)
  {
  test_ones<TestType>(5, 5);
  }



TEMPLATE_TEST_CASE("ones_2", "[ones]", float, double, u32, s32, u64, s64)
  {
  test_ones<TestType>(10, 50);
  }



TEMPLATE_TEST_CASE("ones_3", "[ones]", float, double, u32, s32, u64, s64)
  {
  test_ones<TestType>(50, 10);
  }



TEMPLATE_TEST_CASE("ones_empty", "[ones]", float, double, u32, s32, u64, s64)
  {
  // This just checks that there is no crash.
  test_ones<TestType>(0, 0);
  }



TEMPLATE_TEST_CASE("ones_standalone", "[ones]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = ones<Mat<eT>>(10, 10);

  for (uword c = 0; c < x.n_cols; ++c)
    {
    for (uword r = 0; r < x.n_rows; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(1)) );
      }
    }
  }



TEMPLATE_TEST_CASE("ones_standalone_sizemat", "[ones]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> y(10, 10);
  Mat<eT> x = ones<Mat<eT>>(size(y));

  for (uword c = 0; c < x.n_cols; ++c)
    {
    for (uword r = 0; r < x.n_rows; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(1)) );
      }
    }
  }
