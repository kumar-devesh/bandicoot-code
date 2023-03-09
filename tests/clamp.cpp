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

TEMPLATE_TEST_CASE("clamp_empty", "[clamp]", double, float, u32, s32, u64, s64)
  {
  std::cout << "clamp empty, don't do anything\n";
  }

TEMPLATE_TEST_CASE("clamp_basic", "[clamp]", double, float, u32, s32, u64, s64)
  {
  std::cout << "enter test\n";
  Mat<TestType> x = randi<Mat<TestType>>(40, 50, distr_param(0, 50));
  std::cout << "created x\n";
  Mat<TestType> y = clamp(x, TestType(10), TestType(20));
  std::cout << "did clamp operation\n";

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );
  std::cout << "checked size\n";

  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( TestType(y[i]) >= TestType(10) );
    REQUIRE( TestType(y[i]) <= TestType(20) );
    std::cout << "checked element " << i << "\n";
    }
  }



TEMPLATE_TEST_CASE("clamp_member_basic", "[clamp]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(40, 50, distr_param(0, 50));
  x.clamp(TestType(10), TestType(20));

  REQUIRE( x.n_rows == 40 );
  REQUIRE( x.n_cols == 50 );

  for (uword i = 0; i < x.n_elem; ++i)
    {
    REQUIRE( TestType(x[i]) >= TestType(10) );
    REQUIRE( TestType(x[i]) <= TestType(20) );
    }
  }



TEMPLATE_TEST_CASE("clamp_single_value", "[clamp]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(60, 30, distr_param(0, 50));
  Mat<TestType> y = clamp(x, TestType(1), TestType(1));

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );
  for (uword i = 0; i < x.n_elem; ++i)
    {
    REQUIRE( TestType(y[i]) == Approx(TestType(1)) );
    }
  }



TEMPLATE_TEST_CASE("clamp_all_outside_range", "[clamp]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(20, 30, distr_param(0, 1));
  x *= TestType(10);
  Mat<TestType> y = clamp(x, TestType(5), TestType(6));

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    if (TestType(x[i]) == Approx(TestType(0)))
      {
      REQUIRE( TestType(y[i]) == Approx(TestType(5)) );
      }
    else
      {
      REQUIRE( TestType(y[i]) == Approx(TestType(6)) );
      }
    }
  }



TEMPLATE_TEST_CASE("clamp_empty_matrix", "[clamp]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x;
  Mat<TestType> y = clamp(x, TestType(10), TestType(11));

  REQUIRE( y.n_rows == 0 );
  REQUIRE( y.n_cols == 0 );
  REQUIRE( y.n_elem == 0 );
  }



TEMPLATE_TEST_CASE("clamp_single_element", "[clamp]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x(1, 1);
  x[0] = TestType(5);
  x.clamp(TestType(3), TestType(4));

  REQUIRE( x.n_elem == 1 );
  REQUIRE( TestType(x[0]) == Approx(TestType(4)) );
  }



TEMPLATE_TEST_CASE("clamp_unaligned_subview", "[clamp]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(30, 40, distr_param(0, 50));
  x = clamp(x.submat(0, 0, 10, 10), TestType(5), TestType(10));

  REQUIRE( x.n_rows == 11 );
  REQUIRE( x.n_cols == 11 );
  for (uword i = 0; i < x.n_elem; ++i)
    {
    REQUIRE( TestType(x[i]) >= TestType(5) );
    REQUIRE( TestType(x[i]) <= TestType(10) );
    }
  }



TEMPLATE_TEST_CASE("clamp_op", "[clamp]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(30, 40, distr_param(0, 50));
  Mat<TestType> y = clamp(repmat(x, 2, 2) + 4, TestType(10), TestType(15));

  REQUIRE( y.n_rows == x.n_rows * 2 );
  REQUIRE( y.n_cols == x.n_cols * 2 );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( TestType(y[i]) >= TestType(10) );
    REQUIRE( TestType(y[i]) <= TestType(15) );
    }
  }



TEMPLATE_TEST_CASE("op_with_clamp", "[clamp]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(30, 40, distr_param(0, 50));
  Mat<TestType> y = repmat(clamp(x, TestType(10), TestType(20)), 2, 2) + 4;

  REQUIRE( y.n_rows == x.n_rows * 2 );
  REQUIRE( y.n_cols == x.n_cols * 2 );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( TestType(y[i]) >= TestType(14) );
    REQUIRE( TestType(y[i]) <= TestType(24) );
    }
  }



TEMPLATE_TEST_CASE("subview_inplace_clamp", "[clamp]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x(50, 50);
  x.fill(TestType(50));
  x.submat(5, 5, 10, 10).clamp(TestType(10), TestType(20));

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      if (c >= 5 && c <= 10 && r >= 5 && r <= 10)
        {
        REQUIRE( TestType(x(r, c)) == Approx(TestType(20)) );
        }
      else
        {
        REQUIRE( TestType(x(r, c)) == Approx(TestType(50)) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE(
  "clamp_pre_conv_to",
  "[clamp]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  Mat<eT1> x = randi<Mat<eT1>>(30, 25, distr_param(0, 50));
  Mat<eT2> y = conv_to<Mat<eT2>>::from(clamp(x, eT1(10), eT1(20)));

  REQUIRE(y.n_rows == 30);
  REQUIRE(y.n_cols == 25);

  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT2(y[i]) >= eT2(10) );
    REQUIRE( eT2(y[i]) <= eT2(20) );
    }
  }



TEMPLATE_TEST_CASE(
  "clamp_post_conv_to",
  "[clamp]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  Mat<eT1> x = randi<Mat<eT1>>(30, 25, distr_param(0, 50));
  Mat<eT2> y = clamp(conv_to<Mat<eT2>>::from(x), eT2(10), eT2(20));

  REQUIRE(y.n_rows == 30);
  REQUIRE(y.n_cols == 25);

  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT2(y[i]) >= eT2(10) );
    REQUIRE( eT2(y[i]) <= eT2(20) );
    }
  }
