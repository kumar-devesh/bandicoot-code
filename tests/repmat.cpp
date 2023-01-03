// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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

TEMPLATE_TEST_CASE("repmat_basic", "[repmat]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(5, 5, distr_param(0, 50));

  Mat<TestType> y = repmat(x, 5, 5);

  REQUIRE(y.n_rows == 25);
  REQUIRE(y.n_cols == 25);
  for (size_t c = 0; c < 25; ++c)
    {
    for (size_t r = 0; r < 25; ++r)
      {
      // Approx() is for floating-point imprecision (though that shouldn't be an issue really).
      REQUIRE( TestType(y(r, c)) == Approx(TestType(x(r % 5, c % 5))) );
      }
    }
  }


TEMPLATE_TEST_CASE("repmat_zero_size", "[repmat]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(5, 5, distr_param(0, 50));
  Mat<TestType> y = repmat(x, 0, 0);

  REQUIRE(y.n_rows == 0);
  REQUIRE(y.n_cols == 0);

  y = repmat(x, 0, 5);

  REQUIRE(y.n_rows == 0);
  REQUIRE(y.n_cols == 25);

  y = repmat(x, 5, 0);

  REQUIRE(y.n_rows == 25);
  REQUIRE(y.n_cols == 0);
  }



TEMPLATE_TEST_CASE("repmat_same_size", "[repmat]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(5, 5, distr_param(0, 50));
  Mat<TestType> y = repmat(x, 1, 1);

  REQUIRE(x.n_rows == y.n_rows);
  REQUIRE(x.n_cols == y.n_cols);

  for (size_t c = 0; c < y.n_cols; ++c)
    {
    for (size_t r = 0; r < y.n_rows; ++r)
      {
      REQUIRE( TestType(y(r, c)) == Approx(TestType(x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("repmat_alias", "[repmat]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(5, 5, distr_param(0, 50));
  Mat<TestType> x_old(x);
  x = repmat(x, 2, 2);

  REQUIRE(x.n_rows == 10);
  REQUIRE(x.n_cols == 10);

  for (size_t c = 0; c < x.n_cols; ++c)
    {
    for (size_t r = 0; r < x.n_rows; ++r)
      {
      REQUIRE( TestType(x(r, c)) == Approx(TestType(x_old(r % x_old.n_rows, c % x_old.n_cols))) );
      }
    }
  }



TEMPLATE_TEST_CASE("repmat_alias_same_size", "[repmat]", double, float, u32, s32, u64, s64)
  {
  Mat<TestType> x = randi<Mat<TestType>>(5, 5, distr_param(0, 50));
  Mat<TestType> x_old(x);
  x = repmat(x, 1, 1);

  REQUIRE(x.n_rows == 5);
  REQUIRE(x.n_cols == 5);

  for (size_t c = 0; c < x.n_cols; ++c)
    {
    for (size_t r = 0; r < x.n_rows; ++r)
      {
      REQUIRE( TestType(x(r, c)) == Approx(TestType(x_old(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE(
  "repmat_pre_conv_to",
  "[repmat]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  Mat<eT1> x = randi<Mat<eT1>>(5, 5, distr_param(0, 50));
  Mat<eT2> y = conv_to<Mat<eT2>>::from(repmat(x, 5, 5));

  REQUIRE(y.n_rows == 25);
  REQUIRE(y.n_cols == 25);

  for (uword c = 0; c < y.n_cols; ++c)
    {
    for (uword r = 0; r < y.n_rows; ++r)
      {
      REQUIRE( eT2(y(r, c)) == Approx(eT2(eT1(x(r % x.n_rows, c % x.n_cols)))) );
      }
    }
  }



TEMPLATE_TEST_CASE(
  "repmat_post_conv_to",
  "[repmat]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  Mat<eT1> x = randi<Mat<eT1>>(5, 5, distr_param(0, 50));
  Mat<eT2> y = repmat(conv_to<Mat<eT2>>::from(x), 5, 5);

  REQUIRE(y.n_rows == 25);
  REQUIRE(y.n_cols == 25);

  for (uword c = 0; c < y.n_cols; ++c)
    {
    for (uword r = 0; r < y.n_rows; ++r)
      {
      REQUIRE( eT2(y(r, c)) == Approx(eT2(eT1(x(r % x.n_rows, c % x.n_cols)))) );
      }
    }
  }
