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

TEMPLATE_TEST_CASE("clamp_basic", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(40, 50, distr_param(0, 50));
  Mat<eT> y = clamp(x, eT(10), eT(20));

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );

  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y[i]) >= eT(10) );
    REQUIRE( eT(y[i]) <= eT(20) );
    }
  }



TEMPLATE_TEST_CASE("clamp_member_basic", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(40, 50, distr_param(0, 50));
  x.clamp(eT(10), eT(20));

  REQUIRE( x.n_rows == 40 );
  REQUIRE( x.n_cols == 50 );

  for (uword i = 0; i < x.n_elem; ++i)
    {
    REQUIRE( eT(x[i]) >= eT(10) );
    REQUIRE( eT(x[i]) <= eT(20) );
    }
  }



TEMPLATE_TEST_CASE("clamp_single_value", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(60, 30, distr_param(0, 50));
  Mat<eT> y = clamp(x, eT(1), eT(1));

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );
  for (uword i = 0; i < x.n_elem; ++i)
    {
    REQUIRE( eT(y[i]) == Approx(eT(1)) );
    }
  }



TEMPLATE_TEST_CASE("clamp_all_outside_range", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 30, distr_param(0, 1));
  x *= eT(10);
  Mat<eT> y = clamp(x, eT(5), eT(6));

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    if (eT(x[i]) == Approx(eT(0)))
      {
      REQUIRE( eT(y[i]) == Approx(eT(5)) );
      }
    else
      {
      REQUIRE( eT(y[i]) == Approx(eT(6)) );
      }
    }
  }



TEMPLATE_TEST_CASE("clamp_empty_matrix", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x;
  Mat<eT> y = clamp(x, eT(10), eT(11));

  REQUIRE( y.n_rows == 0 );
  REQUIRE( y.n_cols == 0 );
  REQUIRE( y.n_elem == 0 );
  }



TEMPLATE_TEST_CASE("clamp_single_element", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(1, 1);
  x[0] = eT(5);
  x.clamp(eT(3), eT(4));

  REQUIRE( x.n_elem == 1 );
  REQUIRE( eT(x[0]) == Approx(eT(4)) );
  }



TEMPLATE_TEST_CASE("clamp_unaligned_subview", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 40, distr_param(0, 50));
  x = clamp(x.submat(0, 0, 10, 10), eT(5), eT(10));

  REQUIRE( x.n_rows == 11 );
  REQUIRE( x.n_cols == 11 );
  for (uword i = 0; i < x.n_elem; ++i)
    {
    REQUIRE( eT(x[i]) >= eT(5) );
    REQUIRE( eT(x[i]) <= eT(10) );
    }
  }



TEMPLATE_TEST_CASE("clamp_op", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 40, distr_param(0, 50));
  Mat<eT> y = clamp(repmat(x, 2, 2) + 4, eT(10), eT(15));

  REQUIRE( y.n_rows == x.n_rows * 2 );
  REQUIRE( y.n_cols == x.n_cols * 2 );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y[i]) >= eT(10) );
    REQUIRE( eT(y[i]) <= eT(15) );
    }
  }



TEMPLATE_TEST_CASE("op_with_clamp", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 40, distr_param(0, 50));
  Mat<eT> y = repmat(clamp(x, eT(10), eT(20)), 2, 2) + 4;

  REQUIRE( y.n_rows == x.n_rows * 2 );
  REQUIRE( y.n_cols == x.n_cols * 2 );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y[i]) >= eT(14) );
    REQUIRE( eT(y[i]) <= eT(24) );
    }
  }



TEMPLATE_TEST_CASE("subview_inplace_clamp", "[clamp]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(50, 50);
  x.fill(eT(50));
  x.submat(5, 5, 10, 10).clamp(eT(10), eT(20));

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      if (c >= 5 && c <= 10 && r >= 5 && r <= 10)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(20)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(50)) );
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

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

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
