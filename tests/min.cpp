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

#include <armadillo>
#include <bandicoot>
#include "catch.hpp"

using namespace coot;

TEMPLATE_TEST_CASE("min_small", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(16);
  for (uword i = 0; i < 16; ++i)
    x[i] = i + 1;

  eT min_val = min(x);

  REQUIRE(min_val == Approx(eT(1)) );
  }



TEMPLATE_TEST_CASE("min_1", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(6400);
  for (uword i = 0; i < 6400; ++i)
    x[i] = (6400 - i);

  eT min_val = min(x);

  REQUIRE(min_val == Approx(eT(1)) );
  }



TEMPLATE_TEST_CASE("min_strange_size", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(608);

  for(uword i = 0; i < 608; ++i)
    x[i] = i + 1;

  eT min_val = min(x);

  REQUIRE(min_val == Approx(eT(1)));
  }



TEMPLATE_TEST_CASE("min_large", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Col<eT> cpu_x = arma::conv_to<arma::Col<eT>>::from(arma::randu<arma::Col<double>>(100000) * 10.0);
  cpu_x.randu();
  Col<eT> x(cpu_x);

  eT cpu_min = min(cpu_x);
  eT min_val = min(x);

  REQUIRE(min_val == Approx(cpu_min));
  }



TEMPLATE_TEST_CASE("min_2", "[min]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(50);
  x.randu();
  x += eT(1);

  eT min_val = min(x);

  REQUIRE( min_val >= eT(1) );
  REQUIRE( min_val <= eT(2) );
  }



TEMPLATE_TEST_CASE("min_colwise_1", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = min(x, 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(c)) );
    }
  }



TEMPLATE_TEST_CASE("min_colwise_2", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = min(x, 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(0)) );
    }
  }



TEMPLATE_TEST_CASE("min_rowwise_1", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = min(x, 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(0)) );
    }
  }



TEMPLATE_TEST_CASE("min_rowwise_2", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = min(x, 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(r)) );
    }
  }



TEMPLATE_TEST_CASE("subview_min_colwise_1", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = min(x.submat(1, 1, 8, 8), 0);

  REQUIRE( s.n_rows == 1 );
  REQUIRE( s.n_cols == 8 );
  for (uword c = 0; c < 8; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(c + 1)) );
    }
  }



TEMPLATE_TEST_CASE("subview_min_colwise_2", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = min(x.submat(1, 1, 8, 8), 0);

  REQUIRE( s.n_rows == 1 );
  REQUIRE( s.n_cols == 8 );
  for (uword c = 0; c < 8; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(1)) );
    }
  }



TEMPLATE_TEST_CASE("subview_min_colwise_full", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = min(x.submat(0, 0, 9, 9), 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(c)) );
    }
  }



TEMPLATE_TEST_CASE("subview_min_rowwise_1", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = min(x.submat(1, 1, 8, 8), 1);

  REQUIRE( s.n_rows == 8 );
  REQUIRE( s.n_cols == 1 );
  for (uword r = 0; r < 8; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(1)) );
    }
  }



TEMPLATE_TEST_CASE("subview_min_rowwise_2", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = min(x.submat(1, 1, 8, 8), 1);

  REQUIRE( s.n_rows == 8 );
  REQUIRE( s.n_cols == 1 );
  for (uword r = 0; r < 8; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(r + 1)) );
    }
  }



TEMPLATE_TEST_CASE("subview_min_rowwise_full", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = min(x.submat(0, 0, 9, 9), 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(r)) );
    }
  }



TEMPLATE_TEST_CASE("two_mat_min_simple", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(50, 50);
  x.zeros();
  Mat<eT> y(50, 50);
  y.ones();

  Mat<eT> z = min(x, y);

  REQUIRE( z.n_rows == 50 );
  REQUIRE( z.n_cols == 50 );

  REQUIRE( all( all( z == eT(0) ) ) );
  }



TEMPLATE_TEST_CASE("two_mat_min_expr", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 100, fill::randu);
  Mat<eT> y(100, 100, fill::randn);

  Mat<eT> z = min(x.t(), (3 * y + 1));

  Mat<eT> x_expr = x.t();
  Mat<eT> y_expr = (3 * y + 1);
  Mat<eT> z_expr = min(x_expr, y_expr);

  REQUIRE( z.n_rows == 100 );
  REQUIRE( z.n_cols == 100 );

  REQUIRE( all( all( z == z_expr ) ) );
  }



TEMPLATE_TEST_CASE(
  "two_mat_min_conv_to_inputs",
  "[min]",
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

  Mat<eT1> in_a(150, 150, fill::randu);
  Mat<eT2> in_b(150, 150, fill::randu);

  Mat<eT1> out = min(in_a, conv_to<Mat<eT1>>::from(in_b));

  Mat<eT1> in_b_conv = conv_to<Mat<eT1>>::from(in_b);
  Mat<eT1> out_ref = min(in_a, in_b_conv);

  REQUIRE( out.n_rows == out_ref.n_rows );
  REQUIRE( out.n_cols == out_ref.n_cols );
  REQUIRE( all( all( out == out_ref ) ) );

  Mat<eT2> out2 = min(conv_to<Mat<eT2>>::from(in_a), in_b);

  Mat<eT2> in_a_conv = conv_to<Mat<eT2>>::from(in_a);
  Mat<eT2> out2_ref = min(in_a_conv, in_b);

  REQUIRE( out2.n_rows == out2_ref.n_rows );
  REQUIRE( out2.n_cols == out2_ref.n_cols );
  REQUIRE( all( all( out2 == out2_ref ) ) );

  Mat<eT2> out3 = min(conv_to<Mat<eT2>>::from(in_a), conv_to<Mat<eT2>>::from(in_b_conv));

  Mat<eT2> in_b_conv2 = conv_to<Mat<eT2>>::from(in_b_conv);
  Mat<eT2> out3_ref = min(in_a_conv, in_b_conv2);

  REQUIRE( out3.n_rows == out3_ref.n_rows );
  REQUIRE( out3.n_cols == out3_ref.n_cols );
  REQUIRE( all( all( out3 == out3_ref ) ) );
  }



TEMPLATE_TEST_CASE(
  "two_mat_min_conv_to_outputs",
  "[min]",
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

  Mat<eT1> in_a(150, 150, fill::randu);
  Mat<eT1> in_b(150, 150, fill::randu);

  Mat<eT2> out = conv_to<Mat<eT2>>::from(min(in_a, in_b));

  Mat<eT1> out_pre_conv = min(in_a, in_b);
  Mat<eT2> out_ref = conv_to<Mat<eT2>>::from(out_pre_conv);

  REQUIRE( out.n_rows == out_ref.n_rows );
  REQUIRE( out.n_cols == out_ref.n_cols );
  REQUIRE( all( all( out == out_ref ) ) );
  }



TEMPLATE_TEST_CASE("two_subview_min", "[min]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 100, fill::randu);
  Mat<eT> y(100, 100, fill::randn);

  Mat<eT> z = min(x.submat(50, 50, 74, 74), y.submat(25, 25, 49, 49));

  Mat<eT> x_extracted = x.submat(50, 50, 74, 74);
  Mat<eT> y_extracted = y.submat(25, 25, 49, 49);

  Mat<eT> z_ref = min(x_extracted, y_extracted);

  REQUIRE( z.n_rows == 25 );
  REQUIRE( z.n_cols == 25 );
  REQUIRE( all( all( z == z_ref ) ) );
  }
