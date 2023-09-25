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

TEMPLATE_TEST_CASE("max_small", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(16);
  for (uword i = 0; i < 16; ++i)
    x[i] = i + 1;

  eT max_val = max(x);

  REQUIRE(max_val == Approx(eT(16)) );
  }



TEMPLATE_TEST_CASE("max_1", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(6400);
  for (uword i = 0; i < 6400; ++i)
    x[i] = i + 1;

  eT max_val = max(x);

  REQUIRE(max_val == Approx(eT(6400)) );
  }



TEMPLATE_TEST_CASE("max_strange_size", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(608);

  for(uword i = 0; i < 608; ++i)
    x[i] = i + 1;

  eT max_val = max(x);

  REQUIRE(max_val == Approx(eT(608)));
  }



TEMPLATE_TEST_CASE("max_large", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Col<eT> cpu_x = arma::conv_to<arma::Col<eT>>::from(arma::randu<arma::Col<double>>(100000) * 10.0);
  cpu_x.randu();
  Col<eT> x(cpu_x);

  eT cpu_max = max(cpu_x);
  eT max_val = max(x);

  REQUIRE(max_val == Approx(cpu_max));
  }


TEMPLATE_TEST_CASE("max_2", "[max]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(50);
  x.randu();
  x += eT(1);

  eT max_val = max(x);

  REQUIRE( max_val >= eT(1) );
  REQUIRE( max_val <= eT(2) );
  }



TEMPLATE_TEST_CASE("max_3", "[max]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 1);
  x[0] = 4.7394;
  x[1] = 4.7299;
  x[2] = 6.2287;
  x[3] = 4.5893;
  x[4] = 4.4460;
  x[5] = 4.7376;
  x[6] = 4.7432;
  x[7] = 3.8152;
  x[8] = 4.1442;
  x[9] = 5.2339;

  eT max_val  = max(max(x));
  eT max_val2 = max(max(abs(x)));

  REQUIRE( max_val  == Approx(6.2287) );
  REQUIRE( max_val2 == Approx(6.2287) );
  }




TEMPLATE_TEST_CASE("max_colwise_1", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x, 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(c)) );
    }
  }



TEMPLATE_TEST_CASE("max_colwise_2", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x, 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(9)) );
    }
  }



TEMPLATE_TEST_CASE("max_rowwise_1", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x, 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(9)) );
    }
  }



TEMPLATE_TEST_CASE("max_rowwise_2", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x, 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(r)) );
    }
  }



TEMPLATE_TEST_CASE("subview_max_colwise_1", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x.submat(1, 1, 8, 8), 0);

  REQUIRE( s.n_rows == 1 );
  REQUIRE( s.n_cols == 8 );
  for (uword c = 0; c < 8; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(c + 1)) );
    }
  }



TEMPLATE_TEST_CASE("subview_max_colwise_2", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x.submat(1, 1, 8, 8), 0);

  REQUIRE( s.n_rows == 1 );
  REQUIRE( s.n_cols == 8 );
  for (uword c = 0; c < 8; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(8)) );
    }
  }



TEMPLATE_TEST_CASE("subview_max_colwise_full", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x.submat(0, 0, 9, 9), 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(c)) );
    }
  }



TEMPLATE_TEST_CASE("subview_max_rowwise_1", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x.submat(1, 1, 8, 8), 1);

  REQUIRE( s.n_rows == 8 );
  REQUIRE( s.n_cols == 1 );
  for (uword r = 0; r < 8; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(8)) );
    }
  }



TEMPLATE_TEST_CASE("subview_max_rowwise_2", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x.submat(1, 1, 8, 8), 1);

  REQUIRE( s.n_rows == 8 );
  REQUIRE( s.n_cols == 1 );
  for (uword r = 0; r < 8; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(r + 1)) );
    }
  }



TEMPLATE_TEST_CASE("subview_max_rowwise_full", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = max(x.submat(0, 0, 9, 9), 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(r)) );
    }
  }



TEMPLATE_TEST_CASE("two_mat_max_simple", "[max]", float, double, u32, s32, u64, s64)
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

  Mat<eT> z = max(x, y);

  REQUIRE( z.n_rows == 50 );
  REQUIRE( z.n_cols == 50 );

  REQUIRE( all( all( z == eT(1) ) ) );
  }



TEMPLATE_TEST_CASE("two_mat_max_expr", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 100, fill::randu);
  Mat<eT> y(100, 100, fill::randn);

  Mat<eT> z = max(x.t(), (3 * y + 1));

  Mat<eT> x_expr = x.t();
  Mat<eT> y_expr = (3 * y + 1);
  Mat<eT> z_expr = max(x_expr, y_expr);

  REQUIRE( z.n_rows == 100 );
  REQUIRE( z.n_cols == 100 );

  REQUIRE( all( all( z == z_expr ) ) );
  }



TEMPLATE_TEST_CASE(
  "two_mat_max_conv_to_inputs",
  "[max]",
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

  Mat<eT1> out = max(in_a, conv_to<Mat<eT1>>::from(in_b));

  Mat<eT1> in_b_conv = conv_to<Mat<eT1>>::from(in_b);
  Mat<eT1> out_ref = max(in_a, in_b_conv);

  REQUIRE( out.n_rows == out_ref.n_rows );
  REQUIRE( out.n_cols == out_ref.n_cols );
  REQUIRE( all( all( out == out_ref ) ) );

  Mat<eT2> out2 = max(conv_to<Mat<eT2>>::from(in_a), in_b);

  Mat<eT2> in_a_conv = conv_to<Mat<eT2>>::from(in_a);
  Mat<eT2> out2_ref = max(in_a_conv, in_b);

  REQUIRE( out2.n_rows == out2_ref.n_rows );
  REQUIRE( out2.n_cols == out2_ref.n_cols );
  REQUIRE( all( all( out2 == out2_ref ) ) );

  Mat<eT2> out3 = max(conv_to<Mat<eT2>>::from(in_a), conv_to<Mat<eT2>>::from(in_b_conv));

  Mat<eT2> in_b_conv2 = conv_to<Mat<eT2>>::from(in_b_conv);
  Mat<eT2> out3_ref = max(in_a_conv, in_b_conv2);

  REQUIRE( out3.n_rows == out3_ref.n_rows );
  REQUIRE( out3.n_cols == out3_ref.n_cols );
  REQUIRE( all( all( out3 == out3_ref ) ) );
  }



TEMPLATE_TEST_CASE(
  "two_mat_max_conv_to_outputs",
  "[max]",
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

  Mat<eT2> out = conv_to<Mat<eT2>>::from(max(in_a, in_b));

  Mat<eT1> out_pre_conv = max(in_a, in_b);
  Mat<eT2> out_ref = conv_to<Mat<eT2>>::from(out_pre_conv);

  REQUIRE( out.n_rows == out_ref.n_rows );
  REQUIRE( out.n_cols == out_ref.n_cols );
  REQUIRE( all( all( out == out_ref ) ) );
  }
