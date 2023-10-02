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

TEMPLATE_TEST_CASE("sum_colwise_1", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x, 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(10 * c)) );
    }
  }



TEMPLATE_TEST_CASE("sum_colwise_2", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x, 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(45)) );
    }
  }



TEMPLATE_TEST_CASE("sum_rowwise_1", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x, 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(45)) );
    }
  }



TEMPLATE_TEST_CASE("sum_rowwise_2", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x, 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(10 * r)) );
    }
  }



TEMPLATE_TEST_CASE("subview_sum_colwise_1", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x.submat(1, 1, 8, 8), 0);

  REQUIRE( s.n_rows == 1 );
  REQUIRE( s.n_cols == 8 );
  for (uword c = 0; c < 8; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(8 * (c + 1))) );
    }
  }



TEMPLATE_TEST_CASE("subview_sum_colwise_2", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x.submat(1, 1, 8, 8), 0);

  REQUIRE( s.n_rows == 1 );
  REQUIRE( s.n_cols == 8 );
  for (uword c = 0; c < 8; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(36)) );
    }
  }



TEMPLATE_TEST_CASE("subview_sum_colwise_full", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x.submat(0, 0, 9, 9), 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(10 * c)) );
    }
  }



TEMPLATE_TEST_CASE("subview_sum_rowwise_1", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x.submat(1, 1, 8, 8), 1);

  REQUIRE( s.n_rows == 8 );
  REQUIRE( s.n_cols == 1 );
  for (uword r = 0; r < 8; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(36)) );
    }
  }



TEMPLATE_TEST_CASE("subview_sum_rowwise_2", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x.submat(1, 1, 8, 8), 1);

  REQUIRE( s.n_rows == 8 );
  REQUIRE( s.n_cols == 1 );
  for (uword r = 0; r < 8; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(8 * (r + 1))) );
    }
  }



TEMPLATE_TEST_CASE("subview_sum_rowwise_full", "[sum]", float, double, u32, s32, u64, s64)
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

  Mat<eT> s = sum(x.submat(0, 0, 9, 9), 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(10 * r)) );
    }
  }
