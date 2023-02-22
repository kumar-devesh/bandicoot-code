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

TEMPLATE_TEST_CASE("max_small", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x(16);
  for (uword i = 0; i < 16; ++i)
    x[i] = i + 1;

  eT max_val = max(x);

  REQUIRE(max_val == Approx(eT(16)) );
  }



TEMPLATE_TEST_CASE("max_1", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x(6400);
  for (uword i = 0; i < 6400; ++i)
    x[i] = i + 1;

  eT max_val = max(x);

  REQUIRE(max_val == Approx(eT(6400)) );
  }



TEMPLATE_TEST_CASE("max_strange_size", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x(608);

  for(uword i = 0; i < 608; ++i)
    x[i] = i + 1;

  eT max_val = max(x);

  REQUIRE(max_val == Approx(eT(608)));
  }



TEMPLATE_TEST_CASE("max_large", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

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

  Col<eT> x(50);
  x.randu();
  x += eT(1);

  eT max_val = max(x);

  REQUIRE( max_val >= eT(1) );
  REQUIRE( max_val <= eT(2) );
  }



// This was found to be failing for the CUDA backend.
// TODO: fix this!
TEMPLATE_TEST_CASE("max_3", "[max]", float, double)
  {
  typedef TestType eT;

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

  eT max_val = max(max(x));

  REQUIRE( max_val == Approx(6.2287) );
  }




TEMPLATE_TEST_CASE("max_colwise_1", "[max]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

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
