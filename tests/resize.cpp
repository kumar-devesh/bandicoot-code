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

// Resize does not have a custom kernel, so, no need to test with different element types.

TEST_CASE("resize_simple", "[resize]")
  {
  fmat a(5, 5);
  a.ones();

  fmat b = resize(a, 10, 10);

  REQUIRE( b.n_rows == 10 );
  REQUIRE( b.n_cols == 10 );

  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      if (c >= 5 || r >= 5)
        {
        REQUIRE( float(b(r, c)) == Approx(0.0) );
        }
      else
        {
        REQUIRE( float(b(r, c)) == Approx(1.0) );
        }
      }
    }
  }



TEST_CASE("resize_row_only", "[resize]")
  {
  fmat a(5, 5);
  a.ones();

  fmat b = resize(a, 10, 5);

  REQUIRE( b.n_rows == 10 );
  REQUIRE( b.n_cols == 5 );

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      if (r >= 5)
        {
        REQUIRE( float(b(r, c)) == Approx(0.0) );
        }
      else
        {
        REQUIRE( float(b(r, c)) == Approx(1.0) );
        }
      }
    }
  }



TEST_CASE("resize_col_only", "[resize]")
  {
  fmat a(5, 5);
  a.ones();

  fmat b = resize(a, 5, 10);

  REQUIRE( b.n_rows == 5);
  REQUIRE( b.n_cols == 10);

  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (c >= 5)
        {
        REQUIRE( float(b(r, c)) == Approx(0.0) );
        }
      else
        {
        REQUIRE( float(b(r, c)) == Approx(1.0) );
        }
      }
    }
  }



TEST_CASE("resize_to_empty", "[resize]")
  {
  fmat a(5, 5);
  a.ones();

  fmat b = resize(a, 0, 0);

  REQUIRE( b.n_rows == 0 );
  REQUIRE( b.n_cols == 0 );
  }



TEST_CASE("resize_to_same_size", "[resize]")
  {
  fmat a(5, 5);
  a.ones();

  fmat b = resize(a, 5, 5);

  REQUIRE( b.n_rows == 5 );
  REQUIRE( b.n_cols == 5 );

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( float(b(r, c)) == float(a(r, c)) );
      }
    }
  }



TEST_CASE("resize_to_zero_rows", "[resize]")
  {
  fmat a(5, 5);
  a.ones();

  fmat b = resize(a, 0, 5);

  REQUIRE( b.n_rows == 0 );
  REQUIRE( b.n_cols == 5 );
  }



TEST_CASE("resize_to_zero_cols", "[resize]")
  {
  fmat a(5, 5);
  a.ones();

  fmat b = resize(a, 5, 0);

  REQUIRE( b.n_rows == 5 );
  REQUIRE( b.n_cols == 0 );
  }



TEST_CASE("resize_from_empty", "[resize]")
  {
  fmat a;

  fmat b = resize(a, 5, 5);

  REQUIRE( b.n_rows == 5 );
  REQUIRE( b.n_cols == 5 );

  for (uword i = 0; i < b.n_elem; ++i)
    {
    REQUIRE( float(b[i]) == Approx(0.0) );
    }
  }



TEST_CASE("resize_shrink", "[resize]")
  {
  fmat a(10, 10);
  a.ones();

  fmat b = resize(a, 5, 5);

  REQUIRE( b.n_rows == 5 );
  REQUIRE( b.n_cols == 5 );

  for (uword i = 0; i < b.n_elem; ++i)
    {
    REQUIRE( float(b[i]) == Approx(1.0) );
    }
  }



TEST_CASE("resize_inplace_same_size", "[resize]")
  {
  fmat a(5, 5);
  a.ones();
  a = resize(a, 5, 5);

  REQUIRE( a.n_rows == 5 );
  REQUIRE( a.n_cols == 5 );

  for (uword i = 0; i < a.n_elem; ++i)
    {
    REQUIRE( float(a[i]) == Approx(1.0) );
    }
  }



TEST_CASE("resize_inplace_shrink", "[resize]")
  {
  fmat a(10, 10);
  a.ones();
  a = resize(a, 5, 5);

  REQUIRE( a.n_rows == 5 );
  REQUIRE( a.n_cols == 5 );

  for (uword i = 0; i < a.n_elem; ++i)
    {
    REQUIRE( float(a[i]) == Approx(1.0) );
    }
  }



TEST_CASE("resize_inplace", "[resize]")
  {
  fmat a(5, 5);
  a.ones();
  a = resize(a, 10, 10);

  REQUIRE( a.n_rows == 10 );
  REQUIRE( a.n_cols == 10 );

  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      if (c >= 5 || r >= 5)
        {
        REQUIRE( float(a(r, c)) == Approx(0.0) );
        }
      else
        {
        REQUIRE( float(a(r, c)) == Approx(1.0) );
        }
      }
    }
  }



TEST_CASE("resize_member_onedim", "[resize]")
  {
  fmat a(5, 5);
  a.ones();
  a.resize(100);

  REQUIRE( a.n_rows == 100 );
  REQUIRE( a.n_cols == 1 );

  for (uword i = 0; i < 5; ++i)
    {
    REQUIRE( float(a[i]) == Approx(1.0) );
    }

  for (uword i = 6; i < a.n_elem; ++i)
    {
    REQUIRE( float(a[i]) == Approx(0.0) );
    }
  }



TEST_CASE("resize_member_twodim", "[resize]")
  {
  fmat a(5, 5);
  a.ones();
  a.resize(10, 10);

  REQUIRE( a.n_rows == 10 );
  REQUIRE( a.n_cols == 10 );

  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      if (c >= 5 || r >= 5)
        {
        REQUIRE( float(a(r, c)) == Approx(0.0) );
        }
      else
        {
        REQUIRE( float(a(r, c)) == Approx(1.0) );
        }
      }
    }
  }



TEST_CASE("resize_before_conv_to", "[resize]")
  {
  fmat a(5, 5);
  a.ones();

  imat b = conv_to<imat>::from(resize(a, 10, 10));

  REQUIRE( b.n_rows == 10 );
  REQUIRE( b.n_cols == 10 );

  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      if (c >= 5 || r >= 5)
        {
        REQUIRE( int(b(r, c)) == 0 );
        }
      else
        {
        REQUIRE( int(b(r, c)) == 1 );
        }
      }
    }
  }



TEST_CASE("resize_after_conv_to", "[resize]")
  {
  fmat a(5, 5);
  a.ones();

  imat b = resize(conv_to<imat>::from(a), 10, 10);

  REQUIRE( b.n_rows == 10 );
  REQUIRE( b.n_cols == 10 );

  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      if (c >= 5 || r >= 5)
        {
        REQUIRE( int(b(r, c)) == 0 );
        }
      else
        {
        REQUIRE( int(b(r, c)) == 1 );
        }
      }
    }
  }
