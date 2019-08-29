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

TEST_CASE("fill_1")
  {
  mat x(5, 5);

  x.fill(0.0);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( x(r, c) == Approx(0.0) );
      }
    }
  }



TEST_CASE("fill_2")
  {
  mat x(5, 5);

  x.fill(50.0);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( x(r, c) == Approx(50.0) );
      }
    }
  }



TEST_CASE("scalar_plus")
  {
  mat x(5, 5);
  x.fill(0.0);

  x += 1.5;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( x(r, c) == Approx(1.5) );
      }
    }
  }



TEST_CASE("scalar_minus")
  {
  mat x(5, 5);
  x.fill(0.0);

  x -= 1.5;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( x(r, c) == Approx(-1.5) );
      }
    }
  }



TEST_CASE("scalar_mul")
  {
  mat x(5, 5);
  x.fill(1.0);

  x *= 1.5;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( x(r, c) == Approx(1.5) );
      }
    }
  }



TEST_CASE("scalar_div")
  {
  mat x(5, 5);
  x.fill(1.0);

  x /= 2.0;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( x(r, c) == Approx(0.5) );
      }
    }
  }
