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

template<typename eT>
void test_randu()
  {
  Mat<eT> x(5, 5);

  x.randu();

  // Terrible test: make sure numbers are between 0 and 1 and are not all the same.
  bool all_same = true;
  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (eT(x(r, c)) != eT(x(0, 0)))
        all_same = false;

      REQUIRE( eT(x(r, c)) >= eT(0) );
      REQUIRE( eT(x(r, c)) <= eT(1) );
      }
    }

  REQUIRE( all_same == false );
  }



TEST_CASE("randu_1_old")
  {
//  test_randu<double>();
  test_randu<float>();
  }



template<typename eT>
void test_randn()
  {
  Mat<eT> x(5, 5);

  x.randn();

  // Terrible test: make sure the numbers are not all the same, and make sure the mean is in [-1, 1].
  eT sum = eT(0);
  bool all_same = true;
  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (eT(x(r, c)) != eT(x(0, 0)))
        all_same = false;

      sum += eT(x(r, c));
      }
    }

  REQUIRE( std::abs(sum / eT(25)) <= eT(1) );
  REQUIRE( all_same == false );
  }



TEST_CASE("randn_1")
  {
  test_randn<float>();
  test_randn<double>();
  }
