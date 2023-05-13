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

TEMPLATE_TEST_CASE("chol_1", "[chol]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(5, 5);
  x.zeros();
  for (uword i = 0; i < 5; ++i)
    x(i, i) = 1;

  Mat<eT> y;
  bool status = chol(y, x);

  REQUIRE(status == true);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (r > c)
        REQUIRE(eT(y(r, c)) == eT(0));
      else
        REQUIRE( eT(x(r, c)) == Approx(eT(y(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("chol_2", "[chol]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(5, 5);
  x.fill(eT(1));
  for (uword i = 0; i < 5; ++i)
    x(i, i) = eT(x(i, i)) + eT(1);

  Mat<eT> y;
  bool success = chol(y, x);

  REQUIRE(success == true);

  // Manually computed with GNU Octave.
  REQUIRE( eT(y(0, 0)) == Approx(eT(1.41421)).epsilon(0.01) );
  REQUIRE( eT(y(1, 0)) ==        eT(0.0    )                );
  REQUIRE( eT(y(2, 0)) ==        eT(0.0    )                );
  REQUIRE( eT(y(3, 0)) ==        eT(0.0    )                );
  REQUIRE( eT(y(4, 0)) ==        eT(0.0    )                );
  REQUIRE( eT(y(0, 1)) == Approx(eT(0.70711)).epsilon(0.01) );
  REQUIRE( eT(y(1, 1)) == Approx(eT(1.22474)).epsilon(0.01) );
  REQUIRE( eT(y(2, 1)) ==        eT(0.0    )                );
  REQUIRE( eT(y(3, 1)) ==        eT(0.0    )                );
  REQUIRE( eT(y(4, 1)) ==        eT(0.0    )                );
  REQUIRE( eT(y(0, 2)) == Approx(eT(0.70711)).epsilon(0.01) );
  REQUIRE( eT(y(1, 2)) == Approx(eT(0.40825)).epsilon(0.01) );
  REQUIRE( eT(y(2, 2)) == Approx(eT(1.15470)).epsilon(0.01) );
  REQUIRE( eT(y(3, 2)) ==        eT(0.0    )                );
  REQUIRE( eT(y(4, 2)) ==        eT(0.0    )                );
  REQUIRE( eT(y(0, 3)) == Approx(eT(0.70711)).epsilon(0.01) );
  REQUIRE( eT(y(1, 3)) == Approx(eT(0.40825)).epsilon(0.01) );
  REQUIRE( eT(y(2, 3)) == Approx(eT(0.28868)).epsilon(0.01) );
  REQUIRE( eT(y(3, 3)) == Approx(eT(1.11803)).epsilon(0.01) );
  REQUIRE( eT(y(4, 3)) ==        eT(0.0    )                );
  REQUIRE( eT(y(0, 4)) == Approx(eT(0.70711)).epsilon(0.01) );
  REQUIRE( eT(y(1, 4)) == Approx(eT(0.40825)).epsilon(0.01) );
  REQUIRE( eT(y(2, 4)) == Approx(eT(0.28868)).epsilon(0.01) );
  REQUIRE( eT(y(3, 4)) == Approx(eT(0.22361)).epsilon(0.01) );
  REQUIRE( eT(y(4, 4)) == Approx(eT(1.09545)).epsilon(0.01) );
  }



TEMPLATE_TEST_CASE("chol_3", "[chol]", float, double)
  {
  typedef TestType eT;

  Mat<eT> x(50, 50);
  x.randu();
  // force symmetry
  x *= x.t();

  for (uword i = 0; i < 50; ++i)
    x(i, i) = eT(x(i, i)) + eT(3); // force positive definiteness

  Mat<eT> y;
  bool success = chol(y, x);

  REQUIRE(success == true);

  // Check that the lower diagonal is zeros.
  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = c + 1; r < 50; ++r)
      {
      REQUIRE( eT(y(r, c)) == eT(0) );
      }
    }

  // Now check that we can recompute the original matrix.
  Mat<eT> z = y.t() * y;

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(x(r, c))) );
      }
    }
  }
