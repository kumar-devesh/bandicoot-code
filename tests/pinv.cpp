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

// Trivial test cases.

TEMPLATE_TEST_CASE("identity_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x;
  x.eye(3, 3);

  Mat<eT> out = pinv(x);

  REQUIRE( out.n_rows == 3 );
  REQUIRE( out.n_cols == 3 );
  REQUIRE( eT(out(0, 0)) == Approx(1.0) );
  REQUIRE( eT(out(0, 1)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(0, 2)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(1, 0)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(1, 1)) == Approx(1.0) );
  REQUIRE( eT(out(1, 2)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(2, 0)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(2, 1)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(2, 2)) == Approx(1.0) );
  }



TEMPLATE_TEST_CASE("trivial_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(3, 3);
  // x: [[1, 5, 3],
  //     [9, 6, 2],
  //     [4, 10, 1]]
  x(0, 0) = (eT) 1;
  x(0, 1) = (eT) 5;
  x(0, 2) = (eT) 3;
  x(1, 0) = (eT) 9;
  x(1, 1) = (eT) 6;
  x(1, 2) = (eT) 2;
  x(2, 0) = (eT) 4;
  x(2, 1) = (eT) 10;
  x(2, 2) = (eT) 1;

  Mat<eT> out = pinv(x);

  REQUIRE( out.n_rows == 3 );
  REQUIRE( out.n_cols == 3 );
  // Results computed by Julia.
  REQUIRE( eT(out(0, 0)) == Approx(-0.0782123) );
  REQUIRE( eT(out(0, 1)) == Approx( 0.139665) );
  REQUIRE( eT(out(0, 2)) == Approx(-0.0446927) );
  REQUIRE( eT(out(1, 0)) == Approx(-0.00558659) );
  REQUIRE( eT(out(1, 1)) == Approx(-0.0614525) );
  REQUIRE( eT(out(1, 2)) == Approx( 0.139665) );
  REQUIRE( eT(out(2, 0)) == Approx( 0.368715) );
  REQUIRE( eT(out(2, 1)) == Approx( 0.0558659) );
  REQUIRE( eT(out(2, 2)) == Approx(-0.217877) );
  }

// Diagonal matrix tests:
//  - as vector
//  - as vector with NaNs
//  - as vector with custom tolerance
//  - as vector with tolerance too large
//  - as vector, empty
//  - as vector with trans
//  - as vector with scalar mul
//  - as vector with scalar mul and trans
//  - as vector, output alias
//
//  - as matrix
//  - as matrix with NaNs
//  - as matrix with lots of other nonzero nondiagonal elements
//  - as matrix with custom tolerance
//  - as matrix with tolerance too larger
//  - as matrix, empty
//  - as matrix with trans
//  - as matrix with scalar mul
//  - as matrix with scalar mul and trans
//  - as matrix, output alias

// Symmetric matrix tests, to be added later:
//
//  - empty
//  - random, symmetric, but also with op_symmat
//  - random with op_symmat
//  - random with op_symmat and transpose
//  - random with op_symmat and transpose and scalar mul'
//  - random with NaNs
//  - random with custom tolerance
//  - random with tolerance too large
//  - alias

// General matrix tests:
//
//  - empty
//  - random
//  - random operation
//  - random with NaNs
//  - random with custom tolerance
//  - random with tolerance too large
//  - nonsquare, rows > cols
//  - nonsquare, rows < cols
//  - alias
