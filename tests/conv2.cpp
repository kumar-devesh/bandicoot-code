// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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

// Simple hardcoded test
TEMPLATE_TEST_CASE("hardcoded_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(4, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);
  A(3, 0) = eT(13);
  A(3, 1) = eT(14);
  A(3, 2) = eT(15);
  A(3, 3) = eT(16);

  Mat<eT> B(3, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);
  B(2, 0) = eT(16);
  B(2, 1) = eT(17);
  B(2, 2) = eT(18);

  Mat<eT> C = conv2(A, B);

  REQUIRE( C.n_rows == 6 );
  REQUIRE( C.n_cols == 6 );

  REQUIRE( eT(C(0, 0)) == Approx(eT(  10)) );
  REQUIRE( eT(C(0, 1)) == Approx(eT(  31)) );
  REQUIRE( eT(C(0, 2)) == Approx(eT(  64)) );
  REQUIRE( eT(C(0, 3)) == Approx(eT(  97)) );
  REQUIRE( eT(C(0, 4)) == Approx(eT(  80)) );
  REQUIRE( eT(C(0, 5)) == Approx(eT(  48)) );
  REQUIRE( eT(C(1, 0)) == Approx(eT(  63)) );
  REQUIRE( eT(C(1, 1)) == Approx(eT( 155)) );
  REQUIRE( eT(C(1, 2)) == Approx(eT( 278)) );
  REQUIRE( eT(C(1, 3)) == Approx(eT( 353)) );
  REQUIRE( eT(C(1, 4)) == Approx(eT( 273)) );
  REQUIRE( eT(C(1, 5)) == Approx(eT( 156)) );
  REQUIRE( eT(C(2, 0)) == Approx(eT( 171)) );
  REQUIRE( eT(C(2, 1)) == Approx(eT( 396)) );
  REQUIRE( eT(C(2, 2)) == Approx(eT( 678)) );
  REQUIRE( eT(C(2, 3)) == Approx(eT( 804)) );
  REQUIRE( eT(C(2, 4)) == Approx(eT( 603)) );
  REQUIRE( eT(C(2, 5)) == Approx(eT( 336)) );
  REQUIRE( eT(C(3, 0)) == Approx(eT( 327)) );
  REQUIRE( eT(C(3, 1)) == Approx(eT( 720)) );
  REQUIRE( eT(C(3, 2)) == Approx(eT(1182)) );
  REQUIRE( eT(C(3, 3)) == Approx(eT(1308)) );
  REQUIRE( eT(C(3, 4)) == Approx(eT( 951)) );
  REQUIRE( eT(C(3, 5)) == Approx(eT( 516)) );
  REQUIRE( eT(C(4, 0)) == Approx(eT( 313)) );
  REQUIRE( eT(C(4, 1)) == Approx(eT( 677)) );
  REQUIRE( eT(C(4, 2)) == Approx(eT(1094)) );
  REQUIRE( eT(C(4, 3)) == Approx(eT(1187)) );
  REQUIRE( eT(C(4, 4)) == Approx(eT( 851)) );
  REQUIRE( eT(C(4, 5)) == Approx(eT( 456)) );
  REQUIRE( eT(C(5, 0)) == Approx(eT( 208)) );
  REQUIRE( eT(C(5, 1)) == Approx(eT( 445)) );
  REQUIRE( eT(C(5, 2)) == Approx(eT( 712)) );
  REQUIRE( eT(C(5, 3)) == Approx(eT( 763)) );
  REQUIRE( eT(C(5, 4)) == Approx(eT( 542)) );
  REQUIRE( eT(C(5, 5)) == Approx(eT( 288)) );
  }
