// Copyright 2021 Marcus Edel (http://kurg.org/)
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

TEMPLATE_TEST_CASE("linspace_1", "[linspace]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> a = linspace<Col<eT>>(1,5,5);

  REQUIRE(eT(a(0)) == Approx(eT(1.0)));
  REQUIRE(eT(a(1)) == Approx(eT(2.0)));
  REQUIRE(eT(a(2)) == Approx(eT(3.0)));
  REQUIRE(eT(a(3)) == Approx(eT(4.0)));
  REQUIRE(eT(a(4)) == Approx(eT(5.0)));

  Col<eT> b = linspace<Col<eT>>(1,5,6);

  REQUIRE(eT(b(0)) == Approx(eT(1.0)));
  REQUIRE(eT(b(1)) == Approx(eT(1.8)));
  REQUIRE(eT(b(2)) == Approx(eT(2.6)));
  REQUIRE(eT(b(3)) == Approx(eT(3.4)));
  REQUIRE(eT(b(4)) == Approx(eT(4.2)));
  REQUIRE(eT(b(5)) == Approx(eT(5.0)));

  Row<eT> c = linspace<Row<eT>>(1,5,6);

  REQUIRE(eT(c(0)) == Approx(eT(1.0)));
  REQUIRE(eT(c(1)) == Approx(eT(1.8)));
  REQUIRE(eT(c(2)) == Approx(eT(2.6)));
  REQUIRE(eT(c(3)) == Approx(eT(3.4)));
  REQUIRE(eT(c(4)) == Approx(eT(4.2)));
  REQUIRE(eT(c(5)) == Approx(eT(5.0)));

  Mat<eT> X = linspace<Mat<eT>>(1,5,6);

  REQUIRE(X.n_rows == 6);
  REQUIRE(X.n_cols == 1);
  }



TEMPLATE_TEST_CASE("linspace_2", "[linspace]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> a = linspace<Col<eT>>(1,5,5);

  REQUIRE(eT(a(0)) == Approx(eT(1.0)));
  REQUIRE(eT(a(1)) == Approx(eT(2.0)));
  REQUIRE(eT(a(2)) == Approx(eT(3.0)));
  REQUIRE(eT(a(3)) == Approx(eT(4.0)));
  REQUIRE(eT(a(4)) == Approx(eT(5.0)));

  Mat<eT> X = linspace<Mat<eT>>(1,5,6);

  REQUIRE(X.n_rows == 6);
  REQUIRE(X.n_cols == 1);
  }
