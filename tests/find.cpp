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

TEMPLATE_TEST_CASE("find_basic", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x(100);
  x.zeros();
  x(53) = eT(1);

  uvec y = find(x);

  REQUIRE( y.n_elem == 1 );
  REQUIRE( uword(y[0]) == uword(53) );
  }
