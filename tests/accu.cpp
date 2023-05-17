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

TEMPLATE_TEST_CASE("accu_small", "[accu]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(4, 4);
  for (uword i = 0; i < 16; ++i)
    x[i] = i + 1;

  eT sum = accu(x);

  REQUIRE(sum == Approx(eT(136)) );
  }



TEMPLATE_TEST_CASE("accu_1", "[accu]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(80, 80);
  for (uword i = 0; i < 6400; ++i)
    x[i] = i + 1;

  eT sum = accu(x);

  REQUIRE(sum == Approx(eT(20483200)) );
  }



TEMPLATE_TEST_CASE("accu_strange_size", "[accu]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(608);

  for(uword i = 0; i < 608; ++i)
    x[i] = i + 1;

  eT sum = accu(x);

  REQUIRE(sum == Approx(eT(185136)));
  }



TEMPLATE_TEST_CASE("accu_large", "[accu]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Col<eT> cpu_x = arma::conv_to<arma::Col<eT>>::from(arma::randu<arma::Col<double>>(100000) * 10.0);
  cpu_x.randu();
  Col<eT> x(cpu_x);

  eT cpu_sum = accu(cpu_x);
  eT sum = accu(x);

  REQUIRE(sum == Approx(cpu_sum));
  }



TEMPLATE_TEST_CASE("accu_2", "[accu]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 5);
  x.randu();
  x += eT(1);

  eT sum = accu(x);

  REQUIRE( sum >= eT(50) );
  REQUIRE( sum <= eT(100) );
  }



TEMPLATE_TEST_CASE("accu_subview_1", "[accu]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  for (uword i = 0; i < 25; ++i)
    x[i] = i + 1;

  eT sum1 = accu(x.cols(1, 2));
  eT sum2 = accu(x.rows(3, 4));
  eT sum3 = accu(x.submat(1, 1, 3, 3));
  eT sum4 = accu(x.submat(1, 1, 1, 1));

  REQUIRE( sum1 == Approx(eT(105)) );
  REQUIRE( sum2 == Approx(eT(145)) );
  REQUIRE( sum3 == Approx(eT(117)) );
  REQUIRE( sum4 == Approx(eT(7)) );
  }
