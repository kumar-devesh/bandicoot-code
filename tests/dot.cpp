// Copyright 2020 Ryan Curtin (http://www.ratml.org/)
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

TEMPLATE_TEST_CASE("dot_1", "[dot]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Row<eT> x(10);
  Row<eT> y(10);
  for (uword i = 0; i < 10; ++i)
    {
    x[i] = i + 1;
    y[i] = 10 - i;
    }

  eT d = dot(x, y);

  REQUIRE(d == Approx(eT(220)) );
  }



TEMPLATE_TEST_CASE("dot_2", "[dot]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(1000);
  Row<eT> y(1000);
  x.randu();
  y.randu();

  eT d = dot(x, y);

  eT manual_dot = eT(0);
  for (uword i = 0; i < 1000; ++i)
    {
    manual_dot += eT(x[i]) * eT(y[i]);
    }

  REQUIRE( d == Approx(manual_dot) );
  }



TEMPLATE_TEST_CASE("mat_dot", "[dot]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  Mat<eT> y(10, 10);

  x.randu();
  y.randu();

  eT d = dot(x, y);

  eT manual_dot = eT(0);
  for (uword i = 0; i < 100; ++i)
    {
    manual_dot += eT(x[i]) * eT(y[i]);
    }

  REQUIRE( d == Approx(manual_dot) );
  }



TEMPLATE_TEST_CASE("expr_dot", "[dot]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(10);
  Col<eT> y(10);

  for (uword i = 0; i < 10; ++i)
    {
    x[i] = i + 1;
    y[i] = 10 - i;
    }

  eT d = dot(x % y, (y + eT(2)));

  eT manual_dot = eT(0);
  for (uword i = 0; i < 10; ++i)
    {
    manual_dot += eT(x[i]) * eT(y[i]) * eT(y[i] + eT(2));
    }

  REQUIRE( d == Approx(manual_dot) );
  }



TEMPLATE_TEST_CASE
  (
  "different_eT_dot",
  "[dot]",
  (std::pair<double, double>), (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, float>), (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, u32>), (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, s32>), (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, u64>), (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, s64>), (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>)
  )
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Col<eT1> x(10);
  Col<eT2> y(10);

  for (uword i = 0; i < 10; ++i)
    {
    x[i] = i + 1;
    y[i] = 10 - i;
    }

  typedef typename promote_type<eT1, eT2>::result promoted_eT;
  promoted_eT result = dot(x, y);

  REQUIRE(result == Approx(promoted_eT(220)));
  }



// Make sure that dot() returns the expected results when one type is signed
// and the other is unsigned.
TEMPLATE_TEST_CASE
  (
  "signed_unsigned_dot",
  "[dot]",
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, s32>), (std::pair<u64, s64>)
  )
  {
  typedef typename TestType::first_type ueT1;
  typedef typename TestType::second_type seT2;

  if (!coot_rt_t::is_supported_type<ueT1>() || !coot_rt_t::is_supported_type<seT2>())
    {
    return;
    }

  Col<ueT1> x(10);
  Col<seT2> y(10);

  for (uword i = 0; i < 10; ++i)
    {
    x[i] = i + 1;
    y[i] = -(seT2(i) + 1);
    }

  // This should type-promote to seT1 or similar.
  typedef typename promote_type<ueT1, seT2>::result out_eT;
  out_eT result = dot(x, y);

  REQUIRE(result == Approx(out_eT(-385)));
  }
