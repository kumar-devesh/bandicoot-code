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

TEMPLATE_TEST_CASE("fill_zeros", "[fill_class]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;
  
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }
  
  Mat<eT> X(123, 456, fill::zeros);
  
  for (uword c = 0; c < X.n_cols; ++c)
  for (uword r = 0; r < X.n_rows; ++r)
    {
    REQUIRE( eT(X(r, c)) == eT(0) );
    }
  }



TEMPLATE_TEST_CASE("fill_ones", "[fill_class]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;
  
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }
  
  Mat<eT> X(123, 456, fill::ones);
  
  for (uword c = 0; c < X.n_cols; ++c)
  for (uword r = 0; r < X.n_rows; ++r)
    {
    REQUIRE( eT(X(r, c)) == eT(1) );
    }
  }



TEMPLATE_TEST_CASE("fill_randu", "[fill_class]", float, double)
  {
  typedef TestType eT;
  
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }
  
  Mat<eT> X(1000, 1000, fill::randu);
  
  const eT mean_value = accu(X) / eT(X.n_elem);
  
  REQUIRE( mean_value == Approx(eT(0.5)).margin(1e-1) );
  }



TEMPLATE_TEST_CASE("fill_randn", "[fill_class]", float, double)
  {
  typedef TestType eT;
  
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }
  
  Mat<eT> X(1000, 1000, fill::randn);
  
  const eT mean_value = accu(X) / eT(X.n_elem);
  
  REQUIRE( mean_value == Approx(eT(0.0)).margin(1e-1) );
  }



TEMPLATE_TEST_CASE("fill_instantiation", "[fill_class]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;
  
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }
  
  Mat<eT> A(123, 456, fill::none);
  Mat<eT> B(123, 456, fill::zeros);
  Mat<eT> C(123, 456, fill::ones);
  Mat<eT> D(123, 456, fill::eye);
  Mat<eT> E(123, 456, fill::randu);
  Mat<eT> F(123, 456, fill::randn);
  
  REQUIRE( A.n_elem > 0 );
  REQUIRE( B.n_elem > 0 );
  REQUIRE( C.n_elem > 0 );
  REQUIRE( D.n_elem > 0 );
  REQUIRE( E.n_elem > 0 );
  REQUIRE( F.n_elem > 0 );
  }
