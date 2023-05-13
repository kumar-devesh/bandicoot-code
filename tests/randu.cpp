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

TEMPLATE_TEST_CASE("randu_1", "[randu]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> f = randu<Mat<eT>>(1000, 1000);
  arma::Mat<eT> f_cpu(f);
  for (uword c = 0; c < 1000; ++c)
    {
    for (uword r = 0; r < 1000; ++r)
      {
      REQUIRE( eT(f_cpu(r, c)) >= eT(0) );
      REQUIRE( eT(f_cpu(r, c)) <= eT(1) );
      }
    }
  }



// Use member .randu() function.
TEMPLATE_TEST_CASE("randu_2", "[randu]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> f(1000, 1000);
  f.randu();
  arma::Mat<eT> f_cpu(f);
  for (uword c = 0; c < 1000; ++c)
    {
    for (uword r = 0; r < 1000; ++r)
      {
      REQUIRE( eT(f_cpu(r, c)) >= eT(0) );
      REQUIRE( eT(f_cpu(r, c)) <= eT(1) );
      }
    }
  }



// Use member .randu() function and set size.
TEMPLATE_TEST_CASE("randu_3", "[randu]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> f(5, 5);
  f.randu(1000, 1000);
  REQUIRE( f.n_rows == 1000 );
  REQUIRE( f.n_cols == 1000 );
  arma::Mat<eT> f_cpu(f);

  for (uword r = 0; r < 1000; ++r)
    {
    for (uword c = 0; c < 1000; ++c)
      {
      REQUIRE( eT(f_cpu(r, c)) >= eT(0) );
      REQUIRE( eT(f_cpu(r, c)) <= eT(1) );
      }
    }
  }



// Test Row/Col randu().
TEMPLATE_TEST_CASE("randu_row_col", "[randu]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Row<eT> r1(1000);
  r1.randu();
  Row<eT> r2(10);
  r2.randu(1000);
  Row<eT> r3 = randu<Row<eT>>(1000);
  Row<eT> r4 = randu<Row<eT>>(1, 1000);

  Col<eT> c1(1000);
  c1.randu();
  Col<eT> c2(10);
  c2.randu(1000);
  Col<eT> c3 = randu<Col<eT>>(1000);
  Col<eT> c4 = randu<Col<eT>>(1000, 1);

  REQUIRE( r1.n_elem == 1000 );
  REQUIRE( r2.n_elem == 1000 );
  REQUIRE( r3.n_elem == 1000 );
  REQUIRE( r4.n_elem == 1000 );
  REQUIRE( c1.n_elem == 1000 );
  REQUIRE( c2.n_elem == 1000 );
  REQUIRE( c3.n_elem == 1000 );
  REQUIRE( c4.n_elem == 1000 );

  arma::Row<eT> r1_cpu(r1);
  arma::Row<eT> r2_cpu(r2);
  arma::Row<eT> r3_cpu(r3);
  arma::Row<eT> r4_cpu(r4);
  arma::Col<eT> c1_cpu(c1);
  arma::Col<eT> c2_cpu(c2);
  arma::Col<eT> c3_cpu(c3);
  arma::Col<eT> c4_cpu(c4);

  for (uword i = 0; i < 1000; ++i)
    {
    REQUIRE( eT(r1_cpu[i]) >= eT(0) );
    REQUIRE( eT(r1_cpu[i]) <= eT(1) );
    REQUIRE( eT(r2_cpu[i]) >= eT(0) );
    REQUIRE( eT(r2_cpu[i]) <= eT(1) );
    REQUIRE( eT(r3_cpu[i]) >= eT(0) );
    REQUIRE( eT(r3_cpu[i]) <= eT(1) );
    REQUIRE( eT(r4_cpu[i]) >= eT(0) );
    REQUIRE( eT(r4_cpu[i]) <= eT(1) );
    REQUIRE( eT(c1_cpu[i]) >= eT(0) );
    REQUIRE( eT(c1_cpu[i]) <= eT(1) );
    REQUIRE( eT(c2_cpu[i]) >= eT(0) );
    REQUIRE( eT(c2_cpu[i]) <= eT(1) );
    REQUIRE( eT(c3_cpu[i]) >= eT(0) );
    REQUIRE( eT(c3_cpu[i]) <= eT(1) );
    REQUIRE( eT(c4_cpu[i]) >= eT(0) );
    REQUIRE( eT(c4_cpu[i]) <= eT(1) );
    }
  }



// For floating-point types only.
TEMPLATE_TEST_CASE("randu_distr", "[randu]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  // Sample a large number of random values, then bin them into 5 bins.
  // The empirically observed probability of each bin should match 0.2, plus or minus some variance (calculation detailed below).

  Row<eT> f = randu<Row<eT>>(10000);
  arma::Row<size_t> bin_counts(5);
  bin_counts.zeros();
  arma::Row<eT> f_cpu(f);
  for (size_t i = 0; i < 10000; ++i)
    {
    const eT val = f_cpu[i];
    if (val >= 0.8)
      ++bin_counts[4];
    else if (val >= 0.6)
      ++bin_counts[3];
    else if (val >= 0.4)
      ++bin_counts[2];
    else if (val >= 0.2)
      ++bin_counts[1];
    else
      ++bin_counts[0];
    }

  // Each bin contains the sum of samples of n = 10k Bernoulli trials with p = 0.2.
  // So, their sum is ~ B(n, p) = B(10k, 0.2).
  // If randu() produced samples that are actually uniformly randomly distributed,
  // then with 0.9999999 probability, the sum will lie in the range [1795, 2210].
  // (Note that the binomial distribution is not symmetric for p != 0.5, like in our case.)
  REQUIRE( bin_counts[0] >= 1795 );
  REQUIRE( bin_counts[0] <= 2210 );
  REQUIRE( bin_counts[1] >= 1795 );
  REQUIRE( bin_counts[1] <= 2210 );
  REQUIRE( bin_counts[2] >= 1795 );
  REQUIRE( bin_counts[2] <= 2210 );
  REQUIRE( bin_counts[3] >= 1795 );
  REQUIRE( bin_counts[3] <= 2210 );
  REQUIRE( bin_counts[4] >= 1795 );
  REQUIRE( bin_counts[4] <= 2210 );
  }
