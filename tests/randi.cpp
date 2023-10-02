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

#include <armadillo>
#include <bandicoot>
#include "catch.hpp"

using namespace coot;

TEMPLATE_TEST_CASE("randi_1", "[randi]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> f = randi<Mat<eT>>(1000, 1000);
  arma::Mat<eT> f_cpu(f);

  for (uword c = 0; c < 1000; ++c)
    {
    for (uword r = 0; r < 1000; ++r)
      {
      REQUIRE( eT(f_cpu(r, c)) >= eT(0) );
      REQUIRE( eT(f_cpu(r, c)) <= eT(std::numeric_limits<int>::max()) );
      }
    }
  }



template<typename eT>
void test_randi_range(int lo, int hi)
  {
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> f = randi<Mat<eT>>(1000, 1000, distr_param(lo, hi));
  arma::Mat<eT> f_cpu(f);

  for (uword c = 0; c < 1000; ++c)
    {
    for (uword r = 0; r < 1000; ++r)
      {
      REQUIRE( eT(f_cpu(r, c)) >= eT(lo) );
      REQUIRE( eT(f_cpu(r, c)) <= eT(hi) );
      }
    }
  }



TEMPLATE_TEST_CASE("randi_range_1", "[randi]", float, double, u32, s32, u64, s64)
  {
  test_randi_range<TestType>(0, 50);
  }



TEMPLATE_TEST_CASE("randi_range_2", "[randi]", float, double, s32, s64)
  {
  test_randi_range<TestType>(-75, 12);
  }



// Test Row/Col randi().
template<typename eT>
void test_row_col_randi(int lo, int hi)
  {
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Row<eT> r1 = randi<Row<eT>>(1000, distr_param(lo, hi));
  Row<eT> r2 = randi<Row<eT>>(1, 1000, distr_param(lo, hi));

  Col<eT> c1 = randi<Col<eT>>(1000, distr_param(lo, hi));
  Col<eT> c2 = randi<Col<eT>>(1000, 1, distr_param(lo, hi));

  REQUIRE( r1.n_elem == 1000 );
  REQUIRE( r2.n_elem == 1000 );
  REQUIRE( c1.n_elem == 1000 );
  REQUIRE( c2.n_elem == 1000 );

  arma::Row<eT> r1_cpu(r1);
  arma::Row<eT> r2_cpu(r2);
  arma::Col<eT> c1_cpu(c1);
  arma::Col<eT> c2_cpu(c2);

  for (uword i = 0; i < 1000; ++i)
    {
    REQUIRE( eT(r1_cpu[i]) >= eT(lo) );
    REQUIRE( eT(r1_cpu[i]) <= eT(hi) );
    REQUIRE( eT(r2_cpu[i]) >= eT(lo) );
    REQUIRE( eT(r2_cpu[i]) <= eT(hi) );
    REQUIRE( eT(c1_cpu[i]) >= eT(lo) );
    REQUIRE( eT(c1_cpu[i]) <= eT(hi) );
    REQUIRE( eT(c2_cpu[i]) >= eT(lo) );
    REQUIRE( eT(c2_cpu[i]) <= eT(hi) );
    }
  }



TEMPLATE_TEST_CASE("randi_row_col_1", "[randi]", float, double, u32, s32, u64, s64)
  {
  test_row_col_randi<TestType>(0, 500);
  }



TEMPLATE_TEST_CASE("randi_row_col_2", "[randi]", float, double, s32, s64)
  {
  test_row_col_randi<TestType>(-125, 500);
  }



template<typename eT>
void test_randi_distr(int lo, int hi)
  {
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  // Sample a large number of random values.
  // They are already binned, so the empirically observed probability of each value is (1 / (hi - lo)).

  Row<eT> f = randi<Row<eT>>(10000, distr_param(lo, hi));
  arma::Row<size_t> bin_counts(hi - lo + 1);
  bin_counts.zeros();
  arma::Row<eT> f_cpu(f);
  for (uword i = 0; i < 10000; ++i)
    {
    const eT val = f_cpu[i];
    bin_counts[val - lo]++;
    }

  // Each bin contains the sum of samples of n = 10k Bernoulli trials with p = (1 / (hi - lo)).
  // These bounds are computed in the same way as the randn() tests.

  arma::rowvec log_facts(10001);
  log_facts[0] = 0.0;
  log_facts[1] = 0.0;
  for (uword i = 2; i < 10001; ++i)
    {
    log_facts[i] = log_facts[i - 1] + std::log(i);
    }

  arma::rowvec log_combs(10001);
  for (uword i = 0; i < 10001; ++i)
    {
    // Fill with log(10k! / (i! * (10k - i)!))
    log_combs[i] = log_facts[10000] - (log_facts[i] + log_facts[10000 - i]);
    }

  size_t x = 0;
  size_t lower = 10001, upper = 0;
  double cdf = 0.0;
  const double p = 1.0 / double(hi - lo);
  while (x <= 10000)
    {
    const double log_comb = log_combs[x];
    const double log_prob = x * std::log(p) + (10000 - x) * std::log(1.0 - p);
    const double log_pdf = log_comb + log_prob;
    cdf += std::exp(log_pdf);
    if (cdf > 0.5e-9 && lower > 10000) // only set it once
      {
      lower = x - 1;
      }
    else if (cdf > 0.9999999)
      {
      upper = x;
      break;
      }

    ++x;
    }

  for (size_t i = 0; i < bin_counts.n_elem; ++i)
    {
    REQUIRE( bin_counts[i] > lower );
    REQUIRE( bin_counts[i] < upper );
    }
  }



TEMPLATE_TEST_CASE("randi_distr_1", "[randi]", float, double, u32, s32, u64, s64)
  {
  test_randi_distr<TestType>(0, 67);
  }



TEMPLATE_TEST_CASE("randi_distr_2", "[randi]", float, double, s32, s64)
  {
  test_randi_distr<TestType>(-11, 17);
  }
