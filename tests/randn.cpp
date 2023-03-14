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

template<typename eT>
void test_randn()
  {
  Mat<eT> f = randn<Mat<eT>>(1000, 1000);
  arma::Mat<eT> f_cpu(f);
  for (uword r = 0; r < 1000; ++r)
    {
    for (uword c = 0; c < 1000; ++c)
      {
      REQUIRE( abs(eT(f_cpu(r, c))) <= eT(500) );
      }
    }
  }



TEST_CASE("randn_1", "[randn]")
{
  test_randn<float>();
  test_randn<double>();
  test_randn<s32>();
  test_randn<s64>();
}



// Use member .randn() function.
template<typename eT>
void test_randn_2()
  {
  Mat<eT> f(1000, 1000);
  f.randn();
  arma::Mat<eT> f_cpu(f);
  for (uword r = 0; r < 1000; ++r)
    {
    for (uword c = 0; c < 1000; ++c)
      {
      REQUIRE( abs(eT(f_cpu(r, c))) <= eT(500) );
      }
    }
  }



TEST_CASE("randn_2", "[randn]")
{
  test_randn_2<float>();
  test_randn_2<double>();
  test_randn_2<s32>();
  test_randn_2<s64>();
}



// Use member .randu() function and set size.
template<typename eT>
void test_randn_3()
  {
  Mat<eT> f(5, 5);
  f.randn(1000, 1000);
  REQUIRE( f.n_rows == 1000 );
  REQUIRE( f.n_cols == 1000 );
  arma::Mat<eT> f_cpu(f);

  for (uword r = 0; r < 1000; ++r)
    {
    for (uword c = 0; c < 1000; ++c)
      {
      REQUIRE( abs(eT(f_cpu(r, c))) <= eT(500) );
      }
    }
  }



TEST_CASE("randn_3", "[randn]")
{
  test_randn_3<float>();
  test_randn_3<double>();
  test_randn_3<s32>();
  test_randn_3<s64>();
}



// Test Row/Col randn().
template<typename eT>
void test_row_col_randn()
  {
  Row<eT> r1(1000);
  r1.randn();
  Row<eT> r2(10);
  r2.randn(1000);
  Row<eT> r3 = randn<Row<eT>>(1000);
  Row<eT> r4 = randn<Row<eT>>(1, 1000);

  Col<eT> c1(1000);
  c1.randn();
  Col<eT> c2(10);
  c2.randn(1000);
  Col<eT> c3 = randn<Col<eT>>(1000);
  Col<eT> c4 = randn<Col<eT>>(1000, 1);

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
    REQUIRE( abs(eT(r1_cpu[i])) <= eT(500) );
    REQUIRE( abs(eT(r2_cpu[i])) <= eT(500) );
    REQUIRE( abs(eT(r3_cpu[i])) <= eT(500) );
    REQUIRE( abs(eT(r4_cpu[i])) <= eT(500) );
    REQUIRE( abs(eT(c1_cpu[i])) <= eT(500) );
    REQUIRE( abs(eT(c2_cpu[i])) <= eT(500) );
    REQUIRE( abs(eT(c3_cpu[i])) <= eT(500) );
    REQUIRE( abs(eT(c4_cpu[i])) <= eT(500) );
    }
  }



TEST_CASE("randn_row_col", "[randn]")
  {
  test_row_col_randn<float>();
  test_row_col_randn<double>();
  test_row_col_randn<s32>();
  test_row_col_randn<s64>();
  }



inline double log_fact(const size_t n)
  {
  if (n == 0)
    {
    // 0! = 1; in logspace this is just 0.
    return 0.0;
    }
  else
    {
    // log(n!) = sum_{i = 2}^{n} log(n)
    // (skip i = 1 since log(1) = 0)
    double result = 0.0;
    for (size_t i = 2; i < n; ++i)
      {
      result += std::log(i);
      }

    return result;
    }
  }



// For floating-point types only.
//   mode 0: default parameters, no distr_param
//   mode 1: use distr_param with default values
//   mode 2: use distr_param with integer values
//   mode 3: use distr_param with double values
template<typename eT>
void test_randn_distr(const double mu = 0.0, const double sd = 1.0, const size_t mode = 0)
  {
  // Sample a large number of random values, then bin them, starting with a bin centered at the mean.
  // We use 10 bins per standard deviation, and ensure that the number of points observed in each bin,
  // with high probability, matches a Gaussian distribution with the specified parameters.
  // We check up to 3 standard deviations on each side, and we also check the tails on either side.

  Row<eT> f;
  switch (mode)
    {
    case 0:
      f = randn<Row<eT>>(50000);
      break;
    case 1:
      f = randn<Row<eT>>(50000, distr_param());
      break;
    case 2:
      f = randn<Row<eT>>(50000, distr_param((int) mu, (int) sd));
      break;
    case 3:
      f = randn<Row<eT>>(50000, distr_param(mu, sd));
      break;
    }

  arma::Row<size_t> bin_counts(60);
  bin_counts.zeros();
  size_t left_tail_count = 0;
  size_t right_tail_count = 0;
  arma::Row<eT> f_cpu(f);
  for (size_t i = 0; i < 50000; ++i)
    {
    const eT val = f_cpu[i];
    if (val >= mu - 3.0 * sd && val <= mu + 3.0 * sd)
      {
      const eT normalized_val = val - (mu - 3.0 * sd);
      const size_t bin_index = (normalized_val * 10) / sd;
      bin_counts[bin_index]++;
      }
    else if (val < mu - 3.0 * sd)
      {
      left_tail_count++;
      }
    else
      {
      right_tail_count++;
      }
    }

  // Bin i contains the count of points that fell into a range between [mu - (3 - 0.1 * i) * sd, mu - (3 - 0.1 * (i + 1)) * sd).
  // (For convenience call this range [a, b).)
  // The probability a single sample falls into this bin is (cdf(b) - cdf(a)).
  // As with the randu() test, this is a Bernoulli trial with p = (cdf(b) - cdf(a)).
  // This means that the number of points falling in the bin follows B(n, p) = B(50k, (cdf(b) - cdf(a))).
  // We want to compute the range such that with 0.9999999 probability, the samples we observed match what we would get from a correct Gaussian distribution.
  // Since we will be dealing with really large and really small numbers, we'll work in logspace.

  // First compute some convenience values we'll use later.
  arma::rowvec log_facts(50001); // log_facts[i] = log(i!)
  log_facts[0] = 0.0;
  log_facts[1] = 0.0;
  for (uword i = 2; i < 50001; ++i)
    {
    log_facts[i] = log_facts[i - 1] + std::log(i);
    }

  arma::rowvec log_combs(50001);
  for (uword i = 0; i < 50001; ++i)
    {
    // Fill with log(50k! / (i! * (50k - i)!))
    log_combs[i] = log_facts[50000] - (log_facts[i] + log_facts[50000 - i]);
    }

  for (size_t i = 0; i < 60; ++i)
    {
    const double p = 0.5 * (1 + std::erf(0.1 * (double(i) - 30 + 1) / std::sqrt(2))) - 0.5 * (1 + std::erf(0.1 * (double(i) - 30) / std::sqrt(2)));

    // The CDF cdf_b(x) for integer x of the distribution B(n, p) is sum_{i = 0}^{x} (n! / (i! (n - i)!)) * p^i * (1 - p)^(n - i).
    // We are searching for the largest x value such that cdf_b(x) < log(0.5e-9),
    // and also for the smallest x value such that cdf_b(x) > log(0.9999999).
    size_t x = 0;
    size_t lower = 50001, upper = 0;
    double cdf = 0.0;
    while (x <= 50000)
      {
      // Compute the PDF in log-space, but to add it to the CDF we have to convert out of logspace (due to the addition).
      const double log_comb = log_combs[x];
      const double log_prob = x * std::log(p) + (50000 - x) * std::log(1.0 - p);
      const double log_pdf = log_comb + log_prob;
      cdf += std::exp(log_pdf);
      if (cdf > 0.5e-9 && lower > 50000) // only set it once
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

    REQUIRE( bin_counts[i] > lower );
    REQUIRE( bin_counts[i] < upper );
    }

  // Now let's check the tails.  Here we will only have an upper bound on the number of points.
  // Note that the tails are symmetric, so we only need to do it once.
  const double p = 0.5 * (1 + std::erf(-3.0 / std::sqrt(2)));
  double cdf = 0.0;
  size_t x = 0;
  while (x <= 50000)
    {
    const double log_comb = log_combs[x];
    const double log_prob = x * std::log(p) + (50000 - x) * std::log(1.0 - p);
    const double log_pdf = log_comb + log_prob;
    cdf += std::exp(log_pdf);

    if (cdf > 0.9999999)
      {
      --x;
      break;
      }

    ++x;
    }

  REQUIRE( left_tail_count < x );
  REQUIRE( right_tail_count < x );
  }



template<typename eT>
void run_randn_distr_tests()
  {
  test_randn_distr<eT>(0.0, 1.0, 0);
  test_randn_distr<eT>(0.0, 1.0, 1);

  // Now try changing around the mean and standard deviation, with them taking integer values.
  for (int mu = -10; mu <= 10; mu += 5)
    {
    for (int sd = 1; sd < 5; sd += 1)
      {
      test_randn_distr<eT>(mu, sd, 2);
      }
    }

  // Now use floating-point values for the mean and standard deviations.
  for (uword i = 0; i < 5; ++i)
    {
    eT mu = pow(2.0, i);
    for (uword j = 0; j < 5; ++j)
      {
      eT sd = pow(2.0, j);
      test_randn_distr<eT>(mu, sd, 3);
      }
    }
  }



TEST_CASE("randn_distr", "[randn]")
  {
  run_randn_distr_tests<float>();
  run_randn_distr_tests<double>();
  }
