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

TEMPLATE_TEST_CASE("fill_1", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);

  x.fill(eT(0));

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(0)) );
      }
    }
  }



TEMPLATE_TEST_CASE("fill_2", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);

  x.fill(eT(50));

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(50)) );
      }
    }
  }



TEMPLATE_TEST_CASE("scalar_plus", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(0));

  x += eT(1);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(1)) );
      }
    }
  }



TEMPLATE_TEST_CASE("scalar_minus", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));

  x -= eT(5);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
      }
    }
  }



TEMPLATE_TEST_CASE("scalar_mul", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(1));

  x *= eT(10);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
      }
    }
  }



TEMPLATE_TEST_CASE("scalar_div", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));

  x /= eT(2);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
      }
    }
  }



TEMPLATE_TEST_CASE("submat_scalar_fill", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));

  x.submat(1, 1, 3, 3).fill(eT(5));

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (c >= 1 && c <= 3 && r >= 1 && r <= 3)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("submat_scalar_add", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));

  x.submat(1, 1, 3, 3) += eT(5);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (c >= 1 && c <= 3 && r >= 1 && r <= 3)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(15)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("submat_scalar_minus", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));

  x.submat(1, 1, 3, 3) -= eT(5);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (c >= 1 && c <= 3 && r >= 1 && r <= 3)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("submat_scalar_mul", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));

  x.submat(1, 1, 3, 3) *= eT(5);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (c >= 1 && c <= 3 && r >= 1 && r <= 3)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(50)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("submat_scalar_div", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));

  x.submat(1, 1, 3, 3) /= eT(5);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (c >= 1 && c <= 3 && r >= 1 && r <= 3)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(2)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
        }
      }
    }
  }



/* (this one takes a long time; best to leave it commented out)
template<typename eT>
void test_huge_submat_fill()
  {
  Mat<eT> x(50000, 10);
  x.fill(eT(10));

  x.submat(1, 1, 49999, 3).fill(eT(5));

  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 50000; ++r)
      {
      if (c >= 1 && c <= 3 && r >= 1 && r <= 49999)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
        }
      }
    }
  }



TEST_CASE("submat_huge_scalar_fill")
  {
  test_huge_submat_fill<float>();
  test_huge_submat_fill<double>();
  test_huge_submat_fill<u32>();
  test_huge_submat_fill<s32>();
  test_huge_submat_fill<u64>();
  test_huge_submat_fill<s64>();
  }
*/



TEMPLATE_TEST_CASE
  (
  "eop_scalar_plus",
  "[eop_scalar]",
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

  Mat<eT1> x(5, 5);
  x.fill(eT1(3));

  Mat<eT2> y = conv_to<Mat<eT2>>::from(x) + eT2(1);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT2(y(r, c)) == Approx(eT2(eT1(x(r, c)) + eT2(1))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_neg", "[eop_scalar]", double, float, s32, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = -x;

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(-eT(x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_scalar_minus_pre", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = eT(5) - x;

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(5) - eT(x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_scalar_minus_post", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = x - eT(1);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(2)) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_scalar_times", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = eT(2) * x;

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx( eT(2) * eT(x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_scalar_div_pre", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = eT(9) / x;

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(9) / eT(x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_scalar_div_post", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(4));

  Mat<eT> y = x / eT(2);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(x(r, c)) / eT(2)) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_square", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = square(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(x(r, c)) * eT(x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_sqrt", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(9));

  Mat<eT> y = sqrt(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(x(r, c)) == Approx( eT(y(r, c)) * eT(y(r, c)) ) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_log", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));

  Mat<eT> y = log(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(std::log(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_exp", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = exp(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(std::exp(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_abs", "[eop_scalar]", double, float, s32, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(-3));

  Mat<eT> y = abs(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(std::abs(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_abs_2", "[eop_scalar]", u32, u64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(5));

  Mat<eT> y = abs(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_log2", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = log2(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(std::log2(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_log10", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = log10(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(std::log10(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_trunc_log", "[eop_scalar]", double, float, s32, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));
  x[0] = std::numeric_limits<eT>::infinity();
  x[1] = eT(0);
  x[2] = eT(-1);

  Mat<eT> y = trunc_log(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(arma::trunc_log(eT(x(r, c))))) );
      }
    }
  }



// We can't test anything that will return a negative value for unsigned types.
TEMPLATE_TEST_CASE("eop_trunc_log_pos", "[eop_scalar]", u32, u64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));
  x[0] = std::numeric_limits<eT>::infinity() + 1;
  x[1] = eT(1);
  x[2] = eT(2);

  Mat<eT> y = trunc_log(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(arma::trunc_log(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_exp2", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = exp2(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(exp2(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_exp10", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = exp10(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(std::pow(10.0, eT(x(r, c))))).epsilon(0.01) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_trunc_exp", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(3));
  x[0] = eT(10);
  x[1] = eT(100);
  x[2] = eT(1000);
  if ((double) std::numeric_limits<eT>::max() < exp(100))
    {
    x[1] = eT(log(double(std::numeric_limits<eT>::max())));
    x[2] = eT(x[1]);
    }
  else if ((double) std::numeric_limits<eT>::max() < exp(100))
    {
    x[2] = eT(log(double(std::numeric_limits<eT>::max())));
    }

  Mat<eT> y = trunc_exp(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(arma::trunc_exp(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_cos", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_sin", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_tan", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_acos", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu[3] = 1.0;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_asin", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu[3] = 1.0;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_atan", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_cosh", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_sinh", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_tanh", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_acosh", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  x_cpu += 1;
  x_cpu[2] = 1.0;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_asinh", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_atanh", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<double> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  Mat<eT> x(arma::conv_to<arma::Mat<eT>>::from(x_cpu));

  Mat<eT> y = sin(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(sin(eT(x(r, c))))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eglue_atan2", "[eop_scalar]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> x_cpu(5, 5, arma::fill::randn);
  x_cpu *= 50;
  arma::Mat<eT> y_cpu(5, 5, arma::fill::randn);
  y_cpu *= 50;

  Mat<eT> x(x_cpu);
  Mat<eT> y(y_cpu);

  Mat<eT> z = atan2(x, y);

  arma::Mat<eT> z_cpu = arma::atan2(x_cpu, y_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(z_cpu(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eglue_hypot", "[eop_scalar]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> x_cpu(5, 5, arma::fill::randu);
  x_cpu *= 50;
  arma::Mat<eT> y_cpu(5, 5, arma::fill::randu);
  y_cpu *= 50;

  Mat<eT> x(x_cpu);
  Mat<eT> y(y_cpu);

  Mat<eT> z = hypot(x, y);
  arma::Mat<eT> z_cpu = arma::hypot(x_cpu, y_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(z_cpu(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_sinc", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randn);
  xd_cpu *= 50;
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = sinc(x);
  arma::Mat<eT> y_cpu = arma::sinc(x_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))).epsilon(0.01) );
      }
    }
  }



template<typename eT>
void test_pow(eT exponent)
  {
  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randn);
  xd_cpu *= 20;
  xd_cpu[0] = 5;
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = pow(x, exponent);
  arma::Mat<eT> y_cpu = pow(x_cpu, exponent);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      // Casting issues combined with pow() implementational differences could result in the GPU solution appearing to be off by 1.
      REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))).margin(1) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_pow_1", "[eop_scalar]", double, float)
  {
  typedef TestType eT;

  test_pow<eT>(0);
  test_pow<eT>(1);
  test_pow<eT>(2);
  test_pow<eT>(3);
  test_pow<eT>(-1);
  }



TEMPLATE_TEST_CASE("eop_pow_2", "[eop_scalar]", u32, s32, u64, s64)
  {
  typedef TestType eT;

  test_pow<eT>(0);
  test_pow<eT>(1);
  test_pow<eT>(2);
  test_pow<eT>(3);
  }



TEMPLATE_TEST_CASE("eop_floor", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randu);
  xd_cpu *= 50;
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = floor(x);
  arma::Mat<eT> y_cpu = floor(x_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_ceil", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randu);
  xd_cpu *= 50;
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = ceil(x);
  arma::Mat<eT> y_cpu = ceil(x_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_round", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randu);
  xd_cpu *= 50;
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = round(x);
  arma::Mat<eT> y_cpu = round(x_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_trunc", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randu);
  xd_cpu *= 50;
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = trunc(x);
  arma::Mat<eT> y_cpu = trunc(x_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_sign", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randu);
  xd_cpu *= 50;
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = sign(x);
  arma::Mat<eT> y_cpu = sign(x_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_erf", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randu);
  xd_cpu *= 50;
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = erf(x);
  arma::Mat<eT> y_cpu = erf(x_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      // Small floating-point implementational differences can result in an off-by-one error.
      if (!is_real<eT>::value)
        REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))).margin(1) );
      else
        REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_erfc", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randu);
  xd_cpu *= 50;
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = erfc(x);
  arma::Mat<eT> y_cpu = erfc(x_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      // Small floating-point implementational differences can result in an off-by-one error.
      if (!is_real<eT>::value)
        REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))).margin(1) );
      else
        REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("eop_lgamma", "[eop_scalar]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::mat xd_cpu(5, 5, arma::fill::randu);
  xd_cpu *= 5;
  xd_cpu += 1; // lgamma(x) is undefined for x <= 0
  arma::Mat<eT> x_cpu = arma::conv_to<arma::Mat<eT>>::from(xd_cpu);

  Mat<eT> x(x_cpu);

  Mat<eT> y = lgamma(x);
  arma::Mat<eT> y_cpu = lgamma(x_cpu);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y_cpu(r, c))) );
      }
    }
  }
