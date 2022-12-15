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
void test_fill()
  {
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



TEST_CASE("fill_1")
  {
  test_fill<double>();
  test_fill<float>();
  test_fill<u32>();
  test_fill<s32>();
  test_fill<u64>();
  test_fill<s64>();
  }



template<typename eT>
void test_fill_2()
  {
  Mat<eT> x(5, 5);

  x.fill(eT(50));;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(50)) );
      }
    }
  }



TEST_CASE("fill_2")
  {
  test_fill_2<float>();
  test_fill_2<double>();
  test_fill_2<u32>();
  test_fill_2<s32>();
  test_fill_2<u64>();
  test_fill_2<s64>();
  }



template<typename eT>
void test_scalar_plus()
  {
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



TEST_CASE("scalar_plus")
  {
  test_scalar_plus<float>();
  test_scalar_plus<double>();
  test_scalar_plus<u32>();
  test_scalar_plus<s32>();
  test_scalar_plus<u64>();
  test_scalar_plus<s64>();
  }



template<typename eT>
void test_scalar_minus()
  {
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



TEST_CASE("scalar_minus")
  {
  test_scalar_minus<float>();
  test_scalar_minus<double>();
  test_scalar_minus<u32>();
  test_scalar_minus<s32>();
  test_scalar_minus<u64>();
  test_scalar_minus<s64>();
  }



template<typename eT>
void test_scalar_mul()
  {
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



TEST_CASE("scalar_mul")
  {
  test_scalar_mul<float>();
  test_scalar_mul<double>();
  test_scalar_mul<u32>();
  test_scalar_mul<s32>();
  test_scalar_mul<u64>();
  test_scalar_mul<s64>();
  }



template<typename eT>
void test_scalar_div()
  {
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



TEST_CASE("scalar_div")
  {
  test_scalar_div<float>();
  test_scalar_div<double>();
  test_scalar_div<u32>();
  test_scalar_div<s32>();
  test_scalar_div<u64>();
  test_scalar_div<s64>();
  }



template<typename eT>
void test_submat_scalar_fill()
  {
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



TEST_CASE("submat_scalar_fill")
  {
  test_submat_scalar_fill<float>();
  test_submat_scalar_fill<double>();
  test_submat_scalar_fill<u32>();
  test_submat_scalar_fill<s32>();
  test_submat_scalar_fill<u64>();
  test_submat_scalar_fill<s64>();
  }



template<typename eT>
void test_submat_scalar_add()
  {
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



TEST_CASE("submat_scalar_add")
  {
  test_submat_scalar_add<float>();
  test_submat_scalar_add<double>();
  test_submat_scalar_add<u32>();
  test_submat_scalar_add<s32>();
  test_submat_scalar_add<u64>();
  test_submat_scalar_add<s64>();
  }




template<typename eT>
void test_submat_scalar_minus()
  {
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



TEST_CASE("submat_scalar_minus")
  {
  test_submat_scalar_minus<float>();
  test_submat_scalar_minus<double>();
  test_submat_scalar_minus<u32>();
  test_submat_scalar_minus<s32>();
  test_submat_scalar_minus<u64>();
  test_submat_scalar_minus<s64>();
  }




template<typename eT>
void test_submat_scalar_mul()
  {
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



TEST_CASE("submat_scalar_mul")
  {
  test_submat_scalar_mul<float>();
  test_submat_scalar_mul<double>();
  test_submat_scalar_mul<u32>();
  test_submat_scalar_mul<s32>();
  test_submat_scalar_mul<u64>();
  test_submat_scalar_mul<s64>();
  }




template<typename eT>
void test_submat_scalar_div()
  {
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



TEST_CASE("submat_scalar_div")
  {
  test_submat_scalar_div<float>();
  test_submat_scalar_div<double>();
  test_submat_scalar_div<u32>();
  test_submat_scalar_div<s32>();
  test_submat_scalar_div<u64>();
  test_submat_scalar_div<s64>();
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



template<typename eT1, typename eT2>
void test_eop_scalar_plus()
  {
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



TEST_CASE("eop_scalar_plus_1")
  {
  test_eop_scalar_plus<u32, u32>();
  test_eop_scalar_plus<u32, s32>();
  test_eop_scalar_plus<u32, u64>();
  test_eop_scalar_plus<u32, s64>();
  test_eop_scalar_plus<u32, float>();
  test_eop_scalar_plus<u32, double>();
  test_eop_scalar_plus<s32, u32>();
  test_eop_scalar_plus<s32, s32>();
  test_eop_scalar_plus<s32, u64>();
  test_eop_scalar_plus<s32, s64>();
  test_eop_scalar_plus<s32, float>();
  test_eop_scalar_plus<s32, double>();
  test_eop_scalar_plus<u64, u32>();
  test_eop_scalar_plus<u64, s32>();
  test_eop_scalar_plus<u64, u64>();
  test_eop_scalar_plus<u64, s64>();
  test_eop_scalar_plus<u64, float>();
  test_eop_scalar_plus<u64, double>();
  test_eop_scalar_plus<s64, u32>();
  test_eop_scalar_plus<s64, s32>();
  test_eop_scalar_plus<s64, u64>();
  test_eop_scalar_plus<s64, s64>();
  test_eop_scalar_plus<s64, float>();
  test_eop_scalar_plus<s64, double>();
  test_eop_scalar_plus<float, u32>();
  test_eop_scalar_plus<float, s32>();
  test_eop_scalar_plus<float, u64>();
  test_eop_scalar_plus<float, s64>();
  test_eop_scalar_plus<float, float>();
  test_eop_scalar_plus<float, double>();
  test_eop_scalar_plus<double, u32>();
  test_eop_scalar_plus<double, s32>();
  test_eop_scalar_plus<double, u64>();
  test_eop_scalar_plus<double, s64>();
  test_eop_scalar_plus<double, float>();
  test_eop_scalar_plus<double, double>();
  }



template<typename eT>
void test_eop_neg()
  {
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



TEST_CASE("eop_neg_1")
  {
  test_eop_neg<float>();
  test_eop_neg<double>();
  test_eop_neg<s32>();
  test_eop_neg<s64>();
  }



template<typename eT>
void test_eop_scalar_minus_pre()
  {
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



TEST_CASE("eop_scalar_minus_pre")
  {
  test_eop_scalar_minus_pre<float>();
  test_eop_scalar_minus_pre<double>();
  test_eop_scalar_minus_pre<u32>();
  test_eop_scalar_minus_pre<s32>();
  test_eop_scalar_minus_pre<u64>();
  test_eop_scalar_minus_pre<s64>();
  }



template<typename eT>
void test_eop_scalar_minus_post()
  {
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



TEST_CASE("eop_scalar_minus_post")
  {
  test_eop_scalar_minus_post<float>();
  test_eop_scalar_minus_post<double>();
  test_eop_scalar_minus_post<u32>();
  test_eop_scalar_minus_post<s32>();
  test_eop_scalar_minus_post<u64>();
  test_eop_scalar_minus_post<s64>();
  }



template<typename eT>
void test_eop_scalar_times()
  {
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



TEST_CASE("eop_scalar_times")
  {
  test_eop_scalar_times<float>();
  test_eop_scalar_times<double>();
  test_eop_scalar_times<u32>();
  test_eop_scalar_times<s32>();
  test_eop_scalar_times<u64>();
  test_eop_scalar_times<s64>();
  }



template<typename eT>
void test_eop_scalar_div_pre()
  {
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



TEST_CASE("eop_scalar_div_pre")
  {
  test_eop_scalar_div_pre<float>();
  test_eop_scalar_div_pre<double>();
  test_eop_scalar_div_pre<u32>();
  test_eop_scalar_div_pre<s32>();
  test_eop_scalar_div_pre<u64>();
  test_eop_scalar_div_pre<s64>();
  }



template<typename eT>
void test_eop_scalar_div_post()
  {
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



TEST_CASE("eop_scalar_div_post")
  {
  test_eop_scalar_div_post<float>();
  test_eop_scalar_div_post<double>();
  test_eop_scalar_div_post<u32>();
  test_eop_scalar_div_post<s32>();
  test_eop_scalar_div_post<u64>();
  test_eop_scalar_div_post<s64>();
  }



template<typename eT>
void test_eop_square()
  {
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



TEST_CASE("eop_square")
  {
  test_eop_square<float>();
  test_eop_square<double>();
  test_eop_square<u32>();
  test_eop_square<s32>();
  test_eop_square<u64>();
  test_eop_square<s64>();
  }



template<typename eT>
void test_eop_sqrt()
  {
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



TEST_CASE("eop_sqrt")
  {
  test_eop_sqrt<float>();
  test_eop_sqrt<double>();
  test_eop_sqrt<u32>();
  test_eop_sqrt<s32>();
  test_eop_sqrt<u64>();
  test_eop_sqrt<s64>();
  }



template<typename eT>
void test_eop_log()
  {
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



TEST_CASE("eop_log")
  {
  test_eop_log<float>();
  test_eop_log<double>();
  test_eop_log<u32>();
  test_eop_log<s32>();
  test_eop_log<u64>();
  test_eop_log<s64>();
  }



template<typename eT>
void test_eop_exp()
  {
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



TEST_CASE("eop_exp")
  {
  test_eop_exp<float>();
  test_eop_exp<double>();
  test_eop_exp<u32>();
  test_eop_exp<s32>();
  test_eop_exp<u64>();
  test_eop_exp<s64>();
  }



template<typename eT>
void test_eop_abs()
  {
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



TEST_CASE("eop_abs")
  {
  test_eop_exp<float>();
  test_eop_exp<double>();
  test_eop_exp<u32>();
  test_eop_exp<s32>();
  test_eop_exp<u64>();
  test_eop_exp<s64>();
  }



template<typename eT>
void test_eop_log2()
  {
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



TEST_CASE("eop_log2")
  {
  test_eop_log2<float>();
  test_eop_log2<double>();
  test_eop_log2<u32>();
  test_eop_log2<s32>();
  test_eop_log2<u64>();
  test_eop_log2<s64>();
  }



template<typename eT>
void test_eop_log10()
  {
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



TEST_CASE("eop_log10")
  {
  test_eop_log10<float>();
  test_eop_log10<double>();
  test_eop_log10<u32>();
  test_eop_log10<s32>();
  test_eop_log10<u64>();
  test_eop_log10<s64>();
  }



template<typename eT>
void test_eop_trunc_log()
  {
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



template<typename eT>
void test_eop_trunc_log_pos()
  {
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



// We can't test anything that will return a negative value for unsigned types.
TEST_CASE("eop_trunc_log")
  {
  test_eop_trunc_log<float>();
  test_eop_trunc_log<double>();
  test_eop_trunc_log_pos<u32>();
  test_eop_trunc_log<s32>();
  test_eop_trunc_log_pos<u64>();
  test_eop_trunc_log<s64>();
  }



template<typename eT>
void test_eop_exp2()
  {
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



TEST_CASE("eop_exp2")
  {
  test_eop_exp2<float>();
  test_eop_exp2<double>();
  test_eop_exp2<u32>();
  test_eop_exp2<s32>();
  test_eop_exp2<u64>();
  test_eop_exp2<s64>();
  }



template<typename eT>
void test_eop_exp10()
  {
  Mat<eT> x(5, 5);
  x.fill(eT(3));

  Mat<eT> y = exp10(x);

  for (uword r = 0; r < 5; ++r)
    {
    for (uword c = 0; c < 5; ++c)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(exp10(eT(x(r, c))))).epsilon(0.01) );
      }
    }
  }



TEST_CASE("eop_exp10")
  {
  test_eop_exp10<float>();
  test_eop_exp10<double>();
  test_eop_exp10<u32>();
  test_eop_exp10<s32>();
  test_eop_exp10<u64>();
  test_eop_exp10<s64>();
  }




template<typename eT>
void test_eop_trunc_exp()
  {
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



TEST_CASE("eop_trunc_exp")
  {
  test_eop_trunc_exp<float>();
  test_eop_trunc_exp<double>();
  test_eop_trunc_exp<u32>();
  test_eop_trunc_exp<s32>();
  test_eop_trunc_exp<u64>();
  test_eop_trunc_exp<s64>();
  }



template<typename eT>
void test_eop_cos()
  {
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



TEST_CASE("eop_cos")
  {
  test_eop_cos<float>();
  test_eop_cos<double>();
  test_eop_cos<u32>();
  test_eop_cos<s32>();
  test_eop_cos<u64>();
  test_eop_cos<s64>();
  }



template<typename eT>
void test_eop_sin()
  {
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



TEST_CASE("eop_sin")
  {
  test_eop_sin<float>();
  test_eop_sin<double>();
  test_eop_sin<u32>();
  test_eop_sin<s32>();
  test_eop_sin<u64>();
  test_eop_sin<s64>();
  }



template<typename eT>
void test_eop_tan()
  {
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



TEST_CASE("eop_tan")
  {
  test_eop_tan<float>();
  test_eop_tan<double>();
  test_eop_tan<u32>();
  test_eop_tan<s32>();
  test_eop_tan<u64>();
  test_eop_tan<s64>();
  }



template<typename eT>
void test_eop_acos()
  {
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



TEST_CASE("eop_acos")
  {
  test_eop_acos<float>();
  test_eop_acos<double>();
  test_eop_acos<u32>();
  test_eop_acos<s32>();
  test_eop_acos<u64>();
  test_eop_acos<s64>();
  }



template<typename eT>
void test_eop_asin()
  {
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



TEST_CASE("eop_asin")
  {
  test_eop_asin<float>();
  test_eop_asin<double>();
  test_eop_asin<u32>();
  test_eop_asin<s32>();
  test_eop_asin<u64>();
  test_eop_asin<s64>();
  }



template<typename eT>
void test_eop_atan()
  {
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



TEST_CASE("eop_atan")
  {
  test_eop_atan<float>();
  test_eop_atan<double>();
  test_eop_atan<u32>();
  test_eop_atan<s32>();
  test_eop_atan<u64>();
  test_eop_atan<s64>();
  }



template<typename eT>
void test_eop_cosh()
  {
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



TEST_CASE("eop_cosh")
  {
  test_eop_cosh<float>();
  test_eop_cosh<double>();
  test_eop_cosh<u32>();
  test_eop_cosh<s32>();
  test_eop_cosh<u64>();
  test_eop_cosh<s64>();
  }



template<typename eT>
void test_eop_sinh()
  {
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



TEST_CASE("eop_sinh")
  {
  test_eop_sinh<float>();
  test_eop_sinh<double>();
  test_eop_sinh<u32>();
  test_eop_sinh<s32>();
  test_eop_sinh<u64>();
  test_eop_sinh<s64>();
  }



template<typename eT>
void test_eop_tanh()
  {
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



TEST_CASE("eop_tanh")
  {
  test_eop_tanh<float>();
  test_eop_tanh<double>();
  test_eop_tanh<u32>();
  test_eop_tanh<s32>();
  test_eop_tanh<u64>();
  test_eop_tanh<s64>();
  }



template<typename eT>
void test_eop_acosh()
  {
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



TEST_CASE("eop_acosh")
  {
  test_eop_acosh<float>();
  test_eop_acosh<double>();
  test_eop_acosh<u32>();
  test_eop_acosh<s32>();
  test_eop_acosh<u64>();
  test_eop_acosh<s64>();
  }



template<typename eT>
void test_eop_asinh()
  {
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



TEST_CASE("eop_asinh")
  {
  test_eop_asinh<float>();
  test_eop_asinh<double>();
  test_eop_asinh<u32>();
  test_eop_asinh<s32>();
  test_eop_asinh<u64>();
  test_eop_asinh<s64>();
  }



template<typename eT>
void test_eop_atanh()
  {
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



TEST_CASE("eop_atanh")
  {
  test_eop_atanh<float>();
  test_eop_atanh<double>();
  test_eop_atanh<u32>();
  test_eop_atanh<s32>();
  test_eop_atanh<u64>();
  test_eop_atanh<s64>();
  }



template<typename eT>
void test_eglue_atan2()
  {
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



TEST_CASE("eglue_atan2")
  {
  test_eglue_atan2<float>();
  test_eglue_atan2<double>();
  }



template<typename eT>
void test_eglue_hypot()
  {
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



TEST_CASE("eglue_hypot")
  {
  test_eglue_hypot<float>();
  test_eglue_hypot<double>();
  }



template<typename eT>
void test_eop_sinc()
  {
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



TEST_CASE("eop_sinc")
  {
  test_eop_sinc<float>();
  test_eop_sinc<double>();
  test_eop_sinc<u32>();
  test_eop_sinc<s32>();
  test_eop_sinc<u64>();
  test_eop_sinc<s64>();
  }
