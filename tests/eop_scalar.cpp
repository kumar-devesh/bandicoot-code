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
