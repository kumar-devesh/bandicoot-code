// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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

// For the non-conv-to tests, we didn't implement any custom kernel for vectorise(), so we'll just tests with regular double matrices.

TEST_CASE("direct_vectorise_1", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);

  fvec y = vectorise(x);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x[i])) );
    }
  }



TEST_CASE("direct_vectorise_2", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);

  fvec y = vectorise(x, 0);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x[i])) );
    }
  }



TEST_CASE("direct_vectorise_3", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);

  fmat y = vectorise(x);

  REQUIRE( y.n_elem == x.n_elem );
  REQUIRE( y.n_cols == 1 );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x[i])) );
    }
  }



TEST_CASE("direct_vectorise_row", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat xt = x.t();

  frowvec y = vectorise(x, 1);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(xt[i])) );
    }
  }



TEST_CASE("empty_vectorise", "[vectorise]")
  {
  fmat x;
  fvec y = vectorise(x);

  REQUIRE( y.n_elem == 0 );
  }



TEST_CASE("empty_row_vectorise", "[vectorise]")
  {
  fmat x;
  frowvec y = vectorise(x, 1);

  REQUIRE( y.n_elem == 0 );
  }



TEST_CASE("vectorise_then_trans", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);

  frowvec y = trans(vectorise(x));

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x[i])) );
    }
  }



TEST_CASE("chained_vectorise", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);

  fvec y = vectorise(vectorise(x));

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x[i])) );
    }
  }



TEST_CASE("chained_row_vectorise", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat xt = trans(x);

  frowvec y = vectorise(vectorise(x, 1), 1);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(xt[i])) );
    }
  }


TEST_CASE("long_chained_vectorise", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);

  fvec y = vectorise(vectorise(vectorise(vectorise(vectorise(x, 0)), 0)), 0);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x[i])) );
    }
  }



TEST_CASE("long_chained_row_vectorise", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat xt = trans(x);

  frowvec y = vectorise(vectorise(vectorise(vectorise(vectorise(x, 1)), 1)), 1);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(xt[i])) );
    }
  }



TEST_CASE("long_mixed_chained_vectorise", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);

  frowvec y = vectorise(vectorise(vectorise(vectorise(vectorise(x), 1), 0)), 1);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x[i])) );
    }
  }



TEST_CASE("vectorise_eop", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);

  fvec y = vectorise((x + 3) * 2);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx((float(x[i]) + 3.0) * 2.0) );
    }
  }



TEST_CASE("vectorise_row_eop", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat xt = trans(x);

  frowvec y = vectorise((x + 3) * 2, 1);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx((float(xt[i]) + 3.0) * 2.0) );
    }
  }



TEST_CASE("vectorise_op", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);

  fvec y = vectorise(repmat(x, 1, 1));

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x[i])) );
    }
  }



TEST_CASE("vectorise_row_op", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat xt = trans(x);

  frowvec y = vectorise(repmat(x, 1, 1), 1);

  REQUIRE( y.n_elem == x.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(xt[i])) );
    }
  }



TEST_CASE("vectorise_eglue", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(10, 10);
  fmat z = x + y.t();

  fvec t = vectorise(x + y.t());

  REQUIRE( t.n_elem == x.n_elem );

  for (size_t i = 0; i < t.n_elem; ++i)
    {
    REQUIRE( float(t[i]) == Approx(float(z[i])) );
    }
  }



TEST_CASE("vectorise_row_eglue", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(10, 10);
  fmat z = x + y.t();
  fmat zt = trans(z);

  frowvec t = vectorise(x + y.t(), 1);

  REQUIRE( t.n_elem == x.n_elem );

  for (size_t i = 0; i < t.n_elem; ++i)
    {
    REQUIRE( float(t[i]) == Approx(float(zt[i])) );
    }
  }



TEST_CASE("vectorise_glue", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(10, 10);
  fmat z = x * y.t();

  fvec t = vectorise(x * y.t());

  REQUIRE( t.n_elem == z.n_elem );

  for (size_t i = 0; i < t.n_elem; ++i)
    {
    REQUIRE( float(t[i]) == Approx(float(z[i])) );
    }
  }



TEST_CASE("vectorise_row_glue", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(10, 10);
  fmat z = x * y.t();
  fmat zt = trans(z);

  frowvec t = vectorise(x * y.t(), 1);

  REQUIRE( t.n_elem == z.n_elem );

  for (size_t i = 0; i < t.n_elem; ++i)
    {
    REQUIRE( float(t[i]) == Approx(float(zt[i])) );
    }
  }



TEST_CASE("vectorises_inside_eop", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(10, 10);

  fvec z = vectorise(x) + vectorise(y, 0) - 3.0;

  REQUIRE( z.n_elem == x.n_elem );

  for (size_t i = 0; i < z.n_elem; ++i)
    {
    REQUIRE( float(z[i]) == Approx(float(x[i]) + float(y[i]) - 3.0) );
    }
  }



TEST_CASE("row_vectorises_inside_eop", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(10, 10);
  fmat xt = trans(x);
  fmat yt = trans(y);

  frowvec z = vectorise(x, 1) + vectorise(y, 1) - 3.0;

  REQUIRE( z.n_elem == x.n_elem );

  for (size_t i = 0; i < z.n_elem; ++i)
    {
    REQUIRE( float(z[i]) == Approx(float(xt[i]) + float(yt[i]) - 3.0) );
    }
  }



TEST_CASE("vectorises_inside_glue", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(10, 10);

  fmat z = vectorise(x) * trans(vectorise(y));

  fvec xv = vectorise(x);
  fvec yv = vectorise(y);
  fmat zz = xv * trans(yv);

  REQUIRE( z.n_elem == x.n_elem * y.n_elem );

  for (size_t c = 0; c < z.n_cols; ++c)
    {
    for (size_t r = 0; r < z.n_rows; ++r)
      {
      REQUIRE( float(z(r, c)) == Approx(float(zz(r, c))) );
      }
    }
  }



TEST_CASE("row_vectorises_inside_glue", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(10, 10);

  fmat z = trans(vectorise(x, 1)) * vectorise(y, 1);

  fvec xv = vectorise(x, 1).t();
  frowvec yv = vectorise(y, 1);
  fmat zz = xv * yv;

  REQUIRE( z.n_elem == x.n_elem * y.n_elem );

  for (size_t c = 0; c < z.n_cols; ++c)
    {
    for (size_t r = 0; r < z.n_rows; ++r)
      {
      REQUIRE( float(z(r, c)) == Approx(float(zz(r, c))) );
      }
    }
  }



TEST_CASE("vectorise_inplace_subview", "[vectorise]")
  {
  fmat x = randu<fmat>(20, 20);
  fmat x_old(x);
  fmat x_sub = x.submat(0, 0, 3, 3);

  x.submat(0, 0, 15, 0) = vectorise(x.submat(0, 0, 3, 3));

  for (size_t c = 0; c < x.n_cols; ++c)
    {
    for (size_t r = 0; r < x.n_rows; ++r)
      {
      if (c == 0 && r <= 15)
        {
        REQUIRE( float(x(r, c)) == Approx(float(x_sub[r])) );
        }
      else
        {
        REQUIRE( float(x(r, c)) == Approx(float(x_old(r, c))) );
        }
      }
    }
  }



TEST_CASE("vectorise_row_inplace_subview", "[vectorise]")
  {
  fmat x = randu<fmat>(20, 20);
  fmat x_old(x);
  fmat x_sub = x.submat(0, 0, 3, 3);

  x.submat(0, 0, 0, 15) = vectorise(x.submat(0, 0, 3, 3), 1);

  for (size_t c = 0; c < x.n_cols; ++c)
    {
    for (size_t r = 0; r < x.n_rows; ++r)
      {
      if (c <= 15 && r == 0)
        {
        REQUIRE( float(x(r, c)) == Approx(float(x_sub[c])) );
        }
      else
        {
        REQUIRE( float(x(r, c)) == Approx(float(x_old(r, c))) );
        }
      }
    }
  }



TEST_CASE("vectorise_non_inplace_subview", "[vectorise]")
  {
  fmat x = randu<fmat>(20, 20);

  fvec y = vectorise(x.submat(0, 0, 9, 9));
  fmat x_sub = x.submat(0, 0, 9, 9);

  REQUIRE( y.n_elem == x_sub.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x_sub[i])) );
    }
  }



TEST_CASE("vectorise_row_non_inplace_subview", "[vectorise]")
  {
  fmat x = randu<fmat>(20, 20);

  frowvec y = vectorise(x.submat(0, 0, 9, 9), 1);
  fmat x_sub = x.submat(0, 0, 9, 9);

  REQUIRE( y.n_elem == x_sub.n_elem );

  for (size_t i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( float(y[i]) == Approx(float(x_sub[i])) );
    }
  }



TEST_CASE("vectorise_bonanza", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(2, 50);
  fmat z = randu<fmat>(100, 100);

  fvec out = vectorise(z % ((vectorise(x) * 3) * trans(vectorise(y) - 2)));

  // Now assemble by hand for comparison...
  fvec tmp1 = vectorise(x) * 3;
  fvec tmp2 = vectorise(y) - 2;
  fmat tmp3 = tmp1 * trans(tmp2);
  fmat tmp4 = z % tmp3;

  REQUIRE( out.n_elem == tmp4.n_elem );

  for (size_t i = 0; i < out.n_elem; ++i)
    {
    REQUIRE( float(out[i]) == Approx(float(tmp4[i])) );
    }
  }



TEST_CASE("vectorise_row_bonanza", "[vectorise]")
  {
  fmat x = randu<fmat>(10, 10);
  fmat y = randu<fmat>(2, 50);
  fmat z = randu<fmat>(100, 100);

  frowvec out = vectorise(z % (trans(vectorise(x, 1) * 3) * (vectorise(y, 1) - 2)), 1).t().t();

  // Now assemble by hand for comparison...
  frowvec tmp1 = vectorise(x, 1) * 3;
  frowvec tmp2 = vectorise(y, 1) - 2;
  fmat tmp3 = trans(tmp1) * tmp2;
  fmat tmp4 = z % tmp3;
  fmat tmp4t = trans(tmp4); // has size 100x100 but is in the same order as out

  REQUIRE( out.n_elem == tmp4t.n_elem );

  for (size_t i = 0; i < out.n_elem; ++i)
    {
    REQUIRE( float(out[i]) == Approx(float(tmp4t[i])) );
    }
  }



TEMPLATE_TEST_CASE(
  "vectorise_pre_conv_to",
  "[vectorise]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(5, 5, distr_param(0, 50));
  Col<eT2> y = conv_to<Mat<eT2>>::from(vectorise(x));

  REQUIRE(y.n_elem == x.n_elem);

  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT2(y[i]) == Approx(eT2(eT1(x[i]))) );
    }
  }



TEMPLATE_TEST_CASE(
  "vectorise_row_pre_conv_to",
  "[vectorise]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(5, 5, distr_param(0, 50));
  Mat<eT1> xt = trans(x);
  Row<eT2> y = conv_to<Mat<eT2>>::from(vectorise(x, 1));

  REQUIRE(y.n_elem == x.n_elem);

  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT2(y[i]) == Approx(eT2(eT1(xt[i]))) );
    }
  }



TEMPLATE_TEST_CASE(
  "vectorise_post_conv_to",
  "[vectorise]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(5, 5, distr_param(0, 50));
  Mat<eT2> y = vectorise(conv_to<Mat<eT2>>::from(x));

  REQUIRE(y.n_elem == x.n_elem);

  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT2(y[i]) == Approx(eT2(eT1(x[i]))) );
    }
  }



TEMPLATE_TEST_CASE(
  "vectorise_row_post_conv_to",
  "[vectorise]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(5, 5, distr_param(0, 50));
  Mat<eT1> xt = trans(x);
  Row<eT2> y = vectorise(conv_to<Mat<eT2>>::from(x), 1);

  REQUIRE(y.n_elem == x.n_elem);

  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT2(y[i]) == Approx(eT2(eT1(xt[i]))) );
    }
  }
