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
#include <typeinfo>

#define TWOWAY_TEST_HELPER(TEST_NAME, ET1) \
    TEST_NAME<ET1, u32>(); \
    TEST_NAME<ET1, s32>(); \
    TEST_NAME<ET1, u64>(); \
    TEST_NAME<ET1, s64>(); \
    TEST_NAME<ET1, float>(); \
    TEST_NAME<ET1, double>();

#define TWOWAY_TESTS(TEST_NAME) \
    TWOWAY_TEST_HELPER(TEST_NAME, u32) \
    TWOWAY_TEST_HELPER(TEST_NAME, s32) \
    TWOWAY_TEST_HELPER(TEST_NAME, u64) \
    TWOWAY_TEST_HELPER(TEST_NAME, s64) \
    TWOWAY_TEST_HELPER(TEST_NAME, float) \
    TWOWAY_TEST_HELPER(TEST_NAME, double)

using namespace coot;

// The simplest possible conv_to test.
template<typename eT1, typename eT2>
void test_conv_to_simple()
  {
  Mat<eT1> x(5, 5);
  for (uword i = 0; i < 25; ++i)
    x[i] = i + 1;

  Mat<eT2> y = conv_to<Mat<eT2>>::from(x);

  for (uword i = 0; i < 25; ++i)
    REQUIRE( (eT2) y[i] == Approx((eT2) (i + 1)) );
  }



TEST_CASE("conv_to_simple")
  {
  TWOWAY_TESTS(test_conv_to_simple);
  }



// Really the operation is easy here, the test is more that it compiles.
template<typename eT1, typename eT2>
void test_conv_to_accu()
  {
  Mat<eT1> x(5, 5);
  for (uword i = 0; i < 25; ++i)
    x[i] = i + 1;

  eT2 sum = accu(conv_to<Mat<eT2>>::from(x));

  REQUIRE(sum == Approx(eT2(325)) );
  }



TEST_CASE("conv_to_accu")
  {
  TWOWAY_TESTS(test_conv_to_simple);
  }



// TODO: chol() only supports Mat arguments for now
/*
template<typename eT1, typename eT2>
void test_conv_to_chol()
  {
  Mat<eT1> x(5, 5);
  x.zeros();
  for (uword i = 0; i < 5; ++i)
    x(i, i) = 1;

  Mat<eT2> y;
  bool status = chol(y, conv_to<Mat<eT2>>::from(x));

  REQUIRE(status == true);

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      if (r > c)
        REQUIRE(eT2(y(r, c)) == eT2(0));
      else
        REQUIRE( eT2(x(r, c)) == Approx(eT2(y(r, c))) );
      }
    }
  }



TEST_CASE("conv_to_chol")
  {
  TWOWAY_TESTS(test_conv_to_chol);
  }
*/



template<typename eT1, typename eT2>
void test_conv_to_dot()
  {
  Col<eT1> x(25);
  for (uword i = 0; i < 25; ++i)
    x[i] = i + 1;

  Col<eT2> y(25);
  for (uword i = 0; i < 25; ++i)
    y[i] = i + 1;

  eT1 out1 = dot(x, conv_to<Col<eT1>>::from(y));
  eT2 out2 = dot(conv_to<Col<eT2>>::from(x), y);

  REQUIRE( out1 == eT1(5525) );
  REQUIRE( out2 == eT2(5525) );
  }



TEST_CASE("conv_to_dot")
  {
  TWOWAY_TESTS(test_conv_to_dot);
  }



template<typename eT1, typename eT2>
void test_conv_to_eop_scalar_plus()
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



TEST_CASE("conv_to_eop_scalar_plus")
  {
  TWOWAY_TESTS(test_conv_to_eop_scalar_plus);
  }



template<typename eT1, typename eT2>
void test_conv_to_matmul()
  {
  arma::Mat<eT1> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT2> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT1> cpu_z1 = cpu_x * arma::conv_to<arma::Mat<eT1>>::from(cpu_y.t());
  arma::Mat<eT2> cpu_z2 = arma::conv_to<arma::Mat<eT2>>::from(cpu_x) * cpu_y.t();

  Mat<eT1> x(cpu_x);
  Mat<eT2> y(cpu_y);
  Mat<eT1> z1 = x * conv_to<Mat<eT1>>::from(y.t());
  Mat<eT2> z2 = conv_to<Mat<eT2>>::from(x) * y.t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT1(z1(r, c)) == Approx(eT1(cpu_z1(r, c))) );
      REQUIRE( eT2(z2(r, c)) == Approx(eT2(cpu_z2(r, c))) );
      }
    }
  }



TEST_CASE("conv_to_matmul")
  {
  test_conv_to_matmul<float, float>();
  test_conv_to_matmul<float, double>();
  test_conv_to_matmul<double, float>();
  test_conv_to_matmul<double, double>();
  }



template<typename eT1, typename eT2>
void test_conv_to_scalar_matmul()
  {
  arma::Mat<eT1> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT2> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT1> cpu_z1 = -cpu_x * arma::conv_to<arma::Mat<eT1>>::from(cpu_y.t());
  arma::Mat<eT2> cpu_z2 = arma::conv_to<arma::Mat<eT2>>::from(-cpu_x) * cpu_y.t();
  arma::Mat<eT1> cpu_z3 = 2.0 * cpu_x * arma::conv_to<arma::Mat<eT1>>::from(cpu_y.t());
  arma::Mat<eT2> cpu_z4 = arma::conv_to<arma::Mat<eT2>>::from(2.0 * cpu_x) * cpu_y.t();

  Mat<eT1> x(cpu_x);
  Mat<eT2> y(cpu_y);
  Mat<eT1> z1 = -x * conv_to<Mat<eT1>>::from(y.t());
  Mat<eT2> z2 = conv_to<Mat<eT2>>::from(-x) * y.t();
  Mat<eT1> z3 = 2.0 * x * conv_to<Mat<eT1>>::from(y.t());
  Mat<eT2> z4 = conv_to<Mat<eT2>>::from(2.0 * x) * y.t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT1(z1(r, c)) == Approx(eT1(cpu_z1(r, c))) );
      REQUIRE( eT2(z2(r, c)) == Approx(eT2(cpu_z2(r, c))) );
      REQUIRE( eT2(z3(r, c)) == Approx(eT2(cpu_z3(r, c))) );
      REQUIRE( eT2(z4(r, c)) == Approx(eT2(cpu_z4(r, c))) );
      }
    }
  }



TEST_CASE("conv_to_scalar_matmul")
  {
  test_conv_to_scalar_matmul<float, float>();
  test_conv_to_scalar_matmul<float, double>();
  test_conv_to_scalar_matmul<double, float>();
  test_conv_to_scalar_matmul<double, double>();
  }



// eT1 should be unsigned; eT2 should be signed
template<typename eT1, typename eT2>
void test_conv_to_outside_negation()
  {
  Mat<eT1> x(2, 2);
  x.fill(eT1(3));

  Mat<eT2> y = -conv_to<Mat<eT2>>::from(x);
  Mat<eT2> y2 = -conv_to<Mat<eT2>>::from(x + 1);

  REQUIRE( y.n_rows == 2 );
  REQUIRE( y.n_cols == 2 );
  REQUIRE( y2.n_rows == 2 );
  REQUIRE( y2.n_cols == 2 );
  for (uword i = 0; i < 4; ++i)
    {
    REQUIRE( eT2(y[i]) == eT2(-3) );
    REQUIRE( eT2(y2[i]) == eT2(-4) );
    }
  }



TEST_CASE("conv_to_outside_negation")
  {
  test_conv_to_outside_negation<u32, s32>();
  test_conv_to_outside_negation<u32, s64>();
  test_conv_to_outside_negation<u32, float>();
  test_conv_to_outside_negation<u32, double>();
  test_conv_to_outside_negation<u64, s32>();
  test_conv_to_outside_negation<u64, s64>();
  test_conv_to_outside_negation<u64, float>();
  test_conv_to_outside_negation<u64, double>();
  }



// eT1 should be signed; eT2 should be unsigned
template<typename eT1, typename eT2>
void test_conv_to_inside_negation()
  {
  Mat<eT1> x(2, 2);
  x.fill(eT1(-3));

  Mat<eT2> y = conv_to<Mat<eT2>>::from(-x);
  Mat<eT2> y2 = conv_to<Mat<eT2>>::from(-(x - 1));

  REQUIRE( y.n_rows == 2 );
  REQUIRE( y.n_cols == 2 );
  REQUIRE( y2.n_rows == 2 );
  REQUIRE( y2.n_cols == 2 );
  for (uword i = 0; i < 4; ++i)
    {
    REQUIRE( eT2(y[i]) == eT2(3) );
    REQUIRE( eT2(y2[i]) == eT2(4) );
    }
  }



TEST_CASE("conv_to_inside_negation")
  {
  test_conv_to_inside_negation<s32, u32>();
  test_conv_to_inside_negation<s32, u64>();
  test_conv_to_inside_negation<s64, u32>();
  test_conv_to_inside_negation<s64, u64>();
  test_conv_to_inside_negation<float, u32>();
  test_conv_to_inside_negation<float, u64>();
  test_conv_to_inside_negation<double, u32>();
  test_conv_to_inside_negation<double, u64>();
  }



// eT1 should be integer; eT2 should be floating-point
template<typename eT1, typename eT2>
void test_conv_to_inside_scalar_mul()
  {
  Mat<eT1> x(2, 2);
  x.fill(eT1(3));

  Mat<eT2> y = eT2(0.5) * conv_to<Mat<eT2>>::from(x);
  Mat<eT2> y2 = eT2(0.5) * conv_to<Mat<eT2>>::from(x + 2);

  REQUIRE( y.n_rows == 2 );
  REQUIRE( y.n_cols == 2 );
  REQUIRE( y2.n_rows == 2 );
  REQUIRE( y2.n_cols == 2 );
  for (uword i = 0; i < 4; ++i)
    {
    REQUIRE( eT2(y[i]) == Approx(eT2(1.5)) );
    REQUIRE( eT2(y2[i]) == Approx(eT2(2.5)) );
    }
  }



TEST_CASE("conv_to_inside_scalar_mul")
  {
  test_conv_to_inside_scalar_mul<u32, float>();
  test_conv_to_inside_scalar_mul<s32, float>();
  test_conv_to_inside_scalar_mul<u64, float>();
  test_conv_to_inside_scalar_mul<s64, float>();
  test_conv_to_inside_scalar_mul<u32, double>();
  test_conv_to_inside_scalar_mul<s32, double>();
  test_conv_to_inside_scalar_mul<u64, double>();
  test_conv_to_inside_scalar_mul<s64, double>();
  }



// eT1 should be floating-point; eT2 should be integer
template<typename eT1, typename eT2>
void test_conv_to_outside_scalar_mul()
  {
  Mat<eT1> x(2, 2);
  x.fill(eT1(-1.5));

  Mat<eT2> y = conv_to<Mat<eT2>>::from(eT1(-2.0) * x);
  Mat<eT2> y2 = conv_to<Mat<eT2>>::from(eT1(-2.0) * (x - 1));

  REQUIRE( y.n_rows == 2 );
  REQUIRE( y.n_cols == 2 );
  REQUIRE( y2.n_rows == 2 );
  REQUIRE( y2.n_cols == 2 );
  for (uword i = 0; i < 4; ++i)
    {
    REQUIRE( eT2(y[i]) == eT2(3) );
    REQUIRE( eT2(y2[i]) == eT2(5) );
    }
  }



TEST_CASE("conv_to_outside_scalar_mul")
  {
  test_conv_to_outside_scalar_mul<float, u32>();
  test_conv_to_outside_scalar_mul<float, s32>();
  test_conv_to_outside_scalar_mul<float, u64>();
  test_conv_to_outside_scalar_mul<float, s64>();
  test_conv_to_outside_scalar_mul<double, u32>();
  test_conv_to_outside_scalar_mul<double, s32>();
  test_conv_to_outside_scalar_mul<double, u64>();
  test_conv_to_outside_scalar_mul<double, s64>();
  }



// Test eOps; make sure that conversions at the same time as an eOp give the
// same results as doing it separately.
template<typename out_eT, typename in_eT, typename eop_type>
void test_eop_conv_to(const out_eT aux = out_eT(0))
  {
  Mat<in_eT> x(3, 3);
  for (uword i = 0; i < 9; ++i)
    {
    x[i] = (i + 1);
    }

  typedef mtOp<out_eT, Mat<in_eT>, mtop_conv_to> internal_conv_to_type;

  typedef eOp<internal_conv_to_type, eop_type> eop_with_internal_conv_to_type;
  typedef eOp<Mat<out_eT>, eop_type> eop_after_conv_to_type;

  Mat<out_eT> xc = conv_to<Mat<out_eT>>::from(x);
  // Manually assemble eOp with conv_to internally.
  Mat<out_eT> y1 = eop_with_internal_conv_to_type(internal_conv_to_type(x), aux);
  Mat<out_eT> y2 = eop_after_conv_to_type(xc, aux);

  for (uword i = 0; i < 9; ++i)
    {
    REQUIRE( out_eT(y1[i]) == Approx(out_eT(y2[i])) );
    }

  // Now have the conv_to externally.
  typedef eOp<Mat<in_eT>, eop_type> eop_before_conv_to_type;
  typedef mtOp<out_eT, eop_before_conv_to_type, mtop_conv_to> eop_with_external_conv_to_type;

  Mat<out_eT> z1(eop_with_external_conv_to_type(eop_before_conv_to_type(x, aux)));
  Mat<in_eT> xop(eop_before_conv_to_type(x, aux));
  Mat<out_eT> z2(conv_to<Mat<out_eT>>::from(xop));

  for (uword i = 0; i < 9; ++i)
    {
    REQUIRE( out_eT(z1[i]) == Approx(out_eT(z2[i])) );
    }
  }



#define TWOWAY_EOP_TEST_HELPER(TEST_NAME, ET1, EOP_TYPE, ARG) \
    TEST_NAME<ET1, u32,    EOP_TYPE>(ARG); \
    TEST_NAME<ET1, s32,    EOP_TYPE>(ARG); \
    TEST_NAME<ET1, u64,    EOP_TYPE>(ARG); \
    TEST_NAME<ET1, s64,    EOP_TYPE>(ARG); \
    TEST_NAME<ET1, float,  EOP_TYPE>(ARG); \
    TEST_NAME<ET1, double, EOP_TYPE>(ARG);



#define TWOWAY_EOP_TESTS(TEST_NAME, EOP_TYPE, ARG) \
    TWOWAY_EOP_TEST_HELPER(TEST_NAME, u32,    EOP_TYPE, ARG) \
    TWOWAY_EOP_TEST_HELPER(TEST_NAME, s32,    EOP_TYPE, ARG) \
    TWOWAY_EOP_TEST_HELPER(TEST_NAME, u64,    EOP_TYPE, ARG) \
    TWOWAY_EOP_TEST_HELPER(TEST_NAME, s64,    EOP_TYPE, ARG) \
    TWOWAY_EOP_TEST_HELPER(TEST_NAME, float,  EOP_TYPE, ARG) \
    TWOWAY_EOP_TEST_HELPER(TEST_NAME, double, EOP_TYPE, ARG)



TEST_CASE("eop_conv_to")
  {
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_plus, 1);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_neg, 0);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_minus_pre, 100);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_minus_post, 1);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_times, 2);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_div_pre, 1);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_div_post, 2);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_square, 0);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_sqrt, 0);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_log, 0);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_exp, 0);
  }



// We need to make sure that the conversion actually happens here.
TEST_CASE("two_chained_conv_tos")
  {
  Mat<s32> x(2, 2);
  x.fill(s32(-1));

  Mat<s32> z = conv_to<Mat<s32>>::from(conv_to<Mat<u32>>::from(x));

  REQUIRE( z.n_rows == 2 );
  REQUIRE( z.n_cols == 2 );
  for (uword i = 0; i < 4; ++i)
    {
    REQUIRE( s32(z[i]) == s32(u32(s32(-1))) );
    }
  }



// Make sure that chaining together three conv_tos actually works too.
TEST_CASE("three_chained_conv_tos")
  {
  Mat<s32> x(2, 2);
  x.fill(s32(-1));

  Mat<s32> z1 = conv_to<Mat<s32>>::from(conv_to<Mat<s64>>::from(conv_to<Mat<u32>>::from(x)));
  Mat<s32> z2 = conv_to<Mat<s32>>::from(conv_to<Mat<u32>>::from(conv_to<Mat<s64>>::from(x)));

  REQUIRE( z1.n_rows == 2 );
  REQUIRE( z1.n_cols == 2 );
  REQUIRE( z2.n_rows == 2 );
  REQUIRE( z2.n_cols == 2 );
  for (uword i = 0; i < 4; ++i)
    {
    REQUIRE( s32(z1[i]) == s32(s64(u32(s32(-1)))) );
    REQUIRE( s32(z2[i]) == s32(u32(s64(s32(-1)))) );
    }
  }



// Since eOps can be merged with conv_to operations, make sure that we
// can properly "double-merge" an eOp<conv_to<eOp<...>>> into just one eOp.
// First, we'll do a simple test.
template<typename out_eT, typename in_eT>
void test_simple_double_merge_eop_conv_to()
  {
  Mat<in_eT> x(3, 3);
  for (uword i = 0; i < 9; ++i)
    {
    x[i] = (i + 1);
    }

  Mat<out_eT> y = conv_to<Mat<out_eT>>::from(x + 1) + 1;

  REQUIRE( y.n_rows == 3 );
  REQUIRE( y.n_cols == 3 );
  for (uword i = 0; i < 9; ++i)
    {
    REQUIRE( out_eT(y[i]) == out_eT(i + 3) );
    }
  }



TEST_CASE("simple_double_merge_eop_conv_to")
  {
  TWOWAY_TESTS(test_simple_double_merge_eop_conv_to);
  }



// Now, test every eop_type to make sure that the kernels are right.
// (However, we have to hand-assemble the eOps and Ops.)
template<typename out_eT, typename in_eT, typename eop_type>
void test_double_merged_eop_conv_to(out_eT aux_val)
  {
  out_eT aux_out = aux_val;
  in_eT  aux_in  = in_eT(aux_val);

  Mat<in_eT> x(3, 3);
  for (uword i = 0; i < 9; ++i)
    {
    x[i] = (i + 1);
    }

  typedef eOp<Mat<in_eT>, eop_type> inner_eop_type;
  typedef mtOp<out_eT, inner_eop_type, mtop_conv_to> conv_to_type;
  typedef eOp<conv_to_type, eop_type> outer_eop_type;
  typedef eOp<Mat<out_eT>, eop_type> standalone_outer_eop_type;

  // Do all operations at once (hopefully).
  Mat<out_eT> y1 = outer_eop_type(conv_to_type(inner_eop_type(x, aux_in)), aux_out);

  // Perform three separate operations (no chaining).
  Mat<in_eT> after_inner_eop = inner_eop_type(x, aux_in);
  Mat<out_eT> after_conv = conv_to<Mat<out_eT>>::from(after_inner_eop);
  Mat<out_eT> y2 = standalone_outer_eop_type(after_conv, aux_out);

  REQUIRE( y1.n_rows == 3 );
  REQUIRE( y1.n_cols == 3 );
  REQUIRE( y2.n_rows == 3 );
  REQUIRE( y2.n_cols == 3 );

  for (uword i = 0; i < 9; ++i)
    {
    REQUIRE( out_eT(y1[i]) == Approx(out_eT(y2[i])) );
    }
  }



TEST_CASE("double_merged_eop_conv_to")
  {
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_plus, 1);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_neg, 0);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_minus_pre, 100);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_minus_post, 1);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_times, 2);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_div_pre, 1);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_scalar_div_post, 2);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_square, 0);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_sqrt, 0);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_log, 0);
  TWOWAY_EOP_TESTS(test_eop_conv_to, eop_exp, 0);
  }



// Test colwise/rowwise sums mixed with conv_to operations.
// We can only do this test successfully with u32/u64s.
TEST_CASE("colwise_sum_conv_to")
  {
  Mat<u32> x(3, 3);
  x.fill(u32(-1));

  // If the conversion happens *after* the sum, then the sums will overflow.
  // Otherwise, they won't overflow.
  Mat<u64> y1 = conv_to<Mat<u64>>::from(sum(x, 0));
  Mat<u64> y2 = sum(conv_to<Mat<u64>>::from(x), 0);

  REQUIRE( y1.n_rows == 1 );
  REQUIRE( y1.n_cols == 3 );
  REQUIRE( y2.n_rows == 1 );
  REQUIRE( y2.n_cols == 3 );
  for (uword i = 0; i < 3; ++i)
    {
    REQUIRE( u64(y1[i]) == u64(u32(-3)) );
    REQUIRE( u64(y2[i]) == u64(3 * u64(u32(-1))) );
    }
  }



TEST_CASE("rowwise_sum_conv_to")
  {
  Mat<u32> x(3, 3);
  x.fill(u32(-1));

  // If the conversion happens *after* the sum, then the sums will overflow.
  // Otherwise, they won't overflow.
  Mat<u64> y1 = conv_to<Mat<u64>>::from(sum(x, 1));
  Mat<u64> y2 = sum(conv_to<Mat<u64>>::from(x), 1);

  REQUIRE( y1.n_rows == 3 );
  REQUIRE( y1.n_cols == 1 );
  REQUIRE( y2.n_rows == 3 );
  REQUIRE( y2.n_cols == 1 );
  for (uword i = 0; i < 3; ++i)
    {
    REQUIRE( u64(y1[i]) == u64(u32(-3)) );
    REQUIRE( u64(y2[i]) == u64(3 * u64(u32(-1))) );
    }
  }


TEST_CASE("submat_colwise_sum_conv_to")
  {
  Mat<u32> x(3, 3);
  x.fill(u32(-1));

  // If the conversion happens *after* the sum, then the sums will overflow.
  // Otherwise, they won't overflow.
  Mat<u64> y1 = conv_to<Mat<u64>>::from(sum(x.cols(0, 1), 0));
  Mat<u64> y2 = sum(conv_to<Mat<u64>>::from(x.cols(0, 1)), 0);

  REQUIRE( y1.n_rows == 1 );
  REQUIRE( y1.n_cols == 2 );
  REQUIRE( y2.n_rows == 1 );
  REQUIRE( y2.n_cols == 2 );
  for (uword i = 0; i < 2; ++i)
    {
    REQUIRE( u64(y1[i]) == u64(u32(-3)) );
    REQUIRE( u64(y2[i]) == u64(3 * u64(u32(-1))) );
    }
  }



TEST_CASE("submat_rowwise_sum_conv_to")
  {
  Mat<u32> x(3, 3);
  x.fill(u32(-1));

  // If the conversion happens *after* the sum, then the sums will overflow.
  // Otherwise, they won't overflow.
  Mat<u64> y1 = conv_to<Mat<u64>>::from(sum(x.rows(0, 1), 1));
  Mat<u64> y2 = sum(conv_to<Mat<u64>>::from(x.rows(0, 1)), 1);

  REQUIRE( y1.n_rows == 2 );
  REQUIRE( y1.n_cols == 1 );
  REQUIRE( y2.n_rows == 2 );
  REQUIRE( y2.n_cols == 1 );
  for (uword i = 0; i < 2; ++i)
    {
    REQUIRE( u64(y1[i]) == u64(u32(-3)) );
    REQUIRE( u64(y2[i]) == u64(3 * u64(u32(-1))) );
    }
  }


// inplace_set/plus/minus/schur/div_mat
template<typename eT2, typename eT1>
void test_inplace_set_conv_to()
  {
  Mat<eT1> x(5, 3);
  x.fill(eT1(3));

  Mat<eT2> y1(conv_to<Mat<eT2>>::from(x));
  Mat<eT2> y2 = conv_to<Mat<eT2>>::from(x);

  REQUIRE( y1.n_rows == 5 );
  REQUIRE( y1.n_cols == 3 );
  REQUIRE( y2.n_rows == 5 );
  REQUIRE( y2.n_cols == 3 );
  for (uword i = 0; i < 15; ++i)
    {
    REQUIRE( eT2(y1[i]) == eT2(3) );
    REQUIRE( eT2(y2[i]) == eT2(3) );
    }
  }



TEST_CASE("inplace_set_conv_to")
  {
  TWOWAY_TESTS(test_inplace_set_conv_to);
  }



template<typename eT2, typename eT1>
void test_inplace_plus_conv_to()
  {
  Mat<eT1> x(5, 3);
  x.fill(eT1(3));

  Mat<eT2> y(5, 3);
  y.fill(eT2(1));

  y += conv_to<Mat<eT2>>::from(x);

  REQUIRE( y.n_rows == 5 );
  REQUIRE( y.n_cols == 3 );
  for (uword i = 0; i < 15; ++i)
    {
    REQUIRE( eT2(y[i]) == eT2(4) );
    }
  }



TEST_CASE("inplace_plus_conv_to")
  {
  TWOWAY_TESTS(test_inplace_plus_conv_to);
  }



template<typename eT2, typename eT1>
void test_inplace_minus_conv_to()
  {
  Mat<eT1> x(5, 3);
  x.fill(eT1(1));

  Mat<eT2> y(5, 3);
  y.fill(eT2(3));

  y -= conv_to<Mat<eT2>>::from(x);

  REQUIRE( y.n_rows == 5 );
  REQUIRE( y.n_cols == 3 );
  for (uword i = 0; i < 15; ++i)
    {
    REQUIRE( eT2(y[i]) == eT2(2) );
    }
  }



TEST_CASE("inplace_minus_conv_to")
  {
  TWOWAY_TESTS(test_inplace_minus_conv_to);
  }



template<typename eT2, typename eT1>
void test_inplace_mul_conv_to()
  {
  Mat<eT1> x(5, 3);
  x.fill(eT1(3));

  Mat<eT2> y(5, 3);
  y.fill(eT2(4));

  y %= conv_to<Mat<eT2>>::from(x);

  REQUIRE( y.n_rows == 5 );
  REQUIRE( y.n_cols == 3 );
  for (uword i = 0; i < 15; ++i)
    {
    REQUIRE( eT2(y[i]) == eT2(12) );
    }
  }



TEST_CASE("inplace_mul_conv_to")
  {
  TWOWAY_TESTS(test_inplace_mul_conv_to);
  }



template<typename eT2, typename eT1>
void test_inplace_div_conv_to()
  {
  Mat<eT1> x(5, 3);
  x.fill(eT1(2));

  Mat<eT2> y(5, 3);
  y.fill(eT2(6));

  y /= conv_to<Mat<eT2>>::from(x);

  REQUIRE( y.n_rows == 5 );
  REQUIRE( y.n_cols == 3 );
  for (uword i = 0; i < 15; ++i)
    {
    REQUIRE( eT2(y[i]) == eT2(3) );
    }
  }



TEST_CASE("inplace_div_conv_to")
  {
  TWOWAY_TESTS(test_inplace_div_conv_to);
  }



// inplace_plus/minus/mul/div_array
template<typename eT2, typename eT1>
void test_inplace_plus_subview_conv_to()
  {
  Mat<eT1> x(5, 3);
  x.fill(eT1(1));
  Mat<eT2> y(6, 4);
  y.fill(eT1(2));

  y.submat(0, 0, 4, 2) += conv_to<Mat<eT2>>::from(x);

  for (uword c = 0; c < 3; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT2(y(r, c)) == eT2(3) );
      }

    REQUIRE( eT2(y(5, c)) == eT2(2) );
    }

  for (uword r = 0; r < 6; ++r)
    {
    REQUIRE( eT2(y(r, 3)) == eT2(2) );
    }
  }



TEST_CASE("inplace_plus_subview_conv_to")
  {
  TWOWAY_TESTS(test_inplace_plus_subview_conv_to)
  }



template<typename eT2, typename eT1>
void test_inplace_minus_subview_conv_to()
  {
  Mat<eT1> x(5, 3);
  x.fill(eT1(1));
  Mat<eT2> y(6, 4);
  y.fill(eT2(2));

  y.submat(0, 0, 4, 2) -= conv_to<Mat<eT2>>::from(x);

  for (uword c = 0; c < 3; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT2(y(r, c)) == eT2(1) );
      }

    REQUIRE( eT2(y(5, c)) == eT2(2) );
    }

  for (uword r = 0; r < 6; ++r)
    {
    REQUIRE( eT2(y(r, 3)) == eT2(2) );
    }
  }



TEST_CASE("inplace_minus_subview_conv_to")
  {
  TWOWAY_TESTS(test_inplace_minus_subview_conv_to)
  }



template<typename eT2, typename eT1>
void test_inplace_mul_subview_conv_to()
  {
  Mat<eT1> x(5, 3);
  x.fill(eT1(3));
  Mat<eT2> y(6, 4);
  y.fill(eT2(2));

  y.submat(0, 0, 4, 2) %= conv_to<Mat<eT2>>::from(x);

  for (uword c = 0; c < 3; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT2(y(r, c)) == eT2(6) );
      }

    REQUIRE( eT2(y(5, c)) == eT2(2) );
    }

  for (uword r = 0; r < 6; ++r)
    {
    REQUIRE( eT2(y(r, 3)) == eT2(2) );
    }
  }



TEST_CASE("inplace_mul_subview_conv_to")
  {
  TWOWAY_TESTS(test_inplace_mul_subview_conv_to)
  }



template<typename eT2, typename eT1>
void test_inplace_div_subview_conv_to()
  {
  Mat<eT1> x(5, 3);
  x.fill(eT1(2));
  Mat<eT2> y(6, 4);
  y.fill(eT2(6));

  y.submat(0, 0, 4, 2) /= conv_to<Mat<eT2>>::from(x);

  for (uword c = 0; c < 3; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT2(y(r, c)) == eT2(3) );
      }

    REQUIRE( eT2(y(5, c)) == eT2(6) );
    }

  for (uword r = 0; r < 6; ++r)
    {
    REQUIRE( eT2(y(r, 3)) == eT2(6) );
    }
  }



TEST_CASE("inplace_div_subview_conv_to")
  {
  TWOWAY_TESTS(test_inplace_div_subview_conv_to)
  }



