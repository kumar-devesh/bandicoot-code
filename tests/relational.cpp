// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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

//
// scalar/matrix tests
//

TEMPLATE_TEST_CASE("simple_relational_scalar_ops", "[relational]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(3, 3);
  x(0, 0) = eT(1);
  x(0, 1) = eT(2);
  x(0, 2) = eT(3);
  x(1, 0) = eT(4);
  x(1, 1) = eT(5);
  x(1, 2) = eT(6);
  x(2, 0) = eT(7);
  x(2, 1) = eT(8);
  x(2, 2) = eT(9);

  umat y1 = x < 5;
  umat y2 = x <= 5;
  umat y3 = x > 5;
  umat y4 = x >= 5;
  umat y5 = x == 5;
  umat y6 = x != 5;

  REQUIRE( y1.n_rows == 3 );
  REQUIRE( y1.n_cols == 3 );
  REQUIRE( y2.n_rows == 3 );
  REQUIRE( y2.n_cols == 3 );
  REQUIRE( y3.n_rows == 3 );
  REQUIRE( y3.n_cols == 3 );
  REQUIRE( y4.n_rows == 3 );
  REQUIRE( y4.n_cols == 3 );
  REQUIRE( y5.n_rows == 3 );
  REQUIRE( y5.n_cols == 3 );
  REQUIRE( y6.n_rows == 3 );
  REQUIRE( y6.n_cols == 3 );

  // x[0, 0] = 1
  REQUIRE( uword(y1(0, 0)) == 1 );  // <  5
  REQUIRE( uword(y2(0, 0)) == 1 );  // <= 5
  REQUIRE( uword(y3(0, 0)) == 0 );  // >  5
  REQUIRE( uword(y4(0, 0)) == 0 );  // >= 5
  REQUIRE( uword(y5(0, 0)) == 0 );  // == 5
  REQUIRE( uword(y6(0, 0)) == 1 );  // != 5
  // x(0, 1) = 2
  REQUIRE( uword(y1(0, 1)) == 1 );  // <  5
  REQUIRE( uword(y2(0, 1)) == 1 );  // <= 5
  REQUIRE( uword(y3(0, 1)) == 0 );  // >  5
  REQUIRE( uword(y4(0, 1)) == 0 );  // >= 5
  REQUIRE( uword(y5(0, 1)) == 0 );  // == 5
  REQUIRE( uword(y6(0, 1)) == 1 );  // != 5
  // x(0, 2) = 3
  REQUIRE( uword(y1(0, 2)) == 1 );  // <  5
  REQUIRE( uword(y2(0, 2)) == 1 );  // <= 5
  REQUIRE( uword(y3(0, 2)) == 0 );  // >  5
  REQUIRE( uword(y4(0, 2)) == 0 );  // >= 5
  REQUIRE( uword(y5(0, 2)) == 0 );  // == 5
  REQUIRE( uword(y6(0, 2)) == 1 );  // != 5
  // x(1, 0) = 4
  REQUIRE( uword(y1(1, 0)) == 1 );  // <  5
  REQUIRE( uword(y2(1, 0)) == 1 );  // <= 5
  REQUIRE( uword(y3(1, 0)) == 0 );  // >  5
  REQUIRE( uword(y4(1, 0)) == 0 );  // >= 5
  REQUIRE( uword(y5(1, 0)) == 0 );  // == 5
  REQUIRE( uword(y6(1, 0)) == 1 );  // != 5
  // x(1, 1) = 5
  REQUIRE( uword(y1(1, 1)) == 0 );  // <  5
  REQUIRE( uword(y2(1, 1)) == 1 );  // <= 5
  REQUIRE( uword(y3(1, 1)) == 0 );  // >  5
  REQUIRE( uword(y4(1, 1)) == 1 );  // >= 5
  REQUIRE( uword(y5(1, 1)) == 1 );  // == 5
  REQUIRE( uword(y6(1, 1)) == 0 );  // != 5
  // x(1, 2) = 6
  REQUIRE( uword(y1(1, 2)) == 0 );  // <  5
  REQUIRE( uword(y2(1, 2)) == 0 );  // <= 5
  REQUIRE( uword(y3(1, 2)) == 1 );  // >  5
  REQUIRE( uword(y4(1, 2)) == 1 );  // >= 5
  REQUIRE( uword(y5(1, 2)) == 0 );  // == 5
  REQUIRE( uword(y6(1, 2)) == 1 );  // != 5
  // x(2, 0) = 7
  REQUIRE( uword(y1(2, 0)) == 0 );  // <  5
  REQUIRE( uword(y2(2, 0)) == 0 );  // <= 5
  REQUIRE( uword(y3(2, 0)) == 1 );  // >  5
  REQUIRE( uword(y4(2, 0)) == 1 );  // >= 5
  REQUIRE( uword(y5(2, 0)) == 0 );  // == 5
  REQUIRE( uword(y6(2, 0)) == 1 );  // != 5
  // x(2, 1) = 8
  REQUIRE( uword(y1(2, 1)) == 0 );  // <  5
  REQUIRE( uword(y2(2, 1)) == 0 );  // <= 5
  REQUIRE( uword(y3(2, 1)) == 1 );  // >  5
  REQUIRE( uword(y4(2, 1)) == 1 );  // >= 5
  REQUIRE( uword(y5(2, 1)) == 0 );  // == 5
  REQUIRE( uword(y6(2, 1)) == 1 );  // != 5
  // x[2, 2] = 9
  REQUIRE( uword(y1(2, 2)) == 0 );  // <  5
  REQUIRE( uword(y2(2, 2)) == 0 );  // <= 5
  REQUIRE( uword(y3(2, 2)) == 1 );  // >  5
  REQUIRE( uword(y4(2, 2)) == 1 );  // >= 5
  REQUIRE( uword(y5(2, 2)) == 0 );  // == 5
  REQUIRE( uword(y6(2, 2)) == 1 );  // != 5

  // Now do the same with the scalar coming first.
  y1 = 5 > x;
  y2 = 5 >= x;
  y3 = 5 < x;
  y4 = 5 <= x;
  y5 = 5 == x;
  y6 = 5 != x;

  // x[0, 0] = 1
  REQUIRE( uword(y1(0, 0)) == 1 );  // <  5
  REQUIRE( uword(y2(0, 0)) == 1 );  // <= 5
  REQUIRE( uword(y3(0, 0)) == 0 );  // >  5
  REQUIRE( uword(y4(0, 0)) == 0 );  // >= 5
  REQUIRE( uword(y5(0, 0)) == 0 );  // == 5
  REQUIRE( uword(y6(0, 0)) == 1 );  // != 5
  // x(0, 1) = 2
  REQUIRE( uword(y1(0, 1)) == 1 );  // <  5
  REQUIRE( uword(y2(0, 1)) == 1 );  // <= 5
  REQUIRE( uword(y3(0, 1)) == 0 );  // >  5
  REQUIRE( uword(y4(0, 1)) == 0 );  // >= 5
  REQUIRE( uword(y5(0, 1)) == 0 );  // == 5
  REQUIRE( uword(y6(0, 1)) == 1 );  // != 5
  // x(0, 2) = 3
  REQUIRE( uword(y1(0, 2)) == 1 );  // <  5
  REQUIRE( uword(y2(0, 2)) == 1 );  // <= 5
  REQUIRE( uword(y3(0, 2)) == 0 );  // >  5
  REQUIRE( uword(y4(0, 2)) == 0 );  // >= 5
  REQUIRE( uword(y5(0, 2)) == 0 );  // == 5
  REQUIRE( uword(y6(0, 2)) == 1 );  // != 5
  // x(1, 0) = 4
  REQUIRE( uword(y1(1, 0)) == 1 );  // <  5
  REQUIRE( uword(y2(1, 0)) == 1 );  // <= 5
  REQUIRE( uword(y3(1, 0)) == 0 );  // >  5
  REQUIRE( uword(y4(1, 0)) == 0 );  // >= 5
  REQUIRE( uword(y5(1, 0)) == 0 );  // == 5
  REQUIRE( uword(y6(1, 0)) == 1 );  // != 5
  // x(1, 1) = 5
  REQUIRE( uword(y1(1, 1)) == 0 );  // <  5
  REQUIRE( uword(y2(1, 1)) == 1 );  // <= 5
  REQUIRE( uword(y3(1, 1)) == 0 );  // >  5
  REQUIRE( uword(y4(1, 1)) == 1 );  // >= 5
  REQUIRE( uword(y5(1, 1)) == 1 );  // == 5
  REQUIRE( uword(y6(1, 1)) == 0 );  // != 5
  // x(1, 2) = 6
  REQUIRE( uword(y1(1, 2)) == 0 );  // <  5
  REQUIRE( uword(y2(1, 2)) == 0 );  // <= 5
  REQUIRE( uword(y3(1, 2)) == 1 );  // >  5
  REQUIRE( uword(y4(1, 2)) == 1 );  // >= 5
  REQUIRE( uword(y5(1, 2)) == 0 );  // == 5
  REQUIRE( uword(y6(1, 2)) == 1 );  // != 5
  // x(2, 0) = 7
  REQUIRE( uword(y1(2, 0)) == 0 );  // <  5
  REQUIRE( uword(y2(2, 0)) == 0 );  // <= 5
  REQUIRE( uword(y3(2, 0)) == 1 );  // >  5
  REQUIRE( uword(y4(2, 0)) == 1 );  // >= 5
  REQUIRE( uword(y5(2, 0)) == 0 );  // == 5
  REQUIRE( uword(y6(2, 0)) == 1 );  // != 5
  // x(2, 1) = 8
  REQUIRE( uword(y1(2, 1)) == 0 );  // <  5
  REQUIRE( uword(y2(2, 1)) == 0 );  // <= 5
  REQUIRE( uword(y3(2, 1)) == 1 );  // >  5
  REQUIRE( uword(y4(2, 1)) == 1 );  // >= 5
  REQUIRE( uword(y5(2, 1)) == 0 );  // == 5
  REQUIRE( uword(y6(2, 1)) == 1 );  // != 5
  // x[2, 2] = 9
  REQUIRE( uword(y1(2, 2)) == 0 );  // <  5
  REQUIRE( uword(y2(2, 2)) == 0 );  // <= 5
  REQUIRE( uword(y3(2, 2)) == 1 );  // >  5
  REQUIRE( uword(y4(2, 2)) == 1 );  // >= 5
  REQUIRE( uword(y5(2, 2)) == 0 );  // == 5
  REQUIRE( uword(y6(2, 2)) == 1 );  // != 5
  }



TEMPLATE_TEST_CASE("large_relational_scalar_test", "[relational]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(100000, distr_param(1, 50));

  Col<uword> y1 = x > 25;
  Col<uword> y2 = 25 < x;
  Col<uword> y3 = x >= 25;
  Col<uword> y4 = 25 <= x;
  Col<uword> y5 = x < 25;
  Col<uword> y6 = 25 > x;
  Col<uword> y7 = x <= 25;
  Col<uword> y8 = 25 >= x;
  Col<uword> y9 = x == 25;
  Col<uword> y10 = 25 == x;
  Col<uword> y11 = x != 25;
  Col<uword> y12 = 25 != x;

  REQUIRE( y1.n_rows == 100000 );
  REQUIRE( y1.n_cols == 1 );
  REQUIRE( y1.n_elem == 100000 );
  REQUIRE( y2.n_rows == 100000 );
  REQUIRE( y2.n_cols == 1 );
  REQUIRE( y2.n_elem == 100000 );
  REQUIRE( y3.n_rows == 100000 );
  REQUIRE( y3.n_cols == 1 );
  REQUIRE( y3.n_elem == 100000 );
  REQUIRE( y4.n_rows == 100000 );
  REQUIRE( y4.n_cols == 1 );
  REQUIRE( y4.n_elem == 100000 );
  REQUIRE( y5.n_rows == 100000 );
  REQUIRE( y5.n_cols == 1 );
  REQUIRE( y5.n_elem == 100000 );
  REQUIRE( y6.n_rows == 100000 );
  REQUIRE( y6.n_cols == 1 );
  REQUIRE( y6.n_elem == 100000 );
  REQUIRE( y7.n_rows == 100000 );
  REQUIRE( y7.n_cols == 1 );
  REQUIRE( y7.n_elem == 100000 );
  REQUIRE( y8.n_rows == 100000 );
  REQUIRE( y8.n_cols == 1 );
  REQUIRE( y8.n_elem == 100000 );
  REQUIRE( y9.n_rows == 100000 );
  REQUIRE( y9.n_cols == 1 );
  REQUIRE( y9.n_elem == 100000 );
  REQUIRE( y10.n_rows == 100000 );
  REQUIRE( y10.n_cols == 1 );
  REQUIRE( y10.n_elem == 100000 );
  REQUIRE( y11.n_rows == 100000 );
  REQUIRE( y11.n_cols == 1 );
  REQUIRE( y11.n_elem == 100000 );
  REQUIRE( y12.n_rows == 100000 );
  REQUIRE( y12.n_cols == 1 );
  REQUIRE( y12.n_elem == 100000 );

  arma::Col<eT> x_cpu(x);
  arma::Col<uword> y1_cpu(y1);
  arma::Col<uword> y2_cpu(y2);
  arma::Col<uword> y3_cpu(y3);
  arma::Col<uword> y4_cpu(y4);
  arma::Col<uword> y5_cpu(y5);
  arma::Col<uword> y6_cpu(y6);
  arma::Col<uword> y7_cpu(y7);
  arma::Col<uword> y8_cpu(y8);
  arma::Col<uword> y9_cpu(y9);
  arma::Col<uword> y10_cpu(y10);
  arma::Col<uword> y11_cpu(y11);
  arma::Col<uword> y12_cpu(y12);

  for (uword i = 0; i < 100000; ++i)
    {
    REQUIRE( ((x_cpu[i] >  eT(25)) ? (y1_cpu[i] == 1) : (y1_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] >  eT(25)) ? (y2_cpu[i] == 1) : (y2_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] >= eT(25)) ? (y3_cpu[i] == 1) : (y3_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] >= eT(25)) ? (y4_cpu[i] == 1) : (y4_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] <  eT(25)) ? (y5_cpu[i] == 1) : (y5_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] <  eT(25)) ? (y6_cpu[i] == 1) : (y6_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] <= eT(25)) ? (y7_cpu[i] == 1) : (y7_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] <= eT(25)) ? (y8_cpu[i] == 1) : (y8_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] == eT(25)) ? (y9_cpu[i] == 1) : (y9_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] == eT(25)) ? (y10_cpu[i] == 1) : (y10_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] != eT(25)) ? (y11_cpu[i] == 1) : (y11_cpu[i] == 0)) );
    REQUIRE( ((x_cpu[i] != eT(25)) ? (y12_cpu[i] == 1) : (y12_cpu[i] == 0)) );
    }
  }



TEST_CASE("alias_scalar_relational_test", "[relational]")
  {
  umat x = randi<umat>(10, 10, distr_param(0, 10));
  umat x_old(x);

  x = (x > 5);

  REQUIRE( x.n_rows == 10 );
  REQUIRE( x.n_cols == 10 );
  REQUIRE( x.n_elem == 100 );

  arma::Mat<uword> x_cpu(x);
  arma::Mat<uword> x_old_cpu(x_old);

  for (uword i = 0; i < 100; ++i)
    {
    if (x_old_cpu[i] > 5)
      {
      REQUIRE( x_cpu[i] == 1 );
      }
    else
      {
      REQUIRE( x_cpu[i] == 0 );
      }
    }
  }



// empty tests
TEST_CASE("empty_relational_test", "[relational]")
  {
  fmat x;

  umat y1 = (x > 3);
  umat y2 = (3 < x);
  umat y3 = (x >= 3);
  umat y4 = (3 <= x);
  umat y5 = (x < 3);
  umat y6 = (3 > x);
  umat y7 = (x <= 3);
  umat y8 = (3 >= x);
  umat y9 = (x == 3);
  umat y10 = (3 == x);
  umat y11 = (x != 3);
  umat y12 = (3 != x);

  REQUIRE( y1.n_rows == 0 );
  REQUIRE( y1.n_cols == 0 );
  REQUIRE( y1.n_elem == 0 );
  REQUIRE( y2.n_rows == 0 );
  REQUIRE( y2.n_cols == 0 );
  REQUIRE( y2.n_elem == 0 );
  REQUIRE( y3.n_rows == 0 );
  REQUIRE( y3.n_cols == 0 );
  REQUIRE( y3.n_elem == 0 );
  REQUIRE( y4.n_rows == 0 );
  REQUIRE( y4.n_cols == 0 );
  REQUIRE( y4.n_elem == 0 );
  REQUIRE( y5.n_rows == 0 );
  REQUIRE( y5.n_cols == 0 );
  REQUIRE( y5.n_elem == 0 );
  REQUIRE( y6.n_rows == 0 );
  REQUIRE( y6.n_cols == 0 );
  REQUIRE( y6.n_elem == 0 );
  REQUIRE( y7.n_rows == 0 );
  REQUIRE( y7.n_cols == 0 );
  REQUIRE( y7.n_elem == 0 );
  REQUIRE( y8.n_rows == 0 );
  REQUIRE( y8.n_cols == 0 );
  REQUIRE( y8.n_elem == 0 );
  REQUIRE( y9.n_rows == 0 );
  REQUIRE( y9.n_cols == 0 );
  REQUIRE( y9.n_elem == 0 );
  REQUIRE( y10.n_rows == 0 );
  REQUIRE( y10.n_cols == 0 );
  REQUIRE( y10.n_elem == 0 );
  REQUIRE( y11.n_rows == 0 );
  REQUIRE( y11.n_cols == 0 );
  REQUIRE( y11.n_elem == 0 );
  REQUIRE( y12.n_rows == 0 );
  REQUIRE( y12.n_cols == 0 );
  REQUIRE( y12.n_elem == 0 );
  }



TEMPLATE_TEST_CASE(
  "relational_conv_to_scalar",
  "[relational]",
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

  Mat<eT1> x = randi<Mat<eT1>>(10, 10, distr_param(1, 50));
  Mat<eT2> x_conv = conv_to<Mat<eT2>>::from(x);

  umat y = conv_to<Mat<eT2>>::from(x) < eT2(25);
  umat y_ref = x_conv < eT2(25);

  arma::Mat<uword> y_cpu(y);
  arma::Mat<uword> y_ref_cpu(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = eT2(25) < conv_to<Mat<eT2>>::from(x);
  y_ref = eT2(25) < x_conv;

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = conv_to<Mat<eT2>>::from(x) > eT2(25);
  y_ref = x_conv > eT2(25);

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = eT2(25) > conv_to<Mat<eT2>>::from(x);
  y_ref = eT2(25) > x_conv;

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = conv_to<Mat<eT2>>::from(x) <= eT2(25);
  y_ref = x_conv <= eT2(25);

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = eT2(25) <= conv_to<Mat<eT2>>::from(x);
  y_ref = eT2(25) <= x_conv;

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = conv_to<Mat<eT2>>::from(x) >= eT2(25);
  y_ref = x_conv >= eT2(25);

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = eT2(25) >= conv_to<Mat<eT2>>::from(x);
  y_ref = eT2(25) >= x_conv;

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = conv_to<Mat<eT2>>::from(x) == eT2(25);
  y_ref = x_conv == eT2(25);

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = eT2(25) == conv_to<Mat<eT2>>::from(x);
  y_ref = eT2(25) == x_conv;

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = conv_to<Mat<eT2>>::from(x) != eT2(25);
  y_ref = x_conv != eT2(25);

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );

  y = eT2(25) != conv_to<Mat<eT2>>::from(x);
  y_ref = eT2(25) != x_conv;

  y_cpu = arma::Mat<uword>(y);
  y_ref_cpu = arma::Mat<uword>(y_ref);

  REQUIRE( arma::all( arma::all( y_cpu == y_ref_cpu ) ) );
  }



//
// relational array operations
//



TEMPLATE_TEST_CASE("simple_array_relational_op", "[relational]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X(3, 3);
  Mat<eT> Y(3, 3);

  for (uword i = 0; i < 9; ++i)
    {
    X[i] = eT(i);
    Y[i] = eT(8 - i);
    }

  umat z1 = X < Y;
  umat z2 = Y < X;
  umat z3 = X > Y;
  umat z4 = Y > X;
  umat z5 = X <= Y;
  umat z6 = Y <= X;
  umat z7 = X >= Y;
  umat z8 = Y >= X;
  umat z9 = X == Y;
  umat z10 = Y == X;
  umat z11 = X != Y;
  umat z12 = Y != X;
  umat z13 = X && Y;
  umat z14 = Y && X;
  umat z15 = X || Y;
  umat z16 = Y || X;

  REQUIRE( z1.n_rows == 3 );
  REQUIRE( z1.n_cols == 3 );
  REQUIRE( z1.n_elem == 9 );
  REQUIRE( z2.n_rows == 3 );
  REQUIRE( z2.n_cols == 3 );
  REQUIRE( z2.n_elem == 9 );
  REQUIRE( z3.n_rows == 3 );
  REQUIRE( z3.n_cols == 3 );
  REQUIRE( z3.n_elem == 9 );
  REQUIRE( z4.n_rows == 3 );
  REQUIRE( z4.n_cols == 3 );
  REQUIRE( z4.n_elem == 9 );
  REQUIRE( z5.n_rows == 3 );
  REQUIRE( z5.n_cols == 3 );
  REQUIRE( z5.n_elem == 9 );
  REQUIRE( z6.n_rows == 3 );
  REQUIRE( z6.n_cols == 3 );
  REQUIRE( z6.n_elem == 9 );
  REQUIRE( z7.n_rows == 3 );
  REQUIRE( z7.n_cols == 3 );
  REQUIRE( z7.n_elem == 9 );
  REQUIRE( z8.n_rows == 3 );
  REQUIRE( z8.n_cols == 3 );
  REQUIRE( z8.n_elem == 9 );
  REQUIRE( z9.n_rows == 3 );
  REQUIRE( z9.n_cols == 3 );
  REQUIRE( z9.n_elem == 9 );
  REQUIRE( z10.n_rows == 3 );
  REQUIRE( z10.n_cols == 3 );
  REQUIRE( z10.n_elem == 9 );
  REQUIRE( z11.n_rows == 3 );
  REQUIRE( z11.n_cols == 3 );
  REQUIRE( z11.n_elem == 9 );
  REQUIRE( z12.n_rows == 3 );
  REQUIRE( z12.n_cols == 3 );
  REQUIRE( z12.n_elem == 9 );
  REQUIRE( z13.n_rows == 3 );
  REQUIRE( z13.n_cols == 3 );
  REQUIRE( z13.n_elem == 9 );
  REQUIRE( z14.n_rows == 3 );
  REQUIRE( z14.n_cols == 3 );
  REQUIRE( z14.n_elem == 9 );
  REQUIRE( z15.n_rows == 3 );
  REQUIRE( z15.n_cols == 3 );
  REQUIRE( z15.n_elem == 9 );
  REQUIRE( z16.n_rows == 3 );
  REQUIRE( z16.n_cols == 3 );
  REQUIRE( z16.n_elem == 9 );

  for (uword i = 0; i < 9; ++i)
    {
    REQUIRE( uword(z1[i]) == (X[i] < Y[i]) );
    REQUIRE( uword(z2[i]) == (Y[i] < X[i]) );
    REQUIRE( uword(z3[i]) == (X[i] > Y[i]) );
    REQUIRE( uword(z4[i]) == (Y[i] > X[i]) );
    REQUIRE( uword(z5[i]) == (X[i] <= Y[i]) );
    REQUIRE( uword(z6[i]) == (Y[i] <= X[i]) );
    REQUIRE( uword(z7[i]) == (X[i] >= Y[i]) );
    REQUIRE( uword(z8[i]) == (Y[i] >= X[i]) );
    REQUIRE( uword(z9[i]) == (X[i] == Y[i]) );
    REQUIRE( uword(z10[i]) == (Y[i] == X[i]) );
    REQUIRE( uword(z11[i]) == (X[i] != Y[i]) );
    REQUIRE( uword(z12[i]) == (Y[i] != X[i]) );
    REQUIRE( uword(z13[i]) == (X[i] && Y[i]) );
    REQUIRE( uword(z14[i]) == (Y[i] && X[i]) );
    REQUIRE( uword(z15[i]) == (X[i] || Y[i]) );
    REQUIRE( uword(z16[i]) == (Y[i] || X[i]) );
    }
  }



// large array test

TEMPLATE_TEST_CASE("large_relational_array_test", "[relational]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X = randi<Mat<eT>>(100, 100, distr_param(1, 100));
  Mat<eT> Y = randi<Mat<eT>>(100, 100, distr_param(1, 100));

  arma::Mat<eT> X_cpu(X);
  arma::Mat<eT> Y_cpu(Y);

  umat z = X < Y;
  arma::Mat<uword> z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(X_cpu < Y_cpu);
  arma::Mat<uword> z_cpu(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = Y < X;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(Y_cpu < X_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = X > Y;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(X_cpu > Y_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = Y > X;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(Y_cpu > X_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = X <= Y;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(X_cpu <= Y_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = Y <= X;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(Y_cpu <= X_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = X >= Y;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(X_cpu >= Y_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = Y >= X;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(Y_cpu >= X_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = X == Y;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(X_cpu == Y_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = Y == X;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(Y_cpu == X_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = X != Y;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(X_cpu != Y_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = Y != X;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(Y_cpu != X_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = X && Y;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(X_cpu && Y_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = Y && X;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(Y_cpu && X_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = X || Y;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(X_cpu || Y_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );

  z = Y || X;
  z_cpu_ref = arma::conv_to<arma::Mat<uword>>::from(Y_cpu || X_cpu);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( z_cpu == z_cpu_ref ) ) );
  }



TEST_CASE("empty_relational_array_op", "[relational]")
  {
  fmat x;
  fmat y;

  umat z1 = x < y;
  umat z2 = y < x;
  umat z3 = x > y;
  umat z4 = y > x;
  umat z5 = x <= y;
  umat z6 = y <= x;
  umat z7 = x >= y;
  umat z8 = y >= x;
  umat z9 = x == y;
  umat z10 = y == x;
  umat z11 = x != y;
  umat z12 = y != x;
  umat z13 = x && y;
  umat z14 = y && x;
  umat z15 = x || y;
  umat z16 = y || x;

  REQUIRE( z1.n_rows == 0 );
  REQUIRE( z1.n_cols == 0 );
  REQUIRE( z1.n_elem == 0 );

  REQUIRE( z2.n_rows == 0 );
  REQUIRE( z2.n_cols == 0 );
  REQUIRE( z2.n_elem == 0 );

  REQUIRE( z3.n_rows == 0 );
  REQUIRE( z3.n_cols == 0 );
  REQUIRE( z3.n_elem == 0 );

  REQUIRE( z4.n_rows == 0 );
  REQUIRE( z4.n_cols == 0 );
  REQUIRE( z4.n_elem == 0 );

  REQUIRE( z5.n_rows == 0 );
  REQUIRE( z5.n_cols == 0 );
  REQUIRE( z5.n_elem == 0 );

  REQUIRE( z6.n_rows == 0 );
  REQUIRE( z6.n_cols == 0 );
  REQUIRE( z6.n_elem == 0 );

  REQUIRE( z7.n_rows == 0 );
  REQUIRE( z7.n_cols == 0 );
  REQUIRE( z7.n_elem == 0 );

  REQUIRE( z8.n_rows == 0 );
  REQUIRE( z8.n_cols == 0 );
  REQUIRE( z8.n_elem == 0 );

  REQUIRE( z9.n_rows == 0 );
  REQUIRE( z9.n_cols == 0 );
  REQUIRE( z9.n_elem == 0 );

  REQUIRE( z10.n_rows == 0 );
  REQUIRE( z10.n_cols == 0 );
  REQUIRE( z10.n_elem == 0 );

  REQUIRE( z11.n_rows == 0 );
  REQUIRE( z11.n_cols == 0 );
  REQUIRE( z11.n_elem == 0 );

  REQUIRE( z12.n_rows == 0 );
  REQUIRE( z12.n_cols == 0 );
  REQUIRE( z12.n_elem == 0 );

  REQUIRE( z13.n_rows == 0 );
  REQUIRE( z13.n_cols == 0 );
  REQUIRE( z13.n_elem == 0 );

  REQUIRE( z14.n_rows == 0 );
  REQUIRE( z14.n_cols == 0 );
  REQUIRE( z14.n_elem == 0 );

  REQUIRE( z15.n_rows == 0 );
  REQUIRE( z15.n_cols == 0 );
  REQUIRE( z15.n_elem == 0 );

  REQUIRE( z16.n_rows == 0 );
  REQUIRE( z16.n_cols == 0 );
  REQUIRE( z16.n_elem == 0 );
  }



TEST_CASE("alias_relational_array_op", "[relational]")
  {
  umat x = randi<umat>(10, 10, distr_param(1, 20));
  umat y = randi<umat>(10, 10, distr_param(1, 20));

  umat x_orig(x);
  umat y_orig(y);

  x = x < y;
  umat z = x_orig < y_orig;

  REQUIRE( x.n_rows == 10 );
  REQUIRE( x.n_cols == 10 );
  REQUIRE( x.n_elem == 100 );

  arma::Mat<uword> x_cpu(x);
  arma::Mat<uword> z_cpu(z);

  REQUIRE( arma::all( arma::all( x_cpu == z_cpu ) ) );

  x = x_orig;
  y = x > y;

  z = x_orig > y_orig;

  REQUIRE( y.n_rows == 10 );
  REQUIRE( y.n_cols == 10 );
  REQUIRE( y.n_elem == 100 );

  arma::Mat<uword> y_cpu(y);
  z_cpu = arma::Mat<uword>(z);

  REQUIRE( arma::all( arma::all( y_cpu == z_cpu ) ) );
  }
