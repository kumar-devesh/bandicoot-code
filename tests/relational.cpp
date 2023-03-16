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

#include <bandicoot>
#include "catch.hpp"

using namespace coot;

//
// scalar/matrix tests
//

TEMPLATE_TEST_CASE("simple_relational_scalar_ops", "[relational]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

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
  mat x;

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

// conv_to tests
