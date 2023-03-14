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

// create Mat, Row, Col using SizeMat
TEST_CASE("sizemat_constructors", "[size]")
  {
  mat X(SizeMat(5, 10));
  vec C(SizeMat(10, 1));
  rowvec R(SizeMat(1, 10));

  REQUIRE( X.n_rows == 5 );
  REQUIRE( X.n_cols == 10 );
  REQUIRE( X.n_elem == 50 );

  REQUIRE( C.n_rows == 10 );
  REQUIRE( C.n_cols == 1 );
  REQUIRE( C.n_elem == 10 );

  REQUIRE( R.n_rows == 1 );
  REQUIRE( R.n_cols == 10 );
  REQUIRE( R.n_elem == 10 );
  }



// .set_size with SizeMat
TEST_CASE("mat_set_size_sizemat", "[size]")
  {
  mat X;
  X.zeros(20, 20);

  X.set_size(SizeMat(8, 9));

  REQUIRE( X.n_rows == 8 );
  REQUIRE( X.n_cols == 9 );
  REQUIRE( X.n_elem == 72 );
  }



// .resize with SizeMat
TEST_CASE("mat_resize_sizemat", "[size]")
  {
  mat X;
  X.ones(20, 20);

  X.set_size(SizeMat(30, 40));

  REQUIRE( X.n_rows == 30 );
  REQUIRE( X.n_cols == 40 );
  REQUIRE( X.n_elem == 1200 );
  }



// get subview using SizeMat
TEST_CASE("subview_using_sizemat", "[size]")
  {
  mat X = randi<mat>(30, 40, distr_param(20, 30));

  mat Y = X.submat(3, 3, SizeMat(10, 11));
  mat Y2 = X.submat(3, 3, 12, 13);

  REQUIRE( Y.n_rows == 10 );
  REQUIRE( Y.n_cols == 11 );
  REQUIRE( Y.n_elem == 110 );

  arma::mat Y_cpu(Y);
  arma::mat Y2_cpu(Y2);

  REQUIRE( arma::approx_equal( Y_cpu, Y2_cpu, "reldiff", 1e-5 ) );
  }



// get subvec using SizeMat
TEST_CASE("subvec_using_sizemat", "[size]")
  {
  vec C = randi<vec>(100, distr_param(10, 20));
  rowvec R = randi<rowvec>(100, distr_param(10, 20));

  vec C_sub = C.subvec(3, SizeMat(10, 1));
  vec C2_sub = C.subvec(3, 12);

  rowvec R_sub = R.subvec(3, SizeMat(1, 10));
  rowvec R2_sub = R.subvec(3, 12);

  REQUIRE( C_sub.n_rows == 10 );
  REQUIRE( C_sub.n_cols == 1 );
  REQUIRE( C_sub.n_elem == 10 );

  REQUIRE( R_sub.n_rows == 1 );
  REQUIRE( R_sub.n_cols == 10 );
  REQUIRE( R_sub.n_elem == 10 );

  arma::vec C_sub_cpu(C_sub);
  arma::vec C2_sub_cpu(C2_sub);
  arma::rowvec R_sub_cpu(R_sub);
  arma::rowvec R2_sub_cpu(R2_sub);

  REQUIRE( arma::approx_equal( C_sub_cpu, C2_sub_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( R_sub_cpu, R2_sub_cpu, "reldiff", 1e-5 ) );
  }



// add two SizeMats to make a Mat
TEST_CASE("sizemat_add_sizemat", "[size]")
  {
  SizeMat s1(10, 20);
  SizeMat s2(5, 10);

  SizeMat s3 = s1 + s2;

  REQUIRE( s3.n_rows == 15 );
  REQUIRE( s3.n_cols == 30 );
  }



// subtract two SizeMats to make a Mat
TEST_CASE("sizemat_minus_sizemat", "[size]")
  {
  SizeMat s1(10, 20);
  SizeMat s2(5, 10);

  SizeMat s3 = s1 - s2;

  REQUIRE( s3.n_rows == 5 );
  REQUIRE( s3.n_cols == 10 );

  SizeMat s4 = s3 - s1;

  REQUIRE( s4.n_rows == 0 );
  REQUIRE( s4.n_cols == 0 );
  }



// add scalar to SizeMat
TEST_CASE("sizemat_add_scalar", "[size]")
  {
  SizeMat s(10, 20);
  SizeMat s2 = s + 5;

  REQUIRE( s2.n_rows == 15 );
  REQUIRE( s2.n_cols == 25 );
  }



// subtract scalar from SizeMat
TEST_CASE("sizemat_minus_scalar", "[size]")
  {
  SizeMat s(10, 20);
  SizeMat s2 = s - 5;

  REQUIRE( s2.n_rows == 5 );
  REQUIRE( s2.n_cols == 15 );

  SizeMat s3 = s2 - 10;

  REQUIRE( s3.n_rows == 0 );
  REQUIRE( s3.n_cols == 5 );
  }



// multiply SizeMat by scalar
TEST_CASE("sizemat_times_scalar", "[size]")
  {
  SizeMat s(10, 20);
  SizeMat s2 = s * 3;

  REQUIRE( s2.n_rows == 30 );
  REQUIRE( s2.n_cols == 60 );
  }



// divide SizeMat by scalar
TEST_CASE("sizemat_div_scalar", "[size]")
  {
  SizeMat s(20, 40);
  SizeMat s2 = s / 2;

  REQUIRE( s2.n_rows == 10 );
  REQUIRE( s2.n_cols == 20 );
  }



// use SizeMat with resize
TEST_CASE("sizemat_resize", "[size]")
  {
  mat X(30, 30);
  X.ones();

  mat Y = resize(X, SizeMat(20, 15));

  REQUIRE( Y.n_rows == 20 );
  REQUIRE( Y.n_cols == 15 );
  REQUIRE( Y.n_elem == 300 );
  }



// use SizeMat with reshape
TEST_CASE("sizemat_reshape", "[size]")
  {
  mat X(30, 40);
  X.ones();

  mat Y = reshape(X, SizeMat(20, 15));

  REQUIRE( Y.n_rows == 20 );
  REQUIRE( Y.n_cols == 15 );
  REQUIRE( Y.n_elem == 300 );
  }



// create Mat using size of arbitrary object
TEST_CASE("mat_sizemat_object", "[size]")
  {
  mat X(30, 30);
  X.ones();

  mat Y(size(trans(repmat(X, 2, 3))));

  REQUIRE( Y.n_rows == 90 );
  REQUIRE( Y.n_cols == 60 );
  }



// randi/randn/randu with SizeMat
TEST_CASE("rand_sizemat", "[size]")
  {
  mat X = randu<mat>(SizeMat(10, 20));
  mat Y = randn<mat>(SizeMat(5, 8));
  mat Z = randi<mat>(SizeMat(20, 10), distr_param(10, 20));

  REQUIRE( X.n_rows == 10 );
  REQUIRE( X.n_cols == 20 );
  REQUIRE( X.n_elem == 200 );

  REQUIRE( Y.n_rows == 5 );
  REQUIRE( Y.n_cols == 8 );
  REQUIRE( Y.n_elem == 40 );

  REQUIRE( Z.n_rows == 20 );
  REQUIRE( Z.n_cols == 10 );
  REQUIRE( Z.n_elem == 200 );
  }



// max/min of SizeMat
TEST_CASE("sizemat_max_min", "[size]")
  {
  SizeMat s(20, 30);

  REQUIRE( max(s) == 30 );
  REQUIRE( min(s) == 20 );
  }
