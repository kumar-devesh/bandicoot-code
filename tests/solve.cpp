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



// trivial solve: diagonal matrix
TEMPLATE_TEST_CASE("trivial_diagonal_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(10, 10, fill::eye);
  Col<eT> B(10, fill::ones);

  Col<eT> X1 = solve(A, B);
  Col<eT> X2(10);
  const bool status = solve(X2, A, B);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == B.n_rows );
  REQUIRE( X1.n_cols == B.n_cols );
  REQUIRE( all( abs(X1 - 1) < 1e-5 ) );

  REQUIRE( X2.n_rows == B.n_rows );
  REQUIRE( X2.n_cols == B.n_cols );
  REQUIRE( all( abs(X2 - 1) < 1e-5 ) );
  }



// trivial large solve: diagonal matrix
TEMPLATE_TEST_CASE("trivial_large_diagonal_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(1000, 1000, fill::eye);
  Mat<eT> B(1000, 200, fill::ones);

  Mat<eT> X1 = solve(A, B);
  Mat<eT> X2;
  const bool status = solve(X2, A, B);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == B.n_rows );
  REQUIRE( X1.n_cols == B.n_cols );
  REQUIRE( all( all( abs(X1 - 1) < 1e-5 ) ) );

  REQUIRE( X2.n_rows == B.n_rows );
  REQUIRE( X2.n_cols == B.n_cols );
  REQUIRE( all( all( abs(X2 - 1) < 1e-5 ) ) );
  }



// solve hardcoded system
TEMPLATE_TEST_CASE("hardcoded_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  // [[11,  4,   7]
  //  [2,  15,   8]
  //  [3,   6,  19]]
  Mat<eT> A = reshape(linspace<Col<eT>>(1, 9, 9), 3, 3);
  A.diag() += 10; // force positive definiteness

  // [[1, 2, 3]
  //  [4, 5, 6]
  //  [7, 8, 9]]
  Mat<eT> B = reshape(linspace<Col<eT>>(1, 9, 9), 3, 3).t();

  Mat<eT> X1 = solve(A, B);
  Mat<eT> X2;
  const bool status = solve(X2, A, B);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == 3 );
  REQUIRE( X1.n_cols == 3 );

  REQUIRE( X2.n_rows == 3 );
  REQUIRE( X2.n_cols == 3 );

  // Computed by GNU Octave: A \ B
  REQUIRE( eT(X1(0, 0)) == Approx(eT(-0.17672413793)) );
  REQUIRE( eT(X1(1, 0)) == Approx(eT( 0.09482758621)) );
  REQUIRE( eT(X1(2, 0)) == Approx(eT( 0.36637931034)) );
  REQUIRE( eT(X1(0, 1)) == Approx(eT(-0.12068965517)) );
  REQUIRE( eT(X1(1, 1)) == Approx(eT( 0.13793103448)) );
  REQUIRE( eT(X1(2, 1)) == Approx(eT( 0.39655172414)) );
  REQUIRE( eT(X1(0, 2)) == Approx(eT(-0.06465517241)) );
  REQUIRE( eT(X1(1, 2)) == Approx(eT( 0.18103448276)) );
  REQUIRE( eT(X1(2, 2)) == Approx(eT( 0.42672413793)) );

  REQUIRE( eT(X2(0, 0)) == Approx(eT(-0.17672413793)) );
  REQUIRE( eT(X2(1, 0)) == Approx(eT( 0.09482758621)) );
  REQUIRE( eT(X2(2, 0)) == Approx(eT( 0.36637931034)) );
  REQUIRE( eT(X2(0, 1)) == Approx(eT(-0.12068965517)) );
  REQUIRE( eT(X2(1, 1)) == Approx(eT( 0.13793103448)) );
  REQUIRE( eT(X2(2, 1)) == Approx(eT( 0.39655172414)) );
  REQUIRE( eT(X2(0, 2)) == Approx(eT(-0.06465517241)) );
  REQUIRE( eT(X2(1, 2)) == Approx(eT( 0.18103448276)) );
  REQUIRE( eT(X2(2, 2)) == Approx(eT( 0.42672413793)) );
  }



// solve random square matrix
TEMPLATE_TEST_CASE("random_square_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(250, 250, fill::randu);
  A *= A.t();
  A.diag() += 3; // force positive definiteness

  Mat<eT> B(250, 500, fill::randn);

  Mat<eT> X1 = solve(A, B);
  Mat<eT> X2;
  const bool status = solve(X2, A, B);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == 250 );
  REQUIRE( X1.n_cols == 500 );
  REQUIRE( X2.n_rows == 250 );
  REQUIRE( X2.n_cols == 500 );

  Mat<eT> B1_rec = A * X1;
  Mat<eT> B2_rec = A * X2;

  const eT tol = (is_float<eT>::value) ? 1e-3 : 1e-6;
  REQUIRE( accu( abs(B - B1_rec) ) / B1_rec.n_elem < tol );
  REQUIRE( accu( abs(B - B2_rec) ) / B2_rec.n_elem < tol );
  }



// solve random square matrix with "fast" option
TEMPLATE_TEST_CASE("random_fast_square_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(250, 250, fill::randu);
  A *= A.t();
  A.diag() += 3; // force positive definiteness

  Mat<eT> B(250, 500, fill::randn);

  Mat<eT> X1 = solve(A, B, solve_opts::fast);
  Mat<eT> X2;
  const bool status = solve(X2, A, B, solve_opts::fast);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == 250 );
  REQUIRE( X1.n_cols == 500 );
  REQUIRE( X2.n_rows == 250 );
  REQUIRE( X2.n_cols == 500 );

  Mat<eT> B1_rec = A * X1;
  Mat<eT> B2_rec = A * X2;

  const eT tol = (is_float<eT>::value) ? 1e-3 : 1e-6;
  REQUIRE( accu( abs(B - B1_rec) ) / B1_rec.n_elem < tol );
  REQUIRE( accu( abs(B - B2_rec) ) / B2_rec.n_elem < tol );
  }



// solve transposed random square matrix
TEMPLATE_TEST_CASE("random_square_trans_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(456, 456, fill::randu);
  A.diag() += 5; // force positive definiteness

  Mat<eT> B(456, 100, fill::randn);

  Mat<eT> X1 = solve(A.t(), B);
  Mat<eT> X2;
  const bool status = solve(X2, A.t(), B);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == 456 );
  REQUIRE( X1.n_cols == 100 );
  REQUIRE( X2.n_rows == 456 );
  REQUIRE( X2.n_cols == 100 );

  Mat<eT> B1_rec = A.t() * X1;
  Mat<eT> B2_rec = A.t() * X2;

  const eT tol = (is_float<eT>::value) ? 1e-3 : 1e-6;
  REQUIRE( accu( abs(B - B1_rec) ) / B1_rec.n_elem < tol );
  REQUIRE( accu( abs(B - B2_rec) ) / B2_rec.n_elem < tol );
  }



// solve (a * A^T X = B)
TEMPLATE_TEST_CASE("random_square_htrans2_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(456, 456, fill::randu);
  A.diag() += 5; // force positive definiteness

  Mat<eT> B(456, 100, fill::randn);

  Mat<eT> X1 = solve(3 * A.t(), B);
  Mat<eT> X2;
  const bool status = solve(X2, 3 * A.t(), B);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == 456 );
  REQUIRE( X1.n_cols == 100 );
  REQUIRE( X2.n_rows == 456 );
  REQUIRE( X2.n_cols == 100 );

  Mat<eT> B1_rec = 3 * A.t() * X1;
  Mat<eT> B2_rec = 3 * A.t() * X2;

  const eT tol = (is_float<eT>::value) ? 1e-3 : 1e-6;
  REQUIRE( accu( abs(B - B1_rec) ) / B1_rec.n_elem < tol );
  REQUIRE( accu( abs(B - B2_rec) ) / B2_rec.n_elem < tol );
  }



// solve 1x1 matrix
TEMPLATE_TEST_CASE("1x1_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(1, 1, fill::ones);
  Mat<eT> B(1, 1, fill::ones);

  Mat<eT> X1 = solve(A, B);
  Mat<eT> X2;
  const bool status = solve(X2, A, B);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == 1 );
  REQUIRE( X1.n_cols == 1 );
  REQUIRE( eT(X1(0, 0)) == Approx(eT(1)) );

  REQUIRE( X2.n_rows == 1 );
  REQUIRE( X2.n_cols == 1 );
  REQUIRE( eT(X2(0, 0)) == Approx(eT(1)) );
  }


// solve empty matrix
TEST_CASE("empty_solve", "[solve]")
  {
  fmat A;
  fmat B;

  fmat X1 = solve(A, B);
  fmat X2;
  const bool status = solve(X2, A, B);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == 0 );
  REQUIRE( X1.n_cols == 0 );
  REQUIRE( X2.n_rows == 0 );
  REQUIRE( X2.n_cols == 0 );
  }



// solve expression
TEMPLATE_TEST_CASE("solve_expr", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(256, 256, fill::randu);
  Mat<eT> B(256, 50, fill::randn);

  Mat<eT> X1 = solve((A + 1) - (3 * A.t()) + 10 * diagmat(ones<Col<eT>>(256)), 2 * B + 1);
  Mat<eT> X2;
  const bool status = solve(X2, (A + 1) - (3 * A.t()) + 10 * diagmat(ones<Col<eT>>(256)), 2 * B + 1);

  REQUIRE( status == true );

  REQUIRE( X1.n_rows == 256 );
  REQUIRE( X1.n_cols == 50 );
  REQUIRE( X2.n_rows == 256 );
  REQUIRE( X2.n_cols == 50 );

  Mat<eT> A_ref = (A + 1) - (3 * A.t()) + 10 * diagmat(ones<Col<eT>>(256));
  Mat<eT> B_ref = 2 * B + 1;

  Mat<eT> X_ref = solve(A_ref, B_ref);

  REQUIRE( all( all( abs( X1 - X_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( X2 - X_ref ) < 1e-5 ) ) );
  }



// solve with conversion to another type
TEMPLATE_TEST_CASE
  (
  "conv_to_solve",
  "[solve]",
  (std::pair<float, double>),
  (std::pair<double, float>)
  )
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> A(100, 100, fill::randu);
  A.diag() += 3;
  Mat<eT1> B(100, 50, fill::randn);

  Mat<eT2> X = conv_to<Mat<eT2>>::from(solve(A, B));

  REQUIRE( X.n_rows == 100 );
  REQUIRE( X.n_cols == 50 );

  Mat<eT1> X_pre_conv = solve(A, B);
  Mat<eT2> X_ref = conv_to<Mat<eT2>>::from(X_pre_conv);

  REQUIRE( all( all( abs( X - X_ref ) < 1e-5 ) ) );
  }



// non-square should throw exception
TEST_CASE("nonsquare_solve_exception", "[solve]")
  {
  fmat A(250, 500, fill::randu);
  fmat B(250, 400, fill::randu);

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  fmat X;
  bool status;
  REQUIRE_THROWS( X = solve(A, B) );
  REQUIRE_THROWS( status = solve(X, A, B) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// solve where output is an alias of the input
TEMPLATE_TEST_CASE("output_input_alias_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(133, 133, fill::randu);
  A.diag() += 3;

  Mat<eT> B(133, 133, fill::randu);
  Mat<eT> B_orig(B);

  Mat<eT> X_ref = solve(A, B);
  B = solve(A, B);

  REQUIRE( B.n_rows == X_ref.n_rows );
  REQUIRE( B.n_cols == X_ref.n_cols );
  REQUIRE( all( all( abs( B - X_ref ) < 1e-5 ) ) );

  B = B_orig;
  const bool status = solve(B, A, B);

  REQUIRE( B.n_rows == X_ref.n_rows );
  REQUIRE( B.n_cols == X_ref.n_cols );
  REQUIRE( all( all( abs( B - X_ref ) < 1e-5 ) ) );
  }



// solve where output is an alias of the A input
TEMPLATE_TEST_CASE("output_A_alias_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(200, 200, fill::randu);
  A.diag() += 3;

  Mat<eT> A_orig(A);

  Mat<eT> B(200, 200, fill::randu);

  Mat<eT> X_ref = solve(A, B);
  A = solve(A, B);

  REQUIRE( A.n_rows == X_ref.n_rows );
  REQUIRE( A.n_cols == X_ref.n_cols );
  REQUIRE( all( all( abs( A - X_ref ) < 1e-5 ) ) );

  A = A_orig;
  const bool status = solve(A, A, B);

  REQUIRE( status == true );
  REQUIRE( A.n_rows == X_ref.n_rows );
  REQUIRE( A.n_cols == X_ref.n_cols );
  REQUIRE( all( all( abs( A - X_ref ) < 1e-5 ) ) );
  }



// solve where inputs are the same
TEMPLATE_TEST_CASE("same_input_alias_solve", "[solve]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(200, 200, fill::randu);
  A.diag() += 3;
  Mat<eT> A_orig(A);

  Mat<eT> X_ref = solve(A, A_orig);

  A = solve(A, A);

  REQUIRE( A.n_rows == X_ref.n_rows );
  REQUIRE( A.n_cols == X_ref.n_cols );
  REQUIRE( all( all( abs( A - X_ref ) < 1e-5 ) ) );

  A = A_orig;

  const bool status = solve(A, A, A);

  REQUIRE( A.n_rows == X_ref.n_rows );
  REQUIRE( A.n_cols == X_ref.n_cols );
  REQUIRE( all( all( abs( A - X_ref ) < 1e-5 ) ) );
  }
