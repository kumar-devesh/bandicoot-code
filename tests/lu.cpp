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

// First, let's test the non-pivot versions.

TEMPLATE_TEST_CASE("lu_small", "[lu]", float, double)
  {
  typedef TestType eT;

  // [[1, 3, 5, 7],
  //  [9, 2, 6, 8],
  //  [3, 4, 5, 6],
  //  [0, 0, 1, 2]]
  Mat<eT> X(4, 4);
  X(0, 0) = eT(1);
  X(1, 0) = eT(9);
  X(2, 0) = eT(3);
  X(3, 0) = eT(0);
  X(0, 1) = eT(3);
  X(1, 1) = eT(2);
  X(2, 1) = eT(4);
  X(3, 1) = eT(0);
  X(0, 2) = eT(5);
  X(1, 2) = eT(6);
  X(2, 2) = eT(5);
  X(3, 2) = eT(1);
  X(0, 3) = eT(7);
  X(1, 3) = eT(8);
  X(2, 3) = eT(6);
  X(3, 3) = eT(2);

  Mat<eT> L, U;
  const bool status = lu(L, U, X);
  REQUIRE( status == true );

  REQUIRE( L.n_rows == 4 );
  REQUIRE( L.n_cols == 4 );
  REQUIRE( U.n_rows == 4 );
  REQUIRE( U.n_cols == 4 );

  for (uword c = 0; c < 4; ++c)
    {
    for (uword r = 0; r < 4; ++r)
      {
      if (r > c)
        {
        // Check that U is upper triangular.
        REQUIRE( eT(U(r, c)) == eT(0) );
        }
      }
    }

  // Check that the solution is approximately accurate.
  // These solutions were computed by Julia (with pivoting).
  REQUIRE( eT(L(0, 0)) == Approx(eT(1.0 / 9.0) ).margin(1e-5) );
  REQUIRE( eT(L(1, 0)) == Approx(eT(1.0)       ).margin(1e-5) );
  REQUIRE( eT(L(2, 0)) == Approx(eT(1.0 / 3.0) ).margin(1e-5) );
  REQUIRE( eT(L(3, 0)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(0, 1)) == Approx(eT(5.0 / 6.0) ).margin(1e-5) );
  REQUIRE( eT(L(1, 1)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(2, 1)) == Approx(eT(1.0)       ).margin(1e-5) );
  REQUIRE( eT(L(3, 1)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(0, 2)) == Approx(eT(1.0)       ).margin(1e-5) );
  REQUIRE( eT(L(1, 2)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(2, 2)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(3, 2)) == Approx(eT(6.0 / 11.0)).margin(1e-5) );
  REQUIRE( eT(L(0, 3)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(1, 3)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(2, 3)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(3, 3)) == Approx(eT(1.0)       ).margin(1e-5) );

  REQUIRE( eT(U(0, 0)) == Approx(eT(9.0)       ).margin(1e-5) );
  REQUIRE( eT(U(0, 1)) == Approx(eT(2.0)       ).margin(1e-5) );
  REQUIRE( eT(U(1, 1)) == Approx(eT(10.0 / 3.0)).margin(1e-5) );
  REQUIRE( eT(U(0, 2)) == Approx(eT(6.0)       ).margin(1e-5) );
  REQUIRE( eT(U(1, 2)) == Approx(eT(3.0)       ).margin(1e-5) );
  REQUIRE( eT(U(2, 2)) == Approx(eT(11.0 / 6.0)).margin(1e-5) );
  REQUIRE( eT(U(0, 3)) == Approx(eT(8.0)       ).margin(1e-5) );
  REQUIRE( eT(U(1, 3)) == Approx(eT(10.0 / 3.0)).margin(1e-5) );
  REQUIRE( eT(U(2, 3)) == Approx(eT(10.0 / 3.0)).margin(1e-5) );
  REQUIRE( eT(U(3, 3)) == Approx(eT(2.0 / 11.0)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("lu_random_large", "[lu]", float, double)
  {
  typedef TestType eT;

  // Check random large square LU decompositions against Armadillo.
  for (uword t = 8; t < 13; ++t)
    {
    const uword n = (double) std::pow(2.0, (double) t) + 1;

    Mat<eT> X = randu<Mat<eT>>(n, n);
    X.diag() += 0.5;

    Mat<eT> L, U;

    const bool status = lu(L, U, X);
    REQUIRE( status == true );

    REQUIRE( L.n_rows == n );
    REQUIRE( L.n_cols == n );
    REQUIRE( U.n_rows == n );
    REQUIRE( U.n_cols == n );

    // Check properties of U.
    // Due to the pivoting, L is not necessarily lower triangular.
    arma::Mat<eT> U_cpu(U);
    REQUIRE( U_cpu.is_trimatu() );

    // Check reconstruction error of X.
    Mat<eT> Xr = L * U;

//    const double error = norm(X - Xr, 2) / X.n_elem;
    Xr -= X;
    const double error = std::sqrt(accu(square(Xr))) / Xr.n_elem;
    const double tol = (is_float<eT>::value ? 1e-5 : 1e-8);

    REQUIRE( error < tol );
    }
  }



TEMPLATE_TEST_CASE("lu_triangular", "[lu]", float, double)
  {
  typedef TestType eT;

  arma::Mat<eT> X_cpu = arma::trimatu(arma::randu<arma::Mat<eT>>(503, 503));
  Mat<eT> X(X_cpu);

  Mat<eT> L, U;
  const bool status = lu(L, U, X);
  REQUIRE( status == true );

  Mat<eT> Xr = L * U;

  const double error = norm(X - Xr, 2);

  REQUIRE( error < 1e-5 );
  }



TEST_CASE("lu_empty", "[lu]")
  {
  mat X1;
  mat X2(5, 0);
  mat X3(0, 10);

  mat L, U;

  bool status = lu(L, U, X1);

  REQUIRE( status == true );
  REQUIRE( L.n_rows == 0 );
  REQUIRE( L.n_cols == 0 );
  REQUIRE( U.n_rows == 0 );
  REQUIRE( U.n_cols == 0 );

  status = lu(L, U, X2);

  REQUIRE( status == true );
  REQUIRE( L.n_rows == 5 );
  REQUIRE( L.n_cols == 0 );
  REQUIRE( U.n_rows == 0 );
  REQUIRE( U.n_cols == 0 );

  status = lu(L, U, X3);

  REQUIRE( status == true );
  REQUIRE( L.n_rows == 0 );
  REQUIRE( L.n_cols == 0 );
  REQUIRE( U.n_rows == 0 );
  REQUIRE( U.n_cols == 10 );
  }



TEMPLATE_TEST_CASE("lu_single_element", "[lu]", float, double)
  {
  typedef TestType eT;

  Mat<eT> X(1, 1);
  X(0, 0) = 5;

  Mat<eT> L, U;

  const bool status = lu(L, U, X);
  REQUIRE( status == true );

  REQUIRE( L.n_rows == 1 );
  REQUIRE( L.n_cols == 1 );
  REQUIRE( U.n_rows == 1 );
  REQUIRE( U.n_cols == 1 );

  REQUIRE( eT(L(0, 0)) == Approx(1.0) );
  REQUIRE( eT(U(0, 0)) == Approx(5.0) );
  }



TEMPLATE_TEST_CASE("lu_random_sizes_arma_comparison", "[lu]", float, double)
  {
  typedef TestType eT;

  const size_t seed = std::time(NULL);
  arma::arma_rng::set_seed(seed);

  arma::uvec x_sizes = arma::randi<arma::uvec>(15, arma::distr_param(100, 2000));
  arma::uvec y_sizes = arma::randi<arma::uvec>(15, arma::distr_param(100, 2000));

  x_sizes[0] = 91;
  y_sizes[0] = 109;

  for (size_t t = 0; t < 15; ++t)
    {
    const uword m = x_sizes[t];
    const uword n = y_sizes[t];

    Mat<eT> X = randi<Mat<eT>>(m, n, distr_param(-100, 100)) % randu<Mat<eT>>(m, n);
    X.diag() += 0.5;
    Mat<eT> L, U;

    const bool status = lu(L, U, X);

    REQUIRE( status == true );

    const uword min_mn = std::min(m, n);

    REQUIRE( L.n_rows == X.n_rows );
    REQUIRE( L.n_cols == min_mn   );
    REQUIRE( U.n_rows == min_mn   );
    REQUIRE( U.n_cols == X.n_cols );

    arma::Mat<eT> X_cpu(X);
    arma::Mat<eT> L_ref, U_ref;
    const bool arma_status = arma::lu(L_ref, U_ref, X_cpu);
    REQUIRE( arma_status == true );

    Mat<eT> Xr = L * U;
    arma::Mat<eT> L_cpu(L);
    arma::Mat<eT> U_cpu(U);
    arma::Mat<eT> Xr_cpu = L_ref * U_ref;

    const double cpu_error = arma::norm(X_cpu - Xr_cpu, 2);
    Xr -= X;
    const double error = std::sqrt(accu(square(Xr)));

    if (cpu_error > 1e-7)
      {
      // The GPU version can be a bit more inaccurate.
      REQUIRE( error < 100 * cpu_error );
      }
    else
      {
      REQUIRE( error < 1e-6 );
      }
    }
  }



// Now, let's test the pivot versions.

TEMPLATE_TEST_CASE("lup_small", "[lu]", float, double)
  {
  typedef TestType eT;

  // [[1, 3, 5, 7],
  //  [9, 2, 6, 8],
  //  [3, 4, 5, 6],
  //  [0, 0, 1, 2]]
  Mat<eT> X(4, 4);
  X(0, 0) = eT(1);
  X(1, 0) = eT(9);
  X(2, 0) = eT(3);
  X(3, 0) = eT(0);
  X(0, 1) = eT(3);
  X(1, 1) = eT(2);
  X(2, 1) = eT(4);
  X(3, 1) = eT(0);
  X(0, 2) = eT(5);
  X(1, 2) = eT(6);
  X(2, 2) = eT(5);
  X(3, 2) = eT(1);
  X(0, 3) = eT(7);
  X(1, 3) = eT(8);
  X(2, 3) = eT(6);
  X(3, 3) = eT(2);

  Mat<eT> L, U, P;
  const bool status = lu(L, U, P, X);
  REQUIRE( status == true );

  REQUIRE( L.n_rows == 4 );
  REQUIRE( L.n_cols == 4 );
  REQUIRE( U.n_rows == 4 );
  REQUIRE( U.n_cols == 4 );
  REQUIRE( P.n_rows == 4 );
  REQUIRE( P.n_cols == 4 );

  for (uword c = 0; c < 4; ++c)
    {
    for (uword r = 0; r < 4; ++r)
      {
      if (c > r)
        {
        // Check that L is lower triangular.
        REQUIRE( eT(L(r, c)) == eT(0) );
        }
      else if (c == r)
        {
        // The diagonal of L should be 1.
        REQUIRE( eT(L(r, c)) == eT(1) );
        }
      else
        {
        // Check that U is upper triangular.
        REQUIRE( eT(U(r, c)) == eT(0) );
        }
      }
    }

  // Check that the solution is approximately accurate.
  // These solutions were computed by Julia.
  REQUIRE( eT(L(1, 0)) == Approx(eT(1.0 / 3.0 )).margin(1e-5) );
  REQUIRE( eT(L(2, 0)) == Approx(eT(1.0 / 9.0 )).margin(1e-5) );
  REQUIRE( eT(L(3, 0)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(2, 1)) == Approx(eT(5.0 / 6.0 )).margin(1e-5) );
  REQUIRE( eT(L(3, 1)) == Approx(eT(0.0)       ).margin(1e-5) );
  REQUIRE( eT(L(3, 2)) == Approx(eT(6.0 / 11.0)).margin(1e-5) );

  REQUIRE( eT(U(0, 0)) == Approx(eT(9.0       )).margin(1e-5) );
  REQUIRE( eT(U(0, 1)) == Approx(eT(2.0       )).margin(1e-5) );
  REQUIRE( eT(U(1, 1)) == Approx(eT(10.0 / 3.0)).margin(1e-5) );
  REQUIRE( eT(U(0, 2)) == Approx(eT(6.0       )).margin(1e-5) );
  REQUIRE( eT(U(1, 2)) == Approx(eT(3.0       )).margin(1e-5) );
  REQUIRE( eT(U(2, 2)) == Approx(eT(11.0 / 6.0)).margin(1e-5) );
  REQUIRE( eT(U(0, 3)) == Approx(eT(8.0       )).margin(1e-5) );
  REQUIRE( eT(U(1, 3)) == Approx(eT(10.0 / 3.0)).margin(1e-5) );
  REQUIRE( eT(U(2, 3)) == Approx(eT(10.0 / 3.0)).margin(1e-5) );
  REQUIRE( eT(U(3, 3)) == Approx(eT(2.0 / 11.0)).margin(1e-5) );
  }



TEMPLATE_TEST_CASE("lup_random_large", "[lu]", float, double)
  {
  typedef TestType eT;

  // Check random large square LU decompositions against Armadillo.
  for (uword t = 8; t < 13; ++t)
    {
    const uword n = (double) std::pow(2.0, (double) t);

    arma::Mat<eT> X_in = arma::trimatl(arma::randu<arma::Mat<eT>>(n, n));
    Mat<eT> X(X_in);
    X.diag() += 0.5;

    Mat<eT> L, U, P;

    const bool status = lu(L, U, P, X);
    REQUIRE( status == true );

    REQUIRE( L.n_rows == n );
    REQUIRE( L.n_cols == n );
    REQUIRE( U.n_rows == n );
    REQUIRE( U.n_cols == n );
    REQUIRE( P.n_rows == n );
    REQUIRE( P.n_cols == n );

    // Check properties of L and U.
    arma::Mat<eT> L_cpu(L);
    REQUIRE( L_cpu.is_trimatl() );
    arma::Mat<eT> U_cpu(U);
    REQUIRE( U_cpu.is_trimatu() );

    arma::Mat<eT> P_cpu(P);

    REQUIRE( arma::all( L_cpu.diag() == eT(1) ) );

    // Check reconstruction error of X.
    arma::Mat<eT> Xr = P_cpu.t() * L_cpu * U_cpu;

    arma::Mat<eT> X_cpu(X);
    Xr -= X_cpu;
    const double error = std::sqrt(arma::accu(arma::square(Xr))) / Xr.n_elem;
    const double tol = (is_float<eT>::value ? 1e-6 : 1e-10);

    REQUIRE( error < tol );
    }
  }



TEMPLATE_TEST_CASE("lup_triangular", "[lu]", float, double)
  {
  typedef TestType eT;

  arma::Mat<eT> X_cpu = arma::trimatu(arma::randu<arma::Mat<eT>>(503, 503));
  Mat<eT> X(X_cpu);

  Mat<eT> L, U, P;
  const bool status = lu(L, U, P, X);
  REQUIRE( status == true );

  Mat<eT> Xr = P.t() * L * U;

  const double error = norm(X - Xr, 2);

  REQUIRE( error < 1e-5 );
  }



TEST_CASE("lup_empty", "[lu]")
  {
  mat X1;
  mat X2(5, 0);
  mat X3(0, 10);

  mat L, U, P;

  bool status = lu(L, U, P, X1);

  REQUIRE( status == true );
  REQUIRE( L.n_rows == 0 );
  REQUIRE( L.n_cols == 0 );
  REQUIRE( U.n_rows == 0 );
  REQUIRE( U.n_cols == 0 );
  REQUIRE( P.n_rows == 0 );
  REQUIRE( P.n_cols == 0 );

  status = lu(L, U, P, X2);

  REQUIRE( status == true );
  REQUIRE( L.n_rows == 5 );
  REQUIRE( L.n_cols == 0 );
  REQUIRE( U.n_rows == 0 );
  REQUIRE( U.n_cols == 0 );
  REQUIRE( P.n_rows == 5 );
  REQUIRE( P.n_cols == 5 );
  // The permutation matrix should be the identity matrix.
  arma::mat P_cpu(P);
  REQUIRE( P_cpu.is_diagmat() == true );
  REQUIRE( arma::all( P_cpu.diag() == 1.0 ) );

  status = lu(L, U, P, X3);

  REQUIRE( status == true );
  REQUIRE( L.n_rows == 0 );
  REQUIRE( L.n_cols == 0 );
  REQUIRE( U.n_rows == 0 );
  REQUIRE( U.n_cols == 10 );
  REQUIRE( P.n_rows == 0 );
  REQUIRE( P.n_cols == 0 );
  }



TEMPLATE_TEST_CASE("lup_single_element", "[lu]", float, double)
  {
  typedef TestType eT;

  Mat<eT> X(1, 1);
  X(0, 0) = 5;

  Mat<eT> L, U, P;

  const bool status = lu(L, U, P, X);
  REQUIRE( status == true );

  REQUIRE( L.n_rows == 1 );
  REQUIRE( L.n_cols == 1 );
  REQUIRE( U.n_rows == 1 );
  REQUIRE( U.n_cols == 1 );
  REQUIRE( P.n_rows == 1 );
  REQUIRE( P.n_cols == 1 );

  REQUIRE( eT(L(0, 0)) == Approx(1.0) );
  REQUIRE( eT(U(0, 0)) == Approx(5.0) );
  REQUIRE( eT(P(0, 0)) == Approx(1.0) );
  }



TEMPLATE_TEST_CASE("lup_random_sizes_arma_comparison", "[lu]", float, double)
  {
  typedef TestType eT;

  arma::uvec x_sizes = arma::randi<arma::uvec>(15, arma::distr_param(100, 2000));
  arma::uvec y_sizes = arma::randi<arma::uvec>(15, arma::distr_param(100, 2000));

  for (size_t t = 0; t < 15; ++t)
    {
    const uword m = x_sizes[t];
    const uword n = y_sizes[t];

    Mat<eT> X = randi<Mat<eT>>(m, n, distr_param(-100, 100)) % randu<Mat<eT>>(m, n);
    Mat<eT> L, U, P;

    const bool status = lu(L, U, P, X);

    REQUIRE( status == true );

    const uword min_mn = std::min(m, n);

    REQUIRE( L.n_rows == X.n_rows );
    REQUIRE( L.n_cols == min_mn   );
    REQUIRE( U.n_rows == min_mn   );
    REQUIRE( U.n_cols == X.n_cols );
    REQUIRE( P.n_rows == X.n_rows );
    REQUIRE( P.n_cols == X.n_rows );

    arma::Mat<eT> X_cpu(X);
    arma::Mat<eT> L_ref, U_ref, P_ref;
    const bool arma_status = arma::lu(L_ref, U_ref, P_ref, X_cpu);
    REQUIRE( arma_status == true );

    Mat<eT> Xr = P.t() * L * U;
    arma::Mat<eT> Xr_cpu = P_ref.t() * L_ref * U_ref;

    Xr -= X;
    // TODO: debug issues that are observed with norm(X - Xr, 2) here!
    const double error = std::sqrt(accu(square(Xr)));
    const double cpu_error = arma::norm(X_cpu - Xr_cpu, 2);

    if (cpu_error > 1e-7)
      {
      // The GPU version can be a bit more inaccurate.
      REQUIRE( error < 50 * cpu_error );
      }
    else
      {
      REQUIRE( error < 1e-6 );
      }
    }
  }
