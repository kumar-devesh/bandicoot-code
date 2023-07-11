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

TEMPLATE_TEST_CASE("eig_sym_trivial", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  // [[1,  3,  5, 4]
  //  [3,  6, -1, 0]
  //  [5, -1, 10, 0]
  //  [4,  0,  0, 7]]
  Mat<eT> x(4, 4);
  x(0, 0) = eT(1);
  x(1, 0) = eT(3);
  x(2, 0) = eT(5);
  x(3, 0) = eT(4);
  x(0, 1) = eT(3);
  x(1, 1) = eT(6);
  x(2, 1) = eT(-1);
  x(3, 1) = eT(0);
  x(0, 2) = eT(5);
  x(1, 2) = eT(-1);
  x(2, 2) = eT(10);
  x(3, 2) = eT(0);
  x(0, 3) = eT(4);
  x(1, 3) = eT(0);
  x(2, 3) = eT(0);
  x(3, 3) = eT(7);

  Col<eT> eigvals1;
  Col<eT> eigvals2;
  Col<eT> eigvals3;
  Mat<eT> eigvecs3;

  const bool status1 = eig_sym(eigvals1, x);
  REQUIRE( status1 == true );

  eigvals2 = eig_sym(x);

  const bool status3 = eig_sym(eigvals3, eigvecs3, x);
  REQUIRE( status3 == true );

  REQUIRE( eigvals1.n_elem == 4 );
  REQUIRE( eigvals2.n_elem == 4 );
  REQUIRE( eigvals3.n_elem == 4 );
  REQUIRE( eigvecs3.n_rows == 4 );
  REQUIRE( eigvecs3.n_cols == 4 );

  // Eigenvalues should be sorted in ascending order.
  // These results were computed by Julia.
  REQUIRE( eT(eigvals1[0]) == Approx(eT(-3.55656569922)) );
  REQUIRE( eT(eigvals2[0]) == Approx(eT(-3.55656569922)) );
  REQUIRE( eT(eigvals3[0]) == Approx(eT(-3.55656569922)) );
  REQUIRE( eT(eigvals1[1]) == Approx(eT(6.281708994020)) );
  REQUIRE( eT(eigvals2[1]) == Approx(eT(6.281708994020)) );
  REQUIRE( eT(eigvals3[1]) == Approx(eT(6.281708994020)) );
  REQUIRE( eT(eigvals1[2]) == Approx(eT(8.448683435065)) );
  REQUIRE( eT(eigvals2[2]) == Approx(eT(8.448683435065)) );
  REQUIRE( eT(eigvals3[2]) == Approx(eT(8.448683435065)) );
  REQUIRE( eT(eigvals1[3]) == Approx(eT(12.82617327014)) );
  REQUIRE( eT(eigvals2[3]) == Approx(eT(12.82617327014)) );
  REQUIRE( eT(eigvals3[3]) == Approx(eT(12.82617327014)) );

  // Note that eigenvectors can be reversed, so we check the absolute values.
  REQUIRE( std::abs(eT(eigvecs3(0, 0))) == Approx(std::abs(eT(-0.8374640329669))) );
  REQUIRE( std::abs(eT(eigvecs3(1, 0))) == Approx(std::abs(eT(0.29751439870808))) );
  REQUIRE( std::abs(eT(eigvecs3(2, 0))) == Approx(std::abs(eT(0.33082379881798))) );
  REQUIRE( std::abs(eT(eigvecs3(3, 0))) == Approx(std::abs(eT(0.31732442418410))) );
  REQUIRE( std::abs(eT(eigvecs3(0, 1))) == Approx(std::abs(eT(0.10266826166865))) );
  REQUIRE( std::abs(eT(eigvecs3(1, 1))) == Approx(std::abs(eT(0.81006698736672))) );
  REQUIRE( std::abs(eT(eigvecs3(2, 1))) == Approx(std::abs(eT(0.07980162890593))) );
  REQUIRE( std::abs(eT(eigvecs3(3, 1))) == Approx(std::abs(eT(-0.5717363063935))) );
  REQUIRE( std::abs(eT(eigvecs3(0, 2))) == Approx(std::abs(eT(0.24707154423556))) );
  REQUIRE( std::abs(eT(eigvecs3(1, 2))) == Approx(std::abs(eT(0.49705641569448))) );
  REQUIRE( std::abs(eT(eigvecs3(2, 2))) == Approx(std::abs(eT(-0.4759191786973))) );
  REQUIRE( std::abs(eT(eigvecs3(3, 2))) == Approx(std::abs(eT(0.68219609130665))) );
  REQUIRE( std::abs(eT(eigvecs3(0, 3))) == Approx(std::abs(eT(-0.4765174430824))) );
  REQUIRE( std::abs(eT(eigvecs3(1, 3))) == Approx(std::abs(eT(-0.0906177585076))) );
  REQUIRE( std::abs(eT(eigvecs3(2, 3))) == Approx(std::abs(eT(-0.8109798083230))) );
  REQUIRE( std::abs(eT(eigvecs3(3, 3))) == Approx(std::abs(eT(-0.3271563827494))) );
  }



// 1x1 decomposition
TEMPLATE_TEST_CASE("single_element_eigendecomposition", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(1, 1);
  x(0, 0) = eT(5);

  Col<eT> eigvals1;
  Col<eT> eigvals2;
  Col<eT> eigvals3;
  Mat<eT> eigvecs3;

  const bool status1 = eig_sym(eigvals1, x);
  REQUIRE( status1 == true );

  eigvals2 = eig_sym(x);

  const bool status3 = eig_sym(eigvals3, eigvecs3, x);
  REQUIRE( status3 == true );

  REQUIRE( eigvals1.n_elem == 1 );
  REQUIRE( eigvals2.n_elem == 1 );
  REQUIRE( eigvals3.n_elem == 1 );
  REQUIRE( eigvecs3.n_rows == 1 );
  REQUIRE( eigvecs3.n_cols == 1 );

  REQUIRE( eT(eigvals1[0]) == Approx(eT(5)) );
  REQUIRE( eT(eigvals2[0]) == Approx(eT(5)) );
  REQUIRE( eT(eigvals3[0]) == Approx(eT(5)) );
  REQUIRE( eT(eigvecs3[0]) == Approx(eT(1)) );
  }



// empty decomposition
TEMPLATE_TEST_CASE("empty_eigendecomposition", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x;

  Col<eT> eigvals1;
  Col<eT> eigvals2;
  Col<eT> eigvals3;
  Mat<eT> eigvecs3;

  const bool status1 = eig_sym(eigvals1, x);
  REQUIRE( status1 == true );

  eigvals2 = eig_sym(x);

  const bool status3 = eig_sym(eigvals3, eigvecs3, x);
  REQUIRE( status3 == true );

  REQUIRE( eigvals1.n_elem == 0 );
  REQUIRE( eigvals2.n_elem == 0 );
  REQUIRE( eigvals3.n_elem == 0 );
  REQUIRE( eigvecs3.n_rows == 0 );
  REQUIRE( eigvecs3.n_cols == 0 );
  }



// eye decomposition
TEMPLATE_TEST_CASE("identity_eigendecomposition", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 100);
  x.eye();

  Col<eT> eigvals1;
  Col<eT> eigvals2;
  Col<eT> eigvals3;
  Mat<eT> eigvecs3;

  const bool status1 = eig_sym(eigvals1, x);
  REQUIRE( status1 == true );

  eigvals2 = eig_sym(x);

  const bool status3 = eig_sym(eigvals3, eigvecs3, x);
  REQUIRE( status3 == true );

  REQUIRE( eigvals1.n_elem == 100 );
  REQUIRE( eigvals2.n_elem == 100 );
  REQUIRE( eigvals3.n_elem == 100 );
  REQUIRE( eigvecs3.n_rows == 100 );
  REQUIRE( eigvecs3.n_cols == 100 );

  // All eigenvalues should be 1.
  REQUIRE( all( abs(eigvals1 - 1) < 1e-5 ) );
  REQUIRE( all( abs(eigvals2 - 1) < 1e-5 ) );
  REQUIRE( all( abs(eigvals3 - 1) < 1e-5 ) );

  // The eigenvectors should be the identity matrix.
  arma::Mat<eT> eigvecs3_cpu(eigvecs3);
  arma::Mat<eT> eigvecs3_cpu_ref = arma::eye<arma::Mat<eT>>(100, 100);

  REQUIRE( arma::approx_equal( eigvecs3_cpu, eigvecs3_cpu_ref, "absdiff", 1e-5 ) );
  }



// random diagonal matrix decomposition
TEMPLATE_TEST_CASE("diagonal_random_decomposition", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(250, 250);
  x.zeros();
  x.diag() = randu<Col<eT>>(250);

  Col<eT> eigvals1;
  Col<eT> eigvals2;
  Col<eT> eigvals3;
  Mat<eT> eigvecs3;

  const bool status1 = eig_sym(eigvals1, x);
  REQUIRE( status1 == true );

  eigvals2 = eig_sym(x);

  const bool status3 = eig_sym(eigvals3, eigvecs3, x);
  REQUIRE( status3 == true );

  // Get sorted eigenvalues.
  Col<eT> eigvals_ref = sort(x.diag());

  REQUIRE( eigvals1.n_elem == 250 );
  REQUIRE( eigvals2.n_elem == 250 );
  REQUIRE( eigvals3.n_elem == 250 );
  REQUIRE( eigvecs3.n_rows == 250 );
  REQUIRE( eigvecs3.n_cols == 250 );

  REQUIRE( all( abs(eigvals1 - eigvals_ref) < 1e-5 ) );
  REQUIRE( all( abs(eigvals2 - eigvals_ref) < 1e-5 ) );
  REQUIRE( all( abs(eigvals3 - eigvals_ref) < 1e-5 ) );

  // The eigenvectors should be a permuted identity matrix.
  arma::Mat<eT> eigvecs3_cpu(eigvecs3);
  // Check that each column only has one 1, and that each row only has one 1.
  REQUIRE( arma::all( arma::abs( arma::abs( arma::sum( eigvecs3_cpu, 0 ) ) - 1 ) <= 1e-5 ) );
  REQUIRE( arma::all( arma::abs( arma::abs( arma::sum( eigvecs3_cpu, 1 ) ) - 1 ) <= 1e-5 ) );
  }



// random decomposition without vectors: sum eigenvalue vs. trace test
TEMPLATE_TEST_CASE("sum_eigenvalue_decomposition", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (size_t i = 6; i < 11; ++i)
    {
    const size_t dim = std::pow((size_t) 2, i);

    Mat<eT> x = randu<Mat<eT>>(dim, dim);
    x *= x.t(); // make symmetric
    x.diag() += 0.1 * randu<Col<eT>>(dim); // make positive definite

    const eT trace_val = trace(x);

    Col<eT> eigvals1;
    Col<eT> eigvals2;
    Col<eT> eigvals3;
    Mat<eT> eigvecs3; // ignored for this test

    const bool status1 = eig_sym(eigvals1, x);
    REQUIRE( status1 == true );

    eigvals2 = eig_sym(x);

    const bool status3 = eig_sym(eigvals3, eigvecs3, x);
    REQUIRE( status3 == true );

    REQUIRE( eigvals1.n_elem == dim );
    REQUIRE( eigvals2.n_elem == dim );
    REQUIRE( eigvals3.n_elem == dim );

    const eT sum_eigvals1 = accu(eigvals1);
    const eT sum_eigvals2 = accu(eigvals2);
    const eT sum_eigvals3 = accu(eigvals3);

    REQUIRE( sum_eigvals1 == Approx(trace_val) );
    REQUIRE( sum_eigvals2 == Approx(trace_val) );
    REQUIRE( sum_eigvals3 == Approx(trace_val) );
    }
  }



// random decomposition with/without vectors: armadillo comparison test
TEMPLATE_TEST_CASE("eigenvalue_arma_comparison", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (size_t i = 6; i < 11; ++i)
    {
    const size_t dim = std::pow((size_t) 2, i) - 7;

    Mat<eT> x = randu<Mat<eT>>(dim, dim);
    x *= x.t(); // make symmetric
    x.diag() += 0.5 + 0.5 * randu<Col<eT>>(dim); // make positive definite

    arma::Mat<eT> x_cpu(x);

    Col<eT> eigvals1;
    Col<eT> eigvals2;
    Col<eT> eigvals3;
    Mat<eT> eigvecs3;

    const bool status1 = eig_sym(eigvals1, x);
    REQUIRE( status1 == true );

    eigvals2 = eig_sym(x);

    const bool status3 = eig_sym(eigvals3, eigvecs3, x);
    REQUIRE( status3 == true );

    REQUIRE( eigvals1.n_elem == dim );
    REQUIRE( eigvals2.n_elem == dim );
    REQUIRE( eigvals3.n_elem == dim );

    arma::Col<eT> eigvals_ref;
    arma::Mat<eT> eigvecs_ref;
    const bool status4 = arma::eig_sym(eigvals_ref, eigvecs_ref, x_cpu);
    REQUIRE( status4 == true );

    arma::Col<eT> eigvals1_cpu(eigvals1);
    arma::Col<eT> eigvals2_cpu(eigvals2);
    arma::Col<eT> eigvals3_cpu(eigvals3);
    arma::Mat<eT> eigvecs3_cpu(eigvecs3);

    // Note that we can get slightly different convergences on GPU and CPU.
    // For this reason, we don't check the eigenvectors directly in this test.
    // (They get checked in other tests.)
    const double tol = (is_float<eT>::value) ? 0.075 : 1e-4;
    for (size_t j = 0; j < dim; ++j)
      {
      if (std::abs(eigvals_ref[j]) < 1.0)
        {
        REQUIRE( (eigvals1_cpu[j] - eigvals_ref[j]) < tol );
        REQUIRE( (eigvals2_cpu[j] - eigvals_ref[j]) < tol );
        REQUIRE( (eigvals3_cpu[j] - eigvals_ref[j]) < tol );
        }
      else
        {
        REQUIRE( (eigvals1_cpu[j] - eigvals_ref[j]) < tol * eigvals_ref[j] );
        REQUIRE( (eigvals2_cpu[j] - eigvals_ref[j]) < tol * eigvals_ref[j] );
        REQUIRE( (eigvals3_cpu[j] - eigvals_ref[j]) < tol * eigvals_ref[j] );
        }
      }
    }
  }



// random decomposition with vectors: reconstruction test
TEMPLATE_TEST_CASE("random_eigendecomposition_reconstruction", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (size_t i = 6; i < 11; ++i)
    {
    const size_t dim = std::pow((size_t) 2, i) + 12;

    Mat<eT> x = randu<Mat<eT>>(dim, dim);
    x *= x.t(); // make symmetric
    x.diag() += 0.1 * randu<Col<eT>>(dim); // make positive definite

    Col<eT> eigvals;
    Mat<eT> eigvecs;

    const bool status = eig_sym(eigvals, eigvecs, x);
    REQUIRE( status == true );

    REQUIRE( eigvals.n_elem == dim );
    REQUIRE( eigvecs.n_rows == dim );
    REQUIRE( eigvecs.n_cols == dim );

    Mat<eT> xr = eigvecs * diagmat(eigvals) * eigvecs.t();
    Mat<eT> diff = x - xr;
    const eT norm_diff = accu(square(x - xr)) / xr.n_elem;

    const eT tol = (is_float<eT>::value) ? 1e-4 : 1e-8;
    REQUIRE( norm_diff < tol );
    }
  }



// random decomposition with vectors: orthonormal basis test
TEMPLATE_TEST_CASE("orthonormal_basis_eigendecomposition", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  const size_t dim = 224;

  Mat<eT> x = randu<Mat<eT>>(dim, dim);
  x *= x.t(); // make symmetric
  x.diag() += 0.1 * randu<Col<eT>>(dim); // make positive definite

  Col<eT> eigvals;
  Mat<eT> eigvecs;

  const bool status = eig_sym(eigvals, eigvecs, x);
  REQUIRE( status == true );

  REQUIRE( eigvals.n_elem == dim );
  REQUIRE( eigvecs.n_rows == dim );
  REQUIRE( eigvecs.n_cols == dim );

  // Ensure that each eigenvector is approximately normal.
  const eT tol = is_float<eT>::value ? 1e-5 : 1e-10;
  for (size_t v = 0; v < dim; ++v)
    {
    const eT norm = std::sqrt(dot(eigvecs.col(v), eigvecs.col(v)));
    REQUIRE( norm == Approx(eT(1)) );

    for (size_t v2 = v + 1; v2 < dim; ++v2)
      {
      const eT inner_product = dot(eigvecs.col(v), eigvecs.col(v2));
      REQUIRE( inner_product == Approx(eT(0)).margin(tol) );
      }
    }
  }



// alias test
TEMPLATE_TEST_CASE("alias_eigendecomposition_1", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  const size_t dim = 155;
  Mat<eT> x = randu<Mat<eT>>(dim, dim);
  x *= x.t(); // make symmetric
  x.diag() += 0.1 * randu<Col<eT>>(dim); // make positive definite

  Mat<eT> x_copy(x);

  Col<eT> eigvals2;
  Mat<eT> eigvecs_ref;
  const bool status = eig_sym(eigvals2, eigvecs_ref, x);
  REQUIRE( status == true );

  Col<eT> eigvals;
  const bool status2 = eig_sym(eigvals, x, x); // x now holds eigenvectors
  REQUIRE( status2 == true );

  REQUIRE( eigvals.n_elem == dim );
  REQUIRE( x.n_rows == dim );
  REQUIRE( x.n_cols == dim );

  REQUIRE( all( abs(eigvals - eigvals2) < 1e-5 ) );

  arma::Mat<eT> eigvecs_ref_cpu(eigvecs_ref);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( arma::approx_equal( x_cpu, eigvecs_ref_cpu, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("alias_eigendecomposition_2", "[eig_sym]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(1);
  x(0) = eT(6);

  const bool status1 = eig_sym(x, x);

  REQUIRE( status1 == true );
  REQUIRE( x.n_elem == 1);
  REQUIRE( eT(x[0]) == Approx(eT(6)) );

  x = eig_sym(x);

  REQUIRE( x.n_elem == 1);
  REQUIRE( eT(x[0]) == Approx(eT(6)) );

  Mat<eT> eigvecs;
  const bool status2 = eig_sym(x, eigvecs, x);
  REQUIRE( status2 == true );

  REQUIRE( x.n_elem == 1 );
  REQUIRE( eigvecs.n_elem == 1 );
  REQUIRE( eT(x[0]) == Approx(eT(6)) );
  REQUIRE( eT(eigvecs[0]) == Approx(eT(1)) );
  }



// nonsquare matrices should throw an exception
TEST_CASE("nonsquare_eigendecomposition", "[eig_sym]")
  {
  fmat x(5, 6);
  x.randu();
  fvec y;
  fmat z;
  bool status;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( status = eig_sym(y, x) );
  REQUIRE_THROWS( y = eig_sym(x) );
  REQUIRE_THROWS( status = eig_sym(y, z, x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// "dc" is not a supported decomposition type (yet)
TEST_CASE("dc_eigendecomposition", "[eig_sym]")
  {
  fmat x(5, 5);
  x.randu();
  fvec y;
  fmat z;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  bool status;
  REQUIRE_THROWS( status = eig_sym(y, z, x, "dc") );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }
