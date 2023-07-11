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

// trivial det on small 2x2 matrix
TEMPLATE_TEST_CASE("trivial_2x2_det", "[det]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X(2, 2);
  X(0, 0) = 1.0;
  X(0, 1) = 2.0;
  X(1, 0) = 3.0;
  X(1, 1) = 4.0;

  const eT manual_det = 1.0 * 4.0 - 2.0 * 3.0;

  REQUIRE( det(X) == Approx(manual_det) );

  eT det_val;
  det(det_val, X);
  REQUIRE( det_val == Approx(manual_det) );
  }



// trivial det on small 3x3 matrix
TEMPLATE_TEST_CASE("trivial_3x3_det", "[det]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X(3, 3);
  X(0, 0) = 1.0;
  X(0, 1) = 2.0;
  X(0, 2) = 3.0;
  X(1, 0) = 4.0;
  X(1, 1) = 5.0;
  X(1, 2) = 6.0;
  X(2, 0) = 7.0;
  X(2, 1) = 8.0;
  X(2, 2) = 19.0;

  const eT manual_det = (1.0 * 5.0 * 19.0) + // aei
                        (2.0 * 6.0 * 7.0) +  // bfg
                        (3.0 * 4.0 * 8.0) -  // cdh
                        (3.0 * 5.0 * 7.0) -  // ceg
                        (2.0 * 4.0 * 19.0) - // bdi
                        (1.0 * 6.0 * 8.0);

  REQUIRE( det(X) == Approx(manual_det) );

  eT det_val;
  det(det_val, X);
  REQUIRE( det_val == Approx(manual_det) );
  }



// comparison with armadillo for reasonably sized matrices
TEMPLATE_TEST_CASE("det_arma_comparison", "[det]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword p = 3; p < 7; ++p)
    {
    const size_t dim = std::pow((uword) 2, p);

    Mat<eT> x(dim, dim);
    x.randu();
    x.diag() += 1.5;

    arma::Mat<eT> x_cpu(x);

    const eT det_val = det(x);
    const eT arma_det_val = arma::det(x_cpu);

    REQUIRE( det_val == Approx(arma_det_val).epsilon(1e-3) );

    eT det_val2;
    REQUIRE( det(det_val2, x) == true );
    REQUIRE( det_val2 == Approx(arma_det_val).epsilon(1e-3) );
    }
  }



// empty det
TEST_CASE("empty_det", "[det]")
  {
  fmat x;

  REQUIRE( det(x) == Approx(1.0) );
  }



// non-square det should throw error
TEST_CASE("non_square_det", "[det]")
  {
  fmat x(3, 4);
  x.randu();

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  float out;
  REQUIRE_THROWS( out = det(x) );
  REQUIRE_THROWS( det(out, x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// det on expression
TEMPLATE_TEST_CASE("expr_det", "[det]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(133, 133);
  x.randu();
  x -= 0.5;

  Mat<eT> xtx = x.t() * x;

  const eT det_val1 = det(x.t() * x);
  eT det_val2;
  REQUIRE( det(det_val2, x.t() * x) == true );

  const eT ref_det_val = det(xtx);

  REQUIRE( det_val1 == Approx(ref_det_val) );
  REQUIRE( det_val2 == Approx(ref_det_val) );
  }



// det on diagmat from a vector
TEMPLATE_TEST_CASE("diagmat_det", "[det]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(130) + 0.5;

  const eT det_val1 = det(diagmat(x));
  eT det_val2;
  REQUIRE( det(det_val2, diagmat(x)) == true );

  Mat<eT> x_ref = diagmat(x);

  const eT ref_det_val = det(x_ref);

  REQUIRE( det_val1 == Approx(ref_det_val).margin(1e-8) );
  REQUIRE( det_val2 == Approx(ref_det_val).margin(1e-8) );
  }



// det on diagmat from a matrix
TEMPLATE_TEST_CASE("diagmat_mat_det", "[det]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(130, 130);

  const eT det_val1 = det(diagmat(x));
  eT det_val2;
  REQUIRE( det(det_val2, diagmat(x)) == true );

  Mat<eT> x_ref = diagmat(x);

  const eT ref_det_val = det(x_ref);

  REQUIRE( det_val1 == Approx(ref_det_val).margin(1e-8) );
  REQUIRE( det_val2 == Approx(ref_det_val).margin(1e-8) );
  }



// det on big eye matrix
TEMPLATE_TEST_CASE("big_eye_det", "[det]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(1500, 1500);
  x.eye();

  const eT det_val1 = det(x);
  eT det_val2;
  REQUIRE( det(det_val2, x) == true );

  REQUIRE( det_val1 == Approx(1.0) );
  REQUIRE( det_val2 == Approx(1.0) );
  }



// det on non-invertible matrix
TEMPLATE_TEST_CASE("non_invertible_det", "[det]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(100, 100);
  x.zeros();
  x(99, 99) = -1.0;

  eT det_val;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE( det(det_val, x) == false );
  REQUIRE( det_val == Approx(eT(0)).margin(1e-10) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }
