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

// Trivial test cases.

TEMPLATE_TEST_CASE("identity_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x;
  x.eye(3, 3);

  Mat<eT> out = pinv(x);

  REQUIRE( out.n_rows == 3 );
  REQUIRE( out.n_cols == 3 );
  REQUIRE( eT(out(0, 0)) == Approx(1.0) );
  REQUIRE( eT(out(0, 1)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(0, 2)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(1, 0)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(1, 1)) == Approx(1.0) );
  REQUIRE( eT(out(1, 2)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(2, 0)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(2, 1)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out(2, 2)) == Approx(1.0) );
  }



TEMPLATE_TEST_CASE("trivial_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(3, 3);
  // x: [[1, 5, 3],
  //     [9, 6, 2],
  //     [4, 10, 1]]
  x(0, 0) = (eT) 1;
  x(0, 1) = (eT) 5;
  x(0, 2) = (eT) 3;
  x(1, 0) = (eT) 9;
  x(1, 1) = (eT) 6;
  x(1, 2) = (eT) 2;
  x(2, 0) = (eT) 4;
  x(2, 1) = (eT) 10;
  x(2, 2) = (eT) 1;

  Mat<eT> out = pinv(x);

  REQUIRE( out.n_rows == 3 );
  REQUIRE( out.n_cols == 3 );
  // Results computed by Julia.
  REQUIRE( eT(out(0, 0)) == Approx(-0.0782123) );
  REQUIRE( eT(out(0, 1)) == Approx( 0.139665) );
  REQUIRE( eT(out(0, 2)) == Approx(-0.0446927) );
  REQUIRE( eT(out(1, 0)) == Approx(-0.00558659) );
  REQUIRE( eT(out(1, 1)) == Approx(-0.0614525) );
  REQUIRE( eT(out(1, 2)) == Approx( 0.139665) );
  REQUIRE( eT(out(2, 0)) == Approx( 0.368715) );
  REQUIRE( eT(out(2, 1)) == Approx( 0.0558659) );
  REQUIRE( eT(out(2, 2)) == Approx(-0.217877) );
  }



//
// Tests for pinv() on diagonal matrices that are represented as vectors
//



// invert a diagonal matrix that is given as a vector
TEMPLATE_TEST_CASE("diagonal_vec_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(100) + 2.5;
  // Make a value negative because why not.
  x(33) *= -1.0;

  Mat<eT> out = pinv(diagmat(x));
  Mat<eT> out2;
  const bool status = pinv(out2, diagmat(x));

  REQUIRE( status == true );

  REQUIRE( out.n_rows == 100 );
  REQUIRE( out.n_cols == 100 );
  REQUIRE( out2.n_rows == 100 );
  REQUIRE( out2.n_cols == 100 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();

  Col<eT> ref_diag = 1.0 / x;

  REQUIRE( all( abs(out_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out2_diag - ref_diag) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  }


// fail to invert a diagonal matrix with NaNs that is given as a vector
TEMPLATE_TEST_CASE("nan_diagonal_vec_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(100) + 2.5;
  x(22) = std::numeric_limits<eT>::quiet_NaN();

  Mat<eT> out;
  Mat<eT> out2;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( out = pinv(diagmat(x)) );
  const bool status = pinv(out2, diagmat(x));
  REQUIRE( status == false );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// invert a diagonal matrix that is given as a vector with a custom tolerance
TEMPLATE_TEST_CASE("custom_tol_diagonal_vec_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(1007) + 2.5;
  x(622) = 1e-5;
  x(633) = -0.1;

  Mat<eT> out = pinv(diagmat(x), 1e-4);
  Mat<eT> out2;
  const bool status = pinv(out2, diagmat(x), 1e-4);

  REQUIRE( status == true );

  REQUIRE( out.n_rows == 1007 );
  REQUIRE( out.n_cols == 1007 );
  REQUIRE( out2.n_rows == 1007 );
  REQUIRE( out2.n_cols == 1007 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();

  Col<eT> ref_diag = 1.0 / x;

  REQUIRE( all( abs( out_diag.subvec(0, 621) - ref_diag.subvec(0, 621) ) < 1e-5 ) );
  REQUIRE( all( abs( out2_diag.subvec(0, 621) - ref_diag.subvec(0, 621) ) < 1e-5 ) );

  REQUIRE( eT(out_diag(622)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out2_diag(622)) == Approx(0.0).margin(1e-5) );

  REQUIRE( all( abs( out_diag.subvec(623, 1006) - ref_diag.subvec(623, 1006) ) < 1e-5 ) );
  REQUIRE( all( abs( out2_diag.subvec(623, 1006) - ref_diag.subvec(623, 1006) ) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  }



// invert a diagonal matrix that is given as a vector,
// where the tolerance is large enough to filter everything
TEMPLATE_TEST_CASE("too_large_tol_diagonal_vec_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(514);

  Mat<eT> out = pinv(diagmat(x), 10.0);
  Mat<eT> out2;
  const bool status = pinv(out2, diagmat(x), 10.0);

  REQUIRE( status == true );

  REQUIRE( out.n_rows == 514 );
  REQUIRE( out.n_cols == 514 );
  REQUIRE( out2.n_rows == 514 );
  REQUIRE( out2.n_cols == 514 );

  // Make sure the returned matrices are empty.
  REQUIRE( all( all( abs(out) < 1e-5 ) ) );
  REQUIRE( all( all( abs(out2) < 1e-5 ) ) );
  }



// invert an empty diagonal matrix given as a vector
TEST_CASE("empty_diagonal_vec_pinv", "[pinv]")
  {
  fvec x;

  fmat out = pinv(diagmat(x));
  fmat out2;
  const bool status = pinv(out2, diagmat(x));

  REQUIRE( status == true );

  REQUIRE( out.n_rows == 0 );
  REQUIRE( out.n_cols == 0 );
  REQUIRE( out2.n_rows == 0 );
  REQUIRE( out2.n_cols == 0 );
  }



// invert a transposed diagonal matrix that is given as a vector
TEMPLATE_TEST_CASE("trans_diagonal_vec_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(100) + 2.5;
  // Make a value negative because why not.
  x(33) *= -1.0;

  Mat<eT> out = pinv(diagmat(x).t());
  Mat<eT> out2 = pinv(diagmat(x.t()));
  Mat<eT> out3;
  Mat<eT> out4;
  const bool status3 = pinv(out3, diagmat(x).t());
  const bool status4 = pinv(out4, diagmat(x.t()));

  REQUIRE( status3 == true );
  REQUIRE( status4 == true );

  REQUIRE( out.n_rows == 100 );
  REQUIRE( out.n_cols == 100 );
  REQUIRE( out2.n_rows == 100 );
  REQUIRE( out2.n_cols == 100 );
  REQUIRE( out3.n_rows == 100 );
  REQUIRE( out3.n_cols == 100 );
  REQUIRE( out4.n_rows == 100 );
  REQUIRE( out4.n_cols == 100 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();
  Col<eT> out3_diag = out3.diag();
  Col<eT> out4_diag = out4.diag();

  Col<eT> ref_diag = 1.0 / x;

  REQUIRE( all( abs(out_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out2_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out3_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out4_diag - ref_diag) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);
  arma::Mat<eT> out3_cpu(out);
  arma::Mat<eT> out4_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  REQUIRE( out3_cpu.is_diagmat() );
  REQUIRE( out4_cpu.is_diagmat() );
  }



// invert a scaled diagonal matrix that is given as a vector
TEMPLATE_TEST_CASE("scaled_diagonal_vec_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(100) + 2.5;
  // Make a value negative because why not.
  x(33) *= -1.0;

  Mat<eT> out = pinv(3.0 * diagmat(x));
  Mat<eT> out2 = pinv(diagmat(3.0 * x));
  Mat<eT> out3;
  Mat<eT> out4;
  const bool status3 = pinv(out3, 3.0 * diagmat(x));
  const bool status4 = pinv(out4, diagmat(3.0 * x));

  REQUIRE( status3 == true );
  REQUIRE( status4 == true );

  REQUIRE( out.n_rows == 100 );
  REQUIRE( out.n_cols == 100 );
  REQUIRE( out2.n_rows == 100 );
  REQUIRE( out2.n_cols == 100 );
  REQUIRE( out3.n_rows == 100 );
  REQUIRE( out3.n_cols == 100 );
  REQUIRE( out4.n_rows == 100 );
  REQUIRE( out4.n_cols == 100 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();
  Col<eT> out3_diag = out3.diag();
  Col<eT> out4_diag = out4.diag();

  Col<eT> ref_diag = 1.0 / (3.0 * x);

  REQUIRE( all( abs(out_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out2_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out3_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out4_diag - ref_diag) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);
  arma::Mat<eT> out3_cpu(out);
  arma::Mat<eT> out4_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  REQUIRE( out3_cpu.is_diagmat() );
  REQUIRE( out4_cpu.is_diagmat() );
  }



// invert a scaled, transposed diagonal matrix that is given as a vector
TEMPLATE_TEST_CASE("scaled_trans_diagonal_vec_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randu<Col<eT>>(100) + 2.5;
  // Make a value negative because why not.
  x(33) *= -1.0;

  Mat<eT> out = pinv(3.0 * diagmat(x.t()));
  Mat<eT> out2 = pinv(diagmat(3.0 * x.t()));
  Mat<eT> out3 = pinv(3.0 * diagmat(x.t()).t());
  Mat<eT> out4 = pinv(diagmat(3.0 * x.t()).t());
  Mat<eT> out5;
  Mat<eT> out6;
  Mat<eT> out7;
  Mat<eT> out8;
  const bool status5 = pinv(out5, 3.0 * diagmat(x.t()));
  const bool status6 = pinv(out6, diagmat(3.0 * x.t()));
  const bool status7 = pinv(out7, 3.0 * diagmat(x.t()).t());
  const bool status8 = pinv(out8, diagmat(3.0 * x.t()).t());

  REQUIRE( status5 == true );
  REQUIRE( status6 == true );
  REQUIRE( status7 == true );
  REQUIRE( status8 == true );

  REQUIRE( out.n_rows == 100 );
  REQUIRE( out.n_cols == 100 );
  REQUIRE( out2.n_rows == 100 );
  REQUIRE( out2.n_cols == 100 );
  REQUIRE( out3.n_rows == 100 );
  REQUIRE( out3.n_cols == 100 );
  REQUIRE( out4.n_rows == 100 );
  REQUIRE( out4.n_cols == 100 );
  REQUIRE( out5.n_rows == 100 );
  REQUIRE( out5.n_cols == 100 );
  REQUIRE( out6.n_rows == 100 );
  REQUIRE( out6.n_cols == 100 );
  REQUIRE( out7.n_rows == 100 );
  REQUIRE( out7.n_cols == 100 );
  REQUIRE( out8.n_rows == 100 );
  REQUIRE( out8.n_cols == 100 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();
  Col<eT> out3_diag = out3.diag();
  Col<eT> out4_diag = out4.diag();
  Col<eT> out5_diag = out5.diag();
  Col<eT> out6_diag = out6.diag();
  Col<eT> out7_diag = out7.diag();
  Col<eT> out8_diag = out8.diag();

  Col<eT> ref_diag = 1.0 / (3.0 * x);

  REQUIRE( all( abs(out_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out2_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out3_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out4_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out5_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out6_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out7_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out8_diag - ref_diag) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);
  arma::Mat<eT> out3_cpu(out);
  arma::Mat<eT> out4_cpu(out);
  arma::Mat<eT> out5_cpu(out);
  arma::Mat<eT> out6_cpu(out);
  arma::Mat<eT> out7_cpu(out);
  arma::Mat<eT> out8_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  REQUIRE( out3_cpu.is_diagmat() );
  REQUIRE( out4_cpu.is_diagmat() );
  REQUIRE( out5_cpu.is_diagmat() );
  REQUIRE( out6_cpu.is_diagmat() );
  REQUIRE( out7_cpu.is_diagmat() );
  REQUIRE( out8_cpu.is_diagmat() );
  }



// invert a diagonal matrix that is given as a vector, where the output is an alias of the input
TEMPLATE_TEST_CASE("alias_diagonal_vec_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(100, 1) + 0.5;
  Mat<eT> old_x(x);
  x = pinv(diagmat(x));

  REQUIRE( x.n_rows == 100 );
  REQUIRE( x.n_cols == 100 );

  Col<eT> x_diag = x.diag();
  Col<eT> ref_diag = (1.0 / old_x);

  REQUIRE( all( abs( x_diag - ref_diag ) < 1e-5 ) );
  }



// convert an inverted diagonal matrix that is given as a vector
TEMPLATE_TEST_CASE
  (
  "conv_to_diagonal_vec_pinv",
  "[pinv]",
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

  Col<eT1> x = randu<Col<eT1>>(500) + 0.5;

  Mat<eT2> out = conv_to<Mat<eT2>>::from(pinv(diagmat(x)));

  Mat<eT1> out_pre_conv = pinv(diagmat(x));
  Mat<eT2> out_ref = conv_to<Mat<eT2>>::from(out_pre_conv);

  REQUIRE( out.n_rows == out_ref.n_rows );
  REQUIRE( out.n_cols == out_ref.n_cols );

  REQUIRE( all( all( abs( out - out_ref ) < 1e-5 ) ) );
  }



//
// tests for pinv() on diagonal matrices that are represented as matrices
//



// invert a diagonal matrix
TEMPLATE_TEST_CASE("diagonal_mat_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(100, 100) + 2.5;
  // Make a diagonal value negative because why not.
  x(33, 33) *= -1.0;

  Mat<eT> out = pinv(diagmat(x));
  Mat<eT> out2;
  const bool status = pinv(out2, diagmat(x));

  REQUIRE( status == true );

  REQUIRE( out.n_rows == 100 );
  REQUIRE( out.n_cols == 100 );
  REQUIRE( out2.n_rows == 100 );
  REQUIRE( out2.n_cols == 100 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();

  Col<eT> ref_diag = 1.0 / x.diag();

  REQUIRE( all( abs(out_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out2_diag - ref_diag) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  }


// fail to invert a diagonal matrix with NaNs
TEMPLATE_TEST_CASE("nan_diagonal_mat_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(100, 100) + 2.5;
  x(22, 22) = std::numeric_limits<eT>::quiet_NaN();

  Mat<eT> out;
  Mat<eT> out2;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( out = pinv(diagmat(x)) );
  const bool status = pinv(out2, diagmat(x));
  REQUIRE( status == false );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// invert a diagonal matrix that has NaNs that are not on the diagonal
TEMPLATE_TEST_CASE("diagonal_mat_pinv_offdiag_nans", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(100, 100) + 2.5;
  // Make a diagonal value negative because why not.
  x(33, 33) *= -1.0;
  // Some off-diagonal entries are NaNs.
  x(0, 1) = std::numeric_limits<eT>::quiet_NaN();
  x(10, 15) = std::numeric_limits<eT>::quiet_NaN();
  x(76, 4) = std::numeric_limits<eT>::quiet_NaN();

  Mat<eT> out = pinv(diagmat(x));
  Mat<eT> out2;
  const bool status = pinv(out2, diagmat(x));

  REQUIRE( status == true );

  REQUIRE( out.n_rows == 100 );
  REQUIRE( out.n_cols == 100 );
  REQUIRE( out2.n_rows == 100 );
  REQUIRE( out2.n_cols == 100 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();

  Col<eT> ref_diag = 1.0 / x.diag();

  REQUIRE( all( abs(out_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out2_diag - ref_diag) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  }




// invert a diagonal matrix with a custom tolerance
TEMPLATE_TEST_CASE("custom_tol_diagonal_mat_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(1007, 1007) + 2.5;
  x(622, 622) = 1e-5;
  x(633, 633) = -0.1;
  x(12, 13) = 0.0; // why not?

  Mat<eT> out = pinv(diagmat(x), 1e-4);
  Mat<eT> out2;
  const bool status = pinv(out2, diagmat(x), 1e-4);

  REQUIRE( status == true );

  REQUIRE( out.n_rows == 1007 );
  REQUIRE( out.n_cols == 1007 );
  REQUIRE( out2.n_rows == 1007 );
  REQUIRE( out2.n_cols == 1007 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();

  Col<eT> ref_diag = 1.0 / x.diag();

  REQUIRE( all( abs( out_diag.subvec(0, 621) - ref_diag.subvec(0, 621) ) < 1e-5 ) );
  REQUIRE( all( abs( out2_diag.subvec(0, 621) - ref_diag.subvec(0, 621) ) < 1e-5 ) );

  REQUIRE( eT(out_diag(622)) == Approx(0.0).margin(1e-5) );
  REQUIRE( eT(out2_diag(622)) == Approx(0.0).margin(1e-5) );

  REQUIRE( all( abs( out_diag.subvec(623, 1006) - ref_diag.subvec(623, 1006) ) < 1e-5 ) );
  REQUIRE( all( abs( out2_diag.subvec(623, 1006) - ref_diag.subvec(623, 1006) ) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  }



// invert a diagonal matrix, where the tolerance is large enough to filter everything
TEMPLATE_TEST_CASE("too_large_tol_diagonal_mat_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(514, 517); // non-square for fun

  Mat<eT> out = pinv(diagmat(x), 10.0);
  Mat<eT> out2;
  const bool status = pinv(out2, diagmat(x), 10.0);

  REQUIRE( status == true );

  REQUIRE( out.n_rows == 514 );
  REQUIRE( out.n_cols == 514 );
  REQUIRE( out2.n_rows == 514 );
  REQUIRE( out2.n_cols == 514 );

  // Make sure the returned matrices are empty.
  REQUIRE( all( all( abs(out) < 1e-5 ) ) );
  REQUIRE( all( all( abs(out2) < 1e-5 ) ) );
  }



// invert an empty diagonal matrix
TEST_CASE("empty_diagonal_mat_pinv", "[pinv]")
  {
  fmat x;

  fmat out = pinv(diagmat(x));
  fmat out2;
  const bool status = pinv(out2, diagmat(x));

  REQUIRE( status == true );

  REQUIRE( out.n_rows == 0 );
  REQUIRE( out.n_cols == 0 );
  REQUIRE( out2.n_rows == 0 );
  REQUIRE( out2.n_cols == 0 );
  }



// invert a transposed diagonal matrix
TEMPLATE_TEST_CASE("trans_diagonal_mat_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(100, 100) + 2.5;
  // Make a value negative because why not.
  x(33) *= -1.0;

  Mat<eT> out = pinv(diagmat(x).t());
  Mat<eT> out2 = pinv(diagmat(x.t()));
  Mat<eT> out3;
  Mat<eT> out4;
  const bool status3 = pinv(out3, diagmat(x).t());
  const bool status4 = pinv(out4, diagmat(x.t()));

  REQUIRE( status3 == true );
  REQUIRE( status4 == true );

  REQUIRE( out.n_rows == 100 );
  REQUIRE( out.n_cols == 100 );
  REQUIRE( out2.n_rows == 100 );
  REQUIRE( out2.n_cols == 100 );
  REQUIRE( out3.n_rows == 100 );
  REQUIRE( out3.n_cols == 100 );
  REQUIRE( out4.n_rows == 100 );
  REQUIRE( out4.n_cols == 100 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();
  Col<eT> out3_diag = out3.diag();
  Col<eT> out4_diag = out4.diag();

  Col<eT> ref_diag = 1.0 / x.diag();

  REQUIRE( all( abs(out_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out2_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out3_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out4_diag - ref_diag) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);
  arma::Mat<eT> out3_cpu(out);
  arma::Mat<eT> out4_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  REQUIRE( out3_cpu.is_diagmat() );
  REQUIRE( out4_cpu.is_diagmat() );
  }



// invert a scaled diagonal matrix
TEMPLATE_TEST_CASE("scaled_diagonal_mat_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(100, 100) + 2.5;
  // Make a value negative because why not.
  x(33) *= -1.0;

  Mat<eT> out = pinv(3.0 * diagmat(x));
  Mat<eT> out2 = pinv(diagmat(3.0 * x));
  Mat<eT> out3;
  Mat<eT> out4;
  const bool status3 = pinv(out3, 3.0 * diagmat(x));
  const bool status4 = pinv(out4, diagmat(3.0 * x));

  REQUIRE( status3 == true );
  REQUIRE( status4 == true );

  REQUIRE( out.n_rows == 100 );
  REQUIRE( out.n_cols == 100 );
  REQUIRE( out2.n_rows == 100 );
  REQUIRE( out2.n_cols == 100 );
  REQUIRE( out3.n_rows == 100 );
  REQUIRE( out3.n_cols == 100 );
  REQUIRE( out4.n_rows == 100 );
  REQUIRE( out4.n_cols == 100 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();
  Col<eT> out3_diag = out3.diag();
  Col<eT> out4_diag = out4.diag();

  Col<eT> ref_diag = 1.0 / (3.0 * x.diag());

  REQUIRE( all( abs(out_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out2_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out3_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out4_diag - ref_diag) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);
  arma::Mat<eT> out3_cpu(out);
  arma::Mat<eT> out4_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  REQUIRE( out3_cpu.is_diagmat() );
  REQUIRE( out4_cpu.is_diagmat() );
  }



// invert a scaled, transposed diagonal matrix
TEMPLATE_TEST_CASE("scaled_trans_diagonal_mat_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(100, 100) + 2.5;
  // Make a value negative because why not.
  x(33) *= -1.0;

  Mat<eT> out = pinv(3.0 * diagmat(x.t()));
  Mat<eT> out2 = pinv(diagmat(3.0 * x.t()));
  Mat<eT> out3 = pinv(3.0 * diagmat(x.t()).t());
  Mat<eT> out4 = pinv(diagmat(3.0 * x.t()).t());
  Mat<eT> out5;
  Mat<eT> out6;
  Mat<eT> out7;
  Mat<eT> out8;
  const bool status5 = pinv(out5, 3.0 * diagmat(x.t()));
  const bool status6 = pinv(out6, diagmat(3.0 * x.t()));
  const bool status7 = pinv(out7, 3.0 * diagmat(x.t()).t());
  const bool status8 = pinv(out8, diagmat(3.0 * x.t()).t());

  REQUIRE( status5 == true );
  REQUIRE( status6 == true );
  REQUIRE( status7 == true );
  REQUIRE( status8 == true );

  REQUIRE( out.n_rows == 100 );
  REQUIRE( out.n_cols == 100 );
  REQUIRE( out2.n_rows == 100 );
  REQUIRE( out2.n_cols == 100 );
  REQUIRE( out3.n_rows == 100 );
  REQUIRE( out3.n_cols == 100 );
  REQUIRE( out4.n_rows == 100 );
  REQUIRE( out4.n_cols == 100 );
  REQUIRE( out5.n_rows == 100 );
  REQUIRE( out5.n_cols == 100 );
  REQUIRE( out6.n_rows == 100 );
  REQUIRE( out6.n_cols == 100 );
  REQUIRE( out7.n_rows == 100 );
  REQUIRE( out7.n_cols == 100 );
  REQUIRE( out8.n_rows == 100 );
  REQUIRE( out8.n_cols == 100 );

  Col<eT> out_diag = out.diag();
  Col<eT> out2_diag = out2.diag();
  Col<eT> out3_diag = out3.diag();
  Col<eT> out4_diag = out4.diag();
  Col<eT> out5_diag = out5.diag();
  Col<eT> out6_diag = out6.diag();
  Col<eT> out7_diag = out7.diag();
  Col<eT> out8_diag = out8.diag();

  Col<eT> ref_diag = 1.0 / (3.0 * x.diag());

  REQUIRE( all( abs(out_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out2_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out3_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out4_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out5_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out6_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out7_diag - ref_diag) < 1e-5 ) );
  REQUIRE( all( abs(out8_diag - ref_diag) < 1e-5 ) );

  // Make sure the returned matrices are diagonal.
  arma::Mat<eT> out_cpu(out);
  arma::Mat<eT> out2_cpu(out);
  arma::Mat<eT> out3_cpu(out);
  arma::Mat<eT> out4_cpu(out);
  arma::Mat<eT> out5_cpu(out);
  arma::Mat<eT> out6_cpu(out);
  arma::Mat<eT> out7_cpu(out);
  arma::Mat<eT> out8_cpu(out);

  REQUIRE( out_cpu.is_diagmat() );
  REQUIRE( out2_cpu.is_diagmat() );
  REQUIRE( out3_cpu.is_diagmat() );
  REQUIRE( out4_cpu.is_diagmat() );
  REQUIRE( out5_cpu.is_diagmat() );
  REQUIRE( out6_cpu.is_diagmat() );
  REQUIRE( out7_cpu.is_diagmat() );
  REQUIRE( out8_cpu.is_diagmat() );
  }



// invert a diagonal matrix, where the output is an alias of the input
TEMPLATE_TEST_CASE("alias_diagonal_mat_pinv", "[pinv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(100, 100) + 0.5;
  Mat<eT> old_x(x);
  x = pinv(diagmat(x));

  REQUIRE( x.n_rows == 100 );
  REQUIRE( x.n_cols == 100 );

  Col<eT> x_diag = x.diag();
  Col<eT> ref_diag = (1.0 / old_x.diag());

  REQUIRE( all( abs( x_diag - ref_diag ) < 1e-5 ) );
  }



// convert an inverted diagonal matrix
TEMPLATE_TEST_CASE
  (
  "conv_to_diagonal_mat_pinv",
  "[pinv]",
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

  Mat<eT1> x = randu<Mat<eT1>>(500, 500) + 0.5;

  Mat<eT2> out = conv_to<Mat<eT2>>::from(pinv(diagmat(x)));

  Mat<eT1> out_pre_conv = pinv(diagmat(x));
  Mat<eT2> out_ref = conv_to<Mat<eT2>>::from(out_pre_conv);

  REQUIRE( out.n_rows == out_ref.n_rows );
  REQUIRE( out.n_cols == out_ref.n_cols );

  REQUIRE( all( all( abs( out - out_ref ) < 1e-5 ) ) );
  }




//  - as matrix
//  - as matrix with NaNs
//  - as matrix with lots of other nonzero nondiagonal elements
//  - as matrix with custom tolerance
//  - as matrix with tolerance too larger
//  - as matrix, empty
//  - as matrix with trans
//  - as matrix with scalar mul
//  - as matrix with scalar mul and trans
//  - as matrix, output alias



//
// tests for pinv() on symmetric matrices
//

//  - empty
//  - random, symmetric, but also with op_symmat
//  - random with op_symmat
//  - random with op_symmat and transpose
//  - random with op_symmat and transpose and scalar mul'
//  - random with NaNs
//  - random with custom tolerance
//  - random with tolerance too large
//  - alias



//
// tests for pinv() on general matrices
//

//  - empty
//  - random
//  - random operation
//  - random with NaNs
//  - random with custom tolerance
//  - random with tolerance too large
//  - nonsquare, rows > cols
//  - nonsquare, rows < cols
//  - alias
