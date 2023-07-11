// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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

// first test the "full" size variant

TEMPLATE_TEST_CASE("hardcoded_full_conv_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> A(6);
  A[0] = eT( 1);
  A[1] = eT( 2);
  A[2] = eT( 3);
  A[3] = eT( 4);
  A[4] = eT( 5);
  A[5] = eT( 6);

  Col<eT> B(4);
  B[0] = eT(10);
  B[1] = eT(11);
  B[2] = eT(12);
  B[3] = eT(13);

  // Computed by GNU Octave.
  Col<eT> C_ref(9);
  C_ref[0] = eT(  10);
  C_ref[1] = eT(  31);
  C_ref[2] = eT(  64);
  C_ref[3] = eT( 110);
  C_ref[4] = eT( 156);
  C_ref[5] = eT( 202);
  C_ref[6] = eT( 178);
  C_ref[7] = eT( 137);
  C_ref[8] = eT(  78);

  Col<eT> C1 = conv(A, B, "full");
  Col<eT> C2 = conv(B, A, "full");
  Col<eT> C3 = conv(A, B);
  Col<eT> C4 = conv(B, A);

  REQUIRE( C1.n_elem == 9 );
  REQUIRE( C2.n_elem == 9 );
  REQUIRE( C3.n_elem == 9 );
  REQUIRE( C4.n_elem == 9 );

  REQUIRE( all( abs( C1 - C_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C2 - C_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C3 - C_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C4 - C_ref ) < 1e-5 ) );
  }



TEMPLATE_TEST_CASE("conv_full_equal_size_arma_comparison_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword t = 0; t < 5; ++t)
    {
    const uword dim = std::pow((uword) 2, t + 4);

    Col<eT> A = randu<Col<eT>>(dim) - 0.5;
    Col<eT> B = randu<Col<eT>>(dim) * 2.0;

    Col<eT> C1 = conv2(A, B);
    Col<eT> C2 = conv2(B, A);
    Col<eT> C3 = conv2(A, B, "full");
    Col<eT> C4 = conv2(B, A, "full");

    arma::Col<eT> A_cpu(A);
    arma::Col<eT> B_cpu(B);
    arma::Col<eT> C_ref = arma::conv(A_cpu, B_cpu, "full");

    REQUIRE( C1.n_elem == C_ref.n_elem );
    REQUIRE( C2.n_elem == C_ref.n_elem );
    REQUIRE( C3.n_elem == C_ref.n_elem );
    REQUIRE( C4.n_elem == C_ref.n_elem );

    arma::Col<eT> C1_cpu(C1);
    arma::Col<eT> C2_cpu(C2);
    arma::Col<eT> C3_cpu(C3);
    arma::Col<eT> C4_cpu(C4);

    const eT tol = (is_same_type<eT, float>::value) ? 1e-5 : 1e-8;
    REQUIRE( arma::approx_equal( C1_cpu, C_ref, "absdiff", tol ) );
    REQUIRE( arma::approx_equal( C2_cpu, C_ref, "absdiff", tol ) );
    REQUIRE( arma::approx_equal( C3_cpu, C_ref, "absdiff", tol ) );
    REQUIRE( arma::approx_equal( C4_cpu, C_ref, "absdiff", tol ) );
    }
  }



TEMPLATE_TEST_CASE("1x1_kernel_conv_full_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> X = randu<Col<eT>>(763);
  Col<eT> K(1);
  K[0] = (eT) 1.0;

  // The output should be the same as the input.
  Col<eT> Y1 = conv(X, K);
  Col<eT> Y2 = conv(K, X);
  Col<eT> Y3 = conv(X, K, "full");
  Col<eT> Y4 = conv(K, X, "full");

  REQUIRE( Y1.n_elem == X.n_elem );
  REQUIRE( Y2.n_elem == X.n_elem );
  REQUIRE( Y3.n_elem == X.n_elem );
  REQUIRE( Y4.n_elem == X.n_elem );

  REQUIRE( all( abs( Y1 - X ) < 1e-5 ) );
  REQUIRE( all( abs( Y2 - X ) < 1e-5 ) );
  REQUIRE( all( abs( Y3 - X ) < 1e-5 ) );
  REQUIRE( all( abs( Y4 - X ) < 1e-5 ) );
  }



TEMPLATE_TEST_CASE("conv_full_random_sizes_arma_comparison_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword t = 0; t < 5; ++t)
    {
    const Mat<uword> sizes = randi<Mat<uword>>(2, distr_param(200, 20000));

    const uword A_elem = sizes[0];
    const uword B_elem = sizes[1];

    Col<eT> A = randu<Col<eT>>(A_elem);
    Col<eT> B = randi<Col<eT>>(B_elem, distr_param(-50, 50));

    Col<eT> C1 = conv(A, B);
    Col<eT> C2 = conv(B, A);
    Col<eT> C3 = conv(A, B, "full");
    Col<eT> C4 = conv(B, A, "full");

    arma::Col<eT> A_cpu(A);
    arma::Col<eT> B_cpu(B);
    arma::Col<eT> C_ref = arma::conv(A_cpu, B_cpu);

    REQUIRE( C1.n_elem == C_ref.n_elem );
    REQUIRE( C2.n_elem == C_ref.n_elem );
    REQUIRE( C3.n_elem == C_ref.n_elem );
    REQUIRE( C4.n_elem == C_ref.n_elem );

    arma::Col<eT> C1_cpu(C1);
    arma::Col<eT> C2_cpu(C2);
    arma::Col<eT> C3_cpu(C3);
    arma::Col<eT> C4_cpu(C4);

    const eT tol = (is_same_type<eT, float>::value) ? 1e-4 : 1e-8;
    // Since the matrices can get big, we'll use a slightly relaxed check that accounts for large norms.
    REQUIRE( arma::norm(C1_cpu - C_ref) / arma::norm(C1_cpu) < tol );
    REQUIRE( arma::norm(C2_cpu - C_ref) / arma::norm(C2_cpu) < tol );
    REQUIRE( arma::norm(C3_cpu - C_ref) / arma::norm(C3_cpu) < tol );
    REQUIRE( arma::norm(C4_cpu - C_ref) / arma::norm(C4_cpu) < tol );
    }
  }



TEST_CASE("conv_full_empty_test", "[conv]")
  {
  fvec a;
  fvec b = randu<fvec>(3);

  fvec c1 = conv(a, b);
  fvec c2 = conv(b, a);
  fvec c3 = conv(a, b, "full");
  fvec c4 = conv(b, a, "full");

  REQUIRE( c1.n_elem == 0 );
  REQUIRE( c2.n_elem == 0 );
  REQUIRE( c3.n_elem == 0 );
  REQUIRE( c4.n_elem == 0 );

  // Now try with both matrices empty.
  b.set_size(0, 0);

  c1 = conv(a, b);
  c2 = conv(b, a);
  c3 = conv(a, b, "full");
  c4 = conv(b, a, "full");

  REQUIRE( c1.n_elem == 0 );
  REQUIRE( c2.n_elem == 0 );
  REQUIRE( c3.n_elem == 0 );
  REQUIRE( c4.n_elem == 0 );
  }



TEST_CASE("conv_full_alias_test", "[conv]")
  {
  fvec a = randu<fvec>(2000);
  fvec b = randu<fvec>(50);

  fvec a_orig(a);

  a = conv(a, b);
  fvec a_ref = conv(a_orig, b);

  REQUIRE( a.n_elem == a_ref.n_elem );
  REQUIRE( all( abs( a - a_ref ) < 1e-5 ) );

  a = a_orig;
  a = conv(b, a);

  REQUIRE( a.n_elem == a_ref.n_elem );
  REQUIRE( all( abs( a - a_ref ) < 1e-5 ) );

  a = a_orig;
  a = conv(a, b, "full");

  REQUIRE( a.n_elem == a_ref.n_elem );
  REQUIRE( all( abs( a - a_ref ) < 1e-5 ) );

  a = a_orig;
  a = conv(b, a, "full");

  REQUIRE( a.n_elem == a_ref.n_elem );
  REQUIRE( all( abs( a - a_ref ) < 1e-5 ) );
  }



TEMPLATE_TEST_CASE("conv_full_expr_inputs_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> X = randu<Col<eT>>(15001);
  Mat<eT> Y = randu<Mat<eT>>(25, 10);

  Col<eT> C1 = conv(X + 3, vectorise(trans(2 * Y)));
  Col<eT> C2 = conv(vectorise(trans(2 * Y)), X + 3);
  Col<eT> C3 = conv2(X + 3, vectorise(trans(2 * Y)), "full");
  Col<eT> C4 = conv2(vectorise(trans(2 * Y)), X + 3, "full");

  Col<eT> X_expr_ref(X + 3);
  Col<eT> Y_expr_ref(vectorise(trans(2 * Y)));

  Col<eT> C_ref = conv2(X_expr_ref, Y_expr_ref);

  REQUIRE( C1.n_elem == C_ref.n_elem );
  REQUIRE( C2.n_elem == C_ref.n_elem );
  REQUIRE( C3.n_elem == C_ref.n_elem );
  REQUIRE( C4.n_elem == C_ref.n_elem );

  REQUIRE( all( abs( C1 - C_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C2 - C_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C3 - C_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C4 - C_ref ) < 1e-5 ) );
  }



TEMPLATE_TEST_CASE("col_vs_row_conv_full_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A = randu<Mat<eT>>(512, 1);
  Mat<eT> B = randu<Mat<eT>>(1, 128);

  Col<eT> C1 = conv(A, B);
  Row<eT> C2 = conv(B, A);
  Col<eT> C3 = conv(A, B, "full");
  Row<eT> C4 = conv(B, A, "full");

  arma::Mat<eT> A_cpu(A);
  arma::Mat<eT> B_cpu(B);

  arma::Col<eT> C_ref = arma::conv(A_cpu, B_cpu, "full");

  REQUIRE( C1.n_elem == C_ref.n_elem );
  REQUIRE( C2.n_elem == C_ref.n_elem );
  REQUIRE( C3.n_elem == C_ref.n_elem );
  REQUIRE( C4.n_elem == C_ref.n_elem );

  arma::Col<eT> C1_cpu(C1);
  arma::Row<eT> C2_cpu(C2);
  arma::Col<eT> C3_cpu(C3);
  arma::Row<eT> C4_cpu(C4);

  REQUIRE( arma::approx_equal( C1_cpu, C_ref,     "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( C2_cpu, C_ref.t(), "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( C3_cpu, C_ref,     "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( C4_cpu, C_ref.t(), "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("conv_full_mat_input_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A = randu<Mat<eT>>(20, 50);
  Mat<eT> B = randu<Mat<eT>>(120, 130);

  Row<eT> C1 = conv(A, B);
  Row<eT> C2 = conv(B, A);
  Row<eT> C3 = conv(A, B, "full");
  Row<eT> C4 = conv(B, A, "full");

  Row<eT> A_vec = vectorise(A).t();
  Row<eT> B_vec = vectorise(B).t();

  Row<eT> C_ref = conv(A_vec, B_vec);

  REQUIRE( C1.n_elem == C_ref.n_elem );
  REQUIRE( C2.n_elem == C_ref.n_elem );
  REQUIRE( C3.n_elem == C_ref.n_elem );
  REQUIRE( C4.n_elem == C_ref.n_elem );

  REQUIRE( all( abs( C1 - C_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C2 - C_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C3 - C_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C4 - C_ref ) < 1e-5 ) );
  }



// now test the "same" size variant

TEMPLATE_TEST_CASE("hardcoded_same_conv_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> A(6);
  A[0] = eT( 1);
  A[1] = eT( 2);
  A[2] = eT( 3);
  A[3] = eT( 4);
  A[4] = eT( 5);
  A[5] = eT( 6);

  Col<eT> B(4);
  B[0] = eT(10);
  B[1] = eT(11);
  B[2] = eT(12);
  B[3] = eT(13);

  // Computed by GNU Octave.
  Col<eT> C1_ref(6);
  C1_ref[0] = eT(  64);
  C1_ref[1] = eT( 110);
  C1_ref[2] = eT( 156);
  C1_ref[3] = eT( 202);
  C1_ref[4] = eT( 178);
  C1_ref[5] = eT( 137);

  Col<eT> C2_ref(4);
  C2_ref[0] = eT( 110);
  C2_ref[1] = eT( 156);
  C2_ref[2] = eT( 202);
  C2_ref[3] = eT( 178);

  Col<eT> C1 = conv(A, B, "same");
  Col<eT> C2 = conv(B, A, "same");

  REQUIRE( C1.n_elem == 6 );
  REQUIRE( C2.n_elem == 4 );

  REQUIRE( all( abs( C1 - C1_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C2 - C2_ref ) < 1e-5 ) );
  }



TEMPLATE_TEST_CASE("conv_same_equal_size_arma_comparison_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword t = 0; t < 5; ++t)
    {
    const uword dim = std::pow((uword) 2, t + 4);

    Col<eT> A = randu<Col<eT>>(dim) - 0.5;
    Col<eT> B = randu<Col<eT>>(dim) * 2.0;

    Col<eT> C1 = conv(A, B, "same");
    Col<eT> C2 = conv(B, A, "same");

    arma::Col<eT> A_cpu(A);
    arma::Col<eT> B_cpu(B);
    arma::Col<eT> C1_ref = arma::conv(A_cpu, B_cpu, "same");
    arma::Col<eT> C2_ref = arma::conv(B_cpu, A_cpu, "same");

    REQUIRE( C1.n_elem == C1_ref.n_elem );
    REQUIRE( C2.n_elem == C2_ref.n_elem );

    arma::Col<eT> C1_cpu(C1);
    arma::Col<eT> C2_cpu(C2);

    const eT tol = (is_same_type<eT, float>::value) ? 1e-5 : 1e-8;
    REQUIRE( arma::approx_equal( C1_cpu, C1_ref, "absdiff", tol ) );
    REQUIRE( arma::approx_equal( C2_cpu, C2_ref, "absdiff", tol ) );
    }
  }



TEMPLATE_TEST_CASE("1x1_kernel_conv_same_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> X = randu<Col<eT>>(257);
  Col<eT> K(1);
  K[0] = (eT) 1.0;

  // The output should be the same as the input.
  Col<eT> Y1 = conv(X, K, "same");

  REQUIRE( Y1.n_elem == X.n_elem );
  REQUIRE( all( abs( Y1 - X ) < 1e-5 ) );

  // The output should only have a single element.
  Col<eT> Y2 = conv(K, X, "same");

  REQUIRE( Y2.n_elem == 1 );
  REQUIRE( eT(Y2[0]) == Approx(eT(X[128])) );
  }



TEMPLATE_TEST_CASE("conv_same_random_sizes_arma_comparison_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword t = 0; t < 5; ++t)
    {
    const Mat<uword> sizes = randi<Mat<uword>>(2, distr_param(200, 20000));

    const uword A_elem = sizes[0];
    const uword B_elem = sizes[1];

    Col<eT> A = randu<Col<eT>>(A_elem);
    Col<eT> B = randi<Col<eT>>(B_elem, distr_param(-50, 50));

    Col<eT> C1 = conv(A, B, "same");
    Col<eT> C2 = conv(B, A, "same");

    arma::Col<eT> A_cpu(A);
    arma::Col<eT> B_cpu(B);
    arma::Col<eT> C1_ref = arma::conv(A_cpu, B_cpu, "same");
    arma::Col<eT> C2_ref = arma::conv(B_cpu, A_cpu, "same");

    REQUIRE( C1.n_elem == C1_ref.n_elem );
    REQUIRE( C2.n_elem == C2_ref.n_elem );

    arma::Col<eT> C1_cpu(C1);
    arma::Col<eT> C2_cpu(C2);

    const eT tol = (is_same_type<eT, float>::value) ? 1e-4 : 1e-6;
    // Since the matrices can get big, we'll use a slightly relaxed check that accounts for large norms.
    REQUIRE( arma::norm(C1_cpu - C1_ref) / arma::norm(C1_cpu) < tol );
    REQUIRE( arma::norm(C2_cpu - C2_ref) / arma::norm(C2_cpu) < tol );
    }
  }



TEST_CASE("conv_same_empty_test", "[conv]")
  {
  fmat a;
  fmat b = randu<fmat>(3, 3);

  fmat c1 = conv(a, b, "same");
  fmat c2 = conv(b, a, "same");

  REQUIRE( c1.n_elem == 0 );
  REQUIRE( c2.n_elem == 0 );

  // Now try with both matrices empty.
  b.set_size(0, 0);

  c1 = conv(a, b, "same");
  c2 = conv(b, a, "same");

  REQUIRE( c1.n_elem == 0 );
  REQUIRE( c2.n_elem == 0 );
  }



TEST_CASE("conv_same_alias_test", "[conv]")
  {
  fvec a = randu<fvec>(2000);
  fvec b = randu<fvec>(50);

  fvec a_orig(a);

  a = conv(a, b, "same");
  fvec a_ref = conv(a_orig, b, "same");

  REQUIRE( a.n_elem == a_ref.n_elem );
  REQUIRE( all( abs( a - a_ref ) < 1e-5 ) );

  a = a_orig;
  a = conv(b, a, "same");
  a_ref = conv(b, a_orig, "same");

  REQUIRE( a.n_elem == a_ref.n_elem );
  REQUIRE( all( abs( a - a_ref ) < 1e-5 ) );
  }



TEMPLATE_TEST_CASE("conv_same_expr_inputs_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X = randu<Mat<eT>>(255, 500);
  Col<eT> Y = randu<Col<eT>>(25);

  Col<eT> C1 = conv(vectorise(X + 3), trans(2 * Y), "same");
  Row<eT> C2 = conv(trans(2 * Y), vectorise(X + 3), "same");

  Col<eT> X_expr_ref(vectorise(X + 3));
  Row<eT> Y_expr_ref(trans(2 * Y));

  Col<eT> C1_ref = conv(X_expr_ref, Y_expr_ref, "same");
  Row<eT> C2_ref = conv(Y_expr_ref, X_expr_ref, "same");

  REQUIRE( C1.n_elem == C1_ref.n_elem );
  REQUIRE( C2.n_elem == C2_ref.n_elem );

  REQUIRE( all( abs( C1 - C1_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C2 - C2_ref ) < 1e-5 ) );
  }



TEMPLATE_TEST_CASE("col_vs_row_conv_same_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A = randu<Mat<eT>>(512, 1);
  Mat<eT> B = randu<Mat<eT>>(1, 128);

  Col<eT> C1 = conv(A, B, "same");
  Row<eT> C2 = conv(B, A, "same");

  arma::Mat<eT> A_cpu(A);
  arma::Mat<eT> B_cpu(B);

  arma::Col<eT> C1_ref = arma::conv(A_cpu, B_cpu, "same");
  arma::Row<eT> C2_ref = arma::conv(B_cpu, A_cpu, "same");

  REQUIRE( C1.n_elem == C1_ref.n_elem );
  REQUIRE( C2.n_elem == C2_ref.n_elem );

  arma::Col<eT> C1_cpu(C1);
  arma::Row<eT> C2_cpu(C2);

  REQUIRE( arma::approx_equal( C1_cpu, C1_ref, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( C2_cpu, C2_ref, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("conv_same_mat_input_test", "[conv]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A = randu<Mat<eT>>(20, 50);
  Mat<eT> B = randu<Mat<eT>>(120, 130);

  Row<eT> C1 = conv(A, B, "same");
  Row<eT> C2 = conv(B, A, "same");

  Row<eT> A_vec = vectorise(A).t();
  Row<eT> B_vec = vectorise(B).t();

  Row<eT> C1_ref = conv(A_vec, B_vec, "same");
  Row<eT> C2_ref = conv(B_vec, A_vec, "same");

  REQUIRE( C1.n_elem == C1_ref.n_elem );
  REQUIRE( C2.n_elem == C2_ref.n_elem );

  REQUIRE( all( abs( C1 - C1_ref ) < 1e-5 ) );
  REQUIRE( all( abs( C2 - C2_ref ) < 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "conv_conv_to_test",
  "[conv]",
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

  Col<eT1> A = randu<Col<eT1>>(15000);
  Col<eT1> B = randu<Col<eT1>>(255);

  Col<eT2> C1 = conv_to<Col<eT2>>::from(conv(A, B));
  Col<eT2> C2 = conv_to<Col<eT2>>::from(conv(B, A));
  Col<eT2> C3 = conv_to<Col<eT2>>::from(conv(A, B, "full"));
  Col<eT2> C4 = conv_to<Col<eT2>>::from(conv(B, A, "full"));
  Col<eT2> C5 = conv_to<Col<eT2>>::from(conv(A, B, "same"));
  Col<eT2> C6 = conv_to<Col<eT2>>::from(conv(B, A, "same"));

  Col<eT1> C1_pre_conv = conv2(A, B);
  Col<eT1> C2_pre_conv = conv2(B, A);
  Col<eT1> C3_pre_conv = conv2(A, B, "full");
  Col<eT1> C4_pre_conv = conv2(B, A, "full");
  Col<eT1> C5_pre_conv = conv2(A, B, "same");
  Col<eT1> C6_pre_conv = conv2(B, A, "same");

  Col<eT2> C1_ref = conv_to<Col<eT2>>::from(C1_pre_conv);
  Col<eT2> C2_ref = conv_to<Col<eT2>>::from(C2_pre_conv);
  Col<eT2> C3_ref = conv_to<Col<eT2>>::from(C3_pre_conv);
  Col<eT2> C4_ref = conv_to<Col<eT2>>::from(C4_pre_conv);
  Col<eT2> C5_ref = conv_to<Col<eT2>>::from(C5_pre_conv);
  Col<eT2> C6_ref = conv_to<Col<eT2>>::from(C6_pre_conv);

  REQUIRE( C1.n_elem == C1_ref.n_elem );
  REQUIRE( C2.n_elem == C2_ref.n_elem );
  REQUIRE( C3.n_elem == C3_ref.n_elem );
  REQUIRE( C4.n_elem == C4_ref.n_elem );
  REQUIRE( C5.n_elem == C5_ref.n_elem );
  REQUIRE( C6.n_elem == C6_ref.n_elem );

  REQUIRE( all(abs(C1 - C1_ref) < 1e-5) );
  REQUIRE( all(abs(C2 - C2_ref) < 1e-5) );
  REQUIRE( all(abs(C3 - C3_ref) < 1e-5) );
  REQUIRE( all(abs(C4 - C4_ref) < 1e-5) );
  REQUIRE( all(abs(C5 - C5_ref) < 1e-5) );
  REQUIRE( all(abs(C6 - C6_ref) < 1e-5) );
  }
