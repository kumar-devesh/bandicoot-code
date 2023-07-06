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

TEMPLATE_TEST_CASE("hardcoded_full_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(4, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);
  A(3, 0) = eT(13);
  A(3, 1) = eT(14);
  A(3, 2) = eT(15);
  A(3, 3) = eT(16);

  Mat<eT> B(3, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);
  B(2, 0) = eT(16);
  B(2, 1) = eT(17);
  B(2, 2) = eT(18);

  // Computed by GNU Octave.
  Mat<eT> C_ref(6, 6);
  C_ref(0, 0) = eT(  10);
  C_ref(0, 1) = eT(  31);
  C_ref(0, 2) = eT(  64);
  C_ref(0, 3) = eT(  97);
  C_ref(0, 4) = eT(  80);
  C_ref(0, 5) = eT(  48);
  C_ref(1, 0) = eT(  63);
  C_ref(1, 1) = eT( 155);
  C_ref(1, 2) = eT( 278);
  C_ref(1, 3) = eT( 353);
  C_ref(1, 4) = eT( 273);
  C_ref(1, 5) = eT( 156);
  C_ref(2, 0) = eT( 171);
  C_ref(2, 1) = eT( 396);
  C_ref(2, 2) = eT( 678);
  C_ref(2, 3) = eT( 804);
  C_ref(2, 4) = eT( 603);
  C_ref(2, 5) = eT( 336);
  C_ref(3, 0) = eT( 327);
  C_ref(3, 1) = eT( 720);
  C_ref(3, 2) = eT(1182);
  C_ref(3, 3) = eT(1308);
  C_ref(3, 4) = eT( 951);
  C_ref(3, 5) = eT( 516);
  C_ref(4, 0) = eT( 313);
  C_ref(4, 1) = eT( 677);
  C_ref(4, 2) = eT(1094);
  C_ref(4, 3) = eT(1187);
  C_ref(4, 4) = eT( 851);
  C_ref(4, 5) = eT( 456);
  C_ref(5, 0) = eT( 208);
  C_ref(5, 1) = eT( 445);
  C_ref(5, 2) = eT( 712);
  C_ref(5, 3) = eT( 763);
  C_ref(5, 4) = eT( 542);
  C_ref(5, 5) = eT( 288);

  Mat<eT> C1 = conv2(A, B, "full");
  Mat<eT> C2 = conv2(B, A, "full");
  Mat<eT> C3 = conv2(A, B);
  Mat<eT> C4 = conv2(B, A);

  REQUIRE( C1.n_rows == 6 );
  REQUIRE( C1.n_cols == 6 );
  REQUIRE( C2.n_rows == 6 );
  REQUIRE( C2.n_cols == 6 );
  REQUIRE( C3.n_rows == 6 );
  REQUIRE( C3.n_cols == 6 );
  REQUIRE( C4.n_rows == 6 );
  REQUIRE( C4.n_cols == 6 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C2 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C3 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C4 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_full_k_rows_gt_cols_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(4, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);
  A(3, 0) = eT(13);
  A(3, 1) = eT(14);
  A(3, 2) = eT(15);
  A(3, 3) = eT(16);

  Mat<eT> B(3, 2);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(1, 0) = eT(12);
  B(1, 1) = eT(13);
  B(2, 0) = eT(14);
  B(2, 1) = eT(15);

  // Computed by GNU Octave.
  Mat<eT> C_ref(6, 5);
  C_ref(0, 0) = eT( 10);
  C_ref(0, 1) = eT( 31);
  C_ref(0, 2) = eT( 52);
  C_ref(0, 3) = eT( 73);
  C_ref(0, 4) = eT( 44);
  C_ref(1, 0) = eT( 62);
  C_ref(1, 1) = eT(152);
  C_ref(1, 2) = eT(198);
  C_ref(1, 3) = eT(244);
  C_ref(1, 4) = eT(140);
  C_ref(2, 0) = eT(164);
  C_ref(2, 1) = eT(379);
  C_ref(2, 2) = eT(454);
  C_ref(2, 3) = eT(529);
  C_ref(2, 4) = eT(296);
  C_ref(3, 0) = eT(308);
  C_ref(3, 1) = eT(679);
  C_ref(3, 2) = eT(754);
  C_ref(3, 3) = eT(829);
  C_ref(3, 4) = eT(452);
  C_ref(4, 0) = eT(282);
  C_ref(4, 1) = eT(612);
  C_ref(4, 2) = eT(666);
  C_ref(4, 3) = eT(720);
  C_ref(4, 4) = eT(388);
  C_ref(5, 0) = eT(182);
  C_ref(5, 1) = eT(391);
  C_ref(5, 2) = eT(420);
  C_ref(5, 3) = eT(449);
  C_ref(5, 4) = eT(240);

  Mat<eT> C1 = conv2(A, B, "full");
  Mat<eT> C2 = conv2(B, A, "full");
  Mat<eT> C3 = conv2(A, B);
  Mat<eT> C4 = conv2(B, A);

  REQUIRE( C1.n_rows == 6 );
  REQUIRE( C1.n_cols == 5 );
  REQUIRE( C2.n_rows == 6 );
  REQUIRE( C2.n_cols == 5 );
  REQUIRE( C3.n_rows == 6 );
  REQUIRE( C3.n_cols == 5 );
  REQUIRE( C4.n_rows == 6 );
  REQUIRE( C4.n_cols == 5 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C2 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C3 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C4 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_full_k_rows_lt_cols_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(4, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);
  A(3, 0) = eT(13);
  A(3, 1) = eT(14);
  A(3, 2) = eT(15);
  A(3, 3) = eT(16);

  Mat<eT> B(2, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);

  // Computed by GNU Octave.
  Mat<eT> C_ref(5, 6);
  C_ref(0, 0) = eT( 10);
  C_ref(0, 1) = eT( 31);
  C_ref(0, 2) = eT( 64);
  C_ref(0, 3) = eT( 97);
  C_ref(0, 4) = eT( 80);
  C_ref(0, 5) = eT( 48);
  C_ref(1, 0) = eT( 63);
  C_ref(1, 1) = eT(155);
  C_ref(1, 2) = eT(278);
  C_ref(1, 3) = eT(353);
  C_ref(1, 4) = eT(273);
  C_ref(1, 5) = eT(156);
  C_ref(2, 0) = eT(155);
  C_ref(2, 1) = eT(347);
  C_ref(2, 2) = eT(578);
  C_ref(2, 3) = eT(653);
  C_ref(2, 4) = eT(481);
  C_ref(2, 5) = eT(264);
  C_ref(3, 0) = eT(247);
  C_ref(3, 1) = eT(539);
  C_ref(3, 2) = eT(878);
  C_ref(3, 3) = eT(953);
  C_ref(3, 4) = eT(689);
  C_ref(3, 5) = eT(372);
  C_ref(4, 0) = eT(169);
  C_ref(4, 1) = eT(364);
  C_ref(4, 2) = eT(586);
  C_ref(4, 3) = eT(628);
  C_ref(4, 4) = eT(449);
  C_ref(4, 5) = eT(240);

  Mat<eT> C1 = conv2(A, B, "full");
  Mat<eT> C2 = conv2(B, A, "full");
  Mat<eT> C3 = conv2(A, B);
  Mat<eT> C4 = conv2(B, A);

  REQUIRE( C1.n_rows == 5 );
  REQUIRE( C1.n_cols == 6 );
  REQUIRE( C2.n_rows == 5 );
  REQUIRE( C2.n_cols == 6 );
  REQUIRE( C3.n_rows == 5 );
  REQUIRE( C3.n_cols == 6 );
  REQUIRE( C4.n_rows == 5 );
  REQUIRE( C4.n_cols == 6 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C2 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C3 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C4 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_full_A_rows_gt_cols_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(4, 3);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(1, 0) = eT( 4);
  A(1, 1) = eT( 5);
  A(1, 2) = eT( 6);
  A(2, 0) = eT( 7);
  A(2, 1) = eT( 8);
  A(2, 2) = eT( 9);
  A(3, 0) = eT(10);
  A(3, 1) = eT(11);
  A(3, 2) = eT(12);

  Mat<eT> B(3, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);
  B(2, 0) = eT(16);
  B(2, 1) = eT(17);
  B(2, 2) = eT(18);

  Mat<eT> C_ref(6, 5);
  C_ref(0, 0) = eT( 10);
  C_ref(0, 1) = eT( 31);
  C_ref(0, 2) = eT( 64);
  C_ref(0, 3) = eT( 57);
  C_ref(0, 4) = eT( 36);
  C_ref(1, 0) = eT( 53);
  C_ref(1, 1) = eT(134);
  C_ref(1, 2) = eT(245);
  C_ref(1, 3) = eT(198);
  C_ref(1, 4) = eT(117);
  C_ref(2, 0) = eT(138);
  C_ref(2, 1) = eT(327);
  C_ref(2, 2) = eT(570);
  C_ref(2, 3) = eT(441);
  C_ref(2, 4) = eT(252);
  C_ref(3, 0) = eT(255);
  C_ref(3, 1) = eT(570);
  C_ref(3, 2) = eT(948);
  C_ref(3, 3) = eT(702);
  C_ref(3, 4) = eT(387);
  C_ref(4, 0) = eT(242);
  C_ref(4, 1) = eT(530);
  C_ref(4, 2) = eT(866);
  C_ref(4, 3) = eT(630);
  C_ref(4, 4) = eT(342);
  C_ref(5, 0) = eT(160);
  C_ref(5, 1) = eT(346);
  C_ref(5, 2) = eT(559);
  C_ref(5, 3) = eT(402);
  C_ref(5, 4) = eT(216);

  Mat<eT> C1 = conv2(A, B, "full");
  Mat<eT> C2 = conv2(B, A, "full");
  Mat<eT> C3 = conv2(A, B);
  Mat<eT> C4 = conv2(B, A);

  REQUIRE( C1.n_rows == 6 );
  REQUIRE( C1.n_cols == 5 );
  REQUIRE( C2.n_rows == 6 );
  REQUIRE( C2.n_cols == 5 );
  REQUIRE( C3.n_rows == 6 );
  REQUIRE( C3.n_cols == 5 );
  REQUIRE( C4.n_rows == 6 );
  REQUIRE( C4.n_cols == 5 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C2 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C3 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C4 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_full_A_rows_lt_cols_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(3, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);

  Mat<eT> B(3, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);
  B(2, 0) = eT(16);
  B(2, 1) = eT(17);
  B(2, 2) = eT(18);

  // Computed by GNU Octave.
  Mat<eT> C_ref(5, 6);
  C_ref(0, 0) = eT(  10);
  C_ref(0, 1) = eT(  31);
  C_ref(0, 2) = eT(  64);
  C_ref(0, 3) = eT(  97);
  C_ref(0, 4) = eT(  80);
  C_ref(0, 5) = eT(  48);
  C_ref(1, 0) = eT(  63);
  C_ref(1, 1) = eT( 155);
  C_ref(1, 2) = eT( 278);
  C_ref(1, 3) = eT( 353);
  C_ref(1, 4) = eT( 273);
  C_ref(1, 5) = eT( 156);
  C_ref(2, 0) = eT( 171);
  C_ref(2, 1) = eT( 396);
  C_ref(2, 2) = eT( 678);
  C_ref(2, 3) = eT( 804);
  C_ref(2, 4) = eT( 603);
  C_ref(2, 5) = eT( 336);
  C_ref(3, 0) = eT( 197);
  C_ref(3, 1) = eT( 437);
  C_ref(3, 2) = eT( 722);
  C_ref(3, 3) = eT( 815);
  C_ref(3, 4) = eT( 595);
  C_ref(3, 5) = eT( 324);
  C_ref(4, 0) = eT( 144);
  C_ref(4, 1) = eT( 313);
  C_ref(4, 2) = eT( 508);
  C_ref(4, 3) = eT( 559);
  C_ref(4, 4) = eT( 402);
  C_ref(4, 5) = eT( 216);

  Mat<eT> C1 = conv2(A, B, "full");
  Mat<eT> C2 = conv2(B, A, "full");
  Mat<eT> C3 = conv2(A, B);
  Mat<eT> C4 = conv2(B, A);

  REQUIRE( C1.n_rows == 5 );
  REQUIRE( C1.n_cols == 6 );
  REQUIRE( C2.n_rows == 5 );
  REQUIRE( C2.n_cols == 6 );
  REQUIRE( C3.n_rows == 5 );
  REQUIRE( C3.n_cols == 6 );
  REQUIRE( C4.n_rows == 5 );
  REQUIRE( C4.n_cols == 6 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C2 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C3 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C4 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_full_all_dims_unequal_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(5, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);
  A(3, 0) = eT(13);
  A(3, 1) = eT(14);
  A(3, 2) = eT(15);
  A(3, 3) = eT(16);
  A(4, 0) = eT(17);
  A(4, 1) = eT(18);
  A(4, 2) = eT(19);
  A(4, 3) = eT(20);

  Mat<eT> B(2, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);

  // Computed by GNU Octave.
  Mat<eT> C_ref(6, 6);
  C_ref(0, 0) = eT(  10);
  C_ref(0, 1) = eT(  31);
  C_ref(0, 2) = eT(  64);
  C_ref(0, 3) = eT(  97);
  C_ref(0, 4) = eT(  80);
  C_ref(0, 5) = eT(  48);
  C_ref(1, 0) = eT(  63);
  C_ref(1, 1) = eT( 155);
  C_ref(1, 2) = eT( 278);
  C_ref(1, 3) = eT( 353);
  C_ref(1, 4) = eT( 273);
  C_ref(1, 5) = eT( 156);
  C_ref(2, 0) = eT( 155);
  C_ref(2, 1) = eT( 347);
  C_ref(2, 2) = eT( 578);
  C_ref(2, 3) = eT( 653);
  C_ref(2, 4) = eT( 481);
  C_ref(2, 5) = eT( 264);
  C_ref(3, 0) = eT( 247);
  C_ref(3, 1) = eT( 539);
  C_ref(3, 2) = eT( 878);
  C_ref(3, 3) = eT( 953);
  C_ref(3, 4) = eT( 689);
  C_ref(3, 5) = eT( 372);
  C_ref(4, 0) = eT( 339);
  C_ref(4, 1) = eT( 731);
  C_ref(4, 2) = eT(1178);
  C_ref(4, 3) = eT(1253);
  C_ref(4, 4) = eT( 897);
  C_ref(4, 5) = eT( 480);
  C_ref(5, 0) = eT( 221);
  C_ref(5, 1) = eT( 472);
  C_ref(5, 2) = eT( 754);
  C_ref(5, 3) = eT( 796);
  C_ref(5, 4) = eT( 565);
  C_ref(5, 5) = eT( 300);

  Mat<eT> C1 = conv2(A, B, "full");
  Mat<eT> C2 = conv2(B, A, "full");
  Mat<eT> C3 = conv2(A, B);
  Mat<eT> C4 = conv2(B, A);

  REQUIRE( C1.n_rows == 6 );
  REQUIRE( C1.n_cols == 6 );
  REQUIRE( C2.n_rows == 6 );
  REQUIRE( C2.n_cols == 6 );
  REQUIRE( C3.n_rows == 6 );
  REQUIRE( C3.n_cols == 6 );
  REQUIRE( C4.n_rows == 6 );
  REQUIRE( C4.n_cols == 6 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C2 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C3 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C4 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("conv2_full_square_arma_comparison_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword t = 0; t < 5; ++t)
    {
    const uword A_dim = std::pow((uword) 2, t + 4);
    const uword B_dim = t + 3;

    Mat<eT> A = randu<Mat<eT>>(A_dim, A_dim) - 0.5;
    Mat<eT> B = randu<Mat<eT>>(B_dim, B_dim) * 2.0;

    Mat<eT> C1 = conv2(A, B);
    Mat<eT> C2 = conv2(B, A);
    Mat<eT> C3 = conv2(A, B, "full");
    Mat<eT> C4 = conv2(B, A, "full");

    arma::Mat<eT> A_cpu(A);
    arma::Mat<eT> B_cpu(B);
    arma::Mat<eT> C_ref = arma::conv2(A_cpu, B_cpu, "full");

    REQUIRE( C1.n_rows == C_ref.n_rows );
    REQUIRE( C1.n_cols == C_ref.n_cols );
    REQUIRE( C2.n_rows == C_ref.n_rows );
    REQUIRE( C2.n_cols == C_ref.n_cols );
    REQUIRE( C3.n_rows == C_ref.n_rows );
    REQUIRE( C3.n_cols == C_ref.n_cols );
    REQUIRE( C4.n_rows == C_ref.n_rows );
    REQUIRE( C4.n_cols == C_ref.n_cols );

    arma::Mat<eT> C1_cpu(C1);
    arma::Mat<eT> C2_cpu(C2);
    arma::Mat<eT> C3_cpu(C3);
    arma::Mat<eT> C4_cpu(C4);

    const eT tol = (is_same_type<eT, float>::value) ? 1e-5 : 1e-8;
    REQUIRE( arma::approx_equal( C1_cpu, C_ref, "absdiff", tol ) );
    REQUIRE( arma::approx_equal( C2_cpu, C_ref, "absdiff", tol ) );
    REQUIRE( arma::approx_equal( C3_cpu, C_ref, "absdiff", tol ) );
    REQUIRE( arma::approx_equal( C4_cpu, C_ref, "absdiff", tol ) );
    }
  }



TEMPLATE_TEST_CASE("1x1_kernel_conv2_full_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X = randu<Mat<eT>>(763, 256);
  Mat<eT> K(1, 1);
  K(0, 0) = (eT) 1.0;

  // The output should be the same as the input.
  Mat<eT> Y1 = conv2(X, K);
  Mat<eT> Y2 = conv2(K, X);
  Mat<eT> Y3 = conv2(X, K, "full");
  Mat<eT> Y4 = conv2(K, X, "full");

  REQUIRE( Y1.n_rows == X.n_rows );
  REQUIRE( Y1.n_cols == X.n_cols );
  REQUIRE( Y2.n_rows == X.n_rows );
  REQUIRE( Y2.n_cols == X.n_cols );
  REQUIRE( Y3.n_rows == X.n_rows );
  REQUIRE( Y3.n_cols == X.n_cols );
  REQUIRE( Y4.n_rows == X.n_rows );
  REQUIRE( Y4.n_cols == X.n_cols );

  REQUIRE( all( all( abs( Y1 - X ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( Y2 - X ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( Y3 - X ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( Y4 - X ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("conv2_full_random_sizes_arma_comparison_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword t = 0; t < 5; ++t)
    {
    const Mat<uword> sizes = randi<Mat<uword>>(4, distr_param(50, 200));

    const uword A_rows = sizes[0];
    const uword A_cols = sizes[1];
    const uword B_rows = sizes[2];
    const uword B_cols = sizes[3];

    Mat<eT> A = randu<Mat<eT>>(A_rows, A_cols);
    Mat<eT> B = randi<Mat<eT>>(B_rows, B_cols, distr_param(-50, 50));

    Mat<eT> C1 = conv2(A, B);
    Mat<eT> C2 = conv2(B, A);
    Mat<eT> C3 = conv2(A, B, "full");
    Mat<eT> C4 = conv2(B, A, "full");

    arma::Mat<eT> A_cpu(A);
    arma::Mat<eT> B_cpu(B);
    arma::Mat<eT> C_ref = arma::conv2(A_cpu, B_cpu);

    REQUIRE( C1.n_rows == C_ref.n_rows );
    REQUIRE( C1.n_cols == C_ref.n_cols );
    REQUIRE( C2.n_rows == C_ref.n_rows );
    REQUIRE( C2.n_cols == C_ref.n_cols );
    REQUIRE( C3.n_rows == C_ref.n_rows );
    REQUIRE( C3.n_cols == C_ref.n_cols );
    REQUIRE( C4.n_rows == C_ref.n_rows );
    REQUIRE( C4.n_cols == C_ref.n_cols );

    arma::Mat<eT> C1_cpu(C1);
    arma::Mat<eT> C2_cpu(C2);
    arma::Mat<eT> C3_cpu(C3);
    arma::Mat<eT> C4_cpu(C4);

    const eT tol = (is_same_type<eT, float>::value) ? 1e-6 : 1e-12;
    // Since the matrices can get big, we'll use a slightly relaxed check that accounts for large norms.
    REQUIRE( arma::norm(C1_cpu - C_ref) / arma::norm(C1_cpu) < tol );
    REQUIRE( arma::norm(C2_cpu - C_ref) / arma::norm(C2_cpu) < tol );
    REQUIRE( arma::norm(C3_cpu - C_ref) / arma::norm(C3_cpu) < tol );
    REQUIRE( arma::norm(C4_cpu - C_ref) / arma::norm(C4_cpu) < tol );
    }
  }



TEST_CASE("conv2_full_empty_test", "[conv2]")
  {
  fmat a;
  fmat b = randu<fmat>(3, 3);

  fmat c1 = conv2(a, b);
  fmat c2 = conv2(b, a);
  fmat c3 = conv2(a, b, "full");
  fmat c4 = conv2(b, a, "full");

  REQUIRE( c1.n_rows == 0 );
  REQUIRE( c1.n_cols == 0 );
  REQUIRE( c2.n_rows == 0 );
  REQUIRE( c2.n_cols == 0 );
  REQUIRE( c3.n_rows == 0 );
  REQUIRE( c3.n_cols == 0 );
  REQUIRE( c4.n_rows == 0 );
  REQUIRE( c4.n_cols == 0 );

  // Now try with both matrices empty.
  b.set_size(0, 0);

  c1 = conv2(a, b);
  c2 = conv2(b, a);
  c3 = conv2(a, b, "full");
  c4 = conv2(b, a, "full");

  REQUIRE( c1.n_rows == 0 );
  REQUIRE( c1.n_cols == 0 );
  REQUIRE( c2.n_rows == 0 );
  REQUIRE( c2.n_cols == 0 );
  REQUIRE( c3.n_rows == 0 );
  REQUIRE( c3.n_cols == 0 );
  REQUIRE( c4.n_rows == 0 );
  REQUIRE( c4.n_cols == 0 );
  }



TEST_CASE("conv2_full_alias_test", "[conv2]")
  {
  // Has to be large enough that we process the matrix in pieces.
  fmat a = randu<fmat>(2000, 2000);
  fmat b = randu<fmat>(50, 50);

  fmat a_orig(a);

  a = conv2(a, b);
  fmat a_ref = conv2(a_orig, b);

  REQUIRE( a.n_rows == a_ref.n_rows );
  REQUIRE( a.n_cols == a_ref.n_cols );
  REQUIRE( all( all( abs( a - a_ref ) < 1e-5 ) ) );

  a = a_orig;
  a = conv2(b, a);

  REQUIRE( a.n_rows == a_ref.n_rows );
  REQUIRE( a.n_cols == a_ref.n_cols );
  REQUIRE( all( all( abs( a - a_ref ) < 1e-5 ) ) );

  a = a_orig;
  a = conv2(a, b, "full");

  REQUIRE( a.n_rows == a_ref.n_rows );
  REQUIRE( a.n_cols == a_ref.n_cols );
  REQUIRE( all( all( abs( a - a_ref ) < 1e-5 ) ) );

  a = a_orig;
  a = conv2(b, a, "full");

  REQUIRE( a.n_rows == a_ref.n_rows );
  REQUIRE( a.n_cols == a_ref.n_cols );
  REQUIRE( all( all( abs( a - a_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("conv2_full_expr_inputs_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X = randu<Mat<eT>>(255, 500);
  Mat<eT> Y = randu<Mat<eT>>(25, 10);

  Mat<eT> C1 = conv2(X + 3, trans(2 * Y));
  Mat<eT> C2 = conv2(trans(2 * Y), X + 3);
  Mat<eT> C3 = conv2(X + 3, trans(2 * Y), "full");
  Mat<eT> C4 = conv2(trans(2 * Y), X + 3, "full");

  Mat<eT> X_expr_ref(X + 3);
  Mat<eT> Y_expr_ref(trans(2 * Y));

  Mat<eT> C_ref = conv2(X_expr_ref, Y_expr_ref);

  REQUIRE( C1.n_rows == C_ref.n_rows );
  REQUIRE( C1.n_cols == C_ref.n_cols );
  REQUIRE( C2.n_rows == C_ref.n_rows );
  REQUIRE( C2.n_cols == C_ref.n_cols );
  REQUIRE( C3.n_rows == C_ref.n_rows );
  REQUIRE( C3.n_cols == C_ref.n_cols );
  REQUIRE( C4.n_rows == C_ref.n_rows );
  REQUIRE( C4.n_cols == C_ref.n_cols );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C2 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C3 - C_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C4 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("col_vs_row_conv2_full_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A = randu<Mat<eT>>(512, 1);
  Mat<eT> B = randu<Mat<eT>>(1, 128);

  Mat<eT> C1 = conv2(A, B);
  Mat<eT> C2 = conv2(B, A);
  Mat<eT> C3 = conv2(A, B, "full");
  Mat<eT> C4 = conv2(B, A, "full");

  arma::Mat<eT> A_cpu(A);
  arma::Mat<eT> B_cpu(B);

  arma::Mat<eT> C_ref = arma::conv2(A_cpu, B_cpu, "full");

  REQUIRE( C1.n_rows == C_ref.n_rows );
  REQUIRE( C1.n_cols == C_ref.n_cols );
  REQUIRE( C2.n_rows == C_ref.n_rows );
  REQUIRE( C2.n_cols == C_ref.n_cols );
  REQUIRE( C3.n_rows == C_ref.n_rows );
  REQUIRE( C3.n_cols == C_ref.n_cols );
  REQUIRE( C4.n_rows == C_ref.n_rows );
  REQUIRE( C4.n_cols == C_ref.n_cols );

  arma::Mat<eT> C1_cpu(C1);
  arma::Mat<eT> C2_cpu(C2);
  arma::Mat<eT> C3_cpu(C3);
  arma::Mat<eT> C4_cpu(C4);

  REQUIRE( arma::approx_equal( C1_cpu, C_ref, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( C2_cpu, C_ref, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( C3_cpu, C_ref, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( C4_cpu, C_ref, "reldiff", 1e-5 ) );
  }



// now test the "same" size variant

TEMPLATE_TEST_CASE("hardcoded_same_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(4, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);
  A(3, 0) = eT(13);
  A(3, 1) = eT(14);
  A(3, 2) = eT(15);
  A(3, 3) = eT(16);

  Mat<eT> B(3, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);
  B(2, 0) = eT(16);
  B(2, 1) = eT(17);
  B(2, 2) = eT(18);

  Mat<eT> C_ref(4, 4);
  C_ref(0, 0) = eT( 155);
  C_ref(0, 1) = eT( 278);
  C_ref(0, 2) = eT( 353);
  C_ref(0, 3) = eT( 273);
  C_ref(1, 0) = eT( 396);
  C_ref(1, 1) = eT( 678);
  C_ref(1, 2) = eT( 804);
  C_ref(1, 3) = eT( 603);
  C_ref(2, 0) = eT( 720);
  C_ref(2, 1) = eT(1182);
  C_ref(2, 2) = eT(1308);
  C_ref(2, 3) = eT( 951);
  C_ref(3, 0) = eT( 677);
  C_ref(3, 1) = eT(1094);
  C_ref(3, 2) = eT(1187);
  C_ref(3, 3) = eT( 851);

  Mat<eT> C1 = conv2(A, B, "same");

  REQUIRE( C1.n_rows == 4 );
  REQUIRE( C1.n_cols == 4 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_same_k_rows_gt_cols_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(4, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);
  A(3, 0) = eT(13);
  A(3, 1) = eT(14);
  A(3, 2) = eT(15);
  A(3, 3) = eT(16);

  Mat<eT> B(3, 2);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(1, 0) = eT(12);
  B(1, 1) = eT(13);
  B(2, 0) = eT(14);
  B(2, 1) = eT(15);

  Mat<eT> C_ref(4, 4);
  C_ref(0, 0) = eT( 152);
  C_ref(0, 1) = eT( 198);
  C_ref(0, 2) = eT( 244);
  C_ref(0, 3) = eT( 140);
  C_ref(1, 0) = eT( 379);
  C_ref(1, 1) = eT( 454);
  C_ref(1, 2) = eT( 529);
  C_ref(1, 3) = eT( 296);
  C_ref(2, 0) = eT( 679);
  C_ref(2, 1) = eT( 754);
  C_ref(2, 2) = eT( 829);
  C_ref(2, 3) = eT( 452);
  C_ref(3, 0) = eT( 612);
  C_ref(3, 1) = eT( 666);
  C_ref(3, 2) = eT( 720);
  C_ref(3, 3) = eT( 388);

  Mat<eT> C1 = conv2(A, B, "same");

  REQUIRE( C1.n_rows == 4 );
  REQUIRE( C1.n_cols == 4 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_same_k_rows_lt_cols_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(4, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);
  A(3, 0) = eT(13);
  A(3, 1) = eT(14);
  A(3, 2) = eT(15);
  A(3, 3) = eT(16);

  Mat<eT> B(2, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);

  Mat<eT> C_ref(4, 4);
  C_ref(0, 0) = eT( 155);
  C_ref(0, 1) = eT( 278);
  C_ref(0, 2) = eT( 353);
  C_ref(0, 3) = eT( 273);
  C_ref(1, 0) = eT( 347);
  C_ref(1, 1) = eT( 578);
  C_ref(1, 2) = eT( 653);
  C_ref(1, 3) = eT( 481);
  C_ref(2, 0) = eT( 539);
  C_ref(2, 1) = eT( 878);
  C_ref(2, 2) = eT( 953);
  C_ref(2, 3) = eT( 689);
  C_ref(3, 0) = eT( 364);
  C_ref(3, 1) = eT( 586);
  C_ref(3, 2) = eT( 628);
  C_ref(3, 3) = eT( 449);

  Mat<eT> C1 = conv2(A, B, "same");

  REQUIRE( C1.n_rows == 4 );
  REQUIRE( C1.n_cols == 4 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_same_A_rows_gt_cols_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(4, 3);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(1, 0) = eT( 4);
  A(1, 1) = eT( 5);
  A(1, 2) = eT( 6);
  A(2, 0) = eT( 7);
  A(2, 1) = eT( 8);
  A(2, 2) = eT( 9);
  A(3, 0) = eT(10);
  A(3, 1) = eT(11);
  A(3, 2) = eT(12);

  Mat<eT> B(3, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);
  B(2, 0) = eT(16);
  B(2, 1) = eT(17);
  B(2, 2) = eT(18);

  Mat<eT> C_ref(4, 3);
  C_ref(0, 0) = eT( 134);
  C_ref(0, 1) = eT( 245);
  C_ref(0, 2) = eT( 198);
  C_ref(1, 0) = eT( 327);
  C_ref(1, 1) = eT( 570);
  C_ref(1, 2) = eT( 441);
  C_ref(2, 0) = eT( 570);
  C_ref(2, 1) = eT( 948);
  C_ref(2, 2) = eT( 702);
  C_ref(3, 0) = eT( 530);
  C_ref(3, 1) = eT( 866);
  C_ref(3, 2) = eT( 630);

  Mat<eT> C1 = conv2(A, B, "same");

  REQUIRE( C1.n_rows == 4 );
  REQUIRE( C1.n_cols == 3 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_same_A_rows_lt_cols_conv2_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(3, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);

  Mat<eT> B(3, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);
  B(2, 0) = eT(16);
  B(2, 1) = eT(17);
  B(2, 2) = eT(18);

  Mat<eT> C_ref(3, 4);
  C_ref(0, 0) = eT( 155);
  C_ref(0, 1) = eT( 278);
  C_ref(0, 2) = eT( 353);
  C_ref(0, 3) = eT( 273);
  C_ref(1, 0) = eT( 396);
  C_ref(1, 1) = eT( 678);
  C_ref(1, 2) = eT( 804);
  C_ref(1, 3) = eT( 603);
  C_ref(2, 0) = eT( 437);
  C_ref(2, 1) = eT( 722);
  C_ref(2, 2) = eT( 815);
  C_ref(2, 3) = eT( 595);

  Mat<eT> C1 = conv2(A, B, "same");

  REQUIRE( C1.n_rows == 3 );
  REQUIRE( C1.n_cols == 4 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("hardcoded_same_all_dims_unequal_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A(5, 4);
  A(0, 0) = eT( 1);
  A(0, 1) = eT( 2);
  A(0, 2) = eT( 3);
  A(0, 3) = eT( 4);
  A(1, 0) = eT( 5);
  A(1, 1) = eT( 6);
  A(1, 2) = eT( 7);
  A(1, 3) = eT( 8);
  A(2, 0) = eT( 9);
  A(2, 1) = eT(10);
  A(2, 2) = eT(11);
  A(2, 3) = eT(12);
  A(3, 0) = eT(13);
  A(3, 1) = eT(14);
  A(3, 2) = eT(15);
  A(3, 3) = eT(16);
  A(4, 0) = eT(17);
  A(4, 1) = eT(18);
  A(4, 2) = eT(19);
  A(4, 3) = eT(20);

  Mat<eT> B(2, 3);
  B(0, 0) = eT(10);
  B(0, 1) = eT(11);
  B(0, 2) = eT(12);
  B(1, 0) = eT(13);
  B(1, 1) = eT(14);
  B(1, 2) = eT(15);

  Mat<eT> C_ref(5, 4);
  C_ref(0, 0) = eT( 155);
  C_ref(0, 1) = eT( 278);
  C_ref(0, 2) = eT( 353);
  C_ref(0, 3) = eT( 273);
  C_ref(1, 0) = eT( 347);
  C_ref(1, 1) = eT( 578);
  C_ref(1, 2) = eT( 653);
  C_ref(1, 3) = eT( 481);
  C_ref(2, 0) = eT( 539);
  C_ref(2, 1) = eT( 878);
  C_ref(2, 2) = eT( 953);
  C_ref(2, 3) = eT( 689);
  C_ref(3, 0) = eT( 731);
  C_ref(3, 1) = eT(1178);
  C_ref(3, 2) = eT(1253);
  C_ref(3, 3) = eT( 897);
  C_ref(4, 0) = eT( 472);
  C_ref(4, 1) = eT( 754);
  C_ref(4, 2) = eT( 796);
  C_ref(4, 3) = eT( 565);

  Mat<eT> C1 = conv2(A, B, "same");

  REQUIRE( C1.n_rows == 5 );
  REQUIRE( C1.n_cols == 4 );

  REQUIRE( all( all( abs( C1 - C_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("conv2_same_square_arma_comparison_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword t = 0; t < 5; ++t)
    {
    const uword A_dim = std::pow((uword) 2, t + 4);
    const uword B_dim = t + 3;

    Mat<eT> A = randu<Mat<eT>>(A_dim, A_dim) - 0.5;
    Mat<eT> B = randu<Mat<eT>>(B_dim, B_dim) * 2.0;

    Mat<eT> C1 = conv2(A, B, "same");
    Mat<eT> C2 = conv2(B, A, "same");

    arma::Mat<eT> A_cpu(A);
    arma::Mat<eT> B_cpu(B);
    arma::Mat<eT> C1_ref = arma::conv2(A_cpu, B_cpu, "same");
    arma::Mat<eT> C2_ref = arma::conv2(B_cpu, A_cpu, "same");

    REQUIRE( C1.n_rows == C1_ref.n_rows );
    REQUIRE( C1.n_cols == C1_ref.n_cols );
    REQUIRE( C2.n_rows == C2_ref.n_rows );
    REQUIRE( C2.n_cols == C2_ref.n_cols );

    arma::Mat<eT> C1_cpu(C1);
    arma::Mat<eT> C2_cpu(C2);

    const eT tol = (is_same_type<eT, float>::value) ? 1e-5 : 1e-8;
    REQUIRE( arma::approx_equal( C1_cpu, C1_ref, "absdiff", tol ) );
    REQUIRE( arma::approx_equal( C2_cpu, C2_ref, "absdiff", tol ) );
    }
  }



TEMPLATE_TEST_CASE("1x1_kernel_conv2_same_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X = randu<Mat<eT>>(763, 257);
  Mat<eT> K(1, 1);
  K(0, 0) = (eT) 1.0;

  // The output should be the same as the input.
  Mat<eT> Y1 = conv2(X, K, "same");

  REQUIRE( Y1.n_rows == X.n_rows );
  REQUIRE( Y1.n_cols == X.n_cols );

  REQUIRE( all( all( abs( Y1 - X ) < 1e-5 ) ) );

  // The output should only have a single element.
  Mat<eT> Y2 = conv2(K, X, "same");

  REQUIRE( Y2.n_rows == 1 );
  REQUIRE( Y2.n_cols == 1 );

  REQUIRE( eT(Y2(0, 0)) == Approx(eT(X(381, 128))) );
  }



TEMPLATE_TEST_CASE("conv2_same_random_sizes_arma_comparison_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword t = 0; t < 5; ++t)
    {
    const Mat<uword> sizes = randi<Mat<uword>>(4, distr_param(20, 200));

    const uword A_rows = sizes[0];
    const uword A_cols = sizes[1];
    const uword B_rows = sizes[2];
    const uword B_cols = sizes[3];

    Mat<eT> A = randu<Mat<eT>>(A_rows, A_cols);
    Mat<eT> B = randi<Mat<eT>>(B_rows, B_cols, distr_param(-50, 50));

    Mat<eT> C1 = conv2(A, B, "same");
    Mat<eT> C2 = conv2(B, A, "same");

    arma::Mat<eT> A_cpu(A);
    arma::Mat<eT> B_cpu(B);
    arma::Mat<eT> C1_ref = arma::conv2(A_cpu, B_cpu, "same");
    arma::Mat<eT> C2_ref = arma::conv2(B_cpu, A_cpu, "same");

    REQUIRE( C1.n_rows == C1_ref.n_rows );
    REQUIRE( C1.n_cols == C1_ref.n_cols );
    REQUIRE( C2.n_rows == C2_ref.n_rows );
    REQUIRE( C2.n_cols == C2_ref.n_cols );

    arma::Mat<eT> C1_cpu(C1);
    arma::Mat<eT> C2_cpu(C2);

    const eT tol = (is_same_type<eT, float>::value) ? 1e-6 : 1e-12;
    // Since the matrices can get big, we'll use a slightly relaxed check that accounts for large norms.
    REQUIRE( arma::norm(C1_cpu - C1_ref) / arma::norm(C1_cpu) < tol );
    REQUIRE( arma::norm(C2_cpu - C2_ref) / arma::norm(C2_cpu) < tol );
    }
  }



TEST_CASE("conv2_same_empty_test", "[conv2]")
  {
  fmat a;
  fmat b = randu<fmat>(3, 3);

  fmat c1 = conv2(a, b, "same");
  fmat c2 = conv2(b, a, "same");

  REQUIRE( c1.n_rows == 0 );
  REQUIRE( c1.n_cols == 0 );
  REQUIRE( c2.n_rows == 0 );
  REQUIRE( c2.n_cols == 0 );

  // Now try with both matrices empty.
  b.set_size(0, 0);

  c1 = conv2(a, b, "same");
  c2 = conv2(b, a, "same");

  REQUIRE( c1.n_rows == 0 );
  REQUIRE( c1.n_cols == 0 );
  REQUIRE( c2.n_rows == 0 );
  REQUIRE( c2.n_cols == 0 );
  }



TEST_CASE("conv2_same_alias_test", "[conv2]")
  {
  // Has to be large enough that we process the matrix in pieces.
  fmat a = randu<fmat>(2000, 2000);
  fmat b = randu<fmat>(50, 50);

  fmat a_orig(a);

  a = conv2(a, b, "same");
  fmat a_ref = conv2(a_orig, b, "same");

  REQUIRE( a.n_rows == a_ref.n_rows );
  REQUIRE( a.n_cols == a_ref.n_cols );
  REQUIRE( all( all( abs( a - a_ref ) < 1e-5 ) ) );

  a = a_orig;
  a = conv2(b, a, "same");
  a_ref = conv2(b, a_orig, "same");

  REQUIRE( a.n_rows == a_ref.n_rows );
  REQUIRE( a.n_cols == a_ref.n_cols );
  REQUIRE( all( all( abs( a - a_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("conv2_same_expr_inputs_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> X = randu<Mat<eT>>(255, 500);
  Col<eT> Y = randu<Col<eT>>(25);

  Mat<eT> C1 = conv2(X + 3, trans(2 * diagmat(Y)), "same");
  Mat<eT> C2 = conv2(trans(2 * diagmat(Y)), X + 3, "same");

  Mat<eT> X_expr_ref(X + 3);
  Mat<eT> Y_expr_ref(trans(2 * diagmat(Y)));

  Mat<eT> C1_ref = conv2(X_expr_ref, Y_expr_ref, "same");
  Mat<eT> C2_ref = conv2(Y_expr_ref, X_expr_ref, "same");

  REQUIRE( C1.n_rows == C1_ref.n_rows );
  REQUIRE( C1.n_cols == C1_ref.n_cols );
  REQUIRE( C2.n_rows == C2_ref.n_rows );
  REQUIRE( C2.n_cols == C2_ref.n_cols );

  REQUIRE( all( all( abs( C1 - C1_ref ) < 1e-5 ) ) );
  REQUIRE( all( all( abs( C2 - C2_ref ) < 1e-5 ) ) );
  }



TEMPLATE_TEST_CASE("col_vs_row_conv2_same_test", "[conv2]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> A = randu<Mat<eT>>(512, 1);
  Mat<eT> B = randu<Mat<eT>>(1, 128);

  Mat<eT> C1 = conv2(A, B, "same");
  Mat<eT> C2 = conv2(B, A, "same");

  arma::Mat<eT> A_cpu(A);
  arma::Mat<eT> B_cpu(B);

  arma::Mat<eT> C1_ref = arma::conv2(A_cpu, B_cpu, "same");
  arma::Mat<eT> C2_ref = arma::conv2(B_cpu, A_cpu, "same");

  REQUIRE( C1.n_rows == C1_ref.n_rows );
  REQUIRE( C1.n_cols == C1_ref.n_cols );
  REQUIRE( C2.n_rows == C2_ref.n_rows );
  REQUIRE( C2.n_cols == C2_ref.n_cols );

  arma::Mat<eT> C1_cpu(C1);
  arma::Mat<eT> C2_cpu(C2);

  REQUIRE( arma::approx_equal( C1_cpu, C1_ref, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( C2_cpu, C2_ref, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "conv2_conv_to_test",
  "[conv2]",
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

  Mat<eT1> A = randu<Mat<eT1>>(200, 200);
  Mat<eT1> B = randu<Mat<eT1>>(50, 50);

  Mat<eT2> C1 = conv_to<Mat<eT2>>::from(conv2(A, B));
  Mat<eT2> C2 = conv_to<Mat<eT2>>::from(conv2(B, A));
  Mat<eT2> C3 = conv_to<Mat<eT2>>::from(conv2(A, B, "full"));
  Mat<eT2> C4 = conv_to<Mat<eT2>>::from(conv2(B, A, "full"));
  Mat<eT2> C5 = conv_to<Mat<eT2>>::from(conv2(A, B, "same"));
  Mat<eT2> C6 = conv_to<Mat<eT2>>::from(conv2(B, A, "same"));

  Mat<eT1> C1_pre_conv = conv2(A, B);
  Mat<eT1> C2_pre_conv = conv2(B, A);
  Mat<eT1> C3_pre_conv = conv2(A, B, "full");
  Mat<eT1> C4_pre_conv = conv2(B, A, "full");
  Mat<eT1> C5_pre_conv = conv2(A, B, "same");
  Mat<eT1> C6_pre_conv = conv2(B, A, "same");

  Mat<eT2> C1_ref = conv_to<Mat<eT2>>::from(C1_pre_conv);
  Mat<eT2> C2_ref = conv_to<Mat<eT2>>::from(C2_pre_conv);
  Mat<eT2> C3_ref = conv_to<Mat<eT2>>::from(C3_pre_conv);
  Mat<eT2> C4_ref = conv_to<Mat<eT2>>::from(C4_pre_conv);
  Mat<eT2> C5_ref = conv_to<Mat<eT2>>::from(C5_pre_conv);
  Mat<eT2> C6_ref = conv_to<Mat<eT2>>::from(C6_pre_conv);

  REQUIRE( C1.n_rows == C1_ref.n_rows );
  REQUIRE( C1.n_cols == C1_ref.n_cols );
  REQUIRE( C2.n_rows == C2_ref.n_rows );
  REQUIRE( C2.n_cols == C2_ref.n_cols );
  REQUIRE( C3.n_rows == C3_ref.n_rows );
  REQUIRE( C3.n_cols == C3_ref.n_cols );
  REQUIRE( C4.n_rows == C4_ref.n_rows );
  REQUIRE( C4.n_cols == C4_ref.n_cols );
  REQUIRE( C5.n_rows == C5_ref.n_rows );
  REQUIRE( C5.n_cols == C5_ref.n_cols );
  REQUIRE( C6.n_rows == C6_ref.n_rows );
  REQUIRE( C6.n_cols == C6_ref.n_cols );

  REQUIRE( all(all(abs(C1 - C1_ref) < 1e-5)) );
  REQUIRE( all(all(abs(C2 - C2_ref) < 1e-5)) );
  REQUIRE( all(all(abs(C3 - C3_ref) < 1e-5)) );
  REQUIRE( all(all(abs(C4 - C4_ref) < 1e-5)) );
  REQUIRE( all(all(abs(C5 - C5_ref) < 1e-5)) );
  REQUIRE( all(all(abs(C6 - C6_ref) < 1e-5)) );
  }
