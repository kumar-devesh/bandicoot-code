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

// Tests for .diag(), diagmat(), diagvec(), and related functions.

#include <bandicoot>
#include "catch.hpp"

using namespace coot;

// Test main diagonal operations.

TEMPLATE_TEST_CASE("main_diag_set", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu_orig(x);

  x.diag().fill(eT(4));
  arma::Mat<eT> x_cpu(x);

  for (uword i = 0; i < 20; ++i)
    {
    REQUIRE( x_cpu(i, i) == Approx(eT(4)) );
    }

  x_cpu_orig.diag().fill(eT(4));
  REQUIRE( arma::approx_equal(x_cpu_orig, x_cpu, "absdiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("main_diag_plus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu_orig(x);

  x.diag() += eT(4);
  arma::Mat<eT> x_cpu(x);

  for (uword i = 0; i < 20; ++i)
    {
    REQUIRE( x_cpu(i, i) == Approx(x_cpu_orig(i, i) + eT(4)) );
    }

  x_cpu_orig.diag() += eT(4);
  REQUIRE( arma::approx_equal(x_cpu_orig, x_cpu, "absdiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("main_diag_minus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  x += eT(5);
  arma::Mat<eT> x_cpu_orig(x);

  x.diag() -= eT(4);
  arma::Mat<eT> x_cpu(x);

  for (uword i = 0; i < 20; ++i)
    {
    REQUIRE( x_cpu(i, i) == Approx(x_cpu_orig(i, i) - eT(4)) );
    }

  x_cpu_orig.diag() -= eT(4);
  REQUIRE( arma::approx_equal(x_cpu_orig, x_cpu, "absdiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("main_diag_mul", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu_orig(x);

  x.diag() *= eT(4);
  arma::Mat<eT> x_cpu(x);

  for (uword i = 0; i < 20; ++i)
    {
    REQUIRE( x_cpu(i, i) == Approx(x_cpu_orig(i, i) * eT(4)) );
    }

  x_cpu_orig.diag() *= eT(4);
  REQUIRE( arma::approx_equal(x_cpu_orig, x_cpu, "absdiff", 1e-5) );
  }



TEMPLATE_TEST_CASE("main_diag_div", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu_orig(x);

  x.diag() /= eT(4);
  arma::Mat<eT> x_cpu(x);

  for (uword i = 0; i < 20; ++i)
    {
    REQUIRE( x_cpu(i, i) == Approx(x_cpu_orig(i, i) / eT(4)) );
    }

  x_cpu_orig.diag() /= eT(4);
  REQUIRE( arma::approx_equal(x_cpu_orig, x_cpu, "absdiff", 1e-5) );
  }



// Test off-diagonal operations.

TEMPLATE_TEST_CASE("off_diag_set", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x);

  for (sword k = -19; k < 20; ++k)
    {
    x = x_orig;
    x.diag(k).fill(eT(4));
    arma::Mat<eT> x_cpu(x);

    REQUIRE( arma::all( x_cpu.diag(k) == eT(4) ) );

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.diag(k).fill(eT(4));

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("off_diag_plus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x_orig);

  for (sword k = -19; k < 20; ++k)
    {
    x = x_orig;
    x.diag(k) += eT(4);
    arma::Mat<eT> x_cpu(x);

    REQUIRE( arma::approx_equal(x_cpu.diag(k), x_cpu_orig.diag(k) + eT(4), "absdiff", 1e-5) );

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.diag(k) += eT(4);

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("off_diag_minus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  x += eT(5);
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x_orig);

  for (sword k = -19; k < 20; ++k)
    {
    x = x_orig;
    x.diag(k) -= eT(4);
    arma::Mat<eT> x_cpu(x);

    REQUIRE( arma::approx_equal(x_cpu.diag(k), x_cpu_orig.diag(k) - eT(4), "absdiff", 1e-5) );

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.diag(k) -= eT(4);

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("off_diag_mul", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x_orig);

  for (sword k = -19; k < 20; ++k)
    {
    x = x_orig;
    x.diag(k) *= eT(4);
    arma::Mat<eT> x_cpu(x);

    REQUIRE( arma::approx_equal(x_cpu.diag(k), x_cpu_orig.diag(k) * eT(4), "absdiff", 1e-5) );

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.diag(k) *= eT(4);

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("off_diag_div", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x_orig);

  for (sword k = -19; k < 20; ++k)
    {
    x = x_orig;
    x.diag(k) /= eT(4);
    arma::Mat<eT> x_cpu(x);

    REQUIRE( arma::approx_equal(x_cpu.diag(k), x_cpu_orig.diag(k) / eT(4), "absdiff", 1e-5) );

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.diag(k) /= eT(4);

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



// Test subview main diagonal operations.

TEMPLATE_TEST_CASE("subview_main_diag_set", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu_orig(x);

  x.submat(2, 2, 11, 11).diag().fill(eT(4));
  arma::Mat<eT> x_cpu(x);
  x_cpu_orig.submat(2, 2, 11, 11).diag().fill(eT(4));

  REQUIRE( arma::approx_equal( x_cpu, x_cpu_orig, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("subview_main_diag_plus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu_orig(x);

  x.submat(2, 2, 11, 11).diag() += eT(4);
  arma::Mat<eT> x_cpu(x);
  x_cpu_orig.submat(2, 2, 11, 11).diag() += eT(4);

  REQUIRE( arma::approx_equal( x_cpu, x_cpu_orig, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("subview_main_diag_minus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  x += eT(5);
  arma::Mat<eT> x_cpu_orig(x);

  x.submat(2, 2, 11, 11).diag() -= eT(4);
  arma::Mat<eT> x_cpu(x);
  x_cpu_orig.submat(2, 2, 11, 11).diag() -= eT(4);

  REQUIRE( arma::approx_equal( x_cpu, x_cpu_orig, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("subview_main_diag_mul", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu_orig(x);

  x.submat(2, 2, 11, 11).diag() *= eT(4);
  arma::Mat<eT> x_cpu(x);
  x_cpu_orig.submat(2, 2, 11, 11).diag() *= eT(4);

  REQUIRE( arma::approx_equal( x_cpu, x_cpu_orig, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("subview_main_diag_div", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu_orig(x);

  x.submat(2, 2, 11, 11).diag() /= eT(4);
  arma::Mat<eT> x_cpu(x);
  x_cpu_orig.submat(2, 2, 11, 11).diag() /= eT(4);

  REQUIRE( arma::approx_equal( x_cpu, x_cpu_orig, "absdiff", 1e-5 ) );
  }



// Test subview off-diagonal operations.

TEMPLATE_TEST_CASE("subview_off_diag_set", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x);

  for (sword k = -9; k < 10; ++k)
    {
    x = x_orig;
    x.submat(2, 2, 11, 11).diag(k).fill(eT(4));
    arma::Mat<eT> x_cpu(x);

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.submat(2, 2, 11, 11).diag(k).fill(eT(4));

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("subview_off_diag_plus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x);

  for (sword k = -9; k < 10; ++k)
    {
    x = x_orig;
    x.submat(2, 2, 11, 11).diag(k) += eT(4);
    arma::Mat<eT> x_cpu(x);

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.submat(2, 2, 11, 11).diag(k) += eT(4);

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("subview_off_diag_minus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  x += eT(5);
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x);

  for (sword k = -9; k < 10; ++k)
    {
    x = x_orig;
    x.submat(2, 2, 11, 11).diag(k) -= eT(4);
    arma::Mat<eT> x_cpu(x);

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.submat(2, 2, 11, 11).diag(k) -= eT(4);

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("subview_off_diag_mul", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x);

  for (sword k = -9; k < 10; ++k)
    {
    x = x_orig;
    x.submat(2, 2, 11, 11).diag(k) *= eT(4);
    arma::Mat<eT> x_cpu(x);

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.submat(2, 2, 11, 11).diag(k) *= eT(4);

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("subview_off_diag_div", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  Mat<eT> x_orig(x);
  arma::Mat<eT> x_cpu_orig(x);

  for (sword k = -9; k < 10; ++k)
    {
    x = x_orig;
    x.submat(2, 2, 11, 11).diag(k) /= eT(4);
    arma::Mat<eT> x_cpu(x);

    arma::Mat<eT> x_cpu_tmp(x_cpu_orig);
    x_cpu_tmp.submat(2, 2, 11, 11).diag(k) /= eT(4);

    REQUIRE( arma::approx_equal( x_cpu, x_cpu_tmp, "absdiff", 1e-5 ) );
    }
  }



// Test main diagonal extraction.

// Test off-diagonal extraction.

// Test subview main diagonal extraction.

// Test subview off-diagonal extraction.

