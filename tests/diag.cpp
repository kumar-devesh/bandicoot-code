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

#include <armadillo>
#include <bandicoot>
#include "catch.hpp"

using namespace coot;

// Test main diagonal operations.

TEMPLATE_TEST_CASE("main_diag_set", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

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

TEMPLATE_TEST_CASE("main_diag_extract", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu(x);

  Col<eT> y = x.diag();
  arma::Col<eT> y_cpu = x_cpu.diag();

  arma::Col<eT> y2_cpu(y);

  REQUIRE( arma::approx_equal(y_cpu, y2_cpu, "absdiff", 1e-5) );
  }



// Test off-diagonal extraction.

TEMPLATE_TEST_CASE("off_diag_extract", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu(x);

  for (sword k = -19; k < 20; ++k)
    {
    Col<eT> y = x.diag(k);
    arma::Col<eT> y_cpu = x_cpu.diag(k);

    arma::Col<eT> y2_cpu(y);

    REQUIRE( arma::approx_equal(y_cpu, y2_cpu, "absdiff", 1e-5) );
    }
  }



// Test subview main diagonal extraction.

TEMPLATE_TEST_CASE("subview_main_diag_extract", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu(x);

  Col<eT> y = x.submat(2, 2, 11, 11).diag();
  arma::Col<eT> y_cpu = x_cpu.submat(2, 2, 11, 11).diag();

  arma::Col<eT> y2_cpu(y);

  REQUIRE( arma::approx_equal(y_cpu, y2_cpu, "absdiff", 1e-5) );
  }



// Test subview off-diagonal extraction.

TEMPLATE_TEST_CASE("subview_off_diag_extract", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu(x);

  for (sword k = -9; k < 10; ++k)
    {
    Col<eT> y = x.submat(2, 2, 11, 11).diag(k);
    arma::Col<eT> y_cpu = x_cpu.submat(2, 2, 11, 11).diag(k);

    arma::Col<eT> y2_cpu(y);

    REQUIRE( arma::approx_equal(y_cpu, y2_cpu, "absdiff", 1e-5) );
    }
  }



// Test non-square diagonal size.

TEMPLATE_TEST_CASE("non_square_diag", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 10, distr_param(0, 100));

  REQUIRE( x.diag().n_elem == 10 );
  REQUIRE( x.diag(-5).n_elem == 10 );
  REQUIRE( x.diag(5).n_elem == 5 );
  }



// Test non-square subview diagonal size.

TEMPLATE_TEST_CASE("non_square_subview_diag", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));

  REQUIRE( x.submat(2, 2, 7, 11).diag().n_elem == 6 );
  REQUIRE( x.submat(2, 2, 7, 11).diag(4).n_elem == 6 );
  REQUIRE( x.submat(2, 2, 7, 11).diag(-4).n_elem == 2 );
  }



// Test extraction of diagonal into subview.

TEMPLATE_TEST_CASE("extract_diag_into_subview", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 10, distr_param(0, 100));
  Mat<eT> y = randi<Mat<eT>>(20, 20, distr_param(0, 100));

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y_cpu(y);

  y.submat(0, 0, 9, 0) = x.diag();

  Col<eT> x_diag = x.diag();
  arma::Col<eT> x_diag_cpu(x_diag);

  Col<eT> y_col = y.submat(0, 0, 9, 0);
  arma::Col<eT> y_col_cpu(y_col);

  REQUIRE( arma::approx_equal(y_col_cpu, x_diag_cpu, "absdiff", 1e-5) );
  }



// Test element access in diagonals.

TEMPLATE_TEST_CASE("element_access_diag", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu(x);

  for (sword k = -19; k < 20; ++k)
    {
    REQUIRE( x.diag(k).n_elem == x_cpu.diag(k).n_elem );
    for (uword i = 0; i < x.diag(k).n_elem; ++i)
      {
      REQUIRE ( eT(x.diag(k)[i])       == Approx(eT(x_cpu.diag(k)[i])) );
      REQUIRE ( eT(x.diag(k)(i))       == Approx(eT(x_cpu.diag(k)(i))) );
      REQUIRE ( eT(x.diag(k).at(i))    == Approx(eT(x_cpu.diag(k).at(i))) );
      REQUIRE ( eT(x.diag(k)(i, 0))    == Approx(eT(x_cpu.diag(k)(i, 0))) );
      REQUIRE ( eT(x.diag(k).at(i, 0)) == Approx(eT(x_cpu.diag(k).at(i, 0))) );
      }
    }
  }



// Test element access in a subview's diagonal.

TEMPLATE_TEST_CASE("element_access_subview_diag", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 100));
  arma::Mat<eT> x_cpu(x);

  for (sword k = -9; k < 10; ++k)
    {
    REQUIRE( x.submat(2, 2, 11, 11).diag(k).n_elem == x_cpu.submat(2, 2, 11, 11).diag(k).n_elem );
    for (uword i = 0; i < x.submat(2, 2, 11, 11).diag(k).n_elem; ++i)
      {
      REQUIRE ( eT(x.submat(2, 2, 11, 11).diag(k)[i])       == Approx(eT(x_cpu.submat(2, 2, 11, 11).diag(k)[i])) );
      REQUIRE ( eT(x.submat(2, 2, 11, 11).diag(k)(i))       == Approx(eT(x_cpu.submat(2, 2, 11, 11).diag(k)(i))) );
      REQUIRE ( eT(x.submat(2, 2, 11, 11).diag(k).at(i))    == Approx(eT(x_cpu.submat(2, 2, 11, 11).diag(k).at(i))) );
      REQUIRE ( eT(x.submat(2, 2, 11, 11).diag(k)(i, 0))    == Approx(eT(x_cpu.submat(2, 2, 11, 11).diag(k)(i, 0))) );
      REQUIRE ( eT(x.submat(2, 2, 11, 11).diag(k).at(i, 0)) == Approx(eT(x_cpu.submat(2, 2, 11, 11).diag(k).at(i, 0))) );
      }
    }
  }



// Test diagview::clamp().

TEMPLATE_TEST_CASE("diagview_clamp", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(500, 500, distr_param(0, 10));
  x.diag().clamp(3, 5);

  for (uword i = 0; i < 500; ++i)
    {
    REQUIRE( eT(x.diag()[i]) >= eT(3) );
    REQUIRE( eT(x.diag()[i]) <= eT(5) );
    }
  }



// Test setting diagonals.

TEMPLATE_TEST_CASE("diagview_set", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(500, 500, distr_param(0, 100));
  arma::Mat<eT> x_cpu(x);

  for (sword k = -19; k < 20; ++k)
    {
    Mat<eT> x2(x);
    arma::Mat<eT> x2_cpu(x_cpu);

    Col<eT> y = randi<Col<eT>>(x2.diag(k).n_elem, distr_param(200, 300));
    arma::Col<eT> y_cpu(y);

    x2.diag(k) = y;
    x2_cpu.diag(k) = y_cpu;

    arma::Mat<eT> x3_cpu(x2);

    REQUIRE( arma::approx_equal( x3_cpu, x2_cpu, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("diagview_set_alias", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(500, 500, distr_param(0, 100));
  arma::Mat<eT> x_cpu(x);

  for (sword k = -19; k < 20; ++k)
    {
    Mat<eT> x2(x);
    arma::Mat<eT> x2_cpu(x_cpu);

    x2.diag(k) = x2.submat(0, 0, x2.diag(k).n_elem - 1, 0);
    x2_cpu.diag(k) = x2_cpu.submat(0, 0, x2_cpu.diag(k).n_elem - 1, 0);

    arma::Mat<eT> x3_cpu(x2);

    REQUIRE( arma::approx_equal( x3_cpu, x2_cpu, "absdiff", 1e-5 ) );
    }
  }



TEMPLATE_TEST_CASE("diagview_copy", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(0, 100));
  Mat<eT> y = randi<Mat<eT>>(50, 50, distr_param(100, 200));

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y_cpu(y);

  for (sword k = -49; k < 50; ++k)
    {
    x.diag(k) = y.diag(k);
    x_cpu.diag(k) = y_cpu.diag(k);

    arma::Mat<eT> x2_cpu(x);

    REQUIRE( arma::approx_equal( x_cpu, x2_cpu, "absdiff", 1e-5 ) );
    }
  }



// Test in-place vector operations on diagonals.

TEMPLATE_TEST_CASE("diag_inplace_plus_vector", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 50));
  arma::Mat<eT> x_cpu(x);

  Col<eT> y = randi<Mat<eT>>(x.diag().n_elem, distr_param(51, 100));
  arma::Col<eT> y_cpu(y);

  x.diag() += y;
  x_cpu.diag() += y_cpu;

  arma::Mat<eT> x2_cpu(x);

  REQUIRE( arma::approx_equal( x2_cpu, x_cpu, "absdiff", 1e-5) );
  }




TEMPLATE_TEST_CASE("diag_inplace_minus_vector", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(200, 300));
  arma::Mat<eT> x_cpu(x);

  Col<eT> y = randi<Mat<eT>>(x.diag().n_elem, distr_param(51, 100));
  arma::Col<eT> y_cpu(y);

  x.diag() -= y;
  x_cpu.diag() -= y_cpu;

  arma::Mat<eT> x2_cpu(x);

  REQUIRE( arma::approx_equal( x2_cpu, x_cpu, "absdiff", 1e-5) );
  }




TEMPLATE_TEST_CASE("diag_inplace_schur_vector", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(0, 50));
  arma::Mat<eT> x_cpu(x);

  Col<eT> y = randi<Mat<eT>>(x.diag().n_elem, distr_param(51, 100));
  arma::Col<eT> y_cpu(y);

  x.diag() %= y;
  x_cpu.diag() %= y_cpu;

  arma::Mat<eT> x2_cpu(x);

  REQUIRE( arma::approx_equal( x2_cpu, x_cpu, "absdiff", 1e-5) );
  }




TEMPLATE_TEST_CASE("diag_inplace_div_vector", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 20, distr_param(500, 600));
  arma::Mat<eT> x_cpu(x);

  Col<eT> y = randi<Mat<eT>>(x.diag().n_elem, distr_param(5, 10));
  arma::Col<eT> y_cpu(y);

  x.diag() /= y;
  x_cpu.diag() /= y_cpu;

  arma::Mat<eT> x2_cpu(x);

  REQUIRE( arma::approx_equal( x2_cpu, x_cpu, "absdiff", 1e-5) );
  }



// Test random number generation.

TEMPLATE_TEST_CASE("diag_randu", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(200, 200, distr_param(500, 600));
  x.diag().randu();

  arma::Mat<eT> x_cpu(x);

  for (uword c = 0; c < 200; ++c)
    {
    for (uword r = 0; r < 200; ++r)
      {
      if ( c == r )
        {
        REQUIRE( x_cpu(r, c) >= eT(0) );
        REQUIRE( x_cpu(r, c) <= eT(1) );
        }
      else
        {
        REQUIRE( x_cpu(r, c) >= eT(500) );
        REQUIRE( x_cpu(r, c) <= eT(600) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("diag_randn", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(200, 200, distr_param(500, 600));
  x.diag().randn();

  arma::Mat<eT> x_cpu(x);

  for (uword c = 0; c < 200; ++c)
    {
    for (uword r = 0; r < 200; ++r)
      {
      if ( c == r )
        {
        // Don't want to make too many assumptions about what the randn results
        // will be... but given a unit-variance zero-mean Gaussian, it should be
        // less than 25...
        REQUIRE( std::abs(x_cpu(r, c)) <= eT(25) );
        }
      else
        {
        REQUIRE( x_cpu(r, c) >= eT(500) );
        REQUIRE( x_cpu(r, c) <= eT(600) );
        }
      }
    }
  }



// Test diagonal in-place operations on matrices.

TEMPLATE_TEST_CASE("mat_inplace_diag_plus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(0, 100));
  Col<eT> y = randi<Col<eT>>(30, distr_param(200, 300));

  arma::Mat<eT> x_cpu(x);
  arma::Col<eT> y_cpu(y);

  y += x.diag();
  y_cpu += x_cpu.diag();

  arma::Mat<eT> y2_cpu(y);

  REQUIRE( arma::approx_equal( y2_cpu, y_cpu, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("mat_inplace_diag_minus", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(0, 100));
  Col<eT> y = randi<Col<eT>>(30, distr_param(200, 300));

  arma::Mat<eT> x_cpu(x);
  arma::Col<eT> y_cpu(y);

  y -= x.diag();
  y_cpu -= x_cpu.diag();

  arma::Mat<eT> y2_cpu(y);

  REQUIRE( arma::approx_equal( y2_cpu, y_cpu, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("mat_inplace_diag_schur", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(0, 100));
  Col<eT> y = randi<Col<eT>>(30, distr_param(200, 300));

  arma::Mat<eT> x_cpu(x);
  arma::Col<eT> y_cpu(y);

  y %= x.diag();
  y_cpu %= x_cpu.diag();

  arma::Mat<eT> y2_cpu(y);

  REQUIRE( arma::approx_equal( y2_cpu, y_cpu, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("mat_inplace_diag_div", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(5, 10));
  Col<eT> y = randi<Col<eT>>(30, distr_param(100, 200));

  arma::Mat<eT> x_cpu(x);
  arma::Col<eT> y_cpu(y);

  y /= x.diag();
  y_cpu /= x_cpu.diag();

  arma::Mat<eT> y2_cpu(y);

  REQUIRE( arma::approx_equal( y2_cpu, y_cpu, "absdiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("mat_inplace_diag_mul", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 30, distr_param(0, 10));
  Mat<eT> y = randi<Mat<eT>>(30, 30, distr_param(0, 10));

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y_cpu(y);

  x *= y.diag();
  x_cpu *= y_cpu.diag();

  arma::Mat<eT> x2_cpu(x);

  REQUIRE( arma::approx_equal( x2_cpu, x_cpu, "absdiff", 1e-5 ) );
  }



// Test bare diagmat.

TEMPLATE_TEST_CASE("diagmat", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> y = randi<Col<eT>>(50, distr_param(10, 20));
  Mat<eT> x = diagmat(y);

  REQUIRE( x.n_rows == 50 );
  REQUIRE( x.n_cols == 50 );

  arma::Col<eT> y_cpu(y);
  arma::Mat<eT> x_cpu(x);

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      if ( r == c )
        {
        REQUIRE( x_cpu(r, c) == Approx(y_cpu(r)) );
        }
      else
        {
        REQUIRE( x_cpu(r, c) == Approx(eT(0)).margin(1e-5) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("diagmat2", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> y = randi<Col<eT>>(50, distr_param(10, 20));

  for (sword k = -4; k < 5; ++k)
    {
    Mat<eT> x = diagmat(y, k);

    REQUIRE( x.n_rows == 50 + uword(std::abs(k)) );
    REQUIRE( x.n_cols == 50 + uword(std::abs(k)) );

    arma::Col<eT> y_cpu(y);
    arma::Mat<eT> x_cpu(x);

    for (uword c = 0; c < 50; ++c)
      {
      for (uword r = 0; r < 50; ++r)
        {
        if ( k < 0 )
          {
          if ( (r >= uword(std::abs(k))) && (r - uword(std::abs(k))) == c )
            {
            REQUIRE( x_cpu(r, c) == Approx(y_cpu(c)) );
            }
          else
            {
            REQUIRE( x_cpu(r, c) == Approx(eT(0)).margin(1e-5) );
            }
          }
        else
          {
          if ( (c >= uword(std::abs(k))) && (c - uword(std::abs(k))) == r )
            {
            REQUIRE( x_cpu(r, c) == Approx(y_cpu(r)) );
            }
          else
            {
            REQUIRE( x_cpu(r, c) == Approx(eT(0)).margin(1e-5) );
            }
          }
        }
      }
    }
  }



// Test diagmat into submatrix.

TEMPLATE_TEST_CASE("diagmat_submatrix", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 10, distr_param(10, 20));
  Col<eT> y = randi<Col<eT>>(8, distr_param(50, 60));

  x.submat(1, 1, 8, 8) = diagmat(y);

  REQUIRE( x.n_rows == 10 );
  REQUIRE( x.n_cols == 10 );

  arma::Mat<eT> x_cpu(x);

  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      if ( r == 0 || r == 9 || c == 0 || c == 9 )
        {
        REQUIRE( eT(x_cpu(r, c)) >= eT(10) );
        REQUIRE( eT(x_cpu(r, c)) <= eT(20) );
        }
      else if ( r == c )
        {
        REQUIRE( eT(x_cpu(r, c)) >= eT(50) );
        REQUIRE( eT(x_cpu(r, c)) <= eT(60) );
        }
      else
        {
        REQUIRE( eT(x_cpu(r, c)) == Approx(eT(0)).margin(1e-5) );
        }
      }
    }
  }



// Test trace(diagmat).

TEMPLATE_TEST_CASE("diagmat_trace", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> y = randi<Col<eT>>(10, distr_param(10, 20));

  REQUIRE( trace(diagmat(y)) == Approx(accu(y)) );
  }



// Test diagmat(A) * B

TEMPLATE_TEST_CASE("diagmat_times_1", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> a_diag = randi<Col<eT>>(10, distr_param(10, 20));
  Mat<eT> b = randi<Mat<eT>>(10, 20, distr_param(10, 20));

  Mat<eT> c1 = diagmat(a_diag) * b;
  Mat<eT> c2 = diagmat(trans(a_diag)) * b;
  Mat<eT> c3 = trans(diagmat(a_diag)) * b;

  Mat<eT> a_ref(10, 10);
  a_ref.zeros();
  a_ref.diag() = a_diag;

  Mat<eT> c_ref = a_ref * b;

  REQUIRE( c1.n_rows == 10 );
  REQUIRE( c1.n_cols == 20 );
  REQUIRE( c2.n_rows == 10 );
  REQUIRE( c2.n_cols == 20 );
  REQUIRE( c3.n_rows == 10 );
  REQUIRE( c3.n_cols == 20 );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c_ref_cpu(c_ref);

  REQUIRE( arma::approx_equal(c1_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c2_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c3_cpu, c_ref_cpu, "reldiff", 1e-6) );
  }



// Test diagmat(A) * B'

TEMPLATE_TEST_CASE("diagmat_times_2", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> a_diag = randi<Col<eT>>(10, distr_param(10, 20));
  Mat<eT> b = randi<Mat<eT>>(20, 10, distr_param(10, 20));

  Mat<eT> c1 = diagmat(a_diag) * trans(b);
  Mat<eT> c2 = diagmat(trans(a_diag)) * trans(b);
  Mat<eT> c3 = trans(diagmat(a_diag)) * trans(b);

  Mat<eT> a_ref(10, 10);
  a_ref.zeros();
  a_ref.diag() = a_diag;

  Mat<eT> c_ref = a_ref * trans(b);

  REQUIRE( c1.n_rows == 10 );
  REQUIRE( c1.n_cols == 20 );
  REQUIRE( c2.n_rows == 10 );
  REQUIRE( c2.n_cols == 20 );
  REQUIRE( c3.n_rows == 10 );
  REQUIRE( c3.n_cols == 20 );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c_ref_cpu(c_ref);

  REQUIRE( arma::approx_equal(c1_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c2_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c3_cpu, c_ref_cpu, "reldiff", 1e-6) );
  }



// Test diagmat(a * A) * (b * B')

TEMPLATE_TEST_CASE("diagmat_times_3", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> a_diag = randi<Col<eT>>(10, distr_param(10, 20));
  Mat<eT> b = randi<Mat<eT>>(20, 10, distr_param(10, 20));

  Mat<eT> c1 = diagmat(3 * a_diag) * (5 * trans(b));
  Mat<eT> c2 = (3 * diagmat(trans(a_diag))) * (5 * trans(b));
  Mat<eT> c3 = trans(diagmat(3 * a_diag)) * (5 * trans(b));

  Mat<eT> a_ref(10, 10);
  a_ref.zeros();
  a_ref.diag() = a_diag;

  Mat<eT> c_ref = (3 * a_ref) * (5 * trans(b));

  REQUIRE( c1.n_rows == 10 );
  REQUIRE( c1.n_cols == 20 );
  REQUIRE( c2.n_rows == 10 );
  REQUIRE( c2.n_cols == 20 );
  REQUIRE( c3.n_rows == 10 );
  REQUIRE( c3.n_cols == 20 );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c_ref_cpu(c_ref);

  REQUIRE( arma::approx_equal(c1_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c2_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c3_cpu, c_ref_cpu, "reldiff", 1e-6) );
  }



// Test A * diagmat(B)

TEMPLATE_TEST_CASE("diagmat_times_4", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> a = randi<Mat<eT>>(20, 10, distr_param(10, 20));
  Col<eT> b_diag = randi<Col<eT>>(10, distr_param(10, 20));

  Mat<eT> c1 = a * diagmat(b_diag);
  Mat<eT> c2 = a * diagmat(trans(b_diag));
  Mat<eT> c3 = a * trans(diagmat(b_diag));

  Mat<eT> b_ref(10, 10);
  b_ref.zeros();
  b_ref.diag() = b_diag;

  Mat<eT> c_ref = a * b_ref;

  REQUIRE( c1.n_rows == 20 );
  REQUIRE( c1.n_cols == 10 );
  REQUIRE( c2.n_rows == 20 );
  REQUIRE( c2.n_cols == 10 );
  REQUIRE( c3.n_rows == 20 );
  REQUIRE( c3.n_cols == 10 );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c_ref_cpu(c_ref);

  REQUIRE( arma::approx_equal(c1_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c2_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c3_cpu, c_ref_cpu, "reldiff", 1e-6) );
  }



// Test A' * diagmat(B)

TEMPLATE_TEST_CASE("diagmat_times_5", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> a = randi<Mat<eT>>(10, 20, distr_param(10, 20));
  Col<eT> b_diag = randi<Col<eT>>(10, distr_param(10, 20));

  Mat<eT> c1 = trans(a) * diagmat(b_diag);
  Mat<eT> c2 = trans(a) * diagmat(trans(b_diag));
  Mat<eT> c3 = trans(a) * trans(diagmat(b_diag));

  Mat<eT> b_ref(10, 10);
  b_ref.zeros();
  b_ref.diag() = b_diag;

  Mat<eT> c_ref = trans(a) * b_ref;

  REQUIRE( c1.n_rows == 20 );
  REQUIRE( c1.n_cols == 10 );
  REQUIRE( c2.n_rows == 20 );
  REQUIRE( c2.n_cols == 10 );
  REQUIRE( c3.n_rows == 20 );
  REQUIRE( c3.n_cols == 10 );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c_ref_cpu(c_ref);

  REQUIRE( arma::approx_equal(c1_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c2_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c3_cpu, c_ref_cpu, "reldiff", 1e-6) );
  }



// Test (a * A') * (b * diagmat(B))

TEMPLATE_TEST_CASE("diagmat_times_6", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> a = randi<Mat<eT>>(10, 20, distr_param(10, 20));
  Col<eT> b_diag = randi<Col<eT>>(10, distr_param(10, 20));

  Mat<eT> c1 = (2 * trans(a)) * (6 * diagmat(b_diag));
  Mat<eT> c2 = (2 * trans(a)) * (6 * diagmat(trans(b_diag)));
  Mat<eT> c3 = (2 * trans(a)) * (6 * trans(diagmat(b_diag)));

  Mat<eT> b_ref(10, 10);
  b_ref.zeros();
  b_ref.diag() = b_diag;

  Mat<eT> c_ref = (2 * trans(a)) * (6 * b_ref);

  REQUIRE( c1.n_rows == 20 );
  REQUIRE( c1.n_cols == 10 );
  REQUIRE( c2.n_rows == 20 );
  REQUIRE( c2.n_cols == 10 );
  REQUIRE( c3.n_rows == 20 );
  REQUIRE( c3.n_cols == 10 );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c_ref_cpu(c_ref);

  REQUIRE( arma::approx_equal(c1_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c2_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c3_cpu, c_ref_cpu, "reldiff", 1e-6) );
  }



// Test diagmat(A) * diagmat(B)

TEMPLATE_TEST_CASE("diagmat_times_7", "[diag]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> a_diag = randi<Col<eT>>(10, distr_param(10, 20));
  Col<eT> b_diag = randi<Col<eT>>(10, distr_param(20, 30));

  Mat<eT> c1 = diagmat(a_diag) * diagmat(b_diag);
  Mat<eT> c2 = diagmat(trans(a_diag)) * diagmat(trans(b_diag));
  Mat<eT> c3 = trans(diagmat(a_diag)) * trans(diagmat(b_diag));

  Mat<eT> a_ref(10, 10);
  a_ref.zeros();
  a_ref.diag() = a_diag;
  Mat<eT> b_ref(10, 10);
  b_ref.zeros();
  b_ref.diag() = b_diag;

  Mat<eT> c_ref = a_ref * b_ref;

  REQUIRE( c1.n_rows == 10 );
  REQUIRE( c1.n_cols == 10 );
  REQUIRE( c2.n_rows == 10 );
  REQUIRE( c2.n_cols == 10 );
  REQUIRE( c3.n_rows == 10 );
  REQUIRE( c3.n_cols == 10 );

  arma::Mat<eT> c1_cpu(c1);
  arma::Mat<eT> c2_cpu(c2);
  arma::Mat<eT> c3_cpu(c3);
  arma::Mat<eT> c_ref_cpu(c_ref);

  REQUIRE( arma::approx_equal(c1_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c2_cpu, c_ref_cpu, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(c3_cpu, c_ref_cpu, "reldiff", 1e-6) );
  }


// diagmat2(trans())

TEMPLATE_TEST_CASE("diagmat2_trans", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(50, distr_param(10, 20));

  Mat<eT> y1 = diagmat(trans(x), 3);
  Mat<eT> y1_ref = diagmat(x, 3);

  Mat<eT> y2 = diagmat(trans(x), -3);
  Mat<eT> y2_ref = diagmat(x, -3);

  REQUIRE( y1.n_rows == y1_ref.n_rows );
  REQUIRE( y1.n_cols == y1_ref.n_cols );
  REQUIRE( y1.n_elem == y1_ref.n_elem );

  REQUIRE( y2.n_rows == y2_ref.n_rows );
  REQUIRE( y2.n_cols == y2_ref.n_cols );
  REQUIRE( y2.n_elem == y2_ref.n_elem );

  arma::Mat<eT> y1_cpu(y1);
  arma::Mat<eT> y1_cpu_ref(y1_ref);
  arma::Mat<eT> y2_cpu(y2);
  arma::Mat<eT> y2_cpu_ref(y2_ref);

  REQUIRE( arma::approx_equal(y1_cpu, y1_cpu_ref, "reldiff", 1e-6) );
  REQUIRE( arma::approx_equal(y2_cpu, y2_cpu_ref, "reldiff", 1e-6) );
  }



// diagmat(mat)

TEMPLATE_TEST_CASE("diagmat_mat", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));

  Mat<eT> y = diagmat(x);

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );

  Col<eT> x_diag = x.diag();
  Col<eT> y_diag = y.diag();

  REQUIRE( all( x_diag == y_diag ) );
  }



// diagmat2(mat)

TEMPLATE_TEST_CASE("diagmat2_mat", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));

  for (sword k = -49; k < 49; ++k)
    {
    Mat<eT> y = diagmat(x, k);

    REQUIRE( y.n_rows == x.n_rows );
    REQUIRE( y.n_cols == x.n_cols );

    Col<eT> x_diag = x.diag(k);
    Col<eT> y_diag = y.diag(k);

    REQUIRE( all( x_diag == y_diag ) );
    }
  }



// diagmat(trans(mat))

TEMPLATE_TEST_CASE("diagmat_trans_mat", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));

  Mat<eT> y = diagmat(x.t());

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );

  Col<eT> x_diag = x.diag();
  Col<eT> y_diag = y.diag();

  REQUIRE( all( x_diag == y_diag ) );
  }



// diagmat2(trans(mat))

TEMPLATE_TEST_CASE("diagmat2_trans_mat", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));

  for (sword k = -49; k < 49; ++k)
    {
    Mat<eT> y = diagmat(x.t(), k);

    REQUIRE( y.n_rows == x.n_rows );
    REQUIRE( y.n_cols == x.n_cols );

    Col<eT> x_diag = x.diag(-k);
    Col<eT> y_diag = y.diag(k);

    REQUIRE( all( x_diag == y_diag ) );
    }
  }



// diagvec(mat)

TEMPLATE_TEST_CASE("diagvec_mat", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));
  Col<eT> y1 = diagvec(x);
  Col<eT> y2 = x.diag();

  REQUIRE( y1.n_elem == y2.n_elem );

  arma::Col<eT> y1_cpu(y1);
  arma::Col<eT> y2_cpu(y2);

  REQUIRE( arma::approx_equal(y1_cpu, y2_cpu, "reldiff", 1e-6) );
  }



// diagvec(subview)

TEMPLATE_TEST_CASE("diagvec_subview", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));
  Col<eT> y1 = diagvec(x.submat(11, 11, 30, 30));
  Col<eT> y2 = x.submat(11, 11, 30, 30).diag();

  REQUIRE( y1.n_elem == y2.n_elem );

  arma::Col<eT> y1_cpu(y1);
  arma::Col<eT> y2_cpu(y2);

  REQUIRE( arma::approx_equal(y1_cpu, y2_cpu, "reldiff", 1e-6) );
  }



// diagvec(op)

TEMPLATE_TEST_CASE("diagvec_op", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));

  Col<eT> y1 = diagvec(trans(x) + 3);
  Mat<eT> x2 = trans(x) + 3;
  Col<eT> y2 = x2.diag();

  REQUIRE( y1.n_elem == y2.n_elem );

  arma::Col<eT> y1_cpu(y1);
  arma::Col<eT> y2_cpu(y2);

  REQUIRE( arma::approx_equal(y1_cpu, y2_cpu, "reldiff", 1e-6) );
  }



// diagvec(mat alias)

TEMPLATE_TEST_CASE("diagvec_mat_alias", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));
  Mat<eT> x2 = x.diag();
  x = diagvec(x);

  REQUIRE( x.n_elem == x2.n_elem );

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> x2_cpu(x2);

  REQUIRE( arma::approx_equal(x_cpu, x2_cpu, "reldiff", 1e-6) );
  }



// diagvec(mat, k)

TEMPLATE_TEST_CASE("diagvec_k_mat", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));
  for (sword k = -49; k < 50; ++k)
    {
    Col<eT> y1 = diagvec(x, k);
    Col<eT> y2 = x.diag(k);

    REQUIRE( y1.n_elem == y2.n_elem );

    arma::Col<eT> y1_cpu(y1);
    arma::Col<eT> y2_cpu(y2);

    REQUIRE( arma::approx_equal(y1_cpu, y2_cpu, "reldiff", 1e-6) );
    }
  }



// diagvec(subview, k)

TEMPLATE_TEST_CASE("diagvec_k_subview", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));
  for (sword k = -19; k < 20; ++k)
    {
    Col<eT> y1 = diagvec(x.submat(11, 11, 30, 30), k);
    Col<eT> y2 = x.submat(11, 11, 30, 30).diag(k);

    REQUIRE( y1.n_elem == y2.n_elem );

    arma::Col<eT> y1_cpu(y1);
    arma::Col<eT> y2_cpu(y2);

    REQUIRE( arma::approx_equal(y1_cpu, y2_cpu, "reldiff", 1e-6) );
    }
  }



// diagvec(op, k)

TEMPLATE_TEST_CASE("diagvec_k_op", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));
  for (sword k = -49; k < 50; ++k)
    {
    Mat<eT> x2 = trans(x) * 4;

    Col<eT> y1 = diagvec(trans(x) * 4, k);
    Col<eT> y2 = x2.diag(k);

    REQUIRE( y1.n_elem == y2.n_elem );

    arma::Col<eT> y1_cpu(y1);
    arma::Col<eT> y2_cpu(y2);

    REQUIRE( arma::approx_equal(y1_cpu, y2_cpu, "reldiff", 1e-6) );
    }
  }



// diagvec(mat alias, k)

TEMPLATE_TEST_CASE("diagvec_k_mat_alias", "[diag]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (sword k = -49; k < 50; ++k)
    {
    Mat<eT> x = randi<Mat<eT>>(50, 50, distr_param(10, 20));
    Mat<eT> x2 = x.diag(k);
    x = diagvec(x, k);

    REQUIRE( x.n_elem == x2.n_elem );

    arma::Mat<eT> x_cpu(x);
    arma::Mat<eT> x2_cpu(x2);

    REQUIRE( arma::approx_equal(x_cpu, x2_cpu, "reldiff", 1e-6) );
    }
  }



TEMPLATE_TEST_CASE
  (
  "diagvec_conv_to",
  "[diag]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(20, 20, distr_param(1, 20));
  Col<eT2> y1 = conv_to<Col<eT2>>::from(diagvec(x));
  Col<eT2> y2 = conv_to<Col<eT2>>::from(diagvec(x, 3));

  Col<eT1> y1_ref_1 = diagvec(x);
  Col<eT1> y2_ref_1 = diagvec(x, 3);

  Col<eT2> y1_ref = conv_to<Col<eT2>>::from(y1_ref_1);
  Col<eT2> y2_ref = conv_to<Col<eT2>>::from(y2_ref_1);

  REQUIRE( y1.n_elem == y1_ref.n_elem );
  REQUIRE( y2.n_elem == y2_ref.n_elem );

  // Seemingly huge tolerance works for both integers and floats.
  REQUIRE( all( abs( y1 - y1_ref ) < 1 ) == true );
  REQUIRE( all( abs( y2 - y2_ref ) < 1 ) == true );
  }



TEMPLATE_TEST_CASE
  (
  "diagmat_conv_to_1",
  "[diag]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Col<eT1> x = randi<Mat<eT1>>(20, distr_param(1, 20));
  Mat<eT2> y1 = conv_to<Mat<eT2>>::from(diagmat(x));
  Mat<eT2> y2 = conv_to<Mat<eT2>>::from(diagmat(x, 3));

  Mat<eT1> y1_ref_1 = diagmat(x);
  Mat<eT1> y2_ref_1 = diagmat(x, 3);

  Mat<eT2> y1_ref = conv_to<Mat<eT2>>::from(y1_ref_1);
  Mat<eT2> y2_ref = conv_to<Mat<eT2>>::from(y2_ref_1);

  REQUIRE( y1.n_rows == y1_ref.n_rows );
  REQUIRE( y1.n_cols == y1_ref.n_cols );
  REQUIRE( y2.n_rows == y2_ref.n_rows );
  REQUIRE( y2.n_cols == y2_ref.n_cols );

  // Seemingly huge tolerance works for both integers and floats.
  REQUIRE( all( all( abs( y1 - y1_ref ) < 1 ) ) );
  REQUIRE( all( all( abs( y2 - y2_ref ) < 1 ) ) );
  }



TEMPLATE_TEST_CASE
  (
  "diagmat_conv_to_2",
  "[diag]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(20, 20, distr_param(1, 20));
  Mat<eT2> y1 = conv_to<Mat<eT2>>::from(diagmat(x));
  Mat<eT2> y2 = conv_to<Mat<eT2>>::from(diagmat(x, 3));

  Mat<eT1> y1_ref_1 = diagmat(x);
  Mat<eT1> y2_ref_1 = diagmat(x, 3);

  Mat<eT2> y1_ref = conv_to<Mat<eT2>>::from(y1_ref_1);
  Mat<eT2> y2_ref = conv_to<Mat<eT2>>::from(y2_ref_1);

  REQUIRE( y1.n_rows == y1_ref.n_rows );
  REQUIRE( y1.n_cols == y1_ref.n_cols );
  REQUIRE( y2.n_rows == y2_ref.n_rows );
  REQUIRE( y2.n_cols == y2_ref.n_cols );

  // Seemingly huge tolerance works for both integers and floats.
  REQUIRE( all( all( abs( y1 - y1_ref ) < 1 ) ) );
  REQUIRE( all( all( abs( y2 - y2_ref ) < 1 ) ) );
  }
