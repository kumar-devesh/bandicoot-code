// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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

// symmatu()

TEMPLATE_TEST_CASE("symmatu_basic", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(4, 4);
  x(0, 0) = 0;
  x(0, 1) = 1;
  x(0, 2) = 2;
  x(0, 3) = 3;
  x(1, 1) = 4;
  x(1, 2) = 5;
  x(1, 3) = 6;
  x(2, 2) = 7;
  x(2, 3) = 8;
  x(3, 3) = 9;

  Mat<eT> y = symmatu(x);

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );
  REQUIRE( y.n_elem == x.n_elem );

  REQUIRE( eT(y(0, 0)) == 0 );
  REQUIRE( eT(y(1, 0)) == 1 );
  REQUIRE( eT(y(2, 0)) == 2 );
  REQUIRE( eT(y(3, 0)) == 3 );
  REQUIRE( eT(y(0, 1)) == 1 );
  REQUIRE( eT(y(1, 1)) == 4 );
  REQUIRE( eT(y(2, 1)) == 5 );
  REQUIRE( eT(y(3, 1)) == 6 );
  REQUIRE( eT(y(0, 2)) == 2 );
  REQUIRE( eT(y(1, 2)) == 5 );
  REQUIRE( eT(y(2, 2)) == 7 );
  REQUIRE( eT(y(3, 2)) == 8 );
  REQUIRE( eT(y(0, 3)) == 3 );
  REQUIRE( eT(y(1, 3)) == 6 );
  REQUIRE( eT(y(2, 3)) == 8 );
  REQUIRE( eT(y(3, 3)) == 9 );
  }



TEMPLATE_TEST_CASE("symmatu_inplace_basic", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(4, 4);
  x(0, 0) = 0;
  x(0, 1) = 1;
  x(0, 2) = 2;
  x(0, 3) = 3;
  x(1, 1) = 4;
  x(1, 2) = 5;
  x(1, 3) = 6;
  x(2, 2) = 7;
  x(2, 3) = 8;
  x(3, 3) = 9;

  x = symmatu(x);

  REQUIRE( x.n_rows == 4 );
  REQUIRE( x.n_cols == 4 );
  REQUIRE( x.n_elem == 16 );

  REQUIRE( eT(x(0, 0)) == 0 );
  REQUIRE( eT(x(1, 0)) == 1 );
  REQUIRE( eT(x(2, 0)) == 2 );
  REQUIRE( eT(x(3, 0)) == 3 );
  REQUIRE( eT(x(0, 1)) == 1 );
  REQUIRE( eT(x(1, 1)) == 4 );
  REQUIRE( eT(x(2, 1)) == 5 );
  REQUIRE( eT(x(3, 1)) == 6 );
  REQUIRE( eT(x(0, 2)) == 2 );
  REQUIRE( eT(x(1, 2)) == 5 );
  REQUIRE( eT(x(2, 2)) == 7 );
  REQUIRE( eT(x(3, 2)) == 8 );
  REQUIRE( eT(x(0, 3)) == 3 );
  REQUIRE( eT(x(1, 3)) == 6 );
  REQUIRE( eT(x(2, 3)) == 8 );
  REQUIRE( eT(x(3, 3)) == 9 );

  }



TEST_CASE("symmatu_empty", "[symmat]")
  {
  fmat x;

  fmat y = symmatu(x);

  REQUIRE( y.n_rows == 0 );
  REQUIRE( y.n_cols == 0 );
  REQUIRE( y.n_elem == 0 );
  }



TEST_CASE("symmatu_empty_inplace", "[symmat]")
  {
  fmat x;

  x = symmatu(x);

  REQUIRE( x.n_rows == 0 );
  REQUIRE( x.n_cols == 0 );
  REQUIRE( x.n_elem == 0 );
  }



TEST_CASE("symmatu_one_element", "[symmat]")
  {
  fmat x(1, 1);
  x(0, 0) = 3.0;

  fmat y = symmatu(x);

  REQUIRE( y.n_rows == 1 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 1 );

  REQUIRE( float(y(0, 0)) == Approx(3.0) );
  }



TEST_CASE("symmatu_one_element_inplace", "[symmat]")
  {
  fmat x(1, 1);
  x(0, 0) = 3.0;

  x = symmatu(x);

  REQUIRE( x.n_rows == 1 );
  REQUIRE( x.n_cols == 1 );
  REQUIRE( x.n_elem == 1 );

  REQUIRE( float(x(0, 0)) == Approx(3.0) );
  }



TEMPLATE_TEST_CASE("symmatu_big_random", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(1500, 1500, distr_param(1, 1000));

  Mat<eT> y = symmatu(x, false /* ignored parameter */);

  REQUIRE( y.n_rows == 1500 );
  REQUIRE( y.n_cols == 1500 );

  arma::Mat<eT> y_cpu(y);

  REQUIRE( y_cpu.is_symmetric() );

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y_ref_cpu = arma::symmatu(x_cpu);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("symmatu_big_random_inplace", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(1500, 1500, distr_param(1, 1000));
  Mat<eT> x_ref(x);

  x = symmatu(x);

  REQUIRE( x.n_rows == 1500 );
  REQUIRE( x.n_cols == 1500 );

  arma::Mat<eT> x_cpu(x);

  REQUIRE( x_cpu.is_symmetric() );

  arma::Mat<eT> x_ref_cpu(x);
  arma::Mat<eT> x_symm_ref_cpu = arma::symmatu(x_cpu);

  REQUIRE( arma::approx_equal( x_cpu, x_symm_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("symmatu_in_expr", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(153, 153, distr_param(1, 100));

  Mat<eT> y = repmat(symmatu(x), 3, 2);

  REQUIRE( y.n_rows == x.n_rows * 3 );
  REQUIRE( y.n_cols == x.n_cols * 2 );

  Mat<eT> z = symmatu(x);
  Mat<eT> y_ref = repmat(z, 3, 2);

  arma::Mat<eT> y_cpu(y);
  arma::Mat<eT> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("expr_in_symmatu", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 40, distr_param(1, 100));

  Mat<eT> y = symmatu(repmat(x, 2, 1).t() + 4);

  REQUIRE( y.n_rows == x.n_cols );
  REQUIRE( y.n_cols == x.n_cols );

  Mat<eT> z = repmat(x, 2, 1).t() + 4;
  Mat<eT> y_ref = symmatu(z);

  arma::Mat<eT> y_cpu(y);
  arma::Mat<eT> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "symmatu_conv_to_1",
  "[symmat]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>)
  )
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(65, 65, distr_param(1, 10));
  Mat<eT2> y = symmatu(conv_to<Mat<eT2>>::from(x));

  Mat<eT2> x_conv = conv_to<Mat<eT2>>::from(x);
  Mat<eT2> y_ref = symmatu(x_conv);

  REQUIRE( y.n_rows == y_ref.n_rows );
  REQUIRE( y.n_cols == y_ref.n_cols );

  arma::Mat<eT2> y_cpu(y);
  arma::Mat<eT2> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "symmatu_conv_to_2",
  "[symmat]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>)
  )
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(65, 65, distr_param(1, 10));
  Mat<eT2> y = conv_to<Mat<eT2>>::from(symmatu(x));

  Mat<eT1> x_sym = symmatu(x);
  Mat<eT2> y_ref = conv_to<Mat<eT2>>::from(x_sym);

  REQUIRE( y.n_rows == y_ref.n_rows );
  REQUIRE( y.n_cols == y_ref.n_cols );

  arma::Mat<eT2> y_cpu(y);
  arma::Mat<eT2> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEST_CASE("symmatu_nonsquare_matrix_1", "[symmat]")
  {
  fmat x(5, 6);
  x.zeros();
  fmat y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = symmatu(x) );
  REQUIRE_THROWS( y = symmatu(x, false) );
  REQUIRE_THROWS( x = symmatu(x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



TEST_CASE("symmatu_nonsquare_matrix_2", "[symmat]")
  {
  fmat x(5, 0);
  x.zeros();
  fmat y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = symmatu(x) );
  REQUIRE_THROWS( y = symmatu(x, false) );
  REQUIRE_THROWS( x = symmatu(x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



TEST_CASE("symmatu_nonsquare_matrix_3", "[symmat]")
  {
  fmat x(0, 5);
  x.zeros();
  fmat y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = symmatu(x) );
  REQUIRE_THROWS( y = symmatu(x, false) );
  REQUIRE_THROWS( x = symmatu(x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// symmatl()

TEMPLATE_TEST_CASE("symmatl_basic", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(4, 4);
  x(0, 0) = 0;
  x(1, 0) = 1;
  x(2, 0) = 2;
  x(3, 0) = 3;
  x(1, 1) = 4;
  x(2, 1) = 5;
  x(3, 1) = 6;
  x(2, 2) = 7;
  x(3, 2) = 8;
  x(3, 3) = 9;

  Mat<eT> y = symmatl(x);

  REQUIRE( y.n_rows == x.n_rows );
  REQUIRE( y.n_cols == x.n_cols );
  REQUIRE( y.n_elem == x.n_elem );

  REQUIRE( eT(y(0, 0)) == 0 );
  REQUIRE( eT(y(1, 0)) == 1 );
  REQUIRE( eT(y(2, 0)) == 2 );
  REQUIRE( eT(y(3, 0)) == 3 );
  REQUIRE( eT(y(0, 1)) == 1 );
  REQUIRE( eT(y(1, 1)) == 4 );
  REQUIRE( eT(y(2, 1)) == 5 );
  REQUIRE( eT(y(3, 1)) == 6 );
  REQUIRE( eT(y(0, 2)) == 2 );
  REQUIRE( eT(y(1, 2)) == 5 );
  REQUIRE( eT(y(2, 2)) == 7 );
  REQUIRE( eT(y(3, 2)) == 8 );
  REQUIRE( eT(y(0, 3)) == 3 );
  REQUIRE( eT(y(1, 3)) == 6 );
  REQUIRE( eT(y(2, 3)) == 8 );
  REQUIRE( eT(y(3, 3)) == 9 );
  }



TEMPLATE_TEST_CASE("symmatl_inplace_basic", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(4, 4);
  x(0, 0) = 0;
  x(1, 0) = 1;
  x(2, 0) = 2;
  x(3, 0) = 3;
  x(1, 1) = 4;
  x(2, 1) = 5;
  x(3, 1) = 6;
  x(2, 2) = 7;
  x(3, 2) = 8;
  x(3, 3) = 9;

  x = symmatl(x);

  REQUIRE( x.n_rows == 4 );
  REQUIRE( x.n_cols == 4 );
  REQUIRE( x.n_elem == 16 );

  REQUIRE( eT(x(0, 0)) == 0 );
  REQUIRE( eT(x(1, 0)) == 1 );
  REQUIRE( eT(x(2, 0)) == 2 );
  REQUIRE( eT(x(3, 0)) == 3 );
  REQUIRE( eT(x(0, 1)) == 1 );
  REQUIRE( eT(x(1, 1)) == 4 );
  REQUIRE( eT(x(2, 1)) == 5 );
  REQUIRE( eT(x(3, 1)) == 6 );
  REQUIRE( eT(x(0, 2)) == 2 );
  REQUIRE( eT(x(1, 2)) == 5 );
  REQUIRE( eT(x(2, 2)) == 7 );
  REQUIRE( eT(x(3, 2)) == 8 );
  REQUIRE( eT(x(0, 3)) == 3 );
  REQUIRE( eT(x(1, 3)) == 6 );
  REQUIRE( eT(x(2, 3)) == 8 );
  REQUIRE( eT(x(3, 3)) == 9 );

  }



TEST_CASE("symmatl_empty", "[symmat]")
  {
  fmat x;

  fmat y = symmatl(x);

  REQUIRE( y.n_rows == 0 );
  REQUIRE( y.n_cols == 0 );
  REQUIRE( y.n_elem == 0 );
  }



TEST_CASE("symmatl_empty_inplace", "[symmat]")
  {
  fmat x;

  x = symmatl(x);

  REQUIRE( x.n_rows == 0 );
  REQUIRE( x.n_cols == 0 );
  REQUIRE( x.n_elem == 0 );
  }



TEST_CASE("symmatl_one_element", "[symmat]")
  {
  fmat x(1, 1);
  x(0, 0) = 3.0;

  fmat y = symmatl(x);

  REQUIRE( y.n_rows == 1 );
  REQUIRE( y.n_cols == 1 );
  REQUIRE( y.n_elem == 1 );

  REQUIRE( float(y(0, 0)) == Approx(3.0) );
  }



TEST_CASE("symmatl_one_element_inplace", "[symmat]")
  {
  fmat x(1, 1);
  x(0, 0) = 3.0;

  x = symmatl(x);

  REQUIRE( x.n_rows == 1 );
  REQUIRE( x.n_cols == 1 );
  REQUIRE( x.n_elem == 1 );

  REQUIRE( float(x(0, 0)) == Approx(3.0) );
  }



TEMPLATE_TEST_CASE("symmatl_big_random", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(1500, 1500, distr_param(1, 1000));

  Mat<eT> y = symmatl(x, false /* ignored parameter */);

  REQUIRE( y.n_rows == 1500 );
  REQUIRE( y.n_cols == 1500 );

  arma::Mat<eT> y_cpu(y);

  REQUIRE( y_cpu.is_symmetric() );

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y_ref_cpu = arma::symmatl(x_cpu);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("symmatl_big_random_inplace", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(1500, 1500, distr_param(1, 1000));
  Mat<eT> x_ref(x);

  x = symmatu(x);

  REQUIRE( x.n_rows == 1500 );
  REQUIRE( x.n_cols == 1500 );

  arma::Mat<eT> x_cpu(x);

  REQUIRE( x_cpu.is_symmetric() );

  arma::Mat<eT> x_ref_cpu(x);
  arma::Mat<eT> x_symm_ref_cpu = arma::symmatl(x_cpu);

  REQUIRE( arma::approx_equal( x_cpu, x_symm_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("symmatl_in_expr", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(153, 153, distr_param(1, 100));

  Mat<eT> y = repmat(symmatl(x), 3, 2);

  REQUIRE( y.n_rows == x.n_rows * 3 );
  REQUIRE( y.n_cols == x.n_cols * 2 );

  Mat<eT> z = symmatl(x);
  Mat<eT> y_ref = repmat(z, 3, 2);

  arma::Mat<eT> y_cpu(y);
  arma::Mat<eT> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("expr_in_symmatl", "[symmat]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(20, 40, distr_param(1, 100));

  Mat<eT> y = symmatl(repmat(x, 2, 1).t() + 4);

  REQUIRE( y.n_rows == x.n_cols );
  REQUIRE( y.n_cols == x.n_cols );

  Mat<eT> z = repmat(x, 2, 1).t() + 4;
  Mat<eT> y_ref = symmatl(z);

  arma::Mat<eT> y_cpu(y);
  arma::Mat<eT> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "symmatl_conv_to_1",
  "[symmat]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>)
  )
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(65, 65, distr_param(1, 10));
  Mat<eT2> y = symmatl(conv_to<Mat<eT2>>::from(x));

  Mat<eT2> x_conv = conv_to<Mat<eT2>>::from(x);
  Mat<eT2> y_ref = symmatl(x_conv);

  REQUIRE( y.n_rows == y_ref.n_rows );
  REQUIRE( y.n_cols == y_ref.n_cols );

  arma::Mat<eT2> y_cpu(y);
  arma::Mat<eT2> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE
  (
  "symmatl_conv_to_2",
  "[symmat]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>)
  )
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Mat<eT1> x = randi<Mat<eT1>>(65, 65, distr_param(1, 10));
  Mat<eT2> y = conv_to<Mat<eT2>>::from(symmatl(x));

  Mat<eT1> x_sym = symmatl(x);
  Mat<eT2> y_ref = conv_to<Mat<eT2>>::from(x_sym);

  REQUIRE( y.n_rows == y_ref.n_rows );
  REQUIRE( y.n_cols == y_ref.n_cols );

  arma::Mat<eT2> y_cpu(y);
  arma::Mat<eT2> y_ref_cpu(y_ref);

  REQUIRE( arma::approx_equal( y_cpu, y_ref_cpu, "reldiff", 1e-5 ) );
  }



TEST_CASE("symmatl_nonsquare_matrix_1", "[symmat]")
  {
  fmat x(5, 6);
  x.zeros();
  fmat y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = symmatl(x) );
  REQUIRE_THROWS( y = symmatl(x, false) );
  REQUIRE_THROWS( x = symmatl(x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



TEST_CASE("symmatl_nonsquare_matrix_2", "[symmat]")
  {
  fmat x(5, 0);
  x.zeros();
  fmat y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = symmatl(x) );
  REQUIRE_THROWS( y = symmatl(x, false) );
  REQUIRE_THROWS( x = symmatl(x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



TEST_CASE("symmatl_nonsquare_matrix_3", "[symmat]")
  {
  fmat x(0, 5);
  x.zeros();
  fmat y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = symmatl(x) );
  REQUIRE_THROWS( y = symmatl(x, false) );
  REQUIRE_THROWS( x = symmatl(x) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }
