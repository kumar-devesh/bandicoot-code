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

TEMPLATE_TEST_CASE("simple_sort_test", "[sort]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(10);
  x(0) = 6;
  x(1) = 5;
  x(2) = 1;
  x(3) = 7;
  x(4) = 0;
  x(5) = 9;
  x(6) = 3;
  x(7) = 2;
  x(8) = 8;
  x(9) = 4;

  Col<eT> x1_asc = sort(x);
  Col<eT> x2_asc = sort(x, "ascend");
  Col<eT> x1_desc = sort(x, "descend");

  REQUIRE( x1_asc.n_elem == 10 );
  REQUIRE( x2_asc.n_elem == 10 );
  REQUIRE( x1_desc.n_elem == 10 );

  REQUIRE( eT(x1_asc[0]) == eT(0) );
  REQUIRE( eT(x1_asc[1]) == eT(1) );
  REQUIRE( eT(x1_asc[2]) == eT(2) );
  REQUIRE( eT(x1_asc[3]) == eT(3) );
  REQUIRE( eT(x1_asc[4]) == eT(4) );
  REQUIRE( eT(x1_asc[5]) == eT(5) );
  REQUIRE( eT(x1_asc[6]) == eT(6) );
  REQUIRE( eT(x1_asc[7]) == eT(7) );
  REQUIRE( eT(x1_asc[8]) == eT(8) );
  REQUIRE( eT(x1_asc[9]) == eT(9) );
  REQUIRE( eT(x2_asc[0]) == eT(0) );
  REQUIRE( eT(x2_asc[1]) == eT(1) );
  REQUIRE( eT(x2_asc[2]) == eT(2) );
  REQUIRE( eT(x2_asc[3]) == eT(3) );
  REQUIRE( eT(x2_asc[4]) == eT(4) );
  REQUIRE( eT(x2_asc[5]) == eT(5) );
  REQUIRE( eT(x2_asc[6]) == eT(6) );
  REQUIRE( eT(x2_asc[7]) == eT(7) );
  REQUIRE( eT(x2_asc[8]) == eT(8) );
  REQUIRE( eT(x2_asc[9]) == eT(9) );
  REQUIRE( eT(x1_desc[0]) == eT(9) );
  REQUIRE( eT(x1_desc[1]) == eT(8) );
  REQUIRE( eT(x1_desc[2]) == eT(7) );
  REQUIRE( eT(x1_desc[3]) == eT(6) );
  REQUIRE( eT(x1_desc[4]) == eT(5) );
  REQUIRE( eT(x1_desc[5]) == eT(4) );
  REQUIRE( eT(x1_desc[6]) == eT(3) );
  REQUIRE( eT(x1_desc[7]) == eT(2) );
  REQUIRE( eT(x1_desc[8]) == eT(1) );
  REQUIRE( eT(x1_desc[9]) == eT(0) );
  }



TEMPLATE_TEST_CASE("random_vector_float_data_test", "[sort]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = (std::numeric_limits<eT>::max() / 2) * (2 * randu<Col<eT>>(50000) - 0.5);

  Col<eT> x1_asc = sort(x);
  Col<eT> x2_asc = sort(x, "ascend");
  Col<eT> x1_desc = sort(x, "descend");

  REQUIRE( x1_asc.n_elem == 50000 );
  REQUIRE( x2_asc.n_elem == 50000 );
  REQUIRE( x1_desc.n_elem == 50000 );

  arma::Col<eT> x1_asc_cpu(x1_asc);
  arma::Col<eT> x2_asc_cpu(x2_asc);
  arma::Col<eT> x1_desc_cpu(x1_desc);

  arma::Col<eT> x_cpu(x);

  arma::Col<eT> x1_ref_asc_cpu = arma::sort(x_cpu);
  arma::Col<eT> x2_ref_asc_cpu = arma::sort(x_cpu, "ascend");
  arma::Col<eT> x1_ref_desc_cpu = arma::sort(x_cpu, "descend");

  REQUIRE( arma::approx_equal( x1_asc_cpu, x1_ref_asc_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x2_asc_cpu, x2_ref_asc_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x1_desc_cpu, x1_ref_desc_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("random_vector_integer_data_test", "[sort]", u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x;
  if (std::is_same<eT, u32>::value)
    {
    x = 2 * randi<Col<eT>>(50000, distr_param((int) 0, (int) std::numeric_limits<s32>::max()));
    x[1210] += 1;
    x[301] += 1; // just a couple elements with the last bit high
    }
  if (std::is_same<eT, s32>::value)
    {
    x = randi<Col<eT>>(50000, distr_param((int) std::numeric_limits<s32>::min(), (int) std::numeric_limits<s32>::max()));
    }
  if (std::is_same<eT, u64>::value)
    {
    x = 2 * randi<Col<eT>>(50000, distr_param((int) 0, (int) std::numeric_limits<s32>::max()));
    x[1210] += 1;
    x[301] += 1; // just a couple elements with the last bit high
    Col<eT> y = 2 * randi<Col<eT>>(50000, distr_param((int) 0, (int) std::numeric_limits<s32>::max()));
    y[11] += 1;
    y[165] += 1; // just a couple elements with the last bit high
    x %= y; // should get us to the whole range of u64s
    }
  if (std::is_same<eT, s64>::value)
    {
    x = randi<Col<eT>>(50000, distr_param((int) std::numeric_limits<s32>::min(), (int) std::numeric_limits<s32>::max()));
    Col<eT> y = randi<Col<eT>>(50000, distr_param((int) std::numeric_limits<s32>::min(), (int) std::numeric_limits<s32>::max()));
    x %= y; // should get us to the whole range of s64s
    }

  Col<eT> x1_asc = sort(x);
  Col<eT> x2_asc = sort(x, "ascend");
  Col<eT> x1_desc = sort(x, "descend");

  REQUIRE( x1_asc.n_elem == 50000 );
  REQUIRE( x2_asc.n_elem == 50000 );
  REQUIRE( x1_desc.n_elem == 50000 );

  arma::Col<eT> x1_asc_cpu(x1_asc);
  arma::Col<eT> x2_asc_cpu(x2_asc);
  arma::Col<eT> x1_desc_cpu(x1_desc);

  arma::Col<eT> x_cpu(x);

  arma::Col<eT> x1_ref_asc_cpu = arma::sort(x_cpu);
  arma::Col<eT> x2_ref_asc_cpu = arma::sort(x_cpu, "ascend");
  arma::Col<eT> x1_ref_desc_cpu = arma::sort(x_cpu, "descend");

  REQUIRE( arma::approx_equal( x1_asc_cpu, x1_ref_asc_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x2_asc_cpu, x2_ref_asc_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x1_desc_cpu, x1_ref_desc_cpu, "reldiff", 1e-5 ) );
  }



// sort data where every bit is the same (probably unnecessary)
TEMPLATE_TEST_CASE("identical_data_sort_test", "[sort]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(10000);
  x.zeros();

  Col<eT> y1 = sort(x);
  Col<eT> y2 = sort(x, "ascend");
  Col<eT> y3 = sort(x, "descend");

  REQUIRE( y1.n_elem == 10000 );
  REQUIRE( y2.n_elem == 10000 );
  REQUIRE( y3.n_elem == 10000 );

  arma::Col<eT> x_cpu(x);
  arma::Col<eT> y1_cpu(y1);
  arma::Col<eT> y2_cpu(y2);
  arma::Col<eT> y3_cpu(y3);

  REQUIRE( arma::approx_equal( y1_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y2_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y3_cpu, x_cpu, "absdiff", 1e-5 ) );
  }



// sort empty data
TEST_CASE("empty_data_sort", "[sort]")
  {
  vec x;

  vec y1 = sort(x);
  vec y2 = sort(x, "ascend");
  vec y3 = sort(x, "descend");

  REQUIRE( y1.n_elem == 0 );
  REQUIRE( y2.n_elem == 0 );
  REQUIRE( y3.n_elem == 0 );
  }



// sort one element data
TEST_CASE("one_elem_sort", "[sort]")
  {
  fvec x(1);
  x(0) = 1.0;

  fvec y1 = sort(x);
  fvec y2 = sort(x, "ascend");
  fvec y3 = sort(x, "descend");

  REQUIRE( y1.n_elem == 1 );
  REQUIRE( y2.n_elem == 1 );
  REQUIRE( y3.n_elem == 1 );

  REQUIRE( float(y1[0]) == Approx(1.0) );
  REQUIRE( float(y2[0]) == Approx(1.0) );
  REQUIRE( float(y3[0]) == Approx(1.0) );
  }



// sort an expression
TEST_CASE("sort_expr", "[sort]")
  {
  fvec x = randu<fvec>(1023);

  fvec x_mod = 3 * (x + 4);

  fvec y1 = sort(3 * (x + 4));
  fvec y2 = sort(3 * (x + 4), "ascend");
  fvec y3 = sort(3 * (x + 4), "descend");

  REQUIRE( y1.n_elem == 1023 );
  REQUIRE( y2.n_elem == 1023 );
  REQUIRE( y3.n_elem == 1023 );

  fvec y_asc_ref = sort(x_mod, "ascend");
  fvec y_desc_ref = sort(x_mod, "descend");

  arma::fvec y1_cpu(y1);
  arma::fvec y2_cpu(y2);
  arma::fvec y3_cpu(y3);

  arma::fvec y_asc_ref_cpu(y_asc_ref);
  arma::fvec y_desc_ref_cpu(y_desc_ref);

  REQUIRE( arma::approx_equal( y1_cpu, y_asc_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y2_cpu, y_asc_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y3_cpu, y_desc_ref_cpu, "reldiff", 1e-5 ) );
  }



// sort inside expression
TEST_CASE("sort_inside_expr", "[sort]")
  {
  fvec x = randu<fvec>(1025);

  fvec x_sorted_asc = sort(x, "ascend");
  fvec x_sorted_desc = sort(x, "descend");

  fmat y1 = repmat(sort(x), 2, 3);
  fmat y2 = repmat(sort(x, "ascend"), 2, 3);
  fmat y3 = repmat(sort(x, "descend"), 2, 3);

  fmat y_asc_ref = repmat(x_sorted_asc, 2, 3);
  fmat y_desc_ref = repmat(x_sorted_desc, 2, 3);

  REQUIRE( y1.n_rows == y_asc_ref.n_rows );
  REQUIRE( y1.n_cols == y_asc_ref.n_cols );
  REQUIRE( y2.n_rows == y_asc_ref.n_rows );
  REQUIRE( y2.n_cols == y_asc_ref.n_cols );
  REQUIRE( y3.n_rows == y_desc_ref.n_rows );
  REQUIRE( y3.n_cols == y_desc_ref.n_cols );

  arma::fmat y1_cpu(y1);
  arma::fmat y2_cpu(y2);
  arma::fmat y3_cpu(y3);

  arma::fmat y_asc_ref_cpu(y_asc_ref);
  arma::fmat y_desc_ref_cpu(y_desc_ref);

  REQUIRE( arma::approx_equal( y1_cpu, y_asc_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y2_cpu, y_asc_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y3_cpu, y_desc_ref_cpu, "reldiff", 1e-5 ) );
  }



// incorrect sort direction should throw exception
TEST_CASE("invalid_sort_direction", "[sort]")
  {
  fvec x = randu<fvec>(10);
  fvec y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = sort(x, "hello") );
  REQUIRE_THROWS( y = sort(x, "") );
  REQUIRE_THROWS( y = sort(x, "things") );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// all the same stuff, but columnwise or rowwise
TEMPLATE_TEST_CASE("simple_mat_sort_test", "[sort]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  // [[10, 11,  5, 14,  8]
  //  [ 0, 13,  7,  9,  1]
  //  [ 4,  6,  2,  3, 12]]
  Mat<eT> x(3, 5);
  x(0, 0) = eT(10);
  x(1, 0) = eT(0);
  x(2, 0) = eT(4);
  x(0, 1) = eT(11);
  x(1, 1) = eT(13);
  x(2, 1) = eT(6);
  x(0, 2) = eT(5);
  x(1, 2) = eT(7);
  x(2, 2) = eT(2);
  x(0, 3) = eT(14);
  x(1, 3) = eT(9);
  x(2, 3) = eT(3);
  x(0, 4) = eT(8);
  x(1, 4) = eT(1);
  x(2, 4) = eT(12);

  Mat<eT> x1 = sort(x);
  Mat<eT> x2 = sort(x, "ascend");
  Mat<eT> x3 = sort(x, "ascend", 0);
  Mat<eT> x4 = sort(x, "ascend", 1);
  Mat<eT> x5 = sort(x, "descend");
  Mat<eT> x6 = sort(x, "descend", 0);
  Mat<eT> x7 = sort(x, "descend", 1);

  REQUIRE( x1.n_rows == 3 );
  REQUIRE( x1.n_cols == 5 );
  REQUIRE( x2.n_rows == 3 );
  REQUIRE( x2.n_cols == 5 );
  REQUIRE( x3.n_rows == 3 );
  REQUIRE( x3.n_cols == 5 );
  REQUIRE( x4.n_rows == 3 );
  REQUIRE( x4.n_cols == 5 );
  REQUIRE( x5.n_rows == 3 );
  REQUIRE( x5.n_cols == 5 );
  REQUIRE( x6.n_rows == 3 );
  REQUIRE( x6.n_cols == 5 );
  REQUIRE( x7.n_rows == 3 );
  REQUIRE( x7.n_cols == 5 );

  REQUIRE( eT(x1(0, 0)) == Approx(eT(0)) );
  REQUIRE( eT(x1(1, 0)) == Approx(eT(4)) );
  REQUIRE( eT(x1(2, 0)) == Approx(eT(10)) );
  REQUIRE( eT(x1(0, 1)) == Approx(eT(6)) );
  REQUIRE( eT(x1(1, 1)) == Approx(eT(11)) );
  REQUIRE( eT(x1(2, 1)) == Approx(eT(13)) );
  REQUIRE( eT(x1(0, 2)) == Approx(eT(2)) );
  REQUIRE( eT(x1(1, 2)) == Approx(eT(5)) );
  REQUIRE( eT(x1(2, 2)) == Approx(eT(7)) );
  REQUIRE( eT(x1(0, 3)) == Approx(eT(3)) );
  REQUIRE( eT(x1(1, 3)) == Approx(eT(9)) );
  REQUIRE( eT(x1(2, 3)) == Approx(eT(14)) );
  REQUIRE( eT(x1(0, 4)) == Approx(eT(1)) );
  REQUIRE( eT(x1(1, 4)) == Approx(eT(8)) );
  REQUIRE( eT(x1(2, 4)) == Approx(eT(12)) );

  REQUIRE( eT(x2(0, 0)) == Approx(eT(0)) );
  REQUIRE( eT(x2(1, 0)) == Approx(eT(4)) );
  REQUIRE( eT(x2(2, 0)) == Approx(eT(10)) );
  REQUIRE( eT(x2(0, 1)) == Approx(eT(6)) );
  REQUIRE( eT(x2(1, 1)) == Approx(eT(11)) );
  REQUIRE( eT(x2(2, 1)) == Approx(eT(13)) );
  REQUIRE( eT(x2(0, 2)) == Approx(eT(2)) );
  REQUIRE( eT(x2(1, 2)) == Approx(eT(5)) );
  REQUIRE( eT(x2(2, 2)) == Approx(eT(7)) );
  REQUIRE( eT(x2(0, 3)) == Approx(eT(3)) );
  REQUIRE( eT(x2(1, 3)) == Approx(eT(9)) );
  REQUIRE( eT(x2(2, 3)) == Approx(eT(14)) );
  REQUIRE( eT(x2(0, 4)) == Approx(eT(1)) );
  REQUIRE( eT(x2(1, 4)) == Approx(eT(8)) );
  REQUIRE( eT(x2(2, 4)) == Approx(eT(12)) );

  REQUIRE( eT(x3(0, 0)) == Approx(eT(0)) );
  REQUIRE( eT(x3(1, 0)) == Approx(eT(4)) );
  REQUIRE( eT(x3(2, 0)) == Approx(eT(10)) );
  REQUIRE( eT(x3(0, 1)) == Approx(eT(6)) );
  REQUIRE( eT(x3(1, 1)) == Approx(eT(11)) );
  REQUIRE( eT(x3(2, 1)) == Approx(eT(13)) );
  REQUIRE( eT(x3(0, 2)) == Approx(eT(2)) );
  REQUIRE( eT(x3(1, 2)) == Approx(eT(5)) );
  REQUIRE( eT(x3(2, 2)) == Approx(eT(7)) );
  REQUIRE( eT(x3(0, 3)) == Approx(eT(3)) );
  REQUIRE( eT(x3(1, 3)) == Approx(eT(9)) );
  REQUIRE( eT(x3(2, 3)) == Approx(eT(14)) );
  REQUIRE( eT(x3(0, 4)) == Approx(eT(1)) );
  REQUIRE( eT(x3(1, 4)) == Approx(eT(8)) );
  REQUIRE( eT(x3(2, 4)) == Approx(eT(12)) );

  // sorted rowwise
  REQUIRE( eT(x4(0, 0)) == Approx(eT(5)) );
  REQUIRE( eT(x4(0, 1)) == Approx(eT(8)) );
  REQUIRE( eT(x4(0, 2)) == Approx(eT(10)) );
  REQUIRE( eT(x4(0, 3)) == Approx(eT(11)) );
  REQUIRE( eT(x4(0, 4)) == Approx(eT(14)) );
  REQUIRE( eT(x4(1, 0)) == Approx(eT(0)) );
  REQUIRE( eT(x4(1, 1)) == Approx(eT(1)) );
  REQUIRE( eT(x4(1, 2)) == Approx(eT(7)) );
  REQUIRE( eT(x4(1, 3)) == Approx(eT(9)) );
  REQUIRE( eT(x4(1, 4)) == Approx(eT(13)) );
  REQUIRE( eT(x4(2, 0)) == Approx(eT(2)) );
  REQUIRE( eT(x4(2, 1)) == Approx(eT(3)) );
  REQUIRE( eT(x4(2, 2)) == Approx(eT(4)) );
  REQUIRE( eT(x4(2, 3)) == Approx(eT(6)) );
  REQUIRE( eT(x4(2, 4)) == Approx(eT(12)) );

  // sorted in descending order
  REQUIRE( eT(x5(0, 0)) == Approx(eT(10)) );
  REQUIRE( eT(x5(1, 0)) == Approx(eT(4)) );
  REQUIRE( eT(x5(2, 0)) == Approx(eT(0)) );
  REQUIRE( eT(x5(0, 1)) == Approx(eT(13)) );
  REQUIRE( eT(x5(1, 1)) == Approx(eT(11)) );
  REQUIRE( eT(x5(2, 1)) == Approx(eT(6)) );
  REQUIRE( eT(x5(0, 2)) == Approx(eT(7)) );
  REQUIRE( eT(x5(1, 2)) == Approx(eT(5)) );
  REQUIRE( eT(x5(2, 2)) == Approx(eT(2)) );
  REQUIRE( eT(x5(0, 3)) == Approx(eT(14)) );
  REQUIRE( eT(x5(1, 3)) == Approx(eT(9)) );
  REQUIRE( eT(x5(2, 3)) == Approx(eT(3)) );
  REQUIRE( eT(x5(0, 4)) == Approx(eT(12)) );
  REQUIRE( eT(x5(1, 4)) == Approx(eT(8)) );
  REQUIRE( eT(x5(2, 4)) == Approx(eT(1)) );

  REQUIRE( eT(x6(0, 0)) == Approx(eT(10)) );
  REQUIRE( eT(x6(1, 0)) == Approx(eT(4)) );
  REQUIRE( eT(x6(2, 0)) == Approx(eT(0)) );
  REQUIRE( eT(x6(0, 1)) == Approx(eT(13)) );
  REQUIRE( eT(x6(1, 1)) == Approx(eT(11)) );
  REQUIRE( eT(x6(2, 1)) == Approx(eT(6)) );
  REQUIRE( eT(x6(0, 2)) == Approx(eT(7)) );
  REQUIRE( eT(x6(1, 2)) == Approx(eT(5)) );
  REQUIRE( eT(x6(2, 2)) == Approx(eT(2)) );
  REQUIRE( eT(x6(0, 3)) == Approx(eT(14)) );
  REQUIRE( eT(x6(1, 3)) == Approx(eT(9)) );
  REQUIRE( eT(x6(2, 3)) == Approx(eT(3)) );
  REQUIRE( eT(x6(0, 4)) == Approx(eT(12)) );
  REQUIRE( eT(x6(1, 4)) == Approx(eT(8)) );
  REQUIRE( eT(x6(2, 4)) == Approx(eT(1)) );

  // sorted rowwise descending order
  REQUIRE( eT(x7(0, 0)) == Approx(eT(14)) );
  REQUIRE( eT(x7(0, 1)) == Approx(eT(11)) );
  REQUIRE( eT(x7(0, 2)) == Approx(eT(10)) );
  REQUIRE( eT(x7(0, 3)) == Approx(eT(8)) );
  REQUIRE( eT(x7(0, 4)) == Approx(eT(5)) );
  REQUIRE( eT(x7(1, 0)) == Approx(eT(13)) );
  REQUIRE( eT(x7(1, 1)) == Approx(eT(9)) );
  REQUIRE( eT(x7(1, 2)) == Approx(eT(7)) );
  REQUIRE( eT(x7(1, 3)) == Approx(eT(1)) );
  REQUIRE( eT(x7(1, 4)) == Approx(eT(0)) );
  REQUIRE( eT(x7(2, 0)) == Approx(eT(12)) );
  REQUIRE( eT(x7(2, 1)) == Approx(eT(6)) );
  REQUIRE( eT(x7(2, 2)) == Approx(eT(4)) );
  REQUIRE( eT(x7(2, 3)) == Approx(eT(3)) );
  REQUIRE( eT(x7(2, 4)) == Approx(eT(2)) );
  }



TEMPLATE_TEST_CASE("random_mat_vector_float_data_test", "[sort]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = (std::numeric_limits<eT>::max() / 2) * (2 * randu<Mat<eT>>(323, 600) - 0.5);

  Mat<eT> x1 = sort(x);
  Mat<eT> x2 = sort(x, "ascend");
  Mat<eT> x3 = sort(x, "ascend", 0);
  Mat<eT> x4 = sort(x, "ascend", 1);
  Mat<eT> x5 = sort(x, "descend");
  Mat<eT> x6 = sort(x, "descend", 0);
  Mat<eT> x7 = sort(x, "descend", 1);

  REQUIRE( x1.n_rows == x.n_rows);
  REQUIRE( x1.n_cols == x.n_cols);
  REQUIRE( x2.n_rows == x.n_rows);
  REQUIRE( x2.n_cols == x.n_cols);
  REQUIRE( x3.n_rows == x.n_rows);
  REQUIRE( x3.n_cols == x.n_cols);
  REQUIRE( x4.n_rows == x.n_rows);
  REQUIRE( x4.n_cols == x.n_cols);
  REQUIRE( x5.n_rows == x.n_rows);
  REQUIRE( x5.n_cols == x.n_cols);
  REQUIRE( x6.n_rows == x.n_rows);
  REQUIRE( x6.n_cols == x.n_cols);
  REQUIRE( x7.n_rows == x.n_rows);
  REQUIRE( x7.n_cols == x.n_cols);

  arma::Mat<eT> x1_cpu(x1);
  arma::Mat<eT> x2_cpu(x2);
  arma::Mat<eT> x3_cpu(x3);
  arma::Mat<eT> x4_cpu(x4);
  arma::Mat<eT> x5_cpu(x5);
  arma::Mat<eT> x6_cpu(x6);
  arma::Mat<eT> x7_cpu(x7);

  arma::Mat<eT> x_cpu(x);

  arma::Mat<eT> x1_ref_cpu = arma::sort(x_cpu, "ascend", 0);
  arma::Mat<eT> x2_ref_cpu = arma::sort(x_cpu, "ascend", 1);
  arma::Mat<eT> x3_ref_cpu = arma::sort(x_cpu, "descend", 0);
  arma::Mat<eT> x4_ref_cpu = arma::sort(x_cpu, "descend", 1);

  REQUIRE( arma::approx_equal( x1_cpu, x1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x2_cpu, x1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x3_cpu, x1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x4_cpu, x2_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x5_cpu, x3_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x6_cpu, x3_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x7_cpu, x4_ref_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("random_mat_vector_integer_data_test", "[sort]", u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x;
  if (std::is_same<eT, u32>::value)
    {
    x = 2 * randi<Mat<eT>>(345, 400, distr_param((int) 0, (int) std::numeric_limits<s32>::max()));
    x(12, 100) += 1;
    x(301, 3) += 1; // just a couple elements with the last bit high
    }
  if (std::is_same<eT, s32>::value)
    {
    x = randi<Mat<eT>>(345, 400, distr_param((int) std::numeric_limits<s32>::min(), (int) std::numeric_limits<s32>::max()));
    }
  if (std::is_same<eT, u64>::value)
    {
    x = 2 * randi<Mat<eT>>(345, 400, distr_param((int) 0, (int) std::numeric_limits<s32>::max()));
    x(12, 100) += 1;
    x(301, 3) += 1; // just a couple elements with the last bit high
    Mat<eT> y = 2 * randi<Mat<eT>>(345, 400, distr_param((int) 0, (int) std::numeric_limits<s32>::max()));
    y(11, 66) += 1;
    y(155, 311) += 1; // just a couple elements with the last bit high
    x %= y; // should get us to the whole range of u64s
    }
  if (std::is_same<eT, s64>::value)
    {
    x = randi<Mat<eT>>(345, 400, distr_param((int) std::numeric_limits<s32>::min(), (int) std::numeric_limits<s32>::max()));
    Mat<eT> y = randi<Mat<eT>>(345, 400, distr_param((int) std::numeric_limits<s32>::min(), (int) std::numeric_limits<s32>::max()));
    x %= y; // should get us to the whole range of s64s
    }

  Mat<eT> x1 = sort(x);
  Mat<eT> x2 = sort(x, "ascend");
  Mat<eT> x3 = sort(x, "ascend", 0);
  Mat<eT> x4 = sort(x, "ascend", 1);
  Mat<eT> x5 = sort(x, "descend");
  Mat<eT> x6 = sort(x, "descend", 0);
  Mat<eT> x7 = sort(x, "descend", 1);

  REQUIRE( x1.n_rows == x.n_rows);
  REQUIRE( x1.n_cols == x.n_cols);
  REQUIRE( x2.n_rows == x.n_rows);
  REQUIRE( x2.n_cols == x.n_cols);
  REQUIRE( x3.n_rows == x.n_rows);
  REQUIRE( x3.n_cols == x.n_cols);
  REQUIRE( x4.n_rows == x.n_rows);
  REQUIRE( x4.n_cols == x.n_cols);
  REQUIRE( x5.n_rows == x.n_rows);
  REQUIRE( x5.n_cols == x.n_cols);
  REQUIRE( x6.n_rows == x.n_rows);
  REQUIRE( x6.n_cols == x.n_cols);
  REQUIRE( x7.n_rows == x.n_rows);
  REQUIRE( x7.n_cols == x.n_cols);

  arma::Mat<eT> x1_cpu(x1);
  arma::Mat<eT> x2_cpu(x2);
  arma::Mat<eT> x3_cpu(x3);
  arma::Mat<eT> x4_cpu(x4);
  arma::Mat<eT> x5_cpu(x5);
  arma::Mat<eT> x6_cpu(x6);
  arma::Mat<eT> x7_cpu(x7);

  arma::Mat<eT> x_cpu(x);

  arma::Mat<eT> x1_ref_cpu = arma::sort(x_cpu, "ascend", 0);
  arma::Mat<eT> x2_ref_cpu = arma::sort(x_cpu, "ascend", 1);
  arma::Mat<eT> x3_ref_cpu = arma::sort(x_cpu, "descend", 0);
  arma::Mat<eT> x4_ref_cpu = arma::sort(x_cpu, "descend", 1);

  REQUIRE( arma::approx_equal( x1_cpu, x1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x2_cpu, x1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x3_cpu, x1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x4_cpu, x2_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x5_cpu, x3_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x6_cpu, x3_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x7_cpu, x4_ref_cpu, "reldiff", 1e-5 ) );
  }



// sort data where every bit is the same (probably unnecessary)
TEMPLATE_TEST_CASE("identical_data_mat_sort_test", "[sort]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(150, 200);
  x.zeros();

  Mat<eT> y1 = sort(x);
  Mat<eT> y2 = sort(x, "ascend");
  Mat<eT> y3 = sort(x, "ascend", 0);
  Mat<eT> y4 = sort(x, "ascend", 1);
  Mat<eT> y5 = sort(x, "descend");
  Mat<eT> y6 = sort(x, "descend", 0);
  Mat<eT> y7 = sort(x, "descend", 1);

  REQUIRE( y1.n_rows == 150 );
  REQUIRE( y1.n_cols == 200 );
  REQUIRE( y2.n_rows == 150 );
  REQUIRE( y2.n_cols == 200 );
  REQUIRE( y3.n_rows == 150 );
  REQUIRE( y3.n_cols == 200 );
  REQUIRE( y4.n_rows == 150 );
  REQUIRE( y4.n_cols == 200 );
  REQUIRE( y5.n_rows == 150 );
  REQUIRE( y5.n_cols == 200 );
  REQUIRE( y6.n_rows == 150 );
  REQUIRE( y6.n_cols == 200 );
  REQUIRE( y7.n_rows == 150 );
  REQUIRE( y7.n_cols == 200 );

  arma::Mat<eT> x_cpu(x);
  arma::Mat<eT> y1_cpu(y1);
  arma::Mat<eT> y2_cpu(y2);
  arma::Mat<eT> y3_cpu(y3);
  arma::Mat<eT> y4_cpu(y4);
  arma::Mat<eT> y5_cpu(y5);
  arma::Mat<eT> y6_cpu(y6);
  arma::Mat<eT> y7_cpu(y7);

  REQUIRE( arma::approx_equal( y1_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y2_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y3_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y4_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y5_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y6_cpu, x_cpu, "absdiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y7_cpu, x_cpu, "absdiff", 1e-5 ) );
  }



// sort empty data
TEST_CASE("empty_data_mat_sort", "[sort]")
  {
  fmat x;

  fmat y1 = sort(x);
  fmat y2 = sort(x, "ascend");
  fmat y3 = sort(x, "ascend", 0);
  fmat y4 = sort(x, "ascend", 1);
  fmat y5 = sort(x, "descend");
  fmat y6 = sort(x, "descend", 0);
  fmat y7 = sort(x, "descend", 1);

  REQUIRE( y1.n_rows == 0 );
  REQUIRE( y1.n_cols == 0 );
  REQUIRE( y2.n_rows == 0 );
  REQUIRE( y2.n_cols == 0 );
  REQUIRE( y3.n_rows == 0 );
  REQUIRE( y3.n_cols == 0 );
  REQUIRE( y4.n_rows == 0 );
  REQUIRE( y4.n_cols == 0 );
  REQUIRE( y5.n_rows == 0 );
  REQUIRE( y5.n_cols == 0 );
  REQUIRE( y6.n_rows == 0 );
  REQUIRE( y6.n_cols == 0 );
  REQUIRE( y7.n_rows == 0 );
  REQUIRE( y7.n_cols == 0 );
  }



// sort one element data
TEST_CASE("one_elem_mat_sort", "[sort]")
  {
  fmat x(1, 1);
  x(0) = 1.0;

  fmat y1 = sort(x);
  fmat y2 = sort(x, "ascend");
  fmat y3 = sort(x, "ascend", 0);
  fmat y4 = sort(x, "ascend", 1);
  fmat y5 = sort(x, "descend");
  fmat y6 = sort(x, "descend", 0);
  fmat y7 = sort(x, "descend", 1);

  REQUIRE( y1.n_rows == 1 );
  REQUIRE( y1.n_cols == 1 );
  REQUIRE( y2.n_rows == 1 );
  REQUIRE( y2.n_cols == 1 );
  REQUIRE( y3.n_rows == 1 );
  REQUIRE( y3.n_cols == 1 );
  REQUIRE( y4.n_rows == 1 );
  REQUIRE( y4.n_cols == 1 );
  REQUIRE( y5.n_rows == 1 );
  REQUIRE( y5.n_cols == 1 );
  REQUIRE( y6.n_rows == 1 );
  REQUIRE( y6.n_cols == 1 );
  REQUIRE( y7.n_rows == 1 );
  REQUIRE( y7.n_cols == 1 );

  REQUIRE( float(y1(0, 0)) == Approx(1.0) );
  REQUIRE( float(y2(0, 0)) == Approx(1.0) );
  REQUIRE( float(y3(0, 0)) == Approx(1.0) );
  REQUIRE( float(y4(0, 0)) == Approx(1.0) );
  REQUIRE( float(y5(0, 0)) == Approx(1.0) );
  REQUIRE( float(y6(0, 0)) == Approx(1.0) );
  REQUIRE( float(y7(0, 0)) == Approx(1.0) );
  }



// sort an expression
TEST_CASE("sort_mat_expr", "[sort]")
  {
  fmat x = randu<fmat>(63, 125);

  fmat x_mod = 3 * (x.t() + 4);

  fmat y1 = sort(3 * (x.t() + 4));
  fmat y2 = sort(3 * (x.t() + 4), "ascend");
  fmat y3 = sort(3 * (x.t() + 4), "ascend", 0);
  fmat y4 = sort(3 * (x.t() + 4), "ascend", 1);
  fmat y5 = sort(3 * (x.t() + 4), "descend");
  fmat y6 = sort(3 * (x.t() + 4), "descend", 0);
  fmat y7 = sort(3 * (x.t() + 4), "descend", 1);

  REQUIRE( y1.n_rows == x.n_cols );
  REQUIRE( y1.n_cols == x.n_rows );
  REQUIRE( y2.n_rows == x.n_cols );
  REQUIRE( y2.n_cols == x.n_rows );
  REQUIRE( y3.n_rows == x.n_cols );
  REQUIRE( y3.n_cols == x.n_rows );
  REQUIRE( y4.n_rows == x.n_cols );
  REQUIRE( y4.n_cols == x.n_rows );
  REQUIRE( y5.n_rows == x.n_cols );
  REQUIRE( y5.n_cols == x.n_rows );
  REQUIRE( y6.n_rows == x.n_cols );
  REQUIRE( y6.n_cols == x.n_rows );
  REQUIRE( y7.n_rows == x.n_cols );
  REQUIRE( y7.n_cols == x.n_rows );

  fmat y1_ref = sort(x_mod, "ascend", 0);
  fmat y2_ref = sort(x_mod, "ascend", 1);
  fmat y3_ref = sort(x_mod, "descend", 0);
  fmat y4_ref = sort(x_mod, "descend", 1);

  arma::fmat y1_cpu(y1);
  arma::fmat y2_cpu(y2);
  arma::fmat y3_cpu(y3);
  arma::fmat y4_cpu(y4);
  arma::fmat y5_cpu(y5);
  arma::fmat y6_cpu(y6);
  arma::fmat y7_cpu(y7);

  arma::fmat y1_ref_cpu(y1_ref);
  arma::fmat y2_ref_cpu(y2_ref);
  arma::fmat y3_ref_cpu(y3_ref);
  arma::fmat y4_ref_cpu(y4_ref);

  REQUIRE( arma::approx_equal( y1_cpu, y1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y2_cpu, y1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y3_cpu, y1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y4_cpu, y2_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y5_cpu, y3_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y6_cpu, y3_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y7_cpu, y4_ref_cpu, "reldiff", 1e-5 ) );
  }



// sort inside expression
TEST_CASE("sort_mat_inside_expr", "[sort]")
  {
  fmat x = randu<fmat>(133, 255);

  fmat x_sorted_col_asc = sort(x, "ascend", 0);
  fmat x_sorted_row_asc = sort(x, "ascend", 1);
  fmat x_sorted_col_desc = sort(x, "descend", 0);
  fmat x_sorted_row_desc = sort(x, "descend", 1);

  fmat y1 = repmat(sort(x), 2, 3);
  fmat y2 = repmat(sort(x, "ascend"), 2, 3);
  fmat y3 = repmat(sort(x, "ascend", 0), 2, 3);
  fmat y4 = repmat(sort(x, "ascend", 1), 2, 3);
  fmat y5 = repmat(sort(x, "descend"), 2, 3);
  fmat y6 = repmat(sort(x, "descend", 0), 2, 3);
  fmat y7 = repmat(sort(x, "descend", 1), 2, 3);

  fmat y1_ref = repmat(x_sorted_col_asc, 2, 3);
  fmat y2_ref = repmat(x_sorted_row_asc, 2, 3);
  fmat y3_ref = repmat(x_sorted_col_desc, 2, 3);
  fmat y4_ref = repmat(x_sorted_row_desc, 2, 3);

  REQUIRE( y1.n_rows == y1_ref.n_rows );
  REQUIRE( y1.n_cols == y1_ref.n_cols );
  REQUIRE( y2.n_rows == y1_ref.n_rows );
  REQUIRE( y2.n_cols == y1_ref.n_cols );
  REQUIRE( y3.n_rows == y1_ref.n_rows );
  REQUIRE( y3.n_cols == y1_ref.n_cols );
  REQUIRE( y4.n_rows == y2_ref.n_rows );
  REQUIRE( y4.n_cols == y2_ref.n_cols );
  REQUIRE( y5.n_rows == y3_ref.n_rows );
  REQUIRE( y5.n_cols == y3_ref.n_cols );
  REQUIRE( y6.n_rows == y3_ref.n_rows );
  REQUIRE( y6.n_cols == y3_ref.n_cols );
  REQUIRE( y7.n_rows == y4_ref.n_rows );
  REQUIRE( y7.n_cols == y4_ref.n_cols );

  arma::fmat y1_cpu(y1);
  arma::fmat y2_cpu(y2);
  arma::fmat y3_cpu(y3);
  arma::fmat y4_cpu(y4);
  arma::fmat y5_cpu(y5);
  arma::fmat y6_cpu(y6);
  arma::fmat y7_cpu(y7);

  arma::fmat y1_ref_cpu(y1_ref);
  arma::fmat y2_ref_cpu(y2_ref);
  arma::fmat y3_ref_cpu(y3_ref);
  arma::fmat y4_ref_cpu(y4_ref);

  REQUIRE( arma::approx_equal( y1_cpu, y1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y2_cpu, y1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y3_cpu, y1_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y4_cpu, y2_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y5_cpu, y3_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y6_cpu, y3_ref_cpu, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( y7_cpu, y4_ref_cpu, "reldiff", 1e-5 ) );
  }



// incorrect sort direction should throw exception
TEST_CASE("invalid_mat_sort_direction", "[sort]")
  {
  fmat x = randu<fmat>(10, 8);
  fmat y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = sort(x, "hello", 0) );
  REQUIRE_THROWS( y = sort(x, "", 1) );
  REQUIRE_THROWS( y = sort(x, "", 0) );
  REQUIRE_THROWS( y = sort(x, "things", 1) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// invalid sort dimension should throw exception
TEST_CASE("invalid_mat_sort_dim", "[sort]")
  {
  fmat x = randu<fmat>(10, 8);
  fmat y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = sort(x, "ascend", 2) );
  REQUIRE_THROWS( y = sort(x, "descend", 1000) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// test that sorting does not change the original matrix
TEST_CASE("sort_does_not_affect_original", "[sort]")
  {
  fvec x = randu<fvec>(1000);
  fvec x_old = x;

  fvec y = sort(x);

  arma::fvec x_cpu(x);
  arma::fvec x_old_cpu(x_old);

  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort(x, "ascend");

  x_cpu = arma::fvec(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort(x, "descend");

  x_cpu = arma::fvec(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );
  }



TEST_CASE("sort_mat_does_not_affect_original", "[sort]")
  {
  fmat x = randu<fmat>(1000);
  fmat x_old = x;

  fmat y = sort(x);

  arma::fmat x_cpu(x);
  arma::fmat x_old_cpu(x_old);

  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort(x, "ascend");

  x_cpu = arma::fmat(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort(x, "ascend", 0);

  x_cpu = arma::fmat(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort(x, "ascend", 1);

  x_cpu = arma::fmat(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort(x, "descend");

  x_cpu = arma::fmat(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort(x, "descend", 0);

  x_cpu = arma::fmat(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort(x, "descend", 1);

  x_cpu = arma::fmat(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );
  }
