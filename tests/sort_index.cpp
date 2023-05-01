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

TEMPLATE_TEST_CASE("simple_sort_index_test", "[sort_index]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

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

  uvec x1_asc = sort_index(x);
  uvec x2_asc = sort_index(x, "ascend");
  uvec x3_asc = stable_sort_index(x);
  uvec x4_asc = stable_sort_index(x, "ascend");
  uvec x1_desc = sort_index(x, "descend");
  uvec x2_desc = stable_sort_index(x, "descend");

  REQUIRE( x1_asc.n_elem == 10 );
  REQUIRE( x2_asc.n_elem == 10 );
  REQUIRE( x3_asc.n_elem == 10 );
  REQUIRE( x4_asc.n_elem == 10 );
  REQUIRE( x1_desc.n_elem == 10 );
  REQUIRE( x2_desc.n_elem == 10 );

  REQUIRE( uword(x1_asc[0]) == uword(4) );
  REQUIRE( uword(x1_asc[1]) == uword(2) );
  REQUIRE( uword(x1_asc[2]) == uword(7) );
  REQUIRE( uword(x1_asc[3]) == uword(6) );
  REQUIRE( uword(x1_asc[4]) == uword(9) );
  REQUIRE( uword(x1_asc[5]) == uword(1) );
  REQUIRE( uword(x1_asc[6]) == uword(0) );
  REQUIRE( uword(x1_asc[7]) == uword(3) );
  REQUIRE( uword(x1_asc[8]) == uword(8) );
  REQUIRE( uword(x1_asc[9]) == uword(5) );
  REQUIRE( uword(x2_asc[0]) == uword(4) );
  REQUIRE( uword(x2_asc[1]) == uword(2) );
  REQUIRE( uword(x2_asc[2]) == uword(7) );
  REQUIRE( uword(x2_asc[3]) == uword(6) );
  REQUIRE( uword(x2_asc[4]) == uword(9) );
  REQUIRE( uword(x2_asc[5]) == uword(1) );
  REQUIRE( uword(x2_asc[6]) == uword(0) );
  REQUIRE( uword(x2_asc[7]) == uword(3) );
  REQUIRE( uword(x2_asc[8]) == uword(8) );
  REQUIRE( uword(x2_asc[9]) == uword(5) );
  REQUIRE( uword(x3_asc[0]) == uword(4) );
  REQUIRE( uword(x3_asc[1]) == uword(2) );
  REQUIRE( uword(x3_asc[2]) == uword(7) );
  REQUIRE( uword(x3_asc[3]) == uword(6) );
  REQUIRE( uword(x3_asc[4]) == uword(9) );
  REQUIRE( uword(x3_asc[5]) == uword(1) );
  REQUIRE( uword(x3_asc[6]) == uword(0) );
  REQUIRE( uword(x3_asc[7]) == uword(3) );
  REQUIRE( uword(x3_asc[8]) == uword(8) );
  REQUIRE( uword(x3_asc[9]) == uword(5) );
  REQUIRE( uword(x4_asc[0]) == uword(4) );
  REQUIRE( uword(x4_asc[1]) == uword(2) );
  REQUIRE( uword(x4_asc[2]) == uword(7) );
  REQUIRE( uword(x4_asc[3]) == uword(6) );
  REQUIRE( uword(x4_asc[4]) == uword(9) );
  REQUIRE( uword(x4_asc[5]) == uword(1) );
  REQUIRE( uword(x4_asc[6]) == uword(0) );
  REQUIRE( uword(x4_asc[7]) == uword(3) );
  REQUIRE( uword(x4_asc[8]) == uword(8) );
  REQUIRE( uword(x4_asc[9]) == uword(5) );
  REQUIRE( uword(x1_desc[0]) == uword(5) );
  REQUIRE( uword(x1_desc[1]) == uword(8) );
  REQUIRE( uword(x1_desc[2]) == uword(3) );
  REQUIRE( uword(x1_desc[3]) == uword(0) );
  REQUIRE( uword(x1_desc[4]) == uword(1) );
  REQUIRE( uword(x1_desc[5]) == uword(9) );
  REQUIRE( uword(x1_desc[6]) == uword(6) );
  REQUIRE( uword(x1_desc[7]) == uword(7) );
  REQUIRE( uword(x1_desc[8]) == uword(2) );
  REQUIRE( uword(x1_desc[9]) == uword(4) );
  REQUIRE( uword(x2_desc[0]) == uword(5) );
  REQUIRE( uword(x2_desc[1]) == uword(8) );
  REQUIRE( uword(x2_desc[2]) == uword(3) );
  REQUIRE( uword(x2_desc[3]) == uword(0) );
  REQUIRE( uword(x2_desc[4]) == uword(1) );
  REQUIRE( uword(x2_desc[5]) == uword(9) );
  REQUIRE( uword(x2_desc[6]) == uword(6) );
  REQUIRE( uword(x2_desc[7]) == uword(7) );
  REQUIRE( uword(x2_desc[8]) == uword(2) );
  REQUIRE( uword(x2_desc[9]) == uword(4) );
  }



TEMPLATE_TEST_CASE("random_vector_float_data_index_test", "[sort_index]", float, double)
  {
  typedef TestType eT;

  Col<eT> x = (std::numeric_limits<eT>::max() / 2) * (2 * randu<Col<eT>>(50000) - 0.5);

  uvec x1_asc = sort_index(x);
  uvec x2_asc = sort_index(x, "ascend");
  uvec x3_asc = stable_sort_index(x);
  uvec x4_asc = stable_sort_index(x, "ascend");
  uvec x1_desc = sort_index(x, "descend");
  uvec x2_desc = stable_sort_index(x, "descend");

  REQUIRE( x1_asc.n_elem == 50000 );
  REQUIRE( x2_asc.n_elem == 50000 );
  REQUIRE( x3_asc.n_elem == 50000 );
  REQUIRE( x4_asc.n_elem == 50000 );
  REQUIRE( x1_desc.n_elem == 50000 );
  REQUIRE( x2_desc.n_elem == 50000 );

  arma::Col<uword> x1_asc_cpu(x1_asc);
  arma::Col<uword> x2_asc_cpu(x2_asc);
  arma::Col<uword> x3_asc_cpu(x3_asc);
  arma::Col<uword> x4_asc_cpu(x4_asc);
  arma::Col<uword> x1_desc_cpu(x1_desc);
  arma::Col<uword> x2_desc_cpu(x2_desc);

  arma::Col<eT> x_cpu(x);

  arma::Col<uword> x1_ref_asc_cpu = arma::conv_to<arma::Col<uword>>::from(arma::stable_sort_index(x_cpu));
  arma::Col<uword> x1_ref_desc_cpu = arma::conv_to<arma::Col<uword>>::from(arma::stable_sort_index(x_cpu, "descend"));

  // Since some vectors were created with an unstable sort, we just use them to sort the original data and make sure it is correct.
  arma::Col<eT> x_cpu_sorted_asc = x_cpu.rows(arma::conv_to<arma::uvec>::from(x1_ref_asc_cpu));
  arma::Col<eT> x_cpu_sorted_desc = x_cpu.rows(arma::conv_to<arma::uvec>::from(x1_ref_desc_cpu));

  arma::Col<eT> x1_asc_cpu_sorted = x_cpu.rows(arma::conv_to<arma::uvec>::from(x1_asc_cpu));
  arma::Col<eT> x2_asc_cpu_sorted = x_cpu.rows(arma::conv_to<arma::uvec>::from(x2_asc_cpu));
  arma::Col<eT> x1_desc_cpu_sorted = x_cpu.rows(arma::conv_to<arma::uvec>::from(x1_desc_cpu));

  REQUIRE( arma::approx_equal( x1_asc_cpu_sorted, x_cpu_sorted_asc, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x2_asc_cpu_sorted, x_cpu_sorted_asc, "reldiff", 1e-5 ) );
  REQUIRE( arma::all( x3_asc_cpu == x1_ref_asc_cpu ) );
  REQUIRE( arma::all( x4_asc_cpu == x1_ref_asc_cpu ) );
  REQUIRE( arma::approx_equal( x1_desc_cpu_sorted, x_cpu_sorted_desc, "reldiff", 1e-5 ) );
  REQUIRE( arma::all( x2_desc_cpu == x1_ref_desc_cpu ) );
  }



TEMPLATE_TEST_CASE("random_vector_integer_data_index_test", "[sort_index]", u32, s32, u64, s64)
  {
  typedef TestType eT;

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

  uvec x1_asc = sort_index(x);
  uvec x2_asc = sort_index(x, "ascend");
  uvec x3_asc = stable_sort_index(x);
  uvec x4_asc = stable_sort_index(x, "ascend");
  uvec x1_desc = sort_index(x, "descend");
  uvec x2_desc = stable_sort_index(x, "descend");

  REQUIRE( x1_asc.n_elem == 50000 );
  REQUIRE( x2_asc.n_elem == 50000 );
  REQUIRE( x3_asc.n_elem == 50000 );
  REQUIRE( x4_asc.n_elem == 50000 );
  REQUIRE( x1_desc.n_elem == 50000 );
  REQUIRE( x2_desc.n_elem == 50000 );

  arma::Col<uword> x1_asc_cpu(x1_asc);
  arma::Col<uword> x2_asc_cpu(x2_asc);
  arma::Col<uword> x3_asc_cpu(x3_asc);
  arma::Col<uword> x4_asc_cpu(x4_asc);
  arma::Col<uword> x1_desc_cpu(x1_desc);
  arma::Col<uword> x2_desc_cpu(x2_desc);

  arma::Col<eT> x_cpu(x);

  arma::Col<uword> x1_ref_asc_cpu = arma::conv_to<arma::Col<uword>>::from(arma::stable_sort_index(x_cpu));
  arma::Col<uword> x1_ref_desc_cpu = arma::conv_to<arma::Col<uword>>::from(arma::stable_sort_index(x_cpu, "descend"));

  // Since some vectors were created with an unstable sort, we just use them to sort the original data and make sure it is correct.
  arma::Col<eT> x_cpu_sorted_asc = x_cpu.rows(arma::conv_to<arma::uvec>::from(x1_ref_asc_cpu));
  arma::Col<eT> x_cpu_sorted_desc = x_cpu.rows(arma::conv_to<arma::uvec>::from(x1_ref_desc_cpu));

  arma::Col<eT> x1_asc_cpu_sorted = x_cpu.rows(arma::conv_to<arma::uvec>::from(x1_asc_cpu));
  arma::Col<eT> x2_asc_cpu_sorted = x_cpu.rows(arma::conv_to<arma::uvec>::from(x2_asc_cpu));
  arma::Col<eT> x1_desc_cpu_sorted = x_cpu.rows(arma::conv_to<arma::uvec>::from(x1_desc_cpu));

  REQUIRE( arma::approx_equal( x1_asc_cpu_sorted, x_cpu_sorted_asc, "reldiff", 1e-5 ) );
  REQUIRE( arma::approx_equal( x2_asc_cpu_sorted, x_cpu_sorted_asc, "reldiff", 1e-5 ) );
  REQUIRE( arma::all( x3_asc_cpu == x1_ref_asc_cpu ) );
  REQUIRE( arma::all( x4_asc_cpu == x1_ref_asc_cpu ) );
  REQUIRE( arma::approx_equal( x1_desc_cpu_sorted, x_cpu_sorted_desc, "reldiff", 1e-5 ) );
  REQUIRE( arma::all( x2_desc_cpu == x1_ref_desc_cpu ) );
  }



// sort data where every bit is the same (probably unnecessary)
TEMPLATE_TEST_CASE("identical_data_sort_index_test", "[sort_index]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> x(10000);
  x.zeros();

  uvec y1 = sort_index(x);
  uvec y2 = sort_index(x, "ascend");
  uvec y3 = stable_sort_index(x);
  uvec y4 = stable_sort_index(x, "ascend");
  uvec y5 = sort_index(x, "descend");
  uvec y6 = stable_sort_index(x, "descend");

  REQUIRE( y1.n_elem == 10000 );
  REQUIRE( y2.n_elem == 10000 );
  REQUIRE( y3.n_elem == 10000 );
  REQUIRE( y4.n_elem == 10000 );
  REQUIRE( y5.n_elem == 10000 );
  REQUIRE( y6.n_elem == 10000 );

  uvec lin = linspace<uvec>(0, 9999, 10000);

  REQUIRE( all( y1 == lin ) );
  REQUIRE( all( y2 == lin ) );
  REQUIRE( all( y3 == lin ) );
  REQUIRE( all( y4 == lin ) );
  REQUIRE( all( y5 == lin ) );
  REQUIRE( all( y6 == lin ) );
  }



// sort empty data
TEST_CASE("empty_data_sort_index", "[sort_index]")
  {
  vec x;

  uvec y1 = sort_index(x);
  uvec y2 = sort_index(x, "ascend");
  uvec y3 = stable_sort_index(x);
  uvec y4 = stable_sort_index(x, "ascend");
  uvec y5 = sort_index(x, "descend");
  uvec y6 = stable_sort_index(x, "descend");

  REQUIRE( y1.n_elem == 0 );
  REQUIRE( y2.n_elem == 0 );
  REQUIRE( y3.n_elem == 0 );
  REQUIRE( y4.n_elem == 0 );
  REQUIRE( y5.n_elem == 0 );
  REQUIRE( y6.n_elem == 0 );
  }



// sort one element data
TEST_CASE("one_elem_sort_index", "[sort_index]")
  {
  vec x(1);
  x(0) = 1.0;

  uvec y1 = sort_index(x);
  uvec y2 = sort_index(x, "ascend");
  uvec y3 = stable_sort_index(x, "ascend");
  uvec y4 = stable_sort_index(x, "ascend");
  uvec y5 = sort_index(x, "descend");
  uvec y6 = stable_sort_index(x, "descend");

  REQUIRE( y1.n_elem == 1 );
  REQUIRE( y2.n_elem == 1 );
  REQUIRE( y3.n_elem == 1 );
  REQUIRE( y4.n_elem == 1 );
  REQUIRE( y5.n_elem == 1 );
  REQUIRE( y6.n_elem == 1 );

  REQUIRE( uword(y1[0]) == uword(0) );
  REQUIRE( uword(y2[0]) == uword(0) );
  REQUIRE( uword(y3[0]) == uword(0) );
  REQUIRE( uword(y4[0]) == uword(0) );
  REQUIRE( uword(y5[0]) == uword(0) );
  REQUIRE( uword(y6[0]) == uword(0) );
  }



// sort an expression
TEST_CASE("sort_index_expr", "[sort_index]")
  {
  vec x = randu<vec>(1023);

  vec x_mod = 3 * (x + 4);

  uvec y1 = sort_index(3 * (x + 4));
  uvec y2 = sort_index(3 * (x + 4), "ascend");
  uvec y3 = stable_sort_index(3 * (x + 4));
  uvec y4 = stable_sort_index(3 * (x + 4), "ascend");
  uvec y5 = sort_index(3 * (x + 4), "descend");
  uvec y6 = stable_sort_index(3 * (x + 4), "descend");

  REQUIRE( y1.n_elem == 1023 );
  REQUIRE( y2.n_elem == 1023 );
  REQUIRE( y3.n_elem == 1023 );
  REQUIRE( y4.n_elem == 1023 );
  REQUIRE( y5.n_elem == 1023 );
  REQUIRE( y6.n_elem == 1023 );

  uvec y_asc_ref = sort_index(x_mod, "ascend");
  uvec y_desc_ref = sort_index(x_mod, "descend");
  uvec y_stable_asc_ref = stable_sort_index(x_mod, "ascend");
  uvec y_stable_desc_ref = stable_sort_index(x_mod, "descend");

  REQUIRE( all( y1 == y_asc_ref ) );
  REQUIRE( all( y2 == y_asc_ref ) );
  REQUIRE( all( y3 == y_stable_asc_ref ) );
  REQUIRE( all( y4 == y_stable_asc_ref ) );
  REQUIRE( all( y5 == y_desc_ref ) );
  REQUIRE( all( y6 == y_stable_desc_ref ) );
  }



// sort inside expression
TEST_CASE("sort_index_inside_expr", "[sort_index]")
  {
  vec x = randu<vec>(1025);

  uvec x_sorted_asc = sort_index(x, "ascend");
  uvec x_sorted_desc = sort_index(x, "descend");
  uvec x_stable_sorted_asc = stable_sort_index(x, "ascend");
  uvec x_stable_sorted_desc = stable_sort_index(x, "descend");

  umat y1 = repmat(sort_index(x), 2, 3);
  umat y2 = repmat(sort_index(x, "ascend"), 2, 3);
  umat y3 = repmat(stable_sort_index(x), 2, 3);
  umat y4 = repmat(stable_sort_index(x, "ascend"), 2, 3);
  umat y5 = repmat(sort_index(x, "descend"), 2, 3);
  umat y6 = repmat(stable_sort_index(x, "descend"), 2, 3);

  umat y_asc_ref = repmat(x_sorted_asc, 2, 3);
  umat y_desc_ref = repmat(x_sorted_desc, 2, 3);
  umat y_stable_asc_ref = repmat(x_stable_sorted_asc, 2, 3);
  umat y_stable_desc_ref = repmat(x_stable_sorted_desc, 2, 3);

  REQUIRE( y1.n_rows == y_asc_ref.n_rows );
  REQUIRE( y1.n_cols == y_asc_ref.n_cols );
  REQUIRE( y2.n_rows == y_asc_ref.n_rows );
  REQUIRE( y2.n_cols == y_asc_ref.n_cols );
  REQUIRE( y3.n_rows == y_asc_ref.n_rows );
  REQUIRE( y3.n_cols == y_asc_ref.n_cols );
  REQUIRE( y4.n_rows == y_asc_ref.n_rows );
  REQUIRE( y4.n_cols == y_asc_ref.n_cols );
  REQUIRE( y5.n_rows == y_desc_ref.n_rows );
  REQUIRE( y5.n_cols == y_desc_ref.n_cols );
  REQUIRE( y6.n_rows == y_desc_ref.n_rows );
  REQUIRE( y6.n_cols == y_desc_ref.n_cols );

  REQUIRE( all( all( y1 == y_asc_ref ) ) );
  REQUIRE( all( all( y2 == y_asc_ref ) ) );
  REQUIRE( all( all( y3 == y_stable_asc_ref ) ) );
  REQUIRE( all( all( y4 == y_stable_asc_ref ) ) );
  REQUIRE( all( all( y5 == y_desc_ref ) ) );
  REQUIRE( all( all( y6 == y_stable_desc_ref ) ) );
  }



// incorrect sort direction should throw exception
TEST_CASE("invalid_sort_index_direction", "[sort_index]")
  {
  vec x = randu<vec>(10);
  uvec y;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = sort_index(x, "hello") );
  REQUIRE_THROWS( y = sort_index(x, "") );
  REQUIRE_THROWS( y = sort_index(x, "things") );
  REQUIRE_THROWS( y = stable_sort_index(x, "hello") );
  REQUIRE_THROWS( y = stable_sort_index(x, "") );
  REQUIRE_THROWS( y = stable_sort_index(x, "things") );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



// test that sorting does not change the original matrix
TEST_CASE("sort_index_does_not_affect_original", "[sort]")
  {
  vec x = randu<vec>(1000);
  vec x_old = x;

  uvec y = sort_index(x);

  arma::vec x_cpu(x);
  arma::vec x_old_cpu(x_old);

  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort_index(x, "ascend");

  x_cpu = arma::vec(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = sort_index(x, "descend");

  x_cpu = arma::vec(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = stable_sort_index(x, "ascend");

  x_cpu = arma::vec(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );

  x = x_old;
  y = stable_sort_index(x, "descend");

  x_cpu = arma::vec(x);
  REQUIRE( arma::approx_equal( x_cpu, x_old_cpu, "reldiff", 1e-5 ) );
  }



TEMPLATE_TEST_CASE("duplicate_elements_signed_stable_sort_index_test", "[sort_index]", float, double, s32, s64)
  {
  typedef TestType eT;

  Row<eT> x = randi<Row<eT>>(1033, distr_param(-10000, 10000));

  x[10] = eT(-5);
  x[11] = eT(-5);

  x[73] = eT(10);
  x[113] = eT(10);

  x[6] = eT(0);
  x[987] = eT(0);

  // Now ensure we get the orderings right for a stable sort.
  uvec y1 = stable_sort_index(x);
  uvec y2 = stable_sort_index(x, "ascend");
  uvec y3 = stable_sort_index(x, "descend");

  arma::Col<uword> y1_cpu(y1);
  arma::Col<uword> y2_cpu(y2);
  arma::Col<uword> y3_cpu(y3);

  arma::uword i1 = arma::as_scalar(arma::find(y1_cpu == 10, 1));
  arma::uword i2 = arma::as_scalar(arma::find(y1_cpu == 11, 1));
  REQUIRE( i1 < i2 );

  i1 = arma::as_scalar(arma::find(y1_cpu == 73, 1));
  i2 = arma::as_scalar(arma::find(y1_cpu == 113, 1));
  REQUIRE( i1 < i2 );

  i1 = arma::as_scalar(arma::find(y1_cpu == 6, 1));
  i2 = arma::as_scalar(arma::find(y1_cpu == 987, 1));
  REQUIRE( i1 < i2 );

  i1 = arma::as_scalar(arma::find(y2_cpu == 10, 1));
  i2 = arma::as_scalar(arma::find(y2_cpu == 11, 1));
  REQUIRE( i1 < i2 );

  i1 = arma::as_scalar(arma::find(y2_cpu == 73, 1));
  i2 = arma::as_scalar(arma::find(y2_cpu == 113, 1));
  REQUIRE( i1 < i2 );

  i1 = arma::as_scalar(arma::find(y2_cpu == 6, 1));
  i2 = arma::as_scalar(arma::find(y2_cpu == 987, 1));
  REQUIRE( i1 < i2 );

  i1 = arma::as_scalar(arma::find(y3_cpu == 10, 1));
  i2 = arma::as_scalar(arma::find(y3_cpu == 11, 1));
  REQUIRE( i1 < i2 );

  i1 = arma::as_scalar(arma::find(y3_cpu == 73, 1));
  i2 = arma::as_scalar(arma::find(y3_cpu == 113, 1));
  REQUIRE( i1 < i2 );

  i1 = arma::as_scalar(arma::find(y3_cpu == 6, 1));
  i2 = arma::as_scalar(arma::find(y3_cpu == 987, 1));
  REQUIRE( i1 < i2 );
  }
