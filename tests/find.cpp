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

TEMPLATE_TEST_CASE("find_basic", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(100);
  x.zeros();
  x(53) = eT(1);

  uvec y1 = find(x);
  uvec y2 = find(x, 0);
  uvec y3 = find(x, 0, "first");
  uvec y4 = find(x, 0, "last");

  REQUIRE( y1.n_elem == 1 );
  REQUIRE( y2.n_elem == 1 );
  REQUIRE( y3.n_elem == 1 );
  REQUIRE( y4.n_elem == 1 );
  REQUIRE( uword(y1[0]) == uword(53) );
  REQUIRE( uword(y2[0]) == uword(53) );
  REQUIRE( uword(y3[0]) == uword(53) );
  REQUIRE( uword(y4[0]) == uword(53) );
  }



TEMPLATE_TEST_CASE("find_basic_2", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(100);
  x.zeros();
  x(53) = eT(1);
  x(77) = eT(1);

  uvec y1 = find(x);
  uvec y2 = find(x, 0);
  uvec y3 = find(x, 0, "first");
  uvec y4 = find(x, 0, "last");

  REQUIRE( y1.n_elem == 2 );
  REQUIRE( y2.n_elem == 2 );
  REQUIRE( y3.n_elem == 2 );
  REQUIRE( y4.n_elem == 2 );
  REQUIRE( uword(y1[0]) == uword(53) );
  REQUIRE( uword(y1[1]) == uword(77) );
  REQUIRE( uword(y2[0]) == uword(53) );
  REQUIRE( uword(y2[1]) == uword(77) );
  REQUIRE( uword(y3[0]) == uword(53) );
  REQUIRE( uword(y3[1]) == uword(77) );
  REQUIRE( uword(y4[0]) == uword(53) );
  REQUIRE( uword(y4[1]) == uword(77) );
  }



// find with 10 nonzeros
TEMPLATE_TEST_CASE("find_10_nonzeros", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(1000);
  x.zeros();
  x(0) = eT(1);
  x(11) = eT(1);
  x(93) = eT(1);
  x(110) = eT(1);
  x(112) = eT(1);
  x(167) = eT(1);
  x(498) = eT(1);
  x(499) = eT(1);
  x(500) = eT(1);
  x(811) = eT(1);

  uvec y1 = find(x);
  uvec y2 = find(x, 0);
  uvec y3 = find(x, 0, "first");
  uvec y4 = find(x, 0, "last");

  REQUIRE( y1.n_elem == 10 );
  REQUIRE( uword(y1[0]) == uword(0) );
  REQUIRE( uword(y1[1]) == uword(11) );
  REQUIRE( uword(y1[2]) == uword(93) );
  REQUIRE( uword(y1[3]) == uword(110) );
  REQUIRE( uword(y1[4]) == uword(112) );
  REQUIRE( uword(y1[5]) == uword(167) );
  REQUIRE( uword(y1[6]) == uword(498) );
  REQUIRE( uword(y1[7]) == uword(499) );
  REQUIRE( uword(y1[8]) == uword(500) );
  REQUIRE( uword(y1[9]) == uword(811) );

  REQUIRE( y2.n_elem == 10 );
  REQUIRE( uword(y2[0]) == uword(0) );
  REQUIRE( uword(y2[1]) == uword(11) );
  REQUIRE( uword(y2[2]) == uword(93) );
  REQUIRE( uword(y2[3]) == uword(110) );
  REQUIRE( uword(y2[4]) == uword(112) );
  REQUIRE( uword(y2[5]) == uword(167) );
  REQUIRE( uword(y2[6]) == uword(498) );
  REQUIRE( uword(y2[7]) == uword(499) );
  REQUIRE( uword(y2[8]) == uword(500) );
  REQUIRE( uword(y2[9]) == uword(811) );

  REQUIRE( y3.n_elem == 10 );
  REQUIRE( uword(y3[0]) == uword(0) );
  REQUIRE( uword(y3[1]) == uword(11) );
  REQUIRE( uword(y3[2]) == uword(93) );
  REQUIRE( uword(y3[3]) == uword(110) );
  REQUIRE( uword(y3[4]) == uword(112) );
  REQUIRE( uword(y3[5]) == uword(167) );
  REQUIRE( uword(y3[6]) == uword(498) );
  REQUIRE( uword(y3[7]) == uword(499) );
  REQUIRE( uword(y3[8]) == uword(500) );
  REQUIRE( uword(y3[9]) == uword(811) );

  REQUIRE( y4.n_elem == 10 );
  REQUIRE( uword(y4[0]) == uword(0) );
  REQUIRE( uword(y4[1]) == uword(11) );
  REQUIRE( uword(y4[2]) == uword(93) );
  REQUIRE( uword(y4[3]) == uword(110) );
  REQUIRE( uword(y4[4]) == uword(112) );
  REQUIRE( uword(y4[5]) == uword(167) );
  REQUIRE( uword(y4[6]) == uword(498) );
  REQUIRE( uword(y4[7]) == uword(499) );
  REQUIRE( uword(y4[8]) == uword(500) );
  REQUIRE( uword(y4[9]) == uword(811) );
  }



// find top 5 of 10 nonzeros
TEMPLATE_TEST_CASE("top_5_nonzeros", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(1000);
  x.zeros();
  x(0) = eT(1);
  x(11) = eT(1);
  x(93) = eT(1);
  x(110) = eT(1);
  x(112) = eT(1);
  x(167) = eT(1);
  x(498) = eT(1);
  x(499) = eT(1);
  x(500) = eT(1);
  x(811) = eT(1);

  uvec y1 = find(x, 5);
  uvec y2 = find(x, 5, "first");

  REQUIRE( y1.n_elem == 5 );
  REQUIRE( uword(y1[0]) == uword(0) );
  REQUIRE( uword(y1[1]) == uword(11) );
  REQUIRE( uword(y1[2]) == uword(93) );
  REQUIRE( uword(y1[3]) == uword(110) );
  REQUIRE( uword(y1[4]) == uword(112) );

  REQUIRE( y2.n_elem == 5 );
  REQUIRE( uword(y2[0]) == uword(0) );
  REQUIRE( uword(y2[1]) == uword(11) );
  REQUIRE( uword(y2[2]) == uword(93) );
  REQUIRE( uword(y2[3]) == uword(110) );
  REQUIRE( uword(y2[4]) == uword(112) );
  }



// find bottom 5 of 10 nonzeros
TEMPLATE_TEST_CASE("bottom_5_nonzeros", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(1000);
  x.zeros();
  x(0) = eT(1);
  x(11) = eT(1);
  x(93) = eT(1);
  x(110) = eT(1);
  x(112) = eT(1);
  x(167) = eT(1);
  x(498) = eT(1);
  x(499) = eT(1);
  x(500) = eT(1);
  x(811) = eT(1);

  uvec y = find(x, 5, "last");

  REQUIRE( y.n_elem == 5 );
  REQUIRE( uword(y[0]) == uword(167) );
  REQUIRE( uword(y[1]) == uword(498) );
  REQUIRE( uword(y[2]) == uword(499) );
  REQUIRE( uword(y[3]) == uword(500) );
  REQUIRE( uword(y[4]) == uword(811) );
  }



// find where everything is nonzero
TEMPLATE_TEST_CASE("find_all_nonzero", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(150, distr_param(10, 1000));

  uvec y1 = find(x);
  uvec y2 = find(x, 0);
  uvec y3 = find(x, 0, "first");
  uvec y4 = find(x, 0, "last");

  REQUIRE( y1.n_elem == x.n_elem );
  REQUIRE( y2.n_elem == x.n_elem );
  REQUIRE( y3.n_elem == x.n_elem );
  REQUIRE( y4.n_elem == x.n_elem );

  uvec lin = linspace<uvec>(0, x.n_elem - 1, x.n_elem);

  REQUIRE( all( y1 == lin ) );
  REQUIRE( all( y2 == lin ) );
  REQUIRE( all( y3 == lin ) );
  REQUIRE( all( y4 == lin ) );
  }



// find where everything except one is nonzero
TEMPLATE_TEST_CASE("find_all_except_one_nonzero", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(150, distr_param(10, 1000));
  x(37) = eT(0);

  uvec y1 = find(x);
  uvec y2 = find(x, 0);
  uvec y3 = find(x, 0, "first");
  uvec y4 = find(x, 0, "last");

  REQUIRE( y1.n_elem == x.n_elem - 1 );
  REQUIRE( y2.n_elem == x.n_elem - 1 );
  REQUIRE( y3.n_elem == x.n_elem - 1 );
  REQUIRE( y4.n_elem == x.n_elem - 1 );

  uvec lin1 = linspace<uvec>(0, x.n_elem - 1, x.n_elem);
  uvec lin2(lin1.n_elem - 1);
  lin2.subvec(0, 36) = lin1.subvec(0, 36);
  lin2.subvec(37, lin2.n_elem - 1) = lin1.subvec(38, lin1.n_elem - 1);

  REQUIRE( all( y1 == lin2 ) );
  REQUIRE( all( y2 == lin2 ) );
  REQUIRE( all( y3 == lin2 ) );
  REQUIRE( all( y4 == lin2 ) );
  }



// find in empty vector
TEST_CASE("find_empty_vector", "[find]")
  {
  fvec x;

  uvec y1 = find(x);
  uvec y2 = find(x, 0);
  uvec y3 = find(x, 0, "first");
  uvec y4 = find(x, 0, "last");

  REQUIRE( y1.n_elem == 0 );
  REQUIRE( y2.n_elem == 0 );
  REQUIRE( y3.n_elem == 0 );
  REQUIRE( y4.n_elem == 0 );
  }



// find in matrix, check column major
TEST_CASE("col_major_find", "[find]")
  {
  fmat x(10, 12);
  x.zeros();

  x(5, 6) = 1;
  x(3, 1) = 1;

  uvec y1 = find(x);
  uvec y2 = find(x, 0);
  uvec y3 = find(x, 0, "first");
  uvec y4 = find(x, 0, "last");
  uvec y5 = find(x, 2);
  uvec y6 = find(x, 2, "first");
  uvec y7 = find(x, 2, "last");

  REQUIRE( y1.n_elem == 2 );
  REQUIRE( y2.n_elem == 2 );
  REQUIRE( y3.n_elem == 2 );
  REQUIRE( y4.n_elem == 2 );
  REQUIRE( y5.n_elem == 2 );
  REQUIRE( y6.n_elem == 2 );
  REQUIRE( y7.n_elem == 2 );

  REQUIRE( uword(y1[0]) == 13 );
  REQUIRE( uword(y1[1]) == 65 );
  REQUIRE( uword(y2[0]) == 13 );
  REQUIRE( uword(y2[1]) == 65 );
  REQUIRE( uword(y3[0]) == 13 );
  REQUIRE( uword(y3[1]) == 65 );
  REQUIRE( uword(y4[0]) == 13 );
  REQUIRE( uword(y4[1]) == 65 );
  REQUIRE( uword(y5[0]) == 13 );
  REQUIRE( uword(y5[1]) == 65 );
  REQUIRE( uword(y6[0]) == 13 );
  REQUIRE( uword(y6[1]) == 65 );
  REQUIRE( uword(y7[0]) == 13 );
  REQUIRE( uword(y7[1]) == 65 );
  }



// find with k > number of nonzeros
TEMPLATE_TEST_CASE("find_k_greater_than_nonzeros", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(5000);
  x.zeros();

  for (uword i = 0; i < 500; ++i)
    {
    x[10 * i] = eT(1);
    }

  uvec y1 = find(x, 600);
  uvec y2 = find(x, 600, "first");
  uvec y3 = find(x, 600, "last");

  REQUIRE( y1.n_elem == 500 );
  REQUIRE( y2.n_elem == 500 );
  REQUIRE( y3.n_elem == 500 );

  uvec lin = 10 * linspace<uvec>(0, 499, 500);

  REQUIRE( all( y1 == lin ) );
  REQUIRE( all( y2 == lin ) );
  REQUIRE( all( y3 == lin ) );
  }



// find relational operators
TEMPLATE_TEST_CASE("find_relational_operators", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = linspace<Col<eT>>(0, 9, 10);

  uvec y1 = find(x);
  uvec y2 = find(x != 0);

  REQUIRE( y2.n_elem == y1.n_elem );
  REQUIRE( all( y2 == y1 ) );

  uvec y3 = find(x == 0);

  REQUIRE( y3.n_elem == 1 );

  uvec y4 = find(x > 5);

  REQUIRE( y4.n_elem == 4 );
  REQUIRE( all( y4 > 5 ) );

  uvec y5 = find(x < 5);

  REQUIRE( y5.n_elem == 5 );
  REQUIRE( all( y5 < 5 ) );
  }



// find inside expression
TEMPLATE_TEST_CASE("find_inside_expression", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(1000, distr_param(0, 2));

  uvec y_ref = find(x);

  umat y1 = repmat(find(x), 2, 3);
  umat y2 = repmat(find(x, 0), 2, 3);
  umat y3 = repmat(find(x, 0, "first"), 2, 3);
  umat y4 = repmat(find(x, 0, "last"), 2, 3);

  REQUIRE( y1.n_rows == 2 * y_ref.n_rows );
  REQUIRE( y1.n_cols == 3 * y_ref.n_cols );
  REQUIRE( y2.n_rows == 2 * y_ref.n_rows );
  REQUIRE( y2.n_cols == 3 * y_ref.n_cols );
  REQUIRE( y3.n_rows == 2 * y_ref.n_rows );
  REQUIRE( y3.n_cols == 3 * y_ref.n_cols );
  REQUIRE( y4.n_rows == 2 * y_ref.n_rows );
  REQUIRE( y4.n_cols == 3 * y_ref.n_cols );

  umat y_rep_ref = repmat(y_ref, 2, 3);

  REQUIRE( all( all( y1 == y_rep_ref ) ) );
  REQUIRE( all( all( y2 == y_rep_ref ) ) );
  REQUIRE( all( all( y3 == y_rep_ref ) ) );
  REQUIRE( all( all( y4 == y_rep_ref ) ) );
  }



// find in find
TEMPLATE_TEST_CASE("find_inside_find", "[find]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(1000, distr_param(0, 2));

  uvec z_ref = find(x, 10, "first");

  uvec y1 = find(find(x, 10, "first"));
  uvec y2 = find(find(x, 10, "first"), 0);
  uvec y3 = find(find(x, 10, "first"), 0, "first");
  uvec y4 = find(find(x, 10, "first"), 0, "last");

  uvec y_ref = find(z_ref);

  REQUIRE( y1.n_elem == y_ref.n_elem );
  REQUIRE( y2.n_elem == y_ref.n_elem );
  REQUIRE( y3.n_elem == y_ref.n_elem );
  REQUIRE( y4.n_elem == y_ref.n_elem );

  REQUIRE( all( y1 == y_ref ) );
  REQUIRE( all( y2 == y_ref ) );
  REQUIRE( all( y3 == y_ref ) );
  REQUIRE( all( y4 == y_ref ) );
  }



// invalid direction for find
TEST_CASE("find_invalid_direction", "[find]")
  {
  fvec x;
  uvec y;

  // Suppress error output.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  REQUIRE_THROWS( y = find(x, 0, "hello") );
  REQUIRE_THROWS( y = find(x, 0, "goodbye") );
  REQUIRE_THROWS( y = find(x, 0, "1") );
  REQUIRE_THROWS( y = find(x, 0, "") );

  std::cerr.rdbuf(orig_cerr_buf);
  }



// alias test
TEST_CASE("find_alias", "[find]")
  {
  uvec x = randi<uvec>(1000, distr_param(0, 2));

  uvec x_ref = x;

  x = find(x);
  uvec x2 = find(x_ref);

  REQUIRE( x.n_elem == x2.n_elem );
  REQUIRE( all( x == x2 ) );
  }



// test that computing the size works
TEST_CASE("find_size_computation", "[find]")
  {
  uvec x = randi<uvec>(1000, distr_param(1, 5));

  REQUIRE( size(find(x)).n_rows == 1000 );
  }
