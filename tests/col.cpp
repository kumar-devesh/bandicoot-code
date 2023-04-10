// Copyright 2020 Ryan Curtin (http://www.ratml.org/)
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

TEMPLATE_TEST_CASE("create_col_1", "[col]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x(10, 10);
  for (uword i = 0; i < x.n_elem; ++i)
    {
    x[i] = eT(i);
    }

  for (uword i = 0; i < 10; ++i)
    {
    Col<eT> c = Col<eT>(x.col(i));

    REQUIRE(c.n_rows == x.n_rows);
    REQUIRE(c.n_cols == 1);

    for (uword j = 0; j < 10; ++j)
      {
      REQUIRE((eT) c[j] == Approx(eT(x(j, i))) );
      }
    }
  }



TEMPLATE_TEST_CASE("create_col_2", "[col]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Mat<eT> x(10, 10);
  for (uword i = 0; i < x.n_elem; ++i)
    {
    x[i] = eT(i);
    }

  for (uword i = 0; i < 10; ++i)
    {
    Col<eT> c(x.col(i));

    REQUIRE(c.n_rows == x.n_rows);
    REQUIRE(c.n_cols == 1);

    for (uword j = 0; j < 10; ++j)
      {
      REQUIRE((eT) c[j] == Approx(eT(x(j, i))) );
      }
    }
  }



TEMPLATE_TEST_CASE("empty_col_constructors", "[col]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;
  Col<eT> c1;
  Col<eT> c2(100);
  Col<eT> c3(100, 1);

  REQUIRE(c1.n_rows == 0);
  REQUIRE(c1.n_cols == 0);
  REQUIRE(c2.n_rows == 100);
  REQUIRE(c2.n_cols == 1);
  REQUIRE(c3.n_rows == 100);
  REQUIRE(c3.n_cols == 1);
  }



TEMPLATE_TEST_CASE("col_move_constructor_and_operator", "[col]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> c1(100);
  c1.fill(eT(2));
  Col<eT> c2(std::move(c1));

  REQUIRE(c1.n_rows == 0);
  REQUIRE(c1.n_cols == 0);
  REQUIRE(c2.n_rows == 100);
  REQUIRE(c2.n_cols == 1);
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (eT) c2[i] == Approx(eT(2)) );
    }

  c1.set_size(50);
  c1.fill(eT(3));
  c2 = std::move(c1);

  REQUIRE(c1.n_rows == 0);
  REQUIRE(c1.n_cols == 0);
  REQUIRE(c2.n_rows == 50);
  REQUIRE(c2.n_cols == 1);
  for (uword i = 0; i < 50; ++i)
    {
    REQUIRE( (eT) c2[i] == Approx(eT(3)) );
    }
  }



TEMPLATE_TEST_CASE("col_arma_conversion", "[col]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> c1(100);
  for (uword i = 0; i < c1.n_elem; ++i)
    {
    c1[i] = eT(i);
    }

  arma::Col<eT> c2(c1);

  REQUIRE(c2.n_rows == c1.n_rows);
  REQUIRE(c2.n_cols == c1.n_cols);
  for (uword i = 0; i < c1.n_elem; ++i)
    {
    REQUIRE(eT(c1[i]) == Approx(eT(c2[i])));
    }
  }



TEMPLATE_TEST_CASE("col_rows_1", "[col]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> c(100);
  for (uword i = 0; i < c.n_elem; ++i)
    {
    c[i] = eT(i);
    }

  Col<eT> c_sub = c.rows(5, 15);

  REQUIRE(c_sub.n_rows == 11);
  REQUIRE(c_sub.n_cols == 1);
  for (uword i = 0; i < c_sub.n_elem; ++i)
    {
    REQUIRE( eT(c_sub[i]) == Approx(eT(c[i + 5])) );
    }
  }



TEMPLATE_TEST_CASE("col_rows_2", "[col]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> c(100);
  for (uword i = 0; i < c.n_elem; ++i)
    {
    c[i] = eT(i);
    }
  Col<eT> c_orig(c);

  c.rows(5, 15) *= eT(2);

  for (uword i = 0; i < 5; ++i)
    {
    REQUIRE( eT(c[i]) == Approx(eT(c_orig[i])) );
    }

  for (uword i = 5; i < 16; ++i)
    {
    REQUIRE( eT(c[i]) == Approx(eT(2 * c_orig[i])) );
    }

  for (uword i = 16; i < 100; ++i)
    {
    REQUIRE( eT(c[i]) == Approx(eT(c_orig[i])) );
    }
  }



TEMPLATE_TEST_CASE("col_subvec_1", "[col]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> c(100);
  for (uword i = 0; i < c.n_elem; ++i)
    {
    c[i] = eT(i);
    }

  Col<eT> c_sub = c.subvec(5, 15);

  REQUIRE(c_sub.n_rows == 11);
  REQUIRE(c_sub.n_cols == 1);
  for (uword i = 0; i < c_sub.n_elem; ++i)
    {
    REQUIRE( eT(c_sub[i]) == Approx(eT(c[i + 5])) );
    }
  }



TEMPLATE_TEST_CASE("col_subvec_2", "[col]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  Col<eT> c(100);
  for (uword i = 0; i < c.n_elem; ++i)
    {
    c[i] = eT(i);
    }
  Col<eT> c_orig(c);

  c.subvec(5, 15) *= eT(2);

  for (uword i = 0; i < 5; ++i)
    {
    REQUIRE( eT(c[i]) == Approx(eT(c_orig[i])) );
    }

  for (uword i = 5; i < 16; ++i)
    {
    REQUIRE( eT(c[i]) == Approx(eT(2 * c_orig[i])) );
    }

  for (uword i = 16; i < 100; ++i)
    {
    REQUIRE( eT(c[i]) == Approx(eT(c_orig[i])) );
    }
  }



TEST_CASE("col_invalid_size", "[col]")
  {
  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  vec x;
  REQUIRE_THROWS( x = randi<rowvec>(100, distr_param(0, 10)) );
  REQUIRE_THROWS( x.set_size(1, 2) );
  REQUIRE_THROWS( x = rowvec(5) );
  REQUIRE_THROWS( x = mat(10, 5) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);

  // This one needs to not throw.
  x = mat(10, 1);
  }
