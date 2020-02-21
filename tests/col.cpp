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

template<typename eT>
void test_create_col_1()
  {
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



TEST_CASE("create_col_1")
  {
  test_create_col_1<double>();
  test_create_col_1<float>();
  test_create_col_1<u32>();
  test_create_col_1<s32>();
  test_create_col_1<u64>();
  test_create_col_1<s64>();
  }



template<typename eT>
void test_create_col_2()
  {
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



TEST_CASE("create_col_2")
  {
  test_create_col_2<double>();
  test_create_col_2<float>();
  test_create_col_2<u32>();
  test_create_col_2<s32>();
  test_create_col_2<u64>();
  test_create_col_2<s64>();
  }



template<typename eT>
void test_empty_col_constructors()
  {
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



TEST_CASE("empty_col_constructors")
  {
  test_empty_col_constructors<double>();
  test_empty_col_constructors<float>();
  test_empty_col_constructors<u32>();
  test_empty_col_constructors<s32>();
  test_empty_col_constructors<u64>();
  test_empty_col_constructors<s64>();
  }



template<typename eT>
void test_move_constructor_and_operator()
  {
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



TEST_CASE("col_move_constructor_and_operator")
  {
  test_move_constructor_and_operator<double>();
  test_move_constructor_and_operator<float>();
  test_move_constructor_and_operator<u32>();
  test_move_constructor_and_operator<s32>();
  test_move_constructor_and_operator<u64>();
  test_move_constructor_and_operator<s64>();
  }



template<typename eT>
void test_col_arma_conversion()
  {
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



TEST_CASE("col_arma_conversion")
  {
  test_col_arma_conversion<double>();
  test_col_arma_conversion<float>();
  test_col_arma_conversion<u32>();
  test_col_arma_conversion<s32>();
  test_col_arma_conversion<u64>();
  test_col_arma_conversion<s64>();
  }



template<typename eT>
void test_col_rows_1()
  {
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



TEST_CASE("col_rows_1")
  {
  test_col_rows_1<double>();
  test_col_rows_1<float>();
  test_col_rows_1<u32>();
  test_col_rows_1<s32>();
  test_col_rows_1<u64>();
  test_col_rows_1<s64>();
  }



template<typename eT>
void test_col_rows_2()
  {
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



TEST_CASE("col_rows_2")
  {
  test_col_rows_2<double>();
  test_col_rows_2<float>();
  test_col_rows_2<u32>();
  test_col_rows_2<s32>();
  test_col_rows_2<u64>();
  test_col_rows_2<s64>();
  }



template<typename eT>
void test_col_subvec_1()
  {
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



TEST_CASE("col_subvec_1")
  {
  test_col_subvec_1<double>();
  test_col_subvec_1<float>();
  test_col_subvec_1<u32>();
  test_col_subvec_1<s32>();
  test_col_subvec_1<u64>();
  test_col_subvec_1<s64>();
  }



template<typename eT>
void test_col_subvec_2()
  {
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



TEST_CASE("col_subvec_2")
  {
  test_col_subvec_2<double>();
  test_col_subvec_2<float>();
  test_col_subvec_2<u32>();
  test_col_subvec_2<s32>();
  test_col_subvec_2<u64>();
  test_col_subvec_2<s64>();
  }
