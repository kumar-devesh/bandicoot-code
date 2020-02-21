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
void test_create_row_1()
  {
  Mat<eT> x(10, 10);
  for (uword i = 0; i < x.n_elem; ++i)
    {
    x[i] = eT(i);
    }

  for (uword i = 0; i < 10; ++i)
    {
    Row<eT> r = Row<eT>(x.row(i));

    REQUIRE(r.n_rows == 1);
    REQUIRE(r.n_cols == x.n_cols);

    for (uword j = 0; j < 10; ++j)
      {
      REQUIRE((eT) r[j] == Approx(eT(x(i, j))) );
      }
    }
  }



TEST_CASE("create_row_1")
  {
  test_create_row_1<double>();
  test_create_row_1<float>();
  test_create_row_1<u32>();
  test_create_row_1<s32>();
  test_create_row_1<u64>();
  test_create_row_1<s64>();
  }



template<typename eT>
void test_create_row_2()
  {
  Mat<eT> x(10, 10);
  for (uword i = 0; i < x.n_elem; ++i)
    {
    x[i] = eT(i);
    }

  for (uword i = 0; i < 10; ++i)
    {
    Row<eT> r(x.row(i));

    REQUIRE(r.n_rows == 1);
    REQUIRE(r.n_cols == x.n_cols);

    for (uword j = 0; j < 10; ++j)
      {
      REQUIRE((eT) r[j] == Approx(eT(x(i, j))) );
      }
    }
  }



TEST_CASE("create_row_2")
  {
  test_create_row_2<double>();
  test_create_row_2<float>();
  test_create_row_2<u32>();
  test_create_row_2<s32>();
  test_create_row_2<u64>();
  test_create_row_2<s64>();
  }



template<typename eT>
void test_empty_row_constructors()
  {
  Row<eT> r1;
  Row<eT> r2(100);
  Row<eT> r3(1, 100);

  REQUIRE(r1.n_rows == 0);
  REQUIRE(r1.n_cols == 0);
  REQUIRE(r2.n_rows == 1);
  REQUIRE(r2.n_cols == 100);
  REQUIRE(r3.n_rows == 1);
  REQUIRE(r3.n_cols == 100);
  }



TEST_CASE("empty_row_constructors")
  {
  test_empty_row_constructors<double>();
  test_empty_row_constructors<float>();
  test_empty_row_constructors<u32>();
  test_empty_row_constructors<s32>();
  test_empty_row_constructors<u64>();
  test_empty_row_constructors<s64>();
  }



template<typename eT>
void test_move_constructor_and_operator()
  {
  Row<eT> r1(100);
  r1.fill(eT(2));
  Row<eT> r2(std::move(r1));

  REQUIRE(r1.n_rows == 0);
  REQUIRE(r1.n_cols == 0);
  REQUIRE(r2.n_rows == 1);
  REQUIRE(r2.n_cols == 100);
  for (uword i = 0; i < 100; ++i)
    {
    REQUIRE( (eT) r2[i] == Approx(eT(2)) );
    }

  r1.set_size(50);
  r1.fill(eT(3));
  r2 = std::move(r1);

  REQUIRE(r1.n_rows == 0);
  REQUIRE(r1.n_cols == 0);
  REQUIRE(r2.n_rows == 1);
  REQUIRE(r2.n_cols == 50);
  for (uword i = 0; i < 50; ++i)
    {
    REQUIRE( (eT) r2[i] == Approx(eT(3)) );
    }
  }



TEST_CASE("row_move_constructor_and_operator")
  {
  test_move_constructor_and_operator<double>();
  test_move_constructor_and_operator<float>();
  test_move_constructor_and_operator<u32>();
  test_move_constructor_and_operator<s32>();
  test_move_constructor_and_operator<u64>();
  test_move_constructor_and_operator<s64>();
  }



template<typename eT>
void test_row_arma_conversion()
  {
  Row<eT> r1(100);
  for (uword i = 0; i < r1.n_elem; ++i)
    {
    r1[i] = eT(i);
    }

  arma::Row<eT> r2(r1);

  REQUIRE(r2.n_rows == r1.n_rows);
  REQUIRE(r2.n_cols == r1.n_cols);
  for (uword i = 0; i < r1.n_elem; ++i)
    {
    REQUIRE(eT(r1[i]) == Approx(eT(r2[i])));
    }
  }



TEST_CASE("row_arma_conversion")
  {
  test_row_arma_conversion<double>();
  test_row_arma_conversion<float>();
  test_row_arma_conversion<u32>();
  test_row_arma_conversion<s32>();
  test_row_arma_conversion<u64>();
  test_row_arma_conversion<s64>();
  }



template<typename eT>
void test_row_cols_1()
  {
  Row<eT> r(100);
  for (uword i = 0; i < r.n_elem; ++i)
    {
    r[i] = eT(i);
    }

  Row<eT> r_sub = r.cols(5, 15);

  REQUIRE(r_sub.n_rows == 1);
  REQUIRE(r_sub.n_cols == 11);
  for (uword i = 0; i < r_sub.n_elem; ++i)
    {
    REQUIRE( eT(r_sub[i]) == Approx(eT(r[i + 5])) );
    }
  }



TEST_CASE("row_cols_1")
  {
  test_row_cols_1<double>();
  test_row_cols_1<float>();
  test_row_cols_1<u32>();
  test_row_cols_1<s32>();
  test_row_cols_1<u64>();
  test_row_cols_1<s64>();
  }



template<typename eT>
void test_row_cols_2()
  {
  Row<eT> r(100);
  for (uword i = 0; i < r.n_elem; ++i)
    {
    r[i] = eT(i);
    }
  Row<eT> r_orig(r);

  r.cols(5, 15) *= eT(2);

  for (uword i = 0; i < 5; ++i)
    {
    REQUIRE( eT(r[i]) == Approx(eT(r_orig[i])) );
    }

  for (uword i = 5; i < 16; ++i)
    {
    REQUIRE( eT(r[i]) == Approx(eT(2 * r_orig[i])) );
    }

  for (uword i = 16; i < 100; ++i)
    {
    REQUIRE( eT(r[i]) == Approx(eT(r_orig[i])) );
    }
  }



TEST_CASE("row_cols_2")
  {
  test_row_cols_2<double>();
  test_row_cols_2<float>();
  test_row_cols_2<u32>();
  test_row_cols_2<s32>();
  test_row_cols_2<u64>();
  test_row_cols_2<s64>();
  }



template<typename eT>
void test_row_subvec_1()
  {
  Row<eT> r(100);
  for (uword i = 0; i < r.n_elem; ++i)
    {
    r[i] = eT(i);
    }

  Row<eT> r_sub = r.subvec(5, 15);

  REQUIRE(r_sub.n_rows == 1);
  REQUIRE(r_sub.n_cols == 11);
  for (uword i = 0; i < r_sub.n_elem; ++i)
    {
    REQUIRE( eT(r_sub[i]) == Approx(eT(r[i + 5])) );
    }
  }



TEST_CASE("row_subvec_1")
  {
  test_row_subvec_1<double>();
  test_row_subvec_1<float>();
  test_row_subvec_1<u32>();
  test_row_subvec_1<s32>();
  test_row_subvec_1<u64>();
  test_row_subvec_1<s64>();
  }



template<typename eT>
void test_row_subvec_2()
  {
  Row<eT> r(100);
  for (uword i = 0; i < r.n_elem; ++i)
    {
    r[i] = eT(i);
    }
  Row<eT> r_orig(r);

  r.subvec(5, 15) *= eT(2);

  for (uword i = 0; i < 5; ++i)
    {
    REQUIRE( eT(r[i]) == Approx(eT(r_orig[i])) );
    }

  for (uword i = 5; i < 16; ++i)
    {
    REQUIRE( eT(r[i]) == Approx(eT(2 * r_orig[i])) );
    }

  for (uword i = 16; i < 100; ++i)
    {
    REQUIRE( eT(r[i]) == Approx(eT(r_orig[i])) );
    }
  }



TEST_CASE("row_subvec_2")
  {
  test_row_subvec_2<double>();
  test_row_subvec_2<float>();
  test_row_subvec_2<u32>();
  test_row_subvec_2<s32>();
  test_row_subvec_2<u64>();
  test_row_subvec_2<s64>();
  }
