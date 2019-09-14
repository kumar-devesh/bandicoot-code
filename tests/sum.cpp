// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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
#include <armadillo>
#include "catch.hpp"

using namespace coot;

template<typename eT>
void test_sum_colwise()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = sum(x, 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(10 * c)) );
    }
  }



TEST_CASE("sum_colwise_1")
  {
  test_sum_colwise<float>();
  test_sum_colwise<double>();
  test_sum_colwise<u32>();
  test_sum_colwise<s32>();
  test_sum_colwise<u64>();
  test_sum_colwise<s64>();
  }



template<typename eT>
void test_sum_colwise_2()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = sum(x, 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(45)) );
    }
  }


TEST_CASE("sum_colwise_2")
  {
  test_sum_colwise_2<float>();
  test_sum_colwise_2<double>();
  test_sum_colwise_2<u32>();
  test_sum_colwise_2<s32>();
  test_sum_colwise_2<u64>();
  test_sum_colwise_2<s64>();
  }



template<typename eT>
void test_sum_rowwise()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = sum(x, 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(45)) );
    }
  }



TEST_CASE("sum_rowwise_1")
  {
  test_sum_rowwise<float>();
  test_sum_rowwise<double>();
  test_sum_rowwise<u32>();
  test_sum_rowwise<s32>();
  test_sum_rowwise<u64>();
  test_sum_rowwise<s64>();
  }



template<typename eT>
void test_sum_rowwise_2()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = sum(x, 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(10 * r)) );
    }
  }



TEST_CASE("sum_rowwise_2")
  {
  test_sum_rowwise_2<float>();
  test_sum_rowwise_2<double>();
  test_sum_rowwise_2<u32>();
  test_sum_rowwise_2<s32>();
  test_sum_rowwise_2<u64>();
  test_sum_rowwise_2<s64>();
  }



template<typename eT>
void test_subview_sum_colwise()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = sum(x.submat(1, 1, 8, 8), 0);

  REQUIRE( s.n_rows == 1 );
  REQUIRE( s.n_cols == 8 );
  for (uword c = 0; c < 8; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(8 * (c + 1))) );
    }
  }



TEST_CASE("subview_sum_colwise_1")
  {
  test_subview_sum_colwise<float>();
  test_subview_sum_colwise<double>();
  test_subview_sum_colwise<u32>();
  test_subview_sum_colwise<s32>();
  test_subview_sum_colwise<u64>();
  test_subview_sum_colwise<s64>();
  }



template<typename eT>
void test_subview_sum_colwise_2()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = sum(x.submat(1, 1, 8, 8), 0);

  REQUIRE( s.n_rows == 1 );
  REQUIRE( s.n_cols == 8 );
  for (uword c = 0; c < 8; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(36)) );
    }
  }



TEST_CASE("subview_sum_colwise_2")
  {
  test_subview_sum_colwise_2<float>();
  test_subview_sum_colwise_2<double>();
  test_subview_sum_colwise_2<u32>();
  test_subview_sum_colwise_2<s32>();
  test_subview_sum_colwise_2<u64>();
  test_subview_sum_colwise_2<s64>();
  }



template<typename eT>
void test_subview_sum_colwise_full()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = sum(x.submat(0, 0, 9, 9), 0);

  REQUIRE( s.n_rows == 1  );
  REQUIRE( s.n_cols == 10 );
  for (uword c = 0; c < 10; ++c)
    {
    REQUIRE( eT(s[c]) == Approx(eT(10 * c)) );
    }
  }



TEST_CASE("subview_sum_colwise_3")
  {
  test_subview_sum_colwise_full<float>();
  test_subview_sum_colwise_full<double>();
  test_subview_sum_colwise_full<u32>();
  test_subview_sum_colwise_full<s32>();
  test_subview_sum_colwise_full<u64>();
  test_subview_sum_colwise_full<s64>();
  }



template<typename eT>
void test_subview_sum_rowwise()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = c;
      }
    }

  Mat<eT> s = sum(x.submat(1, 1, 8, 8), 1);

  REQUIRE( s.n_rows == 8 );
  REQUIRE( s.n_cols == 1 );
  for (uword r = 0; r < 8; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(36)) );
    }
  }



TEST_CASE("subview_sum_rowwise_1")
  {
  test_subview_sum_rowwise<float>();
  test_subview_sum_rowwise<double>();
  test_subview_sum_rowwise<u32>();
  test_subview_sum_rowwise<s32>();
  test_subview_sum_rowwise<u64>();
  test_subview_sum_rowwise<s64>();
  }



template<typename eT>
void test_subview_sum_rowwise_2()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = sum(x.submat(1, 1, 8, 8), 1);

  REQUIRE( s.n_rows == 8 );
  REQUIRE( s.n_cols == 1 );
  for (uword r = 0; r < 8; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(8 * (r + 1))) );
    }
  }



TEST_CASE("subview_sum_rowwise_2")
  {
  test_subview_sum_rowwise_2<float>();
  test_subview_sum_rowwise_2<double>();
  test_subview_sum_rowwise_2<u32>();
  test_subview_sum_rowwise_2<s32>();
  test_subview_sum_rowwise_2<u64>();
  test_subview_sum_rowwise_2<s64>();
  }



template<typename eT>
void test_subview_sum_rowwise_full()
  {
  Mat<eT> x(10, 10);
  for (uword c = 0; c < 10; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      x(r, c) = r;
      }
    }

  Mat<eT> s = sum(x.submat(0, 0, 9, 9), 1);

  REQUIRE( s.n_rows == 10 );
  REQUIRE( s.n_cols == 1  );
  for (uword r = 0; r < 10; ++r)
    {
    REQUIRE( eT(s[r]) == Approx(eT(10 * r)) );
    }
  }



TEST_CASE("subview_sum_rowwise_3")
  {
  test_subview_sum_rowwise_full<float>();
  test_subview_sum_rowwise_full<double>();
  test_subview_sum_rowwise_full<u32>();
  test_subview_sum_rowwise_full<s32>();
  test_subview_sum_rowwise_full<u64>();
  test_subview_sum_rowwise_full<s64>();
  }
