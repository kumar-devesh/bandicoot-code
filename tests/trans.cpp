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

template<typename eT1, typename eT2>
void check_transpose(const Mat<eT1>& A, const Mat<eT2>& B)
  {
  REQUIRE(A.n_rows == B.n_cols);
  REQUIRE(A.n_cols == B.n_rows);

  arma::Mat<eT1> A_cpu(A);
  arma::Mat<eT2> B_cpu(B);

  for (size_t c = 0; c < A_cpu.n_cols; ++c)
    {
    for (size_t r = 0; r < A_cpu.n_rows; ++r)
      {
      REQUIRE( eT1(A_cpu(r, c)) == Approx(eT1(eT2(B_cpu(c, r)))) );
      }
    }
  }



TEMPLATE_TEST_CASE("htrans_basic", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 5, distr_param(0, 50));
  Mat<eT> y = trans(x);
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("strans_basic", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 5, distr_param(0, 50));
  Mat<eT> y = trans(x);
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("member_t", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 5, distr_param(0, 50));
  Mat<eT> y = x.t();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("member_ht", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 5, distr_param(0, 50));
  Mat<eT> y = x.ht();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("member_st", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 5, distr_param(0, 50));
  Mat<eT> y = x.st();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("large_htrans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(1000, 200, distr_param(0, 1000));
  Mat<eT> y = x.t();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("large_strans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(1000, 200, distr_param(0, 1000));
  Mat<eT> y = strans(x);
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("alias_htrans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 5, distr_param(0, 50));
  Mat<eT> x_old(x);
  x = x.t();
  check_transpose(x, x_old);
  }



TEMPLATE_TEST_CASE("alias_strans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 5, distr_param(0, 50));
  Mat<eT> x_old(x);
  x = x.st();
  check_transpose(x, x_old);
  }



TEMPLATE_TEST_CASE("alias_htrans_subview", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 5, distr_param(0, 50));
  Mat<eT> x_old = x.submat(1, 1, 5, 3);
  x = trans(x.submat(1, 1, 5, 3));
  check_transpose(x, x_old);
  }



TEMPLATE_TEST_CASE("alias_strans_subview", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 5, distr_param(0, 50));
  Mat<eT> x_old = x.submat(1, 1, 5, 3);
  x = strans(x.submat(1, 1, 5, 3));
  check_transpose(x, x_old);
  }



TEMPLATE_TEST_CASE("zero_size_htrans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x;
  Mat<eT> y = x.t();
  REQUIRE(y.n_elem == 0);
  }



TEMPLATE_TEST_CASE("zero_size_strans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x;
  Mat<eT> y = x.st();
  REQUIRE(y.n_elem == 0);
  }



TEMPLATE_TEST_CASE("1x1_htrans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(1, 1);
  x[0] = eT(3);
  Mat<eT> y = x.t();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("1x1_strans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(1, 1);
  x[0] = eT(3);
  Mat<eT> y = x.st();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("row_htrans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Row<eT> x = randi<Row<eT>>(10, distr_param(0, 50));
  Col<eT> y = x.t();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("row_strans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Row<eT> x = randi<Row<eT>>(10, distr_param(0, 50));
  Col<eT> y = x.st();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("col_htrans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(10, distr_param(0, 50));
  Row<eT> y = x.t();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("col_strans", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(10, distr_param(0, 50));
  Row<eT> y = x.st();
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE("simple_htrans2", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 20, distr_param(0, 50));
  Mat<eT> y = eT(3) * x.t();
  Mat<eT> y2 = x.t();
  y2 *= eT(3);

  REQUIRE( y.n_rows == y2.n_rows );
  REQUIRE( y.n_cols == y2.n_cols );
  for (uword c = 0; c < y.n_cols; ++c)
    {
    for (uword r = 0; r < y.n_rows; ++r)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y2(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("htrans_after_op", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 20, distr_param(0, 50));
  Mat<eT> z = randi<Mat<eT>>(30, 20, distr_param(10, 100));
  Mat<eT> y = trans((4 * (x + 3)) % z);
  Mat<eT> op_result = (4 * (x + 3)) % z;
  check_transpose(y, op_result);
  }



TEMPLATE_TEST_CASE("strans_after_op", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 20, distr_param(0, 50));
  Mat<eT> z = randi<Mat<eT>>(30, 20, distr_param(10, 100));
  Mat<eT> y = strans((4 * (x + 3)) % z);
  Mat<eT> op_result = (4 * (x + 3)) % z;
  check_transpose(y, op_result);
  }



TEMPLATE_TEST_CASE("htrans_in_op", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 20, distr_param(0, 50));
  Mat<eT> z = randi<Mat<eT>>(20, 30, distr_param(10, 100));
  Mat<eT> y = (4 * abs(x) + z.t());
  Mat<eT> zt = trans(z);
  Mat<eT> y2 = (4 * abs(x) + zt);

  REQUIRE( y.n_rows == y2.n_rows );
  REQUIRE( y.n_cols == y2.n_cols );
  for (uword c = 0; c < y.n_cols; ++c)
    {
    for (uword r = 0; r < y.n_rows; ++r)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y2(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("strans_in_op", "[trans]", double, float, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(30, 20, distr_param(0, 50));
  Mat<eT> z = randi<Mat<eT>>(20, 30, distr_param(10, 100));
  Mat<eT> y = (4 * abs(x) + z.st());
  Mat<eT> zt = strans(z);
  Mat<eT> y2 = (4 * abs(x) + zt);

  REQUIRE( y.n_rows == y2.n_rows );
  REQUIRE( y.n_cols == y2.n_cols );
  for (uword c = 0; c < y.n_cols; ++c)
    {
    for (uword r = 0; r < y.n_rows; ++r)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(y2(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE(
  "conv_before_htrans",
  "[trans]",
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

  Mat<eT1> x = randi<Mat<eT1>>(20, 40, distr_param(0, 50));
  Mat<eT2> y = trans(conv_to<Mat<eT2>>::from(x));
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE(
  "conv_before_strans",
  "[trans]",
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

  Mat<eT1> x = randi<Mat<eT1>>(20, 40, distr_param(0, 50));
  Mat<eT2> y = strans(conv_to<Mat<eT2>>::from(x));
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE(
  "conv_after_htrans",
  "[trans]",
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

  Mat<eT1> x = randi<Mat<eT1>>(20, 40, distr_param(0, 50));
  Mat<eT2> y = conv_to<Mat<eT2>>::from(x.t());
  check_transpose(y, x);
  }



TEMPLATE_TEST_CASE(
  "conv_after_strans",
  "[trans]",
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

  Mat<eT1> x = randi<Mat<eT1>>(20, 40, distr_param(0, 50));
  Mat<eT2> y = conv_to<Mat<eT2>>::from(x.st());
  check_transpose(y, x);
  }
