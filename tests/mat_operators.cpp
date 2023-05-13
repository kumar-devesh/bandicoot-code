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

TEMPLATE_TEST_CASE("two_mat_plus", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(0));
  Mat<eT> y(5, 5);
  y.fill(eT(3));

  Mat<eT> z1 = x + y;
  Mat<eT> z2 = y + x;
  x += y;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(3)) );
      REQUIRE( eT(z1(r, c)) == Approx(eT(3)) );
      REQUIRE( eT(z2(r, c)) == Approx(eT(3)) );
      }
    }
  }



TEMPLATE_TEST_CASE("two_mat_minus", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));
  Mat<eT> y(5, 5);
  y.fill(eT(5));

  Mat<eT> z = x - y;
  x -= y;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(5)) );
      REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
      }
    }
  }



TEMPLATE_TEST_CASE("two_mat_mul", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(1));
  Mat<eT> y(5, 5);
  y.fill(eT(10));

  Mat<eT> z1 = x % y;
  Mat<eT> z2 = y % x;
  x %= y;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
      REQUIRE( eT(z1(r, c)) == Approx(eT(10)) );
      REQUIRE( eT(z2(r, c)) == Approx(eT(10)) );
      }
    }
  }



TEMPLATE_TEST_CASE("two_mat_div", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  x.fill(eT(10));
  Mat<eT> y(5, 5);
  y.fill(eT(2));

  Mat<eT> z = x / y;
  x /= y;

  for (uword c = 0; c < 5; ++c)
    {
    for (uword r = 0; r < 5; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(5)) );
      REQUIRE( eT(z(r, c)) == Approx(eT(5)) );
      }
    }
  }



TEMPLATE_TEST_CASE("simple_mat_mul", "[mat_operators]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(3, 3);
  Mat<eT> y(3, 3);

  x(0, 0) = eT(1);
  x(1, 0) = eT(2);
  x(2, 0) = eT(3);
  x(0, 1) = eT(5);
; x(1, 1) = eT(6);
  x(2, 1) = eT(7);
  x(0, 2) = eT(9);
  x(1, 2) = eT(11);
  x(2, 2) = eT(13);

  y(0, 0) = eT(10);
  y(1, 0) = eT(11);
  y(2, 0) = eT(12);
  y(0, 1) = eT(13);
  y(1, 1) = eT(14);
  y(2, 1) = eT(15);
  y(0, 2) = eT(16);
  y(1, 2) = eT(17);
  y(2, 2) = eT(18);

  Mat<eT> z1 = x * y;
  Mat<eT> z2 = x * y.t();
  Mat<eT> z3 = x.t() * y;
  Mat<eT> z4 = x.t() * y.t();

  y *= x;

  // Check against hand-computed results.
  REQUIRE( eT(z1(0, 0)) == Approx(eT(173)) );
  REQUIRE( eT(z1(1, 0)) == Approx(eT(218)) );
  REQUIRE( eT(z1(2, 0)) == Approx(eT(263)) );
  REQUIRE( eT(z1(0, 1)) == Approx(eT(218)) );
  REQUIRE( eT(z1(1, 1)) == Approx(eT(275)) );
  REQUIRE( eT(z1(2, 1)) == Approx(eT(332)) );
  REQUIRE( eT(z1(0, 2)) == Approx(eT(263)) );
  REQUIRE( eT(z1(1, 2)) == Approx(eT(332)) );
  REQUIRE( eT(z1(2, 2)) == Approx(eT(401)) );

  REQUIRE( eT(z2(0, 0)) == Approx(eT(219)) );
  REQUIRE( eT(z2(1, 0)) == Approx(eT(274)) );
  REQUIRE( eT(z2(2, 0)) == Approx(eT(329)) );
  REQUIRE( eT(z2(0, 1)) == Approx(eT(234)) );
  REQUIRE( eT(z2(1, 1)) == Approx(eT(293)) );
  REQUIRE( eT(z2(2, 1)) == Approx(eT(352)) );
  REQUIRE( eT(z2(0, 2)) == Approx(eT(249)) );
  REQUIRE( eT(z2(1, 2)) == Approx(eT(312)) );
  REQUIRE( eT(z2(2, 2)) == Approx(eT(375)) );

  REQUIRE( eT(z3(0, 0)) == Approx(eT(68)) );
  REQUIRE( eT(z3(1, 0)) == Approx(eT(200)) );
  REQUIRE( eT(z3(2, 0)) == Approx(eT(367)) );
  REQUIRE( eT(z3(0, 1)) == Approx(eT(86)) );
  REQUIRE( eT(z3(1, 1)) == Approx(eT(254)) );
  REQUIRE( eT(z3(2, 1)) == Approx(eT(466)) );
  REQUIRE( eT(z3(0, 2)) == Approx(eT(104)) );
  REQUIRE( eT(z3(1, 2)) == Approx(eT(308)) );
  REQUIRE( eT(z3(2, 2)) == Approx(eT(565)) );

  REQUIRE( eT(z4(0, 0)) == Approx(eT(84)) );
  REQUIRE( eT(z4(1, 0)) == Approx(eT(240)) );
  REQUIRE( eT(z4(2, 0)) == Approx(eT(441)) );
  REQUIRE( eT(z4(0, 1)) == Approx(eT(90)) );
  REQUIRE( eT(z4(1, 1)) == Approx(eT(258)) );
  REQUIRE( eT(z4(2, 1)) == Approx(eT(474)) );
  REQUIRE( eT(z4(0, 2)) == Approx(eT(96)) );
  REQUIRE( eT(z4(1, 2)) == Approx(eT(276)) );
  REQUIRE( eT(z4(2, 2)) == Approx(eT(507)) );

  REQUIRE( eT(y(0, 0)) == Approx(eT(84)) );
  REQUIRE( eT(y(1, 0)) == Approx(eT(90)) );
  REQUIRE( eT(y(2, 0)) == Approx(eT(96)) );
  REQUIRE( eT(y(0, 1)) == Approx(eT(240)) );
  REQUIRE( eT(y(1, 1)) == Approx(eT(258)) );
  REQUIRE( eT(y(2, 1)) == Approx(eT(276)) );
  REQUIRE( eT(y(0, 2)) == Approx(eT(441)) );
  REQUIRE( eT(y(1, 2)) == Approx(eT(474)) );
  REQUIRE( eT(y(2, 2)) == Approx(eT(507)) );
  }



TEMPLATE_TEST_CASE("mat_copy", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(10, 10);
  x.randu();

  Mat<eT> y(10, 10);
  y.randu();

  Mat<eT> z;

  y = x;
  z = x;

  REQUIRE( x.n_rows == y.n_rows );
  REQUIRE( x.n_cols == y.n_cols );
  REQUIRE( y.n_rows == z.n_rows );
  REQUIRE( y.n_cols == z.n_cols );

  for (size_t c = 0; c < 10; ++c)
    {
    for (size_t r = 0; r < 10; ++r)
      {
      REQUIRE( eT(x(r, c)) == eT(y(r, c)) );
      REQUIRE( eT(x(r, c)) == eT(z(r, c)) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_copy_from_dev_mem", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(5, 5);
  for (uword i = 0; i < 25; ++i)
    {
    x(i) = i;
    }

  eT* cpu_mem = new eT[25];

  x.copy_from_dev_mem(cpu_mem, 25);

  for (uword i = 0; i < 25; ++i)
    {
    REQUIRE( cpu_mem[i] == Approx(eT(i)) );
    }

  delete[] cpu_mem;
  }



TEMPLATE_TEST_CASE("mat_copy_into_dev_mem", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  eT* cpu_mem = new eT[25];

  for (uword i = 0; i < 25; ++i)
    {
    cpu_mem[i] = eT(i);
    }

  Mat<eT> x(5, 5);
  x.copy_into_dev_mem(cpu_mem, 25);

  for (uword i = 0; i < 25; ++i)
    {
    REQUIRE( eT(x(i)) == Approx(eT(i)) );
    }

  delete[] cpu_mem;
  }



TEMPLATE_TEST_CASE("mat_val_proxy_ops_1", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x(8, 8);
  x.zeros();

  x(1, 1) = eT(3);
  x(1) = eT(2);

  x(2, 2) += eT(5);
  x(2) += eT(3);

  x(3, 3) += eT(10);
  x(3, 3) -= eT(5);
  x(3) += eT(10);
  x(3) -= eT(5);

  x(4, 4) = eT(3);
  x(4, 4) *= eT(2);
  x(4) = eT(3);
  x(4) *= eT(2);

  x(5, 5) = eT(10);
  x(5, 5) /= eT(2);
  x(5) = eT(10);
  x(5) /= eT(2);

  REQUIRE( eT(x(1, 1)) == Approx(eT(3)) );
  REQUIRE( eT(x(1)   ) == Approx(eT(2)) );
  REQUIRE( eT(x(2, 2)) == Approx(eT(5)) );
  REQUIRE( eT(x(2)   ) == Approx(eT(3)) );
  REQUIRE( eT(x(3, 3)) == Approx(eT(5)) );
  REQUIRE( eT(x(3)   ) == Approx(eT(5)) );
  REQUIRE( eT(x(4, 4)) == Approx(eT(6)) );
  REQUIRE( eT(x(4)   ) == Approx(eT(6)) );
  REQUIRE( eT(x(5, 5)) == Approx(eT(5)) );
  REQUIRE( eT(x(5)   ) == Approx(eT(5)) );

  REQUIRE( eT(x(0)   ) == eT(0) );
  REQUIRE( eT(x(6)   ) == eT(0) );
  REQUIRE( eT(x(7)   ) == eT(0) );

  for (uword c = 1; c < 8; ++c)
    {
    for (uword r = 0; r < 8; ++r)
      {
      if (r != c)
        {
        REQUIRE( eT(x(r, c)) == eT(0) );
        }
      }
    }
  }



TEMPLATE_TEST_CASE("mat_submat_insertion", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) = cpu_y;
  x.submat(5, 5, 14, 19) = y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_submat_add", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) += cpu_y;
  x.submat(5, 5, 14, 19) += y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_submat_minus", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) -= cpu_y;
  x.submat(5, 5, 14, 19) -= y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_submat_schur", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) %= cpu_y;
  x.submat(5, 5, 14, 19) %= y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_submat_div", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1000, 2000));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(10, 20));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_x.submat(5, 5, 14, 19) /= cpu_y;
  x.submat(5, 5, 14, 19) /= y;

  for (uword c = 0; c < 30; ++c)
    {
    for (uword r = 0; r < 25; ++r)
      {
      REQUIRE( eT(x(r, c)) == Approx(eT(cpu_x(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_submat_extract", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x = arma::randi<arma::Mat<eT>>(25, 30, arma::distr_param(1, 100));
  arma::Mat<eT> cpu_y = arma::randi<arma::Mat<eT>>(10, 15, arma::distr_param(100, 200));

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  cpu_y = cpu_x.submat(5, 5, 14, 19);
  y = x.submat(5, 5, 14, 19);

  for (uword c = 0; c < 15; ++c)
    {
    for (uword r = 0; r < 10; ++r)
      {
      REQUIRE( eT(y(r, c)) == Approx(eT(cpu_y(r, c))) );
      REQUIRE( eT(y(r, c)) == Approx(eT(x(r + 5, c + 5))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_submat_fill", "[mat_operators]", float, double, u32, s32, u64, s64)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> x(20, 20);
  x.fill(eT(10));

  x.submat(5, 6, 14, 15).fill(eT(2));

  for (uword r = 0; r < 20; ++r)
    {
    for (uword c = 0; c < 20; ++c)
      {
      if (r >= 5 && r <= 14 && c >= 6 && c <= 15)
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(2)) );
        }
      else
        {
        REQUIRE( eT(x(r, c)) == Approx(eT(10)) );
        }
      }
    }
  }
