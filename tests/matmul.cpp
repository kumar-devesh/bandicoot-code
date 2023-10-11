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

#include <armadillo>
#include <bandicoot>
#include "catch.hpp"

using namespace coot;

TEMPLATE_TEST_CASE("mat_mul_square", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y;

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_subview_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 30, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 24) * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 24) * y;

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_subview_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(30, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y.submat(5, 10, 24, 29);

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y.submat(5, 10, 24, 29);

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_subviews", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 30, arma::fill::randu);
  arma::Mat<eT> cpu_y(30, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 24) * cpu_y.submat(5, 10, 24, 29);

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 24) * y.submat(5, 10, 24, 29);

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t() * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y;

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_subview_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 30, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 24).t() * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 24).t() * y;

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_a_subview_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(30, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t() * cpu_y.submat(5, 10, 24, 29);

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y.submat(5, 10, 24, 29);

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_a_subviews", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 30, arma::fill::randu);
  arma::Mat<eT> cpu_y(30, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 24).t() * cpu_y.submat(5, 10, 24, 29);

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 24).t() * y.submat(5, 10, 24, 29);

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y.t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_b_subview_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 30, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 24) * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 24) * y.t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_subview_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(30, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y.submat(5, 10, 24, 29).t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y.submat(5, 10, 24, 29).t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_b_subviews", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 30, arma::fill::randu);
  arma::Mat<eT> cpu_y(30, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 24) * cpu_y.submat(5, 10, 24, 29).t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 24) * y.submat(5, 10, 24, 29).t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_both", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t()  * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y.t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_both_subview_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 30, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 24).t()  * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 24).t() * y.t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_both_subview_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(30, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t()  * cpu_y.submat(5, 10, 24, 29).t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y.submat(5, 10, 24, 29).t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_square_trans_subview_both", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 30, arma::fill::randu);
  arma::Mat<eT> cpu_y(30, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 24).t()  * cpu_y.submat(5, 10, 24, 29).t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 24).t() * y.submat(5, 10, 24, 29).t();

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_y(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y;

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_subview_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 60, arma::fill::randu);
  arma::Mat<eT> cpu_y(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 54) * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 54) * y;

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_subview_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_y(60, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y.submat(5, 10, 54, 29);

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y.submat(5, 10, 54, 29);

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_subviews", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 60, arma::fill::randu);
  arma::Mat<eT> cpu_y(60, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 54) * cpu_y.submat(5, 10, 54, 29);

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 54) * y.submat(5, 10, 54, 29);

  for (uword c = 0; c < 20; ++c)
    {
    for (uword r = 0; r < 20; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t() * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y;

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_subview_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 60, arma::fill::randu);
  arma::Mat<eT> cpu_y(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 54).t() * cpu_y;

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 54).t() * y;

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_a_subview_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_y(40, 60, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t() * cpu_y.submat(10, 5, 29, 54);

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y.submat(10, 5, 29, 54);

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_a_subviews", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 60, arma::fill::randu);
  arma::Mat<eT> cpu_y(40, 60, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 54).t() * cpu_y.submat(10, 5, 29, 54);

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 54).t() * y.submat(10, 5, 29, 54);

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y.t();

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_b_subview_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(60, 40, arma::fill::randu);
  arma::Mat<eT> cpu_y(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(5, 10, 54, 29) * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(5, 10, 54, 29) * y.t();

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_subview_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_y(60, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x * cpu_y.submat(5, 10, 54, 29).t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x * y.submat(5, 10, 54, 29).t();

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_b_subviews", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(60, 40, arma::fill::randu);
  arma::Mat<eT> cpu_y(60, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(5, 10, 54, 29) * cpu_y.submat(5, 10, 54, 29).t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(5, 10, 54, 29) * y.submat(5, 10, 54, 29).t();

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_both", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_y(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t() * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y.t();

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_both_subview_a", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 60, arma::fill::randu);
  arma::Mat<eT> cpu_y(50, 20, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 54).t() * cpu_y.t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 54).t() * y.t();

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_both_subview_b", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(20, 50, arma::fill::randu);
  arma::Mat<eT> cpu_y(60, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.t() * cpu_y.submat(5, 10, 54, 29).t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.t() * y.submat(5, 10, 54, 29).t();

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_mul_nonsquare_trans_subview_both", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_x(40, 60, arma::fill::randu);
  arma::Mat<eT> cpu_y(60, 40, arma::fill::randu);
  arma::Mat<eT> cpu_z = cpu_x.submat(10, 5, 29, 54).t() * cpu_y.submat(5, 10, 54, 29).t();

  Mat<eT> x(cpu_x);
  Mat<eT> y(cpu_y);

  Mat<eT> z = x.submat(10, 5, 29, 54).t() * y.submat(5, 10, 54, 29).t();

  for (uword c = 0; c < 50; ++c)
    {
    for (uword r = 0; r < 50; ++r)
      {
      REQUIRE( eT(z(r, c)) == Approx(eT(cpu_z(r, c))) );
      }
    }
  }



TEMPLATE_TEST_CASE("mat_vec_mul_square", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(50, 50, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A * cpu_x;

  Mat<eT> A(cpu_A);
  Mat<eT> x(cpu_x);
  Mat<eT> y = A * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("submat_vec_mul_square", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(70, 80, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.submat(10, 15, 59, 64) * cpu_x;

  Mat<eT> A(cpu_A);
  Mat<eT> x(cpu_x);
  Mat<eT> y = A.submat(10, 15, 59, 64) * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("mat_subvec_mul_square", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(50, 50, arma::fill::randu);
  arma::Col<eT> cpu_x(80, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A * cpu_x.subvec(20, 69);

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A * x.subvec(20, 69);

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("submat_subvec_mul_square", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(70, 60, arma::fill::randu);
  arma::Col<eT> cpu_x(80, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.submat(10, 5, 59, 54) * cpu_x.subvec(10, 59);

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A.submat(10, 5, 59, 54) * x.subvec(10, 59);

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("mat_vec_mul_square_trans", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(50, 50, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.t() * cpu_x;

  Mat<eT> A(cpu_A);
  Mat<eT> x(cpu_x);
  Mat<eT> y = A.t() * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("submat_vec_mul_square_trans", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(70, 80, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.submat(10, 15, 59, 64).t() * cpu_x;

  Mat<eT> A(cpu_A);
  Mat<eT> x(cpu_x);
  Mat<eT> y = A.submat(10, 15, 59, 64).t() * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("mat_subvec_mul_square_trans", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(50, 50, arma::fill::randu);
  arma::Col<eT> cpu_x(80, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.t() * cpu_x.subvec(15, 64);

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A.t() * x.subvec(15, 64);

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("submat_subvec_mul_square_trans", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(70, 75, arma::fill::randu);
  arma::Col<eT> cpu_x(80, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.submat(10, 15, 59, 64).t() * cpu_x.subvec(10, 59);

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A.submat(10, 15, 59, 64).t() * x.subvec(10, 59);

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("mat_vec_mul_nonsquare", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(10, 50, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A * cpu_x;

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("submat_vec_mul_nonsquare", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(20, 70, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.submat(5, 10, 14, 59) * cpu_x;

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A.submat(5, 10, 14, 59) * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("mat_subvec_mul_nonsquare", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(10, 50, arma::fill::randu);
  arma::Col<eT> cpu_x(80, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A * cpu_x.subvec(15, 64);

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A * x.subvec(15, 64);

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("submat_subvec_mul_nonsquare", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(40, 70, arma::fill::randu);
  arma::Col<eT> cpu_x(80, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.submat(10, 5, 29, 54) * cpu_x.subvec(15, 64);

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A.submat(10, 5, 29, 54) * x.subvec(15, 64);

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("mat_vec_mul_nonsquare_trans", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(50, 10, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.t() * cpu_x;

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A.t() * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("submat_vec_mul_nonsquare_trans", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(70, 20, arma::fill::randu);
  arma::Col<eT> cpu_x(50, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.submat(10, 5, 59, 14).t() * cpu_x;

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A.submat(10, 5, 59, 14).t() * x;

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("mat_subvec_mul_nonsquare_trans", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(50, 10, arma::fill::randu);
  arma::Col<eT> cpu_x(80, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.t() * cpu_x.subvec(15, 64);

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A.t() * x.subvec(15, 64);

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }



TEMPLATE_TEST_CASE("submat_subvec_mul_nonsquare_trans", "[matmul]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  arma::Mat<eT> cpu_A(70, 20, arma::fill::randu);
  arma::Col<eT> cpu_x(80, arma::fill::randu);
  arma::Col<eT> cpu_y = cpu_A.submat(10, 5, 59, 14).t() * cpu_x.subvec(15, 64);

  Mat<eT> A(cpu_A);
  Col<eT> x(cpu_x);
  Col<eT> y = A.submat(10, 5, 59, 14).t() * x.subvec(15, 64);

  REQUIRE( y.n_elem == cpu_y.n_elem );
  for (uword i = 0; i < y.n_elem; ++i)
    {
    REQUIRE( eT(y(i)) == Approx(eT(cpu_y(i))) );
    }
  }
