// Copyright 2021 Ryan Curtin (http://www.ratml.org/)
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
void test_larfg_manual()
  {
  // Simple test: pass a vector of all ones.
  Col<eT> x(10);
  x.ones();
  Col<eT> x_copy(x);

  const eT tau = lapack::larfg(x);

  REQUIRE(abs(tau) <= eT(2));

  // Now, reconstruct H for the test.
  Mat<eT> H(10, 10);
  H.eye();
  Col<eT> v_aug = x;
  v_aug[0] = eT(1);
  H -= tau * v_aug * v_aug.t();

  Col<eT> out = H * x_copy;

  REQUIRE(((eT) out[0]) == Approx((eT) x[0]));
  for (uword i = 1; i < 10; ++i)
    {
    REQUIRE(abs(out[i]) < eT(1e-5));
    }
  }



TEST_CASE("larfg_manual")
  {
  test_larfg_manual<float>();
  test_larfg_manual<double>();
  }



template<typename eT>
void test_larfg_zeros()
  {
  // Ensure that if we pass x equal to all zeros, then we get the correct result back (tau = 0, H = I).
  Col<eT> x(10);
  x.zeros();

  const eT tau = lapack::larfg(x);

  REQUIRE(tau == Approx(eT(0)).margin(1e-6));

  for (uword i = 0; i < 10; ++i)
    {
    REQUIRE(((eT) x[i]) == Approx(eT(0)).margin(1e-6));
    }

  // In addition, if alpha (e.g. x[0]) is greater than zero but the rest of x is 0, this should still be the case.
  x.zeros();
  x[0] = eT(5);

  const eT tau2 = lapack::larfg(x);

  REQUIRE(tau2 == Approx(eT(0)).margin(1e-6));

  REQUIRE(((eT) x[0]) == Approx(eT(5)));
  for (uword i = 1; i < 10; ++i)
    {
    REQUIRE(((eT) x[i]) == Approx(eT(0)).margin(1e-6));
    }
  }



TEST_CASE("larfg_zeros")
  {
  test_larfg_zeros<float>();
  test_larfg_zeros<double>();
  }



template<typename eT>
void test_larfg_big()
  {
  // Use a big random vector.
  Col<eT> x(1000);
  x.randu();
  Col<eT> x_copy(x);

  const eT tau = lapack::larfg(x);

  REQUIRE(abs(tau) <= eT(2));

  // Now, reconstruct H for the test.
  Mat<eT> H(1000, 1000);
  H.eye();
  Col<eT> v_aug = x;
  v_aug[0] = eT(1);
  H -= tau * v_aug * v_aug.t();

  Col<eT> out = H * x_copy;

  REQUIRE(((eT) out[0]) == Approx((eT) x[0]));
  for (uword i = 1; i < 1000; ++i)
    {
    REQUIRE(abs(out[i]) < eT(1e-5));
    }
  }



TEST_CASE("larfg_big")
  {
  test_larfg_big<float>();
  test_larfg_big<double>();
  }



// This test will not work until dot() can support underflow and safely scale.
/*
template<typename eT>
void test_larfg_small_norm()
  {
  // Create a vector with a very small norm.
  eT min_val = 10000 * std::numeric_limits<eT>::min();

  // All elements will be some multiple of that minimum value.
  Col<eT> x(500);
  for (uword i = 0; i < x.n_elem; ++i)
    {
    x[i] = (rand() % 100) * min_val;
    }

  Col<eT> x_copy(x);

  const eT tau = lapack::larfg(x);

  REQUIRE(abs(tau) <= eT(2));

  // Now reconstruct H for the test.
  Mat<eT> H(500, 500);
  H.eye();
  Col<eT> v_aug = x;
  v_aug[0] = eT(1);
  H -= tau * v_aug * v_aug.t();

  Col<eT> out = H * x_copy;

  REQUIRE(((((eT) x[0]) != eT(0.0)) && (((eT) x[0]) != eT(-0.0))));
  REQUIRE(((eT) out[0]) == Approx((eT) x[0]));
  for (uword i = 1; i < 500; ++i)
    {
    REQUIRE(abs(out[i]) < eT(1e-5));
    }
  }



TEST_CASE("larfg_small_norm")
  {
  test_larfg_small_norm<float>();
  test_larfg_small_norm<double>();
  }
*/



template<typename eT>
void test_larfg_one_elem()
  {
  Col<eT> x(1);
  x[0] = eT(3);

  const eT tau = lapack::larfg(x);

  REQUIRE(tau == Approx(eT(0)).margin(1e-20));

  REQUIRE(eT(x[0]) == Approx(eT(3)));
  }



TEST_CASE("larfg_one_elem")
  {
  test_larfg_one_elem<float>();
  test_larfg_one_elem<double>();
  }



template<typename eT>
void test_larfg_empty()
  {
  Col<eT> x(0);

  const eT tau = lapack::larfg(x);

  REQUIRE(tau == Approx(eT(0)).margin(1e-20));
  }



TEST_CASE("larfg_empty")
  {
  test_larfg_empty<float>();
  test_larfg_empty<double>();
  }
