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

  const double tau = lapack::larfg(x);

  REQUIRE(abs(tau) <= eT(2));

  // Now, reconstruct H for the test.
  Mat<eT> H(10, 10);
  H.eye();
  Col<eT> v_aug = x;
  v_aug[0] = eT(1);
  H -= v_aug * v_aug.t();

  Col<eT> out = H * x_copy;

  REQUIRE(((eT) out[0]) == Approx((eT) x[0]));
  for (uword i = 1; i < 10; ++i)
    {
    REQUIRE(abs(out[i]) < eT(1e-8));
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

  }



TEST_CASE("larfg_zeros")
  {
  test_larfg_zeros<float>();
  test_larfg_zeros<double>();
  }



template<typename eT>
void test_larfg_big()
  {

  }



TEST_CASE("larfg_big")
  {
  test_larfg_big<float>();
  test_larfg_big<double>();
  }




template<typename eT>
void test_larfg_small_norm()
  {

  }



TEST_CASE("larfg_small_norm")
  {
  test_larfg_small_norm<float>();
  test_larfg_small_norm<double>();
  }
