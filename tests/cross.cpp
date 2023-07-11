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

#include <armadillo>
#include <bandicoot>
#include "catch.hpp"

using namespace coot;

TEMPLATE_TEST_CASE("cross_arma_comparison", "[cross]", u32, s32, u64, s64, float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (size_t trial = 0; trial < 50; ++trial)
    {
    Col<eT> a = randi<Col<eT>>(3, distr_param(0, 100));
    Col<eT> b = randi<Col<eT>>(3, distr_param(0, 100));

    if (!is_same_type<eT, u32>::value && !is_same_type<eT, u64>::value)
      {
      a -= 5;
      b -= 10;
      }

    arma::Col<eT> a_cpu(a);
    arma::Col<eT> b_cpu(b);

    Col<eT> out = cross(a, b);
    arma::Col<eT> out_cpu = arma::cross(a_cpu, b_cpu);

    REQUIRE( out.n_elem == out_cpu.n_elem );
    REQUIRE( eT(out[0]) == Approx(eT(out_cpu[0])) );
    REQUIRE( eT(out[1]) == Approx(eT(out_cpu[1])) );
    REQUIRE( eT(out[2]) == Approx(eT(out_cpu[2])) );
    }
  }



TEST_CASE("empty_cross", "[cross]")
  {
  fvec a, b;

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  fvec out;
  REQUIRE_THROWS( out = cross(a, b) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



TEST_CASE("wrong_size_cross", "[cross]")
  {
  fvec a = randu<fvec>(6);
  fvec b = randu<fvec>(2);

  // Disable cerr output for this test.
  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);

  fvec out;
  REQUIRE_THROWS( out = cross(a, b) );

  // Make only one wrong.
  b = randu<fvec>(3);
  REQUIRE_THROWS( out = cross(a, b) );

  b = randu<fvec>(6);
  a = randu<fvec>(3);
  REQUIRE_THROWS( out = cross(a, b) );

  // Restore cerr output.
  std::cerr.rdbuf(orig_cerr_buf);
  }



TEST_CASE("alias_cross", "[cross]")
  {
  fvec a = randu<fvec>(3);
  fvec b = randu<fvec>(3);

  fvec a_orig(a);

  a = cross(a, b);

  fvec out = cross(a_orig, b);

  REQUIRE( a.n_elem == 3 );
  REQUIRE( float(a[0]) == Approx(float(out[0])) );
  REQUIRE( float(a[1]) == Approx(float(out[1])) );
  REQUIRE( float(a[2]) == Approx(float(out[2])) );
  }



TEST_CASE("row_vs_col_cross", "[cross]")
  {
  fvec a = randu<fvec>(3);
  frowvec b = randu<frowvec>(3);
  fvec b_t = b.t();

  fvec out = cross(a, b);
  fvec out2 = cross(a, b_t);

  REQUIRE( out.n_elem == 3 );
  REQUIRE( float(out[0]) == Approx(float(out2[0])) );
  REQUIRE( float(out[1]) == Approx(float(out2[1])) );
  REQUIRE( float(out[2]) == Approx(float(out2[2])) );
  }



TEMPLATE_TEST_CASE
  (
  "conv_to_cross",
  "[cross]",
  (std::pair<double, float>), (std::pair<double, u32>), (std::pair<double, s32>), (std::pair<double, u64>), (std::pair<double, s64>),
  (std::pair<float, double>), (std::pair<float, u32>), (std::pair<float, s32>), (std::pair<float, u64>), (std::pair<float, s64>),
  (std::pair<u32, double>), (std::pair<u32, float>), (std::pair<u32, s32>), (std::pair<u32, u64>), (std::pair<u32, s64>),
  (std::pair<s32, double>), (std::pair<s32, float>), (std::pair<s32, u32>), (std::pair<s32, u64>), (std::pair<s32, s64>),
  (std::pair<u64, double>), (std::pair<u64, float>), (std::pair<u64, u32>), (std::pair<u64, s32>), (std::pair<u64, s64>),
  (std::pair<s64, double>), (std::pair<s64, float>), (std::pair<s64, u32>), (std::pair<s64, s32>), (std::pair<s64, u64>)
  )
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }

  Col<eT1> a = randi<Col<eT1>>(3, distr_param(0, 1000));
  Col<eT1> b = randi<Col<eT1>>(3, distr_param(0, 1000));

  Col<eT2> out = conv_to<Col<eT2>>::from(cross(a, b));

  Col<eT1> out_pre_conv = cross(a, b);
  Col<eT2> out_ref = conv_to<Col<eT2>>::from(out_pre_conv);

  REQUIRE( out.n_elem == 3 );

  REQUIRE( eT2(out[0]) == Approx(eT2(out_ref[0])) );
  REQUIRE( eT2(out[1]) == Approx(eT2(out_ref[1])) );
  REQUIRE( eT2(out[2]) == Approx(eT2(out_ref[2])) );
  }
