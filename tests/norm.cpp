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



// Compare with Armadillo's implementation and ensure that the result is at least close.
TEMPLATE_TEST_CASE("norm_basic", "[norm]", double, float)
  {
  Col<TestType> x = randi<Col<TestType>>(1000, distr_param(0, 50));
  arma::Col<TestType> x_cpu(x);

  REQUIRE( norm(x, 1) == Approx(norm(x_cpu, 1)) );
  REQUIRE( norm(x, 2) == Approx(norm(x_cpu, 2)) );
  REQUIRE( norm(x, 3) == Approx(norm(x_cpu, 3)) );
  REQUIRE( norm(x, 4) == Approx(norm(x_cpu, 4)) );
  REQUIRE( norm(x, 5) == Approx(norm(x_cpu, 5)) );
  REQUIRE( norm(x, 6) == Approx(norm(x_cpu, 6)) );
  REQUIRE( norm(x, 7) == Approx(norm(x_cpu, 7)) );
  REQUIRE( norm(x, 8) == Approx(norm(x_cpu, 8)) );
  REQUIRE( norm(x, 9) == Approx(norm(x_cpu, 9)) );
  REQUIRE( norm(x, 10) == Approx(norm(x_cpu, 10)) );
  REQUIRE( norm(x, 42) == Approx(norm(x_cpu, 42)) );

  REQUIRE( norm(x, "-inf") == Approx(norm(x_cpu, "-inf")) );
  REQUIRE( norm(x, "inf") == Approx(norm(x_cpu, "inf")) );
  REQUIRE( norm(x, "fro") == Approx(norm(x_cpu, "fro")) );
  }



TEMPLATE_TEST_CASE("empty_norm", "[norm]", double, float)
  {
  Col<TestType> x;

  for (uword p = 1; p < 11; ++p)
    {
    REQUIRE( norm(x, p) == TestType(0) );
    }

  REQUIRE( norm(x, "-inf") == TestType(0) );
  REQUIRE( norm(x, "inf") == TestType(0) );
  REQUIRE( norm(x, "fro") == TestType(0) );
  }



TEMPLATE_TEST_CASE("single_element_norm", "[norm]", double, float)
  {
  Row<TestType> x(1);
  x[0] = TestType(10);

  for (uword p = 1; p < 11; ++p)
    {
    REQUIRE( norm(x, p) == Approx(TestType(10)) );
    }

  REQUIRE( norm(x, "-inf") == Approx(TestType(10)) );
  REQUIRE( norm(x, "inf") == Approx(TestType(10)) );
  REQUIRE( norm(x, "fro") == Approx(TestType(10)) );
  }



TEMPLATE_TEST_CASE("expr_norm", "[norm]", double, float)
  {
  Row<TestType> x = randi<Col<TestType>>(1000, distr_param(0, 50));
  Row<TestType> z = vectorise(repmat(trans(x + 3), 2, 2));

  for (uword p = 1; p < 11; ++p)
    {
    REQUIRE( norm(vectorise(repmat(trans(x + 3), 2, 2)), p) == Approx(norm(z, p)) );
    }

  REQUIRE( norm(vectorise(repmat(trans(x + 3), 2, 2)), "-inf") == Approx(norm(z, "-inf")) );
  REQUIRE( norm(vectorise(repmat(trans(x + 3), 2, 2)), "inf") == Approx(norm(z, "inf")) );
  REQUIRE( norm(vectorise(repmat(trans(x + 3), 2, 2)), "fro") == Approx(norm(z, "fro")) );
  }



// Attempt to get norm2 to overflow by using very large elements.
TEMPLATE_TEST_CASE("norm2_overflow", "[norm]", double, float)
  {
  Col<TestType> x(10000);
  Col<TestType> f = randu<Col<TestType>>(10000) * 0.9;
  x = f * std::numeric_limits<TestType>::max();

  const TestType x_max = max(x);
  Col<TestType> x_normalized = x / x_max;

  REQUIRE( norm(x, 2) == Approx(norm(x_normalized, 2) * x_max) );
  }



// Attempt to get norm2 to underflow by using very small elements.
TEMPLATE_TEST_CASE("norm2_underflow", "[norm]", double, float)
  {
  Col<TestType> x(10000);
  Col<TestType> f = randi<Col<TestType>>(10000, distr_param(1, 25));
  x = f * std::numeric_limits<TestType>::min();

  const TestType x_max = max(x);
  Col<TestType> x_normalized = x / x_max;

  REQUIRE( norm(x, 2) == Approx(norm(x_normalized, 2) * x_max) );
  }



// Try a complicated expression with norm() inside of it too.
TEMPLATE_TEST_CASE("double_norm_expr", "[norm]", double, float)
  {
  Col<TestType> x = randi<Col<TestType>>(10000, distr_param(0, 100));
  Col<TestType> y = randi<Col<TestType>>(10000, distr_param(50, 150));

  Col<TestType> z = x * norm(x, 1) / norm(x, 2) % (y + 3);

  REQUIRE( norm(x * norm(x, 1) / norm(x, 2) % (y + 3), 2) == Approx(norm(z, 2)) );
  }



// Make sure norm() with a conv_to works correctly.
TEMPLATE_TEST_CASE(
  "norm_conv_to",
  "[norm]",
  (std::pair<double, float>), (std::pair<float, double>))
  {
  typedef typename TestType::first_type eT1;
  typedef typename TestType::second_type eT2;

  Col<eT1> x = randi<Col<eT1>>(10000, distr_param(0, 100));
  Col<eT2> y = conv_to<Col<eT2>>::from(x);

  for (uword p = 1; p < 11; ++p)
    {
    REQUIRE( norm(conv_to<Col<eT2>>::from(x), p) == Approx(norm(y, p)) );
    }

  REQUIRE( norm(conv_to<Col<eT2>>::from(x), "-inf") == Approx(norm(y, "-inf")) );
  REQUIRE( norm(conv_to<Col<eT2>>::from(x), "inf") == Approx(norm(y, "inf")) );
  REQUIRE( norm(conv_to<Col<eT2>>::from(x), "fro") == Approx(norm(y, "fro")) );
  }
