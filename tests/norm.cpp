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



// Compare with Armadillo's implementation and ensure that the result is at least close.
TEMPLATE_TEST_CASE("norm_basic", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(1000, distr_param(0, 50));
  arma::Col<eT> x_cpu(x);

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
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x;

  for (uword p = 1; p < 11; ++p)
    {
    REQUIRE( norm(x, p) == eT(0) );
    }

  REQUIRE( norm(x, "-inf") == eT(0) );
  REQUIRE( norm(x, "inf") == eT(0) );
  REQUIRE( norm(x, "fro") == eT(0) );
  }



TEMPLATE_TEST_CASE("single_element_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Row<eT> x(1);
  x[0] = eT(10);

  for (uword p = 1; p < 11; ++p)
    {
    REQUIRE( norm(x, p) == Approx(eT(10)) );
    }

  REQUIRE( norm(x, "-inf") == Approx(eT(10)) );
  REQUIRE( norm(x, "inf") == Approx(eT(10)) );
  REQUIRE( norm(x, "fro") == Approx(eT(10)) );
  }



TEMPLATE_TEST_CASE("expr_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Row<eT> x = randi<Row<eT>>(1000, distr_param(0, 50));
  Row<eT> z = vectorise(repmat(trans(x + 3), 2, 2)).t();

  for (uword p = 1; p < 11; ++p)
    {
    REQUIRE( norm(vectorise(repmat(trans(x + 3), 2, 2).t()), p) == Approx(norm(z, p)) );
    }

  REQUIRE( norm(vectorise(repmat(trans(x + 3), 2, 2).t()), "-inf") == Approx(norm(z, "-inf")) );
  REQUIRE( norm(vectorise(repmat(trans(x + 3), 2, 2).t()), "inf") == Approx(norm(z, "inf")) );
  REQUIRE( norm(vectorise(repmat(trans(x + 3), 2, 2).t()), "fro") == Approx(norm(z, "fro")) );
  }



// Attempt to get norm2 to overflow by using very large elements.
TEMPLATE_TEST_CASE("norm2_overflow", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(10000);
  Col<eT> f = randu<Col<eT>>(10000) * 0.9;
  x = f * std::numeric_limits<eT>::max();

  const eT x_max = max(x);
  Col<eT> x_normalized = x / x_max;

  REQUIRE( norm(x, 2) == Approx(norm(x_normalized, 2) * x_max) );
  }



// Attempt to get norm2 to underflow by using very small elements.
TEMPLATE_TEST_CASE("norm2_underflow", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x(10000);
  Col<eT> f = randi<Col<eT>>(10000, distr_param(1, 25));
  x = f * std::numeric_limits<eT>::min();

  const eT x_max = max(x);
  Col<eT> x_normalized = x / x_max;

  REQUIRE( norm(x, 2) == Approx(norm(x_normalized, 2) * x_max) );
  }



// Try a complicated expression with norm() inside of it too.
TEMPLATE_TEST_CASE("double_norm_expr", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(10000, distr_param(0, 100));
  Col<eT> y = randi<Col<eT>>(10000, distr_param(50, 150));

  Col<eT> z = x * norm(x, 1) / norm(x, 2) % (y + 3);

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

  if (!coot_rt_t::is_supported_type<eT1>() || !coot_rt_t::is_supported_type<eT2>())
    {
    return;
    }


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



// Compute norms of a very large vector (hopefully enough to trigger a second reduce).
TEMPLATE_TEST_CASE("large_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Col<eT> x = randi<Col<eT>>(1000000, distr_param(0, 100));
  arma::Col<eT> x_cpu(x);

  REQUIRE( norm(x, 1) == Approx(norm(x_cpu, 1)).epsilon(0.01) );
  REQUIRE( norm(x, 2) == Approx(norm(x_cpu, 2)).epsilon(0.01) );
  REQUIRE( norm(x, 3) == Approx(norm(x_cpu, 3)).epsilon(0.01) );
  REQUIRE( norm(x, 4) == Approx(norm(x_cpu, 4)).epsilon(0.01) );
  REQUIRE( norm(x, 5) == Approx(norm(x_cpu, 5)).epsilon(0.01) );
  REQUIRE( norm(x, "-inf") == Approx(norm(x_cpu, "-inf")).epsilon(0.01) );
  REQUIRE( norm(x, "inf") == Approx(norm(x_cpu, "inf")).epsilon(0.01) );
  REQUIRE( norm(x, "fro") == Approx(norm(x_cpu, "fro")).epsilon(0.01) );
  }



// Compute matrix norms of small random matrix.
TEMPLATE_TEST_CASE("small_matrix_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(10, 10);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( norm(x, 1) == Approx(norm(x_cpu, 1)).epsilon(0.01) );
  REQUIRE( norm(x, 2) == Approx(norm(x_cpu, 2)).epsilon(0.01) );
  REQUIRE( norm(x, "inf") == Approx(norm(x_cpu, "inf")).epsilon(0.01) );
  REQUIRE( norm(x, "fro") == Approx(norm(x_cpu, "fro")).epsilon(0.01) );
  }



// Compute matrix norms of large random matrix.
TEMPLATE_TEST_CASE("large_matrix_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(1000, 1000);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( norm(x, 1) == Approx(norm(x_cpu, 1)).epsilon(0.01) );
  REQUIRE( norm(x, 2) == Approx(norm(x_cpu, 2)).epsilon(0.01) );
  REQUIRE( norm(x, "inf") == Approx(norm(x_cpu, "inf")).epsilon(0.01) );
  REQUIRE( norm(x, "fro") == Approx(norm(x_cpu, "fro")).epsilon(0.01) );
  }



// Compute matrix norms of tall, skinny matrix.
TEMPLATE_TEST_CASE("tall_matrix_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randn<Mat<eT>>(1000, 10);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( norm(x, 1) == Approx(norm(x_cpu, 1)).epsilon(0.01) );
  REQUIRE( norm(x, 2) == Approx(norm(x_cpu, 2)).epsilon(0.01) );
  REQUIRE( norm(x, "inf") == Approx(norm(x_cpu, "inf")).epsilon(0.01) );
  REQUIRE( norm(x, "fro") == Approx(norm(x_cpu, "fro")).epsilon(0.01) );
  }



// Compute matrix norms of short, wide matrix.
TEMPLATE_TEST_CASE("wide_matrix_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randn<Mat<eT>>(10, 1000);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( norm(x, 1) == Approx(norm(x_cpu, 1)).epsilon(0.01) );
  REQUIRE( norm(x, 2) == Approx(norm(x_cpu, 2)).epsilon(0.01) );
  REQUIRE( norm(x, "inf") == Approx(norm(x_cpu, "inf")).epsilon(0.01) );
  REQUIRE( norm(x, "fro") == Approx(norm(x_cpu, "fro")).epsilon(0.01) );
  }



// Compute matrix norms of empty matrix.
TEMPLATE_TEST_CASE("empty_matrix_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x;

  REQUIRE( norm(x, 1) == Approx(eT(0)).margin(1e-6) );
  REQUIRE( norm(x, 2) == Approx(eT(0)).margin(1e-6) );
  REQUIRE( norm(x, "inf") == Approx(eT(0)).margin(1e-6) );
  REQUIRE( norm(x, "fro") == Approx(eT(0)).margin(1e-6) );
  }



// Ensure invalid norm type throws an exception.
TEMPLATE_TEST_CASE("invalid_matrix_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(10, 10);

  std::streambuf* orig_cerr_buf = std::cerr.rdbuf();
  std::cerr.rdbuf(NULL);
  eT out;
  REQUIRE_THROWS( out = norm(x, "-inf") );
  std::cerr.rdbuf(orig_cerr_buf);
  }



// Test subview norms over a single column.
TEMPLATE_TEST_CASE("subview_col_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(250, 250);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( norm(x.submat(10, 10, 10, 150), 1)      == Approx(norm(x_cpu.submat(10, 10, 10, 150), 1))      );
  REQUIRE( norm(x.submat(10, 10, 10, 150), 2)      == Approx(norm(x_cpu.submat(10, 10, 10, 150), 2))      );
  REQUIRE( norm(x.submat(10, 10, 10, 150), 3)      == Approx(norm(x_cpu.submat(10, 10, 10, 150), 3))      );
  REQUIRE( norm(x.submat(10, 10, 10, 150), 5)      == Approx(norm(x_cpu.submat(10, 10, 10, 150), 5))      );
  REQUIRE( norm(x.submat(10, 10, 10, 150), 10)     == Approx(norm(x_cpu.submat(10, 10, 10, 150), 10))     );
  REQUIRE( norm(x.submat(10, 10, 10, 150), "inf")  == Approx(norm(x_cpu.submat(10, 10, 10, 150), "inf"))  );
  REQUIRE( norm(x.submat(10, 10, 10, 150), "-inf") == Approx(norm(x_cpu.submat(10, 10, 10, 150), "-inf")) );
  REQUIRE( norm(x.submat(10, 10, 10, 150), "fro")  == Approx(norm(x_cpu.submat(10, 10, 10, 150), "fro"))  );
  }



// Test subview norms over a single row.
TEMPLATE_TEST_CASE("subview_row_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(250, 250);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( norm(x.submat(10, 10, 150, 10), 1)      == Approx(norm(x_cpu.submat(10, 10, 150, 10), 1))      );
  REQUIRE( norm(x.submat(10, 10, 150, 10), 2)      == Approx(norm(x_cpu.submat(10, 10, 150, 10), 2))      );
  REQUIRE( norm(x.submat(10, 10, 150, 10), 3)      == Approx(norm(x_cpu.submat(10, 10, 150, 10), 3))      );
  REQUIRE( norm(x.submat(10, 10, 150, 10), 5)      == Approx(norm(x_cpu.submat(10, 10, 150, 10), 5))      );
  REQUIRE( norm(x.submat(10, 10, 150, 10), 10)     == Approx(norm(x_cpu.submat(10, 10, 150, 10), 10))     );
  REQUIRE( norm(x.submat(10, 10, 150, 10), "inf")  == Approx(norm(x_cpu.submat(10, 10, 150, 10), "inf"))  );
  REQUIRE( norm(x.submat(10, 10, 150, 10), "-inf") == Approx(norm(x_cpu.submat(10, 10, 150, 10), "-inf")) );
  REQUIRE( norm(x.submat(10, 10, 150, 10), "fro")  == Approx(norm(x_cpu.submat(10, 10, 150, 10), "fro"))  );
  }



// Test subview norms over a vectorised subview.
TEMPLATE_TEST_CASE("subview_vectorised_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(250, 250);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( norm(vectorise(x.submat(10, 10, 150, 150)), 1)      == Approx(norm(vectorise(x_cpu.submat(10, 10, 150, 150)), 1))      );
  REQUIRE( norm(vectorise(x.submat(10, 10, 150, 150)), 2)      == Approx(norm(vectorise(x_cpu.submat(10, 10, 150, 150)), 2))      );
  REQUIRE( norm(vectorise(x.submat(10, 10, 150, 150)), 3)      == Approx(norm(vectorise(x_cpu.submat(10, 10, 150, 150)), 3))      );
  REQUIRE( norm(vectorise(x.submat(10, 10, 150, 150)), 5)      == Approx(norm(vectorise(x_cpu.submat(10, 10, 150, 150)), 5))      );
  REQUIRE( norm(vectorise(x.submat(10, 10, 150, 150)), 10)     == Approx(norm(vectorise(x_cpu.submat(10, 10, 150, 150)), 10))     );
  REQUIRE( norm(vectorise(x.submat(10, 10, 150, 150)), "inf")  == Approx(norm(vectorise(x_cpu.submat(10, 10, 150, 150)), "inf"))  );
  REQUIRE( norm(vectorise(x.submat(10, 10, 150, 150)), "-inf") == Approx(norm(vectorise(x_cpu.submat(10, 10, 150, 150)), "-inf")) );
  REQUIRE( norm(vectorise(x.submat(10, 10, 150, 150)), "fro")  == Approx(norm(vectorise(x_cpu.submat(10, 10, 150, 150)), "fro"))  );
  }



// Test subview matrix norms.
TEMPLATE_TEST_CASE("subview_matrix_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(250, 250);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( norm(x.submat(10, 10, 150, 150), 1)     == Approx(norm(x_cpu.submat(10, 10, 150, 150), 1)) );
  REQUIRE( norm(x.submat(10, 10, 150, 150), 2)     == Approx(norm(x_cpu.submat(10, 10, 150, 150), 2)) );
  REQUIRE( norm(x.submat(10, 10, 150, 150), "inf") == Approx(norm(x_cpu.submat(10, 10, 150, 150), "inf")) );
  REQUIRE( norm(x.submat(10, 10, 150, 150), "fro") == Approx(norm(x_cpu.submat(10, 10, 150, 150), "fro")) );
  }



// Test subview norms when the subview is the size of the full matrix.
TEMPLATE_TEST_CASE("full_subview_matrix_norm", "[norm]", double, float)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  Mat<eT> x = randu<Mat<eT>>(250, 250);
  arma::Mat<eT> x_cpu(x);

  REQUIRE( norm(x.submat(0, 0, 249, 249), 1)     == Approx(norm(x_cpu.submat(0, 0, 249, 249), 1)) );
  REQUIRE( norm(x.submat(0, 0, 249, 249), 2)     == Approx(norm(x_cpu.submat(0, 0, 249, 249), 2)) );
  REQUIRE( norm(x.submat(0, 0, 249, 249), "inf") == Approx(norm(x_cpu.submat(0, 0, 249, 249), "inf")) );
  REQUIRE( norm(x.submat(0, 0, 249, 249), "fro") == Approx(norm(x_cpu.submat(0, 0, 249, 249), "fro")) );
  }



TEST_CASE("subview_norm_1", "[norm]")
  {
  fmat x = randu<fmat>(10, 10);
  arma::fmat x_cpu(x);
  REQUIRE( norm(x.submat(1, 1, 5, 1), "-inf") == Approx(norm(x_cpu.submat(1, 1, 5, 1), "-inf")) );
  }
