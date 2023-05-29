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

TEMPLATE_TEST_CASE("det_arma_comparison", "[det]", float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  for (uword p = 6; p < 10; ++p)
    {
    const size_t dim = std::pow((uword) 2, p);

    Mat<eT> x(dim, dim);
    x.randu();

    arma::Mat<eT> x_cpu(x);

    const eT det_val = det(x);
    const eT arma_det_val = arma::det(x_cpu);

    REQUIRE( det_val == Approx(arma_det_val) );
    }
  }
