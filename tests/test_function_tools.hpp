// Copyright 2021 Marcus Edel (http://kurg.org/)
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

using namespace coot;

// Check the values of two matrices.
template<typename MatType>
inline void check_matrices(const MatType& a, const MatType& b, double tolerance = 1e-5)
  {
  REQUIRE(a.n_rows == b.n_rows);
  REQUIRE(a.n_cols == b.n_cols);

  for(uword i = 0; i < a.n_elem; ++i)
    {
    if(std::abs(a(i)) < tolerance / 2)
      {
      REQUIRE(b(i) == Approx(0.0).margin(tolerance / 2.0));
      }
    else
      {
      REQUIRE(a(i) == Approx(b(i)).epsilon(tolerance));
      }
    }
  }
