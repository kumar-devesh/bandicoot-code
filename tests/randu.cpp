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
#include "catch.hpp"

using namespace coot;

TEST_CASE("randu_1", "[randu]")
{
  //randi(3, 3, distr_param(0, 1));
  mat f = randu<mat>(3, 3);

  f.print();

  mat g = randu<mat>(3, 3);
  g.print();
/*   mat src(4, 1); */
/*   src.randn(); */
/*   mat dst(4, 1); */
/*   dst.randu(); */

/*   src.print(); */
/*   dst.print(); */

/*   // Copy the first three elements from src to dst. */
/*   cudaMemcpy((double*)dst.get_dev_mem().cuda_mem_ptr, src.get_dev_mem().cuda_mem_ptr, sizeof(double) * 3, cudaMemcpyDeviceToDevice); */
/*   // Copy first element from src to last element in dst. */
/*   cudaMemcpy((double*)dst.get_dev_mem().cuda_mem_ptr + 3, src.get_dev_mem().cuda_mem_ptr, sizeof(double) * 1, cudaMemcpyDeviceToDevice); */

/*   src.print(); */
/*   dst.print(); */
}
