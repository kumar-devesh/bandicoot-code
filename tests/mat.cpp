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

#ifdef COOT_USE_OPENCL

TEMPLATE_TEST_CASE("mat_advanced_cl", "[mat]", u32, s32, u64, s64, float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  if (coot::get_rt().backend != CL_BACKEND)
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 10, distr_param(1, 10));
  cl_mem ptr = x.get_dev_mem(false).cl_mem_ptr;

  Mat<eT> y(ptr, 10, 10);

  REQUIRE( all(all(x == y)) );
  }

#endif



#ifdef COOT_USE_CUDA

TEMPLATE_TEST_CASE("mat_advanced_cuda", "[mat]", u32, s32, u64, s64, float, double)
  {
  typedef TestType eT;

  if (!coot_rt_t::is_supported_type<eT>())
    {
    return;
    }

  if (coot::get_rt().backend != CUDA_BACKEND)
    {
    return;
    }

  Mat<eT> x = randi<Mat<eT>>(10, 10, distr_param(1, 10));
  eT* ptr = x.get_dev_mem(false).cuda_mem_ptr;

  Mat<eT> y(ptr, 10, 10);

  REQUIRE( all(all(x == y)) );
  }

#endif
