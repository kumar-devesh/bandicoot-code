// Copyright 2023 Ryan Curtin (http://ratml.org)
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


#include <complex>

#include "bandicoot_bits/config.hpp"

#undef COOT_USE_WRAPPER

#include "bandicoot_bits/compiler_setup.hpp"
#include "bandicoot_bits/include_opencl.hpp"
#include "bandicoot_bits/include_cuda.hpp"
#include "bandicoot_bits/typedef_elem.hpp"

#ifdef COOT_USE_CUDA

namespace coot
  {
  #include "bandicoot_bits/cuda/def_curand.hpp"

  extern "C"
    {



    //
    // setup/teardown functions
    //



    curandStatus_t wrapper_curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type)
      {
      return curandCreateGenerator(generator, rng_type);
      }



    curandStatus_t wrapper_curandDestroyGenerator(curandGenerator_t generator)
      {
      return curandDestroyGenerator(generator);
      }



    curandStatus_t wrapper_curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed)
      {
      return curandSetPseudoRandomGeneratorSeed(generator, seed);
      }



    //
    // generation functions
    //



    extern curandStatus_t coot_wrapper(curandGenerate)(curandGenerator_t generator,
                                                       unsigned int* outputPtr,
                                                       size_t num)
      {
      return curandGenerate(generator, outputPtr, num);
      }



    extern curandStatus_t coot_wrapper(curandGenerateUniform)(curandGenerator_t generator,
                                                              float* outputPtr,
                                                              size_t num)
      {
      return curandGenerateUniform(generator, outputPtr, num);
      }



    extern curandStatus_t coot_wrapper(curandGenerateUniformDouble)(curandGenerator_t generator,
                                                                    double* outputPtr,
                                                                    size_t num)
      {
      return curandGenerateUniformDouble(generator, outputPtr, num);
      }



    extern curandStatus_t coot_wrapper(curandGenerateNormal)(curandGenerator_t generator,
                                                             float* outputPtr,
                                                             size_t n,
                                                             float mean,
                                                             float stddev)
      {
      return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
      }



    extern curandStatus_t coot_wrapper(curandGenerateNormalDouble)(curandGenerator_t generator,
                                                                   double* outputPtr,
                                                                   size_t n,
                                                                   double mean,
                                                                   double stddev)
      {
      return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
      }



    } // extern "C"
  } // namespace coot

#endif
