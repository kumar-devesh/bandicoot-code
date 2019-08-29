// Copyright 2019 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2019 Ryan Curtin (http://www.ratml.org)
// Copyright 2019 National ICT Australia (NICTA)
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

//#define CATCH_CONFIG_MAIN  // catch.hpp will define main()
#define CATCH_CONFIG_RUNNER  // we will define main()
#include "catch.hpp"

int
main(int argc, char** argv)
  {
  std::cout << "Bandicoot version: " << coot::arma_version::as_string() << '\n';

  std::cout << "Run with OpenCL backend:\n";

  coot::coot_rt.backend = CL_BACKEND;
  Catch::Session().run(argc, argv);

  std::cout << "Run with CUDA backend:\n";

  coot::coot_rt.backend = CUDA_BACKEND;
  Catch::Session().run(argc, argv);
  }

