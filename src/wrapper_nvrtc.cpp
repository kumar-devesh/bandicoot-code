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
  #include "bandicoot_bits/cuda/def_nvrtc.hpp"

  extern "C"
    {



    nvrtcResult wrapper_nvrtcCreateProgram(nvrtcProgram* prog,
                                           const char* src,
                                           const char* name,
                                           int numHeaders,
                                           const char** headers,
                                           const char** includeNames)
      {
      return nvrtcCreateProgram(prog, src, name, numHeaders, headers, includeNames);
      }



    nvrtcResult wrapper_nvrtcCompileProgram(nvrtcProgram prog,
                                            int numOptions,
                                            const char** options)
      {
      return nvrtcCompileProgram(prog, numOptions, options);
      }



    nvrtcResult wrapper_nvrtcGetProgramLogSize(nvrtcProgram prog,
                                               size_t* logSizeRet)
      {
      return nvrtcGetProgramLogSize(prog, logSizeRet);
      }



    nvrtcResult wrapper_nvrtcGetProgramLog(nvrtcProgram prog,
                                           char* log)
      {
      return nvrtcGetProgramLog(prog, log);
      }



    nvrtcResult wrapper_nvrtcGetCUBINSize(nvrtcProgram prog,
                                          size_t* cubinSizeRet)
      {
      return nvrtcGetCUBINSize(prog, cubinSizeRet);
      }



    nvrtcResult wrapper_nvrtcGetCUBIN(nvrtcProgram prog,
                                      char* cubin)
      {
      return nvrtcGetCUBIN(prog, cubin);
      }



    } // extern "C"
  } // namespace coot

#endif
