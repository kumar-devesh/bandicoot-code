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

// utility functions for compiled-on-the-fly CUDA kernels

inline
bool
runtime_t::internal_init()
  {
  coot_extra_debug_sigprint();

  valid = false;

  // TODO: device configuration and setup here
  CUresult result = cuInit(0);
  coot_check_cuda_error(result, "cuda::runtime_t::init(): cuInit() failed");

  result = cuDeviceGet(&cuDevice, 0);
  coot_check_cuda_error(result, "cuda::runtime_t::init(): cuDeviceGet() failed");

  result = cuCtxCreate(&context, 0, cuDevice);
  coot_check_cuda_error(result, "cuda::runtime_t::init(): cuCtxCreate() failed");

  bool status = false;

  status = init_kernels<u32>(u32_kernels, get_cuda_kernel_src(), get_cuda_kernel_names());
  if (status == false) { coot_debug_warn("coot_cuda_rt: couldn't set up CUDA u32 kernels"); }

  status = init_kernels<s32>(s32_kernels, get_cuda_kernel_src(), get_cuda_kernel_names());
  if (status == false) { coot_debug_warn("coot_cuda_rt: couldn't set up CUDA s32 kernels"); }

  status = init_kernels<u64>(u64_kernels, get_cuda_kernel_src(), get_cuda_kernel_names());
  if (status == false) { coot_debug_warn("coot_cuda_rt: couldn't set up CUDA u64 kernels"); }

  status = init_kernels<s64>(s64_kernels, get_cuda_kernel_src(), get_cuda_kernel_names());
  if (status == false) { coot_debug_warn("coot_cuda_rt: couldn't set up CUDA s64 kernels"); }

  status = init_kernels<double>(d_kernels, get_cuda_kernel_src(), get_cuda_kernel_names());
  if (status == false) { coot_debug_warn("coot_cuda_rt: couldn't set up CUDA double kernels"); }

  status = init_kernels<float>(f_kernels, get_cuda_kernel_src(), get_cuda_kernel_names());
  if (status == false) { coot_debug_warn("coot_cuda_rt: couldn't set up CUDA float kernels"); }

  valid = true;

  return true;
  }

template<typename eT>
inline
bool
runtime_t::init_kernels(std::vector<CUfunction>& kernels, const std::string& source, const std::vector<std::string>& names)
  {
  // We'll use NVRTC to compile each of the kernels we need on the fly.

  nvrtcProgram prog;
  nvrtcResult result = nvrtcCreateProgram(
      &prog,          // program holder
      source.c_str(), // buffer with source
      "coot_kernels", // name
      0,              // numHeaders
      NULL,           // headers
      NULL);          // includeNames

  if (result != NVRTC_SUCCESS)
    {
    std::cout << "nvrtcCreateProgram() failed with error " << result << "\n";
    }

  // Construct the macros that we need.
  std::string prefix;
  std::string macro1;
  std::string macro2;
  std::string macro3;

  if (is_same_type<eT, u32>::yes)
    {
    prefix = "u32_";
    macro2 = "-D eT=uint";
    macro3 = "-D promoted_eT=float";
    }
  else if (is_same_type<eT, s32>::yes)
    {
    prefix = "s32_";
    macro2 = "-D eT=int";
    macro3 = "-D promoted_eT=float";
    }
  else if (is_same_type<eT, u64>::yes)
    {
    prefix = "u64_";
    macro2 = "-D eT=size_t";
    macro3 = "-D promoted_eT=float";
    }
  else if (is_same_type<eT, s64>::yes)
    {
    prefix = "s64_";
    macro2 = "-D eT=long";
    macro3 = "-D promoted_eT=float";
    }
  else if (is_same_type<eT, float>::yes)
    {
    prefix = "f_";
    macro2 = "-D eT=float";
    macro3 = "-D promoted_eT=float";
    }
  else if (is_same_type<eT, double>::yes)
    {
    prefix = "d_";
    macro2 = "-D eT=double";
    macro3 = "-D promoted_eT=double";
    }

  macro1 = "-D PREFIX=" + prefix;

  const char *opts[] = {"--gpu-architecture=compute_30",
                        "--fmad=false",
                        macro1.c_str(),
                        macro2.c_str(),
                        macro3.c_str(),
                        "-D UWORD=size_t" /* TODO: what about 32-bit? */};

  result  = nvrtcCompileProgram(prog,  // prog
                                6,     // numOptions
                                opts); // options
  if (result != NVRTC_SUCCESS)
    {
    std::cout << "nvrtcCompileProgram() failed with error " << result << "\n";
    }

  size_t logSize;
  result = (nvrtcGetProgramLogSize(prog, &logSize));
  if (result != NVRTC_SUCCESS)
    {
    std::cout << "nvrtcGetProgramLogSize() failed with error " << result <<
"\n";
    }
  char *log = new char[logSize];
  result = (nvrtcGetProgramLog(prog, log));
  if (result != NVRTC_SUCCESS)
    {
    std::cout << "nvrtcGetProgramLog() failed with error " << result << "\n";
    std::cout << log << '\n';
    }

  // Obtain PTX from the program.
  size_t ptxSize;
  result = nvrtcGetPTXSize(prog, &ptxSize);
  if (result != NVRTC_SUCCESS)
    {
    std::cout << "nvrtcGetPTXSize() failed with error " << result << "\n";
    }
  char *ptx = new char[ptxSize];
  result = nvrtcGetPTX(prog, ptx);
  if (result != NVRTC_SUCCESS)
    {
    std::cout << "nvrtcGetPTX() failed with error " << result << "\n";
    std::cout << "ptx is " << ptx << "\n";
    }

  CUresult result2 = cuInit(0);
  CUmodule module;
  result2 = cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
  coot_check_cuda_error(result2, "cuda::runtime_t::init(): cuModuleLoadDataEx() failed");

  // Now that everything is compiled, unpack the results into individual kernels
  // that we can access.
  const uword n_kernels = names.size();

  kernels.resize(n_kernels);

  for (uword i = 0; i < n_kernels; ++i)
    {
    const std::string name = prefix + names.at(i);
    result2 = cuModuleGetFunction(&kernels.at(i), module, name.c_str());
    coot_check_cuda_error(result2, "cuda::runtime_t::init(): cuModuleGetFunction() failed for function " + name);
    }

  return true;
  }



template<typename eT>
inline
CUfunction&
runtime_t::get_kernel(const kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (valid == false), "coot_cuda_rt not valid" );

       if(is_same_type<eT,u32   >::yes)  { return u32_kernels.at(num); }
  else if(is_same_type<eT,s32   >::yes)  { return s32_kernels.at(num); }
  else if(is_same_type<eT,u64   >::yes)  { return u64_kernels.at(num); }
  else if(is_same_type<eT,s64   >::yes)  { return s64_kernels.at(num); }
  else if(is_same_type<eT,float >::yes)  { return   f_kernels.at(num); }
  else if(is_same_type<eT,double>::yes)  { return   d_kernels.at(num); }
  else { coot_debug_check(true, "unsupported element type" ); }
  }



template<typename eT>
inline
eT*
runtime_t::acquire_memory(const uword n_elem)
  {
  void* result;
  cudaError_t error = cudaMalloc(&result, sizeof(eT) * n_elem);

  coot_check_cuda_error(error, "cuda::acquire_memory(): couldn't allocate memory");

  return (eT*) result;
  }

template<typename eT>
inline
void
runtime_t::release_memory(eT* cuda_mem)
  {
  if(cuda_mem)
    {
    cudaError_t error = cudaFree(cuda_mem);

    coot_check_cuda_error(error, "cuda::release_memory(): couldn't free memory");
    }
  }
