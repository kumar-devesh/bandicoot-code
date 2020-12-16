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
runtime_t::init(const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (wanted_platform != 0), "cuda::runtime_t::init(): wanted_platform must be 0 for the CUDA backend" );

  valid = false;

  CUresult result = cuInit(0);
  coot_check_cuda_error(result, "cuda::runtime_t::init(): cuInit() failed");

  int device_count = 0;
  result = cuDeviceGetCount(&device_count);
  coot_check_cuda_error(result, "cuda::runtime_t::init(): cuDeviceGetCount() failed");

  // Ensure that the desired device is within the range of devices we have.
  // TODO: better error message?
  coot_debug_check( ((int) wanted_device >= device_count), "cuda::runtime_t::init(): invalid wanted_device" );

  result = cuDeviceGet(&cuDevice, wanted_device);
  coot_check_cuda_error(result, "cuda::runtime_t::init(): cuDeviceGet() failed");

  result = cuCtxCreate(&context, 0, cuDevice);
  coot_check_cuda_error(result, "cuda::runtime_t::init(): cuCtxCreate() failed");

  // NOTE: it seems size_t will have the same size on the device and host;
  // given the definition of uword, we will assume uword on the host is equivalent
  // to size_t on the device.
  //
  // NOTE: float will also have the same size as the host (generally 32 bits)
  cudaError_t result2 = cudaGetDeviceProperties(&dev_prop, wanted_device);
  coot_check_cuda_error(result2, "cuda::runtime_t::init(): couldn't get device properties");

  std::vector<std::pair<std::string, CUfunction*>> name_map;
  type_to_dev_string type_map;
  std::string src =
      get_cuda_src_preamble() +
      rt_common::get_three_elem_kernel_src(threeway_kernels, get_cuda_threeway_kernel_src(), threeway_kernel_id::get_names(), name_map, type_map) +
      rt_common::get_two_elem_kernel_src(twoway_kernels, get_cuda_twoway_kernel_src(), twoway_kernel_id::get_names(), "", name_map, type_map) +
      rt_common::get_one_elem_kernel_src(oneway_kernels, get_cuda_oneway_kernel_src(), oneway_kernel_id::get_names(), "", name_map, type_map) +
      get_cuda_src_epilogue();

  bool status = compile_kernels(src, name_map);
  if (status == false) { coot_debug_warn("cuda::runtime_t::init(): couldn't set up CUDA kernels"); }

  // Initialize RNG struct.
  curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);

  // Initialize cuBLAS.
  cublasCreate(&cublas_handle);

  valid = true;

  return true;

  // TODO: destroy context in destructor
  }



inline
bool
runtime_t::compile_kernels(const std::string& source,
                           std::vector<std::pair<std::string, CUfunction*>>& names)
  {
  // We'll use NVRTC to compile each of the kernels we need on the fly.
  nvrtcProgram prog;
  nvrtcResult result = nvrtcCreateProgram(
      &prog,          // CUDA runtime compilation program
      source.c_str(), // CUDA program source
      "coot_kernels", // CUDA program name
      0,              // number of headers used
      NULL,           // sources of the headers
      NULL);          // name of each header
  coot_check_nvrtc_error(result, "cuda::runtime_t::init_kernels(): nvrtcCreateProgram() failed");

  std::vector<const char*> opts =
    {
    "--fmad=false",
    "-D UWORD=size_t",
    "--relocatable-device-code=true",
    "--restrict"
    };

  // Get compute capabilities.
  int major, minor = 0;
  CUresult result2 = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
  coot_check_cuda_error(result2, "cuda::runtime_t::init_kernels(): cuDeviceGetAttribute() failed");
  result2 = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
  coot_check_cuda_error(result2, "cuda::runtime_t::init_kernels(): cuDeviceGetAttribute() failed");

  std::stringstream gpu_arch_opt;
  gpu_arch_opt << "--gpu-architecture=compute_" << major << minor;
  const std::string& gpu_arch_opt_tmp = gpu_arch_opt.str();
  opts.push_back(gpu_arch_opt_tmp.c_str());

  result = nvrtcCompileProgram(prog,         // CUDA runtime compilation program
                               opts.size(),  // number of compile options
                               opts.data()); // compile options

  // If compilation failed, display what went wrong.  The NVRTC outputs aren't
  // always very helpful though...
  if (result != NVRTC_SUCCESS)
    {
    size_t logSize;
    result = (nvrtcGetProgramLogSize(prog, &logSize));
    coot_check_nvrtc_error(result, "cuda::runtime_t::init_kernels(): nvrtcGetProgramLogSize() failed");

    char *log = new char[logSize];
    result = (nvrtcGetProgramLog(prog, log));
    coot_check_nvrtc_error(result, "cuda::runtime_t::init_kernels(): nvrtcGetProgramLog() failed");

    coot_stop_runtime_error("cuda::runtime_t::init_kernels(): compilation failed", std::string(log));
    }

  // Obtain PTX from the program.
  size_t ptxSize;
  result = nvrtcGetPTXSize(prog, &ptxSize);
  coot_check_nvrtc_error(result, "cuda::runtime_t::init_kernels(): nvrtcGetPTXSize() failed");

  char *ptx = new char[ptxSize];
  result = nvrtcGetPTX(prog, ptx);
  coot_check_nvrtc_error(result, "cuda::runtime_t::init_kernels(): nvrtcGetPTX() failed");

  result2 = cuInit(0);
  CUmodule module;
  result2 = cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
  coot_check_cuda_error(result2, "cuda::runtime_t::init_kernels(): cuModuleLoadDataEx() failed");

  // Now that everything is compiled, unpack the results into individual kernels
  // that we can access.
  for (uword i = 0; i < names.size(); ++i)
    {
    result2 = cuModuleGetFunction(names.at(i).second, module, names.at(i).first.c_str());
    coot_check_cuda_error(result2, "cuda::runtime_t::init_kernels(): cuModuleGetFunction() failed for function " + names.at(i).first);
    }

  return true;
  }



inline
runtime_t::~runtime_t()
  {
  // Clean up cuBLAS handle.
  cublasDestroy(cublas_handle);
  }



template<typename eT>
inline
const CUfunction&
runtime_t::get_kernel(const oneway_kernel_id::enum_id num)
  {
  return get_kernel<eT>(oneway_kernels, num);
  }



template<typename eT1, typename eT2>
inline
const CUfunction&
runtime_t::get_kernel(const twoway_kernel_id::enum_id num)
  {
  return get_kernel<eT1, eT2>(twoway_kernels, num);
  }



template<typename eT1, typename eT2, typename eT3>
inline
const CUfunction&
runtime_t::get_kernel(const threeway_kernel_id::enum_id num)
  {
  return get_kernel<eT1, eT2, eT3>(threeway_kernels, num);
  }



template<typename eT1, typename... eTs, typename HeldType, typename EnumType>
inline
const CUfunction&
runtime_t::get_kernel(const rt_common::kernels_t<HeldType>& k, const EnumType num)
  {
  coot_extra_debug_sigprint();

       if(is_same_type<eT1,u32   >::yes)  { return get_kernel<eTs...>(k.u32_kernels, num); }
  else if(is_same_type<eT1,s32   >::yes)  { return get_kernel<eTs...>(k.s32_kernels, num); }
  else if(is_same_type<eT1,u64   >::yes)  { return get_kernel<eTs...>(k.u64_kernels, num); }
  else if(is_same_type<eT1,s64   >::yes)  { return get_kernel<eTs...>(k.s64_kernels, num); }
  else if(is_same_type<eT1,float >::yes)  { return get_kernel<eTs...>(  k.f_kernels, num); }
  else if(is_same_type<eT1,double>::yes)  { return get_kernel<eTs...>(  k.d_kernels, num); }
  else { coot_debug_check(true, "unsupported element type" ); }
  }



template<typename eT, typename EnumType>
inline
const CUfunction&
runtime_t::get_kernel(const rt_common::kernels_t<std::vector<CUfunction>>& k, const EnumType num)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (valid == false), "cuda::runtime_t not valid" );

       if(is_same_type<eT,u32   >::yes)  { return k.u32_kernels.at(num); }
  else if(is_same_type<eT,s32   >::yes)  { return k.s32_kernels.at(num); }
  else if(is_same_type<eT,u64   >::yes)  { return k.u64_kernels.at(num); }
  else if(is_same_type<eT,s64   >::yes)  { return k.s64_kernels.at(num); }
  else if(is_same_type<eT,float >::yes)  { return   k.f_kernels.at(num); }
  else if(is_same_type<eT,double>::yes)  { return   k.d_kernels.at(num); }
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



inline
void
runtime_t::synchronise()
  {
  cuCtxSynchronize();
  }
