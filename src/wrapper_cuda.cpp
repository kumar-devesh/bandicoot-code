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
  #include "bandicoot_bits/cuda/def_cuda.hpp"

  extern "C"
    {



    //
    // setup functions
    //



    CUresult wrapper_cuInit(unsigned int flags)
      {
      return cuInit(flags);
      }



    CUresult wrapper_cuDeviceGetCount(int* count)
      {
      return cuDeviceGetCount(count);
      }



    CUresult wrapper_cuDeviceGet(CUdevice* device, int ordinal)
      {
      return cuDeviceGet(device, ordinal);
      }



    CUresult wrapper_cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
      {
      return cuDeviceGetAttribute(pi, attrib, dev);
      }



    CUresult wrapper_cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev)
      {
      return cuCtxCreate(pctx, flags, dev);
      }



    CUresult wrapper_cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues)
      {
      return cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
      }



    CUresult wrapper_cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name)
      {
      return cuModuleGetFunction(hfunc, hmod, name);
      }



    cudaError_t wrapper_cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
      {
      return cudaGetDeviceProperties(prop, device);
      }



    cudaError_t wrapper_cudaRuntimeGetVersion(int* runtimeVersion)
      {
      return cudaRuntimeGetVersion(runtimeVersion);
      }



    //
    // memory handling
    //


    cudaError_t wrapper_cudaMalloc(void** devPtr, size_t size)
      {
      return cudaMalloc(devPtr, size);
      }



    cudaError_t wrapper_cudaFree(void* devPtr)
      {
      return cudaFree(devPtr);
      }



    cudaError_t wrapper_cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
      {
      return cudaMemcpy(dst, src, count, kind);
      }



    cudaError_t wrapper_cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)
      {
      return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
      }



    //
    // running kernels
    //



    CUresult wrapper_cuLaunchKernel(CUfunction f,
                                    unsigned int gridDimX,
                                    unsigned int gridDimY,
                                    unsigned int gridDimZ,
                                    unsigned int blockDimX,
                                    unsigned int blockDimY,
                                    unsigned int blockDimZ,
                                    unsigned int sharedMemBytes,
                                    CUstream hStream,
                                    void** kernelParams,
                                    void** extra)
      {
      return cuLaunchKernel(f,
                            gridDimX,
                            gridDimY,
                            gridDimZ,
                            blockDimX,
                            blockDimY,
                            blockDimZ,
                            sharedMemBytes,
                            hStream,
                            kernelParams,
                            extra);
      }



    //
    // synchronisation
    //


    CUresult wrapper_cuCtxSynchronize()
      {
      return cuCtxSynchronize();
      }



    } // extern "C"
  } // namespace coot

#endif
