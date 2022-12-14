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

#define STR2(A) STR(A)
#define STR(A) #A

// utility functions for compiled-on-the-fly CUDA kernels

inline
std::string
get_cuda_src_preamble()
  {
  std::string source = \

  "#define uint unsigned int\n"
  "#define COOT_FN2(ARG1, ARG2)  ARG1 ## ARG2 \n"
  "#define COOT_FN(ARG1,ARG2) COOT_FN2(ARG1,ARG2)\n"
  "\n"
  // Properties for specific types.
  "__device__ inline bool coot_is_fp(const uint) { return false; } \n"
  "__device__ inline bool coot_is_fp(const int) { return false; } \n"
  "__device__ inline bool coot_is_fp(const size_t) { return false; } \n"
  "__device__ inline bool coot_is_fp(const long) { return false; } \n"
  "__device__ inline bool coot_is_fp(const float) { return true; } \n"
  "__device__ inline bool coot_is_fp(const double) { return true; } \n"
  "\n"
  "extern \"C\" {\n"
  "\n"
  "extern __shared__ char aux_shared_mem[]; \n" // this may be used in some kernels
  "\n"
  // Utility functions to return the correct min/max value for a given type.
  // These constants are not defined in the CUDA compilation environment so we use the host's version.
  "__device__ inline float coot_type_min_float() { return " STR2(FLT_MIN) "; } \n"
  "__device__ inline double coot_type_min_double() { return " STR2(DBL_MIN) "; } \n"
  "__device__ inline float coot_type_max_float() { return " STR2(FLT_MAX) "; } \n"
  "__device__ inline double coot_type_max_double() { return " STR2(DBL_MAX) "; } \n"
  "\n"
  ;

  return source;
  }



inline
std::string
get_cuda_src_epilogue()
  {
  return "}\n";
  }



inline
std::string
read_file(const std::string& filename)
  {
  // This is super hacky!  We eventually need a configuration system to track this.
  const std::string this_file = __FILE__;

  // We need to strip the '_src.hpp' from __FILE__.
  const std::string full_filename = this_file.substr(0, this_file.size() - 8) + "s/" + filename;
  std::ifstream f(full_filename);
  std::string file_contents = "";
  if (!f.is_open())
    {
    std::cout << "Failed to open " << full_filename << " (kernel source)!\n";
    throw std::runtime_error("Cannot open required kernel source.");
    }

  // Allocate memory for file contents.
  f.seekg(0, std::ios::end);
  file_contents.reserve(f.tellg());
  f.seekg(0, std::ios::beg);

  file_contents.assign(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());

  return file_contents;
  }



inline
std::string
get_cuda_oneway_kernel_src()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "accu_warp_reduce.cu",
      "min_warp_reduce.cu",
      "max_warp_reduce.cu"
  };

  std::string result = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "oneway/" + filename;
    result += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : oneway_kernel_id::get_names())
    {
    std::string filename = "oneway/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }




inline
std::string
get_cuda_oneway_real_kernel_src()
  {
  std::string result = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : oneway_real_kernel_id::get_names())
    {
    std::string filename = "oneway_real/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }



inline
std::string
get_cuda_twoway_kernel_src()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  // TODO: adapt so that auxiliary terms have type eT1 not eT2
  // current dogma will be: eT2(x + val) *not* eT2(x) + eT2(val)
  // however, we should also add the overload eT2(x) + val for those situations
  // the operation, I guess, would look like Op<out_eT, eOp<...>, op_conv_to>
  // and we could add an auxiliary out_eT to Op that's 0 by default, but I guess we need bools to indicate usage?
  // they would need to be added to eOp too
  // need to look through Op to see if it's needed there

  std::vector<std::string> aux_function_filenames = {
      "dot_warp_reduce.cu"
  };

  std::string result = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "twoway/" + filename;
    result += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : twoway_kernel_id::get_names())
    {
    std::string filename = "twoway/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }



inline
std::string
get_cuda_threeway_kernel_src()
  {
  std::string result = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : threeway_kernel_id::get_names())
    {
    std::string filename = "threeway/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }
