// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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



struct kernel_src
  {
  static inline const std::string&  get_src_preamble();

  static inline const std::string&  get_zeroway_source();
  static inline       std::string  init_zeroway_source();

  static inline const std::string&  get_oneway_source();
  static inline       std::string  init_oneway_source();

  static inline const std::string&  get_oneway_real_source();
  static inline       std::string  init_oneway_real_source();

  static inline const std::string&  get_oneway_integral_source();
  static inline       std::string  init_oneway_integral_source();

  static inline const std::string&  get_twoway_source();
  static inline       std::string  init_twoway_source();

  static inline const std::string&  get_threeway_source();
  static inline       std::string  init_threeway_source();

  static inline const std::string&  get_magma_real_source();
  static inline       std::string  init_magma_real_source();

  static inline const std::string&  get_src_epilogue();
  };



inline
const std::string&
kernel_src::get_src_preamble()
  {
  char u32_max[32];
  char u64_max[32];
  snprintf(u32_max, 32, "%llu", (unsigned long long) std::numeric_limits<u32>::max());
  snprintf(u64_max, 32, "%llu", (unsigned long long) std::numeric_limits<u64>::max());

  char s32_min[32];
  char s64_min[32];
  snprintf(s32_min, 32, "%llu", (unsigned long long) std::numeric_limits<s32>::min());
  snprintf(s64_min, 32, "%llu", (unsigned long long) std::numeric_limits<s64>::min());

  char s32_max[32];
  char s64_max[32];
  snprintf(s32_max, 32, "%llu", (unsigned long long) std::numeric_limits<s32>::max());
  snprintf(s64_max, 32, "%llu", (unsigned long long) std::numeric_limits<s64>::max());

  static const std::string source = \

  "#ifdef cl_khr_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_khr_pragma_unroll : enable \n"
  "#endif \n"
  "#ifdef cl_amd_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_amd_pragma_unroll : enable \n"
  "#endif \n"
  "#ifdef cl_nv_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable \n"
  "#endif \n"
  "#ifdef cl_intel_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_intel_pragma_unroll : enable \n"
  "#endif \n"
  "\n"
  "#define COOT_FN2(ARG1,ARG2)  ARG1 ## ARG2 \n"
  "#define COOT_FN(ARG1,ARG2) COOT_FN2(ARG1,ARG2) \n"
  "\n"
  "#define COOT_FN_3_2(ARG1,ARG2,ARG3) ARG1 ## ARG2 ## ARG3 \n"
  "#define COOT_FN_3(ARG1,ARG2,ARG3) COOT_FN_3_2(ARG1,ARG2,ARG3) \n"
  "\n"
  // Utility functions to return the correct min/max value for a given type.
  "inline uint coot_type_min_uint() { return 0; } \n"
  "inline ulong coot_type_min_ulong() { return 0; } \n"
  "inline uint coot_type_max_uint() { return " + std::string(u32_max) + "; } \n"
  "inline ulong coot_type_max_ulong() { return " + std::string(u64_max) + "; } \n"
  "\n"
  "inline int coot_type_min_int() { return " + std::string(s32_min) + "; } \n"
  "inline long coot_type_min_long() { return " + std::string(s64_min) + "; } \n"
  "inline int coot_type_max_int() { return " + std::string(s32_max) + "; } \n"
  "inline long coot_type_max_long() { return " + std::string(s64_max) + "; } \n"
  "\n"
  "inline float coot_type_min_float() { return FLT_MIN; } \n"
  "inline double coot_type_min_double() { return DBL_MIN; } \n"
  "inline float coot_type_max_float() { return FLT_MAX; } \n"
  "inline double coot_type_max_double() { return DBL_MAX; } \n"
  "\n"
  "inline bool coot_is_fp_uint() { return false; } \n"
  "inline bool coot_is_fp_int() { return false; } \n"
  "inline bool coot_is_fp_ulong() { return false; } \n"
  "inline bool coot_is_fp_long() { return false; } \n"
  "inline bool coot_is_fp_float() { return true; } \n"
  "inline bool coot_is_fp_double() { return true; } \n"
  "\n"
  "inline bool coot_is_signed_uint() { return false; } \n"
  "inline bool coot_is_signed_int() { return true; } \n"
  "inline bool coot_is_signed_ulong() { return false; } \n"
  "inline bool coot_is_signed_long() { return true; } \n"
  "inline bool coot_is_signed_float() { return true; } \n"
  "inline bool coot_is_signed_double() { return true; } \n"
  "\n"
  // MAGMA-specific macros.
  "#define MAGMA_BLK_X 64 \n"
  "#define MAGMA_BLK_Y 32 \n"
  "#define MAGMA_TRANS_NX 32 \n"
  "#define MAGMA_TRANS_NY 8 \n"
  "#define MAGMA_TRANS_NB 32 \n"
  "#define MAGMA_TRANS_INPLACE_NB 16 \n"
  ;

  return source;
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
const std::string&
kernel_src::get_zeroway_source()
  {
  static const std::string source = init_zeroway_source();

  return source;
  }



inline
std::string
kernel_src::init_zeroway_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "xorwow_rng.cl",
      "philox_rng.cl"
  };

  std::string source = "";

  // First, load any auxiliary functions.
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "zeroway/" + filename;
    source += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : zeroway_kernel_id::get_names())
    {
    std::string filename = "zeroway/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_oneway_source()
  {
  static const std::string source = init_oneway_source();

  return source;
  }



// TODO: inplace_set_scalar() could be replaced with explicit call to clEnqueueFillBuffer()
// present in OpenCL 1.2: http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueFillBuffer.html

// TODO: need submat analogues of all functions

// TODO: need specialised handling for cx_float and cx_double
// for example (cx_double * cx_double) is not simply (double2 * double2)


inline
std::string
kernel_src::init_oneway_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "accu_wavefront_reduce.cl",
      "min_wavefront_reduce.cl",
      "max_wavefront_reduce.cl"
  };

  std::string source = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "oneway/" + filename;
    source += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : oneway_kernel_id::get_names())
    {
    std::string filename = "oneway/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_oneway_real_source()
  {
  static const std::string source = init_oneway_real_source();

  return source;
  }



inline
std::string
kernel_src::init_oneway_real_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::string source = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : oneway_real_kernel_id::get_names())
    {
    std::string filename = "oneway_real/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_oneway_integral_source()
  {
  static const std::string source = init_oneway_integral_source();

  return source;
  }



inline
std::string
kernel_src::init_oneway_integral_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "and_wavefront_reduce.cl",
  };

  std::string source = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "oneway_integral/" + filename;
    source += read_file(full_filename);
    }

  // Load each file for each kernel.
  for (const std::string& kernel_name : oneway_integral_kernel_id::get_names())
    {
    std::string filename = "oneway_integral/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_twoway_source()
  {
  static const std::string source = init_twoway_source();

  return source;
  }



inline
std::string
kernel_src::init_twoway_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "dot_wavefront_reduce.cl"
  };

  std::string source = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "twoway/" + filename;
    source += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : twoway_kernel_id::get_names())
    {
    std::string filename = "twoway/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_threeway_source()
  {
  static const std::string source = init_threeway_source();

  return source;
  }



inline
std::string
kernel_src::init_threeway_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::string source = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : threeway_kernel_id::get_names())
    {
    std::string filename = "threeway/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_magma_real_source()
  {
  static const std::string source = init_magma_real_source();

  return source;
  }



inline
std::string
kernel_src::init_magma_real_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::string source = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : magma_real_kernel_id::get_names())
    {
    std::string filename = "magma_real/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_src_epilogue()
  {
  static const std::string source = "";

  return source;
  }
