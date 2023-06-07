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



inline
coot_rt_t::~coot_rt_t()
  {
  coot_extra_debug_sigprint_this(this);
  }



inline
coot_rt_t::coot_rt_t()
  {
  coot_extra_debug_sigprint_this(this);
  backend = COOT_DEFAULT_BACKEND;
  }



inline
bool
coot_rt_t::init(const bool print_info)
  {
  coot_extra_debug_sigprint();

  // TODO: investigate reading a config file by default; if a config file exist, use the specifed platform and device within the config file
  // TODO: config file may exist in several places: (1) globally accessible, such as /etc/bandicoot_config, or locally, such as ~/.config/bandicoot_config
  // TODO: use case: user puts code on a server which has a different configuration than the user's workstation

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return get_rt().cl_rt.init(false, 0, 0, print_info);
    #else
    coot_stop_runtime_error("coot_rt::init(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return get_rt().cuda_rt.init(false, 0, 0, print_info);
    #else
    coot_stop_runtime_error("coot_rt::init(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::init(): unknown backend");
    }

  return false;
  }



inline
bool
coot_rt_t::init(const char* filename, const bool print_info)
  {
  coot_extra_debug_sigprint();

  return coot_rt_t::init(std::string(filename), print_info);
  }



inline
bool
coot_rt_t::init(const std::string filename, const bool print_info)
  {
  coot_extra_debug_sigprint();

  // TODO: handling of config files is currently rudimentary

  if(print_info)  {std::cout << "coot::opencl::runtime_t::init(): reading " << filename << std::endl; }

  uword wanted_platform = 0;
  uword wanted_device   = 0;

  std::ifstream f;
  f.open(filename.c_str(), std::fstream::binary);

  if(f.is_open() == false)
    {
    std::cout << "coot::opencl::runtime_t::init(): couldn't read " << filename << std::endl;
    return false;
    }

  f >> wanted_platform;
  f >> wanted_device;

  if(f.good() == false)
    {
    wanted_platform = 0;
    wanted_device   = 0;

    std::cout << "coot::opencl::runtime_t::init(): couldn't read " << filename << std::endl;
    return false;
    }
  else
    {
    if(print_info)  { std::cout << "coot::opencl::runtime::init(): wanted_platform = " << wanted_platform << "   wanted_device = " << wanted_device << std::endl; }
    }

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return get_rt().cl_rt.init(true, wanted_platform, wanted_device, print_info);
    #else
    coot_stop_runtime_error("coot_rt::init(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return get_rt().cuda_rt.init(true, wanted_platform, wanted_device, print_info);
    #else
    coot_stop_runtime_error("coot_rt::init(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::init(): unknown backend");
    }

  return false;
  }



inline
bool
coot_rt_t::init(const uword wanted_platform, const uword wanted_device, const bool print_info)
  {
  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return get_rt().cl_rt.init(true, wanted_platform, wanted_device, print_info);
    #else
    coot_stop_runtime_error("coot_rt::init(): OpenCL backend not enabled");

    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return get_rt().cuda_rt.init(true, wanted_platform, wanted_device, print_info);
    #else
    coot_stop_runtime_error("coot_rt::init(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::init(): unknown backend");
    }

  return false;
  }



template<typename eT>
inline
dev_mem_t<eT>
coot_rt_t::acquire_memory(const uword n_elem)
  {
  coot_extra_debug_sigprint();

//  coot_check_runtime_error( (valid == false), "coot_rt::acquire_memory(): runtime not valid" );

  if(n_elem == 0)  { return dev_mem_t<eT>({ NULL }); }

  coot_debug_check
   (
   ( size_t(n_elem) > (std::numeric_limits<size_t>::max() / sizeof(eT)) ),
   "coot_rt::acquire_memory(): requested size is too large"
   );

  dev_mem_t<eT> result;

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    result.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_elem);
    #else
    coot_stop_runtime_error("coot_rt::acquire_memory(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    result.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_elem);
    #else
    coot_stop_runtime_error("coot_rt::acquire_memory(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::acquire_memory(): unknown backend");
    }

  return result;
  }



template<typename eT>
inline
void
coot_rt_t::release_memory(dev_mem_t<eT> dev_mem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    get_rt().cl_rt.release_memory(dev_mem.cl_mem_ptr);
    #else
    coot_stop_runtime_error("coot_rt::release_memory(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    get_rt().cuda_rt.release_memory(dev_mem.cuda_mem_ptr);
    #else
    coot_stop_runtime_error("coot_rt::release_memory(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::release_memory(): unknown backend");
    }
  }



template<typename eT>
inline
bool
coot_rt_t::is_supported_type()
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #ifdef COOT_USE_OPENCL
    return get_rt().cl_rt.is_supported_type<eT>();
    #else
    coot_stop_runtime_error("coot_rt::is_supported_type(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #ifdef COOT_USE_CUDA
    return get_rt().cuda_rt.is_supported_type<eT>();
    #else
    coot_stop_runtime_error("coot_rt::is_supported_type(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::is_supported_type(): unknown backend");
    }

  return false;
  }



inline
void
coot_rt_t::set_rng_seed(const u64 seed)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #ifdef COOT_USE_OPENCL
    get_rt().cl_rt.set_rng_seed(seed);
    #else
    coot_stop_runtime_error("coot_rt::set_rng_seed(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #ifdef COOT_USE_CUDA
    get_rt().cuda_rt.set_rng_seed(seed);
    #else
    coot_stop_runtime_error("coot_rt::set_rng_seed(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::set_rng_seed(): unknown backend");
    }
  }



inline
void
coot_rt_t::synchronise()
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    get_rt().cl_rt.synchronise();
    #else
    coot_stop_runtime_error("coot_rt::synchronise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    get_rt().cuda_rt.synchronise();
    #else
    coot_stop_runtime_error("coot_rt::synchronise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::synchronise(): unknown backend");
    }
  }



template<typename out_eT, typename in_eT>
inline
void
coot_rt_t::copy_array(dev_mem_t<out_eT> dest, dev_mem_t<in_eT> src, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::copy_array(dest, src, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::copy_array(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::copy_array(dest, src, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::copy_array(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::copy_array(): unknown backend");
    }
  }



template<typename out_eT, typename in_eT>
inline
void
coot_rt_t::copy_subview(dev_mem_t<out_eT> dest, dev_mem_t<in_eT> src, const uword aux_row1, const uword aux_col1, const uword M_n_rows, const uword M_n_cols, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::copy_subview(dest, src, aux_row1, aux_col1, M_n_rows, M_n_cols, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::copy_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::copy_subview(dest, src, aux_row1, aux_col1, M_n_rows, M_n_cols, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::copy_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::copy_subview(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::reorder_cols(dev_mem_t<eT> out, const dev_mem_t<eT> mem, const uword n_rows, const dev_mem_t<uword> order, const uword out_n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::reorder_cols(out, mem, n_rows, order, out_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::reorder_cols(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::reorder_cols(out, mem, n_rows, order, out_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::reorder_cols(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::reorder_cols(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::extract_diag(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword in_mem_offset, const uword n_rows, const uword len)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::extract_diag(out, in, in_mem_offset, n_rows, len);
    #else
    coot_stop_runtime_error("coot_rt::extract_diag(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::extract_diag(out, in, in_mem_offset, n_rows, len);
    #else
    coot_stop_runtime_error("coot_rt::extract_diag(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::extract_diag(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::set_diag(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword in_mem_offset, const uword n_rows, const uword len)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::set_diag(out, in, in_mem_offset, n_rows, len);
    #else
    coot_stop_runtime_error("coot_rt::set_diag(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::set_diag(out, in, in_mem_offset, n_rows, len);
    #else
    coot_stop_runtime_error("coot_rt::set_diag(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::set_diag(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::copy_diag(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword out_mem_offset, const uword in_mem_offset, const uword out_n_rows, const uword in_n_rows, const uword len)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::copy_diag(out, in, out_mem_offset, in_mem_offset, out_n_rows, in_n_rows, len);
    #else
    coot_stop_runtime_error("coot_rt::copy_diag(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::copy_diag(out, in, out_mem_offset, in_mem_offset, out_n_rows, in_n_rows, len);
    #else
    coot_stop_runtime_error("coot_rt::copy_diag(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::copy_diag(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::inplace_op_scalar(dev_mem_t<eT> dest, const eT val, const uword n_elem, const oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::inplace_op_scalar(dest, val, n_elem, num);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_scalar(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::inplace_op_scalar(dest, val, n_elem, num);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_scalar(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::inplace_op_scalar(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::inplace_op_array(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_elem, const twoway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::inplace_op_array(dest, src, n_elem, num);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_array(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::inplace_op_array(dest, src, n_elem, num);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_array(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::inplace_op_array(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::inplace_op_subview(dev_mem_t<eT> dest, const eT val, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword M_n_rows, const oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::inplace_op_subview(dest, val, aux_row1, aux_col1, n_rows, n_cols, M_n_rows, num);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::inplace_op_subview(dest, val, aux_row1, aux_col1, n_rows, n_cols, M_n_rows, num);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::inplace_op_subview(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::inplace_op_diag(dev_mem_t<eT> dest, const uword mem_offset, const eT val, const uword n_rows, const uword len, const oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::inplace_op_diag(dest, mem_offset, val, n_rows, len, num);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_diag(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::inplace_op_diag(dest, mem_offset, val, n_rows, len, num);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_diag(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::inplace_op_diag(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::inplace_op_subview(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const twoway_kernel_id::enum_id num, const char* identifier)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::inplace_op_subview(dest, src, M_n_rows, aux_row1, aux_col1, n_rows, n_cols, num, identifier);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::inplace_op_subview(dest, src, M_n_rows, aux_row1, aux_col1, n_rows, n_cols, num, identifier);
    #else
    coot_stop_runtime_error("coot_rt::inplace_op_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::inplace_op_subview(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::replace(dev_mem_t<eT> mem, const uword n_elem, const eT val_find, const eT val_replace)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::replace(mem, n_elem, val_find, val_replace);
    #else
    coot_stop_runtime_error("coot_rt::replace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::replace(mem, n_elem, val_find, val_replace);
    #else
    coot_stop_runtime_error("coot_rt::replace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::replace(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::htrans(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::htrans(dest, src, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::htrans(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::htrans(dest, src, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::htrans(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::htrans(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::strans(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::strans(dest, src, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::strans(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::strans(dest, src, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::strans(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::strans(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::fill_randu(dest, n);
    #else
    coot_stop_runtime_error("coot_rt::fill_randu(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::fill_randu(dest, n);
    #else
    coot_stop_runtime_error("coot_rt::fill_randu(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::fill_randu(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::fill_randn(dest, n, mu, sd);
    #else
    coot_stop_runtime_error("coot_rt::fill_randn(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::fill_randn(dest, n, mu, sd);
    #else
    coot_stop_runtime_error("coot_rt::fill_randn(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::fill_randn(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::fill_randi(dest, n, lo, hi);
    #else
    coot_stop_runtime_error("coot_rt::fill_randi(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::fill_randi(dest, n, lo, hi);
    #else
    coot_stop_runtime_error("coot_rt::fill_randi(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::fill_randi(): unknown backend");
    }
  }



template<typename eT1, typename eT2, typename eT3>
inline
void
coot_rt_t::array_op(dev_mem_t<eT3> dest, const uword n_elem, const dev_mem_t<eT1> A_mem, const dev_mem_t<eT2> B_mem, const threeway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::array_op(dest, n_elem, A_mem, B_mem, num);
    #else
    coot_stop_runtime_error("coot_rt::array_op(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::array_op(dest, n_elem, A_mem, B_mem, num);
    #else
    coot_stop_runtime_error("coot_rt::array_op(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::array_op(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::eop_scalar(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_elem, const eT1 aux_val_pre, const eT2 aux_val_post, const twoway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::eop_scalar(dest, src, n_elem, aux_val_pre, aux_val_post, num);
    #else
    coot_stop_runtime_error("coot_rt::eop_scalar(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::eop_scalar(dest, src, n_elem, aux_val_pre, aux_val_post, num);
    #else
    coot_stop_runtime_error("coot_rt::eop_scalar(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::eop_scalar(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::accu(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::accu(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::accu(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::accu(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::accu(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::accu(): unknown backend");
    }

  return eT(0);
  }



template<typename eT>
inline
eT
coot_rt_t::accu_subview(const dev_mem_t<eT> mem, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::accu_subview(mem, M_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::accu_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::accu_subview(mem, M_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::accu_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::accu_subview(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT>
inline
eT
coot_rt_t::min(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::min(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::min(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::min(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::min(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::min(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT>
inline
eT
coot_rt_t::max(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::max(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::max(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::max(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::max(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::max(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT>
inline
eT
coot_rt_t::max_abs(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::max_abs(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::max_abs(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::max_abs(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::max_abs(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::max_abs(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT1, typename eT2>
inline
bool
coot_rt_t::all_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::all_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::all_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::all_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::all_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::all_vec(): unknown backend");
    }

  return false; // stop compilation warnings
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::all(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::all(out_mem, in_mem, n_rows, n_cols, val, num, colwise);
    #else
    coot_stop_runtime_error("coot_rt::all(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::all(out_mem, in_mem, n_rows, n_cols, val, num, colwise);
    #else
    coot_stop_runtime_error("coot_rt::all(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::all(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
bool
coot_rt_t::any_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::any_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::any_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::any_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::any_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::any_vec(): unknown backend");
    }

  return false; // stop compilation warnings
  }



template<typename eT>
inline
bool
coot_rt_t::any_vec(const dev_mem_t<eT> mem, const uword n_elem, const eT val, const oneway_real_kernel_id::enum_id num, const oneway_real_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::any_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::any_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::any_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::any_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::any_vec(): unknown backend");
    }

  return false; // stop compilation warnings
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::any(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::any(out_mem, in_mem, n_rows, n_cols, val, num, colwise);
    #else
    coot_stop_runtime_error("coot_rt::any(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::any(out_mem, in_mem, n_rows, n_cols, val, num, colwise);
    #else
    coot_stop_runtime_error("coot_rt::any(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::any(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::relational_scalar_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::relational_scalar_op(out_mem, in_mem, n_elem, val, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_scalar_op(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::relational_scalar_op(out_mem, in_mem, n_elem, val, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_scalar_op(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::relational_scalar_op(): unknown backend");
    }
  }



template<typename eT1>
inline
void
coot_rt_t::relational_unary_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const oneway_real_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::relational_unary_array_op(out_mem, in_mem, n_elem, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_unary_array_op(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::relational_unary_array_op(out_mem, in_mem, n_elem, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_unary_array_op(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::relational_unary_array_op(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::relational_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> X_mem, const dev_mem_t<eT2> Y_mem, const uword n_elem, const twoway_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::relational_array_op(out_mem, X_mem, Y_mem, n_elem, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_array_op(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::relational_array_op(out_mem, X_mem, Y_mem, n_elem, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_array_op(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::relational_array_op(): unknown backend");
    }
  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::chol(dev_mem_t<eT> out, const uword n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::chol(out, n_rows);
    #else
    coot_stop_runtime_error("coot_rt::chol(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::chol(out, n_rows);
    #else
    coot_stop_runtime_error("coot_rt::chol(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::chol(): unknown backend");
    }

  return std::make_tuple(false, ""); // fix warnings
  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::lu(dev_mem_t<eT> L, dev_mem_t<eT> U, dev_mem_t<eT> in, const bool pivoting, dev_mem_t<eT> P, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::lu(L, U, in, pivoting, P, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::lu(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::lu(L, U, in, pivoting, P, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::lu(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::lu(): unknown backend");
    }

  return std::make_tuple(false, ""); // fix warnings
  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::svd(dev_mem_t<eT> U, dev_mem_t<eT> S, dev_mem_t<eT> V, dev_mem_t<eT> A, const uword n_rows, const uword n_cols, const bool compute_u_vt)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::svd(U, S, V, A, n_rows, n_cols, compute_u_vt);
    #else
    coot_stop_runtime_error("coot_rt::svd(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::svd(U, S, V, A, n_rows, n_cols, compute_u_vt);
    #else
    coot_stop_runtime_error("coot_rt::svd(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::svd(): unknown backend");
    }

  return std::make_tuple(false, ""); // fix warnings
  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::eig_sym(dev_mem_t<eT> mem, const uword n_rows, const bool eigenvectors, dev_mem_t<eT> eigenvalues)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::eig_sym(mem, n_rows, eigenvectors, eigenvalues);
    #else
    coot_stop_runtime_error("coot_rt::eig_sym(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::eig_sym(mem, n_rows, eigenvectors, eigenvalues);
    #else
    coot_stop_runtime_error("coot_rt::eig_sym(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::eig_sym(): unknown backend");
    }

  return std::make_tuple(false, ""); // fix warnings
  }



template<typename eT>
inline
void
coot_rt_t::copy_from_dev_mem(eT* dest, const dev_mem_t<eT> src, const uword N)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::copy_from_dev_mem(dest, src, N);
    #else
    coot_stop_runtime_error("coot_rt::copy_from_dev_mem(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::copy_from_dev_mem(dest, src, N);
    #else
    coot_stop_runtime_error("coot_rt::copy_from_dev_mem(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::copy_from_dev_mem(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::copy_into_dev_mem(dest, src, N);
    #else
    coot_stop_runtime_error("coot_rt::copy_into_dev_mem(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::copy_into_dev_mem(dest, src, N);
    #else
    coot_stop_runtime_error("coot_rt::copy_into_dev_mem(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::copy_into_dev_mem(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::eye(dev_mem_t<eT> mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::eye(mem, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::eye(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::eye(mem, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::eye(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::eye(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::get_val(const dev_mem_t<eT> mem, const uword index)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::get_val(mem, index);
    #else
    coot_stop_runtime_error("coot_rt::get_val(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::get_val(mem, index);
    #else
    coot_stop_runtime_error("coot_rt::get_val(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::get_val(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT>
inline
void
coot_rt_t::set_val(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::set_val(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::set_val(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::set_val(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::set_val(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::set_val(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_add_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::val_add_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_add_inplace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::val_add_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_add_inplace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::val_add_inplace(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_minus_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::val_minus_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_minus_inplace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::val_minus_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_minus_inplace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::val_minus_inplace(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_mul_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::val_mul_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_mul_inplace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::val_mul_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_mul_inplace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::val_mul_inplace(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_div_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::val_div_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_div_inplace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::val_div_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_div_inplace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::val_div_inplace(): unknown backend");
    }
  }



template<typename eT, const bool do_trans_A, const bool do_trans_B>
inline
void
coot_rt_t::gemm(dev_mem_t<eT> C_mem, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT> A_mem, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> B_mem, const eT alpha, const eT beta)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::gemm<do_trans_A, do_trans_B>::apply(C_mem, C_n_rows, C_n_cols, A_mem, A_n_rows, A_n_cols, B_mem, alpha, beta);
    #else
    coot_stop_runtime_error("coot_rt::gemm(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::gemm<do_trans_A, do_trans_B>::apply(C_mem, C_n_rows, C_n_cols, A_mem, A_n_rows, A_n_cols, B_mem, alpha, beta);
    #else
    coot_stop_runtime_error("coot_rt::gemm(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::gemm(): unknown backend");
    }
  }



template<typename eT, const bool do_trans_A>
inline
void
coot_rt_t::gemv(dev_mem_t<eT> y_mem, const dev_mem_t<eT> A_mem, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT> x_mem, const eT alpha, const eT beta)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::gemv<do_trans_A>::apply(y_mem, A_mem, A_n_rows, A_n_cols, x_mem, alpha, beta);
    #else
    coot_stop_runtime_error("coot_rt::gemv(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::gemv<do_trans_A>::apply(y_mem, A_mem, A_n_rows, A_n_cols, x_mem, alpha, beta);
    #else
    coot_stop_runtime_error("coot_rt::gemv(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::gemv(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::mul_diag(dev_mem_t<eT> C_mem, const uword C_n_rows, const uword C_n_cols, const eT alpha, const dev_mem_t<eT> A_mem, const bool A_is_diag, const bool A_trans, const dev_mem_t<eT> B_mem, const bool B_is_diag, const bool B_trans)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::mul_diag(C_mem, C_n_rows, C_n_cols, alpha, A_mem, A_is_diag, A_trans, B_mem, B_is_diag, B_trans);
    #else
    coot_stop_runtime_error("coot_rt::mul_diag(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::mul_diag(C_mem, C_n_rows, C_n_cols, alpha, A_mem, A_is_diag, A_trans, B_mem, B_is_diag, B_trans);
    #else
    coot_stop_runtime_error("coot_rt::mul_diag(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::mul_diag(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::sum_colwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sum_colwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::sum_colwise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sum_colwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::sum_colwise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sum_colwise(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::sum_rowwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sum_rowwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::sum_rowwise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sum_rowwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::sum_rowwise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sum_rowwise(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::sum_colwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> a_mem, const uword a_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sum_colwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::sum_colwise_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sum_colwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::sum_colwise_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sum_colwise_subview(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::sum_rowwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> a_mem, const uword a_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sum_rowwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::sum_rowwise_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sum_rowwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::sum_rowwise_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sum_rowwise_subview(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::min_colwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::min_colwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::min_colwise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::min_colwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::min_colwise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::min_colwise(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::min_rowwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::min_rowwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::min_rowwise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::min_rowwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::min_rowwise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::min_rowwise(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::min_colwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> a_mem, const uword a_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::min_colwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::min_colwise_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::min_colwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::min_colwise_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::min_colwise_subview(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::min_rowwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> a_mem, const uword a_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::min_rowwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::min_rowwise_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::min_rowwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::min_rowwise_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::min_rowwise_subview(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::max_colwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::max_colwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::max_colwise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::max_colwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::max_colwise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::max_colwise(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::max_rowwise(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> A_mem, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::max_rowwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::max_rowwise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::max_rowwise(out_mem, A_mem, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::max_rowwise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::max_rowwise(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::max_colwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> a_mem, const uword a_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::max_colwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::max_colwise_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::max_colwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::max_colwise_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::max_colwise_subview(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::max_rowwise_subview(dev_mem_t<eT2> out_mem, const dev_mem_t<eT1> a_mem, const uword a_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::max_rowwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::max_rowwise_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::max_rowwise_subview(out_mem, a_mem, a_n_rows, aux_row1, aux_col1, n_rows, n_cols, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::max_rowwise_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::max_rowwise_subview(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::trace(const dev_mem_t<eT> mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::trace(mem, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::trace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::trace(mem, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::trace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::trace(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT1, typename eT2>
inline
typename promote_type<eT1, eT2>::result
coot_rt_t::dot(const dev_mem_t<eT1> mem1, const dev_mem_t<eT2> mem2, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::dot(mem1, mem2, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::dot(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::dot(mem1, mem2, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::dot(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::dot(): unknown backend");
    }

  return typename promote_type<eT1, eT2>::result(0); // fix warning
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::repmat(const dev_mem_t<eT1> src, dev_mem_t<eT2> dest, const uword n_rows, const uword n_cols, const uword copies_per_row, const uword copies_per_col)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::repmat(src, dest, n_rows, n_cols, copies_per_row, copies_per_col);
    #else
    coot_stop_runtime_error("coot_rt::repmat(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::repmat(src, dest, n_rows, n_cols, copies_per_row, copies_per_col);
    #else
    coot_stop_runtime_error("coot_rt::repmat(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::repmat(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::linspace(dev_mem_t<eT> mem, const eT start, const eT end, const uword num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::linspace(mem, start, end, num);
    #else
    coot_stop_runtime_error("coot_rt::linspace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::linspace(mem, start, end, num);
    #else
    coot_stop_runtime_error("coot_rt::linspace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::linspace(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::clamp(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const eT1 min_val, const eT1 max_val, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::clamp(dest, src, min_val, max_val, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::clamp(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::clamp(dest, src, min_val, max_val, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::clamp(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::clamp(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::vec_norm_1(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::vec_norm_1(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_1(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::vec_norm_1(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_1(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::vec_norm_1(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
eT
coot_rt_t::vec_norm_2(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::vec_norm_2(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_2(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::vec_norm_2(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_2(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::vec_norm_2(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
eT
coot_rt_t::vec_norm_k(const dev_mem_t<eT> mem, const uword n_elem, const uword k)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::vec_norm_k(mem, n_elem, k);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_k(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::vec_norm_k(mem, n_elem, k);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_k(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::vec_norm_k(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
eT
coot_rt_t::vec_norm_min(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::vec_norm_min(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_min(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::vec_norm_min(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_min(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::vec_norm_min(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::mean(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword n_rows, const uword n_cols, const uword dim, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::mean(out, in, n_rows, n_cols, dim, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::mean(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::mean(out, in, n_rows, n_cols, dim, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::mean(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::mean(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::mean_subview(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword M_n_rows, const uword start_row, const uword start_col, const uword n_rows, const uword n_cols, const uword dim, const bool post_conv_apply)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::mean_subview(out, in, M_n_rows, start_row, start_col, n_rows, n_cols, dim, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::mean_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::mean_subview(out, in, M_n_rows, start_row, start_col, n_rows, n_cols, dim, post_conv_apply);
    #else
    coot_stop_runtime_error("coot_rt::mean_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::mean_subview(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::median(dev_mem_t<eT2> out, dev_mem_t<eT1> in, const uword n_rows, const uword n_cols, const uword dim)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::median(out, in, n_rows, n_cols, dim);
    #else
    coot_stop_runtime_error("coot_rt::median(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::median(out, in, n_rows, n_cols, dim);
    #else
    coot_stop_runtime_error("coot_rt::median(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::median(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::median_vec(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::median_vec(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::median_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::median_vec(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::median_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::median_vec(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
void
coot_rt_t::var(dev_mem_t<eT> out, const dev_mem_t<eT> in, const dev_mem_t<eT> means, const uword n_rows, const uword n_cols, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::var(out, in, means, n_rows, n_cols, dim, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::var(out, in, means, n_rows, n_cols, dim, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::var(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::var_subview(dev_mem_t<eT> out, const dev_mem_t<eT> in, const dev_mem_t<eT> means, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword dim, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::var_subview(out, in, means, M_n_rows, M_n_cols, aux_row1, aux_col1, n_rows, n_cols, dim, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::var_subview(out, in, means, M_n_rows, M_n_cols, aux_row1, aux_col1, n_rows, n_cols, dim, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::var_subview(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::var_vec(const dev_mem_t<eT> mem, const eT mean, const uword n_elem, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::var_vec(mem, mean, n_elem, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::var_vec(mem, mean, n_elem, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::var_vec(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
eT
coot_rt_t::var_vec_subview(const dev_mem_t<eT> mem, const eT mean, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::var_vec_subview(mem, mean, M_n_rows, M_n_cols, aux_row1, aux_col1, n_rows, n_cols, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_vec_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::var_vec_subview(mem, mean, M_n_rows, M_n_cols, aux_row1, aux_col1, n_rows, n_cols, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_vec_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::var_vec_subview(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
inline
void
coot_rt_t::join_cols(dev_mem_t<eT5> out, const dev_mem_t<eT1> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT2> B, const uword B_n_rows, const uword B_n_cols, const dev_mem_t<eT3> C, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT4> D, const uword D_n_rows, const uword D_n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::join_cols(out, A, A_n_rows, A_n_cols, B, B_n_rows, B_n_cols, C, C_n_rows, C_n_cols, D, D_n_rows, D_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::join_cols(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::join_cols(out, A, A_n_rows, A_n_cols, B, B_n_rows, B_n_cols, C, C_n_rows, C_n_cols, D, D_n_rows, D_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::join_cols(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::join_cols(): unknown backend");
    }
  }



template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
inline
void
coot_rt_t::join_rows(dev_mem_t<eT5> out, const dev_mem_t<eT1> A, const uword A_n_rows, const uword A_n_cols, const dev_mem_t<eT2> B, const uword B_n_rows, const uword B_n_cols, const dev_mem_t<eT3> C, const uword C_n_rows, const uword C_n_cols, const dev_mem_t<eT4> D, const uword D_n_rows, const uword D_n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::join_rows(out, A, A_n_rows, A_n_cols, B, B_n_rows, B_n_cols, C, C_n_rows, C_n_cols, D, D_n_rows, D_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::join_rows(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::join_rows(out, A, A_n_rows, A_n_cols, B, B_n_rows, B_n_cols, C, C_n_rows, C_n_cols, D, D_n_rows, D_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::join_rows(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::join_rows(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::sort_colwise(dev_mem_t<eT> mem, const uword n_rows, const uword n_cols, const uword sort_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sort_colwise(mem, n_rows, n_cols, sort_type);
    #else
    coot_stop_runtime_error("coot_rt::sort_colwise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sort_colwise(mem, n_rows, n_cols, sort_type);
    #else
    coot_stop_runtime_error("coot_rt::sort_colwise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sort_colwise(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::sort_rowwise(dev_mem_t<eT> mem, const uword n_rows, const uword n_cols, const uword sort_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sort_rowwise(mem, n_rows, n_cols, sort_type);
    #else
    coot_stop_runtime_error("coot_rt::sort_rowwise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sort_rowwise(mem, n_rows, n_cols, sort_type);
    #else
    coot_stop_runtime_error("coot_rt::sort_rowwise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sort_rowwise(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::sort_vec(dev_mem_t<eT> mem, const uword n_elem, const uword sort_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sort_vec(mem, n_elem, sort_type);
    #else
    coot_stop_runtime_error("coot_rt::sort_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sort_vec(mem, n_elem, sort_type);
    #else
    coot_stop_runtime_error("coot_rt::sort_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sort_vec(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::sort_index_vec(dev_mem_t<uword> out, dev_mem_t<eT> A, const uword n_elem, const uword sort_type, const uword stable_sort)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sort_index_vec(out, A, n_elem, sort_type, stable_sort);
    #else
    coot_stop_runtime_error("coot_rt::sort_index_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sort_index_vec(out, A, n_elem, sort_type, stable_sort);
    #else
    coot_stop_runtime_error("coot_rt::sort_index_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sort_index_vec(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::find(dev_mem_t<uword>& out, uword& out_len, const dev_mem_t<eT> A, const uword n_elem, const uword k, const uword find_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::find(out, out_len, A, n_elem, k, find_type);
    #else
    coot_stop_runtime_error("coot_rt::find(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::find(out, out_len, A, n_elem, k, find_type);
    #else
    coot_stop_runtime_error("coot_rt::find(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::find(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::symmat(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword size, const uword lower)
  {
  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::symmat(out, in, size, lower);
    #else
    coot_stop_runtime_error("coot_rt::symmat(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::symmat(out, in, size, lower);
    #else
    coot_stop_runtime_error("coot_rt::symmat(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::symmat(): unknown backend");
    }
  }
