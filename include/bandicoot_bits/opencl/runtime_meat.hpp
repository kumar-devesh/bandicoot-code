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
runtime_t::~runtime_t()
  {
  coot_extra_debug_sigprint_this(this);
  
  internal_cleanup();
  
  valid = false;
  }



inline
runtime_t::runtime_t()
  {
  coot_extra_debug_sigprint_this(this);
  
  valid   = false;
  plt_id  = NULL;
  dev_id  = NULL;
  ctxt    = NULL;
  cq      = NULL;
  }



inline
bool
runtime_t::init(const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info)
  {
  coot_extra_debug_sigprint();
  
  internal_cleanup();
  valid = false;
  
  bool status = false;
  
  status = search_devices(plt_id, dev_id, manual_selection, wanted_platform, wanted_device, print_info);
  if(status == false)  { coot_debug_warn("coot::opencl::runtime_t: couldn't find a suitable device"); return false; }
  
  interrogate_device(dev_info, plt_id, dev_id, print_info);
  
  if(dev_info.opencl_ver < 120)  { coot_debug_warn("coot::opencl::runtime_t: selected device has OpenCL version < 1.2"); return false; }
  
  status = setup_queue(ctxt, cq, plt_id, dev_id);
  if(status == false)  { coot_debug_warn("coot::opencl::runtime_t: couldn't setup queue"); return false; }
  
  // setup kernels

  std::vector<std::pair<std::string, cl_kernel*>> name_map;
  type_to_dev_string type_map;
  std::string src =
      kernel_src::get_src_preamble() +
      rt_common::get_three_elem_kernel_src(threeway_kernels, kernel_src::get_threeway_source(), threeway_kernel_id::get_names(), name_map, type_map) +
      rt_common::get_two_elem_kernel_src(twoway_kernels, kernel_src::get_twoway_source(), twoway_kernel_id::get_names(), "", name_map, type_map) +
      rt_common::get_one_elem_kernel_src(oneway_kernels, kernel_src::get_oneway_source(), oneway_kernel_id::get_names(), "", name_map, type_map) +
      rt_common::get_one_elem_real_kernel_src(oneway_real_kernels, kernel_src::get_oneway_real_source(), oneway_real_kernel_id::get_names(), "", name_map, type_map) +
      kernel_src::get_src_epilogue();

  status = compile_kernels(src, name_map);
  if(status == false)  { coot_debug_warn("coot::opencl::runtime_t: couldn't setup OpenCL kernels"); return false; }
  
  // TODO: refactor to allow use the choice of clBLAS or clBLast backends


  // setup clBLAS
  
  
  get_cerr_stream().flush();
  get_cerr_stream() << "setup clBLAS: start" << endl;
  
  cl_int clblas_status = clblasSetup();
  
  get_cerr_stream().flush();
  get_cerr_stream() << "setup clBLAS: end" << endl;
    
  if(clblas_status != CL_SUCCESS)  { coot_debug_warn("coot::opencl::runtime_t: couldn't setup clBLAS"); return false; }
  
  if(status == false)
    {
    internal_cleanup();
    valid = false;
    
    return false;
    }
  
  valid = true;
  
  return true;
  }



inline
void
runtime_t::lock()
  {
  coot_extra_debug_sigprint();
  
  #if defined(COOT_USE_CXX11)
    {
    coot_extra_debug_print("calling mutex.lock()");
    mutex.lock();
    }
  #endif
  }




inline
void
runtime_t::unlock()
  {
  coot_extra_debug_sigprint();
  
  #if defined(COOT_USE_CXX11)
    {
    coot_extra_debug_print("calling mutex.unlock()");
    mutex.unlock();
    }
  #endif
  }



inline
void
runtime_t::internal_cleanup()
  {
  coot_extra_debug_sigprint();
  
  if(cq != NULL)  { clFinish(cq); }
  
  clblasTeardown();
  
  // TODO: go through each kernel vector
  
  //const uword f_kernels_size = f_kernels.size();
  
  //for(uword i=0; i<f_kernels_size; ++i)  { if(f_kernels.at(i) != NULL)  { clReleaseKernel(f_kernels.at(i)); } }
  
  if(cq   != NULL)  { clReleaseCommandQueue(cq); cq   = NULL; }
  if(ctxt != NULL)  { clReleaseContext(ctxt);    ctxt = NULL; }
  }



inline
bool
runtime_t::search_devices(cl_platform_id& out_plt_id, cl_device_id& out_dev_id, const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info) const
  {
  coot_extra_debug_sigprint();
  
  // first, get a list of platforms and the devices on each platform
  
  cl_int  status      = 0;
  cl_uint n_platforms = 0;
  
  status = clGetPlatformIDs(0, NULL, &n_platforms);
  
  if((status != CL_SUCCESS) || (n_platforms == 0))
    {
    coot_debug_warn("coot::opencl::runtime_t::search_devices(): no OpenCL platforms available");
    return false;
    }
  
  std::vector<cl_platform_id> platform_ids(n_platforms);
  
  status = clGetPlatformIDs(n_platforms, &(platform_ids[0]), NULL);
  
  if(status != CL_SUCCESS)
    {
    coot_debug_warn("coot::opencl::runtime_t::search_devices(): couldn't get info on OpenCL platforms");
    return false;
    }
  

  // go through each platform
  
  std::vector< std::vector<cl_device_id> > device_ids(n_platforms);
  std::vector< std::vector<int         > > device_pri(n_platforms);  // device priorities
  
  for(size_t platform_count = 0; platform_count < n_platforms; ++platform_count)
    {
    cl_platform_id tmp_platform_id = platform_ids.at(platform_count);
    
    cl_uint local_n_devices = 0;
    
    status = clGetDeviceIDs(tmp_platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &local_n_devices);
    
    if((status != CL_SUCCESS) || (local_n_devices == 0))
      {
      continue;  // go to the next platform
      }
    
    std::vector<cl_device_id>& local_device_ids = device_ids.at(platform_count);
    std::vector<int>&          local_device_pri = device_pri.at(platform_count);
    
    local_device_ids.resize(local_n_devices);
    local_device_pri.resize(local_n_devices);
    
    status = clGetDeviceIDs(tmp_platform_id, CL_DEVICE_TYPE_ALL, local_n_devices, &(local_device_ids[0]), NULL);
    
    // go through each device on this platform
    for(size_t local_device_count = 0; local_device_count < local_n_devices; ++local_device_count)
      {
      cl_device_id local_device_id = local_device_ids.at(local_device_count);
      
      if(print_info)
        {
        get_cerr_stream().flush();
        get_cerr_stream() << "platform: " << platform_count << " / device: " << local_device_count << std::endl;
        }
      
      runtime_dev_info tmp_info;
      
      const bool ok = interrogate_device(tmp_info, tmp_platform_id, local_device_id, print_info);
      
      if(print_info)
        {
        if(ok == false)
          {
          get_cerr_stream().flush();
          get_cerr_stream() << "problem with getting info about device" << std::endl;
          }
        
        get_cerr_stream() << std::endl;
        }
      
      local_device_pri.at(local_device_count) = 0;
      
      if(tmp_info.is_gpu)           { local_device_pri.at(local_device_count) +=  2; }
      if(tmp_info.has_float64)      { local_device_pri.at(local_device_count) +=  1; }
      if(tmp_info.opencl_ver < 120) { local_device_pri.at(local_device_count)  = -1; }
      }
    }
  
  
  if(manual_selection)
    {
    if(wanted_platform >= platform_ids.size())
      {
      coot_debug_warn("invalid platform number");
      return false;
      }
    
    std::vector<cl_device_id>& local_device_ids = device_ids.at(wanted_platform);
    
    if(wanted_device >= local_device_ids.size())
      {
      coot_debug_warn("invalid device number");
      return false;
      }
    
    if(print_info)
      {
      get_cerr_stream() << "selected: platform: " << wanted_platform << " / device: " << wanted_device << std::endl;
      }
    
    out_plt_id = platform_ids.at(wanted_platform);
    out_dev_id = local_device_ids.at(wanted_device);
    
    return true;
    }
  
  
  // select the device with highest priority
  
  bool found_device = false;
  
  int    best_val          = -1;
  size_t best_platform_num =  0;
  size_t best_device_num   =  0;
  
  for(size_t platform_count = 0; platform_count < n_platforms; ++platform_count)
    {
    std::vector<cl_device_id>& local_device_ids = device_ids.at(platform_count);
    std::vector<int>&          local_device_pri = device_pri.at(platform_count);
    
    size_t local_n_devices = local_device_ids.size();
    
    for(size_t local_device_count = 0; local_device_count < local_n_devices; ++local_device_count)
      {
      const int tmp_val = local_device_pri.at(local_device_count);
      
      // cout << "platform_count: " << platform_count << "  local_device_count: " << local_device_count << "  priority: " << tmp_val << "   best_val: " << best_val << endl;
      
      if(best_val < tmp_val)
        {
        best_val          = tmp_val;
        best_platform_num = platform_count;
        best_device_num   = local_device_count;
        
        found_device = true;
        }
      }
    }
  
  if(found_device)
    {
    if(print_info)
      {
      get_cerr_stream() << "selected: platform: " << best_platform_num << " / device: " << best_device_num << std::endl;
      }
    
    std::vector<cl_device_id>& local_device_ids = device_ids.at(best_platform_num);
    
    out_plt_id = platform_ids.at(best_platform_num);
    out_dev_id = local_device_ids.at(best_device_num);
    }
     
  return found_device;
  }



inline
bool
runtime_t::interrogate_device(runtime_dev_info& out_info, cl_platform_id in_plt_id, cl_device_id in_dev_id, const bool print_info) const
  {
  coot_extra_debug_sigprint();
  
  cl_char dev_name1[1024]; // TODO: use dynamic memory allocation (podarray or std::vector)
  cl_char dev_name2[1024];
  cl_char dev_name3[1024];
  
  dev_name1[0] = cl_char(0);
  dev_name2[0] = cl_char(0);
  dev_name3[0] = cl_char(0);
  
  cl_device_type      dev_type = 0;
  cl_device_fp_config dev_fp64 = 0;
  
  cl_uint dev_n_units     = 0;
  cl_uint dev_sizet_width = 0;
  cl_uint dev_ptr_width   = 0;
  cl_uint dev_opencl_ver  = 0;
  cl_uint dev_align       = 0;

  size_t dev_max_wg         = 0;
  size_t dev_wavefront_size = 0;
  
  
  clGetDeviceInfo(in_dev_id, CL_DEVICE_VENDOR,              sizeof(dev_name1),           &dev_name1,   NULL);
  clGetDeviceInfo(in_dev_id, CL_DEVICE_NAME,                sizeof(dev_name2),           &dev_name2,   NULL);
  clGetDeviceInfo(in_dev_id, CL_DEVICE_VERSION,             sizeof(dev_name3),           &dev_name3,   NULL);
  clGetDeviceInfo(in_dev_id, CL_DEVICE_TYPE,                sizeof(cl_device_type),      &dev_type,    NULL);
  clGetDeviceInfo(in_dev_id, CL_DEVICE_DOUBLE_FP_CONFIG,    sizeof(cl_device_fp_config), &dev_fp64,    NULL);
  clGetDeviceInfo(in_dev_id, CL_DEVICE_MAX_COMPUTE_UNITS,   sizeof(cl_uint),             &dev_n_units, NULL);
  clGetDeviceInfo(in_dev_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint),             &dev_align,   NULL);
  clGetDeviceInfo(in_dev_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),              &dev_max_wg,  NULL);

  // contrary to the official OpenCL specification (OpenCL 1.2, sec 4.2 and sec 6.1.1).
  // certain OpenCL implementations use internal size_t which doesn't correspond to CL_DEVICE_ADDRESS_BITS
  // example: Clover from Mesa 13.0.4, running as AMD OLAND (DRM 2.48.0 / 4.9.14-200.fc25.x86_64, LLVM 3.9.1)
  

  const char* tmp_program_src = \
    "__kernel void coot_interrogate(__global uint* out) \n"
    "  {                                                \n"
    "  const size_t i = get_global_id(0);               \n"
    "  if(i == 0)                                       \n"
    "    {                                              \n"
    "    out[0] = (uint)sizeof(size_t);                 \n"
    "    out[1] = (uint)sizeof(void*);                  \n"
    "    out[2] = (uint)(__OPENCL_VERSION__);           \n"
    "    }                                              \n"
    "  }                                                \n";
  
  cl_context       tmp_context    = NULL;
  cl_command_queue tmp_queue      = NULL;
  cl_program       tmp_program    = NULL;
  cl_kernel        tmp_kernel     = NULL;
  cl_mem           tmp_dev_mem    = NULL;
  cl_uint          tmp_cpu_mem[4] = { 0, 0, 0, 0 };
  
  
  cl_int status = 0;
  
  if(setup_queue(tmp_context, tmp_queue, in_plt_id, in_dev_id))
    {
    tmp_program = clCreateProgramWithSource(tmp_context, 1, (const char **)&(tmp_program_src), NULL, &status);
    
    if(status == CL_SUCCESS)
      {
      status = clBuildProgram(tmp_program, 0, NULL, NULL, NULL, NULL);
      
      // cout << "status: " << coot_cl_error::as_string(status) << endl;
       
      // size_t len = 0;
      // char buffer[10240];
       
      // clGetProgramBuildInfo(tmp_program, in_dev_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      // std::cout << "output from clGetProgramBuildInfo():" << std::endl;
      // std::cout << buffer << std::endl;
      
      if(status == CL_SUCCESS)
        {
        tmp_kernel = clCreateKernel(tmp_program, "coot_interrogate", &status);
        
        if(status == CL_SUCCESS)
          {
          // Extract what might be the warp or wavefront size.
          // It seems possible this could be different per kernel, but we'll hope not.
          status = clGetKernelWorkGroupInfo(tmp_kernel, in_dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &dev_wavefront_size, NULL);

          tmp_dev_mem = clCreateBuffer(tmp_context, CL_MEM_READ_WRITE, sizeof(cl_uint)*4, NULL, &status);
          
          clSetKernelArg(tmp_kernel, 0, sizeof(cl_mem),  &tmp_dev_mem);
          status = clEnqueueTask(tmp_queue, tmp_kernel, 0, NULL, NULL);  // TODO: replace with clEnqueueNDRangeKernel to avoid deprecation warnings
          
          if(status == CL_SUCCESS)
            {
            clFinish(cq);
            
            status = clEnqueueReadBuffer(tmp_queue, tmp_dev_mem, CL_TRUE, 0, sizeof(cl_uint)*4, tmp_cpu_mem, 0, NULL, NULL);
            
            if(status == CL_SUCCESS)
              {
              clFinish(cq);
              
              dev_sizet_width = tmp_cpu_mem[0];
              dev_ptr_width   = tmp_cpu_mem[1];
              dev_opencl_ver  = tmp_cpu_mem[2];
              }
            }
          }
        }
      }
    }
  
  if(status != CL_SUCCESS)
    {
    coot_debug_warn(coot_cl_error::as_string(status));
    }
  
  if(tmp_dev_mem != NULL)  { clReleaseMemObject   (tmp_dev_mem); }
  if(tmp_kernel  != NULL)  { clReleaseKernel      (tmp_kernel ); }
  if(tmp_program != NULL)  { clReleaseProgram     (tmp_program); }
  if(tmp_queue   != NULL)  { clReleaseCommandQueue(tmp_queue);   }
  if(tmp_context != NULL)  { clReleaseContext     (tmp_context); }
  
  if(print_info)
    {
    get_cerr_stream().flush();
    get_cerr_stream() << "name1:          " << dev_name1 << std::endl;
    get_cerr_stream() << "name2:          " << dev_name2 << std::endl;
    get_cerr_stream() << "name3:          " << dev_name3 << std::endl;
    get_cerr_stream() << "is_gpu:         " << (dev_type == CL_DEVICE_TYPE_GPU)  << std::endl;
    get_cerr_stream() << "fp64:           " << dev_fp64 << std::endl;
    get_cerr_stream() << "sizet_width:    " << dev_sizet_width  << std::endl;
    get_cerr_stream() << "ptr_width:      " << dev_ptr_width << std::endl;
    get_cerr_stream() << "n_units:        " << dev_n_units << std::endl;
    get_cerr_stream() << "opencl_ver:     " << dev_opencl_ver << std::endl;
  //get_cerr_stream() << "align:          " << dev_align  << std::endl;
    get_cerr_stream() << "max_wg:         " << dev_max_wg << std::endl;
    get_cerr_stream() << "wavefront_size: " << dev_wavefront_size << std::endl;
    }
  
  out_info.is_gpu         = (dev_type == CL_DEVICE_TYPE_GPU);
  out_info.has_float64    = (dev_fp64 != 0);
  out_info.has_sizet64    = (dev_sizet_width >= 8);
  out_info.ptr_width      = uword(dev_ptr_width);
  out_info.n_units        = uword(dev_n_units);
  out_info.opencl_ver     = uword(dev_opencl_ver);
  out_info.max_wg         = uword(dev_max_wg);
  out_info.wavefront_size = uword(dev_wavefront_size);
  
  return (status == CL_SUCCESS);
  }




inline
bool
runtime_t::setup_queue(cl_context& out_context, cl_command_queue& out_queue, cl_platform_id in_plat_id, cl_device_id in_dev_id) const
  {
  coot_extra_debug_sigprint();
  
  cl_context_properties prop[3] = { CL_CONTEXT_PLATFORM, cl_context_properties(in_plat_id), 0 };
  
  cl_int status = 0;
  
  out_context = clCreateContext(prop, 1, &in_dev_id, NULL, NULL, &status);
  
  if((status != CL_SUCCESS) || (out_context == NULL))
    {
    coot_debug_warn(coot_cl_error::as_string(status));
    return false;
    }
  
  // NOTE: clCreateCommandQueue is deprecated as of OpenCL 2.0, but it will be supported for the "foreseeable future"
  // NOTE: clCreateCommandQueue is replaced with clCreateCommandQueueWithProperties in OpenCL 2.0
  // NOTE: http://stackoverflow.com/questions/28500496/opencl-function-found-deprecated-by-visual-studio
    
  out_queue = clCreateCommandQueue(out_context, in_dev_id, 0, &status);
  
  if((status != CL_SUCCESS) || (out_queue == NULL))
    {
    coot_debug_warn(coot_cl_error::as_string(status));
    return false;
    }
  
  return true;
  }



inline
bool
runtime_t::compile_kernels(const std::string& source, std::vector<std::pair<std::string, cl_kernel*>>& names)
  {
  coot_extra_debug_sigprint();
  
  cl_int status;
  
  // TODO: get info using clquery ?
  
  runtime_t::program_wrapper prog_holder;  // program_wrapper will automatically call clReleaseProgram() when it goes out of scope
  

  // cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret);
  // strings = An array of N pointers (where N = count) to optionally null-terminated character strings that make up the source code. 
  // lengths = An array with the number of chars in each string (the string length). If an element in lengths is zero, its accompanying string is null-terminated.
  //           If lengths is NULL, all strings in the strings argument are considered null-terminated.
  //           Any length value passed in that is greater than zero excludes the null terminator in its count. 
  
  
  status = 0;
  
  const char* source_c_str = source.c_str();
  
  prog_holder.prog = clCreateProgramWithSource(ctxt, 1, &source_c_str, NULL, &status);
  
  if((status != CL_SUCCESS) || (prog_holder.prog == NULL))
    {
    cout << "status: " << coot_cl_error::as_string(status) << endl;
    
    std::cout << "coot_cl_rt::compile_kernels(): couldn't create program" << std::endl;
    return false;
    }
  
  std::string build_options = ((sizeof(uword) >= 8) && dev_info.has_sizet64) ? std::string("-D UWORD=ulong") : std::string("-D UWORD=uint");

  // Add the wavefront size to the build options.
  std::ostringstream wavefront_size_options;
  wavefront_size_options << " -D WAVEFRONT_SIZE=" << dev_info.wavefront_size;
  wavefront_size_options << " -D WAVEFRONT_SIZE_NAME=";
  if (dev_info.wavefront_size == 8 || dev_info.wavefront_size == 16 || dev_info.wavefront_size == 32 || dev_info.wavefront_size == 64 || dev_info.wavefront_size == 128)
    {
    wavefront_size_options << dev_info.wavefront_size;
    }
  else
    {
    wavefront_size_options << "other";
    }
  build_options += wavefront_size_options.str();

  status = clBuildProgram(prog_holder.prog, 0, NULL, build_options.c_str(), NULL, NULL);

  if(status != CL_SUCCESS)
    {
    size_t len = 0;

    // Get the length of the error log and then allocate enough space for it.
    clGetProgramBuildInfo(prog_holder.prog, dev_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char* buffer = new char[len];

    clGetProgramBuildInfo(prog_holder.prog, dev_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    std::cout << "coot::cl_rt::compile_kernels(): couldn't build program;"              << std::endl;
    std::cout << "coot::cl_rt::compile_kernels(): output from clGetProgramBuildInfo():" << std::endl;
    std::cout << buffer << std::endl;
    delete buffer;

    return false;
    }


  for (uword i = 0; i < names.size(); ++i)
    {
    (*names.at(i).second) = clCreateKernel(prog_holder.prog, names.at(i).first.c_str(), &status);

    if((status != CL_SUCCESS) || (names.at(i).second == NULL))
      {
      std::cout << coot_cl_error::as_string(status) << endl;
      std::cout << "kernel_name: " << names.at(i).first << endl;
      return false;
      }
    }

  return true;
  }



inline
uword
runtime_t::get_n_units() const
  {
  return (valid) ? dev_info.n_units : uword(0);
  }



inline
uword
runtime_t::get_max_wg() const
  {
  return (valid) ? dev_info.max_wg : uword(0);
  }



inline
uword
runtime_t::get_wavefront_size() const
  {
  return (valid) ? dev_info.wavefront_size : uword(0);
  }



inline
bool
runtime_t::is_valid() const
  {
  return valid;
  }



inline
bool
runtime_t::has_sizet64() const
  {
  return dev_info.has_sizet64;
  }



inline
bool
runtime_t::has_float64() const
  {
  return dev_info.has_float64;
  }



template<typename eT>
inline
cl_mem
runtime_t::acquire_memory(const uword n_elem)
  {
  coot_extra_debug_sigprint();
  
  coot_check_runtime_error( (valid == false), "coot_cl_rt::acquire_memory(): runtime not valid" );
  
  if(n_elem == 0)  { return NULL; }
  
  coot_debug_check
   (
   ( size_t(n_elem) > (std::numeric_limits<size_t>::max() / sizeof(eT)) ),
   "coot_cl_rt::acquire_memory(): requested size is too large"
   );

  cl_int status = 0;
  cl_mem result = clCreateBuffer(ctxt, CL_MEM_READ_WRITE, sizeof(eT)*(std::max)(uword(1), n_elem), NULL, &status);
  
  coot_check_bad_alloc( ((status != CL_SUCCESS) || (result == NULL)), "coot_cl_rt::acquire_memory(): not enough memory on device" );
  
  return result;
  }



inline
void
runtime_t::release_memory(cl_mem dev_mem)
  {
  coot_extra_debug_sigprint();
  
  coot_debug_check( (valid == false), "coot_cl_rt not valid" );
  
  if(dev_mem)  { clReleaseMemObject(dev_mem); }
  }



inline
void
runtime_t::synchronise()
  {
  clFinish(get_cq());
  }



inline
cl_device_id
runtime_t::get_device()
  {
  coot_extra_debug_sigprint();
  
  coot_debug_check( (valid == false), "coot_cl_rt not valid" );
  
  return dev_id;
  }



inline
cl_context
runtime_t::get_context()
  {
  coot_extra_debug_sigprint();
  
  coot_debug_check( (valid == false), "coot_cl_rt not valid" );
  
  return ctxt;
  }



inline
cl_command_queue
runtime_t::get_cq()
  {
  coot_extra_debug_sigprint();
  
  coot_debug_check( (valid == false), "coot_cl_rt not valid" );
  
  return cq;
  }



inline
bool
runtime_t::create_extra_cq(cl_command_queue& out_queue)
  {
  coot_extra_debug_sigprint();
  
  coot_debug_check( (valid == false), "coot_cl_rt not valid" );
  
  cl_int status = 0;
  
  out_queue = clCreateCommandQueue((*this).ctxt, (*this).dev_id, 0, &status);
  
  if((status != CL_SUCCESS) || (out_queue == NULL))
    {
    coot_debug_warn(coot_cl_error::as_string(status));
    return false;
    }
  
  return true;
  }



inline
void
runtime_t::delete_extra_cq(cl_command_queue& in_queue)
  {
  coot_extra_debug_sigprint();
  
  coot_debug_check( (valid == false), "coot_cl_rt not valid" );
  
  if(in_queue != NULL)  { clReleaseCommandQueue(in_queue); in_queue = NULL; }
  }



template<typename eT>
inline
const cl_kernel&
runtime_t::get_kernel(const oneway_kernel_id::enum_id num)
  {
  return get_kernel<eT>(oneway_kernels, num);
  }



template<typename eT>
inline
const cl_kernel&
runtime_t::get_kernel(const oneway_real_kernel_id::enum_id num)
  {
  return get_kernel<eT>(oneway_real_kernels, num);
  }



template<typename eT1, typename eT2>
inline
const cl_kernel&
runtime_t::get_kernel(const twoway_kernel_id::enum_id num)
  {
  return get_kernel<eT1, eT2>(twoway_kernels, num);
  }



template<typename eT1, typename eT2, typename eT3>
inline
const cl_kernel&
runtime_t::get_kernel(const threeway_kernel_id::enum_id num)
  {
  return get_kernel<eT1, eT2, eT3>(threeway_kernels, num);
  }



template<typename eT1, typename... eTs, typename HeldType, typename EnumType>
inline
const cl_kernel&
runtime_t::get_kernel(const rt_common::kernels_t<HeldType>& k, const EnumType num)
  {
  coot_extra_debug_sigprint();

       if(is_same_type<eT1, u32   >::yes) { return get_kernel<eTs...>(k.u32_kernels, num); }
  else if(is_same_type<eT1, s32   >::yes) { return get_kernel<eTs...>(k.s32_kernels, num); }
  else if(is_same_type<eT1, u64   >::yes) { return get_kernel<eTs...>(k.u64_kernels, num); }
  else if(is_same_type<eT1, s64   >::yes) { return get_kernel<eTs...>(k.s64_kernels, num); }
  else if(is_same_type<eT1, float >::yes) { return get_kernel<eTs...>(  k.f_kernels, num); }
  else if(is_same_type<eT1, double>::yes) { return get_kernel<eTs...>(  k.d_kernels, num); }
  else { coot_debug_check(true, "unsupported element type"); }
  }



template<typename eT, typename EnumType>
inline
const cl_kernel&
runtime_t::get_kernel(const rt_common::kernels_t<std::vector<cl_kernel>>& k, const EnumType num)
  {
  coot_extra_debug_sigprint();

       if(is_same_type<eT, u32   >::yes) { return k.u32_kernels.at(num); }
  else if(is_same_type<eT, s32   >::yes) { return k.s32_kernels.at(num); }
  else if(is_same_type<eT, u64   >::yes) { return k.u64_kernels.at(num); }
  else if(is_same_type<eT, s64   >::yes) { return k.s64_kernels.at(num); }
  else if(is_same_type<eT, float >::yes) { return   k.f_kernels.at(num); }
  else if(is_same_type<eT, double>::yes) { return   k.d_kernels.at(num); }
  else { coot_debug_check(true, "unsupported element type"); }
  }



//
// program_wrapper

inline
runtime_t::program_wrapper::program_wrapper()
  {
  coot_extra_debug_sigprint();
  
  prog = NULL;
  }



inline
runtime_t::program_wrapper::~program_wrapper()
  {
  coot_extra_debug_sigprint();
  
  if(prog != NULL)  { clReleaseProgram(prog); }
  }






//
// cq_guard

inline
runtime_t::cq_guard::cq_guard()
  {
  coot_extra_debug_sigprint();
  
  get_rt().cl_rt.lock();
  
  if(get_rt().cl_rt.is_valid())
    {
    coot_extra_debug_print("calling clFinish()");
    clFinish(get_rt().cl_rt.get_cq());  // force synchronisation
    
    //coot_extra_debug_print("calling clFlush()");
    //clFlush(get_rt().cl_rt.get_cq());  // submit all enqueued commands
    }
  }



inline
runtime_t::cq_guard::~cq_guard()
  {
  coot_extra_debug_sigprint();
  
  if(get_rt().cl_rt.is_valid())
    {
    coot_extra_debug_print("calling clFlush()");
    clFlush(get_rt().cl_rt.get_cq());  // submit all enqueued commands
    }
  
  get_rt().cl_rt.unlock();
  }




//
// adapt_uword

inline
runtime_t::adapt_uword::adapt_uword(const uword val)
  {
  if((sizeof(uword) >= 8) && get_rt().cl_rt.has_sizet64())
    {
    size  = sizeof(u64);
    addr  = (void*)(&val64);
    val64 = u64(val);
    }
  else
    {
    size   = sizeof(u32);
    addr   = (void*)(&val32);
    val32  = u32(val);
    
    coot_check_runtime_error( ((sizeof(uword) >= 8) && (val > 0xffffffffU)), "adapt_uword: given value doesn't fit into unsigned 32 bit integer" );
    }
  }