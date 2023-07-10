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

#ifdef COOT_USE_OPENCL

namespace coot
  {
  #include "bandicoot_bits/opencl/def_opencl.hpp"

  extern "C"
    {



    //
    // setup functions
    //



    cl_context wrapper_clCreateContext(const cl_context_properties* properties,
                                       cl_uint num_devices,
                                       const cl_device_id* devices,
                                       void (CL_CALLBACK* pfn_notify)(const char* errinfo,
                                                                      const void* private_info,
                                                                      size_t cb,
                                                                      void* user_data),
                                       void* user_data,
                                       cl_int* errcode_ret)
      {
      return clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
      }



    cl_command_queue wrapper_clCreateCommandQueue(cl_context context,
                                                  cl_device_id device,
                                                  cl_command_queue_properties properties,
                                                  cl_int* errcode_ret)
      {
      return clCreateCommandQueue(context, device, properties, errcode_ret);
      }



    cl_int wrapper_clGetPlatformIDs(cl_uint num_entries,
                                    cl_platform_id* platforms,
                                    cl_uint* num_platforms)
      {
      return clGetPlatformIDs(num_entries, platforms, num_platforms);
      }



    cl_int wrapper_clGetDeviceIDs(cl_platform_id platform,
                                  cl_device_type device_type,
                                  cl_uint num_entries,
                                  cl_device_id* devices,
                                  cl_uint* num_devices)
      {
      return clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
      }



    cl_int wrapper_clGetDeviceInfo(cl_device_id device,
                                   cl_device_info param_name,
                                   size_t param_value_size,
                                   void* param_value,
                                   size_t* param_value_size_ret)
      {
      return clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
      }



    //
    // kernel compilation
    //



    cl_program wrapper_clCreateProgramWithSource(cl_context context,
                                                 cl_uint count,
                                                 const char** strings,
                                                 const size_t* lengths,
                                                 cl_int* errcode_ret)
      {
      return clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
      }



    cl_program wrapper_clCreateProgramWithBinary(cl_context context,
                                                 cl_uint num_devices,
                                                 const cl_device_id* device_list,
                                                 const size_t* lengths,
                                                 const unsigned char** binaries,
                                                 cl_int* binary_status,
                                                 cl_int* errcode_ret)
      {
      return clCreateProgramWithBinary(context,
                                       num_devices,
                                       device_list,
                                       lengths,
                                       binaries,
                                       binary_status,
                                       errcode_ret);
      }



    cl_int wrapper_clBuildProgram(cl_program program,
                                  cl_uint num_devices,
                                  const cl_device_id* device_list,
                                  const char* options,
                                  void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                                  void* user_data)
      {
      return clBuildProgram(program,
                            num_devices,
                            device_list,
                            options,
                            pfn_notify,
                            user_data);
      }



    cl_int wrapper_clGetProgramBuildInfo(cl_program program,
                                         cl_device_id device,
                                         cl_program_build_info param_name,
                                         size_t param_value_size,
                                         void* param_value,
                                         size_t* param_value_size_ret)
      {
      return clGetProgramBuildInfo(program,
                                   device,
                                   param_name,
                                   param_value_size,
                                   param_value,
                                   param_value_size_ret);
      }



    cl_int wrapper_clGetProgramInfo(cl_program program,
                                    cl_program_info param_name,
                                    size_t param_value_size,
                                    void* param_value,
                                    size_t* param_value_size_ret)
      {
      return clGetProgramInfo(program,
                              param_name,
                              param_value_size,
                              param_value,
                              param_value_size_ret);
      }



    cl_kernel wrapper_clCreateKernel(cl_program program,
                                     const char* kernel_name,
                                     cl_int* errcode_ret)
      {
      return clCreateKernel(program, kernel_name, errcode_ret);
      }



    #if defined(CL_VERSION_2_1)
    cl_int wrapper_clGetKernelSubGroupInfo(cl_kernel kernel,
                                           cl_device_id device,
                                           cl_kernel_sub_group_info param_name,
                                           size_t input_value_size,
                                           const void* input_value,
                                           size_t param_value_size,
                                           void* param_value,
                                           size_t* param_value_size_ret)
      {
      return clGetKernelSubGroupInfo(kernel,
                                     device,
                                     param_name,
                                     input_value_size,
                                     input_value,
                                     param_value_size,
                                     param_value,
                                     param_value_size_ret);
      }
    #elif defined(cl_khr_subgroups) || defined(cl_intel_subgroups)
    cl_int wrapper_clGetKernelSubGroupInfoKHR(cl_kernel kernel,
                                              cl_device_id device,
                                              cl_kernel_sub_group_info param_name,
                                              size_t input_value_size,
                                              const void* input_value,
                                              size_t param_value_size,
                                              void* param_value,
                                              size_t* param_value_size_ret)
      {
      return clGetKernelSubGroupInfoKHR(kernel,
                                        device,
                                        param_name,
                                        input_value_size,
                                        input_value,
                                        param_value_size,
                                        param_value,
                                        param_value_size_ret);
      }
    #endif



    //
    // memory handling functions
    //



    cl_mem wrapper_clCreateBuffer(cl_context context,
                                  cl_mem_flags flags,
                                  size_t size,
                                  void* host_ptr,
                                  cl_int* errcode_ret)
      {
      return clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
      }



    cl_int wrapper_clEnqueueReadBuffer(cl_command_queue command_queue,
                                       cl_mem buffer,
                                       cl_bool blocking_read,
                                       size_t offset,
                                       size_t size,
                                       void* ptr,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list,
                                       cl_event* event)
      {
      return clEnqueueReadBuffer(command_queue,
                                 buffer,
                                 blocking_read,
                                 offset,
                                 size,
                                 ptr,
                                 num_events_in_wait_list,
                                 event_wait_list,
                                 event);
      }



    cl_int wrapper_clEnqueueWriteBuffer(cl_command_queue command_queue,
                                        cl_mem buffer,
                                        cl_bool blocking_write,
                                        size_t offset,
                                        size_t size,
                                        const void* ptr,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event* event_wait_list,
                                        cl_event* event)
      {
      return clEnqueueWriteBuffer(command_queue,
                                  buffer,
                                  blocking_write,
                                  offset,
                                  size,
                                  ptr,
                                  num_events_in_wait_list,
                                  event_wait_list,
                                  event);
      }



    cl_int wrapper_clEnqueueReadBufferRect(cl_command_queue command_queue,
                                           cl_mem buffer,
                                           cl_bool blocking_read,
                                           const size_t* buffer_origin,
                                           const size_t* host_origin,
                                           const size_t* region,
                                           size_t buffer_row_pitch,
                                           size_t buffer_slice_pitch,
                                           size_t host_row_pitch,
                                           size_t host_slice_pitch,
                                           void* ptr,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list,
                                           cl_event* event)
      {
      return clEnqueueReadBufferRect(command_queue,
                                     buffer,
                                     blocking_read,
                                     buffer_origin,
                                     host_origin,
                                     region,
                                     buffer_row_pitch,
                                     buffer_slice_pitch,
                                     host_row_pitch,
                                     host_slice_pitch,
                                     ptr,
                                     num_events_in_wait_list,
                                     event_wait_list,
                                     event);
      }



    cl_int wrapper_clEnqueueWriteBufferRect(cl_command_queue command_queue,
                                            cl_mem buffer,
                                            cl_bool blocking_write,
                                            const size_t* buffer_origin,
                                            const size_t* host_origin,
                                            const size_t* region,
                                            size_t buffer_row_pitch,
                                            size_t buffer_slice_pitch,
                                            size_t host_row_pitch,
                                            size_t host_slice_pitch,
                                            void* ptr,
                                            cl_uint num_events_in_wait_list,
                                            const cl_event* event_wait_list,
                                            cl_event* event)
      {
      return clEnqueueWriteBufferRect(command_queue,
                                      buffer,
                                      blocking_write,
                                      buffer_origin,
                                      host_origin,
                                      region,
                                      buffer_row_pitch,
                                      buffer_slice_pitch,
                                      host_row_pitch,
                                      host_slice_pitch,
                                      ptr,
                                      num_events_in_wait_list,
                                      event_wait_list,
                                      event);
      }



    void* wrapper_clEnqueueMapBuffer(cl_command_queue command_queue,
                                     cl_mem buffer,
                                     cl_bool blocking_map,
                                     cl_map_flags map_flags,
                                     size_t offset,
                                     size_t size,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list,
                                     cl_event* event,
                                     cl_int* errcode_ret)
      {
      return clEnqueueMapBuffer(command_queue,
                                buffer,
                                blocking_map,
                                map_flags,
                                offset,
                                size,
                                num_events_in_wait_list,
                                event_wait_list,
                                event,
                                errcode_ret);
      }



    cl_int wrapper_clEnqueueUnmapMemObject(cl_command_queue command_queue,
                                           cl_mem memobj,
                                           void* mapped_ptr,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list,
                                           cl_event* event)
      {
      return clEnqueueUnmapMemObject(command_queue,
                                     memobj,
                                     mapped_ptr,
                                     num_events_in_wait_list,
                                     event_wait_list,
                                     event);
      }



    cl_int wrapper_clEnqueueCopyBuffer(cl_command_queue command_queue,
                                       cl_mem src_buffer,
                                       cl_mem dst_buffer,
                                       size_t src_offset,
                                       size_t dst_offset,
                                       size_t size,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event* event_wait_list,
                                       cl_event* event)
      {
      return clEnqueueCopyBuffer(command_queue,
                                 src_buffer,
                                 dst_buffer,
                                 src_offset,
                                 dst_offset,
                                 size,
                                 num_events_in_wait_list,
                                 event_wait_list,
                                 event);
      }



    cl_int wrapper_clEnqueueCopyBufferRect(cl_command_queue command_queue,
                                           cl_mem src_buffer,
                                           cl_mem dst_buffer,
                                           const size_t* src_origin,
                                           const size_t* dst_origin,
                                           const size_t* region,
                                           size_t src_row_pitch,
                                           size_t src_slice_pitch,
                                           size_t dst_row_pitch,
                                           size_t dst_slice_pitch,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event* event_wait_list,
                                           cl_event* event)
      {
      return clEnqueueCopyBufferRect(command_queue,
                                     src_buffer,
                                     dst_buffer,
                                     src_origin,
                                     dst_origin,
                                     region,
                                     src_row_pitch,
                                     src_slice_pitch,
                                     dst_row_pitch,
                                     dst_slice_pitch,
                                     num_events_in_wait_list,
                                     event_wait_list,
                                     event);
      }



    //
    // running kernels
    //



    cl_int wrapper_clSetKernelArg(cl_kernel kernel,
                                  cl_uint arg_index,
                                  size_t arg_size,
                                  const void* arg_value)
      {
      return clSetKernelArg(kernel, arg_index, arg_size, arg_value);
      }



    cl_int wrapper_clGetKernelWorkGroupInfo(cl_kernel kernel,
                                            cl_device_id device,
                                            cl_kernel_work_group_info param_name,
                                            size_t param_value_size,
                                            void* param_value,
                                            size_t* param_value_size_ret)
      {
      return clGetKernelWorkGroupInfo(kernel,
                                      device,
                                      param_name,
                                      param_value_size,
                                      param_value,
                                      param_value_size_ret);
      }



    cl_int wrapper_clEnqueueNDRangeKernel(cl_command_queue command_queue,
                                          cl_kernel kernel,
                                          cl_uint work_dim,
                                          const size_t* global_work_offset,
                                          const size_t* global_work_size,
                                          const size_t* local_work_size,
                                          cl_uint num_events_in_wait_list,
                                          const cl_event* event_wait_list,
                                          cl_event* event)
      {
      return clEnqueueNDRangeKernel(command_queue,
                                    kernel,
                                    work_dim,
                                    global_work_offset,
                                    global_work_size,
                                    local_work_size,
                                    num_events_in_wait_list,
                                    event_wait_list,
                                    event);
      }



    cl_int wrapper_clEnqueueTask(cl_command_queue command_queue,
                                 cl_kernel kernel,
                                 cl_uint num_events_in_wait_list,
                                 const cl_event* event_wait_list,
                                 cl_event* event)
      {
      return clEnqueueTask(command_queue,
                           kernel,
                           num_events_in_wait_list,
                           event_wait_list,
                           event);
      }



    //
    // synchronisation
    //



    cl_int wrapper_clFinish(cl_command_queue command_queue)
      {
      return clFinish(command_queue);
      }



    cl_int wrapper_clFlush(cl_command_queue command_queue)
      {
      return clFlush(command_queue);
      }



    //
    // cleanup
    //



    cl_int wrapper_clReleaseMemObject(cl_mem memobj)
      {
      return clReleaseMemObject(memobj);
      }



    cl_int wrapper_clReleaseKernel(cl_kernel kernel)
      {
      return clReleaseKernel(kernel);
      }



    cl_int wrapper_clReleaseProgram(cl_program program)
      {
      return clReleaseProgram(program);
      }



    cl_int wrapper_clReleaseCommandQueue(cl_command_queue command_queue)
      {
      return clReleaseCommandQueue(command_queue);
      }



    cl_int wrapper_clReleaseContext(cl_context context)
      {
      return clReleaseContext(context);
      }



    } // extern "C"
  } // namespace coot

#endif
