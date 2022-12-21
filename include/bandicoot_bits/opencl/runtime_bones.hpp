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



struct runtime_dev_info
  {
  public:

  coot_aligned bool  is_gpu;
  coot_aligned bool  has_float64;
  coot_aligned bool  has_sizet64;
  coot_aligned uword ptr_width;
  coot_aligned uword n_units;
  coot_aligned uword opencl_ver;
  coot_aligned uword max_wg;
  coot_aligned uword wavefront_size;

  inline
  void
  reset()
    {
    is_gpu         = false;
    has_float64    = false;
    has_sizet64    = false;
    n_units        = 0;
    ptr_width      = 0;
    opencl_ver     = 0;
    max_wg         = 0;
    wavefront_size = 0;
    }

  inline runtime_dev_info()  { reset(); }
  };



// TODO: if this is placed into a run-time library and executed there, what happens when two programs use the run-time library at the same time?
class runtime_t
  {
  public:

  inline ~runtime_t();
  inline  runtime_t();

  inline bool init(const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info);

  #if defined(COOT_USE_CXX11)
                   runtime_t(const runtime_t&) = delete;
  runtime_t&       operator=(const runtime_t&) = delete;
  #endif

  inline uword get_n_units() const;
  inline uword get_max_wg() const;
  inline uword get_wavefront_size() const;

  inline bool is_valid()    const;
  inline bool has_sizet64() const;
  inline bool has_float64() const;

  template<typename eT>
  inline cl_mem acquire_memory(const uword n_elem);

  inline void release_memory(cl_mem dev_mem);

  inline void synchronise();

  inline cl_device_id     get_device();
  inline cl_context       get_context();
  inline cl_command_queue get_cq();

  inline bool create_extra_cq(cl_command_queue& out_queue);
  inline void delete_extra_cq(cl_command_queue&  in_queue);

  // TODO: add function to return info about device as a string

  template<typename eT1>
  inline const cl_kernel& get_kernel(const oneway_kernel_id::enum_id num);

  template<typename eT1>
  inline const cl_kernel& get_kernel(const oneway_real_kernel_id::enum_id num);

  template<typename eT2, typename eT1>
  inline const cl_kernel& get_kernel(const twoway_kernel_id::enum_id num);

  template<typename eT3, typename eT2, typename eT1>
  inline const cl_kernel& get_kernel(const threeway_kernel_id::enum_id num);

  // Get random number generator.

  template<typename eT> cl_mem get_xorwow_state() const;
  inline size_t get_num_rng_threads() const;

  class program_wrapper;
  class cq_guard;
  class adapt_uword;

  friend class cq_guard;  // explicitly allows cq_guard to call lock() and unlock()


  private:


  coot_aligned bool             valid;

  coot_aligned cl_platform_id   plt_id;
  coot_aligned cl_device_id     dev_id;
  coot_aligned cl_context       ctxt;
  coot_aligned cl_command_queue cq;

  coot_aligned runtime_dev_info dev_info;

  coot_aligned rt_common::kernels_t<std::vector<cl_kernel>>                                             oneway_kernels;
  coot_aligned rt_common::kernels_t<std::vector<cl_kernel>>                                             oneway_real_kernels;
  coot_aligned rt_common::kernels_t<rt_common::kernels_t<std::vector<cl_kernel>>>                       twoway_kernels;
  coot_aligned rt_common::kernels_t<rt_common::kernels_t<rt_common::kernels_t<std::vector<cl_kernel>>>> threeway_kernels;

  // Internally-held RNG state.
  coot_aligned cl_mem   f_xorwow_state;
  coot_aligned cl_mem   d_xorwow_state;
  coot_aligned size_t   num_rng_threads;

  #if defined(COOT_USE_CXX11)
  coot_aligned std::recursive_mutex mutex;
  #endif

  inline void   lock();  //! NOTE: do not call this function directly; instead instantiate the cq_guard class inside a relevant scope
  inline void unlock();  //! NOTE: do not call this function directly; it's automatically called when an instance of cq_guard goes out of scope

  inline void internal_cleanup();

  inline bool search_devices(cl_platform_id& out_plat_id, cl_device_id& out_dev_id, const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info) const;

  inline bool interrogate_device(runtime_dev_info& out_info, cl_platform_id in_plat_id, cl_device_id in_dev_id, const bool print_info) const;

  inline bool setup_queue(cl_context& out_context, cl_command_queue& out_queue, cl_platform_id in_plat_id, cl_device_id in_dev_id) const;

  inline std::string unique_host_device_id() const;

  inline bool load_cached_kernels(const std::string& unique_host_device_id,
                                  const size_t kernel_size);

  inline bool compile_kernels(const std::string& unique_host_id);

  inline bool create_kernels(const std::vector<std::pair<std::string, cl_kernel*>>& name_map,
                             runtime_t::program_wrapper& prog_holder,
                             const std::string& build_options);

  inline bool cache_kernels(const std::string& unique_host_device_id,
                            runtime_t::program_wrapper& prog_holder) const;

  template<typename eT1, typename... eTs, typename HeldType, typename EnumType>
  inline
  const cl_kernel&
  get_kernel(const rt_common::kernels_t<HeldType>& k, const EnumType num);

  template<typename eT, typename EnumType>
  inline
  const cl_kernel&
  get_kernel(const rt_common::kernels_t<std::vector<cl_kernel>>& k, const EnumType num);
  };



class runtime_t::program_wrapper
  {
  public:

  coot_aligned cl_program prog;  // cl_program is typedef for struct _cl_program*

  inline  program_wrapper();
  inline ~program_wrapper();

  #if defined(COOT_USE_CXX11)
                   program_wrapper(const program_wrapper&) = delete;
  program_wrapper&       operator=(const program_wrapper&) = delete;
  #endif
  };


class runtime_t::cq_guard
  {
  public:

  inline  cq_guard();
  inline ~cq_guard();

  #if defined(COOT_USE_CXX11)
             cq_guard(const cq_guard&) = delete;
  cq_guard& operator=(const cq_guard&) = delete;
  #endif
  };



class runtime_t::adapt_uword
  {
  public:

  coot_aligned size_t size;
  coot_aligned void*  addr;
  coot_aligned u64    val64;
  coot_aligned u32    val32;

  inline adapt_uword(const uword val);
  };
