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

// Utility functions for generating random numbers via OpenCL.

template<typename eT> struct preferred_rng { };
template<> struct preferred_rng<u32> { typedef std::mt19937 result; };
template<> struct preferred_rng<u64> { typedef std::mt19937_64 result; };

// Should be called with an unsigned int as the eT type.
template<typename eT>
inline
void
init_xorwow_state(cl_mem xorwow_state, const size_t num_rng_threads)
  {
  coot_extra_debug_sigprint();

  // TODO: allow modification of seed

  // Since the states are relatively small, and we only do the seeding once, we'll initialize the values on the CPU, then copy them over.
  // We ensure that all values are odd.
  arma::Row<eT> cpu_state(6 * num_rng_threads, arma::fill::none);
  typename preferred_rng<eT>::result rng;
  for (size_t i = 0; i < cpu_state.n_elem; ++i)
    {
    eT val = rng();
    if (val % 2 == 0)
      val += 1;
    cpu_state[i] = val;
    }

  // Copy the state to the GPU memory.
  dev_mem_t<eT> m;
  m.cl_mem_ptr = xorwow_state;
  copy_into_dev_mem(m, cpu_state.memptr(), 6 * num_rng_threads);
  }

template<typename eT>
inline
void
fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // For integral types, truncate to [0, 1] just like Armadillo.
  // We'll generate numbers using a floating-point type of the same width, then pass over it to truncate and cast back tot he right type.

  if (std::is_same<eT, u32>::value || std::is_same<eT, s32>::value)
    {
    dev_mem_t<float> reinterpreted_mem;
    reinterpreted_mem.cl_mem_ptr = dest.cl_mem_ptr;
    fill_randu(reinterpreted_mem, n);
    copy_array(dest, reinterpreted_mem, n);
    }
  else if (std::is_same<eT, u64>::value || std::is_same<eT, s64>::value)
    {
    dev_mem_t<double> reinterpreted_mem;
    reinterpreted_mem.cl_mem_ptr = dest.cl_mem_ptr;
    fill_randu(reinterpreted_mem, n);
    copy_array(dest, reinterpreted_mem, n);
    }
  else if (std::is_same<eT, float>::value || std::is_same<eT, double>::value)
    {
    // Get the kernel and set up to run it.
    cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::inplace_xorwow_randu);

    runtime_t::cq_guard guard;
    runtime_t::adapt_uword n_cl(n);

    cl_int status = 0;

    cl_mem xorwow_state = get_rt().cl_rt.get_xorwow_state<eT>();

    status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr) );
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &(xorwow_state) );
    status |= clSetKernelArg(kernel, 2, n_cl.size,      n_cl.addr );

    // Each thread will do as many elements as it can.
    // This avoids memory synchronization issues, since each RNG state will be local to only a single run of the kernel.
    const size_t num_rng_threads = get_rt().cl_rt.get_num_rng_threads();
    const size_t num_threads = std::min(num_rng_threads, n);

    status |= clEnqueueNDRangeKernel(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &num_threads, NULL, 0, NULL, NULL);

    coot_check_cl_error(status, "randu()");
    }
  else
    {
    std::ostringstream oss;
    oss << "coot::opencl::fill_randu(): not implemented for type " << typeid(eT).name();
    coot_stop_runtime_error(oss.str());
    }
  }



template<typename eT>
inline
void
fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // TODO: replace awful implementation
/*
  // Generate n random numbers.
  eT* cpu_rand = new eT[n];

  std::mt19937 gen;
  std::normal_distribution<eT> n_distr;
  for (uword i = 0; i < n; ++i)
    {
    cpu_rand[i] = n_distr(gen);
    }

  // Now push it to the device.
  opencl::runtime_t::cq_guard guard;

  cl_int status = clEnqueueWriteBuffer(get_rt().cl_rt.get_cq(), dest.cl_mem_ptr, CL_TRUE, 0, sizeof(eT) * n, cpu_rand, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "coot::opencl::fill_randn(): couldn't access device memory");

  delete[] cpu_rand;
*/
  }



// TODO: fill_randi, etc...
