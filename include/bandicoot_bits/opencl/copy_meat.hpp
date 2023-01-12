// Copyright 2019 Ryan Curtin (http://www.ratml.org)
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



template<typename eT>
inline
void
copy_from_dev_mem(eT* dest, const dev_mem_t<eT> src, const uword N)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  // use a blocking call
  const cl_int status = clEnqueueReadBuffer(get_rt().cl_rt.get_cq(), src.cl_mem_ptr, CL_TRUE, 0, sizeof(eT) * N, dest, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "Mat::copy_from_dev_mem(): couldn't access device memory" );
  }



template<typename eT>
inline
void
copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  // use a blocking call
  cl_int status = clEnqueueWriteBuffer(get_rt().cl_rt.get_cq(), dest.cl_mem_ptr, CL_TRUE, 0, sizeof(eT)*N, src, 0, NULL, NULL);

  coot_check_runtime_error( (status != CL_SUCCESS), "Mat::write_dev_mem(): couldn't access device memory" );
  }
