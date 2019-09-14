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

// utility wrapper functions for CUDA; if Bandicoot is not configured for CUDA,
// then every function here throws an exception.  The CUDA headers have all
// already been imported by the 'bandicoot' header if they are needed.

struct cuda_wrapper
  {
  public:

  template<typename eT>
  static eT* acquire_memory(const uword n_elem);

  static void release_memory(void* memory);

  static void throw_unsupported();
  };
