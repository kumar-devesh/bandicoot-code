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
// TODO: these currently generate random numbers on the CPU and then send them to the GPU.  That's awful!

template<typename eT>
inline void fill_randu(dev_mem_t<eT> dest, const uword n);

template<typename eT>
inline void fill_randn(dev_mem_t<eT> dest, const uword n);

// TODO: fill_randi, etc...
