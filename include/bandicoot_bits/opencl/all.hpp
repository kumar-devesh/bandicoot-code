// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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



// Determine whether all elements in the memory satisfy the conditions imposed by the kernel `num` (and its small version `num_small`).
template<typename eT1, typename eT2>
inline
bool
all_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small)
  {

  // TODO

  return false;
  }
