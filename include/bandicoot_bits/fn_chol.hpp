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



// TODO: add optional 'layout' argument
template<typename T1>
inline
bool
chol(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type, T1>& X)
  {
  coot_extra_debug_sigprint();

  out = X.get_ref();

  return coot_rt_t::chol(out.get_dev_mem(true), out.n_rows);
  }
