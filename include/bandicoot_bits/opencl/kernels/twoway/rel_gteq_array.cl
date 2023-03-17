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

__kernel
void
COOT_FN(PREFIX,rel_gteq_array)(__global UWORD* out,
                               __global const eT1* X,
                               __global const eT2* Y,
                               const UWORD n_elem)
  {
  const UWORD i = get_global_id(0);

  if (i < n_elem)
    {
    const eT1 val1 = X[i];
    const eT2 val2 = Y[i];
    out[i] = (val1 >= val2);
    }
  }
