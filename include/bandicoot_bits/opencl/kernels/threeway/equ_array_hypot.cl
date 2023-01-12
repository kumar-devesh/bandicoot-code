// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,equ_array_hypot)(__global eT3* out,
                                __global const eT1* A,
                                __global const eT2* B,
                                const UWORD N)
  {
  const UWORD i = get_global_id(0);
  if(i < N)
    {
    const fp_eT3 a_val = (fp_eT3) A[i];
    const fp_eT3 b_val = (fp_eT3) B[i];
    out[i] = (eT3) hypot(a_val, b_val);
    }
  }
