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

// xorwow will use one state per warp item

// Note that we use curand instead!  This kernel is only included for consistency with the OpenCL kernels,
// and so that the list of oneway_real kernels is identical across backends.

// See algorithm "xorwow" from page 5 of "Xorshift RNGs" by George Marsaglia.
__device__
inline
uint_eT1
COOT_FN(PREFIX,inplace_xorwow)(uint_eT1* xorwow_state)
  {
  // xorwow_state[0] through xorwow_state[4] represent the 5 state integers,
  // and xorwow_state[5] holds the counter.
  uint_eT1 t = xorwow_state[4];
  uint_eT1 s = xorwow_state[0];

  xorwow_state[4] = xorwow_state[3];
  xorwow_state[3] = xorwow_state[2];
  xorwow_state[2] = xorwow_state[1];
  xorwow_state[1] = s;

  t ^= (t >> 2);
  t ^= (t << 1);
  t ^= (s ^ (s << 4));
  xorwow_state[0] = t;
  xorwow_state[5] += 362437;
  return t + xorwow_state[5];
  }



__global__
void
COOT_FN(PREFIX,inplace_xorwow_randu)(eT1* mem,
                                     uint_eT1* xorwow_state,
                                     const UWORD n_elem)
  {
  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD tid = threadIdx.x;
  if (i < n_elem)
    {
    uint_eT1 t = COOT_FN(PREFIX,inplace_xorwow)(xorwow_state + 6 * tid);
    // Now normalize to [0, 1] and compute the output.
    mem[i] = (t / (eT1) COOT_FN(coot_type_max_u_,eT1)());
    }
  }
