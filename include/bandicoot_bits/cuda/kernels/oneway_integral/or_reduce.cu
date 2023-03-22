// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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

// this kernel is technically incorrect if the size is not a factor of 2!
__global__
void
COOT_FN(PREFIX,or_reduce)(const eT1* in_mem,
                          const UWORD n_elem,
                          eT1* out_mem)
  {
  eT1* aux_mem = (eT1*) aux_shared_mem;

  const UWORD tid = threadIdx.x;
  UWORD i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  const UWORD grid_size = blockDim.x * 2 * gridDim.x;

  // Make sure all auxiliary memory is initialized to something that won't
  // screw up the final reduce.

  aux_mem[tid] = (eT1) 0;

  while (i + blockDim.x < n_elem)
    {
    aux_mem[tid] |= in_mem[i];
    aux_mem[tid] |= in_mem[i + blockDim.x];
    i += grid_size;
    }
  if (i < n_elem)
    {
    aux_mem[tid] |= in_mem[i];
    }
  __syncthreads();

  for (UWORD s = blockDim.x / 2; s > 32; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] |= aux_mem[tid + s];
      }
    __syncthreads();
  }

  if (tid < 32) // unroll last warp's worth of work
    {
    COOT_FN(PREFIX,or_warp_reduce)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[blockIdx.x] = aux_mem[0];
    }
  }
