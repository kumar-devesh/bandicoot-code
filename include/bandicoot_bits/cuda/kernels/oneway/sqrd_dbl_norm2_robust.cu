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
COOT_FN(PREFIX,sqrd_dbl_norm2_robust)(const eT1* in_mem,
                                      const UWORD n_elem,
                                      double* out_mem,
                                      const double max_val)
  {
  double* aux_mem = (double*) aux_shared_mem;

  const UWORD tid = threadIdx.x;
  UWORD i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  const UWORD grid_size = blockDim.x * 2 * gridDim.x;

  aux_mem[tid] = 0;

  while (i + blockDim.x < n_elem)
    {
    // copy to local shared memory
    double v1 = ((double) in_mem[i]) / max_val;
    double v2 = ((double) in_mem[i + blockDim.x]) / max_val;
    aux_mem[tid] += (v1 * v1) + (v2 * v2);
    i += grid_size;
    }
  if (i < n_elem)
    {
    double v = ((double) in_mem[i]) / max_val;
    aux_mem[tid] += (v * v);
    }
  __syncthreads();

  for (UWORD s = blockDim.x / 2; s > 32; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] += aux_mem[tid + s];
      }
    __syncthreads();
  }

  if (tid < 32) // unroll last warp's worth of work
    {
    // Since we are just accumulating now, we can use the accu() warp-reduce function.
    COOT_FN(PREFIX,accu_warp_reduce)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[blockIdx.x] = aux_mem[0];
    }
  }
