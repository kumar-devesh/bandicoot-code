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

// Forward declarations we may need.
void COOT_FN(PREFIX,min_wavefront_reduce_other)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,min_wavefront_reduce_8)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,min_wavefront_reduce_16)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,min_wavefront_reduce_32)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,min_wavefront_reduce_64)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,min_wavefront_reduce_128)(__local volatile eT1* data, UWORD tid);



__kernel
void
COOT_FN(PREFIX,norm_min)(__global const eT1* in_mem,
                         const UWORD n_elem,
                         __global eT1* out_mem,
                         __local volatile eT1* aux_mem)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  if (i < n_elem)
    {
    aux_mem[tid] = ET1_ABS(in_mem[i]);
    }
  if (i + get_local_size(0) < n_elem)
    {
    const eT1 v = ET1_ABS(in_mem[i + get_local_size(0)]);
    aux_mem[tid] = min(aux_mem[tid], v);
    }
  i += grid_size;

  while (i + get_local_size(0) < n_elem)
    {
    const eT1 v = min(ET1_ABS(in_mem[i]), ET1_ABS(in_mem[i + get_local_size(0)]));
    aux_mem[tid] = min(aux_mem[tid], v);
    i += grid_size;
    }
  if (i < n_elem)
    {
    aux_mem[tid] = min(aux_mem[tid], ET1_ABS(in_mem[i]));
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (UWORD s = get_local_size(0) / 2; s > WAVEFRONT_SIZE; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] = min(aux_mem[tid], aux_mem[tid + s]);
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (tid < WAVEFRONT_SIZE)
    {
    COOT_FN_3(PREFIX,min_wavefront_reduce_,WAVEFRONT_SIZE_NAME)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[get_group_id(0)] = aux_mem[0];
    }
  }
