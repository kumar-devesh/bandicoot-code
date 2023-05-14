// Copyright 2021 Ryan Curtin (https://www.ratml.org/)
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

void
COOT_FN(PREFIX,max_wavefront_reduce_other)(__local volatile eT1* data, UWORD tid)
  {
  for(UWORD i = WAVEFRONT_SIZE; i > 0; i >>= 1)
    {
    if (tid < i)
      data[tid] = max(data[tid], data[tid + i]);
    }
  }



void
COOT_FN(PREFIX,max_wavefront_reduce_8)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] = max(data[tid], data[tid + 8]);
  data[tid] = max(data[tid], data[tid + 4]);
  data[tid] = max(data[tid], data[tid + 2]);
  data[tid] = max(data[tid], data[tid + 1]);
  }



void
COOT_FN(PREFIX,max_wavefront_reduce_16)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] = max(data[tid], data[tid + 16]);
  data[tid] = max(data[tid], data[tid + 8]);
  data[tid] = max(data[tid], data[tid + 4]);
  data[tid] = max(data[tid], data[tid + 2]);
  data[tid] = max(data[tid], data[tid + 1]);
  }



void
COOT_FN(PREFIX,max_wavefront_reduce_32)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] = max(data[tid], data[tid + 32]);
  data[tid] = max(data[tid], data[tid + 16]);
  data[tid] = max(data[tid], data[tid + 8]);
  data[tid] = max(data[tid], data[tid + 4]);
  data[tid] = max(data[tid], data[tid + 2]);
  data[tid] = max(data[tid], data[tid + 1]);
  }



void
COOT_FN(PREFIX,max_wavefront_reduce_64)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] = max(data[tid], data[tid + 64]);
  data[tid] = max(data[tid], data[tid + 32]);
  data[tid] = max(data[tid], data[tid + 16]);
  data[tid] = max(data[tid], data[tid + 8]);
  data[tid] = max(data[tid], data[tid + 4]);
  data[tid] = max(data[tid], data[tid + 2]);
  data[tid] = max(data[tid], data[tid + 1]);
  }



void
COOT_FN(PREFIX,max_wavefront_reduce_128)(__local volatile eT1* data, UWORD tid)
  {
  data[tid] = max(data[tid], data[tid + 128]);
  data[tid] = max(data[tid], data[tid + 64]);
  data[tid] = max(data[tid], data[tid + 32]);
  data[tid] = max(data[tid], data[tid + 16]);
  data[tid] = max(data[tid], data[tid + 8]);
  data[tid] = max(data[tid], data[tid + 4]);
  data[tid] = max(data[tid], data[tid + 2]);
  data[tid] = max(data[tid], data[tid + 1]);
  }
