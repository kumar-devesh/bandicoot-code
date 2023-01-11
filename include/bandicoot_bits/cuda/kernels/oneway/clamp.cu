// Copyright 2021 Marcus Edel (http://www.kurg.org/)
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
__global__
void
COOT_FN(PREFIX,clamp)(const eT1* A_mem,
                      const eT1 min_val,
                      const eT1 max_val,
                      const UWORD num,
                      eT1* out_mem)
  {
  UWORD idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(; idx < num; idx += blockDim.x * gridDim.x)
    {
    out_mem[idx] = max(min_val, min(max_val, A_mem[idx]));
    }
  }
