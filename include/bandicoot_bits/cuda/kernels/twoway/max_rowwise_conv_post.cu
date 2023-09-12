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

__global__
void
COOT_FN(PREFIX,max_rowwise_conv_post)(eT2* dest,
                                      const eT1* src,
                                      const UWORD n_rows,
                                      const UWORD n_cols,
                                      const UWORD dest_mem_incr,
                                      const UWORD src_M_n_rows)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < n_rows)
    {
    eT1 acc = (eT1) src[row];
    for (UWORD i = 1; i < n_cols; ++i)
      {
      acc = max(acc, A[(i * src_M_n_rows) + row]);
      }

    dest[row * dest_mem_incr] = (eT2) (acc);
    }
  }
