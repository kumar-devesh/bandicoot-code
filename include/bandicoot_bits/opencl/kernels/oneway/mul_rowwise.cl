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
COOT_FN(PREFIX,mul_rowwise)(__global eT1* out,
                            __global const eT1* A, // expected to have length n_rows
                            __global const eT1* in,
                            const eT1 alpha,
                            const UWORD n_rows,
                            const UWORD n_cols)
  {
  const UWORD row = get_global_id(0);
  if(row < n_rows)
    {
    const eT1 val = alpha * A[col];

    #pragma unroll
    for(UWORD i = 0; i < n_rows; ++i)
      {
      const UWORD offset = i * n_rows + row;
      out[offset] = val * in[offset];
      }
    }
  }
