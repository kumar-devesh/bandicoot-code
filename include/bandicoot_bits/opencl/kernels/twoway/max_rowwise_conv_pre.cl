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
COOT_FN(PREFIX,max_rowwise_conv_pre)(__global eT2* out,
                                     __global const eT1* A,
                                     const UWORD A_n_rows,
                                     const UWORD A_n_cols)
  {
  const UWORD row = get_global_id(0);
  if(row < A_n_rows)
    {
    eT2 acc = (eT2) A[row];
    #pragma unroll
    for(UWORD i=1; i < A_n_cols; ++i)
      {
      acc = max(acc, (eT2) (A[i*A_n_rows + row]));
      }
    out[row] = acc;
    }
  }
