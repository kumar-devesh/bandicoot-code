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
COOT_FN(PREFIX,submat_mean_colwise_conv_pre)(__global eT2* out,
                                             __global const eT1* A,
                                             const UWORD A_n_rows,
                                             const UWORD start_row,
                                             const UWORD start_col,
                                             const UWORD sub_n_rows,
                                             const UWORD sub_n_cols)
  {
  const UWORD col = get_global_id(0);
  if(col < sub_n_cols)
    {
    __global const eT1* colptr = &(A[ (col + start_col)*A_n_rows + start_row ]);
    eT2 acc = (eT2) (0);
    #pragma unroll
    for (UWORD i = 0; i < sub_n_rows; ++i)
      {
      acc += (eT2) colptr[i];
      }
    out[col] = (acc / (eT2) sub_n_rows);
    }
  }
