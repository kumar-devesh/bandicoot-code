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
COOT_FN(PREFIX,submat_var_rowwise)(eT1* out,
                                   const eT1* A,
                                   const eT1* means,
                                   const UWORD A_n_rows,
                                   const UWORD start_row,
                                   const UWORD start_col,
                                   const UWORD sub_n_rows,
                                   const UWORD sub_n_cols,
                                   const UWORD norm_correction)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < sub_n_rows)
    {
    const eT1 mean_val = means[row];
    eT1 acc = (eT1)(0);
    for (UWORD i = 0; i < sub_n_cols; ++i)
      {
      const eT1 val = (A[(i + start_col) * A_n_rows + (row + start_row)] - mean_val);
      acc += (val * val);
      }

    out[row] = (acc / (eT1) (sub_n_cols - norm_correction));
    }
  }
