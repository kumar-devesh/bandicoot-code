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
COOT_FN(PREFIX,var_colwise)(eT1* out,
                            const eT1* A,
                            const eT1* means,
                            const UWORD A_n_rows,
                            const UWORD A_n_cols,
                            const UWORD norm_correction)
  {
  const UWORD col = blockIdx.x * blockDim.x + threadIdx.x;
  if(col < A_n_cols)
    {
    const eT1* colptr = &(A[ col*A_n_rows ]);
    const eT1 mean_val = means[col];
    eT1 acc = (eT1) (0);
    for (UWORD i = 0; i < A_n_rows; ++i)
      {
      eT1 val = (colptr[i] - mean_val);
      acc += (val * val);
      }

    out[col] = (acc / (eT1) (A_n_rows - norm_correction));
    }
  }
