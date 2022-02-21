// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,submat_inplace_plus_mat)(eT2* out,
                                        const eT1* A,
                                        const UWORD out_start_row,
                                        const UWORD out_start_col,
                                        const UWORD out_n_rows,
                                        const UWORD A_n_rows,
                                        const UWORD A_n_cols)
  {
  const UWORD row = blockIdx.x * blockDim.x + threadIdx.x;
  const UWORD col = blockIdx.y * blockDim.y + threadIdx.y;
  if( (row <= A_n_rows) && (col <= A_n_cols) )
    {
    const UWORD out_index = (out_start_row + row) + ((out_start_col + col) * out_n_rows);
    const UWORD   A_index = row + col * A_n_rows;
    out[out_index] += (eT2) A[A_index];
    }
  }