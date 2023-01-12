// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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
COOT_FN(PREFIX,submat_extract)(__global eT2* out,
                               __global const eT1* in,
                               const UWORD in_start_row,
                               const UWORD in_start_col,
                               const UWORD in_n_rows,
                               const UWORD out_n_rows,
                               const UWORD out_n_cols)
  {
  const UWORD row = get_global_id(0); // row in source matrix
  const UWORD col = get_global_id(1); // col in source matrix
  if( (row <= out_n_rows) && (col <= out_n_cols) )
    {
    const UWORD  in_index = (in_start_row + row) + ((in_start_col + col) * in_n_rows);
    const UWORD out_index = row + col * out_n_rows;
    out[out_index] = (eT2) in[in_index];
    }
  }
