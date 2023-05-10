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
COOT_FN(PREFIX,lu_extract_l)(__global eT1* L,
                             __global const eT1* U,
                             const UWORD n_rows,
                             const UWORD n_cols)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);
  const UWORD index = row + n_rows * col;

  if( (row < n_rows) && (col < n_cols))
    {
    L[index] = (row > col) ? U[index] : ((row == col) ? 1 : 0);
    }
  }
