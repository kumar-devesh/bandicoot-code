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

// This extracts L from U, and sets the lower diagonal of U to 0.
__kernel
void
COOT_FN(PREFIX,lu_extract_pivoted_l)(__global eT1* L,
                                     __global eT1* U,
                                     const UWORD n_rows,
                                     const UWORD n_cols,
                                     __global const UWORD* ipiv)
  {
  const UWORD row = get_global_id(0);
  const UWORD col = get_global_id(1);

  // Note that U might not be square, but L must be.
  // If n_cols > n_rows, then U is upper trapezoidal.
  // If n_rows > n_cols, then L is lower trapezoidal.
  // However, L is always square (size n_rows x n_rows).

  const UWORD index = row + n_rows * col;
  // We are extracted a permuted version of L.
  // Instead of extracting row i of U as row i of L,
  // we extract row i of U as row ipiv[i] of L.
  const UWORD L_index = ipiv[row] + n_rows * col;

  if( (row < n_rows) && (col < n_cols))
    {
    if (col < n_rows)
      {
      L[L_index] = (row > col) ? U[index] : ((row == col) ? 1 : 0);
      }
    U[index] = (row > col) ? 0 : U[index];
    }
  else if ( (row < n_rows) && (col < n_rows) ) // L has size n_rows x n_rows
    {
    L[L_index] = (row == col) ? 1 : 0;
    }
  }
