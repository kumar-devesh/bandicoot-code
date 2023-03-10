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
COOT_FN(PREFIX,diag_inplace_minus_scalar)(eT1* out,
                                          const eT1 val,
                                          const UWORD n_rows,
                                          const UWORD len)
  {
  const UWORD tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len)
    {
    const UWORD i = (n_rows + 1) * tid;
    out[i] -= val;
    }
  }
