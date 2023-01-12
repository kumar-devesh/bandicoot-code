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
COOT_FN(PREFIX,equ_array_div_scalar_pre)(eT2* out,
                                         const eT1* A,
                                         const eT1 val_pre,
                                         const eT2 val_post,
                                         const UWORD N)
  {
  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N)
    {
    // if both are 0, we take it as val_pre == 0 and val_post unused
    if (val_post == (eT2) (0))
      {
      out[i] = (eT2) (val_pre / A[i]);
      }
    else if (val_pre == (eT1) (0) && val_post != (eT2) (0))
      {
      out[i] = val_post / ((eT2) A[i]);
      }
    else
      {
      // if both are nonzero, we apply sequentially----be careful!
      out[i] = val_post / ((eT2) (val_pre / A[i]));
      }
    }
  }
