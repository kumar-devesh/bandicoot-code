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
COOT_FN(PREFIX,equ_array_div_array)(eT3* out,
                                    const eT1* A,
                                    const eT2* B,
                                    const UWORD N)
  {
  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N)
    {
    const threeway_promoted_eT a_val = (threeway_promoted_eT) A[i];
    const threeway_promoted_eT b_val = (threeway_promoted_eT) B[i];
    out[i] = (eT3) (a_val / b_val);
    }
  }
