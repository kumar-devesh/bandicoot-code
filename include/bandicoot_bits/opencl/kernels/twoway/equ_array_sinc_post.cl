// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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
COOT_FN(PREFIX,equ_array_sinc_post)(__global eT2* out,
                                    __global const eT1* A,
                                    const eT1 val_pre,
                                    const eT2 val_post,
                                    const UWORD N)
  {
  (void)(val_pre);
  (void)(val_post);
  const UWORD i = get_global_id(0);
  if(i < N)
    {
    const eT1 val = (eT1) A[i];
    // To imitate Armadillo correctly, we use double if the type is not floating point.
    if (COOT_FN(coot_is_fp_,eT1)())
      {
      const fp_eT1 tmp = val * M_PI;
      out[i] = (tmp == (eT1) 0.0) ? (eT2) 1.0 : (eT2) (sin(tmp) / tmp);
      }
    else
      {
      const double fp_val = (double) val;
      const double tmp = fp_val * M_PI;
      out[i] = (tmp == 0.0) ? (eT2) 1.0 : (eT2) (sin(tmp) / tmp);
      }
    }
  }