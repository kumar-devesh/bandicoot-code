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

__global__
void
COOT_FN(PREFIX,equ_array_trunc_log_pre)(eT2* out,
                                        const eT1* A,
                                        const eT1 val_pre,
                                        const eT2 val_post,
                                        const UWORD N)
  {
  (void)(val_pre);
  (void)(val_post);
  const UWORD i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N)
    {
    // To match Armadillo, we always use `double` as the intermediate type for any non-floating point type.
    const eT2 val = (eT2) A[i];
    if (coot_is_fp(val))
      {
      const fp_eT2 fp_val = (fp_eT2) val;
      if (fp_val <= (fp_eT2) 0)
        {
        out[i] = (eT2) log(COOT_FN(coot_type_min_,fp_eT2)());
        }
      else if (isinf(fp_val))
        {
        out[i] = (eT2) log(COOT_FN(coot_type_max_,fp_eT2)());
        }
      else
        {
        out[i] = (eT2) log(fp_val);
        }
      }
    else
      {
      const double fp_val = (double) val;
      if (fp_val <= (double) 0)
        {
        out[i] = (eT2) log(coot_type_min_double());
        }
      else if (isinf(fp_val))
        {
        out[i] = (eT2) log(coot_type_max_double());
        }
      else
        {
        out[i] = (eT2) log(fp_val);
        }
      }
    }
  }