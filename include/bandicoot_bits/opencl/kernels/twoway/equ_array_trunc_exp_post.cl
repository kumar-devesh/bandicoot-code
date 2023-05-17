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
COOT_FN(PREFIX,equ_array_trunc_exp_post)(__global eT2* out,
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
    // To match Armadillo, we always use `double` as the intermediate type for any non-floating point type.
    const eT1 val = A[i];
    if (COOT_FN(coot_is_fp_,eT1)())
      {
      const fp_eT1 fp_val = (fp_eT1) val;
      if (fp_val >= log(COOT_FN(coot_type_max_,fp_eT1)()))
        {
        out[i] = (eT2) ((eT1) COOT_FN(coot_type_max_,fp_eT1)());
        }
      else
        {
        out[i] = (eT2) ((eT1) exp(fp_val));
        }
      }
    else
      {
      const ARMA_FP_TYPE fp_val = (ARMA_FP_TYPE) val;
      if (fp_val >= log(COOT_FN(coot_type_max_,ARMA_FP_TYPE)()))
        {
        out[i] = (eT2) ((eT1) COOT_FN(coot_type_max_,ARMA_FP_TYPE)());
        }
      else
        {
        out[i] = (eT2) ((eT1) exp(fp_val));
        }
      }
    }
  }
