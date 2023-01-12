// Copyright 2021 Ryan Curtin (https://www.ratml.org/)
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

// TODO: extend to complex cases
__kernel
void
COOT_FN(PREFIX,larfg)(__global eT1* x, const UWORD N, __global eT1* norm, const eT1 min_norm)
  {
  const UWORD tid = get_global_id(0);

  if (tid < N)
    {
    const eT1 norm_val = sqrt(norm[0]);
    const eT1 alpha = x[0];
    const eT1 beta = -copysign(norm_val, alpha);

    if (alpha == norm_val)
      {
      // If all elements in x are 0, we'll set tau to 0 at the higher level.
      if (tid == 0)
        {
        norm[2] = -1;
        }
      }
      // This else if is currently not possible, as norm computation will always underflow.
//    else if (norm_val <= min_norm)
//      {
//      // If the norm is too small, then we have to scale.  (We still need beta.)
//      if (tid == 0)
//        {
//        norm[1] = beta;
//        norm[2] = -2;
//        }
//      }
    else
      {
      // Now perform scaling of x in order to produce v.
      if (tid == 0)
        {
        x[tid] = beta;
        norm[0] = alpha;
        norm[1] = beta;
        }
      else
        {
        x[tid] /= (alpha - beta);
        }
      }
    }
  }
