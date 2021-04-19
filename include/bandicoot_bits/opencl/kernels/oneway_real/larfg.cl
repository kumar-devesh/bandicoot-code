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
  const UWORD tid = get_local_id(0);

  if (tid < N)
    {
    const eT1 norm_val = sqrt(norm[0]);
    const eT1 alpha = x[0];
    const eT1 beta = -copysign(norm_val, alpha);

    // Now perform scaling of x in order to produce v.  If beta is too small, do nothing.  (The CPU will catch it after this kernel finishes.)
//    if (beta >= min_norm && alpha != norm_val)
    if (alpha != norm_val)
      {
      if (tid == 0)
        {
        x[tid] = beta;
        norm[0] = alpha;
        }
      else
        {
        x[tid] /= (alpha - beta);
        }
      }
    else if (tid == 0)
      {
      // In this case, x == [0...], but we need to notify the higher level
      // process somehow, so we pass back `alpha` in norm[0].
      norm[0] = norm_val;
      }
    }
  }
