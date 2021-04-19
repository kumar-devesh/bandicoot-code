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
COOT_FN(PREFIX,larfg_work)(__global eT1* x, const UWORD N, __global eT* norm, const eT min_norm)
  {
  const UWORD tid = get_local_id(0);

  if (tid < N)
    {
    const eT alpha = x[0];
    const eT beta = -sign(alpha) * norm_tau[0];

    // Now perform scaling of x in order to produce v.  If beta is too small, do nothing.  (The CPU will catch it after this kernel finishes.)
    if (beta >= min_norm)
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
    }
  }
