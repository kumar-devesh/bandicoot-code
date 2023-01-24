// Copyright 2021 Ryan Curtin (https://www.ratml.org)
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



/**
 * LARFG generates a real or complex elementary reflector H of order n, such
 * that if eT is real,
 *
 *     H * ( alpha ) = ( beta ),   H**T * H = I.
 *         (   x   )   (   0  )
 *
 * and if eT is complex,
 *
 *     H**H * ( alpha ) = ( beta ),   H**H * H = I.
 *            (   x   )   (   0  )
 *
 * where alpha and beta are scalars, and x is an (n-1)-element
 * vector. If eT is real, H is represented in the form
 *
 *      H = I - tau * ( 1 ) * ( 1 v**T ) ,
 *                    ( v )
 *
 * where tau is a real scalar and v is a real (n-1)-element
 * vector.  If eT is complex, H is represented in the form
 *
 *      H = I - tau * ( 1 ) * ( 1 v**H ) ,
 *                    ( v )
 *
 * where tau is a complex scalar and v is a complex (n-1)-element
 * vector. Note that H is not hermitian.
 *
 * If the elements of x are all zero, then tau = 0 and H is taken to be
 * the unit matrix.
 *
 * Otherwise  1 <= real(tau) <= 2 and abs(tau - 1) <= 1.
 *
 * X is expected to be given as a vector such that X = [alpha, x], and when
 * the method is completed, X will hold [beta, v].  tau is the return value.
 */
template<typename eT>
inline
eT
larfg(Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  // Assumption: X is sufficiently small that we can process the entire thing with one kernel call.

  return coot_rt_t::larfg(X.get_dev_mem(false), X.n_elem);
  }
