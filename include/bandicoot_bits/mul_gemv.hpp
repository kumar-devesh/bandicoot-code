// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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



template<const bool do_trans_A=false, const bool use_alpha=false, const bool use_beta=false>
class gemv
  {
  public:

  template<typename eT>
  inline
  static
  void
  apply( Mat<eT>& y, const Mat<eT>& A, const Mat<eT>& x, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    coot_extra_debug_sigprint();

    const float local_alpha = (use_alpha) ? alpha : float(1);
    const float local_beta  = (use_beta)  ? beta  : float(0);

    // TODO: versions that use subviews and offsets instead of using the default values
    coot_rt_t::gemv<eT, do_trans_A>(y.get_dev_mem(true), 0, 1, A.get_dev_mem(true), 0, A.n_rows, A.n_rows, A.n_cols, x.get_dev_mem(true), 0, 1, local_alpha, local_beta);
    }
  };
