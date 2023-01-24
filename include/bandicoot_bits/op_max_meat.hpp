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



template<typename T1>
inline
typename T1::elem_type
op_max::apply(const Op<T1, op_max>& in)
  {
  coot_extra_debug_sigprint();

  if (resolves_to_vector<T1>::value)
    {
    const unwrap<T1> U(in.m);
    const Mat<typename T1::elem_type>& A = U.M;

    return coot_rt_t::max(A.get_dev_mem(false), A.n_elem);
    }
  else
    {
    // TODO: implement kernels that do row-wise or column-wise max...
    throw std::invalid_argument("max(): not yet implemented for anything other than vectors... sorry!");
    }
  }



// Optimization: we have a max-abs kernel available for max(abs(...)) situations.
template<typename T1>
inline
typename T1::elem_type
op_max::apply(const Op<eOp<T1, eop_abs>, op_max>& in)
  {
  coot_extra_debug_sigprint();

  if (resolves_to_vector<T1>::value)
    {
    const unwrap<T1> U(in.m.m);
    const Mat<typename T1::elem_type>& A = U.M;

    return coot_rt_t::max_abs(A.get_dev_mem(false), A.n_elem);
    }
  else
    {
    // TODO: implement kernels that do row-wise or column-wise max...
    throw std::invalid_argument("max(): not yet implemented for anything other than vectors... sorry!");
    }
  }
