// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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


template<typename out_eT, typename T1>
inline
void
op_strans::apply(Mat<out_eT>& out, const Op<T1, op_strans>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(in.m);

  if(U.is_alias(out))
    {
    // TODO: inplace transpose?
    Mat<out_eT> tmp;
    op_strans::apply_noalias(tmp, U.M);
    out.steal_mem(tmp);
    }
  else
    {
    op_strans::apply_noalias(out, U.M);
    }
  }



template<typename out_eT, typename in_eT>
inline
void
op_strans::apply_noalias(Mat<out_eT>& out, const Mat<in_eT>& A)
  {
  coot_extra_debug_sigprint();

  out.set_size(A.n_cols, A.n_rows);

  if (A.n_cols == 1 || A.n_rows == 1)
    {
    // Simply copying the data is sufficient.
    arrayops::copy(out.get_dev_mem(false), A.get_dev_mem(false), A.n_elem);
    }
  else
    {
    coot_rt_t::strans(out.get_dev_mem(false), A.get_dev_mem(false), A.n_rows, A.n_cols);
    }
  }
